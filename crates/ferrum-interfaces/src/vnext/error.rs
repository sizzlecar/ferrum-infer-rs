use thiserror::Error;

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::model_executor::PlanRuntimeResourceSnapshot;

use super::execution::DynamicBackingPoolId;

/// Maximum encoded size accepted by the untrusted failure-envelope decoder.
pub const MAX_FAILURE_ENVELOPE_WIRE_BYTES: usize = 8 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureDomain {
    Device,
    Operation,
    Resource,
    Planning,
    ModelResolution,
    Product,
    Event,
}

/// Portable failure payload. Execution identity is carried by the surrounding
/// event or resource receipt rather than flattened into this message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FailureEnvelope {
    domain: FailureDomain,
    code: String,
    message: String,
    retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource_snapshot: Option<PlanRuntimeResourceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedFailureEnvelope {
    domain: FailureDomain,
    code: String,
    message: String,
    retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource_snapshot: Option<PlanRuntimeResourceSnapshot>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FailureEnvelopeWire {
    domain: FailureDomain,
    code: String,
    message: String,
    retryable: bool,
    #[serde(default)]
    resource_snapshot: Option<PlanRuntimeResourceSnapshot>,
}

impl From<FailureEnvelopeWire> for UnvalidatedFailureEnvelope {
    fn from(wire: FailureEnvelopeWire) -> Self {
        Self {
            domain: wire.domain,
            code: wire.code,
            message: wire.message,
            retryable: wire.retryable,
            resource_snapshot: wire.resource_snapshot,
        }
    }
}

impl UnvalidatedFailureEnvelope {
    pub fn revalidate(self, expected_domain: FailureDomain) -> Result<FailureEnvelope, VNextError> {
        if self.domain != expected_domain {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "failure domain `{:?}` differs from expected `{:?}`",
                    self.domain, expected_domain
                ),
            });
        }
        let resource_snapshot = self.resource_snapshot;
        let envelope = FailureEnvelope::new(self.domain, self.code, self.message, self.retryable)?;
        match resource_snapshot {
            Some(snapshot) => envelope.with_resource_snapshot(snapshot),
            None => Ok(envelope),
        }
    }
}

impl FailureEnvelope {
    pub fn new(
        domain: FailureDomain,
        code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> Result<Self, VNextError> {
        let envelope = Self {
            domain,
            code: code.into(),
            message: message.into(),
            retryable,
            resource_snapshot: None,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), VNextError> {
        if self.code.is_empty()
            || self.code.len() > 64
            || !self
                .code
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            || self.message.trim().is_empty()
            || self.message.len() > 4096
            || self
                .message
                .bytes()
                .any(|byte| byte.is_ascii_control() && !matches!(byte, b'\n' | b'\t'))
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "failure code or message is empty, oversized, or non-portable".to_owned(),
            });
        }
        if let Some(snapshot) = &self.resource_snapshot {
            if self.domain != FailureDomain::Resource {
                return Err(VNextError::InvalidExecutionPlan {
                    reason:
                        "only resource-domain failures may carry a plan runtime resource snapshot"
                            .to_owned(),
                });
            }
            snapshot
                .validate()
                .map_err(|error| VNextError::InvalidExecutionPlan {
                    reason: format!("invalid plan runtime resource snapshot: {error}"),
                })?;
        }
        Ok(())
    }

    pub fn with_resource_snapshot(
        mut self,
        snapshot: PlanRuntimeResourceSnapshot,
    ) -> Result<Self, VNextError> {
        self.resource_snapshot = Some(snapshot);
        self.validate()?;
        Ok(self)
    }

    pub const fn domain(&self) -> FailureDomain {
        self.domain
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub const fn retryable(&self) -> bool {
        self.retryable
    }

    pub const fn resource_snapshot(&self) -> Option<&PlanRuntimeResourceSnapshot> {
        self.resource_snapshot.as_ref()
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedFailureEnvelope, VNextError> {
        if bytes.len() > MAX_FAILURE_ENVELOPE_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted failure envelope",
                message: format!(
                    "payload has {} bytes; maximum is {MAX_FAILURE_ENVELOPE_WIRE_BYTES}",
                    bytes.len()
                ),
            });
        }
        serde_json::from_slice::<FailureEnvelopeWire>(bytes)
            .map(Into::into)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted failure envelope",
                message: error.to_string(),
            })
    }
}

#[cfg(test)]
mod failure_envelope_tests {
    use super::{FailureDomain, FailureEnvelope};
    use crate::model_executor::PlanRuntimeResourceSnapshot;

    fn snapshot() -> PlanRuntimeResourceSnapshot {
        PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).unwrap()
    }

    #[test]
    fn resource_snapshot_round_trips_only_after_revalidation() {
        let envelope = FailureEnvelope::new(
            FailureDomain::Resource,
            "diagnostic_resource_failure",
            "injected resource failure",
            false,
        )
        .unwrap()
        .with_resource_snapshot(snapshot())
        .unwrap();
        let encoded = serde_json::to_vec(&envelope).unwrap();

        let decoded = FailureEnvelope::decode_untrusted(&encoded)
            .unwrap()
            .revalidate(FailureDomain::Resource)
            .unwrap();

        assert_eq!(decoded, envelope);
        assert_eq!(
            decoded
                .resource_snapshot()
                .unwrap()
                .available_bytes()
                .unwrap(),
            400
        );
    }

    #[test]
    fn non_resource_failure_rejects_resource_snapshot() {
        let envelope = FailureEnvelope::new(
            FailureDomain::Operation,
            "operation_failure",
            "operation failure",
            false,
        )
        .unwrap();

        assert!(envelope.with_resource_snapshot(snapshot()).is_err());
    }

    #[test]
    fn untrusted_resource_snapshot_fails_closed_when_capacity_is_incoherent() {
        let wire = serde_json::json!({
            "domain": "resource",
            "code": "forged_resource_failure",
            "message": "forged resource failure",
            "retryable": false,
            "resource_snapshot": {
                "device_capacity_bytes": 1_000,
                "usable_capacity_bytes": 1_001,
                "process_claimed_bytes": 700,
                "plan_claimed_bytes": 700,
                "static_bytes": 400,
                "dynamic_resident_bytes": 300,
                "dynamic_free_bytes": 200,
                "pending_growth_bytes": 0,
                "quarantined_bytes": 0
            }
        });

        let unvalidated = FailureEnvelope::decode_untrusted(&serde_json::to_vec(&wire).unwrap())
            .expect("wire shape should decode before trust validation");

        assert!(unvalidated.revalidate(FailureDomain::Resource).is_err());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicAdmissionFaultKind {
    InvalidContract,
    UnknownDomain,
    ForeignCoordinator,
    Poisoned,
    EpochExhausted,
    EpochRegression,
    AuthorityExhausted,
    ArithmeticOverflow,
    AllocationFailure,
}

/// Capacity boundary that rejected one otherwise valid dynamic growth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCapacityPressureScope {
    PlanBudget,
    ProcessWide,
}

/// Exact device-wide pressure observed while trying to grow dynamic backing.
///
/// This is retry evidence, not an allocation authority. Plan-budget pressure
/// can be paired with plan-local release epochs; process-wide pressure requires
/// device-wide coordination. Contract and allocator failures remain terminal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCapacityPressure {
    scope: DeviceCapacityPressureScope,
    device_id: String,
    requested_bytes: u64,
    plan_claimed_bytes: u64,
    plan_usable_bytes: u64,
    process_claimed_bytes: u64,
    process_usable_bytes: u64,
}

impl DeviceCapacityPressure {
    pub fn new(
        scope: DeviceCapacityPressureScope,
        device_id: String,
        requested_bytes: u64,
        plan_claimed_bytes: u64,
        plan_usable_bytes: u64,
        process_claimed_bytes: u64,
        process_usable_bytes: u64,
    ) -> Result<Self, VNextError> {
        let pressure = Self {
            scope,
            device_id,
            requested_bytes,
            plan_claimed_bytes,
            plan_usable_bytes,
            process_claimed_bytes,
            process_usable_bytes,
        };
        let plan_available = pressure
            .plan_usable_bytes
            .checked_sub(pressure.plan_claimed_bytes);
        let process_available = pressure
            .process_usable_bytes
            .checked_sub(pressure.process_claimed_bytes);
        let scope_matches = match pressure.scope {
            DeviceCapacityPressureScope::PlanBudget => {
                plan_available.is_some_and(|available| available < pressure.requested_bytes)
            }
            DeviceCapacityPressureScope::ProcessWide => {
                plan_available.is_some_and(|available| available >= pressure.requested_bytes)
                    && process_available
                        .is_some_and(|available| available < pressure.requested_bytes)
            }
        };
        if pressure.device_id.trim().is_empty()
            || pressure.requested_bytes == 0
            || process_available.is_none()
            || !scope_matches
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "device capacity pressure evidence is inconsistent".to_owned(),
            });
        }
        Ok(pressure)
    }

    pub fn scope(&self) -> &DeviceCapacityPressureScope {
        &self.scope
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }

    pub const fn plan_claimed_bytes(&self) -> u64 {
        self.plan_claimed_bytes
    }

    pub const fn plan_usable_bytes(&self) -> u64 {
        self.plan_usable_bytes
    }

    pub const fn process_claimed_bytes(&self) -> u64 {
        self.process_claimed_bytes
    }

    pub const fn process_usable_bytes(&self) -> u64 {
        self.process_usable_bytes
    }

    pub const fn available_bytes(&self) -> u64 {
        let plan_available = self
            .plan_usable_bytes
            .saturating_sub(self.plan_claimed_bytes);
        let process_available = self
            .process_usable_bytes
            .saturating_sub(self.process_claimed_bytes);
        if plan_available < process_available {
            plan_available
        } else {
            process_available
        }
    }
}

impl fmt::Display for DeviceCapacityPressure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "device `{}` capacity is temporarily unavailable: requested {}, plan claimed {}/{}, process claimed {}/{}",
            self.device_id,
            self.requested_bytes,
            self.plan_claimed_bytes,
            self.plan_usable_bytes,
            self.process_claimed_bytes,
            self.process_usable_bytes
        )
    }
}

/// Exact pool-local resident ceiling observed while maintaining otherwise
/// valid deferred backing.
///
/// Unlike device capacity pressure, this does not authorize a larger pool. It
/// proves that existing resident owners must be released or a reusable cache
/// entry must be evicted before the deferred transaction can be retried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolResidentPressure {
    pool_id: DynamicBackingPoolId,
    requested_bytes: u64,
    resident_bytes: u64,
    maximum_resident_bytes: u64,
}

impl DynamicPoolResidentPressure {
    pub fn new(
        pool_id: DynamicBackingPoolId,
        requested_bytes: u64,
        resident_bytes: u64,
        maximum_resident_bytes: u64,
    ) -> Result<Self, VNextError> {
        let pressure = Self {
            pool_id,
            requested_bytes,
            resident_bytes,
            maximum_resident_bytes,
        };
        if pressure.requested_bytes == 0
            || pressure.resident_bytes > pressure.maximum_resident_bytes
            || pressure.available_bytes() >= pressure.requested_bytes
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "dynamic pool resident pressure evidence is inconsistent".to_owned(),
            });
        }
        Ok(pressure)
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn maximum_resident_bytes(&self) -> u64 {
        self.maximum_resident_bytes
    }

    pub const fn available_bytes(&self) -> u64 {
        self.maximum_resident_bytes
            .saturating_sub(self.resident_bytes)
    }
}

impl fmt::Display for DynamicPoolResidentPressure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "dynamic pool `{}` resident capacity is temporarily unavailable: requested {}, resident {}/{}",
            self.pool_id.as_str(),
            self.requested_bytes,
            self.resident_bytes,
            self.maximum_resident_bytes
        )
    }
}

/// Recoverable physical pressure returned by deferred backing maintenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "evidence", rename_all = "snake_case")]
pub enum DynamicBackingPressure {
    DeviceCapacity(DeviceCapacityPressure),
    PoolResident(DynamicPoolResidentPressure),
}

impl DynamicBackingPressure {
    pub const fn requested_bytes(&self) -> u64 {
        match self {
            Self::DeviceCapacity(pressure) => pressure.requested_bytes(),
            Self::PoolResident(pressure) => pressure.requested_bytes(),
        }
    }

    pub const fn available_bytes(&self) -> u64 {
        match self {
            Self::DeviceCapacity(pressure) => pressure.available_bytes(),
            Self::PoolResident(pressure) => pressure.available_bytes(),
        }
    }

    pub const fn device_capacity(&self) -> Option<&DeviceCapacityPressure> {
        match self {
            Self::DeviceCapacity(pressure) => Some(pressure),
            Self::PoolResident(_) => None,
        }
    }

    pub const fn pool_resident(&self) -> Option<&DynamicPoolResidentPressure> {
        match self {
            Self::DeviceCapacity(_) => None,
            Self::PoolResident(pressure) => Some(pressure),
        }
    }
}

impl From<DeviceCapacityPressure> for DynamicBackingPressure {
    fn from(pressure: DeviceCapacityPressure) -> Self {
        Self::DeviceCapacity(pressure)
    }
}

impl From<DynamicPoolResidentPressure> for DynamicBackingPressure {
    fn from(pressure: DynamicPoolResidentPressure) -> Self {
        Self::PoolResident(pressure)
    }
}

/// Structured, fail-closed errors produced by the vNext contracts.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VNextError {
    #[error("invalid {kind} identity `{value}`: {reason}")]
    InvalidIdentity {
        kind: &'static str,
        value: String,
        reason: &'static str,
    },
    #[error("unknown model family `{family_id}`")]
    UnknownModelFamily { family_id: String },
    #[error("unknown external model metadata `{metadata_id}`")]
    UnknownExternalModelMetadata { metadata_id: String },
    #[error(
        "ambiguous model family registration for {identity_kind} `{identity}`: {matches} matches"
    )]
    AmbiguousModelFamilyRegistration {
        identity_kind: &'static str,
        identity: String,
        matches: usize,
    },
    #[error("invalid model config for `{family_id}` at `{field}`: {reason}")]
    InvalidModelConfig {
        family_id: String,
        field: String,
        reason: String,
    },
    #[error("unknown weight layout `{layout_id}` for model family `{family_id}`")]
    UnknownWeightLayout {
        family_id: String,
        layout_id: String,
    },
    #[error(
        "operation `{operation_id}` requires version {required_major}.{required_minor}; provider offers {available_major}.{available_minor}"
    )]
    IncompatibleOperationVersion {
        node_id: Option<String>,
        operation_id: String,
        required_major: u16,
        required_minor: u16,
        available_major: u16,
        available_minor: u16,
    },
    #[error("no provider for operation `{operation_id}` on device `{device_id}`: {reason}")]
    UnsupportedOperation {
        node_id: Option<String>,
        operation_id: String,
        device_id: String,
        reason: String,
    },
    #[error("invalid execution plan: {reason}")]
    InvalidExecutionPlan { reason: String },
    #[error("weight materializer `{materializer_id}` requires typed numerical-quality approval")]
    WeightMaterializerQualityApprovalRequired { materializer_id: String },
    #[error("dynamic admission {kind:?}: {reason}")]
    DynamicAdmissionContract {
        kind: DynamicAdmissionFaultKind,
        reason: String,
    },
    #[error("{0}")]
    DeviceCapacityUnavailable(DeviceCapacityPressure),
    #[error("{0}")]
    DynamicPoolResidentUnavailable(DynamicPoolResidentPressure),
    #[error(
        "dynamic resource admission is not connected: {descriptor_count} descriptors require at least {minimum_sequence_bytes} bytes for one runnable sequence"
    )]
    DynamicResourceAdmissionRequired {
        descriptor_count: usize,
        minimum_sequence_bytes: u64,
    },
    #[error(
        "unsupported execution plan schema {actual_major}.{actual_minor}; expected {expected_major}.{expected_minor}"
    )]
    UnsupportedPlanSchema {
        expected_major: u16,
        expected_minor: u16,
        actual_major: u16,
        actual_minor: u16,
    },
    #[error("execution plan hash mismatch: expected `{expected}`, actual `{actual}`")]
    PlanHashMismatch { expected: String, actual: String },
    #[error("invalid resource transition for `{resource_id}`: {from} + {action}")]
    InvalidResourceTransition {
        resource_id: String,
        from: &'static str,
        action: &'static str,
    },
    #[error("invalid resource lease transition for `{lease_id}`: {from} + {action}")]
    InvalidLeaseTransition {
        lease_id: String,
        from: &'static str,
        action: &'static str,
    },
    #[error("invalid resolved model plan at `{field}`: {reason}")]
    InvalidResolvedModelPlan { field: String, reason: String },
    #[error("failed to {context}: {message}")]
    Serialization {
        context: &'static str,
        message: String,
    },
}
