use thiserror::Error;

use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedFailureEnvelope {
    domain: FailureDomain,
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FailureEnvelopeWire {
    domain: FailureDomain,
    code: String,
    message: String,
    retryable: bool,
}

impl From<FailureEnvelopeWire> for UnvalidatedFailureEnvelope {
    fn from(wire: FailureEnvelopeWire) -> Self {
        Self {
            domain: wire.domain,
            code: wire.code,
            message: wire.message,
            retryable: wire.retryable,
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
        FailureEnvelope::new(self.domain, self.code, self.message, self.retryable)
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
        Ok(())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicAdmissionFaultKind {
    InvalidContract,
    UnknownDomain,
    ForeignCoordinator,
    Poisoned,
    EpochExhausted,
    AuthorityExhausted,
    ArithmeticOverflow,
    AllocationFailure,
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
    #[error("dynamic admission {kind:?}: {reason}")]
    DynamicAdmissionContract {
        kind: DynamicAdmissionFaultKind,
        reason: String,
    },
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
