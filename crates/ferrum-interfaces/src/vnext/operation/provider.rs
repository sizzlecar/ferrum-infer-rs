use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

use super::super::{
    CapabilityId, ContractVersion, DeviceId, ExecutionIdentityEnvelope, NodeId, OperationId,
    ProviderId, QuantizationFormatId, UnvalidatedExecutionIdentityParts, VNextError,
    WeightFormatId,
};
use super::foundation::{canonical_sha256, invalid_operation};
use super::{
    DynamicStorageRequirement, ProfilePhase, ProviderStorageBindingRequirement, ResolvedValueRole,
};

pub const PROVIDER_EXECUTION_SEMANTICS_VERSION: ContractVersion = ContractVersion::new(1, 0);
pub const MAX_OPERATION_FAILURE_WIRE_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProviderExecutionContractFingerprint([u8; 32]);

impl ProviderExecutionContractFingerprint {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ProviderExecutionContractFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for ProviderExecutionContractFingerprint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ProviderExecutionContractFingerprint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if !canonical_sha256(&value) {
            return Err(serde::de::Error::custom(
                "provider execution contract fingerprint must be a lowercase SHA256",
            ));
        }
        let mut bytes = [0_u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(Self(bytes))
    }
}

/// Repeatability promised for one immutable plan/provider/runtime binding.
///
/// Bitwise equality covers every declared output and state effect when logical
/// inputs, explicit RNG state, initial state, and initialized workspaces are
/// identical. Approximation against an independent oracle is a separate
/// operation contract and cannot weaken this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderExecutionRepeatability {
    BitwiseSameRuntime,
}

impl ProviderExecutionRepeatability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BitwiseSameRuntime => "bitwise_same_runtime",
        }
    }
}

/// Whether a provider authorizes reusable device execution for the same
/// immutable eager operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderReplayEquivalence {
    Ineligible,
    BitwiseEagerEquivalent,
}

impl ProviderReplayEquivalence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ineligible => "ineligible",
            Self::BitwiseEagerEquivalent => "bitwise_eager_equivalent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProviderExecutionSemantics {
    contract_version: ContractVersion,
    contract_fingerprint: ProviderExecutionContractFingerprint,
    repeatability: ProviderExecutionRepeatability,
    replay_equivalence: ProviderReplayEquivalence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderExecutionSemanticsWire {
    contract_version: ContractVersion,
    contract_fingerprint: ProviderExecutionContractFingerprint,
    repeatability: ProviderExecutionRepeatability,
    replay_equivalence: ProviderReplayEquivalence,
}

impl<'de> Deserialize<'de> for ProviderExecutionSemantics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderExecutionSemanticsWire::deserialize(deserializer)?;
        let semantics = Self::new(
            wire.contract_version,
            wire.repeatability,
            wire.replay_equivalence,
        )
        .map_err(serde::de::Error::custom)?;
        if semantics.contract_fingerprint != wire.contract_fingerprint {
            return Err(serde::de::Error::custom(format!(
                "provider execution contract fingerprint mismatch: expected `{}`, actual `{}`",
                semantics.contract_fingerprint, wire.contract_fingerprint
            )));
        }
        Ok(semantics)
    }
}

impl ProviderExecutionSemantics {
    fn new(
        contract_version: ContractVersion,
        repeatability: ProviderExecutionRepeatability,
        replay_equivalence: ProviderReplayEquivalence,
    ) -> Result<Self, VNextError> {
        if contract_version != PROVIDER_EXECUTION_SEMANTICS_VERSION {
            return Err(invalid_operation(format!(
                "provider execution semantics version {contract_version} is unsupported"
            )));
        }
        let mut digest = Sha256::new();
        digest.update(b"ferrum.runtime-vnext.provider-execution-semantics.v1\0");
        digest.update(contract_version.major.to_le_bytes());
        digest.update(contract_version.minor.to_le_bytes());
        digest.update([match repeatability {
            ProviderExecutionRepeatability::BitwiseSameRuntime => 1,
        }]);
        digest.update([match replay_equivalence {
            ProviderReplayEquivalence::Ineligible => 0,
            ProviderReplayEquivalence::BitwiseEagerEquivalent => 1,
        }]);
        Ok(Self {
            contract_version,
            contract_fingerprint: ProviderExecutionContractFingerprint(digest.finalize().into()),
            repeatability,
            replay_equivalence,
        })
    }

    pub fn bitwise_eager_only() -> Self {
        Self::new(
            PROVIDER_EXECUTION_SEMANTICS_VERSION,
            ProviderExecutionRepeatability::BitwiseSameRuntime,
            ProviderReplayEquivalence::Ineligible,
        )
        .expect("built-in eager execution semantics are valid")
    }

    pub fn bitwise_eager_and_replay() -> Self {
        Self::new(
            PROVIDER_EXECUTION_SEMANTICS_VERSION,
            ProviderExecutionRepeatability::BitwiseSameRuntime,
            ProviderReplayEquivalence::BitwiseEagerEquivalent,
        )
        .expect("built-in reusable execution semantics are valid")
    }

    pub const fn contract_version(self) -> ContractVersion {
        self.contract_version
    }

    pub const fn contract_fingerprint(self) -> ProviderExecutionContractFingerprint {
        self.contract_fingerprint
    }

    pub const fn repeatability(self) -> ProviderExecutionRepeatability {
        self.repeatability
    }

    pub const fn replay_equivalence(self) -> ProviderReplayEquivalence {
        self.replay_equivalence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDeterminismRequirement {
    BitwiseSameRuntime,
    BitwiseSameRuntimeWithReplay,
}

impl ExecutionDeterminismRequirement {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BitwiseSameRuntime => "bitwise_same_runtime",
            Self::BitwiseSameRuntimeWithReplay => "bitwise_same_runtime_with_replay",
        }
    }

    pub const fn requires_replay_equivalence(self) -> bool {
        matches!(self, Self::BitwiseSameRuntimeWithReplay)
    }

    pub fn accepts(self, semantics: ProviderExecutionSemantics) -> bool {
        semantics.repeatability == ProviderExecutionRepeatability::BitwiseSameRuntime
            && (!self.requires_replay_equivalence()
                || semantics.replay_equivalence
                    == ProviderReplayEquivalence::BitwiseEagerEquivalent)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationProviderDescriptor {
    provider_id: ProviderId,
    operation_id: OperationId,
    operation_fingerprint: String,
    provider_implementation_fingerprint: String,
    execution_semantics: ProviderExecutionSemantics,
    version: ContractVersion,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
    accepted_weight_formats: BTreeSet<WeightFormatId>,
    accepted_quantization_formats: BTreeSet<QuantizationFormatId>,
    dynamic_storage_bindings: Vec<ProviderStorageBindingRequirement>,
    resource_estimator_id: String,
    resource_estimator_version: ContractVersion,
    resource_estimator_implementation_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationProviderDescriptorWire {
    provider_id: ProviderId,
    operation_id: OperationId,
    operation_fingerprint: String,
    provider_implementation_fingerprint: String,
    execution_semantics: ProviderExecutionSemantics,
    version: ContractVersion,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
    accepted_weight_formats: BTreeSet<WeightFormatId>,
    accepted_quantization_formats: BTreeSet<QuantizationFormatId>,
    dynamic_storage_bindings: Vec<ProviderStorageBindingRequirement>,
    resource_estimator_id: String,
    resource_estimator_version: ContractVersion,
    resource_estimator_implementation_fingerprint: String,
}

impl<'de> Deserialize<'de> for OperationProviderDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = OperationProviderDescriptorWire::deserialize(deserializer)?;
        let original_bindings = wire.dynamic_storage_bindings.clone();
        let descriptor = Self::new(
            wire.provider_id,
            wire.operation_id,
            wire.operation_fingerprint,
            wire.provider_implementation_fingerprint,
            wire.execution_semantics,
            wire.version,
            wire.device_id,
            wire.capabilities,
            wire.accepted_weight_formats,
            wire.accepted_quantization_formats,
            wire.dynamic_storage_bindings,
            wire.resource_estimator_id,
            wire.resource_estimator_version,
            wire.resource_estimator_implementation_fingerprint,
        )
        .map_err(serde::de::Error::custom)?;
        if descriptor.dynamic_storage_bindings != original_bindings {
            return Err(serde::de::Error::custom(
                "provider storage binding requirements are not canonical",
            ));
        }
        Ok(descriptor)
    }
}

impl OperationProviderDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider_id: ProviderId,
        operation_id: OperationId,
        operation_fingerprint: impl Into<String>,
        provider_implementation_fingerprint: impl Into<String>,
        execution_semantics: ProviderExecutionSemantics,
        version: ContractVersion,
        device_id: DeviceId,
        capabilities: BTreeSet<CapabilityId>,
        accepted_weight_formats: BTreeSet<WeightFormatId>,
        accepted_quantization_formats: BTreeSet<QuantizationFormatId>,
        mut dynamic_storage_bindings: Vec<ProviderStorageBindingRequirement>,
        resource_estimator_id: impl Into<String>,
        resource_estimator_version: ContractVersion,
        resource_estimator_implementation_fingerprint: impl Into<String>,
    ) -> Result<Self, VNextError> {
        let operation_fingerprint = operation_fingerprint.into();
        let provider_implementation_fingerprint = provider_implementation_fingerprint.into();
        let resource_estimator_id = resource_estimator_id.into();
        let resource_estimator_implementation_fingerprint =
            resource_estimator_implementation_fingerprint.into();
        dynamic_storage_bindings.sort_by_key(|binding| (binding.role(), binding.ordinal()));
        if version.major == 0
            || !canonical_sha256(&operation_fingerprint)
            || !canonical_sha256(&provider_implementation_fingerprint)
            || resource_estimator_id.is_empty()
            || resource_estimator_id.len() > 160
            || !resource_estimator_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
            })
            || resource_estimator_version.major == 0
            || !canonical_sha256(&resource_estimator_implementation_fingerprint)
            || dynamic_storage_bindings.is_empty()
            || dynamic_storage_bindings.windows(2).any(|pair| {
                (pair[0].role(), pair[0].ordinal()) == (pair[1].role(), pair[1].ordinal())
            })
        {
            return Err(invalid_operation(
                "operation provider or resource-estimator identity is invalid",
            ));
        }
        Ok(Self {
            provider_id,
            operation_id,
            operation_fingerprint,
            provider_implementation_fingerprint,
            execution_semantics,
            version,
            device_id,
            capabilities,
            accepted_weight_formats,
            accepted_quantization_formats,
            dynamic_storage_bindings,
            resource_estimator_id,
            resource_estimator_version,
            resource_estimator_implementation_fingerprint,
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn operation_fingerprint(&self) -> &str {
        &self.operation_fingerprint
    }

    pub fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub const fn execution_semantics(&self) -> ProviderExecutionSemantics {
        self.execution_semantics
    }

    pub fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.capabilities
    }

    pub fn accepted_weight_formats(&self) -> &BTreeSet<WeightFormatId> {
        &self.accepted_weight_formats
    }

    pub fn accepted_quantization_formats(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.accepted_quantization_formats
    }

    pub fn dynamic_storage_bindings(&self) -> &[ProviderStorageBindingRequirement] {
        &self.dynamic_storage_bindings
    }

    pub fn dynamic_storage_for(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Option<&DynamicStorageRequirement> {
        self.dynamic_storage_bindings
            .binary_search_by_key(&(role, ordinal), |binding| {
                (binding.role(), binding.ordinal())
            })
            .ok()
            .map(|index| self.dynamic_storage_bindings[index].storage())
    }

    pub fn resource_estimator_id(&self) -> &str {
        &self.resource_estimator_id
    }

    pub const fn resource_estimator_version(&self) -> ContractVersion {
        self.resource_estimator_version
    }

    pub fn resource_estimator_implementation_fingerprint(&self) -> &str {
        &self.resource_estimator_implementation_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EngineProviderDescriptor {
    provider_id: ProviderId,
    contract_version: ContractVersion,
    implementation_fingerprint: String,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineProviderDescriptorWire {
    provider_id: ProviderId,
    contract_version: ContractVersion,
    implementation_fingerprint: String,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
}

impl<'de> Deserialize<'de> for EngineProviderDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = EngineProviderDescriptorWire::deserialize(deserializer)?;
        Self::new(
            wire.provider_id,
            wire.contract_version,
            wire.implementation_fingerprint,
            wire.device_id,
            wire.capabilities,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl EngineProviderDescriptor {
    pub fn new(
        provider_id: ProviderId,
        contract_version: ContractVersion,
        implementation_fingerprint: impl Into<String>,
        device_id: DeviceId,
        capabilities: BTreeSet<CapabilityId>,
    ) -> Result<Self, VNextError> {
        let implementation_fingerprint = implementation_fingerprint.into();
        if contract_version.major == 0 || !canonical_sha256(&implementation_fingerprint) {
            return Err(invalid_operation(
                "engine provider contract version or implementation fingerprint is invalid",
            ));
        }
        Ok(Self {
            provider_id,
            contract_version,
            implementation_fingerprint,
            device_id,
            capabilities,
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub const fn contract_version(&self) -> ContractVersion {
        self.contract_version
    }

    pub fn implementation_fingerprint(&self) -> &str {
        &self.implementation_fingerprint
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.capabilities
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderCompatibilityRequest {
    operation_id: OperationId,
    required_version: ContractVersion,
    required_capabilities: BTreeSet<CapabilityId>,
    required_weight_formats: BTreeSet<WeightFormatId>,
    required_quantization_formats: BTreeSet<QuantizationFormatId>,
    execution_determinism: ExecutionDeterminismRequirement,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderCompatibilityRequestWire {
    operation_id: OperationId,
    required_version: ContractVersion,
    required_capabilities: BTreeSet<CapabilityId>,
    required_weight_formats: BTreeSet<WeightFormatId>,
    required_quantization_formats: BTreeSet<QuantizationFormatId>,
    execution_determinism: ExecutionDeterminismRequirement,
}

impl<'de> Deserialize<'de> for ProviderCompatibilityRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderCompatibilityRequestWire::deserialize(deserializer)?;
        Self::new(
            wire.operation_id,
            wire.required_version,
            wire.required_capabilities,
            wire.required_weight_formats,
            wire.required_quantization_formats,
            wire.execution_determinism,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ProviderCompatibilityRequest {
    pub fn new(
        operation_id: OperationId,
        required_version: ContractVersion,
        required_capabilities: BTreeSet<CapabilityId>,
        required_weight_formats: BTreeSet<WeightFormatId>,
        required_quantization_formats: BTreeSet<QuantizationFormatId>,
        execution_determinism: ExecutionDeterminismRequirement,
    ) -> Result<Self, VNextError> {
        if required_version.major == 0 {
            return Err(invalid_operation(
                "provider compatibility request has a zero major version",
            ));
        }
        Ok(Self {
            operation_id,
            required_version,
            required_capabilities,
            required_weight_formats,
            required_quantization_formats,
            execution_determinism,
        })
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub const fn required_version(&self) -> ContractVersion {
        self.required_version
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn required_weight_formats(&self) -> &BTreeSet<WeightFormatId> {
        &self.required_weight_formats
    }

    pub fn required_quantization_formats(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.required_quantization_formats
    }

    pub const fn execution_determinism(&self) -> ExecutionDeterminismRequirement {
        self.execution_determinism
    }

    pub(super) fn extend_required_capabilities(
        &mut self,
        capabilities: impl IntoIterator<Item = CapabilityId>,
    ) {
        self.required_capabilities.extend(capabilities);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCompatibilityRejectReason {
    OperationVersionMismatch {
        required: ContractVersion,
        available: ContractVersion,
    },
    ProviderVersionMismatch {
        required: ContractVersion,
        available: ContractVersion,
    },
    MissingCapabilities {
        capabilities: BTreeSet<CapabilityId>,
    },
    UnsupportedWeightFormats {
        formats: BTreeSet<WeightFormatId>,
    },
    UnsupportedQuantizationFormats {
        formats: BTreeSet<QuantizationFormatId>,
    },
    InsufficientExecutionDeterminism {
        required: ExecutionDeterminismRequirement,
        available: ProviderExecutionSemantics,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCompatibilityRejection {
    pub provider_id: ProviderId,
    pub reasons: Vec<ProviderCompatibilityRejectReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderCompatibilityReport {
    request: ProviderCompatibilityRequest,
    compatible_provider_ids: Vec<ProviderId>,
    rejected: Vec<ProviderCompatibilityRejection>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderCompatibilityReportWire {
    request: ProviderCompatibilityRequest,
    compatible_provider_ids: Vec<ProviderId>,
    rejected: Vec<ProviderCompatibilityRejection>,
}

impl<'de> Deserialize<'de> for ProviderCompatibilityReport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderCompatibilityReportWire::deserialize(deserializer)?;
        let report = Self {
            request: wire.request,
            compatible_provider_ids: wire.compatible_provider_ids,
            rejected: wire.rejected,
        };
        report.validate_shape().map_err(serde::de::Error::custom)?;
        Ok(report)
    }
}

impl ProviderCompatibilityReport {
    pub(super) fn from_classification(
        request: ProviderCompatibilityRequest,
        compatible_provider_ids: Vec<ProviderId>,
        rejected: Vec<ProviderCompatibilityRejection>,
    ) -> Result<Self, VNextError> {
        let report = Self {
            request,
            compatible_provider_ids,
            rejected,
        };
        report.validate_shape()?;
        Ok(report)
    }

    fn rejection_summary(&self) -> String {
        serde_json::to_string(&self.rejected)
            .map(|rejected| format!("all providers were rejected: {rejected}"))
            .unwrap_or_else(|_| "all providers were rejected".to_owned())
    }

    fn validate_shape(&self) -> Result<(), VNextError> {
        let compatible = self.compatible_provider_ids.iter().collect::<BTreeSet<_>>();
        let rejected = self
            .rejected
            .iter()
            .map(|rejection| &rejection.provider_id)
            .collect::<BTreeSet<_>>();
        if compatible.len() != self.compatible_provider_ids.len()
            || rejected.len() != self.rejected.len()
            || (compatible.is_empty() && rejected.is_empty())
            || !compatible.is_disjoint(&rejected)
            || self
                .rejected
                .iter()
                .any(|rejection| rejection.reasons.is_empty())
            || self
                .compatible_provider_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .rejected
                .windows(2)
                .any(|pair| pair[0].provider_id >= pair[1].provider_id)
        {
            return Err(invalid_operation(
                "provider compatibility report is duplicate, overlapping, empty, or non-canonical",
            ));
        }
        Ok(())
    }

    pub fn request(&self) -> &ProviderCompatibilityRequest {
        &self.request
    }

    pub fn compatible_provider_ids(&self) -> &[ProviderId] {
        &self.compatible_provider_ids
    }

    pub fn rejected(&self) -> &[ProviderCompatibilityRejection] {
        &self.rejected
    }

    pub fn require_compatible(&self, device_id: &DeviceId) -> Result<(), VNextError> {
        if self.compatible_provider_ids.is_empty() {
            return Err(VNextError::UnsupportedOperation {
                node_id: None,
                operation_id: self.request.operation_id.to_string(),
                device_id: device_id.to_string(),
                reason: self.rejection_summary(),
            });
        }
        Ok(())
    }

    /// Requires one compatible provider while retaining the plan node that
    /// caused a missing capability or version failure.
    pub fn require_compatible_for_node(
        &self,
        device_id: &DeviceId,
        node_id: &NodeId,
    ) -> Result<(), VNextError> {
        if !self.compatible_provider_ids.is_empty() {
            return Ok(());
        }
        let operation_version_mismatch = self
            .rejected
            .iter()
            .flat_map(|rejection| &rejection.reasons)
            .find_map(|reason| match reason {
                ProviderCompatibilityRejectReason::OperationVersionMismatch {
                    required,
                    available,
                } => Some((*required, *available)),
                _ => None,
            });
        let provider_version_mismatch = self
            .rejected
            .iter()
            .map(|rejection| {
                rejection.reasons.iter().find_map(|reason| match reason {
                    ProviderCompatibilityRejectReason::ProviderVersionMismatch {
                        required,
                        available,
                    } => Some((*required, *available)),
                    _ => None,
                })
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|versions| {
                versions
                    .into_iter()
                    .max_by_key(|(_, available)| (available.major, available.minor))
            });
        if let Some((required, available)) =
            operation_version_mismatch.or(provider_version_mismatch)
        {
            return Err(VNextError::IncompatibleOperationVersion {
                node_id: Some(node_id.to_string()),
                operation_id: self.request.operation_id.to_string(),
                required_major: required.major,
                required_minor: required.minor,
                available_major: available.major,
                available_minor: available.minor,
            });
        }
        Err(VNextError::UnsupportedOperation {
            node_id: Some(node_id.to_string()),
            operation_id: self.request.operation_id.to_string(),
            device_id: device_id.to_string(),
            reason: self.rejection_summary(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationFailure {
    identity: ExecutionIdentityEnvelope,
    phase: ProfilePhase,
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedOperationFailure {
    identity: UnvalidatedExecutionIdentityParts,
    phase: ProfilePhase,
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnvalidatedOperationFailureWire {
    identity: UnvalidatedExecutionIdentityParts,
    phase: ProfilePhase,
    code: String,
    message: String,
    retryable: bool,
}

impl From<UnvalidatedOperationFailureWire> for UnvalidatedOperationFailure {
    fn from(wire: UnvalidatedOperationFailureWire) -> Self {
        Self {
            identity: wire.identity,
            phase: wire.phase,
            code: wire.code,
            message: wire.message,
            retryable: wire.retryable,
        }
    }
}

impl UnvalidatedOperationFailure {
    pub fn revalidate(
        self,
        expected_identity: &ExecutionIdentityEnvelope,
        expected_phase: ProfilePhase,
    ) -> Result<OperationFailure, VNextError> {
        let identity = ExecutionIdentityEnvelope::new(self.identity.into())?;
        if &identity != expected_identity || self.phase != expected_phase {
            return Err(invalid_operation(
                "serialized operation failure differs from the expected execution context",
            ));
        }
        OperationFailure::new(
            identity,
            self.phase,
            self.code,
            self.message,
            self.retryable,
        )
    }
}

impl OperationFailure {
    pub fn new(
        identity: ExecutionIdentityEnvelope,
        phase: ProfilePhase,
        code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> Result<Self, VNextError> {
        let code = code.into();
        let message = message.into();
        let parts = identity.parts();
        if parts.frame_id.is_none()
            || parts.node_invocation_id.is_none()
            || parts.node_id.is_none()
            || parts.operation_id.is_none()
            || parts.provider_id.is_none()
            || parts.device_id.is_none()
            || parts.plan_id.is_none()
            || parts.plan_hash.is_none()
            || parts.transaction_id.is_none()
            || parts.resource_pool_id.is_none()
            || parts.resource_pool_identity_fingerprint.is_none()
            || parts.provisioning_run_id.is_none()
            || parts.provisioning_request_id.is_none()
            || parts.active_sequence_slot.is_none()
            || parts.admission_generation.is_none()
            || parts.activation_epoch.is_none()
            || parts.runtime_implementation_fingerprint.is_none()
            || parts.active_sequence_fingerprint.is_none()
            || parts.completed_sequence_fingerprint.is_some()
            || parts.aborted_sequence_fingerprint.is_some()
            || parts.resource_id.is_some()
            || parts.resource_generation.is_some()
            || parts.resource_batch_fingerprint.is_some()
            || code.trim().is_empty()
            || message.trim().is_empty()
            || code.len() > 64
            || message
                .bytes()
                .any(|byte| byte.is_ascii_control() && !matches!(byte, b'\n' | b'\t'))
            || message.len() > 4096
            || !code
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(invalid_operation(
                "operation failure requires complete execution identity, code, and message",
            ));
        }
        Ok(Self {
            identity,
            phase,
            code,
            message,
            retryable,
        })
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub const fn phase(&self) -> ProfilePhase {
        self.phase
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

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedOperationFailure, VNextError> {
        if bytes.len() > MAX_OPERATION_FAILURE_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted operation failure",
                message: format!(
                    "operation failure wire size {} exceeds limit {}",
                    bytes.len(),
                    MAX_OPERATION_FAILURE_WIRE_BYTES
                ),
            });
        }
        serde_json::from_slice::<UnvalidatedOperationFailureWire>(bytes)
            .map(UnvalidatedOperationFailure::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted operation failure",
                message: error.to_string(),
            })
    }
}
