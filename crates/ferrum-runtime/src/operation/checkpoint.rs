//! Checkpoint continuation capabilities of a chosen operation implementation.
//! Reusable device execution (`ProviderReplayEquivalence`) is independent.

use std::num::NonZeroU64;

use serde::{Deserialize, Deserializer, Serialize};

use super::super::{CheckpointInputDependency, ContractVersion, VNextError};
use super::foundation::invalid_operation;

mod state_port;
pub use state_port::{ProviderCheckpointStateLayout, ProviderCheckpointStatePort};

pub const PROVIDER_CHECKPOINT_CONTRACT_VERSION: ContractVersion = ContractVersion::new(1, 0);

/// A positive token span accepted by an implementation. Both the minimum and
/// alignment are explicit; construction rejects an unreachable rounded minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointTokenSpanConstraint {
    minimum_tokens: NonZeroU64,
    alignment: NonZeroU64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointTokenSpanConstraintWire {
    minimum_tokens: NonZeroU64,
    alignment: NonZeroU64,
}

impl<'de> Deserialize<'de> for CheckpointTokenSpanConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CheckpointTokenSpanConstraintWire::deserialize(deserializer)?;
        Self::new(wire.minimum_tokens, wire.alignment).map_err(serde::de::Error::custom)
    }
}

impl CheckpointTokenSpanConstraint {
    pub fn new(minimum_tokens: NonZeroU64, alignment: NonZeroU64) -> Result<Self, VNextError> {
        let span = Self {
            minimum_tokens,
            alignment,
        };
        if span.first_legal_tokens().is_none() {
            return Err(invalid_operation(
                "checkpoint token span has no representable aligned length",
            ));
        }
        Ok(span)
    }

    pub const fn any_positive() -> Self {
        Self {
            minimum_tokens: NonZeroU64::MIN,
            alignment: NonZeroU64::MIN,
        }
    }

    pub const fn minimum_tokens(&self) -> NonZeroU64 {
        self.minimum_tokens
    }

    pub const fn alignment(&self) -> NonZeroU64 {
        self.alignment
    }

    pub fn permits(&self, tokens: u64) -> bool {
        tokens >= self.minimum_tokens.get() && tokens.is_multiple_of(self.alignment.get())
    }

    fn first_legal_tokens(&self) -> Option<u64> {
        let multiple = (self.minimum_tokens.get() - 1) / self.alignment.get() + 1;
        multiple.checked_mul(self.alignment.get())
    }
}

/// Legality of computing a span from M to a boundary N and continuing with a
/// nonempty suffix. These constrain span lengths, not absolute token positions.
/// An implementation that additionally requires absolute-position alignment
/// cannot use this contract without a separate, versioned boundary declaration.
/// Plans intersect all declarations; validating only the first span is insufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointBoundaryConstraint {
    prefix: CheckpointTokenSpanConstraint,
    suffix: CheckpointTokenSpanConstraint,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointBoundaryConstraintWire {
    prefix: CheckpointTokenSpanConstraint,
    suffix: CheckpointTokenSpanConstraint,
}

impl<'de> Deserialize<'de> for CheckpointBoundaryConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CheckpointBoundaryConstraintWire::deserialize(deserializer)?;
        Self::new(wire.prefix, wire.suffix).map_err(serde::de::Error::custom)
    }
}

impl CheckpointBoundaryConstraint {
    pub fn new(
        prefix: CheckpointTokenSpanConstraint,
        suffix: CheckpointTokenSpanConstraint,
    ) -> Result<Self, VNextError> {
        if prefix
            .first_legal_tokens()
            .zip(suffix.first_legal_tokens())
            .and_then(|(prefix, suffix)| prefix.checked_add(suffix))
            .is_none()
        {
            return Err(invalid_operation(
                "checkpoint boundary has no representable prefix and nonempty suffix",
            ));
        }
        Ok(Self { prefix, suffix })
    }

    pub const fn any_positive() -> Self {
        Self {
            prefix: CheckpointTokenSpanConstraint::any_positive(),
            suffix: CheckpointTokenSpanConstraint::any_positive(),
        }
    }

    pub const fn prefix(&self) -> CheckpointTokenSpanConstraint {
        self.prefix
    }

    pub const fn suffix(&self) -> CheckpointTokenSpanConstraint {
        self.suffix
    }

    pub fn permits(&self, prefix_tokens: u64, prompt_tokens: u64) -> bool {
        self.permits_from(0, prefix_tokens, prompt_tokens)
    }

    /// Validates both actual execution spans, including after restoration to M.
    /// Neither a zero-length capture frame nor an empty suffix is permitted.
    pub fn permits_from(
        &self,
        processed_tokens: u64,
        boundary_tokens: u64,
        prompt_tokens: u64,
    ) -> bool {
        boundary_tokens
            .checked_sub(processed_tokens)
            .zip(prompt_tokens.checked_sub(boundary_tokens))
            .is_some_and(|(prefix, suffix)| {
                self.prefix.permits(prefix) && self.suffix.permits(suffix)
            })
    }
}

/// Numerical reference for checkpoint continuation and any stronger promise
/// about repartitioning. Every supported contract requires bitwise continuation
/// from identical complete state with identical suffix inputs, execution
/// partitions, implementation choices, and numerical/runtime identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointPartitionNumerics {
    /// Adopt the complete boundary state of an authenticated successful source
    /// execution. Restoring all of that state preserves subsequent outputs and
    /// state effects under the identical-suffix conditions above. No uncaptured
    /// execution history may affect continuation. This does not promise that
    /// recomputing the prefix with other partitions produces identical state.
    /// The actual capture and native restore identities bind the adopted state;
    /// a caller-supplied expected history is neither needed nor sufficient.
    CapturedExecutionContinuation,
    /// Reuse needs identical execution partitions, also in the matching identity.
    SamePartitionOnly,
    /// Legal repartitioning preserves all outputs and persistent state effects.
    BitwiseEquivalent,
    /// Legal repartitioning obeys the *owning provider descriptor's exact
    /// operation_fingerprint* oracle against the same unpartitioned execution,
    /// including every persistent state effect. This is not a new tolerance.
    /// Plan validation must reject this declaration if that operation's oracle
    /// does not actually cover the state effects. It does not weaken the
    /// identical-partition bitwise continuation requirement.
    OperationOracle,
}

/// Whether a successful frame may be captured at the end of its complete input.
/// This is separate from the existing partial-input boundary declaration: an
/// older provider has not promised that its final frame leaves resumable state.
/// Even when supported, restoration still requires a legal nonempty suffix and
/// all of the provider's input-dependency and numerical conditions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointCompletedInputCapture {
    #[default]
    Unsupported,
    Supported,
}

impl CheckpointCompletedInputCapture {
    pub const fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported)
    }
}

/// An implementation promises all of its persistent state effects are complete
/// at the permitted successful frame boundaries, and legal suffix execution can
/// resume from those values without hidden execution history. Physical export
/// mappings, operation-oracle coverage, and whole-program closure are separate
/// plan checks; this declaration alone does not authorize restoration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderCheckpointContract {
    contract_version: ContractVersion,
    input_dependency: CheckpointInputDependency,
    boundaries: CheckpointBoundaryConstraint,
    partition_numerics: CheckpointPartitionNumerics,
    #[serde(skip_serializing_if = "CheckpointCompletedInputCapture::is_unsupported")]
    completed_input_capture: CheckpointCompletedInputCapture,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    state_ports: Vec<ProviderCheckpointStatePort>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderCheckpointContractWire {
    contract_version: ContractVersion,
    input_dependency: CheckpointInputDependency,
    boundaries: CheckpointBoundaryConstraint,
    partition_numerics: CheckpointPartitionNumerics,
    #[serde(default)]
    completed_input_capture: CheckpointCompletedInputCapture,
    #[serde(default)]
    state_ports: Vec<ProviderCheckpointStatePort>,
}

impl<'de> Deserialize<'de> for ProviderCheckpointContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderCheckpointContractWire::deserialize(deserializer)?;
        if wire.contract_version != PROVIDER_CHECKPOINT_CONTRACT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "provider checkpoint contract version {} is unsupported",
                wire.contract_version
            )));
        }
        let original_ports = wire.state_ports.clone();
        let contract = Self::new(
            wire.input_dependency,
            wire.boundaries,
            wire.partition_numerics,
        )
        .with_completed_input_capture(wire.completed_input_capture)
        .with_state_ports(wire.state_ports)
        .map_err(serde::de::Error::custom)?;
        if contract.state_ports != original_ports {
            return Err(serde::de::Error::custom(
                "checkpoint state ports are not canonical",
            ));
        }
        Ok(contract)
    }
}

impl ProviderCheckpointContract {
    pub const fn new(
        input_dependency: CheckpointInputDependency,
        boundaries: CheckpointBoundaryConstraint,
        partition_numerics: CheckpointPartitionNumerics,
    ) -> Self {
        Self {
            contract_version: PROVIDER_CHECKPOINT_CONTRACT_VERSION,
            input_dependency,
            boundaries,
            partition_numerics,
            completed_input_capture: CheckpointCompletedInputCapture::Unsupported,
            state_ports: Vec::new(),
        }
    }

    pub const fn with_completed_input_capture(
        mut self,
        capability: CheckpointCompletedInputCapture,
    ) -> Self {
        self.completed_input_capture = capability;
        self
    }

    pub const fn completed_input_capture(&self) -> CheckpointCompletedInputCapture {
        self.completed_input_capture
    }

    pub fn with_state_ports(
        mut self,
        mut ports: Vec<ProviderCheckpointStatePort>,
    ) -> Result<Self, VNextError> {
        ports.sort_by_key(ProviderCheckpointStatePort::key);
        if ports.windows(2).any(|pair| pair[0].key() == pair[1].key()) {
            return Err(invalid_operation(
                "duplicate checkpoint state port/storage ABI",
            ));
        }
        self.state_ports = ports;
        Ok(self)
    }

    pub fn state_ports(&self) -> &[ProviderCheckpointStatePort] {
        &self.state_ports
    }

    pub const fn contract_version(&self) -> ContractVersion {
        self.contract_version
    }

    pub const fn input_dependency(&self) -> CheckpointInputDependency {
        self.input_dependency
    }

    pub const fn boundaries(&self) -> CheckpointBoundaryConstraint {
        self.boundaries
    }

    pub const fn partition_numerics(&self) -> CheckpointPartitionNumerics {
        self.partition_numerics
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderCheckpointCapability {
    #[default]
    Unsupported,
    CompletedBoundary(ProviderCheckpointContract),
}

impl ProviderCheckpointCapability {
    pub const fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported)
    }
}

#[cfg(test)]
mod tests;
