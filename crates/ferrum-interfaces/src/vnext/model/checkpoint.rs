//! Semantic declarations only: these do not authorize copying state or prove
//! that a selected provider/storage layout implements checkpoint restoration.

use serde::{Deserialize, Deserializer, Serialize};

use super::{ContractVersion, StateLifetime, VNextError};

mod inputs;
pub use inputs::{ProgramCheckpointInputs, PROGRAM_CHECKPOINT_INPUTS_VERSION};

pub const STATE_CHECKPOINT_CONTRACT_VERSION: ContractVersion = ContractVersion::new(1, 0);

/// Token input that must remain identical when a completed state is reused.
/// Both variants additionally require the same plan, numerical/position
/// semantics, and every non-token conditioning input in the matching identity.
/// An uncovered dependency makes the plan unsupported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointInputDependency {
    /// The state at N is independent of tokens after N. A different legal
    /// suffix may follow the exact token prefix [0, N).
    ExactTokenPrefix,
    /// The complete token input, including its length and suffix, affects the
    /// state at N. Matching only [0, N) is insufficient.
    EntireTokenInput,
}

/// Logical contents valid at one completed boundary N. Physical regions,
/// padding, aliases, and initialization coverage must be resolved separately.
/// This is not inferred from a capacity formula or tensor/state name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateCheckpointContents {
    /// The ordered logical positions [0, N), with one semantic tensor value
    /// per position. Unwritten capacity after N is not part of the checkpoint.
    PrefixPositions,
    /// The entire semantic tensor value at exactly N, with no earlier values.
    /// It cannot be shortened to represent an earlier boundary.
    BoundaryValue,
}

/// Complete continuation state at a successful frame boundary. The program
/// still has to account for *all* persistent effects and conditioning inputs;
/// declaring one state does not establish that closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StateCheckpointContract {
    contract_version: ContractVersion,
    contents: StateCheckpointContents,
    input_dependency: CheckpointInputDependency,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateCheckpointContractWire {
    contract_version: ContractVersion,
    contents: StateCheckpointContents,
    input_dependency: CheckpointInputDependency,
}

impl<'de> Deserialize<'de> for StateCheckpointContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StateCheckpointContractWire::deserialize(deserializer)?;
        if wire.contract_version != STATE_CHECKPOINT_CONTRACT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "state checkpoint contract version {} is unsupported",
                wire.contract_version
            )));
        }
        Ok(Self::new(wire.contents, wire.input_dependency))
    }
}

impl StateCheckpointContract {
    pub const fn new(
        contents: StateCheckpointContents,
        input_dependency: CheckpointInputDependency,
    ) -> Self {
        Self {
            contract_version: STATE_CHECKPOINT_CONTRACT_VERSION,
            contents,
            input_dependency,
        }
    }

    pub const fn contract_version(&self) -> ContractVersion {
        self.contract_version
    }

    pub const fn contents(&self) -> StateCheckpointContents {
        self.contents
    }

    pub const fn input_dependency(&self) -> CheckpointInputDependency {
        self.input_dependency
    }
}

/// Missing declarations remain unsupported, including on old wire payloads.
/// Unsupported is omitted from StateSpec serialization so ordinary-compute
/// identities remain unchanged; an explicit contract changes that identity.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum StateCheckpointCapability {
    #[default]
    Unsupported,
    CompletedBoundary(StateCheckpointContract),
}

impl StateCheckpointCapability {
    pub const fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported)
    }

    pub(super) fn validate_lifetime(self, lifetime: StateLifetime) -> Result<(), VNextError> {
        if !self.is_unsupported() && lifetime != StateLifetime::Sequence {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "checkpoint contracts currently support only Sequence state; Request/Step state requires a separate closure contract".to_owned(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
