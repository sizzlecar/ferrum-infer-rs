use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize};

use crate::vnext::{ContractVersion, ProgramValueId, VNextError};

pub const PROGRAM_CHECKPOINT_INPUTS_VERSION: ContractVersion = ContractVersion::new(1, 0);

/// Explicit checkpoint input roles. Conditioning inputs are compared in full,
/// including shape, dtype and byte contents. Output-only inputs may be omitted
/// from matching only after the execution plan proves they cannot affect any
/// state. Names and tensor widths do not establish an input's semantic role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProgramCheckpointInputs {
    contract_version: ContractVersion,
    token_input: ProgramValueId,
    conditioning_inputs: BTreeSet<ProgramValueId>,
    #[serde(skip_serializing_if = "BTreeSet::is_empty")]
    output_only_inputs: BTreeSet<ProgramValueId>,
}

#[cfg(test)]
mod tests;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    contract_version: ContractVersion,
    token_input: ProgramValueId,
    conditioning_inputs: Vec<ProgramValueId>,
    #[serde(default)]
    output_only_inputs: Vec<ProgramValueId>,
}

impl<'de> Deserialize<'de> for ProgramCheckpointInputs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        if wire.contract_version != PROGRAM_CHECKPOINT_INPUTS_VERSION
            || wire
                .conditioning_inputs
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || wire
                .output_only_inputs
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(serde::de::Error::custom(
                "checkpoint input version or canonical input order is invalid",
            ));
        }
        Self::new(
            wire.token_input,
            wire.conditioning_inputs.into_iter().collect(),
        )
        .and_then(|inputs| {
            inputs.with_output_only_inputs(wire.output_only_inputs.into_iter().collect())
        })
        .map_err(serde::de::Error::custom)
    }
}

impl ProgramCheckpointInputs {
    pub fn new(
        token_input: ProgramValueId,
        conditioning_inputs: BTreeSet<ProgramValueId>,
    ) -> Result<Self, VNextError> {
        if conditioning_inputs.contains(&token_input) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "checkpoint token input cannot also be a conditioning input".to_owned(),
            });
        }
        Ok(Self {
            contract_version: PROGRAM_CHECKPOINT_INPUTS_VERSION,
            token_input,
            conditioning_inputs,
            output_only_inputs: BTreeSet::new(),
        })
    }

    /// Declare inputs that affect only current outputs, not continuation state.
    /// This declaration alone is not authority to skip matching: plan derivation
    /// checks all value, token-work and exact-alias paths to state effects.
    pub fn with_output_only_inputs(
        mut self,
        output_only_inputs: BTreeSet<ProgramValueId>,
    ) -> Result<Self, VNextError> {
        if output_only_inputs.contains(&self.token_input)
            || !output_only_inputs.is_disjoint(&self.conditioning_inputs)
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "checkpoint input roles must be disjoint".to_owned(),
            });
        }
        self.output_only_inputs = output_only_inputs;
        Ok(self)
    }

    pub fn token_input(&self) -> &ProgramValueId {
        &self.token_input
    }

    pub fn conditioning_inputs(&self) -> &BTreeSet<ProgramValueId> {
        &self.conditioning_inputs
    }

    pub fn output_only_inputs(&self) -> &BTreeSet<ProgramValueId> {
        &self.output_only_inputs
    }

    pub(crate) fn covers(&self, inputs: &[ProgramValueId]) -> bool {
        let declared = self
            .conditioning_inputs
            .iter()
            .chain(self.output_only_inputs.iter())
            .chain([&self.token_input])
            .collect::<BTreeSet<_>>();
        declared == inputs.iter().collect::<BTreeSet<_>>()
    }
}
