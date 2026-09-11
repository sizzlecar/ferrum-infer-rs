use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize};

use crate::vnext::{ContractVersion, ProgramValueId, VNextError};

pub const PROGRAM_CHECKPOINT_INPUTS_VERSION: ContractVersion = ContractVersion::new(1, 0);

/// Inputs whose complete canonical content must be supplied to checkpoint
/// matching. Token IDs have an explicit role; all other inputs are compared in
/// full, including shape, dtype and byte contents. Names and tensor widths do
/// not establish an input's semantic role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProgramCheckpointInputs {
    contract_version: ContractVersion,
    token_input: ProgramValueId,
    conditioning_inputs: BTreeSet<ProgramValueId>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    contract_version: ContractVersion,
    token_input: ProgramValueId,
    conditioning_inputs: Vec<ProgramValueId>,
}

impl<'de> Deserialize<'de> for ProgramCheckpointInputs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        if wire.contract_version != PROGRAM_CHECKPOINT_INPUTS_VERSION
            || wire
                .conditioning_inputs
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
        })
    }

    pub fn token_input(&self) -> &ProgramValueId {
        &self.token_input
    }

    pub fn conditioning_inputs(&self) -> &BTreeSet<ProgramValueId> {
        &self.conditioning_inputs
    }

    pub(crate) fn covers(&self, inputs: &[ProgramValueId]) -> bool {
        let declared = self
            .conditioning_inputs
            .iter()
            .chain([&self.token_input])
            .collect::<BTreeSet<_>>();
        declared == inputs.iter().collect::<BTreeSet<_>>()
    }
}
