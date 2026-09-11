use std::collections::BTreeMap;

use super::*;
use crate::vnext::ResolvedTensorLayout;

/// Full canonical content of one non-token input. Tensor identity and all bytes
/// participate in equality; a hash alone is never the final matching proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointCanonicalInput {
    tensor: ResolvedTensorSpec,
    bytes: Vec<u8>,
}

impl CheckpointCanonicalInput {
    pub fn new(tensor: ResolvedTensorSpec, bytes: Vec<u8>) -> Result<Self, VNextError> {
        if !matches!(tensor.layout(), ResolvedTensorLayout::Contiguous)
            || u64::try_from(bytes.len()).ok() != Some(tensor.minimum_storage_bytes()?)
        {
            return Err(invalid_plan(
                "checkpoint input requires exact contiguous canonical tensor bytes",
            ));
        }
        Ok(Self { tensor, bytes })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceCheckpointInputIdentity {
    layout_fingerprint: String,
    token_input: ProgramValueId,
    tokens: Vec<u32>,
    conditioning: BTreeMap<ProgramValueId, CheckpointCanonicalInput>,
    requires_entire_input: bool,
}

impl SequenceCheckpointInputIdentity {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Checks exact contents after an indexed lookup. This is only the input
    /// portion of matching: plan/loading instance, saved N and numerical
    /// partition identity remain mandatory independent checks.
    pub fn matches_at(&self, target: &Self, boundary: usize) -> bool {
        boundary > 0
            && boundary < target.tokens.len()
            && boundary < self.tokens.len()
            && self.layout_fingerprint == target.layout_fingerprint
            && self.token_input == target.token_input
            && self.conditioning == target.conditioning
            && self.requires_entire_input == target.requires_entire_input
            && self.tokens[..boundary] == target.tokens[..boundary]
            && (!self.requires_entire_input || self.tokens == target.tokens)
    }
}

impl SequenceCheckpointLayout {
    pub fn bind_inputs(
        &self,
        token_input: &ProgramValueId,
        tokens: &[u32],
        conditioning: &BTreeMap<ProgramValueId, CheckpointCanonicalInput>,
    ) -> Result<SequenceCheckpointInputIdentity, Vec<SequenceCheckpointUnsupportedReason>> {
        use SequenceCheckpointUnsupportedReason as Reason;
        let declared = self.inputs();
        let mut reasons = Vec::new();
        if token_input != declared.token_input() || tokens.is_empty() {
            reasons.push(Reason::InvalidInput {
                value_id: declared.token_input().clone(),
            });
        }
        for value_id in declared.conditioning_inputs() {
            if !conditioning.contains_key(value_id) {
                reasons.push(Reason::MissingInput {
                    value_id: value_id.clone(),
                });
            }
        }
        for value_id in conditioning.keys() {
            if !declared.conditioning_inputs().contains(value_id) {
                reasons.push(Reason::InvalidInput {
                    value_id: value_id.clone(),
                });
            }
        }
        if !reasons.is_empty() {
            return Err(reasons);
        }
        let layout_fingerprint = self
            .fingerprint()
            .map_err(|_| vec![Reason::InputCoverage])?;
        Ok(SequenceCheckpointInputIdentity {
            layout_fingerprint,
            token_input: token_input.clone(),
            tokens: tokens.to_vec(),
            conditioning: conditioning.clone(),
            requires_entire_input: self.input_dependency()
                == CheckpointInputDependency::EntireTokenInput,
        })
    }
}
