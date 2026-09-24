use super::super::report::Phase;
use super::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(in crate::commands::calibrate_slo) enum ReferenceRequestPolicy {
    OriginalInput {},
    FixedReferenceOutput {
        max_tokens: NonZeroUsize,
        eos: ReferenceEosPolicy,
    },
}
impl Default for ReferenceRequestPolicy {
    fn default() -> Self {
        Self::OriginalInput {}
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::commands::calibrate_slo) enum ReferenceEosPolicy {
    Respect,
    Ignore,
}
impl ReferenceEosPolicy {
    pub(in crate::commands::calibrate_slo) fn ignore(self) -> bool {
        matches!(self, Self::Ignore)
    }
}
impl ReferenceRequestPolicy {
    pub(in crate::commands::calibrate_slo) fn validate(self) -> Result<()> {
        if matches!(self, Self::FixedReferenceOutput { max_tokens, .. } if max_tokens.get() > 1_048_576)
        {
            return Err(invalid("reference max_tokens exceeds the request bound"));
        }
        Ok(())
    }
    pub(in crate::commands::calibrate_slo) fn output_tokens(self, original: usize) -> usize {
        match self {
            Self::OriginalInput {} => original,
            Self::FixedReferenceOutput { max_tokens, .. } => max_tokens.get(),
        }
    }
    pub(in crate::commands::calibrate_slo) fn independent(self, phase: Phase) -> bool {
        matches!(self, Self::FixedReferenceOutput { .. })
            && matches!(phase, Phase::Warmup | Phase::Discovery | Phase::Reference)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::commands::calibrate_slo) struct InputIdentityLedger {
    maximum_inputs: usize,
    entries: Vec<InputIdentity>,
}
#[derive(Debug, Clone, Serialize)]
struct InputIdentity {
    source_index: usize,
    actual_input: CalibrationRequestEvidence,
    /// True only after a real independent-reference frontier was observed.
    independent_reference_observed: bool,
    /// Never inferred from equal prompt bytes or a declared source budget.
    original_policy_observed: bool,
}
impl InputIdentityLedger {
    pub(in crate::commands::calibrate_slo) fn new(maximum_inputs: usize) -> Result<Self> {
        if maximum_inputs == 0 || maximum_inputs > 4096 {
            return Err(invalid(
                "input identity ledger requires a bounded recovered input set",
            ));
        }
        Ok(Self {
            maximum_inputs,
            entries: Vec::with_capacity(maximum_inputs),
        })
    }
    pub(in crate::commands::calibrate_slo) fn observe(
        &mut self,
        source_index: usize,
        input: CalibrationRequestEvidence,
        independent_reference: bool,
    ) -> Result<()> {
        if source_index >= self.maximum_inputs
            || input.original_input_tokens == 0
            || input.original_input_tokens_sha256 == [0; 32]
        {
            return Err(invalid("invalid actual input identity"));
        }
        if let Some(prior) = self
            .entries
            .iter_mut()
            .find(|entry| entry.source_index == source_index)
        {
            if prior.actual_input != input {
                return Err(invalid(
                    "same source changed its actual engine token length or digest",
                ));
            }
            prior.independent_reference_observed |= independent_reference;
            prior.original_policy_observed |= !independent_reference;
            return Ok(());
        }
        if self.entries.len() == self.maximum_inputs {
            return Err(invalid("input identity ledger capacity exhausted"));
        }
        self.entries.push(InputIdentity {
            source_index,
            actual_input: input,
            independent_reference_observed: independent_reference,
            original_policy_observed: !independent_reference,
        });
        Ok(())
    }
}

#[cfg(test)]
#[path = "request_policy_tests.rs"]
mod tests;
