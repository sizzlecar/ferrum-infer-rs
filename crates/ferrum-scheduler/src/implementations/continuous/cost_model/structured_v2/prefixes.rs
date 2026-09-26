//! Immutable preparation declarations, not sampler or calibration authority.
//! The live engine separately checks every token against its actual tokenizer
//! and request constraints before installing a private capability.
use super::{windows::CohortPlanV2, StructuredUnknownV2};
use ferrum_interfaces::output_flow::advance_committed_utf8_fragment;
use ferrum_types::TokenId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PREFIX_PLAN_REVISION_V5: &str = "ferrum.structured-prefix-preparation.v5";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPrefixSlotV5 {
    pub tokenizer_policy_sha256: [u8; 32],
    pub token_ids: Vec<TokenId>,
    /// Actual bounded tokenizer bytes, bound before the first preparation
    /// wave. Replay checks the trajectory; it does not recompute model logits.
    pub token_bytes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPrefixCohortV5 {
    /// Same generated frontier for every slot; measurement starts only after
    /// all slots revoke their preparation capability at this exact frontier.
    pub release_generated: u64,
    pub slots: Vec<StructuredPrefixSlotV5>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPrefixPlanV5 {
    /// Expanded fit/residual/qualification cohorts, position-aligned with the
    /// full request plan. None declares ordinary cohorts including prefill.
    pub phases: [Vec<Option<StructuredPrefixCohortV5>>; 3],
}

impl StructuredPrefixSlotV5 {
    pub fn expected_pending(&self) -> Result<Vec<u8>, StructuredUnknownV2> {
        if self.token_ids.is_empty() || self.token_ids.len() != self.token_bytes.len() {
            return Err(StructuredUnknownV2::InvalidSettings);
        }
        let mut pending = Vec::new();
        for bytes in &self.token_bytes {
            pending = advance_committed_utf8_fragment(&pending, bytes)
                .map_err(|_| StructuredUnknownV2::InvalidInput)?;
        }
        Ok(pending)
    }
}

impl StructuredPrefixPlanV5 {
    /// This declaration check is bounded by the source's existing whole-file
    /// and 8 MiB record limits. Actual context/vocabulary/output checks remain
    /// mandatory in the live engine; this creates no receipt.
    pub fn validate(&self, requests: &CohortPlanV2) -> Result<(), StructuredUnknownV2> {
        requests.validate()?;
        for (phase, declared) in self.phases.iter().zip(&requests.phases) {
            if phase.len() != declared.len() {
                return Err(StructuredUnknownV2::InvalidSettings);
            }
            for (prefix, cohort) in phase.iter().zip(declared) {
                let Some(prefix) = prefix else {
                    continue;
                };
                if prefix.release_generated == 0 || prefix.slots.len() != cohort.requests.len() {
                    return Err(StructuredUnknownV2::InvalidSettings);
                }
                for (slot, request) in prefix.slots.iter().zip(&cohort.requests) {
                    if prefix.release_generated >= request.maximum_output
                        || u64::try_from(slot.token_ids.len()).ok()
                            != Some(prefix.release_generated)
                    {
                        return Err(StructuredUnknownV2::InvalidSettings);
                    }
                    slot.expected_pending()?;
                }
            }
        }
        Ok(())
    }

    pub fn signature(&self, requests: &CohortPlanV2) -> Result<[u8; 32], StructuredUnknownV2> {
        self.validate(requests)?;
        let mut hash = Sha256::new();
        hash.update(PREFIX_PLAN_REVISION_V5.as_bytes());
        hash.update([0]);
        struct HashWriter<'a>(&'a mut Sha256);
        impl std::io::Write for HashWriter<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        serde_json::to_writer(HashWriter(&mut hash), &(requests, self))
            .map_err(|_| StructuredUnknownV2::InvalidInput)?;
        Ok(hash.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::super::windows::{CohortRequestV2, CohortV2};
    use super::*;

    fn declarations() -> (CohortPlanV2, StructuredPrefixPlanV5) {
        let requests = CohortPlanV2 {
            phases: std::array::from_fn(|_| {
                vec![CohortV2 {
                    manifest_case: 0,
                    repetition: 0,
                    requests: vec![
                        CohortRequestV2 {
                            manifest_prompt: 0,
                            maximum_output: 4
                        };
                        2
                    ],
                }]
            }),
        };
        let plan = StructuredPrefixPlanV5 {
            phases: std::array::from_fn(|_| {
                vec![Some(StructuredPrefixCohortV5 {
                    release_generated: 2,
                    slots: vec![
                        StructuredPrefixSlotV5 {
                            tokenizer_policy_sha256: [1; 32],
                            token_ids: vec![TokenId::new(1), TokenId::new(2)],
                            token_bytes: vec![b"a".to_vec(), vec![0xc3]],
                        },
                        StructuredPrefixSlotV5 {
                            tokenizer_policy_sha256: [1; 32],
                            token_ids: vec![TokenId::new(1), TokenId::new(3)],
                            token_bytes: vec![b"a".to_vec(), b"b".to_vec()],
                        },
                    ],
                })]
            }),
        };
        (requests, plan)
    }

    #[test]
    fn prefix_plan_preserves_joint_slot_pending_and_binds_original_full_population() {
        let (requests, plan) = declarations();
        plan.validate(&requests).unwrap();
        let c = plan.phases[0][0].as_ref().unwrap();
        assert_eq!(c.slots[0].expected_pending().unwrap(), [0xc3]);
        assert!(c.slots[1].expected_pending().unwrap().is_empty());
        let signature = plan.signature(&requests).unwrap();
        let mut other = plan.clone();
        other.phases[0][0].as_mut().unwrap().slots.swap(0, 1);
        assert_ne!(signature, other.signature(&requests).unwrap());
        let mut ordinary = plan.clone();
        ordinary.phases[0][0] = None;
        ordinary.validate(&requests).unwrap();
        assert_ne!(signature, ordinary.signature(&requests).unwrap());
    }

    #[test]
    fn prefix_plan_rejects_shortened_length_missing_slot_and_invalid_utf8() {
        let (requests, plan) = declarations();
        let mut bad = plan.clone();
        bad.phases[0][0].as_mut().unwrap().release_generated = 4;
        assert!(bad.validate(&requests).is_err());
        let mut bad = plan.clone();
        bad.phases[0][0].as_mut().unwrap().slots.pop();
        assert!(bad.validate(&requests).is_err());
        let mut bad = plan.clone();
        bad.phases[0][0].as_mut().unwrap().slots[0].token_bytes[1] = vec![0xa9];
        assert!(bad.validate(&requests).is_err());
        let mut bad = plan;
        bad.phases[0].clear();
        assert!(bad.validate(&requests).is_err());
    }
}
