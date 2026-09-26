//! Explicit source5 entry. Static declarations are checked against the actual
//! tokenizer before writing the header; installation stays manual-session only.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixPlanV5;
use std::num::NonZeroUsize;

impl CalibrationSession {
    pub async fn begin_structured_prefix_cost_group_v5(
        &mut self,
        options: StructuredCalibrationGroupOptionsV2,
        prefixes: StructuredPrefixPlanV5,
    ) -> Result<()> {
        let requests = &options
            .children
            .first()
            .ok_or_else(|| FerrumError::config("source5 needs declared children"))?
            .cohort_plan;
        prefixes
            .validate(requests)
            .map_err(|e| FerrumError::config(format!("source5 prefix plan: {e:?}")))?;
        let tokenizer = self.engine.inner.tokenizer.as_ref();
        let byte_bound = tokenizer.bounded_token_bytes_bound().ok_or_else(|| {
            FerrumError::unsupported("source5 requires actual bounded tokenizer bytes")
        })?;
        let mut bytes = vec![0; byte_bound.get()];
        for (cohorts, original) in prefixes.phases.iter().zip(&requests.phases) {
            for (cohort, original) in cohorts.iter().zip(original) {
                let Some(cohort) = cohort else {
                    continue;
                };
                for (slot, request) in cohort.slots.iter().zip(&original.requests) {
                    let maximum = usize::try_from(request.maximum_output)
                        .ok()
                        .and_then(NonZeroUsize::new)
                        .ok_or_else(|| {
                            FerrumError::config("source5 request output bound overflow")
                        })?;
                    let release = usize::try_from(cohort.release_generated)
                        .map_err(|_| FerrumError::config("source5 prefix bound overflow"))?;
                    if release > self.context_capacity() {
                        return Err(FerrumError::config(
                            "source5 prefix exceeds actual context capacity",
                        ));
                    }
                    CalibrationPrefixTokensV1 {
                        tokenizer_policy_sha256: slot.tokenizer_policy_sha256,
                        token_ids: slot.token_ids.clone(),
                        release_generated: release,
                    }
                    .validate_for_tokenizer(tokenizer, maximum)?;
                    for (&token, expected) in slot.token_ids.iter().zip(&slot.token_bytes) {
                        let n = tokenizer
                            .token_bytes_bounded_into(token, &mut bytes)
                            .map_err(|_| {
                                FerrumError::invalid_request("source5 actual tokenizer read failed")
                            })?
                            .ok_or_else(|| {
                                FerrumError::invalid_request("source5 token is not known")
                            })?;
                        if bytes.get(..n) != Some(expected.as_slice()) {
                            return Err(FerrumError::invalid_request(
                                "source5 declared bytes differ from actual tokenizer",
                            ));
                        }
                    }
                }
            }
        }
        self.begin_structured_group_inner(options, Some(prefixes))
            .await
    }
    pub(super) fn group_phase_boundary(&self) -> Result<()> {
        if self.prefix_source5 {
            self.completed_owner_boundary()
        } else {
            self.selected_phase_boundary()
        }
    }
}
