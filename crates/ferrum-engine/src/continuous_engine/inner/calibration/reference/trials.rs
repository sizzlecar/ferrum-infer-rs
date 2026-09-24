use super::*;

/// Only reference-training phases have keys. Heldout observations never enter
/// this collector and are evaluated separately against the imported profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationReferenceTrial {
    Prefill { curve: usize, repetition: usize },
    Decode { repetition: usize },
}
pub(super) struct Trial {
    pub request: RequestId,
    pub owner: u64,
    pub input: CalibrationRequestEvidence,
    pub prefix: u32,
    pub generated: u64,
    pub expected_generation: u64,
    pub samples: Vec<Witness>,
    pub selected: Vec<usize>,
    pub complete: bool,
}
impl CalibrationReferenceCollector {
    pub fn plan_sha256(&self) -> [u8; 32] {
        self.plan_sha256
    }
    pub fn frozen_accepted_ordinal(&self) -> u64 {
        self.frozen_accepted_ordinal
    }

    /// Bind a fresh actual owner before its first prefill. Decode-unit trials
    /// also record their genuine preparation chain, never a restored prefix.
    pub(super) fn begin_trial(
        &mut self,
        key: CalibrationReferenceTrial,
        frontier: &CalibrationFrontier,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.session, &frontier.session)
            || self.trials.contains_key(&key)
            || frontier.generated_tokens() != 0
            || frontier.prefill_progress()
                != Some((0, frontier.request_evidence().original_input_tokens))
            || frontier.kv_tokens() != 0
            || self.discovery.iter().any(|row| {
                row.witness.commit.owner_incarnation == frontier.owner_incarnation().get()
            })
            || self
                .trials
                .values()
                .any(|trial| trial.owner == frontier.owner_incarnation().get())
        {
            return Err(invalid(
                "reference trial requires a new fresh singleton owner",
            ));
        }
        let (repetition, input) = match key {
            CalibrationReferenceTrial::Prefill { curve, repetition } => {
                let curve = self
                    .plan
                    .curves
                    .get(curve)
                    .ok_or_else(|| invalid("unknown reference curve"))?;
                (
                    repetition,
                    CalibrationRequestEvidence {
                        original_input_tokens: curve.total_prompt_tokens.get() as usize,
                        original_input_tokens_sha256: curve.input_tokens_sha256,
                    },
                )
            }
            CalibrationReferenceTrial::Decode { repetition } => {
                let row = self
                    .discovery
                    .iter()
                    .find(|row| {
                        row.witness.matches(
                            &self.plan.protocol.decode_shape,
                            self.plan.protocol.decode_host,
                        )
                    })
                    .ok_or_else(|| invalid("decode discovery is missing"))?;
                (repetition, row.input)
            }
        };
        if repetition >= self.plan.protocol.repetitions.get()
            || *frontier.request_evidence() != input
        {
            return Err(invalid(
                "reference trial differs from frozen repetition/input",
            ));
        }
        self.trials.insert(
            key,
            Trial {
                request: frontier.request_id().clone(),
                owner: frontier.owner_incarnation().get(),
                input,
                prefix: 0,
                generated: 0,
                expected_generation: frontier.work_generation().get(),
                samples: Vec::new(),
                selected: Vec::new(),
                complete: false,
            },
        );
        Ok(())
    }

    /// Returns true once this predeclared trial's measured target is complete.
    /// Preparation receipts remain in the source join but do not become τ_ref.
    pub fn observe(
        &mut self,
        key: CalibrationReferenceTrial,
        report: &CalibrationWaveReport,
    ) -> Result<bool> {
        if self.retained >= self.plan.limits.max_samples.get() {
            return Err(invalid("reference receipt capacity exhausted"));
        }
        let witness = Witness::capture(report)?;
        let trial = self
            .trials
            .get_mut(&key)
            .ok_or_else(|| invalid("reference trial was not begun"))?;
        if trial.complete
            || witness.accepted <= self.frozen_accepted_ordinal
            || witness.sample.observed_at_ns < self.frozen_at_ns
            || profile::ProfileFingerprint::from(&witness.sample.fingerprint) != self.fingerprint
            || witness.commit.request_id != trial.request
            || witness.commit.owner_incarnation != trial.owner
            || witness.commit.work_generation != trial.expected_generation
            || trial.samples.last().is_some_and(|old| {
                old.accepted >= witness.accepted
                    || old.sample.observed_at_ns > witness.sample.observed_at_ns
            })
        {
            return Err(invalid(
                "reference trial crossed identity/order/freeze boundary",
            ));
        }
        let (prefix, generated) = match witness.commit.work {
            CalibrationCommittedWork::Prefill {
                start,
                end,
                total_prompt_tokens,
                generated_before,
                generated_after,
            } if start == trial.prefix
                && generated_before == trial.generated
                && trial.generated == 0
                && total_prompt_tokens as usize == trial.input.original_input_tokens =>
            {
                (end, generated_after)
            }
            CalibrationCommittedWork::Decode {
                kv_before,
                kv_after,
                generated_before,
                generated_after,
            } if kv_before == trial.prefix
                && generated_before == trial.generated
                && trial.generated > 0 =>
            {
                (kv_after, generated_after)
            }
            _ => {
                return Err(invalid(
                    "reference preparation/commit chain is discontinuous",
                ))
            }
        };
        let (selected, complete) = match key {
            CalibrationReferenceTrial::Prefill { curve, .. } => {
                let partition = &self.plan.curves[curve].partition;
                let expected = partition
                    .get(trial.selected.len())
                    .ok_or_else(|| invalid("reference trial exceeded frozen partition"))?;
                if !witness.matches(expected, self.plan.protocol.prefill_host) {
                    return Err(invalid(
                        "actual prefill shape/host differs from frozen protocol",
                    ));
                }
                (true, trial.selected.len() + 1 == partition.len())
            }
            CalibrationReferenceTrial::Decode { .. } => {
                let target = self.plan.protocol.decode_host.state.generated_tokens_before;
                let selected = matches!(witness.commit.work,CalibrationCommittedWork::Decode{generated_before,..} if generated_before==target);
                if selected
                    && !witness.matches(
                        &self.plan.protocol.decode_shape,
                        self.plan.protocol.decode_host,
                    )
                {
                    return Err(invalid("actual decode unit differs from frozen protocol"));
                }
                if !selected && generated > target {
                    return Err(invalid("decode reference target was skipped"));
                }
                (selected, selected)
            }
        };
        // Physical prefill/decode publication advances the engine cost
        // frontier once. Restore/recompute has its own advance; an omitted
        // transition therefore cannot be relabelled as direct continuation.
        let next_generation = witness
            .commit
            .work_generation
            .checked_add(1)
            .ok_or_else(|| invalid("reference generation exhausted"))?;
        trial.prefix = prefix;
        trial.generated = generated;
        trial.expected_generation = next_generation;
        if selected {
            trial.selected.push(trial.samples.len());
        }
        trial.samples.push(witness);
        trial.complete = complete;
        self.retained += 1;
        Ok(complete)
    }
}
