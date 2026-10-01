//! Streaming query diagnostics over immutable original inputs. This does not
//! enroll tickets, change membership, select a close, or feed future phases.
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::Owner;

pub(super) struct InputAudit {
    pub query: StructuredQueryV2,
    actual_ns: u64,
    issued_at_ns: u64,
    generation: u64,
    block: u64,
}

impl InputAudit {
    pub fn checked(
        h: &StructuredServiceHeaderV7,
        wave: &StructuredServiceWaveV7,
        block: u64,
    ) -> Self {
        let mut contract = h.declaration.nonnegative_envelope.clone().unwrap();
        contract.algorithm_universe = None;
        let (raw, actual_ns, _) = physical::validate_parts(
            &h.fingerprint,
            h.opening.monotonic_ns,
            Some(&contract),
            h.opening.monotonic_ns,
            wave.ticket,
            wave.fifo,
            wave.issued_at_ns,
            &wave.host_stages,
            wave.independent.as_ref(),
            &mut physical::Frontiers::default(),
        )
        .unwrap();
        Self {
            query: StructuredQueryV2::exact(raw),
            actual_ns,
            issued_at_ns: wave.issued_at_ns,
            generation: h.generation,
            block,
        }
    }
}

#[derive(Default, serde::Serialize)]
struct Counts {
    original_inputs: usize,
    preceding_frozen_support_eligible: usize,
    qualified_prospective_support: usize,
    known: usize,
    unknown: BTreeMap<String, usize>,
    first_issue_ns: Option<u64>,
    last_issue_ns: Option<u64>,
    first_known_issue_ns: Option<u64>,
    last_known_issue_ns: Option<u64>,
    known_actual_sum_ns: u64,
    known_planning_sum_ns: u64,
    maximum_underestimate_ns: u64,
    maximum_overestimate_ns: u64,
}
impl Counts {
    fn unknown(&mut self, reason: impl Into<String>) {
        *self.unknown.entry(reason.into()).or_default() += 1;
    }
}

#[derive(Default, serde::Serialize)]
pub(super) struct OriginalCoverage {
    /// All original completed OrdinaryDecode/GreedyToken inputs after the
    /// successor opens. Other physical products are not its declared target.
    by_phase_rows_generation_block: BTreeMap<String, Counts>,
    other_products_or_roles: usize,
}

impl OriginalCoverage {
    pub fn observe(
        &mut self,
        input: &InputAudit,
        owner: Option<&Owner>,
        activated: Option<u64>,
        future_only: bool,
        fingerprint: &ProfileFingerprint,
    ) {
        let raw = input.query.input();
        if raw.owner().role != StructuredWaveRoleV2::OrdinaryDecode
            || raw.owner().product != StructuredProductV2::GreedyToken
        {
            self.other_products_or_roles += 1;
            return;
        }
        let qualified_at_issue = activated.is_some_and(|at| input.issued_at_ns >= at);
        let key = format!(
            "{}/rows={}/generation={}/block={}/{}",
            if qualified_at_issue {
                "after_qualification"
            } else {
                "before_qualification"
            },
            raw.owner().rows,
            input.generation,
            input.block,
            if future_only {
                "future_query_only"
            } else {
                "original_collection_stream"
            }
        );
        let counts = self.by_phase_rows_generation_block.entry(key).or_default();
        counts.original_inputs += 1;
        counts.first_issue_ns.get_or_insert(input.issued_at_ns);
        counts.last_issue_ns = Some(input.issued_at_ns);
        let Some(owner) = owner else {
            counts.unknown("no_discovered_successor_population");
            return;
        };
        let universe = owner
            .contract
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .algorithm_universe
            .as_ref()
            .unwrap();
        let key = match raw.numerical_family_key_for_universe(universe) {
            Ok(key) => key,
            Err(reason) => {
                counts.unknown(format!("population_projection:{reason:?}"));
                return;
            }
        };
        if Some(&key) != owner.scope.numerical_family.as_ref() {
            counts.unknown("declared_population_mismatch");
            return;
        }
        match &owner.state {
            OwnerState::Empty => counts.unknown("fit_collecting_no_qualified_model"),
            OwnerState::Fitted(model) => match model.service_input_membership(raw) {
                Ok(StructuredServiceInputMembershipV1::Eligible) => {
                    counts.preceding_frozen_support_eligible += 1;
                    counts.unknown("fit_support_only_no_qualified_model");
                }
                result => counts.unknown(format!("fit_support:{result:?}")),
            },
            OwnerState::Calibrated(model) => match model.service_input_membership(raw) {
                Ok(StructuredServiceInputMembershipV1::Eligible) => {
                    counts.preceding_frozen_support_eligible += 1;
                    counts.unknown("fit_residual_support_only_no_qualified_model");
                }
                result => counts.unknown(format!("residual_support:{result:?}")),
            },
            OwnerState::Qualified(model) if qualified_at_issue => {
                match model.catalog_input_membership(&input.query) {
                    Ok(Some(true)) => counts.qualified_prospective_support += 1,
                    Ok(Some(false)) => {
                        counts.unknown("outside_frozen_phase_input_support");
                        return;
                    }
                    result => {
                        counts.unknown(format!("catalog_membership:{result:?}"));
                        return;
                    }
                }
                match model.predict_query_detailed(
                    &fingerprint.clone().into(),
                    &input.query,
                    input.issued_at_ns,
                ) {
                    Ok(prediction) => {
                        counts.known += 1;
                        counts
                            .first_known_issue_ns
                            .get_or_insert(input.issued_at_ns);
                        counts.last_known_issue_ns = Some(input.issued_at_ns);
                        counts.known_actual_sum_ns = counts
                            .known_actual_sum_ns
                            .checked_add(input.actual_ns)
                            .unwrap();
                        counts.known_planning_sum_ns = counts
                            .known_planning_sum_ns
                            .checked_add(prediction.planning_ns)
                            .unwrap();
                        counts.maximum_underestimate_ns = counts
                            .maximum_underestimate_ns
                            .max(input.actual_ns.saturating_sub(prediction.planning_ns));
                        counts.maximum_overestimate_ns = counts
                            .maximum_overestimate_ns
                            .max(prediction.planning_ns.saturating_sub(input.actual_ns));
                    }
                    Err(reason) => counts.unknown(format!("prediction:{reason:?}")),
                }
            }
            OwnerState::Qualified(_) => counts.unknown("original_issue_precedes_qualification"),
            OwnerState::Failed(reason) => counts.unknown(format!("successor_failed:{reason}")),
        }
    }
}
