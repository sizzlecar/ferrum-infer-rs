//! Algorithmic fixtures, not evidence of any hardware cost coverage.
use super::*;
use ferrum_interfaces::execution_cost::{
    CanonicalWaveCostFeatures, CanonicalWaveCostShape, CostRowNumericFeatures,
    HostContentCostFeaturesV1,
};
use std::cell::{Cell, RefCell};

struct Routes {
    first_alternative: bool,
    count: usize,
}
impl PlanningShapeResolver for Routes {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        TestResolver.resolve(query, poll)
    }
    fn resolve_domain(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<PlanningShapeDomain<CanonicalWaveCostShape>>, PlanningUnknownReason> {
        let mut shape = self.resolve(query, poll)?.unwrap();
        if query.prior_waves.is_empty() && !self.first_alternative {
            return Ok(Some(PlanningShapeDomain::Exact(shape)));
        }
        shape.host_content_features = Some(HostContentCostFeaturesV1 {
            schema_version: 1,
            output_policy_signature: [9; 32],
        });
        shape.numeric_features = Some(CanonicalWaveCostFeatures {
            schema_version: 1,
            output_policy_signature: [8; 32],
            rows: query
                .rows
                .iter()
                .map(|row| {
                    let n = u64::from(row.request.timing.committed_tokens);
                    CostRowNumericFeatures {
                        generated_tokens_before: n,
                        maximum_output_tokens: u64::from(
                            row.request.timing.maximum_output_tokens.get(),
                        ),
                        sampling_history_tokens: n,
                        repetition_tokens: 0,
                        decoded_prefix_tokens: n + 1,
                        decoded_text_bytes_bound: n + 1,
                        decode_scratch_bytes_bound: 0,
                    }
                })
                .collect(),
        });
        let mut alternatives = Vec::new();
        for index in 0..self.count {
            let mut shape = shape.clone();
            shape.provider_signature = [index as u8; 32];
            alternatives.push(shape);
        }
        Ok(Some(PlanningShapeDomain::HostContentAlternatives(
            alternatives,
        )))
    }
}

struct Envelope {
    enabled: bool,
    missing: bool,
    expiry: u64,
    calls: RefCell<Vec<(u8, u64)>>,
}
impl PlanningCostModel for Envelope {
    fn model_version(&self) -> u64 {
        7
    }
    fn supports_empirical_host_content(&self) -> bool {
        self.enabled
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        now: u64,
    ) -> Option<PlanningCost> {
        let tag = if shape.host_content_features.is_some() {
            shape.provider_signature[0]
        } else {
            2
        };
        self.calls.borrow_mut().push((tag, now));
        if tag == 1 && self.missing {
            return None;
        }
        Some(PlanningCost {
            typical_ns: if tag == 1 { 7 } else { 3 },
            planning_ns: if tag == 1 { 35 } else { 5 },
            model_version: 7,
            valid_for_ns: self.expiry.checked_sub(now)?,
        })
    }
}
fn model() -> Envelope {
    Envelope {
        enabled: true,
        missing: false,
        expiry: 10_000,
        calls: RefCell::new(Vec::new()),
    }
}
fn scenario() -> SchedulerSnapshot {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(3);
    let mut s = snapshot(vec![request]);
    s.capabilities.decode_batch_sizes = vec![nz(1)];
    s.scope.horizon_end_ns = 240;
    s
}
fn routes() -> Routes {
    Routes {
        first_alternative: false,
        count: 2,
    }
}

#[test]
fn host_domain_complete_common_witness_uses_maximum_whole_wave_cost() {
    let model = model();
    let (first, witness, _) =
        feasible(planner(2).propose(&scenario(), &model, &routes(), &mut Clock(100)));
    assert!(first.candidate.execution_shape.exact().is_some());
    assert_eq!(first.predicted_wall_ns, 5);
    assert_eq!(
        witness.completion_at_ns, 140,
        "5 + max(5,35), never a cheap representative or summed percentiles"
    );
    assert_eq!(witness.predicted_output_tokens, 2);
    assert!(model
        .calls
        .borrow()
        .iter()
        .any(|&(tag, now)| tag == 0 && now == 105));
    assert!(model
        .calls
        .borrow()
        .iter()
        .any(|&(tag, now)| tag == 1 && now == 105));
}

#[test]
fn host_domain_missing_branch_and_legacy_models_cannot_produce_a_witness() {
    for (enabled, missing) in [(true, true), (false, false)] {
        let mut model = model();
        model.enabled = enabled;
        model.missing = missing;
        assert!(matches!(
            planner(2).propose(&scenario(), &model, &routes(), &mut Clock(100)),
            PlanningDecision::Unknown {
                reason: PlanningUnknownReason::CostUnavailable,
                ..
            }
        ));
    }
}

#[test]
fn host_domain_minimum_expiry_must_cover_maximum_wave_end_inclusively() {
    for (expiry, pass) in [(140, true), (139, false)] {
        let mut model = model();
        model.expiry = expiry;
        let result = planner(2).propose(&scenario(), &model, &routes(), &mut Clock(100));
        if pass {
            assert_eq!(feasible(result).0.witness_valid_for_ns, 0);
        } else {
            assert!(matches!(
                result,
                PlanningDecision::Unknown {
                    reason: PlanningUnknownReason::CostUnavailable,
                    ..
                }
            ));
        }
    }
}

#[test]
fn host_domain_can_never_supply_a_nonexact_first_submission() {
    let mut r = routes();
    r.first_alternative = true;
    assert!(matches!(
        planner(2).propose(&scenario(), &model(), &r, &mut Clock(100)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidShapeEvidence,
            ..
        }
    ));
}

#[test]
fn host_domain_configured_bound_cannot_silently_truncate_routes() {
    let mut planner = planner(2);
    planner.settings.search.max_shape_alternatives = nz(1);
    assert!(matches!(
        planner.propose(&scenario(), &model(), &routes(), &mut Clock(100)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ShapeCapacity,
            ..
        }
    ));
}

#[test]
fn host_domain_budget_is_checked_between_each_cost_lookup() {
    let s = scenario();
    let candidates =
        candidates::enumerate(&s, &s.requests, 100, 1, None, &TestResolver, &mut || Ok(()))
            .unwrap();
    let mut shape = candidates.waves[0].execution_shape.exact().unwrap().clone();
    shape.host_content_features = Some(HostContentCostFeaturesV1 {
        schema_version: 1,
        output_policy_signature: [4; 32],
    });
    shape.provider_signature = [0; 32];
    let mut second = shape.clone();
    second.provider_signature = [1; 32];
    let domain = PlanningShapeDomain::HostContentAlternatives(vec![shape, second]);
    let model = model();
    let polls = Cell::new(0);
    let result = simulation::domain_cost(&s, &model, &domain, None, 100, &mut || {
        polls.set(polls.get() + 1);
        if polls.get() == 3 {
            Err(PlanningUnknownReason::ComputeBudgetExhausted)
        } else {
            Ok(())
        }
    });
    assert_eq!(result, Err(PlanningUnknownReason::ComputeBudgetExhausted));
    assert_eq!(model.calls.borrow().len(), 1);
}
