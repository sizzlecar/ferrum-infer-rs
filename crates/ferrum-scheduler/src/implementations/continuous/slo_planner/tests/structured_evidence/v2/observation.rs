//! Real typed binding and original lookup loops; no qualified model fabrication.
use super::*;
use crate::implementations::continuous::slo_planner::observation::AttemptObservation;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Default)]
struct Tap {
    phases: Mutex<Vec<PlanningQueryPhase>>,
    constructed: AtomicUsize,
    lookups: Mutex<Vec<(PlanningQueryKey, u64, PlanningObservedCost)>>,
    ends: Mutex<Vec<(usize, usize, PlanningQueryAttemptEnd)>>,
}
impl PlanningQueryObserver for Tap {
    fn begin_replay(&self, _: usize) -> u64 {
        1
    }
    fn end_replay(&self, _: u64, reason: PlanningQueryAttemptEnd) {
        assert_eq!(reason, PlanningQueryAttemptEnd::Completed);
    }
    fn selected_replay(&self, _: u64) {}
    fn begin_attempt(
        &self,
        phase: PlanningQueryPhase,
        _: usize,
        _: &[RequestSchedulingView],
        _: &[CandidateWork],
    ) -> u64 {
        let mut phases = self.phases.lock().unwrap();
        phases.push(phase);
        phases.len() as u64
    }
    fn constructed(
        &self,
        _: PlanningQueryKey,
        query: Result<
            &crate::implementations::continuous::cost_model::structured_v2::StructuredQueryV2,
            StructuredUnknown,
        >,
    ) {
        let query = query.unwrap();
        assert!(query.observation_retained_bytes().unwrap() >= std::mem::size_of_val(query));
        assert!(!query
            .required_coverage()
            .unwrap()
            .joint_support_coordinates
            .is_empty());
        self.constructed.fetch_add(1, Ordering::SeqCst);
    }
    fn lookup(&self, key: PlanningQueryKey, now: u64, cost: PlanningObservedCost) {
        self.lookups.lock().unwrap().push((key, now, cost));
    }
    fn end_attempt(&self, _: u64, built: usize, queried: usize, reason: PlanningQueryAttemptEnd) {
        self.ends.lock().unwrap().push((built, queried, reason));
    }
}
struct ObservedModel<'a> {
    tap: Option<&'a Tap>,
    calls: AtomicUsize,
    unknown: bool,
}
impl PlanningCostModel for ObservedModel<'_> {
    fn query_observer(&self) -> Option<&dyn PlanningQueryObserver> {
        self.tap.map(|x| x as _)
    }
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        PlanningCostEvidenceRequirement::StructuredV2
    }
    fn supports_empirical_host_content(&self) -> bool {
        true
    }
    fn model_version(&self) -> u64 {
        7
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        _: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        panic!("legacy query")
    }
    fn predict_with_evidence(
        &self,
        fp: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now: u64,
    ) -> Option<PlanningCost> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.unknown {
            None
        } else {
            ModelV2.predict_with_evidence(fp, shape, evidence, now)
        }
    }
    fn predict_observed(
        &self,
        fp: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now: u64,
    ) -> PlanningObservedCost {
        let cost = self.predict_with_evidence(fp, shape, evidence, now);
        PlanningObservedCost {
            outcome: cost.map_or(
                PlanningQueryOutcome::StructuredUnknown(StructuredUnknown::JointSupport),
                PlanningQueryOutcome::Known,
            ),
            cost_now_ns: Some(now),
        }
    }
}

#[test]
fn required_query_observation_first_unknown_preserves_not_queried_and_partial_binding() {
    let a = fixture::wave(0, true, 8, "fixture.first");
    let b = fixture::wave(1, true, 16, "fixture.first");
    let forecast = |wave: &ferrum_interfaces::execution_cost::CanonicalStructuredWave| {
        let recipe = wave
            .statistical
            .as_ref()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap();
        HostContentForecastV2::Unresolved(
            HostPendingSetV2::new(&wave.exact, recipe, &[], HostPendingConstraintV2::AnySubset)
                .unwrap(),
        )
    };
    let forecasts = PlanningShapeDomain::HostContentAlternatives(vec![forecast(&a), forecast(&b)]);
    let shapes = PlanningShapeDomain::HostContentAlternatives(vec![
        canonical_cost_shape(&a.exact).unwrap(),
        canonical_cost_shape(&b.exact).unwrap(),
    ]);
    let canonical = PlanningShapeDomain::HostContentAlternatives(vec![a.exact, b.exact]);
    let statistics = PlanningShapeDomain::HostContentAlternatives(vec![
        a.statistical.unwrap(),
        b.statistical.unwrap(),
    ]);
    let s = snapshot(Vec::new());
    let tap = Tap::default();
    let attempt = AttemptObservation::new(&tap, PlanningQueryPhase::Search, 1, &[], &[]);
    let evidence = execution::bind_statistics_observed(
        &canonical,
        &shapes,
        Some(&statistics),
        Some(&forecasts),
        None,
        PlanningCostEvidenceRequirement::StructuredV2,
        &mut || Ok(()),
        Some(&attempt),
    )
    .unwrap()
    .unwrap();
    let model = ObservedModel {
        tap: Some(&tap),
        calls: AtomicUsize::new(0),
        unknown: true,
    };
    let result = simulation::domain_cost_observed(
        &s,
        &model,
        &shapes,
        Some(&evidence),
        100,
        &mut || Ok(()),
        Some(&attempt),
    );
    assert_eq!(result, Err(PlanningUnknownReason::CostUnavailable));
    attempt.finish(PlanningQueryAttemptEnd::Unknown(result.unwrap_err()));
    assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *tap.ends.lock().unwrap(),
        vec![(
            2,
            1,
            PlanningQueryAttemptEnd::Unknown(PlanningUnknownReason::CostUnavailable)
        )]
    );
    assert_eq!(
        tap.lookups.lock().unwrap()[0].2.outcome,
        PlanningQueryOutcome::StructuredUnknown(StructuredUnknown::JointSupport)
    );

    let partial = Tap::default();
    let attempt = AttemptObservation::new(&partial, PlanningQueryPhase::Search, 1, &[], &[]);
    let result = execution::bind_statistics_observed(
        &canonical,
        &shapes,
        Some(&statistics),
        Some(&forecasts),
        None,
        PlanningCostEvidenceRequirement::StructuredV2,
        &mut || {
            if partial.constructed.load(Ordering::SeqCst) == 0 {
                Ok(())
            } else {
                Err(PlanningUnknownReason::ComputeBudgetExhausted)
            }
        },
        Some(&attempt),
    );
    assert!(matches!(
        result,
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
    attempt.finish(PlanningQueryAttemptEnd::Unknown(
        PlanningUnknownReason::ComputeBudgetExhausted,
    ));
    assert_eq!(
        *partial.ends.lock().unwrap(),
        vec![(
            1,
            0,
            PlanningQueryAttemptEnd::Unknown(PlanningUnknownReason::ComputeBudgetExhausted)
        )]
    );
    assert!(partial.lookups.lock().unwrap().is_empty());
}

#[test]
fn required_query_observation_disabled_keeps_lookup_count_and_replay_is_separate() {
    let mut requests = vec![decode(1), decode(2)];
    for (i, r) in requests.iter_mut().enumerate() {
        r.context_tokens = 64;
        r.recurrent_state_bytes = 32;
        r.timing.committed_tokens = 2;
        r.timing.maximum_output_tokens = n32(if i == 0 { 3 } else { 20 });
    }
    let mut s = snapshot(requests);
    s.capabilities.path = WaveExecutionPath::PlanRuntime;
    s.capabilities.graph_state = WaveGraphState::Disabled;
    let work: Vec<_> = s
        .requests
        .iter()
        .map(|r| CandidateWork {
            key: r.key.clone(),
            action: WaveAction::Decode,
        })
        .collect();
    let route = ForecastRoute {
        supplies_forecast: true,
    };
    let tap = Tap::default();
    let mut resulting_waves = Vec::new();
    for enabled in [false, true] {
        let model = ObservedModel {
            tap: enabled.then_some(&tap),
            calls: AtomicUsize::new(0),
            unknown: false,
        };
        let state =
            simulation::begin_with_controller_time(&s, &route, &mut || Ok(()), 100, 0).unwrap();
        let next = simulation::advance(
            &s,
            &state,
            &work,
            &model,
            false,
            &mut || Ok(()),
            false,
            None,
        )
        .unwrap_or_else(|e| panic!("{:?}", e.cause));
        let replay = simulation::replay(
            &s,
            &[next.wave.clone()],
            &model,
            &route,
            false,
            &mut || Ok(()),
            100,
            0,
            false,
            None,
        )
        .unwrap();
        assert_eq!(model.calls.load(Ordering::SeqCst), 2);
        assert_eq!(replay.output_tokens, 2);
        resulting_waves.push(next.wave);
    }
    assert_eq!(resulting_waves[0], resulting_waves[1]);
    assert_eq!(
        *tap.phases.lock().unwrap(),
        vec![
            PlanningQueryPhase::Search,
            PlanningQueryPhase::IndependentReplay { replay: 1 }
        ]
    );
    assert_eq!(tap.constructed.load(Ordering::SeqCst), 2);
    assert!(tap
        .ends
        .lock()
        .unwrap()
        .iter()
        .all(|x| *x == (1, 1, PlanningQueryAttemptEnd::Completed)));
}
