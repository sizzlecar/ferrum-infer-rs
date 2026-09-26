//! The deterministic model below checks planner protocol only. It does not
//! claim calibrated execution times or mint qualified host observations.
use super::*;
use ferrum_interfaces::execution_cost::{
    HostContentForecastV2, HostPendingConstraintV2, HostPendingSetV2,
};

#[derive(Clone)]
struct ForecastRoute {
    supplies_forecast: bool,
}
impl PlanningExecutionContext for ForecastRoute {
    fn begin<'epoch>(
        &'epoch self,
        _: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        poll()?;
        Ok(Arc::new(self.clone()))
    }
}
impl<'epoch> PlanningExecutionState<'epoch> for ForecastRoute {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        let mut projected = Route.project(input, poll)?.unwrap();
        projected.host_content_forecasts = self
            .supplies_forecast
            .then_some(PlanningShapeDomain::Exact(HostContentForecastV2::Exact));
        projected.successor = Arc::new(self.clone());
        Ok(Some(projected))
    }
}

#[test]
fn bounded_execution_session_retains_v2_forecast_through_cost_lookup() {
    let mut requests = vec![decode(1), decode(2)];
    for (position, request) in requests.iter_mut().enumerate() {
        request.context_tokens = 64;
        request.recurrent_state_bytes = 32;
        request.timing.committed_tokens = 2;
        request.timing.maximum_output_tokens = n32(if position == 0 { 3 } else { 20 });
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
    for supplies_forecast in [false, true] {
        let route = ForecastRoute { supplies_forecast };
        // Use the same bounded adapter as product search. A direct producer
        // projection would miss information lost by this intermediate layer.
        let settings = BoundedPlannerSettings::default();
        let session = execution::ExecutionSession::new(&route, &settings);
        let state = simulation::begin_with_controller_time(
            &s,
            &session,
            &mut || Ok(()),
            100,
            settings
                .future_controller_time
                .reserved_ns(&settings.search)
                .unwrap(),
        )
        .unwrap();
        let result = simulation::advance(
            &s,
            &state,
            &work,
            &ModelV2,
            false,
            &mut || Ok(()),
            false,
            None,
        );
        if supplies_forecast {
            let result = result.unwrap_or_else(|e| {
                panic!(
                    "bounded V2 projection lost provider-bound input: {:?}",
                    e.cause
                )
            });
            let replay = simulation::replay(
                &s,
                &[result.wave.clone()],
                &ModelV2,
                &session,
                false,
                &mut || Ok(()),
                100,
                0,
                false,
                None,
            )
            .unwrap();
            let proof = Arc::new(FinalReplayFirstWave::from_replay(
                &s,
                replay.first_wave_candidate.clone().unwrap(),
                replay.first_wave_canonical.clone().unwrap(),
                replay.first_wave_statistics.clone(),
            ));
            let mut delivered = SelectedWave {
                final_replay_first_wave: Some(proof),
                protection: None,
                candidate: result.wave.clone(),
                predicted_wall_ns: 20,
                planning_observed_at_ns: 100,
                snapshot_observed_at_ns: s.observed_at_ns,
                snapshot_generation: s.generation,
                cost_model_version: s.cost_model_version,
                witness_valid_for_ns: 10,
            };
            let (_, stats, query) = delivered.replayed_first_wave_structured_v2(&s).unwrap();
            let statistics_ptr = Arc::as_ptr(stats);
            let query_ptr = query as *const _;
            // Public candidate sidecars can be removed/replaced without
            // changing its equality; they are never the delivered authority.
            delivered.candidate.cost_evidence = None;
            let (_, stats, query) = delivered.replayed_first_wave_structured_v2(&s).unwrap();
            assert_eq!(Arc::as_ptr(stats), statistics_ptr);
            assert_eq!(query as *const _, query_ptr);
            delivered.candidate.work[0].key.incarnation += 1;
            assert!(delivered.replayed_first_wave_structured_v2(&s).is_none());
            assert_eq!(result.state.now_ns, 120);
            assert_eq!(result.state.output_tokens, 2);
            assert!(result.state.requests[0].timing.completed());
            assert!(!result.state.requests[1].timing.completed());
            let evidence = result.wave.cost_evidence.as_ref().unwrap().exact().unwrap();
            assert!(evidence
                .structured_query_v2_for(result.wave.execution_shape.exact().unwrap())
                .is_ok());
        } else {
            assert!(matches!(
                result,
                Err(simulation::TransitionFailure {
                    cause: simulation::SimulationFailure::Unknown(
                        PlanningUnknownReason::CostUnavailable
                    ),
                    ..
                })
            ));
        }
    }
}

#[test]
fn structured_v2_binding_requires_explicit_forecast_and_retains_original_recipe() {
    let wave = fixture::wave(0, true, 8, "fixture.first");
    let shape = canonical_cost_shape(&wave.exact).unwrap();
    let selected = wave.statistical.as_ref().unwrap();
    let requirement = PlanningCostEvidenceRequirement::StructuredV2;
    let absent = PlanningCostEvidence::bind(&wave.exact, &shape, selected, requirement).unwrap();
    assert!(absent.structured_query_v2_for(&shape).is_err());
    let exact = PlanningCostEvidence::bind_with_forecast(
        &wave.exact,
        &shape,
        selected,
        requirement,
        Some(&HostContentForecastV2::Exact),
    )
    .unwrap();
    let query = exact.structured_query_v2_for(&shape).unwrap();
    assert!(std::ptr::eq(
        query,
        exact.structured_query_v2_for(&shape).unwrap()
    ));
    assert!(exact.structured_input_for(&shape).is_err());
    assert!(exact.input_for(&shape).is_none());
    let recipe = selected.structured_capture().unwrap().unwrap();
    let pending = |recipe| {
        HostContentForecastV2::Unresolved(
            HostPendingSetV2::new(&wave.exact, recipe, &[], HostPendingConstraintV2::AnySubset)
                .unwrap(),
        )
    };
    let bound = PlanningCostEvidence::bind_with_forecast(
        &wave.exact,
        &shape,
        selected,
        requirement,
        Some(&pending(recipe)),
    )
    .unwrap();
    assert!(bound.structured_query_v2_for(&shape).is_ok());
    let replacement = Arc::new(recipe.as_ref().clone());
    let mismatched = PlanningCostEvidence::bind_with_forecast(
        &wave.exact,
        &shape,
        selected,
        requirement,
        Some(&pending(&replacement)),
    )
    .unwrap();
    assert!(mismatched.structured_query_v2_for(&shape).is_err());
    let mut other_shape = shape.clone();
    other_shape.recurrent_state_bytes += 1;
    assert!(bound.structured_query_v2_for(&other_shape).is_err());
}

struct ModelV2;
impl PlanningCostModel for ModelV2 {
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        PlanningCostEvidenceRequirement::StructuredV2
    }
    fn model_version(&self) -> u64 {
        7
    }
    fn supports_empirical_host_content(&self) -> bool {
        true
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        _: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        panic!("V2 must not invoke a legacy cost callback")
    }
    fn predict_with_evidence(
        &self,
        _: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        evidence?.structured_query_v2_for(shape).ok()?;
        Some(PlanningCost {
            typical_ns: 10,
            planning_ns: 20,
            model_version: 7,
            valid_for_ns: 1000_u64.checked_sub(now_ns)?,
        })
    }
}

#[test]
fn structured_v2_domain_preserves_unknown_alternative_and_rejects_missing_forecast() {
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
    let good = PlanningShapeDomain::HostContentAlternatives(vec![forecast(&a), forecast(&b)]);
    let wrong = PlanningShapeDomain::HostContentAlternatives(vec![forecast(&a), forecast(&a)]);
    let shapes = PlanningShapeDomain::HostContentAlternatives(vec![
        canonical_cost_shape(&a.exact).unwrap(),
        canonical_cost_shape(&b.exact).unwrap(),
    ]);
    let canonical = PlanningShapeDomain::HostContentAlternatives(vec![a.exact, b.exact]);
    let selected = PlanningShapeDomain::HostContentAlternatives(vec![
        a.statistical.unwrap(),
        b.statistical.unwrap(),
    ]);
    let bind = |forecasts| {
        execution::bind_statistics_with_forecasts(
            &canonical,
            &shapes,
            Some(&selected),
            forecasts,
            PlanningCostEvidenceRequirement::StructuredV2,
            &mut || Ok(()),
        )
    };
    assert!(bind(None).unwrap().is_none());
    let incomplete = PlanningShapeDomain::HostContentAlternatives(vec![good.shapes()[0].clone()]);
    assert!(bind(Some(&incomplete)).unwrap().is_none());
    let evidence = bind(Some(&good)).unwrap().unwrap();
    let s = snapshot(Vec::new());
    assert!(
        simulation::domain_cost(&s, &ModelV2, &shapes, Some(&evidence), 100, &mut || Ok(()))
            .is_ok()
    );
    let bad = bind(Some(&wrong)).unwrap().unwrap();
    assert_eq!(bad.shapes().len(), 2);
    assert!(bad.shapes()[0]
        .structured_query_v2_for(&shapes.shapes()[0])
        .is_ok());
    assert!(bad.shapes()[1]
        .structured_query_v2_for(&shapes.shapes()[1])
        .is_err());
    assert_eq!(
        simulation::domain_cost(&s, &ModelV2, &shapes, Some(&bad), 100, &mut || Ok(())),
        Err(PlanningUnknownReason::CostUnavailable)
    );
}

#[test]
fn structured_v2_first_exact_wave_cannot_borrow_a_conditional_forecast() {
    let wave = fixture::wave(0, true, 8, "fixture.first");
    let recipe = wave
        .statistical
        .as_ref()
        .unwrap()
        .structured_capture()
        .unwrap()
        .unwrap();
    let pending = HostContentForecastV2::Unresolved(
        HostPendingSetV2::new(&wave.exact, recipe, &[], HostPendingConstraintV2::AnySubset)
            .unwrap(),
    );
    let shape = PlanningShapeDomain::Exact(canonical_cost_shape(&wave.exact).unwrap());
    let canonical = PlanningShapeDomain::Exact(wave.exact);
    let selected = PlanningShapeDomain::Exact(wave.statistical.unwrap());
    let conditional = PlanningShapeDomain::Exact(pending);
    assert!(execution::bind_statistics_with_forecasts(
        &canonical,
        &shape,
        Some(&selected),
        Some(&conditional),
        PlanningCostEvidenceRequirement::StructuredV2,
        &mut || Ok(()),
    )
    .unwrap()
    .is_none());
    let exact = PlanningShapeDomain::Exact(HostContentForecastV2::Exact);
    assert!(execution::bind_statistics_with_forecasts(
        &canonical,
        &shape,
        Some(&selected),
        Some(&exact),
        PlanningCostEvidenceRequirement::StructuredV2,
        &mut || Ok(()),
    )
    .unwrap()
    .is_some());
}
