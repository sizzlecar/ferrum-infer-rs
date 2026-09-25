//! Planner protocol tests use real typed producer projections. The model below
//! is deliberately a deterministic boundary fixture, not measured GPU cost.
use super::*;
use crate::implementations::continuous::cost_model::structured::StructuredUnknown;
mod fixture;
mod v2;

#[derive(Clone)]
struct Route;
impl PlanningExecutionContext for Route {
    fn begin<'epoch>(
        &'epoch self,
        _: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        poll()?;
        Ok(Arc::new(self.clone()))
    }
}
impl<'epoch> PlanningExecutionState<'epoch> for Route {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        poll()?;
        let wave = fixture::wave(0, true, 8, "fixture.first");
        Ok(Some(ProjectedExecution {
            host_content_forecasts: None,
            ordered_work: input.work.to_vec(),
            canonical_domain: PlanningShapeDomain::Exact(wave.exact),
            statistical_evidence: Some(PlanningShapeDomain::Exact(wave.statistical.unwrap())),
            successor: Arc::new(self.clone()),
        }))
    }
}

#[test]
fn structured_model_requirement_reaches_real_simulation_projection() {
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
    let route = Route;
    let state = simulation::begin(&s, &route, &mut || Ok(()), 100).unwrap();
    let result = simulation::advance(
        &s,
        &state,
        &work,
        &StructuredModel,
        true,
        &mut || Ok(()),
        false,
        None,
    )
    .unwrap_or_else(|_| panic!("typed structured input must reach the cost lookup"));
    assert!(result
        .wave
        .cost_evidence
        .as_ref()
        .unwrap()
        .exact()
        .unwrap()
        .structured_input_for(result.wave.execution_shape.exact().unwrap())
        .is_ok());
    assert_eq!(result.state.now_ns, 120);
    assert_eq!(result.state.output_tokens, 2);
    assert!(result.state.requests[0].timing.completed());
    assert!(!result.state.requests[1].timing.completed());
}

struct StructuredModel;
impl PlanningCostModel for StructuredModel {
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        PlanningCostEvidenceRequirement::Structured
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
        panic!("structured lookup must never call legacy lookup")
    }
    fn predict_with_evidence(
        &self,
        _: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now: u64,
    ) -> Option<PlanningCost> {
        let input = evidence?.structured_input_for(shape).ok()?;
        let first_terminal = input.physical_host_rows()[0].terminal_expectation
            == ferrum_interfaces::execution_cost::HostTerminalExpectationV1::LengthBoundary;
        Some(PlanningCost {
            typical_ns: 10,
            planning_ns: if first_terminal { 20 } else { 30 },
            model_version: 7,
            valid_for_ns: (if first_terminal { 1_000_u64 } else { 900_u64 }).checked_sub(now)?,
        })
    }
}

#[test]
fn structured_binding_caches_the_attached_route_and_does_not_relabel_selected_input() {
    let wave = fixture::wave(0, true, 8, "fixture.first");
    let shape = canonical_cost_shape(&wave.exact).unwrap();
    let selected = wave.statistical.as_ref().unwrap();
    let evidence = PlanningCostEvidence::bind(
        &wave.exact,
        &shape,
        selected,
        PlanningCostEvidenceRequirement::Structured,
    )
    .unwrap();
    let input = evidence.structured_input_for(&shape).unwrap();
    assert_eq!(input.physical_host_rows().len(), 2);
    assert!(evidence.input_for(&shape).is_none());
    assert!(std::ptr::eq(
        input,
        evidence.structured_input_for(&shape).unwrap()
    ));
    let mut wrong = shape.clone();
    wrong.recurrent_state_bytes += 1;
    assert!(matches!(
        evidence.structured_input_for(&wrong),
        Err(StructuredUnknown::MissingEvidence)
    ));
    assert!(PlanningCostEvidence::bind(
        &wave.exact,
        &wrong,
        selected,
        PlanningCostEvidenceRequirement::Structured
    )
    .is_none());
    let legacy = PlanningCostEvidence::bind(
        &wave.exact,
        &shape,
        selected,
        PlanningCostEvidenceRequirement::Selected,
    )
    .unwrap();
    assert!(legacy.input_for(&shape).is_some());
    assert!(matches!(
        legacy.structured_input_for(&shape),
        Err(StructuredUnknown::MissingEvidence)
    ));
}

fn alternatives(
    missing_second: bool,
) -> (
    PlanningShapeDomain<WaveExecutionShape>,
    PlanningShapeDomain<PlanningCostEvidence>,
) {
    let a = fixture::wave(0, true, 8, "fixture.first");
    let b = fixture::wave(1, !missing_second, 8, "fixture.first");
    let shapes = PlanningShapeDomain::HostContentAlternatives(vec![
        canonical_cost_shape(&a.exact).unwrap(),
        canonical_cost_shape(&b.exact).unwrap(),
    ]);
    let evidence = execution::bind_statistics(
        &PlanningShapeDomain::HostContentAlternatives(vec![a.exact, b.exact]),
        &shapes,
        Some(&PlanningShapeDomain::HostContentAlternatives(vec![
            a.statistical.unwrap(),
            b.statistical.unwrap(),
        ])),
        PlanningCostEvidenceRequirement::Structured,
        &mut || Ok(()),
    )
    .unwrap()
    .unwrap();
    (shapes, evidence)
}

#[test]
fn structured_alternatives_take_largest_cost_shortest_ttl_and_keep_unknown_branches() {
    let s = snapshot(Vec::new());
    let (shapes, evidence) = alternatives(false);
    let cost = simulation::domain_cost(
        &s,
        &StructuredModel,
        &shapes,
        Some(&evidence),
        100,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(cost.planning_ns, 30);
    assert_eq!(cost.valid_for_ns, 770);
    let (shapes, evidence) = alternatives(true);
    assert_eq!(evidence.shapes().len(), 2);
    assert!(evidence.shapes()[0]
        .structured_input_for(&shapes.shapes()[0])
        .is_ok());
    assert!(matches!(
        evidence.shapes()[1].structured_input_for(&shapes.shapes()[1]),
        Err(StructuredUnknown::MissingEvidence)
    ));
    assert_eq!(
        simulation::domain_cost(
            &s,
            &StructuredModel,
            &shapes,
            Some(&evidence),
            100,
            &mut || Ok(())
        ),
        Err(PlanningUnknownReason::CostUnavailable)
    );
}

#[test]
fn structured_anchor_forwards_protocol_and_ages_every_branch_without_refitting() {
    let s = snapshot(Vec::new());
    let (shapes, evidence) = alternatives(false);
    let anchored = AnchoredPlanningCostModel::new(
        &StructuredModel,
        PlanningCostClockAnchor::exact(10_000, 100),
    );
    assert_eq!(
        anchored.evidence_requirement(),
        PlanningCostEvidenceRequirement::Structured
    );
    let at =
        |now| simulation::domain_cost(&s, &anchored, &shapes, Some(&evidence), now, &mut || Ok(()));
    assert_eq!(at(10_000).unwrap().valid_for_ns, 770);
    assert_eq!(at(10_050).unwrap().valid_for_ns, 720);
    assert_eq!(at(10_770).unwrap().valid_for_ns, 0);
    assert_eq!(at(10_771), Err(PlanningUnknownReason::CostUnavailable));
}

#[test]
fn structured_binding_propagates_budget_and_rejects_incomplete_domains() {
    let wave = fixture::wave(0, true, 8, "fixture.first");
    let canonical = PlanningShapeDomain::Exact(wave.exact);
    let shapes =
        PlanningShapeDomain::Exact(canonical_cost_shape(canonical.exact().unwrap()).unwrap());
    let selected = PlanningShapeDomain::Exact(wave.statistical.unwrap());
    assert!(matches!(
        execution::bind_statistics(
            &canonical,
            &shapes,
            Some(&selected),
            PlanningCostEvidenceRequirement::Structured,
            &mut || Err(PlanningUnknownReason::ComputeBudgetExhausted)
        ),
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
    let incomplete = PlanningShapeDomain::HostContentAlternatives(Vec::new());
    assert!(execution::bind_statistics(
        &canonical,
        &shapes,
        Some(&incomplete),
        PlanningCostEvidenceRequirement::Structured,
        &mut || Ok(())
    )
    .unwrap()
    .is_none());
}
