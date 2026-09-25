//! CPU contract fixture: real canonical builder, fit/residual model and planner
//! bridge. This does not establish hardware coverage or a serving witness.
use super::*;
use crate::implementations::continuous::cost_model::statistical::model::{
    tests as samples, WholeWaveModelV1, WholeWaveObservationV1,
};
use ferrum_interfaces::execution_cost::{CanonicalWaveCostShape, StatisticalWaveEvidenceV1};

struct SelectedModel(WholeWaveModelV1);
impl PlanningCostModel for SelectedModel {
    fn model_version(&self) -> u64 {
        7
    }
    fn requires_statistical_evidence(&self) -> bool {
        true
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
        panic!("selected model must not fall back to a legacy lookup")
    }
    fn predict_with_evidence(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now: u64,
    ) -> Option<PlanningCost> {
        let p = self
            .0
            .predict_input(fingerprint, evidence?.input_for(shape)?, now)
            .ok()?;
        Some(PlanningCost {
            typical_ns: p.fitted_ns,
            planning_ns: p.planning_ns,
            model_version: 7,
            valid_for_ns: p.valid_until_ns.checked_sub(now)?,
        })
    }
}
fn model() -> SelectedModel {
    SelectedModel(
        samples::fitted()
            .calibrate(&samples::populations().1, 120)
            .unwrap(),
    )
}

#[derive(Clone)]
struct State {
    canonical: PlanningShapeDomain<CanonicalWaveCostShape>,
    statistics: Option<PlanningShapeDomain<StatisticalWaveEvidenceV1>>,
}
impl PlanningExecutionContext for State {
    fn begin<'epoch>(
        &'epoch self,
        _: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        poll()?;
        Ok(Arc::new(self.clone()))
    }
}
impl<'epoch> PlanningExecutionState<'epoch> for State {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        poll()?;
        Ok(Some(ProjectedExecution {
            ordered_work: input.work.to_vec(),
            canonical_domain: self.canonical.clone(),
            statistical_evidence: self.statistics.clone(),
            successor: Arc::new(self.clone()),
        }))
    }
}
fn exact(sample: WholeWaveObservationV1) -> State {
    State {
        canonical: PlanningShapeDomain::Exact(sample.exact),
        statistics: Some(PlanningShapeDomain::Exact(sample.selected)),
    }
}
fn scenario() -> (SchedulerSnapshot, Vec<CandidateWork>) {
    let mut r = prefill(1);
    r.recurrent_state_bytes = 64;
    r.timing.maximum_output_tokens = n32(32);
    let RequestPhaseView::Prefill(ref mut p) = r.phase else {
        unreachable!()
    };
    p.total_prompt_tokens = n32(64);
    p.executable_until = 64;
    p.reference = Arc::new(PrefillReferenceWork {
        evaluation: Default::default(),
        version: 1,
        points: vec![
            ReferenceWorkPoint {
                prompt_tokens: 0,
                cumulative_work_ns: 0,
            },
            ReferenceWorkPoint {
                prompt_tokens: 12,
                cumulative_work_ns: 12,
            },
            ReferenceWorkPoint {
                prompt_tokens: 64,
                cumulative_work_ns: 64,
            },
        ],
    });
    let work = vec![CandidateWork {
        key: r.key.clone(),
        action: WaveAction::Prefill {
            offset: 0,
            count: n32(12),
        },
    }];
    let mut s = snapshot(vec![r]);
    s.observed_at_ns = 120;
    s.capabilities.path = WaveExecutionPath::PlanRuntime;
    s.capabilities.graph_state = WaveGraphState::Disabled;
    s.capabilities.prefill_chunk_sizes = vec![n32(12)];
    (s, work)
}
fn project(s: &SchedulerSnapshot, work: &[CandidateWork], context: &State) -> WaveCandidate {
    let state = context.begin(s, &mut || Ok(())).unwrap();
    execution::project(
        s,
        &s.requests,
        work,
        state.as_ref(),
        true,
        PlanningCostEvidenceRequirement::Selected,
        &mut || Ok(()),
    )
    .unwrap()
    .unwrap()
    .wave
}
fn cost(
    s: &SchedulerSnapshot,
    model: &dyn PlanningCostModel,
    wave: &WaveCandidate,
    now: u64,
) -> Result<PlanningCost, PlanningUnknownReason> {
    simulation::domain_cost(
        s,
        model,
        &wave.execution_shape,
        wave.cost_evidence.as_ref(),
        now,
        &mut || Ok(()),
    )
}

#[test]
fn selected_evidence_flows_through_projection_model_and_anchored_ttl() {
    let (s, work) = scenario();
    let wave = project(&s, &work, &exact(samples::sample(17, 12, 130)));
    let model = model();
    let direct = cost(&s, &model, &wave, 120).unwrap();
    assert_eq!(direct.planning_ns, 125);
    let anchored =
        AnchoredPlanningCostModel::new(&model, PlanningCostClockAnchor::exact(1_000, 120));
    assert!(anchored.requires_statistical_evidence());
    assert_eq!(cost(&s, &anchored, &wave, 1_000).unwrap(), direct);
    let later = cost(&s, &anchored, &wave, 1_010).unwrap();
    assert_eq!(later.planning_ns, direct.planning_ns);
    assert_eq!(later.valid_for_ns + 10, direct.valid_for_ns);
    assert_eq!(
        cost(&s, &anchored, &wave, 30_000),
        Err(PlanningUnknownReason::CostUnavailable)
    );
}

#[test]
fn missing_producer_is_unknown_only_for_the_explicit_selected_model() {
    let (s, work) = scenario();
    let mut context = exact(samples::sample(17, 12, 130));
    let with_evidence = project(&s, &work, &context);
    context.statistics = None;
    let without = project(&s, &work, &context);
    assert_eq!(
        with_evidence, without,
        "passive statistics do not redefine execution identity"
    );
    assert_eq!(
        cost(&s, &model(), &without, 120),
        Err(PlanningUnknownReason::CostUnavailable)
    );
    assert_eq!(
        cost(&s, &Model(|_: &WaveExecutionShape| Some(17)), &without, 120)
            .unwrap()
            .planning_ns,
        17
    );
    let bound = with_evidence
        .cost_evidence
        .as_ref()
        .unwrap()
        .exact()
        .unwrap();
    let mut wrong = with_evidence.execution_shape.exact().unwrap().clone();
    wrong.recurrent_state_bytes += 1;
    assert!(bound.input_for(&wrong).is_none());
}

#[test]
fn all_alternatives_need_matching_evidence_and_independent_cost_support() {
    let (s, work) = scenario();
    let known = samples::sample(17, 12, 130);
    let unseen = samples::sample_route(18, 12, 130, "unobserved.algorithm", false);
    let mut context = State {
        canonical: PlanningShapeDomain::HostContentAlternatives(vec![known.exact, unseen.exact]),
        statistics: Some(PlanningShapeDomain::HostContentAlternatives(vec![
            known.selected,
            unseen.selected,
        ])),
    };
    let wave = project(&s, &work, &context);
    assert!(wave.cost_evidence.is_some());
    assert_eq!(
        cost(&s, &model(), &wave, 120),
        Err(PlanningUnknownReason::CostUnavailable),
        "known first branch cannot stand in for the whole domain"
    );
    let Some(PlanningShapeDomain::HostContentAlternatives(ref mut selected)) = context.statistics
    else {
        unreachable!()
    };
    selected.swap(0, 1);
    assert!(project(&s, &work, &context).cost_evidence.is_none());
    let Some(PlanningShapeDomain::HostContentAlternatives(ref mut selected)) = context.statistics
    else {
        unreachable!()
    };
    selected.pop();
    assert!(project(&s, &work, &context).cost_evidence.is_none());
}

#[test]
fn evidence_binding_propagates_budget_failure_without_legacy_fallback() {
    let sample = samples::sample(17, 12, 130);
    let shape = canonical_cost_shape(&sample.exact).unwrap();
    let result = execution::bind_statistics(
        &PlanningShapeDomain::Exact(sample.exact),
        &PlanningShapeDomain::Exact(shape),
        Some(&PlanningShapeDomain::Exact(sample.selected)),
        PlanningCostEvidenceRequirement::Selected,
        &mut || Err(PlanningUnknownReason::ComputeBudgetExhausted),
    );
    assert!(matches!(
        result,
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
}

#[test]
fn independent_attention_v2_future_binding_consumes_explicit_family_and_rejects_legacy_sidecar() {
    use crate::implementations::continuous::cost_model::statistical::model::FittedWholeWaveModelV1;
    let (fit, residual) = samples::populations();
    let model = SelectedModel(
        FittedWholeWaveModelV1::fit_independent_attention_v2(
            samples::fingerprint(),
            samples::settings(),
            samples::partition(),
            &fit,
            120,
        )
        .unwrap()
        .calibrate(&residual, 120)
        .unwrap(),
    );
    let (s, work) = scenario();
    let sample = samples::sample(17, 12, 130);
    let wave = project(&s, &work, &exact(sample.clone()));
    assert_eq!(cost(&s, &model, &wave, 120).unwrap().planning_ns, 125);
    let mut old = sample.clone();
    old.selected =
        StatisticalWaveEvidenceV1::from_wire_v1(old.selected.to_wire_v1(), &old.exact).unwrap();
    let legacy = project(&s, &work, &exact(old));
    assert_eq!(
        wave, legacy,
        "statistical family does not change canonical execution identity"
    );
    assert_eq!(
        cost(&s, &model, &legacy, 120),
        Err(PlanningUnknownReason::CostUnavailable)
    );
    assert_eq!(
        cost(&s, &model, &wave, 30_000),
        Err(PlanningUnknownReason::CostUnavailable)
    );
    let mut wrong = sample.exact.clone();
    wrong.recurrent_state_bytes += 1;
    let ctx = State {
        canonical: PlanningShapeDomain::Exact(wrong),
        statistics: Some(PlanningShapeDomain::Exact(sample.selected)),
    };
    assert!(execution::bind_statistics(
        &ctx.canonical,
        &PlanningShapeDomain::Exact(canonical_cost_shape(ctx.canonical.exact().unwrap()).unwrap()),
        ctx.statistics.as_ref(),
        PlanningCostEvidenceRequirement::Selected,
        &mut || Ok(())
    )
    .unwrap()
    .is_none());
}
