//! Real canonical/selected recipe and planner binding; no trained model or
//! execution permission is synthesized by these domain-propagation fixtures.
use super::*;
use ferrum_interfaces::execution_cost::CanonicalWaveCostShape;
use ferrum_interfaces::execution_cost::{
    CostWorkloadDomainV1, CostWorkloadLimitsV1, ExecutorCostIdentity, EXECUTOR_COST_IDENTITY_SCHEMA,
};
use std::num::{NonZeroU32, NonZeroU64};

fn captured() -> SchedulerSnapshot {
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
    s
}
fn descriptor(snapshot: &SchedulerSnapshot, context: u32) -> CostWorkloadDomainV1 {
    CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: snapshot.fingerprint.model_weights,
            numerical_policy: snapshot.fingerprint.numerical_policy,
            device_runtime: snapshot.fingerprint.device_runtime,
            execution_config: snapshot.fingerprint.execution_config,
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(context).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
            repetition_slot_capacity: 8,
            fixed_state_bytes_per_row: 32,
        },
    )
    .unwrap()
}
#[derive(Clone)]
struct RuntimeRoute {
    domain: Option<Arc<CostWorkloadDomainV1>>,
}
impl PlanningExecutionContext for RuntimeRoute {
    fn begin<'epoch>(
        &'epoch self,
        _: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        poll()?;
        Ok(Arc::new(self.clone()))
    }
}
impl<'epoch> PlanningExecutionState<'epoch> for RuntimeRoute {
    fn cost_workload_domain(&self) -> Option<&CostWorkloadDomainV1> {
        self.domain.as_deref()
    }
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        let mut projected = ForecastRoute {
            supplies_forecast: true,
        }
        .project(input, poll)?
        .unwrap();
        projected.successor = Arc::new(self.clone());
        Ok(Some(projected))
    }
}

#[test]
fn future_workload_domain_survives_bounded_state_and_preserves_exact_route() {
    let s = captured();
    let domain = Arc::new(descriptor(&s, 128));
    let work: Vec<_> = s
        .requests
        .iter()
        .map(|r| CandidateWork {
            key: r.key.clone(),
            action: WaveAction::Decode,
        })
        .collect();
    let mut original = None;
    for supplied in [None, Some(Arc::clone(&domain))] {
        let route = RuntimeRoute {
            domain: supplied.clone(),
        };
        let settings = BoundedPlannerSettings::default();
        let session = execution::ExecutionSession::new(&route, &settings);
        let state = session.begin(&s, &mut || Ok(())).unwrap();
        assert_eq!(state.cost_workload_domain(), supplied.as_deref());
        let result = execution::project(
            &s,
            &s.requests,
            &work,
            state.as_ref(),
            true,
            PlanningCostEvidenceRequirement::StructuredV2,
            &mut || Ok(()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.successor.cost_workload_domain(), supplied.as_deref());
        let evidence = result.wave.cost_evidence.as_ref().unwrap().exact().unwrap();
        let query = evidence
            .structured_query_v2_for(result.wave.execution_shape.exact().unwrap())
            .unwrap();
        assert_eq!(
            query.input().physical_domain_signature(),
            supplied.as_ref().map(|d| d.sha256())
        );
        let physical = result.first_canonical.unwrap();
        if let Some(expected) = &original {
            assert_eq!(
                &physical, expected,
                "numeric domain never replaces the original exact route"
            );
        } else {
            original = Some(physical);
        }
        assert!(result.first_statistics.is_some());
    }
}

#[test]
fn future_workload_domain_rejects_foreign_runtime_and_out_of_range_input() {
    let s = captured();
    let work: Vec<_> = s
        .requests
        .iter()
        .map(|r| CandidateWork {
            key: r.key.clone(),
            action: WaveAction::Decode,
        })
        .collect();
    let mut foreign = s.clone();
    foreign.fingerprint.device_runtime[0] ^= 1;
    let route = RuntimeRoute {
        domain: Some(Arc::new(descriptor(&foreign, 128))),
    };
    let state = route.begin(&s, &mut || Ok(())).unwrap();
    assert!(matches!(
        execution::project(
            &s,
            &s.requests,
            &work,
            state.as_ref(),
            true,
            PlanningCostEvidenceRequirement::StructuredV2,
            &mut || Ok(())
        ),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    ));
    let route = RuntimeRoute {
        domain: Some(Arc::new(descriptor(&s, 64))),
    };
    let state = route.begin(&s, &mut || Ok(())).unwrap();
    let result = execution::project(
        &s,
        &s.requests,
        &work,
        state.as_ref(),
        true,
        PlanningCostEvidenceRequirement::StructuredV2,
        &mut || Ok(()),
    )
    .unwrap()
    .unwrap();
    let bound = result.wave.cost_evidence.as_ref().unwrap().exact().unwrap();
    assert_eq!(bound.structured_query_v2_for(result.wave.execution_shape.exact().unwrap()).unwrap_err(),
        crate::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2::WrongDomain);
    assert!(result.first_statistics.is_none());
    assert!(
        result.first_canonical.is_some(),
        "finite numeric rejection cannot invent a different route"
    );
}

#[test]
fn future_workload_domain_does_not_replace_recipe_or_legacy_none_contract() {
    let s = captured();
    let d = descriptor(&s, 128);
    let wave = fixture::wave(0, true, 8, "fixture.first");
    let shape = canonical_cost_shape(&wave.exact).unwrap();
    let selected = wave.statistical.as_ref().unwrap();
    let bind = |domain| {
        PlanningCostEvidence::bind_with_forecast_and_domain(
            &wave.exact,
            &shape,
            selected,
            PlanningCostEvidenceRequirement::StructuredV2,
            Some(&HostContentForecastV2::Exact),
            domain,
        )
        .unwrap()
    };
    let old = PlanningCostEvidence::bind_with_forecast(
        &wave.exact,
        &shape,
        selected,
        PlanningCostEvidenceRequirement::StructuredV2,
        Some(&HostContentForecastV2::Exact),
    )
    .unwrap();
    assert_eq!(
        old.structured_query_v2_for(&shape).unwrap().input(),
        bind(None).structured_query_v2_for(&shape).unwrap().input()
    );
    assert!(bind(None)
        .structured_query_v2_for(&shape)
        .unwrap()
        .input()
        .physical_domain_signature()
        .is_none());
    assert_eq!(
        bind(Some(&d))
            .structured_query_v2_for(&shape)
            .unwrap()
            .input()
            .physical_domain_signature(),
        Some(d.sha256())
    );
    let other = fixture::wave(1, true, 8, "fixture.first");
    let wrong = PlanningCostEvidence::bind_with_forecast_and_domain(
        &wave.exact,
        &shape,
        other.statistical.as_ref().unwrap(),
        PlanningCostEvidenceRequirement::StructuredV2,
        Some(&HostContentForecastV2::Exact),
        Some(&d),
    )
    .unwrap();
    assert!(wrong.structured_query_v2_for(&shape).is_err());
    let mut wrong_shape = shape.clone();
    wrong_shape.recurrent_state_bytes += 1;
    assert!(PlanningCostEvidence::bind_with_forecast_and_domain(
        &wave.exact,
        &wrong_shape,
        selected,
        PlanningCostEvidenceRequirement::StructuredV2,
        Some(&HostContentForecastV2::Exact),
        Some(&d)
    )
    .is_none());
}

struct EmptyResolver;
impl PlanningShapeResolver for EmptyResolver {
    fn resolve(
        &self,
        _: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        poll()?;
        Ok(None)
    }
}
struct RuntimeResolver(CostWorkloadDomainV1);
impl PlanningShapeResolver for RuntimeResolver {
    fn cost_workload_domain(&self) -> Option<&CostWorkloadDomainV1> {
        Some(&self.0)
    }
    fn resolve(
        &self,
        q: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        EmptyResolver.resolve(q, poll)
    }
}
#[test]
fn future_workload_domain_replay_adapter_forwards_runtime_and_defaults_to_none() {
    let s = captured();
    let resolver = RuntimeResolver(descriptor(&s, 128));
    for resolver in [&EmptyResolver as &dyn PlanningShapeResolver, &resolver] {
        let replay = execution::ReplayContext {
            resolver,
            resources: None,
        };
        let settings = BoundedPlannerSettings::default();
        let bounded = execution::ExecutionSession::new(&replay, &settings);
        let state = bounded.begin(&s, &mut || Ok(())).unwrap();
        assert_eq!(
            state.cost_workload_domain(),
            resolver.cost_workload_domain()
        );
        if let Some(domain) = state.cost_workload_domain() {
            assert!(std::ptr::eq(
                domain,
                resolver.cost_workload_domain().unwrap()
            ));
        }
    }
}
