//! Automatic owner-block publication reaches the public complete planner. The CPU
//! backend projection below describes the same controlled one-row kernel as
//! the actual recorder fixture; it neither executes nor supplies a cost model.
//! This does not replace the real ControllerSnapshot/backend hardware gate.
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::{
        BatchOrderSemantics, ExecutionFingerprint, WaveExecutionPath, WaveExecutionShape,
        WaveGraphState,
    },
    slo_planner::*,
    LogicalWorkGeneration,
};
use std::cell::{Cell, RefCell};

const ALGORITHM: &str = "fixture.automatic-published-planner";

#[derive(Clone)]
struct CpuProjection {
    domain: Arc<CostWorkloadDomainV1>,
    host: HostCostFeaturesV1,
    key: RequestWorkKey,
    projections: Arc<AtomicU64>,
}
impl PlanningExecutionContext for CpuProjection {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        poll()?;
        if snapshot.requests.len() != 1 || snapshot.requests[0].key != self.key {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        Ok(Arc::new(self.clone()))
    }
}
impl<'epoch> PlanningExecutionState<'epoch> for CpuProjection {
    fn cost_workload_domain(&self) -> Option<&CostWorkloadDomainV1> {
        Some(&self.domain)
    }

    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        poll()?;
        let [row] = input.rows else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        let [work] = input.work else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        let ActualRowWork::Decode { kv_tokens } = row.work else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        if input.kind != ActualWaveKind::Decode
            || input.requests.len() != 1
            || input.requests[0] != *row.request
            || row.request.key != self.key
            || work.key != self.key
            || work.action != WaveAction::Decode
            || row.request.context_tokens != kv_tokens
            || row.request.recurrent_state_bytes != 64
            || input.recurrent_state_bytes != 64
            || row.request.output_policy_signature != [6; 32]
            || row.request.timing.completed()
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        // Generate the canonical work from this projected frontier, not a
        // replay of the training shape. FullGeneration makes history exact.
        let mut host = self.host;
        host.state.generated_tokens_before = u64::from(row.request.timing.committed_tokens);
        host.state.maximum_output_tokens =
            u64::from(row.request.timing.maximum_output_tokens.get());
        host.state.sampling_history_tokens = host.state.generated_tokens_before;
        let projected = wave_with_host(ALGORITHM, kv_tokens, host);
        self.projections.fetch_add(1, Ordering::Relaxed);
        Ok(Some(ProjectedExecution {
            host_content_forecasts: Some(PlanningShapeDomain::Exact(
                ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
            )),
            statistical_evidence: Some(PlanningShapeDomain::Exact(projected.prepared.selected)),
            ordered_work: input.work.to_vec(),
            canonical_domain: PlanningShapeDomain::Exact(projected.prepared.exact),
            successor: Arc::new(self.clone()),
        }))
    }
}

struct Lookup<'a> {
    model: &'a profile::EngineCostSnapshot,
    known: Cell<usize>,
    failures: RefCell<Vec<PlanningQueryOutcome>>,
}
impl PlanningCostModel for Lookup<'_> {
    fn model_version(&self) -> u64 {
        self.model.model_version()
    }
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        self.model.evidence_requirement()
    }
    fn supports_empirical_host_content(&self) -> bool {
        self.model.supports_empirical_host_content()
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        _: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        panic!("published structured model must be queried with planner-bound evidence")
    }
    fn predict_with_evidence(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        // Observe the normal model call without creating or replacing its answer.
        let result = self
            .model
            .predict_observed(fingerprint, shape, evidence, now_ns);
        if let Some(cost) = result.cost() {
            assert_eq!(cost.model_version, self.model.model_version());
            assert!(cost.planning_ns > 0 && cost.valid_for_ns > 0);
            self.known.set(self.known.get() + 1);
        } else {
            self.failures.borrow_mut().push(result.outcome);
        }
        result.cost()
    }
}
struct FixedClock(u64);
impl PlanningClock for FixedClock {
    fn now_ns(&mut self) -> u64 {
        self.0
    }
}

#[tokio::test]
async fn automatic_publication_reaches_complete_planner_and_selected_replay_before_execution() {
    let identity = identity();
    let ExecutorCostIdentityAvailability::Known(executor) = &identity else {
        panic!("fixture identity");
    };
    let domain = CostWorkloadDomainV1::new_vnext(
        executor,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(1).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(1).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
            repetition_slot_capacity: 0,
            fixed_state_bytes_per_row: 64,
        },
    )
    .unwrap();
    let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 {
        settings: SloAutomaticCalibrationSettingsV1 {
            discovery_offered_waves: NonZeroUsize::new(8).unwrap(),
            phase_offered_waves: [NonZeroUsize::new(8).unwrap(); 3],
            ..Default::default()
        },
    };
    let runtime = EngineCostRuntime::build_with_profile_and_domain(
        identity,
        clock.clone(),
        &config,
        false,
        None,
        None,
        Some(domain.clone()),
    )
    .unwrap();
    assert!(runtime.snapshot().is_none());
    runtime.begin_automatic_calibration().unwrap();
    runtime.consume_samples();
    let (_, hosts) = selected_shape(&[3]);
    let host = hosts[0]; // Genuine ordinary decode: two committed tokens, one left.
    for phase in 0..4 {
        // Advance the completed original block's FIFO barrier without an offer.
        runtime.consume_samples();
        for _ in 0..8 {
            fixture::record_route(
                &runtime,
                &clock,
                wave_with_host(ALGORITHM, 7, host),
                false,
                false,
            )
            .unwrap();
            runtime.consume_samples();
        }
        let audit = runtime.training.live.as_ref().unwrap().audit();
        assert_eq!(audit.failed_generations, 0, "{audit:#?}");
        assert!(audit.publication_error.is_none(), "{audit:#?}");
        assert_eq!(
            audit.qualified_publications,
            u64::from(phase == 3),
            "{audit:#?}"
        );
    }
    let receipt = runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.storage, SloCostProfileStorage::Memory);
    assert!(receipt.path.is_none());
    assert_eq!(
        receipt.offered_samples, 32,
        "complete source7 prefix includes discovery"
    );
    let children = runtime
        .training
        .live_catalog_children(clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(
        children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|phase| phase.members),
        [8; 3]
    );
    let published = runtime.snapshot().unwrap();
    let now = clock.now_ns().unwrap();
    let key = RequestWorkKey {
        request_id: RequestId::new(),
        incarnation: 1,
        work_generation: LogicalWorkGeneration::default(),
    };
    let request = RequestSchedulingView {
        key: key.clone(),
        timing: RequestTimingView {
            ingress_at_ns: now - 3,
            first_commit_at_ns: Some(now - 2),
            last_commit_at_ns: Some(now - 1),
            committed_tokens: 2,
            maximum_output_tokens: NonZeroU32::new(3).unwrap(),
            budgets: PlannerLatencyBudgets {
                ttft_ns: NonZeroU64::new(1_000_000_000).unwrap(),
                tpot_ns: NonZeroU64::new(1_000_000_000).unwrap(),
                itl_ns: NonZeroU64::new(1_000_000_000).unwrap(),
            },
            slo_failed: false,
        },
        phase: RequestPhaseView::Decode,
        readiness: RequestReadiness::Ready,
        context_tokens: 8,
        recurrent_state_bytes: 64,
        output_credit: OutputCreditView {
            available_token_commands: 1,
            byte_backing: OutputByteBacking::PrepaidLifetime {
                remaining_token_commands: 1,
                remaining_wire_bytes: 1024,
            },
        },
        output_policy_signature: [6; 32],
        fairness_rank: 0,
        recovery_service: RecoveryServiceDebt::new(NonZeroUsize::new(256).unwrap()),
        ranking_service_cost_ns: None,
        optimistic_next_service: None,
    };
    let snapshot = SchedulerSnapshot {
        observed_at_ns: now,
        generation: 1,
        cost_model_version: published.model_version(),
        fingerprint: published.fingerprint().clone(),
        requests: vec![request],
        capabilities: BackendPlanningCapabilities {
            work_policy: Default::default(),
            path: WaveExecutionPath::PlanRuntime,
            graph_state: WaveGraphState::Disabled,
            order: BatchOrderSemantics::Ordered,
            decode_batch_sizes: vec![NonZeroUsize::MIN],
            prefill_batch_sizes: vec![],
            prefill_chunk_sizes: vec![],
            prefill_alignment: NonZeroU32::MIN,
            allow_final_short_chunk: true,
            native_mixed: false,
            max_wave_rows: NonZeroUsize::MIN,
            max_prefill_tokens_per_wave: NonZeroU64::MIN,
            workspace_bytes_upper_bound: 0,
        },
        capacity: CapacityReadView {
            evidence_known: true,
            available_kv_tokens: 128,
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            available_workspace_bytes: 0,
            available_output_bytes: 0,
        },
        scope: PlanningScope {
            horizon_end_ns: now + 1_000_000_000,
            reference_decode_token_ns: NonZeroU64::new(10).unwrap(),
            reference_work_version: 1,
        },
        has_unmodeled_maintenance: false,
    };
    let before = snapshot.clone();
    let original_calls = runtime.sink.stats().raw_offered;
    let original_issued = runtime
        .training
        .live
        .as_ref()
        .unwrap()
        .audit()
        .population
        .issued;
    let projections = Arc::new(AtomicU64::new(0));
    let backend = CpuProjection {
        domain: Arc::new(domain),
        host,
        key: key.clone(),
        projections: projections.clone(),
    };
    let lookup = Lookup {
        model: &published,
        known: Cell::new(0),
        failures: RefCell::new(Vec::new()),
    };
    let decision = BoundedSloPlanner::default().propose_with_execution(
        &snapshot,
        &lookup,
        &backend,
        &mut FixedClock(now),
    );
    let PlanningDecision::FeasibleWithinHorizon {
        first_wave,
        witness,
        search,
    } = decision
    else {
        panic!(
            "published model failed full planner: {decision:?}; lookups={:?}",
            lookup.failures.borrow()
        );
    };
    assert!(lookup.failures.borrow().is_empty());
    assert!(
        lookup.known.get() > 0,
        "real planner must query the published model"
    );
    assert_eq!(search.phase, PlanningSearchPhase::Finalization);
    assert!(first_wave.final_replay_first_wave.is_some());
    let (canonical, _, query) = first_wave
        .replayed_first_wave_structured_v2(&snapshot)
        .unwrap();
    assert_eq!(canonical.rows, vec![ActualRowWork::Decode { kv_tokens: 8 }]);
    assert_eq!(query.owner().role, StructuredWaveRoleV2::OrdinaryDecode);
    assert_eq!(
        query.input().physical_domain_signature(),
        Some(backend.domain.sha256())
    );
    assert_eq!(
        first_wave.candidate.work,
        vec![CandidateWork {
            key,
            action: WaveAction::Decode
        }]
    );
    assert_eq!(first_wave.cost_model_version, receipt.model_version);
    assert!(first_wave.predicted_wall_ns > 0 && first_wave.witness_valid_for_ns > 0);
    assert_eq!(witness.predicted_output_tokens, 1);
    assert_eq!(witness.requests_with_obligations_beyond_horizon, 0);
    assert_eq!(
        snapshot, before,
        "search/replay cannot advance the live request"
    );
    assert!(projections.load(Ordering::Relaxed) > 0);
    assert_eq!(
        runtime.sink.stats().raw_offered,
        original_calls,
        "the new request was never physically executed to obtain its prediction"
    );
    assert_eq!(
        runtime
            .training
            .live
            .as_ref()
            .unwrap()
            .audit()
            .population
            .issued,
        original_issued,
        "planning cannot issue another calibration ticket"
    );
    runtime.shutdown().await.unwrap();
}
