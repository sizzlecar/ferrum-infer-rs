//! Genuine provider -> resource route -> V2 input binding -> unknown lookup.
//! No profile, timing estimate or qualified cost sample is manufactured.
use super::*;
use crate::continuous_engine::inner::cost_observation::EngineCostRuntime;
use ferrum_interfaces::ModelExecutor;
use ferrum_scheduler::implementations::continuous::cost_model::{
    structured_v2::{StructuredQueryV2, StructuredUnknownV2},
    ExecutionFingerprint, WaveExecutionShape,
};

#[derive(Default)]
struct Tap {
    depths: Mutex<Vec<usize>>,
    constructed: AtomicUsize,
    queried: AtomicUsize,
}
impl PlanningQueryObserver for Tap {
    fn begin_replay(&self, _: usize) -> u64 {
        panic!("unknown cost cannot create a replayable solution")
    }
    fn end_replay(&self, _: u64, _: PlanningQueryAttemptEnd) {
        panic!("unknown cost cannot replay")
    }
    fn selected_replay(&self, _: u64) {
        panic!("unknown cost cannot select")
    }
    fn begin_attempt(
        &self,
        phase: PlanningQueryPhase,
        depth: usize,
        _: &[RequestSchedulingView],
        _: &[CandidateWork],
    ) -> u64 {
        assert_eq!(phase, PlanningQueryPhase::Search);
        let mut depths = self.depths.lock();
        depths.push(depth);
        depths.len() as u64
    }
    fn constructed(
        &self,
        _: PlanningQueryKey,
        query: std::result::Result<&StructuredQueryV2, StructuredUnknownV2>,
    ) {
        let query = query.expect("real selected provider must bind a V2 query");
        assert!(!query
            .required_coverage()
            .unwrap()
            .joint_support_coordinates
            .is_empty());
        self.constructed.fetch_add(1, Ordering::Relaxed);
    }
    fn lookup(&self, _: PlanningQueryKey, _: u64, result: PlanningObservedCost) {
        assert_eq!(result.outcome, PlanningQueryOutcome::ModelUnavailable);
        assert!(result.cost_now_ns.is_some(), "retain the real cost clock");
        assert!(result.cost().is_none());
        self.queried.fetch_add(1, Ordering::Relaxed);
    }
    fn end_attempt(&self, _: u64, constructed: usize, queried: usize, _: PlanningQueryAttemptEnd) {
        assert!(queried <= constructed);
        assert!(queried <= 1, "first unknown must stop each edge");
    }
}
struct TappedModel {
    model: Arc<dyn PlanningCostModel + Send + Sync>,
    tap: Arc<Tap>,
}
impl PlanningCostModel for TappedModel {
    fn query_observer(&self) -> Option<&dyn PlanningQueryObserver> {
        Some(self.tap.as_ref())
    }
    fn model_version(&self) -> u64 {
        self.model.model_version()
    }
    fn requires_statistical_evidence(&self) -> bool {
        self.model.requires_statistical_evidence()
    }
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        self.model.evidence_requirement()
    }
    fn predict(
        &self,
        fp: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        now: u64,
    ) -> Option<PlanningCost> {
        self.model.predict(fp, shape, now)
    }
    fn predict_observed(
        &self,
        fp: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now: u64,
    ) -> PlanningObservedCost {
        self.model.predict_observed(fp, shape, evidence, now)
    }
}

async fn uncalibrated_fixture() -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (mut engine, scheduler, executor) = fixture().await;
    executor.enable_structured_query_route();
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.cost_observation =
        ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    inner.config.scheduler.slo.required_query_observation =
        ferrum_types::SloRequiredQueryObservationConfig::StructuredUncalibratedV1 {
            path: "unused-fixture-queries.jsonl".into(),
            limits: Default::default(),
        };
    // The fixture already installed a validated reference runtime; avoid file IO
    // and leave the shared startup writer tests responsible for queue/file behavior.
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::new(
            executor.execution_cost_identity(),
            &inner.config.scheduler.slo.cost_observation,
            None,
        )
        .unwrap(),
    ));
    assert!(inner.cost_runtime.as_ref().unwrap().snapshot().is_none());
    (engine, scheduler, executor)
}

#[tokio::test]
async fn uncalibrated_required_query_observation_reaches_real_root_lookup_and_legacy() {
    let (mut engine, _, executor) = uncalibrated_fixture().await;
    let journal = std::env::temp_dir().join(format!(
        "ferrum-uncalibrated-shared-entry-{}.jsonl",
        uuid::Uuid::new_v4()
    ));
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.required_query_observation =
        ferrum_types::SloRequiredQueryObservationConfig::StructuredUncalibratedV1 {
            path: journal.clone(),
            limits: Default::default(),
        };
    let writer = crate::continuous_engine::query_observation::Writer::open(
        &inner.config.scheduler.slo.required_query_observation,
    )
    .unwrap()
    .unwrap();
    writer
        .bind_identity(&inner.config.scheduler.slo, None)
        .unwrap();
    inner.required_query_observation = Some(writer.clone());
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    let before = prefill::UnsubmittedState::capture(&engine, &executor);
    let mut captured =
        prefill::captured_with_hint(&engine, &executor, &ferrum_interfaces::BatchHint::simple(1))
            .await;
    assert_eq!(captured.snapshot.cost_model_version, 0);
    assert_eq!(
        captured.model.evidence_requirement(),
        PlanningCostEvidenceRequirement::StructuredV2
    );
    assert!(!captured.model.supports_empirical_host_content());
    let tap = Arc::new(Tap::default());
    captured.model = Arc::new(TappedModel {
        model: captured.model.clone(),
        tap: tap.clone(),
    });
    let decision = engine.inner.propose_slo_controller(&captured);
    let PlanningDecision::Unknown { search, .. } = decision else {
        panic!("uncalibrated input authorized a finite cost witness: {decision:?}")
    };
    assert!(search.cost_unknown_candidates > 0, "{search:?}");
    // Search stats count the root layer as one; observer depth is zero-based.
    assert_eq!(search.max_depth_reached, 1);
    assert!(tap.constructed.load(Ordering::Relaxed) > 0);
    assert!(tap.queried.load(Ordering::Relaxed) > 0);
    assert!(tap.depths.lock().iter().all(|depth| *depth == 0));
    before.assert_unchanged(&engine, &executor);
    drop(captured);

    // Exercise the real shared controller exit, not just the adapter method.
    assert!(matches!(
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1),)
            .unwrap(),
        SloIterationPlan::Legacy
    ));
    let audit = engine.inner.slo_controller.lock().last_audit.unwrap();
    assert_eq!(audit.outcome, "observed");
    assert_eq!(audit.decision, "unknown");
    assert_eq!(audit.reason, "cost_unavailable");
    // CompleteRequests goes through TimeAdmissionProposal::unknown, whose
    // existing API retains the reason but resets aggregate search stats.
    // Inspect the actual shared-entry callbacks, not that lossy summary.
    writer.close().unwrap();
    let records = std::fs::read_to_string(&journal)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(
        records
            .iter()
            .any(|r| r["event"] == "query_constructed" && r["data"]["demand"].is_object()),
        "{records:?}"
    );
    assert!(
        records.iter().any(|r| r["event"] == "query_lookup"
            && r["data"]["outcome"]["kind"] == "model_unavailable"
            && r["data"]["cost_now_ns"].as_u64().is_some()),
        "{records:?}"
    );
    assert!(records
        .iter()
        .filter(|r| r["event"] == "attempt_begin")
        .all(|r| r["data"]["depth"] == 0 && r["data"]["phase"] == "Search"));
    assert!(!records.iter().any(|r| matches!(
        r["event"].as_str(),
        Some("replay_begin" | "replay_end" | "selected_replay")
    )));
    assert_eq!(records.last().unwrap()["recording_complete"], true);
    assert!(!audit.backend_submitted);
    before.assert_unchanged(&engine, &executor);
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    cleanup(engine, session).await;
    std::fs::remove_file(journal).unwrap();
}

#[tokio::test]
async fn uncalibrated_required_query_observation_keeps_reference_and_enforce_boundaries() {
    let (mut engine, _, executor) = uncalibrated_fixture().await;
    // Remove the actual reference before publishing any request.
    // Missing input never creates queries.
    let reference = Arc::get_mut(&mut engine.inner)
        .unwrap()
        .prefill_reference_runtime
        .take();
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    let before = prefill::UnsubmittedState::capture(&engine, &executor);
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let error = engine
        .inner
        .capture_slo_controller_snapshot(
            &hint,
            ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
        )
        .err()
        .expect("reference must remain required");
    assert_eq!(error.reason, "missing_reference_work");
    before.assert_unchanged(&engine, &executor);
    // Live requests retain engine owners, so use the same configuration pattern
    // as other controller fixtures only after cleanup in a new independent fixture.
    drop(reference);
    cleanup(engine, session).await;

    let (mut engine, _, executor) = uncalibrated_fixture().await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    let before = prefill::UnsubmittedState::capture(&engine, &executor);
    let error = engine
        .inner
        .capture_slo_controller_snapshot(
            &hint,
            ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
        )
        .err()
        .expect("Enforce must never get an uncalibrated adapter");
    assert_eq!(error.reason, "uncalibrated_observation_scope_invalid");
    before.assert_unchanged(&engine, &executor);
    cleanup(engine, session).await;
}
