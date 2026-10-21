//! Controlled snapshots test the Engine consumer and waits. Actual reference
//! learning is exercised separately by ferrum-slo's real CPU integration.

use super::*;
use crate::continuous_engine::attempt_observation_tests::AttemptRecorder;
use ferrum_scheduler::cost_prediction::{
    CostPrediction, CostQuery, CostRisk, CostSnapshot, CostSupport, CostUnknownReason, CostWork,
};
use tokio::sync::watch;

struct Snapshot {
    revision: u64,
    known: bool,
    created: Instant,
    queries: Arc<Mutex<Vec<u64>>>,
    work_queries: Arc<Mutex<Vec<(u64, Vec<CostWork>)>>>,
    queried: Arc<Notify>,
}

impl CostSnapshot for Snapshot {
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        ferrum_types::CostQueryLimits::default()
    }
    fn revision(&self) -> u64 {
        self.revision
    }

    fn predict(&self, query: &CostQuery<'_>, _now: Instant) -> CostPrediction {
        assert!(query.work_is_exact);
        assert!(!query.work.is_empty());
        self.queries.lock().push(self.revision);
        self.work_queries
            .lock()
            .push((self.revision, query.work.to_vec()));
        self.queried.notify_one();
        if self.known {
            CostPrediction::Known {
                central_service_time: Duration::from_micros(10),
                upper_service_time: Duration::from_micros(10),
                risk: CostRisk::EngineeringMargin,
                support: CostSupport::ExactObservedWork,
                samples: 4,
                observed_at: self.created,
                expires_at: self.created + Duration::from_secs(3600),
            }
        } else {
            CostPrediction::Unknown(CostUnknownReason::NoCoverage)
        }
    }
}

pub(super) struct Source {
    current: Mutex<Arc<dyn CostSnapshot>>,
    publication: Mutex<Option<watch::Sender<u64>>>,
    queries: Arc<Mutex<Vec<u64>>>,
    work_queries: Arc<Mutex<Vec<(u64, Vec<CostWork>)>>>,
    queried: Arc<Notify>,
    loads: AtomicU64,
    subscriptions: AtomicU64,
    pub(super) replace_after_load: AtomicBool,
}

impl Source {
    pub(super) fn new(known: bool) -> Arc<Self> {
        let queries = Arc::new(Mutex::new(Vec::new()));
        let work_queries = Arc::new(Mutex::new(Vec::new()));
        let queried = Arc::new(Notify::new());
        let (publication, _) = watch::channel(1);
        Arc::new(Self {
            current: Mutex::new(Arc::new(Snapshot {
                revision: 1,
                known,
                created: Instant::now(),
                queries: Arc::clone(&queries),
                work_queries: Arc::clone(&work_queries),
                queried: queried.clone(),
            })),
            publication: Mutex::new(Some(publication)),
            queries,
            work_queries,
            queried,
            loads: AtomicU64::new(0),
            subscriptions: AtomicU64::new(0),
            replace_after_load: AtomicBool::new(false),
        })
    }

    pub(super) fn take_work_queries(&self) -> Vec<(u64, Vec<CostWork>)> {
        std::mem::take(&mut *self.work_queries.lock())
    }

    pub(super) fn load_count(&self) -> u64 {
        self.loads.load(Ordering::Relaxed)
    }

    fn publish(&self, revision: u64, known: bool) {
        *self.current.lock() = Arc::new(Snapshot {
            revision,
            known,
            created: Instant::now(),
            queries: Arc::clone(&self.queries),
            work_queries: Arc::clone(&self.work_queries),
            queried: self.queried.clone(),
        });
        if let Some(publication) = self.publication.lock().as_ref() {
            publication.send_replace(revision);
        }
    }
}

impl CostSnapshotSource for Source {
    fn load(&self) -> Arc<dyn CostSnapshot> {
        self.loads.fetch_add(1, Ordering::Relaxed);
        let original = self.current.lock().clone();
        // Simulate a concurrent publication immediately after this load. Any
        // accidental reload during an admission subround would now see v2.
        if self.replace_after_load.swap(false, Ordering::AcqRel) {
            self.publish(2, true);
        }
        original
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        self.subscriptions.fetch_add(1, Ordering::Relaxed);
        self.publication
            .lock()
            .as_ref()
            .expect("test source is open")
            .subscribe()
    }
}

fn policy() -> ferrum_types::SchedulerSloConfig {
    "ttft:1000,tpot:100,itl:100".parse().unwrap()
}

fn engine(
    mut config: EngineConfig,
    source: Arc<Source>,
    serial_growth: bool,
) -> (
    ContinuousBatchEngine,
    Arc<PlanRuntimeAdmissionObservation>,
    Arc<dyn Tokenizer + Send + Sync>,
) {
    config.runtime.profile_detail = ObservabilityProfileDetail::Off;
    let implementation = if serial_growth {
        PlanRuntimeAdmissionTestExecutor::new_with_serial_backing_growth(128)
    } else {
        PlanRuntimeAdmissionTestExecutor::new(128)
    };
    let observation = implementation.observer();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        128,
        &[("test", 5), ("ok", 6), ("generated", 42)],
    ));
    let engine = ContinuousBatchEngine::new_observed(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        ExecutorOwner::new(Box::new(implementation)),
        None,
        Some(Arc::new(AttemptRecorder::default())),
        Some(source),
    )
    .unwrap();
    (engine, observation, tokenizer)
}

pub(super) async fn submit(
    engine: &ContinuousBatchEngine,
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
    prompt: &str,
) -> RequestId {
    let mut request = policy_request();
    request.prompt = prompt.to_owned();
    request.sampling_params.max_tokens = 3;
    let tokens = tokenizer.encode(prompt, true).unwrap();
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.to_owned(),
        serde_json::json!(tokens.len()),
    );
    request.ensure_ingress(ferrum_types::RequestIngressBoundary::EngineApi);
    let id = request.id.clone();
    engine.active_inner().unwrap().sequences.insert(
        id.clone(),
        SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens,
            Some(tokenizer.clone()),
            Some(128),
        ),
    );
    engine.scheduler.submit(request).await.unwrap();
    id
}

#[tokio::test]
async fn cost_feedback_requires_both_inputs_and_keeps_default_off() {
    let source = Source::new(true);
    let observer = Arc::new(AttemptRecorder::default());
    let mut config = EngineConfig::default();
    config.scheduler.slo = Some(policy());
    for observed in [false, true] {
        let mut builder = crate::EngineBuilder::new(config.clone());
        if observed {
            builder = builder.with_execution_attempt_observer(observer.clone());
        }
        let error = builder
            .build()
            .await
            .err()
            .expect("missing source must fail construction");
        assert!(error
            .to_string()
            .contains("requires an execution observer and cost snapshot source"));
    }
    let error = crate::EngineBuilder::new(EngineConfig::default())
        .with_execution_feedback(observer, source.clone())
        .build()
        .await
        .err()
        .unwrap();
    assert!(error
        .to_string()
        .contains("requires an enabled scheduler SLO policy"));
    assert_eq!(source.loads.load(Ordering::Relaxed), 0);
    assert_eq!(source.subscriptions.load(Ordering::Relaxed), 0);

    let (off, _, _, _) = plan_runtime_test_engine(
        EngineConfig::default(),
        PlanRuntimeAdmissionTestExecutor::new(128),
    );
    let inner = off.active_inner().unwrap();
    assert!(inner.execution_attempt_observer.is_none());
    assert!(inner.cost_snapshot_source.is_none());
    drop(inner);
    off.shutdown().await.unwrap();
    off.wait_for_shutdown_completion().await.unwrap();
}

#[tokio::test]
async fn cost_feedback_one_snapshot_spans_backing_growth_rounds_and_final_selection() {
    let mut config = EngineConfig::default();
    config.scheduler.slo = Some(policy());
    config.scheduler.max_running_requests = 3;
    config.scheduler.prefill_first_until_active = Some(2);
    config.batching.max_batch_size = 2;
    config.batching.max_num_batched_tokens = 16;
    let source = Source::new(true);
    source.replace_after_load.store(true, Ordering::Release);
    let (engine, executor, tokenizer) = engine(config, source.clone(), true);
    submit(&engine, &tokenizer, "test").await;
    submit(&engine, &tokenizer, "ok").await;
    engine
        .active_inner()
        .unwrap()
        .run_iteration()
        .await
        .unwrap();
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 3);
    assert_eq!(executor.batch_prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(source.loads.load(Ordering::Relaxed), 1);
    let queries = std::mem::take(&mut *source.queries.lock());
    assert!(
        queries.len() > 2,
        "admission subrounds and final selection must query"
    );
    assert!(queries.iter().all(|revision| *revision == 1));
    assert_eq!(source.current.lock().revision(), 2);
    assert_eq!(engine.scheduler.decoding_count(), 2);
    submit(&engine, &tokenizer, "test").await;
    engine
        .active_inner()
        .unwrap()
        .run_iteration()
        .await
        .unwrap();
    assert_eq!(source.loads.load(Ordering::Relaxed), 2);
    let late = source.queries.lock().clone();
    assert!(
        !late.is_empty(),
        "the next candidate must query the new revision"
    );
    assert!(late.iter().all(|revision| *revision == 2));
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    assert!(executor.retained.lock().unwrap().is_empty());
    assert!(executor.tokens.0.upgrade().is_none());
}

#[tokio::test]
async fn cost_feedback_wait_receives_publication_between_planning_and_sleep() {
    let mut config = EngineConfig::default();
    let mut slo = policy();
    slo.unknown_cost = ferrum_types::UnknownCostPolicy::Wait {
        max_wait_ms: 60_000,
        on_timeout: ferrum_types::CostFallback::Reject,
    };
    config.scheduler.slo = Some(slo);
    let source = Source::new(false);
    let (engine, executor, tokenizer) = engine(config, source.clone(), false);
    let inner = engine.active_inner().unwrap();
    let mut updates = Some(source.subscribe());
    drop(updates.as_mut().unwrap().borrow_and_update());
    submit(&engine, &tokenizer, "test").await;
    let EngineIterationOutcome::Waiting { capacity, retry_at } =
        inner.run_iteration().await.unwrap()
    else {
        panic!("time-only blocked work must wait rather than yield");
    };
    assert!(capacity.is_none());
    assert!(retry_at.is_some_and(|deadline| deadline > Instant::now()));
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 0);
    assert_eq!(engine.scheduler.waiting_count(), 1);
    source.publish(2, true);
    let mut wait = tokio_test::task::spawn(managed_driver::wait_for_iteration_wake(
        &inner,
        capacity,
        retry_at,
        &mut updates,
    ));
    tokio_test::assert_ready!(wait.poll());
    drop(wait);
    inner.run_iteration().await.unwrap();
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(engine.scheduler.waiting_count(), 0);
    drop(inner);
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    assert!(executor.tokens.0.upgrade().is_none());
}

#[tokio::test]
async fn cost_feedback_original_driver_resumes_held_request_on_snapshot_publication() {
    use futures::StreamExt;
    let mut config = EngineConfig::default();
    let mut slo = policy();
    slo.unknown_cost = ferrum_types::UnknownCostPolicy::Wait {
        max_wait_ms: 60_000,
        on_timeout: ferrum_types::CostFallback::Reject,
    };
    config.scheduler.slo = Some(slo);
    let source = Source::new(false);
    let (engine, executor, _) = engine(config, source.clone(), false);
    let original_inner = Arc::downgrade(&engine.active_inner().unwrap());
    let mut request = policy_request();
    request.prompt = "test".to_owned();
    request.sampling_params.max_tokens = 2;
    let stream = engine.infer_stream(request).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), source.queried.notified())
        .await
        .unwrap();
    assert_eq!(source.subscriptions.load(Ordering::Relaxed), 1);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 0);
    assert_eq!(engine.scheduler.waiting_count(), 1);
    source.publish(2, true);
    let chunks = tokio::time::timeout(Duration::from_secs(2), stream.collect::<Vec<_>>())
        .await
        .unwrap();
    let chunks = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
    assert_eq!(
        chunks
            .iter()
            .filter_map(|chunk| chunk.chunk().token)
            .count(),
        2
    );
    assert!(executor.admission_probes.load(Ordering::Relaxed) > 0);
    assert_eq!(engine.scheduler.waiting_count(), 0);
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    assert!(original_inner.upgrade().is_none());
    assert!(executor.tokens.0.upgrade().is_none());
}

#[tokio::test(start_paused = true)]
async fn cost_feedback_closed_watch_preserves_deadline_and_shutdown_waits() {
    let mut config = EngineConfig::default();
    config.scheduler.slo = Some(policy());
    let source = Source::new(false);
    let (engine, _, _) = engine(config, source.clone(), false);
    let inner = engine.active_inner().unwrap();
    let mut updates = Some(source.subscribe());
    drop(source.publication.lock().take());
    let mut wait = tokio_test::task::spawn(managed_driver::wait_for_iteration_wake(
        &inner,
        None,
        Some(Instant::now() + Duration::from_secs(10)),
        &mut updates,
    ));
    tokio_test::assert_pending!(wait.poll());
    tokio::time::advance(Duration::from_secs(11)).await;
    tokio_test::assert_ready!(wait.poll());
    drop(wait);
    assert!(
        updates.is_none(),
        "closed source must be removed from select"
    );
    let mut wait = tokio_test::task::spawn(managed_driver::wait_for_iteration_wake(
        &inner,
        None,
        None,
        &mut updates,
    ));
    tokio_test::assert_pending!(wait.poll());
    inner.shutdown_notify.notify_one();
    tokio_test::assert_ready!(wait.poll());
    drop(wait);
    drop(inner);
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
}
