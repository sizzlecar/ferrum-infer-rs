//! Admission scheduling is injected; the retained cache and its terminal
//! identity belong to the existing real CPU Llama fixture.
use super::*;
use crate::continuous_engine::token_completion_tests::{
    observed_cpu_model_owned, ModelObservation,
};
use ferrum_execution::model_executor::{ExecutorCapabilities, ExecutorStatus};
use ferrum_execution::token_execution::{
    ExecutionBatchOutcome, ExecutionCloseDisposition, ExecutionCloseOutcome, ExecutionWork,
    ExecutorSequence,
};
use ferrum_models::test_support::TinyTokenizer;
use std::time::Duration;
use tokio::sync::oneshot;

#[path = "tests/readiness.rs"]
mod readiness;
#[path = "tests/sequence_slots.rs"]
mod sequence_slots;

struct AdmissionGate {
    inner: Arc<ExecutorScope>,
    owner: ExecutorOwner,
    observation: Arc<AdmissionObservation>,
    ready: Mutex<Option<oneshot::Sender<ExecutorSequence>>>,
    resume: Mutex<Option<oneshot::Receiver<()>>>,
}

#[derive(Default)]
struct AdmissionObservation {
    closed: Mutex<Vec<ExecutorSequence>>,
    slot_capacity: Mutex<Option<usize>>,
    slot_owners: Mutex<Vec<ExecutorSequence>>,
    admission_attempts: std::sync::atomic::AtomicUsize,
    close_gate: Mutex<Option<(oneshot::Sender<ExecutorSequence>, oneshot::Receiver<()>)>>,
}

impl std::ops::Deref for AdmissionGate {
    type Target = AdmissionObservation;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

impl AdmissionGate {
    fn into_shutdown(self) -> ferrum_execution::ExecutorShutdownObserver {
        let Self {
            inner,
            owner,
            observation,
            ready,
            resume,
        } = self;
        drop((inner, observation, ready, resume));
        owner.begin_shutdown()
    }
}

// Test-only envelope: cancelling the injected await must not discard an
// already admitted real cache. The production executor has its own envelope.
struct PendingAdmission<'a> {
    executor: &'a AdmissionGate,
    frontier: Option<ExecutionFrontier>,
}

impl Drop for PendingAdmission<'_> {
    fn drop(&mut self) {
        if let Some(frontier) = self.frontier.take() {
            let outcome = self.executor.cancel_tokens(frontier.sequence()).unwrap();
            assert!(outcome.error.is_none());
            assert!(matches!(
                outcome.disposition,
                ExecutionCloseDisposition::Ended(_)
            ));
        }
    }
}

#[async_trait::async_trait]
impl ModelExecutor for AdmissionGate {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::ExecutorShutdownPermit,
    ) -> ferrum_execution::ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ferrum_execution::ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        match Arc::try_unwrap(self) {
            Ok(original) => original.into_shutdown(),
            Err(original) => ferrum_execution::ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal("admission fixture has an unscoped owner"),
            ),
        }
    }
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        self.admission_attempts.fetch_add(1, Ordering::SeqCst);
        if let Some(capacity) = *self.slot_capacity.lock() {
            let occupied = self
                .slot_owners
                .lock()
                .iter()
                .filter(|owner| owner.observed_end().is_none())
                .count();
            if occupied >= capacity {
                return Ok(TokenAdmissionOutcome::SequenceSlotsDeferred(
                    ferrum_execution::token_execution::TokenSequenceSlotDeferral::new(
                        capacity, occupied, 1,
                    )?,
                ));
            }
        }
        let outcome = self.inner.admit_tokens(input).await?;
        let TokenAdmissionOutcome::Admitted(frontier) = outcome else {
            return Ok(outcome);
        };
        self.slot_owners.lock().push(frontier.sequence().clone());
        let mut pending = PendingAdmission {
            executor: self,
            frontier: Some(frontier),
        };
        let resume = self.resume.lock().take();
        if let Some(resume) = resume {
            self.ready
                .lock()
                .take()
                .unwrap()
                .send(pending.frontier.as_ref().unwrap().sequence().clone())
                .unwrap();
            resume
                .await
                .map_err(|_| FerrumError::internal("admission test gate was dropped"))?;
        }
        Ok(TokenAdmissionOutcome::Admitted(
            pending.frontier.take().unwrap(),
        ))
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        self.inner.execute_tokens(work).await
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        let gate = self.close_gate.lock().take();
        if let Some((entered, resume)) = gate {
            entered.send(sequence.clone()).unwrap();
            resume
                .await
                .map_err(|_| FerrumError::internal("physical close test gate dropped"))?;
        }
        self.inner.finish_tokens(sequence, reason).await
    }

    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        let outcome = self.inner.cancel_tokens(sequence)?;
        if let ExecutionCloseDisposition::Ended(ended) = &outcome.disposition {
            assert_eq!(ended.sequence(), sequence);
            self.closed.lock().push(sequence.clone());
        }
        Ok(outcome)
    }
}

struct Fixture {
    engine: ContinuousBatchEngine,
    executor: Arc<AdmissionObservation>,
    model: Arc<Mutex<ModelObservation>>,
    request: InferenceRequest,
    response: oneshot::Receiver<Result<InferenceOutput>>,
    ready: Option<oneshot::Receiver<ExecutorSequence>>,
    resume: Option<oneshot::Sender<()>>,
    trace: tempfile::TempDir,
    layers: usize,
}

async fn install_outer(
    engine: &ContinuousBatchEngine,
    mut request: InferenceRequest,
    prompt: Vec<TokenId>,
) -> oneshot::Receiver<Result<InferenceOutput>> {
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.into(),
        serde_json::json!(prompt.len()),
    );
    let (sender, response) = oneshot::channel();
    let mut state = SequenceState::new(request.clone(), prompt);
    state.response_sender = Some(sender);
    let mut slot = RequestSlotLease::open(
        &engine.active_inner().expect("engine execution is open"),
        request.id.clone(),
    );
    slot.admit(&engine.active_inner().expect("engine execution is open"));
    state.request_slot = Some(slot);
    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .insert(request.id.clone(), state)
        .is_none());
    engine.scheduler.submit(request).await.unwrap();
    response
}

async fn fixture() -> Fixture {
    let (inner, model, cfg) = observed_cpu_model_owned();
    let owner = ExecutorOwner::new(inner);
    let inner = Arc::new(owner.scope().unwrap());
    let (ready, ready_receiver) = oneshot::channel();
    let (resume, resume_receiver) = oneshot::channel();
    let executor = Arc::new(AdmissionObservation {
        closed: Mutex::new(Vec::new()),
        ..Default::default()
    });
    let implementation = AdmissionGate {
        inner,
        owner,
        observation: Arc::clone(&executor),
        ready: Mutex::new(Some(ready)),
        resume: Mutex::new(Some(resume_receiver)),
    };
    let trace = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.backend.device = Device::CPU;
    config.runtime.scheduler_trace_jsonl = Some(trace.path().join("trace.jsonl"));
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        Arc::new(TinyTokenizer::new(cfg.vocab_size)),
        Arc::new(crate::registry::GreedySampler),
        ExecutorOwner::new(Box::new(implementation)),
    )
    .unwrap();
    let mut request = InferenceRequest::new("original admission", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    let response = install_outer(&engine, request.clone(), vec![TokenId(1), TokenId(2)]).await;
    Fixture {
        engine,
        executor,
        model,
        request,
        response,
        ready: Some(ready_receiver),
        resume: Some(resume),
        trace,
        layers: cfg.num_layers,
    }
}

async fn start_round(
    fixture: &mut Fixture,
) -> (
    tokio::task::JoinHandle<Result<(usize, Vec<ExecutorPrefillMaintenanceDeferral>)>>,
    ExecutorSequence,
) {
    let inner = Arc::clone(
        &fixture
            .engine
            .active_inner()
            .expect("engine execution is open"),
    );
    let task = tokio::spawn(async move { inner.prepare_dynamic_admission_round(1, None).await });
    let identity = tokio::time::timeout(Duration::from_secs(5), fixture.ready.take().unwrap())
        .await
        .expect("admission must reach its explicit gate")
        .unwrap();
    // All preparation resources already exist, but no product result has been
    // published and neither outer nor capacity-cache lock may span the await.
    {
        assert!(
            fixture
                .engine
                .active_inner()
                .expect("engine execution is open")
                .sequences
                .index_is_unlocked(),
            "outer index must be available during executor admission"
        );
        let sequences = &fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let state_handle = sequences.get(&fixture.request.id).unwrap();
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        assert!(state.pending_token_admission.is_some());
        assert!(state.execution_frontier.is_none());
        assert!(state.request_slot.as_ref().unwrap().armed);
    }
    assert!(fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .dynamic_admission_availability
        .try_lock()
        .is_some());
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    assert!(fixture.model.lock().releases.is_empty());
    assert!(identity.observed_end().is_none());
    (task, identity)
}

fn slot_events(fixture: &Fixture, action: ResourceAction) -> usize {
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .scheduler_trace_jsonl
        .as_ref()
        .unwrap()
        .flush()
        .unwrap();
    std::fs::read_to_string(fixture.trace.path().join("trace.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<FerrumProfileEvent>(line).unwrap())
        .filter_map(|event| event.resource)
        .filter(|resource| resource.resource_kind == "request_slot" && resource.action == action)
        .count()
}

async fn finish_round(
    fixture: &mut Fixture,
    task: tokio::task::JoinHandle<Result<(usize, Vec<ExecutorPrefillMaintenanceDeferral>)>>,
    identity: &ExecutorSequence,
) {
    fixture.resume.take().unwrap().send(()).unwrap();
    let (admitted, maintenance) = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("cancelled admission must finish without another scheduler tick")
        .unwrap()
        .unwrap();
    assert_eq!(admitted, 0);
    assert!(maintenance.is_empty());
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .drain_execution_closes()
        .unwrap();
    assert_eq!(
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
    assert_eq!(identity.observed_end().unwrap().sequence(), identity);
    assert_eq!(*fixture.executor.closed.lock(), vec![identity.clone()]);
    let model = fixture.model.lock();
    assert!(model.model.kv_caches.is_empty());
    assert_eq!(model.releases, vec![vec![0; fixture.layers]]);
}

#[tokio::test]
async fn token_admission_cancelled_ticket_closes_original_outer() {
    let mut fixture = fixture().await;
    let (task, identity) = start_round(&mut fixture).await;
    assert!(fixture
        .engine
        .scheduler
        .cancel_request(fixture.request.id.clone())
        .unwrap());
    // The receiver remains open and the exact original outer still owns its
    // request slot: client-disconnect cleanup cannot hide an orphan here.
    assert!(fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&fixture.request.id)
        .is_some());
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 0);
    finish_round(&mut fixture, task, &identity).await;
    assert!(!fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&fixture.request.id)
        .is_some());
    assert!(fixture
        .engine
        .scheduler
        .trace_phase(&fixture.request.id)
        .is_none());
    assert!(fixture.response.try_recv().unwrap().is_err());
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 1);
    assert_eq!(slot_events(&fixture, ResourceAction::RequestClose), 1);
    assert_eq!(
        fixture
            .engine
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
}

#[tokio::test]
async fn token_admission_replacement_keeps_new_outer() {
    let mut fixture = fixture().await;
    let (task, identity) = start_round(&mut fixture).await;
    assert!(fixture
        .engine
        .scheduler
        .cancel_request(fixture.request.id.clone())
        .unwrap());
    let mut old = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .take(&fixture.request.id)
        .unwrap();
    assert!(old.execution_frontier.is_none());
    assert!(old.pending_token_admission.is_some());
    // Model preparation is still owned by the executor envelope. Simulate the
    // original outer's completed cancellation before publishing a new owner.
    old.request_slot.take().unwrap().close(
        &fixture
            .engine
            .active_inner()
            .expect("engine execution is open"),
    );
    drop(old);
    let mut replacement = fixture.request.clone();
    replacement.prompt = "replacement with distinct input".into();
    let replacement_prompt = vec![TokenId(3), TokenId(4), TokenId(5)];
    let mut replacement_response = install_outer(
        &fixture.engine,
        replacement.clone(),
        replacement_prompt.clone(),
    )
    .await;
    finish_round(&mut fixture, task, &identity).await;
    {
        let sequences = &fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let state_handle = sequences
            .get(&fixture.request.id)
            .expect("replacement must stay present");
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        assert_eq!(state.original_request.prompt, replacement.prompt);
        assert_eq!(state.input_tokens, replacement_prompt);
        assert!(state.generated_tokens.is_empty());
        assert!(state.pending_token_admission.is_none());
        assert!(state.execution_frontier.is_none());
        assert!(state.execution_close.is_none());
        assert!(state.request_slot.as_ref().unwrap().armed);
    }
    assert_eq!(
        fixture.engine.scheduler.trace_phase(&fixture.request.id),
        Some(RequestPhase::Waiting)
    );
    assert_eq!(fixture.engine.scheduler.waiting_count(), 1);
    assert!(matches!(
        replacement_response.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 1);
    assert_eq!(slot_events(&fixture, ResourceAction::RequestClose), 1);
    assert_eq!(
        fixture
            .engine
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );

    // The replacement remains usable, then retires its own distinct actual
    // cache and slot. No stale result may authorize or cancel this admission.
    let (admitted, maintenance) = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .prepare_dynamic_admission_round(1, None)
        .await
        .unwrap();
    assert_eq!(admitted, 1);
    assert!(maintenance.is_empty());
    let replacement_identity = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&fixture.request.id)
        .unwrap()
        .lock()
        .as_ref()
        .unwrap()
        .execution_frontier
        .as_ref()
        .unwrap()
        .sequence()
        .clone();
    assert_ne!(replacement_identity, identity);
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_with_error(
            &fixture.request.id,
            FerrumError::request_validation("replacement test cleanup"),
        )
        .await
        .unwrap();
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .drain_execution_closes()
        .unwrap();
    assert!(replacement_response.try_recv().unwrap().is_err());
    assert_eq!(
        *fixture.executor.closed.lock(),
        vec![identity, replacement_identity.clone()]
    );
    assert!(replacement_identity.observed_end().is_some());
    assert!(fixture.model.lock().model.kv_caches.is_empty());
    assert_eq!(
        fixture.model.lock().releases,
        vec![vec![0; fixture.layers]; 2]
    );
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 2);
    assert_eq!(slot_events(&fixture, ResourceAction::RequestClose), 2);
    assert_eq!(
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
}

mod faulted_round;
