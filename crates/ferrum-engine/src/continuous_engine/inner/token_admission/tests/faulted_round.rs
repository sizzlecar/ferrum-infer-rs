//! An earlier real, malformed admission must still be retired when publishing
//! a later admission fails. Neither error may erase the other's physical owner.
use super::*;
use ferrum_execution::model_executor::PrefillChunk;
use ferrum_execution::token_execution::{ExecutionProduct, TokenWork};

struct AdvancedThenGatedAdmission {
    gate: Arc<AdmissionGate>,
    first_request: RequestId,
    advanced: Mutex<Option<ExecutionFrontier>>,
}

#[async_trait::async_trait]
impl ModelExecutor for AdvancedThenGatedAdmission {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.gate.info()
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.gate.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.gate.status()
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        if input.context.request_id != &self.first_request {
            return self.gate.admit_tokens(input).await;
        }
        let TokenAdmissionOutcome::Admitted(start) = self.gate.inner.admit_tokens(input).await?
        else {
            panic!("first request must obtain a real CPU admission");
        };
        let mut retained = PendingAdmission {
            executor: self.gate.as_ref(),
            frontier: Some(start),
        };
        let policy = LogitsReturnPolicy::FullLogits;
        let work = ExecutionWork::new(
            retained.frontier.as_ref().unwrap(),
            TokenWork::Prefill(PrefillChunk::new(0, 1, input.context.input_tokens.len())?),
            &policy,
        )?;
        let ExecutionBatchOutcome::Completed(completed) =
            self.gate.inner.execute_tokens(&[work]).await?
        else {
            panic!("injected malformed admission requires a successful real partial prefill");
        };
        let mut completions = completed.into_completions();
        assert_eq!(completions.len(), 1);
        let completion = completions.pop().unwrap();
        assert!(matches!(
            completion.product(),
            ExecutionProduct::Intermediate
        ));
        let advanced = completion.next_frontier().clone();
        assert_eq!(advanced.consumed_tokens(), 1);
        assert_eq!(
            advanced.sequence(),
            retained.frontier.as_ref().unwrap().sequence()
        );
        assert!(self.advanced.lock().replace(advanced.clone()).is_none());
        // This is deliberately invalid *admission*, but its consumed position
        // and cache contents were produced by the actual model token core.
        drop(retained.frontier.take());
        Ok(TokenAdmissionOutcome::Admitted(advanced))
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        self.gate.execute_tokens(work).await
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.gate.finish_tokens(sequence, reason).await
    }

    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.gate.cancel_tokens(sequence)
    }
}

#[tokio::test]
async fn token_admission_later_publish_error_retires_earlier_real_faulted_frontier() {
    let (inner, model, cfg) = observed_cpu_model();
    let mut first_request = InferenceRequest::new("invalid initial frontier", "tiny-llama-test");
    first_request.sampling_params.max_tokens = 4;
    first_request.sampling_params.temperature = 0.0;
    let mut second_request = InferenceRequest::new("disappearing outer", "tiny-llama-test");
    second_request.sampling_params.max_tokens = 4;
    second_request.sampling_params.temperature = 0.0;
    let (ready, ready_receiver) = oneshot::channel();
    let (resume, resume_receiver) = oneshot::channel();
    let gate = Arc::new(AdmissionGate {
        inner,
        ready: Mutex::new(Some(ready)),
        resume: Mutex::new(Some(resume_receiver)),
        closed: Mutex::new(Vec::new()),
    });
    let executor = Arc::new(AdvancedThenGatedAdmission {
        gate: Arc::clone(&gate),
        first_request: first_request.id.clone(),
        advanced: Mutex::new(None),
    });
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
        Arc::new(MockKvCacheManager::new(256)),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    let first_response = install_outer(
        &engine,
        first_request.clone(),
        vec![TokenId(1), TokenId(2), TokenId(3)],
    )
    .await;
    let _second_response = install_outer(
        &engine,
        second_request.clone(),
        vec![TokenId(4), TokenId(5)],
    )
    .await;
    let mut fixture = Fixture {
        engine,
        executor: gate,
        model,
        request: first_request,
        response: first_response,
        ready: Some(ready_receiver),
        resume: Some(resume),
        trace,
        layers: cfg.num_layers,
    };
    let engine_inner = Arc::clone(&fixture.engine.inner);
    let task = tokio::spawn(async move { engine_inner.prepare_dynamic_admission_round(2).await });
    let second_identity =
        tokio::time::timeout(Duration::from_secs(5), fixture.ready.take().unwrap())
            .await
            .expect("second admission must reach the explicit gate")
            .unwrap();
    let first_frontier = executor.advanced.lock().clone().unwrap();
    let first_identity = first_frontier.sequence().clone();
    {
        let sequences = fixture.engine.inner.sequences.try_write().unwrap();
        let first = &sequences[&fixture.request.id];
        assert_eq!(first.execution_frontier.as_ref(), Some(&first_frontier));
        assert!(first.request_slot.as_ref().unwrap().armed);
        assert!(first.pending_token_admission.is_none());
        assert!(sequences[&second_request.id]
            .pending_token_admission
            .is_some());
    }
    assert!(fixture
        .engine
        .inner
        .dynamic_admission_availability
        .try_lock()
        .is_some());
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_phase(&fixture.request.id),
        Some(RequestPhase::AdmissionFailed)
    );
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_phase(&second_request.id),
        Some(RequestPhase::Waiting)
    );
    assert!(matches!(
        fixture.response.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    {
        let observed = fixture.model.lock();
        let mut lengths: Vec<_> = observed
            .model
            .kv_caches
            .values()
            .map(|layers| layers.iter().map(|layer| layer.len).collect::<Vec<_>>())
            .collect();
        lengths.sort();
        assert_eq!(
            lengths,
            vec![vec![0; fixture.layers], vec![1; fixture.layers]]
        );
        assert!(observed.releases.is_empty());
    }
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 0);

    // Deliberately retain the scheduler ticket, so apply succeeds and publish
    // fails. The fresh result still belongs to its independent executor owner.
    let mut removed = fixture
        .engine
        .inner
        .sequences
        .write()
        .remove(&second_request.id)
        .unwrap();
    assert!(removed.execution_frontier.is_none());
    removed
        .request_slot
        .take()
        .unwrap()
        .close(&fixture.engine.inner);
    drop(removed);
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_phase(&second_request.id),
        Some(RequestPhase::Waiting)
    );
    fixture.resume.take().unwrap().send(()).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("round failure must still drain the earlier fault")
        .unwrap();
    assert!(
        result.is_err(),
        "second publication must report its missing outer"
    );
    fixture.engine.inner.execution_closes.join_observers().await;
    fixture.engine.inner.drain_execution_closes().unwrap();

    assert!(fixture.response.try_recv().unwrap().is_err());
    assert!(fixture.engine.inner.sequences.read().is_empty());
    assert!(fixture
        .engine
        .inner
        .scheduler
        .trace_phase(&fixture.request.id)
        .is_none());
    assert!(fixture
        .engine
        .inner
        .scheduler
        .trace_phase(&second_request.id)
        .is_none());
    for identity in [&first_identity, &second_identity] {
        assert_eq!(identity.observed_end().unwrap().sequence(), identity);
        assert_eq!(
            fixture
                .executor
                .closed
                .lock()
                .iter()
                .filter(|closed| *closed == identity)
                .count(),
            1
        );
    }
    assert_eq!(fixture.executor.closed.lock().len(), 2);
    {
        let observed = fixture.model.lock();
        assert!(observed.model.kv_caches.is_empty());
        let mut released = observed.releases.clone();
        released.sort();
        assert_eq!(
            released,
            vec![vec![0; fixture.layers], vec![1; fixture.layers]]
        );
    }
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 2);
    assert_eq!(slot_events(&fixture, ResourceAction::RequestClose), 2);
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        2
    );
}
