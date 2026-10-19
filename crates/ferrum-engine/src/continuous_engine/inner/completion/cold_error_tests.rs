//! Client terminal-error delivery must survive cancellation independently of
//! the real CPU model's close owner and the bounded stream's backpressure.

use super::*;
use crate::continuous_engine::token_completion_tests::{
    observed_cpu_model_owned, ModelObservation,
};
use ferrum_execution::model_executor::PrefillChunk;
use ferrum_execution::token_execution::{
    ExecutionBatchOutcome, ExecutionProduct, ExecutionWork, ExecutorSequence, TokenWork,
};
use ferrum_models::test_support::{tiny_llama_model, TinyTokenizer};
use ferrum_models::DecoderOnlyLLM;

struct Fixture {
    engine: ContinuousBatchEngine,
    model: Arc<Mutex<ModelObservation>>,
    identity: ExecutorSequence,
    request_id: RequestId,
    response: tokio::sync::oneshot::Receiver<Result<InferenceOutput>>,
    stream: OutputReceiver,
    trace: tempfile::TempDir,
    layers: usize,
}

async fn fixture() -> Fixture {
    let (executor, model, cfg) = observed_cpu_model_owned();
    let trace = tempfile::tempdir().unwrap();
    let mut config = EngineConfig::default();
    config.backend.device = Device::CPU;
    config.runtime.scheduler_trace_jsonl = Some(trace.path().join("trace.jsonl"));
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer = Arc::new(TinyTokenizer::new(cfg.vocab_size));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        ExecutorOwner::new(executor),
    )
    .unwrap();
    let executor = Arc::clone(&engine.active_inner().unwrap().model_executor);
    let prompt = [TokenId(1), TokenId(2)];
    let mut request = InferenceRequest::new("known terminal failure", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.into(),
        serde_json::json!(prompt.len()),
    );
    let request_id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    assert_eq!(
        scheduler
            .next_batch(ferrum_interfaces::BatchHint::simple(1))
            .await
            .unwrap()
            .requests
            .len(),
        1
    );
    let TokenAdmissionOutcome::Admitted(frontier) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&request_id, &prompt, cfg.max_seq_len),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("real CPU admission");
    };
    let policy = LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, prompt.len(), prompt.len()).unwrap()),
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("real CPU prefill");
    };
    let completion = batch.into_completions().pop().unwrap();
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        completion.product()
    else {
        panic!("actual full-vocabulary output");
    };
    assert_eq!(logits.len(), cfg.vocab_size);
    assert_eq!(
        logits,
        &tiny_llama_model(&cfg).prefill("reference", &[1, 2])
    );
    let identity = completion.next_frontier().sequence().clone();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        prompt.to_vec(),
        Some(tokenizer),
        Some(cfg.vocab_size),
    );
    sequence.execution_frontier = Some(completion.next_frontier().clone());
    sequence.prefill_complete = true;
    sequence.prefill_tokens_processed = prompt.len();
    let (response_sender, response) = tokio::sync::oneshot::channel();
    let (stream_sender, stream) = engine
        .active_inner()
        .expect("engine execution is open")
        .output_channel(request_id.clone(), 1)
        .unwrap();
    stream_sender
        .send(Ok(StreamChunk {
            request_id: request_id.clone(),
            text: "already visible".into(),
            token: None,
            finish_reason: None,
            usage: None,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
            api_response: None,
            execution_evidence: None,
        }))
        .await
        .unwrap();
    sequence.response_sender = Some(response_sender);
    sequence.stream_sender = Some(stream_sender);
    let mut slot = RequestSlotLease::open(
        &engine.active_inner().expect("engine execution is open"),
        request_id.clone(),
    );
    slot.admit(&engine.active_inner().expect("engine execution is open"));
    sequence.request_slot = Some(slot);
    engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .insert(request_id.clone(), sequence);
    Fixture {
        engine,
        model,
        identity,
        request_id,
        response,
        stream,
        trace,
        layers: cfg.num_layers,
    }
}

fn slot_releases(fixture: &Fixture) -> usize {
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
        .filter(|resource| {
            resource.resource_kind == "request_slot" && resource.action == ResourceAction::Release
        })
        .count()
}

async fn cancelled_error_publication(close_before_caller_drop: bool) {
    let mut fixture = fixture().await;
    let error = FerrumError::backend("original terminal execution failure");
    let original_inner = fixture.engine.active_inner().unwrap();
    let mut completion =
        Box::pin(original_inner.complete_request_with_error(&fixture.request_id, error.clone()));
    // Current-thread execution guarantees the newly spawned actual close has
    // not run before this poll returns Pending. No forged Held/Ended authority.
    let waker = futures::task::noop_waker();
    assert!(matches!(
        completion.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(!fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&fixture.request_id)
        .is_some());
    assert_eq!(
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        1
    );
    assert!(fixture.identity.observed_end().is_none());
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    assert_eq!(slot_releases(&fixture), 0);
    assert_eq!(
        fixture
            .response
            .try_recv()
            .unwrap()
            .unwrap_err()
            .to_string(),
        error.to_string()
    );

    if close_before_caller_drop {
        // Leave the channel full while the independent close actually ends.
        // Terminal stream delivery still cannot complete under backpressure.
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .join_observers()
            .await;
        assert!(fixture.identity.observed_end().is_some());
        assert_eq!(slot_releases(&fixture), 0);
    }
    drop(completion);
    drop(original_inner);
    assert_eq!(
        fixture.stream.try_recv().unwrap().unwrap().chunk().text,
        "already visible"
    );
    let terminal = tokio::time::timeout(Duration::from_secs(3), fixture.stream.recv())
        .await
        .expect("terminal publication must outlive caller")
        .expect("terminal error must not become a closed channel")
        .unwrap_err();
    assert_eq!(terminal.to_string(), error.to_string());
    assert!(fixture.stream.recv().await.is_none());

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
        fixture.identity.observed_end().unwrap().sequence(),
        &fixture.identity
    );
    assert_eq!(
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
    {
        let model = fixture.model.lock();
        assert!(model.model.kv_caches.is_empty());
        assert_eq!(model.releases, vec![vec![2; fixture.layers]]);
    }
    assert_eq!(slot_releases(&fixture), 1);
    assert_eq!(
        fixture
            .engine
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_with_error(&fixture.request_id, error)
        .await
        .unwrap();
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .drain_execution_closes()
        .unwrap();
    assert_eq!(slot_releases(&fixture), 1);
    assert_eq!(fixture.model.lock().releases.len(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn completion_known_error_survives_drop_before_actual_close() {
    cancelled_error_publication(false).await;
}

#[tokio::test(flavor = "current_thread")]
async fn completion_known_error_survives_drop_with_backpressured_stream() {
    cancelled_error_publication(true).await;
}
