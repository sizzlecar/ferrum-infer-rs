//! Only text decoding is faulted. Admission, full-vocabulary CPU prefill,
//! model KV, close receipts and the product slot use the existing real fixture.
use super::*;
use crate::continuous_engine::token_completion_tests as actual;
use ferrum_execution::token_execution::ExecutionCloseDisposition;
use ferrum_models::test_support::TinyTokenizer;
use std::sync::atomic::AtomicUsize;

const DECODE_ERROR: &str = "injected terminal tokenizer failure";

#[derive(Default)]
struct DecodeFault {
    remaining: AtomicUsize,
    failures: AtomicUsize,
}

struct FaultTokenizer {
    inner: TinyTokenizer,
    fault: Arc<DecodeFault>,
}

impl Tokenizer for FaultTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.inner.encode(text, add_special)
    }
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        if skip_special
            && !tokens.is_empty()
            && self
                .fault
                .remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
        {
            self.fault.failures.fetch_add(1, Ordering::SeqCst);
            return Err(FerrumError::tokenizer(DECODE_ERROR));
        }
        self.inner.decode(tokens, skip_special)
    }
    fn decode_incremental(&self, previous: &[TokenId], next: TokenId) -> Result<String> {
        self.inner.decode_incremental(previous, next)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        self.inner.special_tokens()
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.inner.token_id(text)
    }
    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.inner.token_text(token)
    }
    fn token_bytes(&self, token: TokenId) -> Option<Vec<u8>> {
        self.inner.token_bytes(token)
    }
    fn info(&self) -> TokenizerInfo {
        self.inner.info()
    }
}

async fn fixture() -> (actual::Fixture, Arc<DecodeFault>) {
    let fault = Arc::new(DecodeFault::default());
    let tokenizer_fault = Arc::clone(&fault);
    let fixture = actual::fixture_with_tokenizer(false, false, move |vocab| {
        Arc::new(FaultTokenizer {
            inner: TinyTokenizer::new(vocab),
            fault: tokenizer_fault,
        })
    })
    .await;
    // The original fixture has already compared every real prefill logit to
    // its independent CPU reference. Faults start only after that work.
    assert_eq!(fault.failures.load(Ordering::SeqCst), 0);
    assert!(fixture.identity.observed_end().is_none());
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    (fixture, fault)
}

fn assert_tokenizer_error(error: &FerrumError) {
    assert!(matches!(error, FerrumError::Tokenizer { message } if message == DECODE_ERROR));
}

async fn assert_one_stream_error(fixture: &mut actual::Fixture, expected: &FerrumError) {
    let errors = tokio::time::timeout(Duration::from_secs(3), async {
        let mut errors = Vec::new();
        while let Some(item) = fixture.stream.recv().await {
            match item {
                Ok(chunk) => panic!(
                    "terminal decode failure cannot publish success: {:?}",
                    chunk.chunk()
                ),
                Err(error) => errors.push(error),
            }
        }
        errors
    })
    .await
    .expect("terminal error delivery must finish without a client stall");
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].to_string(), expected.to_string());
}

async fn decode_failure_closes_actual_cpu(once: bool) {
    let (mut fixture, fault) = fixture().await;
    fault
        .remaining
        .store(if once { 1 } else { usize::MAX }, Ordering::SeqCst);
    let error = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_for_test(&fixture.request_id, FinishReason::Length)
        .await
        .unwrap_err();
    assert_tokenizer_error(&error);
    assert_tokenizer_error(&fixture.response.try_recv().unwrap().unwrap_err());
    assert_tokenizer_error(
        &fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .unwrap(),
    );
    assert_one_stream_error(&mut fixture, &error).await;
    assert!(fault.failures.load(Ordering::SeqCst) >= 1);
    if once {
        assert_eq!(fault.failures.load(Ordering::SeqCst), 1);
        assert_eq!(fault.remaining.load(Ordering::SeqCst), 0);
        // Recovery of the decoder cannot erase the earlier failed flush.
        assert!(fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .tokenizer
            .decode(&[TokenId(3)], true)
            .is_ok());
    }
    assert_eq!(fixture.executor.close_call_counts(), (0, 1));
    actual::assert_closed_once(&fixture).await;
}

#[tokio::test]
async fn completion_terminal_decode_error_closes_real_cpu_and_errors_both_clients() {
    decode_failure_closes_actual_cpu(false).await;
}

#[tokio::test]
async fn completion_terminal_decode_error_once_survives_later_decoder_recovery() {
    decode_failure_closes_actual_cpu(true).await;
}

#[tokio::test]
async fn completion_terminal_decode_error_keeps_failed_close_owner_and_control_error() {
    let (mut fixture, fault) = fixture().await;
    fault.remaining.store(usize::MAX, Ordering::SeqCst);
    fixture
        .executor
        .fail_cancellation
        .store(true, Ordering::SeqCst);
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_detached(&fixture.request_id, FinishReason::Length, None)
        .unwrap();
    // This current-thread test has not yielded. The synchronous handoff must
    // publish the first client/control failure before the close task runs.
    let client_error = fixture.response.try_recv().unwrap().unwrap_err();
    assert_tokenizer_error(&client_error);
    assert_tokenizer_error(
        &fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .unwrap(),
    );
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .join_current()
        .await;
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    let close_error = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .take_control_error()
        .unwrap();
    assert!(matches!(&close_error, FerrumError::Backend { message }
        if message == "injected cancellation without terminal proof"));
    assert_one_stream_error(&mut fixture, &client_error).await;
    assert_eq!(fixture.executor.close_call_counts(), (0, 1));
    assert!(fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&fixture.request_id)
        .is_none());
    assert!(fixture.identity.observed_end().is_none());
    assert_eq!(
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        1
    );
    assert_eq!(actual::slot_events(&fixture, ResourceAction::Release), 0);
    assert_eq!(
        actual::slot_events(&fixture, ResourceAction::RequestClose),
        0
    );
    assert_eq!(
        fixture.engine.scheduler.confirmed_capacity_release_epoch(),
        0
    );
    {
        let model = fixture.model.lock();
        assert!(model.releases.is_empty());
        assert_eq!(model.model.kv_caches.len(), 1);
        assert!(model
            .model
            .kv_caches
            .values()
            .next()
            .unwrap()
            .iter()
            .all(|layer| layer.len == fixture.prompt_len));
    }
    // Only a real later close of that same executor incarnation authorizes
    // the original registry to release its retained slot and capacity.
    fixture
        .executor
        .fail_cancellation
        .store(false, Ordering::SeqCst);
    let closed = fixture.executor.cancel_tokens(&fixture.identity).unwrap();
    assert!(closed.error.is_none());
    let ExecutionCloseDisposition::Ended(ended) = closed.disposition else {
        panic!("actual CPU close must prove terminal");
    };
    assert_eq!(ended.sequence(), &fixture.identity);
    actual::assert_closed_once(&fixture).await;
    assert_eq!(fixture.executor.close_call_counts(), (0, 2));
}

#[tokio::test]
async fn completion_terminal_decode_error_preserves_prior_client_error() {
    let (mut fixture, fault) = fixture().await;
    fault.remaining.store(usize::MAX, Ordering::SeqCst);
    let original = FerrumError::request_validation("original explicit client failure");
    let control = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_with_error(&fixture.request_id, original.clone())
        .await
        .unwrap_err();
    assert_tokenizer_error(&control);
    let client = fixture.response.try_recv().unwrap().unwrap_err();
    assert!(matches!(&client, FerrumError::RequestValidation { .. }));
    assert_eq!(client.to_string(), original.to_string());
    assert_tokenizer_error(
        &fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .unwrap(),
    );
    assert_one_stream_error(&mut fixture, &original).await;
    assert!(fault.failures.load(Ordering::SeqCst) >= 1);
    assert_eq!(fixture.executor.close_call_counts(), (0, 1));
    actual::assert_closed_once(&fixture).await;
}

async fn pending_span_terminal_provenance(decode_fails: bool) {
    let (mut fixture, fault) = fixture().await;
    let handle = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&fixture.request_id)
        .unwrap();
    let (sender, text, token) = {
        let state = handle.lock();
        let state = state.as_ref().unwrap();
        (
            state.stream_sender.as_ref().unwrap().clone(),
            fixture
                .engine
                .active_inner()
                .expect("engine execution is open")
                .tokenizer
                .decode(&state.generated_tokens, true)
                .unwrap(),
            *state.generated_tokens.last().unwrap(),
        )
    };
    assert!(!text.is_empty());
    let session = sender.output_session().clone();
    let held = session
        .reserve_queued(session.config().queued_bytes_total)
        .await
        .unwrap();
    let range = 0..text.len() + usize::from(!decode_fails);
    // The positive arm uses the actual previously generated text. The negative
    // arm deliberately violates its byte range while keeping decoding valid.
    sender
        .publish_text(
            &fixture
                .engine
                .active_inner()
                .expect("engine execution is open"),
            StreamChunk {
                request_id: fixture.request_id.clone(),
                text,
                token: Some(token),
                finish_reason: None,
                usage: None,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: None,
                execution_evidence: None,
            },
            range.clone(),
            None,
        )
        .unwrap();
    assert!(matches!(
        fixture.stream.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    if decode_fails {
        fault.remaining.store(usize::MAX, Ordering::SeqCst);
    }
    let original =
        FerrumError::request_validation("prior client failure must not conceal provenance");
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_detached(
            &fixture.request_id,
            FinishReason::Error,
            Some(original.clone()),
        )
        .unwrap();
    assert_eq!(
        fixture
            .response
            .try_recv()
            .unwrap()
            .unwrap_err()
            .to_string(),
        original.to_string()
    );
    let projection = sender.completed_text(range).unwrap_err();
    if decode_fails {
        assert_tokenizer_error(&projection);
    } else {
        assert!(matches!(&projection, FerrumError::Internal { message }
            if message == "pending output span disagrees with completed text"));
    }
    if decode_fails {
        // The driver may consume the first decode error while the reader is
        // still blocked. Later delivery must not report that same failure again.
        let recorded = fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .unwrap();
        assert_eq!(recorded.to_string(), projection.to_string());
    } else {
        assert!(fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .is_none());
    }
    drop(held);
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .join_current()
        .await;
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
        sender.failure().unwrap().to_string(),
        projection.to_string()
    );
    if !decode_fails {
        assert_eq!(
            fixture
                .engine
                .active_inner()
                .expect("engine execution is open")
                .output_tasks
                .take_control_error()
                .unwrap()
                .to_string(),
            projection.to_string()
        );
    }
    assert!(fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .take_control_error()
        .is_none());
    drop(sender);
    assert_one_stream_error(&mut fixture, &original).await;
    actual::assert_closed_once(&fixture).await;
}

#[tokio::test]
async fn completion_pending_span_preserves_actual_decode_failure() {
    pending_span_terminal_provenance(true).await;
}

#[tokio::test]
async fn completion_pending_available_text_rejects_bad_range_despite_prior_error() {
    pending_span_terminal_provenance(false).await;
}
