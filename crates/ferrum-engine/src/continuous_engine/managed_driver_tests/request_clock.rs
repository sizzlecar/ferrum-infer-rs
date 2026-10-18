//! Real CPU numerical execution with the request blocked inside tokenizer.encode.
//! Instant ordering, not a sleep threshold, distinguishes ingress from construction.
use super::*;
use ferrum_interfaces::tokenizer::TokenizerInfo;
use ferrum_models::test_support::TinyTokenizer;
use std::sync::{Condvar, Mutex as StdMutex};

struct EncodeGate {
    armed: AtomicBool,
    reached: Mutex<Option<tokio::sync::oneshot::Sender<Instant>>>,
    released: StdMutex<bool>,
    changed: Condvar,
}
impl EncodeGate {
    fn enter(&self) {
        if !self.armed.swap(false, Ordering::SeqCst) {
            return;
        }
        self.reached
            .lock()
            .take()
            .unwrap()
            .send(Instant::now())
            .unwrap();
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
    }
    fn release(&self) -> Instant {
        let mut released = self.released.lock().unwrap();
        let at = Instant::now();
        *released = true;
        self.changed.notify_all();
        at
    }
}
struct ReleaseEncode(Arc<EncodeGate>);
impl Drop for ReleaseEncode {
    fn drop(&mut self) {
        self.0.release();
    }
}
struct GatedTokenizer {
    inner: TinyTokenizer,
    gate: Arc<EncodeGate>,
}
impl Tokenizer for GatedTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.gate.enter();
        self.inner.encode(text, add_special)
    }
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        self.inner.decode(tokens, skip_special)
    }
    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String> {
        self.inner.decode_incremental(prev, next)
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn request_clock_cpu_infer_includes_tokenization_and_preserves_ingress() {
    for supplied in [false, true] {
        gated_request(false, supplied).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn request_clock_cpu_infer_stream_includes_tokenization_and_preserves_ingress() {
    for supplied in [false, true] {
        gated_request(true, supplied).await;
    }
}

async fn gated_request(streaming: bool, supplied: bool) {
    let (reached, entered) = tokio::sync::oneshot::channel();
    let gate = Arc::new(EncodeGate {
        armed: AtomicBool::new(false),
        reached: Mutex::new(Some(reached)),
        released: StdMutex::new(false),
        changed: Condvar::new(),
    });
    let mut fixture = fixture::Fixture::with_tokenizer(
        |config| {
            config.runtime.profile_detail = ObservabilityProfileDetail::Latency;
        },
        |vocab| {
            Arc::new(GatedTokenizer {
                inner: TinyTokenizer::new(vocab),
                gate: gate.clone(),
            })
        },
    );
    let _release_on_failure = ReleaseEncode(gate.clone());
    let mut request = fixture.request();
    request.evidence_request.capture_engine_token_timing = true;
    let supplied_anchor =
        supplied.then(|| RequestIngressAnchor::capture(RequestIngressBoundary::HttpHandler));
    if let Some(anchor) = supplied_anchor {
        request = request.with_ingress(anchor);
    }
    // A remote wall timestamp cannot become this process's monotonic anchor.
    request.created_at =
        chrono::DateTime::from_timestamp(if supplied { 1 } else { 4_102_444_800 }, 0).unwrap();
    let id = request.id.clone();
    let ingress_engine = ContinuousBatchEngine {
        inner: Arc::clone(&fixture.engine.inner),
        driver: Arc::clone(&fixture.engine.driver),
    };
    gate.armed.store(true, Ordering::SeqCst);
    let inference = tokio::spawn(async move {
        if streaming {
            let stream = ingress_engine.infer_stream(request).await.unwrap();
            let chunks = stream
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .unwrap();
            let terminal = chunks.last().unwrap().chunk();
            assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
            assert_eq!(
                chunks
                    .iter()
                    .filter(|chunk| chunk.chunk().finish_reason.is_some())
                    .count(),
                1
            );
            let evidence = terminal.execution_evidence.as_ref().unwrap().clone();
            assert_eq!(terminal.usage.as_ref().unwrap().completion_tokens, 4);
            (evidence, None)
        } else {
            let output = ingress_engine.infer(request).await.unwrap();
            assert_eq!(output.response().tokens.len(), 4);
            assert_eq!(output.response().finish_reason, FinishReason::Length);
            (
                output
                    .response()
                    .execution_evidence
                    .as_ref()
                    .unwrap()
                    .clone(),
                Some(output.response().latency_ms),
            )
        }
    });
    let encode_entered = tokio::time::timeout(Duration::from_secs(3), entered)
        .await
        .unwrap()
        .unwrap();
    assert!(fixture.engine.inner.sequences.get(&id).is_none());
    assert_eq!(fixture.executor.admissions.load(Ordering::SeqCst), 0);
    let encode_released = gate.release();
    // The existing fixture gate is after actual CPU logits and KV verification,
    // before the first sample is committed, so the original state stays live.
    tokio::time::timeout(Duration::from_secs(3), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    let anchor = {
        let handle = fixture.engine.inner.sequences.get(&id).unwrap();
        let state = handle.lock();
        let sequence = state.as_ref().unwrap();
        let anchor = sequence.original_request.ingress.unwrap();
        assert_eq!(sequence.start_time, anchor.instant());
        assert!(
            anchor.instant() <= encode_entered,
            "tokenization must not precede the request anchor"
        );
        if let Some(original) = supplied_anchor {
            assert_eq!(anchor, original);
        }
        assert_eq!(
            anchor.boundary(),
            if supplied {
                RequestIngressBoundary::HttpHandler
            } else {
                RequestIngressBoundary::EngineApi
            }
        );
        assert!(sequence.generated_tokens.is_empty());
        anchor
    };
    fixture.release.send(()).unwrap();
    let (evidence, latency_ms) = tokio::time::timeout(Duration::from_secs(3), inference)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(evidence.output_token_ids.len(), 4);
    let timing = evidence.engine_token_timing.unwrap();
    timing.validate(4).unwrap();
    assert_eq!(timing.request_start_boundary, Some(anchor.boundary()));
    let correlation = anchor.wall_correlation().unwrap();
    assert_eq!(timing.wall_anchor_unix_nanos, correlation.unix_nanos);
    assert_eq!(
        timing.wall_anchor_max_error_nanos,
        correlation.max_error_nanos
    );
    let tokenize_end = anchor.elapsed_at(encode_released);
    assert!(u128::from(timing.ttft_nanos().unwrap()) >= tokenize_end.as_nanos());
    if let Some(latency_ms) = latency_ms {
        assert!(u128::from(latency_ms) >= tokenize_end.as_millis());
    }
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 4);
    assert_eq!(fixture.executor.admissions.load(Ordering::SeqCst), 1);
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 1);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert!(fixture.engine.inner.sequences.is_empty());
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        fixture::closed_slot_events(
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::Release
        ),
        1
    );
}

#[tokio::test]
async fn request_clock_cpu_pause_quota_failure_closes_exact_admitted_owner_without_replay() {
    // This fixture has completed and compared a genuine CPU prefill, committed
    // its first token, and retained the exact physical KV and product slot.
    let mut fixture = closure::fixture(false, false).await;
    fixture
        .engine
        .inner
        .scheduler
        .mark_prefill_complete(&fixture.request_id, fixture.prompt_len);
    let handle = fixture
        .engine
        .inner
        .sequences
        .get(&fixture.request_id)
        .unwrap();
    let session = handle
        .lock()
        .as_ref()
        .unwrap()
        .output_semantic
        .output_session()
        .clone();
    let occupied = session
        .try_reserve_semantic(
            session.config().semantic_bytes_per_session - session.semantic_usage() - 63,
        )
        .unwrap();
    let frontier = {
        let mut state = handle.lock();
        let sequence = state.as_mut().unwrap();
        assert!(sequence.closes_pending_decode_scheduling());
        sequence.pause_decode_scheduling(Instant::now(), true);
        assert!(matches!(
            sequence.output_accounting_error(),
            Some(FerrumError::ResourceExhausted { .. })
        ));
        assert!(!sequence.closes_pending_decode_scheduling());
        sequence.execution_frontier.clone().unwrap()
    };
    assert_eq!(frontier.sequence(), &fixture.identity);
    let batch = fixture
        .engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    let error = fixture
        .engine
        .inner
        .process_batch(&batch)
        .await
        .unwrap_err();
    assert!(matches!(error, FerrumError::ResourceExhausted { .. }));
    fixture.engine.inner.execution_closes.join_observers().await;
    fixture.engine.inner.output_tasks.join_current().await;
    fixture.engine.inner.drain_execution_closes().unwrap();
    let client_error = (&mut fixture.response).await.unwrap().unwrap_err();
    assert!(matches!(
        client_error,
        FerrumError::ResourceExhausted { .. }
    ));
    assert!(matches!(
        fixture.engine.inner.output_tasks.take_control_error(),
        Some(FerrumError::ResourceExhausted { .. })
    ));
    assert!(fixture
        .engine
        .inner
        .output_tasks
        .take_control_error()
        .is_none());
    assert_eq!(fixture.executor.close_call_counts(), (0, 1));
    closure::assert_closed_once(&fixture).await;
    assert!(fixture.engine.inner.sequences.is_empty());
    // Exact layer lengths at release prove no decode/replayed prefill occurred.
    assert_eq!(
        fixture.model.lock().releases,
        vec![vec![fixture.prompt_len; fixture.layers]]
    );
    drop(occupied);
    drop(session);
    fixture.engine.shutdown().await.unwrap();
}
