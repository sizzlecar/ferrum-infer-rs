//! A real completed CPU forward enters a deliberately blocked tokenizer. Only
//! that request's product state may be locked; this is not parallel model work.
use super::*;
use ferrum_interfaces::tokenizer::TokenizerInfo;
use ferrum_models::test_support::TinyTokenizer;
use std::sync::{Condvar, Mutex as StdMutex};

struct DecoderGate {
    armed: AtomicBool,
    reached: Mutex<Option<tokio::sync::oneshot::Sender<Vec<TokenId>>>>,
    released: StdMutex<bool>,
    changed: Condvar,
}

impl DecoderGate {
    fn enter(&self, tokens: &[TokenId]) {
        if tokens.len() != 1 || !self.armed.swap(false, Ordering::SeqCst) {
            return;
        }
        self.reached
            .lock()
            .take()
            .unwrap()
            .send(tokens.to_vec())
            .unwrap();
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

// A failed progress assertion must also release the blocked runtime worker.
struct ReleaseDecoder(Arc<DecoderGate>);
impl Drop for ReleaseDecoder {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct GatedTokenizer {
    inner: TinyTokenizer,
    gate: Arc<DecoderGate>,
}

impl Tokenizer for GatedTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.inner.encode(text, add_special)
    }
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        if skip_special {
            self.gate.enter(tokens);
        }
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
async fn sequence_decoder_block_does_not_hold_publication_or_peer_close() {
    let (reached, entered) = tokio::sync::oneshot::channel();
    let gate = Arc::new(DecoderGate {
        armed: AtomicBool::new(false),
        reached: Mutex::new(Some(reached)),
        released: StdMutex::new(false),
        changed: Condvar::new(),
    });
    let fixture = fixture::Fixture::with_tokenizer(
        |config| {
            config.output.engine_queue_items = 1;
            config.scheduler.max_running_requests = 3;
            config.batching.max_batch_size = 2;
        },
        |vocab| {
            Arc::new(GatedTokenizer {
                inner: TinyTokenizer::new(vocab),
                gate: gate.clone(),
            })
        },
    );
    let release_on_failure = ReleaseDecoder(gate.clone());
    fixture.executor.remove_execution_gate();

    let peer_request = fixture.request();
    let peer_id = peer_request.id.clone();
    let mut peer = fixture.engine.infer_stream(peer_request).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), fixture.executor.wait_for_outputs(2))
        .await
        .unwrap();

    // The first request is paused by its actual full output queue. Arm only
    // after the next request has completed a genuine numerical prefill, so
    // tokenizer setup/vocabulary inspection cannot consume the decoder gate.
    let gate_after_forward = gate.clone();
    fixture.executor.after_next_prefill(move || {
        gate_after_forward.armed.store(true, Ordering::SeqCst);
    });
    let blocked_request = fixture.request();
    let blocked_id = blocked_request.id.clone();
    let blocked = fixture.engine.infer_stream(blocked_request).await.unwrap();
    let candidate = tokio::time::timeout(Duration::from_secs(3), entered)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 3);
    assert_eq!(fixture.executor.admissions.load(Ordering::SeqCst), 2);

    // This calls the real ingress publication path while the driver holds the
    // blocked request's mutable state, after releasing its planning lock.
    let later_request = fixture.request();
    let later_id = later_request.id.clone();
    let publication_engine = Arc::clone(&fixture.engine);
    let publication =
        tokio::spawn(async move { publication_engine.infer_stream(later_request).await });
    // Poll only a JoinHandle here. A regression that synchronously blocks a
    // worker on the global lock must still let this task time out and release
    // the decoder through ReleaseDecoder before runtime shutdown joins it.
    let later = tokio::time::timeout(Duration::from_secs(3), publication)
        .await
        .expect("a decoder must not hold the global publication index")
        .unwrap()
        .unwrap();
    assert!(fixture
        .engine
        .active_inner()
        .unwrap()
        .sequences
        .get(&later_id)
        .is_some());
    assert!(fixture
        .engine
        .active_inner()
        .unwrap()
        .sequences
        .get(&blocked_id)
        .is_some());
    assert_eq!(
        fixture.executor.admissions.load(Ordering::SeqCst),
        2,
        "publication does not claim a second concurrent model forward"
    );

    let first = tokio::time::timeout(Duration::from_secs(3), peer.next())
        .await
        .expect("already queued output remains consumable")
        .unwrap()
        .unwrap();
    assert!(!first.chunk().text.is_empty());
    let close_inner = Arc::clone(&fixture.engine.active_inner().unwrap());
    let close_id = peer_id.clone();
    let close = tokio::spawn(async move {
        close_inner
            .complete_request_for_test(&close_id, FinishReason::Length)
            .await
    });
    let consume = tokio::spawn(async move { peer.collect::<Vec<_>>().await });
    let (closed, rest) = tokio::time::timeout(Duration::from_secs(3), async {
        tokio::join!(close, consume)
    })
    .await
    .expect("an independent original request can actually close");
    closed.unwrap().unwrap();
    let mut peer_chunks = vec![first];
    peer_chunks.extend(
        rest.unwrap()
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .unwrap(),
    );
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.executor.observed.lock().model.kv_caches.len(), 1);
    assert!(fixture
        .engine
        .active_inner()
        .unwrap()
        .sequences
        .get(&peer_id)
        .is_none());
    assert_eq!(
        peer_chunks
            .iter()
            .filter(|chunk| chunk.chunk().finish_reason.is_some())
            .count(),
        1
    );
    assert_eq!(
        peer_chunks
            .last()
            .unwrap()
            .chunk()
            .usage
            .as_ref()
            .unwrap()
            .completion_tokens,
        2
    );
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 3);

    gate.release();
    let (blocked_chunks, later_chunks) = tokio::time::timeout(Duration::from_secs(3), async {
        tokio::join!(blocked.collect::<Vec<_>>(), later.collect::<Vec<_>>())
    })
    .await
    .unwrap();
    let blocked_chunks = blocked_chunks
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let later_chunks = later_chunks
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    for chunks in [&blocked_chunks, &later_chunks] {
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| chunk.chunk().finish_reason.is_some())
                .count(),
            1
        );
        assert_eq!(
            chunks.last().unwrap().chunk().finish_reason,
            Some(FinishReason::Length)
        );
        assert_eq!(
            chunks
                .last()
                .unwrap()
                .chunk()
                .usage
                .as_ref()
                .unwrap()
                .completion_tokens,
            4
        );
    }
    let tokens = |chunks: &[OutputChunk]| {
        chunks
            .iter()
            .filter_map(|chunk| chunk.chunk().token)
            .collect::<Vec<_>>()
    };
    let text = |chunks: &[OutputChunk]| {
        chunks
            .iter()
            .map(|chunk| chunk.chunk().text.as_str())
            .collect::<String>()
    };
    assert_eq!(tokens(&blocked_chunks), tokens(&later_chunks));
    assert_eq!(tokens(&blocked_chunks).first(), candidate.first());
    assert_eq!(text(&blocked_chunks), text(&later_chunks));
    assert_eq!(
        text(&peer_chunks),
        fixture
            .engine
            .active_inner()
            .unwrap()
            .tokenizer
            .decode(&tokens(&peer_chunks), true)
            .unwrap()
    );
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 10);
    assert_eq!(fixture.executor.maximum_in_flight.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 3);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert!(fixture.original_inner.upgrade().is_none());
    assert_eq!(
        fixture::closed_slot_events(
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::Release
        ),
        3
    );
    drop((
        peer_chunks,
        blocked_chunks,
        later_chunks,
        release_on_failure,
    ));
    assert_eq!(fixture.output_budget.usage().sessions, 0);
    assert_eq!(fixture.output_budget.usage().queued_bytes, 0);
}
