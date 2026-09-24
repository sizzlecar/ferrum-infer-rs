use super::*;
use ferrum_interfaces::tokenizer::{BoundedDecodeBound, BoundedDecodeError, TokenizerInfo};
use ferrum_testkit::{MockKvCacheManager, MockModelExecutor, MockTensorFactory, MockTokenizer};
use ferrum_types::{InferenceRequest, ModelOutputProtocol, SpecialTokens};
use std::sync::{atomic::AtomicBool, Mutex};

#[derive(Default)]
struct DecodeProbe {
    action: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    fail: AtomicBool,
}

struct ProjectionTokenizer {
    base: MockTokenizer,
    probe: DecodeProbe,
}

impl ProjectionTokenizer {
    fn new() -> Self {
        Self {
            base: MockTokenizer::new(32),
            probe: DecodeProbe::default(),
        }
    }

    fn on_decode(&self, action: impl FnOnce() + Send + 'static) {
        *self.probe.action.lock().unwrap() = Some(Box::new(action));
    }

    fn piece(token: TokenId) -> &'static [u8] {
        match token.get() {
            1 => b"hello",
            2 => &[0xe4],
            3 => &[0xb8, 0xad],
            4 => b"ST",
            5 => b"OPtail",
            6 => b"b",
            7 => b"<|return|>",
            8 => b"<|call|>",
            9 => b"<eos>",
            _ => b"",
        }
    }
}

impl Tokenizer for ProjectionTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.base.encode(text, add_special)
    }

    fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
        let action = self.probe.action.lock().unwrap().take();
        if let Some(action) = action {
            action();
        }
        if self.probe.fail.load(Ordering::Relaxed) {
            return Err(FerrumError::internal("injected projection decode failure"));
        }
        // IDs 2 and 3 only form a valid scalar together. Decoding one token
        // at a time cannot reproduce the complete history's Chinese text.
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|token| {
                let piece: &[u8] = match token.get() {
                    1 => b"hello",
                    2 => &[0xe4],
                    3 => &[0xb8, 0xad],
                    4 => b"ST",
                    5 => b"OPtail",
                    6 => b"b",
                    7 => b"<|return|>",
                    8 => b"<|call|>",
                    9 => b"<eos>",
                    _ => b"",
                };
                piece.iter().copied()
            })
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn decode_incremental(&self, _previous: &[TokenId], _next: TokenId) -> Result<String> {
        panic!("stream projection must preserve full-history decode")
    }

    fn vocab_size(&self) -> usize {
        self.base.vocab_size()
    }
    fn special_tokens(&self) -> &SpecialTokens {
        self.base.special_tokens()
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        match text {
            "<|return|>" => Some(TokenId::new(7)),
            "<|call|>" => Some(TokenId::new(8)),
            _ => None,
        }
    }
    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.base.token_text(token)
    }
    fn info(&self) -> TokenizerInfo {
        self.base.info()
    }

    fn bounded_decode_bound(&self) -> Option<BoundedDecodeBound> {
        // Raw pieces contain at most 10 bytes; each invalid raw byte can
        // require one three-byte replacement. Scratch stores the raw pieces.
        Some(BoundedDecodeBound::new(
            std::num::NonZeroUsize::new(30).unwrap(),
            10,
        ))
    }

    fn bounded_token_bytes_bound(&self) -> Option<std::num::NonZeroUsize> {
        std::num::NonZeroUsize::new(10)
    }

    fn token_bytes_bounded_into(
        &self,
        token: TokenId,
        output: &mut [u8],
    ) -> std::result::Result<Option<usize>, BoundedDecodeError> {
        let required = self.bounded_token_bytes_bound().unwrap().get();
        if output.len() < required {
            return Err(BoundedDecodeError::InsufficientScratch {
                required,
                available: output.len(),
            });
        }
        if !(1..=9).contains(&token.get()) {
            return Ok(None);
        }
        // Preserve the incomplete e4 / b8 ad pieces; decoding a token here
        // would replace its bytes and invalidate the UTF-8 continuation test.
        let piece = Self::piece(token);
        output[..piece.len()].copy_from_slice(piece);
        Ok(Some(piece.len()))
    }

    fn decode_bounded_into(
        &self,
        tokens: &[TokenId],
        _: bool,
        scratch: &mut [u8],
        output: &mut String,
    ) -> std::result::Result<(), BoundedDecodeError> {
        let required = self
            .bounded_decode_bound()
            .unwrap()
            .requirements(tokens.len())?;
        if scratch.len() < required.scratch_bytes {
            return Err(BoundedDecodeError::InsufficientScratch {
                required: required.scratch_bytes,
                available: scratch.len(),
            });
        }
        if output.capacity() < required.text_bytes {
            return Err(BoundedDecodeError::InsufficientOutput {
                required: required.text_bytes,
                available: output.capacity(),
            });
        }
        let action = self.probe.action.lock().unwrap().take();
        if let Some(action) = action {
            action();
        }
        if self.probe.fail.load(Ordering::Relaxed) {
            return Err(BoundedDecodeError::Unsupported);
        }
        let mut length = 0;
        for token in tokens {
            let piece = Self::piece(*token);
            scratch[length..length + piece.len()].copy_from_slice(piece);
            length += piece.len();
        }
        output.clear();
        let mut remaining = &scratch[..length];
        loop {
            match std::str::from_utf8(remaining) {
                Ok(valid) => {
                    output.push_str(valid);
                    return Ok(());
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    output.push_str(std::str::from_utf8(&remaining[..valid]).unwrap());
                    output.push('\u{FFFD}');
                    let skipped = error.error_len().unwrap_or(remaining.len() - valid);
                    remaining = &remaining[valid + skipped..];
                }
            }
        }
    }
}

fn sequence(request: InferenceRequest, tokens: &[u32]) -> SequenceState {
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1)]);
    sequence.generated_tokens = tokens.iter().copied().map(TokenId::new).collect();
    sequence
}

fn project(
    sequence: &SequenceState,
    tokenizer: &ProjectionTokenizer,
    terminal: Option<FinishReason>,
) -> StreamTextProjection {
    StreamProjectionSnapshot::capture(sequence)
        .project(tokenizer, terminal)
        .unwrap()
        .unwrap()
}

fn fixture(tokenizer: Arc<ProjectionTokenizer>) -> ContinuousBatchEngine {
    let config = EngineConfig::default();
    ContinuousBatchEngine::new(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        Arc::new(MockKvCacheManager::new(128)),
        Arc::new(MockModelExecutor::instant(32)),
        Arc::new(MockTensorFactory),
    )
    .unwrap()
}

#[tokio::test]
async fn output_projection_decodes_without_global_lock_and_preserves_utf8_history() {
    let tokenizer = Arc::new(ProjectionTokenizer::new());
    let engine = fixture(tokenizer.clone());
    let request = InferenceRequest::new("prompt", "projection-test");
    let id = request.id.clone();
    let (sender, mut receiver) = mpsc::channel(4);
    let mut state = sequence(request, &[2]);
    state.stream_sender = Some(sender);
    engine.inner.sequences.write().insert(id.clone(), state);

    let inner = Arc::downgrade(&engine.inner);
    let checked = Arc::new(AtomicBool::new(false));
    let seen = checked.clone();
    tokenizer.on_decode(move || {
        let inner = inner.upgrade().unwrap();
        let _guard = inner
            .sequences
            .try_write()
            .expect("decoder must run outside sequence lock");
        seen.store(true, Ordering::Relaxed);
    });
    engine.inner.send_stream_update(&id, TokenId::new(2)).await;
    assert!(checked.load(Ordering::Relaxed));
    assert!(receiver.try_recv().is_err());
    assert_eq!(engine.inner.sequences.read()[&id].streamed_text_len, 0);
    engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .generated_tokens
        .push(TokenId::new(3));
    engine.inner.send_stream_update(&id, TokenId::new(3)).await;
    assert_eq!(receiver.try_recv().unwrap().unwrap().text, "中");
    engine.inner.send_stream_update(&id, TokenId::new(3)).await;
    assert!(
        receiver.try_recv().is_err(),
        "same frontier must not be emitted twice"
    );
    let states = engine.inner.sequences.read();
    assert_eq!(states[&id].streamed_text_len, "中".len());
    assert_eq!(states[&id].emitted_chunks, 1);
}

#[tokio::test]
async fn output_projection_cancel_and_same_id_reuse_rejects_old_text() {
    let tokenizer = Arc::new(ProjectionTokenizer::new());
    let engine = fixture(tokenizer.clone());
    let request = InferenceRequest::new("prompt", "projection-test");
    let id = request.id.clone();
    let (old_sender, mut old_receiver) = mpsc::channel(2);
    let mut old = sequence(request.clone(), &[1]);
    old.stream_sender = Some(old_sender);
    engine.inner.sequences.write().insert(id.clone(), old);
    let (new_sender, mut new_receiver) = mpsc::channel(2);
    let inner = Arc::downgrade(&engine.inner);
    tokenizer.on_decode(move || {
        let inner = inner.upgrade().unwrap();
        let mut states = inner
            .sequences
            .try_write()
            .expect("cancel can acquire sequence lock");
        states.remove(&request.id).unwrap();
        // Identical ID, tokens, and watermark: only the incarnation differs.
        let mut replacement = sequence(request.clone(), &[1]);
        replacement.stream_sender = Some(new_sender);
        states.insert(request.id, replacement);
    });
    engine.inner.send_stream_update(&id, TokenId::new(1)).await;
    assert!(old_receiver.try_recv().is_err());
    assert!(new_receiver.try_recv().is_err());
    let states = engine.inner.sequences.read();
    assert_eq!(states[&id].streamed_text_len, 0);
    assert_eq!(states[&id].decoded_text_len, 0);
    assert_eq!(states[&id].emitted_chunks, 0);
    assert!(states[&id].first_emit_at.is_none());
}

#[tokio::test]
async fn output_projection_same_length_frontier_change_cannot_commit() {
    let tokenizer = Arc::new(ProjectionTokenizer::new());
    let engine = fixture(tokenizer.clone());
    let request = InferenceRequest::new("prompt", "projection-test");
    let id = request.id.clone();
    let (sender, mut receiver) = mpsc::channel(2);
    let mut state = sequence(request, &[1]);
    state.stream_sender = Some(sender);
    engine.inner.sequences.write().insert(id.clone(), state);
    let inner = Arc::downgrade(&engine.inner);
    let changed_id = id.clone();
    tokenizer.on_decode(move || {
        inner
            .upgrade()
            .unwrap()
            .sequences
            .write()
            .get_mut(&changed_id)
            .unwrap()
            .generated_tokens[0] = TokenId::new(6);
    });
    engine.inner.send_stream_update(&id, TokenId::new(1)).await;
    assert!(receiver.try_recv().is_err());
    assert_eq!(engine.inner.sequences.read()[&id].streamed_text_len, 0);
    assert_eq!(engine.inner.sequences.read()[&id].decoded_text_len, 0);
    engine.inner.send_stream_update(&id, TokenId::new(6)).await;
    assert_eq!(receiver.try_recv().unwrap().unwrap().text, "b");
}

#[test]
fn output_projection_stop_prefix_rebases_concurrent_terminal_without_duplication() {
    let tokenizer = ProjectionTokenizer::new();
    let mut state = sequence(InferenceRequest::new("prompt", "projection-test"), &[1, 4]);
    state.stop_text_seqs = vec!["STOP".to_owned()];
    let ordinary = project(&state, &tokenizer, None);
    let terminal = project(&state, &tokenizer, Some(FinishReason::Length));
    assert_eq!(ordinary.commit(&mut state).as_deref(), Some("hello"));
    assert_eq!(terminal.commit(&mut state).as_deref(), Some("ST"));
    assert_eq!(state.streamed_text_len, 7);
    assert_eq!(state.decoded_text_len, 7);

    let stale = project(&state, &tokenizer, None);
    state.streamed_text_len = 0;
    assert!(
        stale.commit(&mut state).is_none(),
        "rollback cannot reuse a projection"
    );
    state.generated_tokens.push(TokenId::new(5));
    let matched = project(&state, &tokenizer, Some(FinishReason::Stop));
    assert_eq!(matched.commit(&mut state).as_deref(), Some("hello"));
    assert_eq!(state.decoded_text_len, "helloSTOPtail".len());
    assert_eq!(
        visible_text_end(&["结束".to_owned()], "中结", false),
        "中".len()
    );
    assert_eq!(
        visible_text_end(&["结束".to_owned()], "中结", true),
        "中结".len()
    );
}

#[test]
fn output_projection_terminal_harmony_markers_follow_typed_protocol() {
    let tokenizer = ProjectionTokenizer::new();
    for (id, marker) in [(7, "<|return|>"), (8, "<|call|>"), (9, "<eos>")] {
        for protocol in [
            ModelOutputProtocol::Text,
            ModelOutputProtocol::HarmonyGptOss,
        ] {
            for reason in [
                FinishReason::EOS,
                FinishReason::Stop,
                FinishReason::Error,
                FinishReason::Length,
            ] {
                let mut state =
                    sequence(InferenceRequest::new("prompt", "projection-test"), &[1, id]);
                state.stop_token_ids.insert(id);
                state.sampling_params.model_output_protocol = protocol;
                let keep = reason == FinishReason::Length
                    || (reason != FinishReason::Error
                        && protocol == ModelOutputProtocol::HarmonyGptOss
                        && id != 9);
                let expected = if keep {
                    format!("hello{marker}")
                } else {
                    "hello".to_owned()
                };
                assert_eq!(
                    project(&state, &tokenizer, Some(reason)).commit(&mut state),
                    Some(expected)
                );
            }
        }
    }
}

#[test]
fn output_projection_errors_and_incomplete_utf8_leave_watermarks_unchanged() {
    let tokenizer = ProjectionTokenizer::new();
    let mut state = sequence(InferenceRequest::new("prompt", "projection-test"), &[1]);
    state.decoded_text_len = 17;
    tokenizer.probe.fail.store(true, Ordering::Relaxed);
    assert!(StreamProjectionSnapshot::capture(&state)
        .project(&tokenizer, None)
        .is_err());
    tokenizer.probe.fail.store(false, Ordering::Relaxed);
    state.generated_tokens = vec![TokenId::new(2)];
    for terminal in [None, Some(FinishReason::Length)] {
        assert!(StreamProjectionSnapshot::capture(&state)
            .project(&tokenizer, terminal)
            .unwrap()
            .is_none());
        assert_eq!(state.streamed_text_len, 0);
        assert_eq!(state.decoded_text_len, 17);
    }
}

struct ProjectionPlanRuntime(MockModelExecutor);
#[async_trait::async_trait]
impl ferrum_interfaces::ModelExecutor for ProjectionPlanRuntime {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.0.info()
    }
    fn capabilities(&self) -> ferrum_interfaces::model_executor::ExecutorCapabilities {
        self.0.capabilities()
    }
    fn status(&self) -> ferrum_interfaces::model_executor::ExecutorStatus {
        self.0.status()
    }
    fn execution_resource_authority(
        &self,
    ) -> ferrum_interfaces::model_executor::ExecutionResourceAuthority {
        ferrum_interfaces::model_executor::ExecutionResourceAuthority::PlanRuntime
    }
    async fn prefill(
        &self,
        input: &ferrum_interfaces::model_executor::PrefillInput,
    ) -> Result<ferrum_interfaces::model_executor::PrefillOutput> {
        self.0.prefill(input).await
    }
    async fn decode(
        &self,
        input: &ferrum_interfaces::model_executor::DecodeInput,
    ) -> Result<ferrum_interfaces::model_executor::DecodeOutput> {
        self.0.decode(input).await
    }
}

async fn credited_projection_fixture(
    tokenizer: Arc<ProjectionTokenizer>,
    request: InferenceRequest,
    tokens: &[u32],
) -> (
    ContinuousBatchEngine,
    ferrum_interfaces::output_flow::CreditedOutputSession,
) {
    use crate::continuous_engine::output_flow_runtime::OutputReadiness;
    let config = EngineConfig::default();
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        Arc::new(ProjectionPlanRuntime(MockModelExecutor::instant(32))),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    engine.inner.bg_loop_spawned.store(true, Ordering::Release);
    let id = request.id.clone();
    let session = engine
        .submit_credited_stream(
            request,
            ferrum_interfaces::InferenceRequestContext::capture(),
            Arc::new(ferrum_interfaces::output_flow::OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    let mut changed = engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .subscribe();
    let grant = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let readiness = engine.inner.sequences.read()[&id]
                .credited_output
                .as_ref()
                .unwrap()
                .port
                .try_take();
            if let OutputReadiness::Ready(grant) = readiness {
                break grant;
            }
            changed.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    {
        let mut states = engine.inner.sequences.write();
        let sequence = states.get_mut(&id).unwrap();
        sequence
            .generated_tokens
            .extend(tokens.iter().copied().map(TokenId::new));
        sequence.credited_output.as_mut().unwrap().grant = Some(grant);
    }
    (engine, session)
}

async fn drop_credited_projection_fixture(
    engine: &ContinuousBatchEngine,
    id: &RequestId,
    session: ferrum_interfaces::output_flow::CreditedOutputSession,
) {
    drop(session.frames);
    engine.inner.cancel_abandoned_requests().await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(id));
    assert_eq!(engine.inner.scheduler.trace_phase(id), None);
    let completion = tokio::time::timeout(std::time::Duration::from_secs(3), session.completion)
        .await
        .unwrap()
        .unwrap();
    drop(completion);
}

async fn credited_reused_id_ignores_outcome(tokens: &[u32], fail: bool) {
    use crate::continuous_engine::output_flow_runtime::OutputReadinessState;
    use futures::StreamExt;
    let tokenizer = Arc::new(ProjectionTokenizer::new());
    let mut request = InferenceRequest::new("prompt", "projection-test");
    request.sampling_params.max_tokens = 4;
    let id = request.id.clone();
    let (engine, old_session) =
        credited_projection_fixture(tokenizer.clone(), request.clone(), tokens).await;
    // Independently admitted owners let this focused fence test replace the
    // incarnation synchronously inside decode. Pool admission/reopen races
    // are tested separately by the output credit ledger's generation tests.
    let (replacement_engine, mut replacement_session) =
        credited_projection_fixture(tokenizer.clone(), request, tokens).await;
    let replacement = replacement_engine
        .inner
        .sequences
        .write()
        .remove(&id)
        .unwrap();
    let inner = Arc::downgrade(&engine.inner);
    let swapped_id = id.clone();
    // Keep the displaced admission alive until both incarnations can return
    // to their original schedulers for real cancellation. An ABA map swap is
    // not permission to drop an owned request slot.
    let displaced = Arc::new(Mutex::new(None));
    let displaced_in_decode = displaced.clone();
    tokenizer.on_decode(move || {
        let inner = inner.upgrade().unwrap();
        let old = inner.sequences.write().insert(swapped_id, replacement);
        *displaced_in_decode.lock().unwrap() = old;
    });
    tokenizer.probe.fail.store(fail, Ordering::Relaxed);
    engine
        .inner
        .send_stream_update(&id, TokenId::new(tokens[0]))
        .await;
    {
        let states = engine.inner.sequences.read();
        let state = &states[&id];
        let output = state.credited_output.as_ref().unwrap();
        assert!(
            output.failure.is_none(),
            "stale projection must not fail a replacement"
        );
        assert!(
            output.grant.is_some(),
            "stale empty projection must not consume its grant"
        );
        assert_eq!(output.accepted_ordinal, 0);
        assert_eq!(
            output.port.readiness(),
            OutputReadinessState::ProjectionBusy
        );
        assert_eq!(state.streamed_text_len, 0);
        assert_eq!(state.decoded_text_len, 0);
        assert_eq!(state.emitted_chunks, 0);
        assert!(state.first_emit_at.is_none());
    }
    assert!(futures::poll!(replacement_session.frames.next()).is_pending());
    let old = displaced.lock().unwrap().take().unwrap();
    let replacement = engine
        .inner
        .sequences
        .write()
        .insert(id.clone(), old)
        .unwrap();
    assert!(replacement_engine
        .inner
        .sequences
        .write()
        .insert(id.clone(), replacement)
        .is_none());
    drop_credited_projection_fixture(&engine, &id, old_session).await;
    drop_credited_projection_fixture(&replacement_engine, &id, replacement_session).await;
}

#[tokio::test]
async fn output_projection_credited_reused_id_ignores_old_text_result() {
    credited_reused_id_ignores_outcome(&[1], false).await;
}

#[tokio::test]
async fn output_projection_credited_reused_id_ignores_old_incomplete_utf8() {
    credited_reused_id_ignores_outcome(&[2], false).await;
}

#[tokio::test]
async fn output_projection_credited_reused_id_ignores_old_decode_error() {
    credited_reused_id_ignores_outcome(&[1], true).await;
}

#[tokio::test]
async fn output_projection_inflight_decode_retains_credit_after_sequence_cancel() {
    let tokenizer = Arc::new(ProjectionTokenizer::new());
    let mut request = InferenceRequest::new("prompt", "projection-test");
    request.sampling_params.max_tokens = 4;
    let id = request.id.clone();
    let (engine, session) = credited_projection_fixture(tokenizer.clone(), request, &[1]).await;
    let pool = engine
        .inner
        .output_credit_pool
        .get()
        .unwrap()
        .as_ref()
        .unwrap();
    let charged = pool.snapshot().data_used.projection_bytes;
    let snapshot = StreamProjectionSnapshot::capture(&engine.inner.sequences.read()[&id]);
    let (entered, entered_rx) = tokio::sync::oneshot::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    tokenizer.on_decode(move || {
        entered.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    let decode =
        tokio::task::spawn_blocking(move || snapshot.project_fenced(tokenizer.as_ref(), None));
    tokio::time::timeout(std::time::Duration::from_secs(3), entered_rx)
        .await
        .unwrap()
        .unwrap();
    drop(session.frames);
    engine.inner.cancel_abandoned_requests().await.unwrap();
    assert_eq!(engine.inner.scheduler.trace_phase(&id), None);
    let mut completion = session.completion;
    let mut wake = pool.subscribe();
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while pool.snapshot().open_accounts != 0 {
            wake.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(pool.snapshot().data_used.projection_bytes, charged);
    assert!(
        futures::poll!(&mut completion).is_pending(),
        "decoder snapshot outlives engine port"
    );
    release.send(()).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(3), decode)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        charged,
        "uncommitted decoded result still owns its lifetime"
    );
    drop(result);
    let completion = tokio::time::timeout(std::time::Duration::from_secs(3), completion)
        .await
        .unwrap()
        .unwrap();
    drop(completion);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}
