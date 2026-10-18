use super::*;

#[derive(Debug)]
struct SequenceTokenTiming {
    wall_anchor_unix_nanos: i64,
    wall_anchor_max_error_nanos: u64,
    decode_ready_nanos_since_request_start: Option<u64>,
    token_commit_nanos_since_request_start: Vec<u64>,
    decode_stage_intervals: Vec<EngineDecodeStageInterval>,
    pending_decode_scheduling_started_at: Option<Instant>,
}

impl SequenceTokenTiming {
    fn capture_anchor() -> Result<(Instant, Self)> {
        let wall_before = system_time_unix_nanos(SystemTime::now())?;
        let request_start = Instant::now();
        let wall_after = system_time_unix_nanos(SystemTime::now())?;
        let lower = wall_before.min(wall_after);
        let upper = wall_before.max(wall_after);
        let span = upper.saturating_sub(lower);
        let midpoint = lower.saturating_add(span / 2);
        Ok((
            request_start,
            Self {
                wall_anchor_unix_nanos: midpoint,
                wall_anchor_max_error_nanos: u64::try_from(span).unwrap_or(u64::MAX),
                decode_ready_nanos_since_request_start: None,
                token_commit_nanos_since_request_start: Vec::new(),
                decode_stage_intervals: Vec::new(),
                pending_decode_scheduling_started_at: None,
            },
        ))
    }

    fn nanos_since_request_start(request_start: Instant, at: Instant) -> Option<u64> {
        at.checked_duration_since(request_start)
            .map(|elapsed| u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX))
    }

    fn record_commit(&mut self, request_start: Instant, committed_at: Instant) {
        let elapsed =
            Self::nanos_since_request_start(request_start, committed_at).unwrap_or_else(|| {
                u64::try_from(request_start.elapsed().as_nanos()).unwrap_or(u64::MAX)
            });
        let monotonic = self
            .token_commit_nanos_since_request_start
            .last()
            .copied()
            .map_or(elapsed, |previous| previous.max(elapsed));
        self.token_commit_nanos_since_request_start.push(monotonic);
        self.pending_decode_scheduling_started_at = Some(committed_at);
    }

    fn record_decode_stage_interval(
        &mut self,
        request_start: Instant,
        stage: EngineDecodeStage,
        started_at: Instant,
        ended_at: Instant,
    ) {
        let Some(start_nanos_since_request_start) =
            Self::nanos_since_request_start(request_start, started_at)
        else {
            return;
        };
        let Some(end_nanos_since_request_start) =
            Self::nanos_since_request_start(request_start, ended_at)
        else {
            return;
        };
        if end_nanos_since_request_start < start_nanos_since_request_start {
            return;
        }
        self.decode_stage_intervals.push(EngineDecodeStageInterval {
            stage,
            start_nanos_since_request_start,
            end_nanos_since_request_start,
        });
    }

    fn record_decode_ready(&mut self, request_start: Instant) {
        if self.decode_ready_nanos_since_request_start.is_some() {
            return;
        }
        let ready_at = Instant::now();
        let ready_nanos =
            Self::nanos_since_request_start(request_start, ready_at).unwrap_or_else(|| {
                u64::try_from(request_start.elapsed().as_nanos()).unwrap_or(u64::MAX)
            });
        self.decode_ready_nanos_since_request_start = Some(ready_nanos);
        self.pending_decode_scheduling_started_at = Some(ready_at);
    }

    fn close_decode_scheduling(&mut self, request_start: Instant, ended_at: Instant) {
        let Some(started_at) = self.pending_decode_scheduling_started_at.take() else {
            return;
        };
        self.record_decode_stage_interval(
            request_start,
            EngineDecodeStage::DecodeScheduling,
            started_at,
            ended_at,
        );
    }

    fn into_evidence(self, output_tokens: usize) -> Result<EngineTokenTimingEvidence> {
        let evidence = EngineTokenTimingEvidence {
            clock_source: "rust_std_instant".to_string(),
            wall_anchor_unix_nanos: self.wall_anchor_unix_nanos,
            wall_anchor_max_error_nanos: self.wall_anchor_max_error_nanos,
            decode_ready_nanos_since_request_start: self.decode_ready_nanos_since_request_start,
            token_commit_nanos_since_request_start: self.token_commit_nanos_since_request_start,
            decode_stage_intervals: self.decode_stage_intervals,
        };
        evidence.validate(output_tokens).map_err(|error| {
            FerrumError::internal(format!("invalid engine token timing evidence: {error}"))
        })?;
        Ok(evidence)
    }
}

fn system_time_unix_nanos(time: SystemTime) -> Result<i64> {
    let duration = time.duration_since(UNIX_EPOCH).map_err(|error| {
        FerrumError::internal(format!("system clock predates Unix epoch: {error}"))
    })?;
    i64::try_from(duration.as_nanos())
        .map_err(|_| FerrumError::internal("system clock Unix nanos exceed i64"))
}

/// Selects which generated-token domain is visible to logits processors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum SequenceSamplingHistoryScope {
    #[default]
    FullGeneration,
    HiddenStructuredOutput,
    VisibleStructuredOutput {
        start_token_index: usize,
    },
}

/// Request-local token history visible to logits processors and samplers.
///
/// Hidden reasoning and the visible structured result are distinct output
/// domains. Keeping the typed scope and frequency table together prevents a
/// caller from changing `previous_tokens` without also rebuilding additive
/// and multiplicative penalty counts.
#[derive(Debug, Default)]
pub(super) struct SequenceSamplingHistory {
    scope: SequenceSamplingHistoryScope,
    token_frequencies: HashMap<TokenId, usize>,
}

impl SequenceSamplingHistory {
    pub(super) fn activate_visible_structured_output(
        &mut self,
        start_token_index: usize,
        generated_tokens: &[TokenId],
    ) -> Result<bool> {
        if start_token_index > generated_tokens.len() {
            return Err(FerrumError::internal(format!(
                "sampling history starts at token {start_token_index}, beyond generated length {}",
                generated_tokens.len()
            )));
        }
        let scope = SequenceSamplingHistoryScope::VisibleStructuredOutput { start_token_index };
        if self.scope == scope {
            return Ok(false);
        }
        self.scope = scope;
        self.rebuild_from(start_token_index, generated_tokens);
        Ok(true)
    }

    pub(super) fn enter_hidden_structured_output(&mut self) -> bool {
        if self.scope == SequenceSamplingHistoryScope::HiddenStructuredOutput {
            return false;
        }
        self.scope = SequenceSamplingHistoryScope::HiddenStructuredOutput;
        self.token_frequencies.clear();
        true
    }

    fn rebuild_from(&mut self, start_token_index: usize, generated_tokens: &[TokenId]) {
        self.token_frequencies.clear();
        for &token in &generated_tokens[start_token_index..] {
            *self.token_frequencies.entry(token).or_insert(0) += 1;
        }
    }

    pub(super) fn record(&mut self, token: TokenId) {
        if self.scope == SequenceSamplingHistoryScope::HiddenStructuredOutput {
            return;
        }
        *self.token_frequencies.entry(token).or_insert(0) += 1;
    }

    pub(super) fn previous_tokens<'a>(
        &self,
        generated_tokens: &'a [TokenId],
    ) -> Result<&'a [TokenId]> {
        let start_token_index = match self.scope {
            SequenceSamplingHistoryScope::FullGeneration => 0,
            SequenceSamplingHistoryScope::HiddenStructuredOutput => {
                return Ok(&generated_tokens[..0]);
            }
            SequenceSamplingHistoryScope::VisibleStructuredOutput { start_token_index } => {
                start_token_index
            }
        };
        generated_tokens.get(start_token_index..).ok_or_else(|| {
            FerrumError::internal(format!(
                "sampling history starts at token {}, beyond generated length {}",
                start_token_index,
                generated_tokens.len()
            ))
        })
    }

    pub(super) fn token_frequencies(&self) -> &HashMap<TokenId, usize> {
        &self.token_frequencies
    }

    pub(super) fn scope(&self) -> SequenceSamplingHistoryScope {
        self.scope
    }

    pub(super) fn token_count(&self, generated_tokens: &[TokenId]) -> usize {
        self.previous_tokens(generated_tokens)
            .map_or(0, <[TokenId]>::len)
    }

    pub(super) fn unique_token_count(&self) -> usize {
        self.token_frequencies.len()
    }
}

/// State of a running sequence in the continuous batch.
#[derive(Debug)]
pub struct SequenceState {
    pub request_id: RequestId,
    /// Original request — kept for re-submission after preemption.
    pub original_request: InferenceRequest,
    pub input_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    /// The executor's sole live incarnation and committed token position.
    /// This is an execution observation, not a second physical KV allocation.
    /// Exact existing scheduler queue identity while preparation is in flight.
    pub(super) pending_token_admission:
        Option<ferrum_scheduler::implementations::continuous::PrefixRequestKey>,
    pub(super) execution_frontier: Option<ferrum_execution::token_execution::ExecutionFrontier>,
    /// Preserve an interrupted participant's actual close observation until
    /// terminal processing transfers it into the independent close registry.
    pub(super) execution_close:
        Option<Box<Mutex<ferrum_execution::token_execution::ExecutionCloseOutcome>>>,
    pub(super) sampling_params: SamplingParams,
    /// Immutable logits-processing and sampler plan prepared once per request.
    pub(super) sampling_plan: TokenSamplingPlan,
    pub phase: RequestPhase,
    pub rng: SamplingRng,
    pub prefill_complete: bool,
    /// Number of prompt tokens already written into the model KV cache by
    /// opt-in unified chunked prefill. Zero for the normal full-prefill path.
    pub prefill_tokens_processed: usize,
    pub(super) stream_sender: Option<OutputSender>,
    pub response_sender: Option<tokio::sync::oneshot::Sender<Result<InferenceResponse>>>,
    pub(super) request_slot: Option<RequestSlotLease>,
    pub start_time: Instant,
    /// Present only for callers that explicitly request the latency preset.
    /// One entry is recorded after each generated token becomes committed
    /// engine state, independent of text decoding or stream flushing.
    token_timing: Option<SequenceTokenTiming>,
    /// Wall-clock `Instant` at which the first SSE chunk was actually
    /// sent to the client stream. Populated lazily by `send_stream_update`
    /// the first time a non-empty delta is emitted (multi-byte UTF-8
    /// buffering can defer that past the first scheduler-completed token).
    /// Used to record `ferrum.engine.ttft_seconds` and as the start point
    /// of the TPOT window.
    pub first_emit_at: Option<Instant>,
    /// Wall-clock `Instant` at the most recent successfully-sent chunk.
    /// Used to compute per-token ITL deltas (`ferrum.engine.itl_seconds`).
    pub last_emit_at: Option<Instant>,
    /// Count of stream chunks successfully sent to the client. Lags
    /// `generated_tokens.len()` by the number of tokens currently buffered
    /// for a multi-byte UTF-8 sequence (so a Chinese char split across
    /// 2 BPE tokens emits once, increments the count by 1).
    pub emitted_chunks: u32,
    pub tokens_this_iteration: usize,
    /// Number of times this request has been preempted.
    pub preemption_count: usize,
    /// Tokenizer-aware hard grammar for `json_object` and strict schema.
    pub structured_output_processor: Option<StructuredOutputProcessor>,
    /// Generated-token history scoped to the active typed output domain.
    pub(super) sampling_history: SequenceSamplingHistory,
    /// Single-token stop ids: model's EOS + any `stop_sequences` that encode to
    /// exactly one token. Checked against the last generated token each step
    /// — replaces the old "token id near top of vocab = EOS" placeholder. Built
    /// from `tokenizer.eos_token`, a common-EOS fallback list (`</s>`,
    /// `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`), and one-token encodings of
    /// `sampling_params.stop_sequences`.
    pub stop_token_ids: HashSet<u32>,
    /// Effective automatic EOS IDs; ignore_eos and explicit user-stop
    /// collisions are excluded so completion gates delay only model termination.
    pub model_eos_token_ids: Vec<u32>,
    /// Explicit single-token user stops, including IDs also declared model EOS.
    pub user_stop_token_ids: HashSet<u32>,
    /// Request-local compiled completion boundary. The common satisfied path
    /// is one enum comparison per token.
    pub(super) response_completion_state: ResponseCompletionState,
    /// Token IDs that should never be sampled as normal output. Used for
    /// tokenizer/model vocab holes such as Qwen3's reserved tail IDs and
    /// literal `<unk` / `<unk>` pieces.
    pub forbidden_token_ids: HashSet<u32>,
    /// Token IDs masked only before the first generated token.
    pub initial_forbidden_token_ids: HashSet<u32>,
    /// Base tokenizer vocabulary size. IDs above this are allowed only when
    /// they are explicitly whitelisted in `allowed_extended_token_ids`.
    pub tokenizer_base_vocab_size: Option<usize>,
    pub allowed_extended_token_ids: HashSet<u32>,
    /// User stop strings, including single-token encodings. Checked against
    /// accumulated decoded text to preserve stop boundaries inside tokens.
    pub stop_text_seqs: Vec<String>,
    /// Base token-validity mask for model-side greedy argmax.
    pub argmax_token_mask: Option<TokenSelectionMask>,
    /// First-token variant that also applies `initial_forbidden_token_ids`.
    pub initial_argmax_token_mask: Option<TokenSelectionMask>,
    /// A byte-level token may decode to a trailing replacement character
    /// until a later token supplies the remaining UTF-8 bytes.  The next
    /// step needs full logits so the engine can reject candidates that would
    /// flush that incomplete fragment into user-visible output.
    pub pending_decoded_utf8_fragment: bool,
    /// Raw byte-level tokenizer suffix that is a syntactically valid but
    /// incomplete UTF-8 scalar.  Unlike a lossy decoded string, this preserves
    /// whether a continuation token can actually complete the pending scalar.
    pub pending_decoded_utf8_bytes: Vec<u8>,
    /// Bytes of decoded `generated_tokens` already flushed via the stream
    /// channel. Used by `send_stream_update` to compute per-call delta from
    /// the full-history decode, so multi-byte UTF-8 sequences (Chinese chars,
    /// emoji) that span several BPE tokens don't get rendered as
    /// `\u{FFFD}` replacement chars when decoded one token at a time.
    pub streamed_text_len: usize,
    /// Stable decoded bytes examined by the output-quality filter. This can
    /// exceed `streamed_text_len` while a possible stop-string prefix is held.
    pub decoded_text_len: usize,
}

#[derive(Debug, Default)]
pub(super) struct SequenceCompletionResources {
    pub(super) request_slot: Option<RequestSlotLease>,
    pub(super) execution_frontier: Option<ferrum_execution::token_execution::ExecutionFrontier>,
    pub(super) execution_close: Option<ferrum_execution::token_execution::ExecutionCloseOutcome>,
}

impl SequenceState {
    pub fn new(request: InferenceRequest, input_tokens: Vec<TokenId>) -> Self {
        Self::new_with_tokenizer(request, input_tokens, None)
    }

    /// Build sequence state, optionally wiring a tokenizer for constrained
    /// decoding. Test-only direct constructors build a local grammar factory;
    /// product entrypoints use the fallible shared-factory constructor below.
    pub fn new_with_tokenizer(
        request: InferenceRequest,
        input_tokens: Vec<TokenId>,
        tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
    ) -> Self {
        Self::new_with_tokenizer_and_model_vocab_size(request, input_tokens, tokenizer, None)
    }

    pub fn new_with_tokenizer_and_model_vocab_size(
        request: InferenceRequest,
        input_tokens: Vec<TokenId>,
        tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
        model_vocab_size: Option<usize>,
    ) -> Self {
        Self::try_new_with_tokenizer_model_vocab_and_structured_factory(
            request,
            input_tokens,
            tokenizer,
            model_vocab_size,
            None,
        )
        .expect("direct SequenceState construction requires supported sampling and structured-output contracts")
    }

    pub fn try_new_with_tokenizer_model_vocab_and_structured_factory(
        request: InferenceRequest,
        input_tokens: Vec<TokenId>,
        tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
        model_vocab_size: Option<usize>,
        shared_structured_factory: Option<&StructuredOutputFactory>,
    ) -> Result<Self> {
        request.sampling_params.validate()?;
        if request.sampling_params.tfs.is_some()
            || request.sampling_params.typical_p.is_some()
            || request.sampling_params.mirostat.is_some()
        {
            return Err(FerrumError::unsupported(
                "tfs, typical_p, and mirostat are not supported by the token sampling plan",
            ));
        }
        let sampling_plan = TokenSamplingPlan::from_params(&request.sampling_params);
        let rng = request
            .sampling_params
            .seed
            .map(SamplingRng::seeded)
            .unwrap_or_else(SamplingRng::from_entropy);
        let needs_structured_output = request.requires_structured_output();
        let local_structured_factory = match (
            needs_structured_output,
            shared_structured_factory,
            tokenizer.as_ref(),
        ) {
            (true, None, Some(tokenizer)) => {
                Some(StructuredOutputFactory::new_with_model_vocab_size(
                    Arc::clone(tokenizer),
                    model_vocab_size,
                )?)
            }
            (true, None, None) => {
                return Err(FerrumError::config(
                    "structured output requires a tokenizer-aware grammar factory",
                ));
            }
            _ => None,
        };
        let ignore_eos = request
            .metadata
            .get("ferrum_ignore_eos")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let StopConditions {
            model_eos_token_ids,
            stop_token_ids,
            user_stop_token_ids,
            stop_text_seqs,
        } = resolve_stop_conditions(&request.sampling_params, tokenizer.as_deref(), ignore_eos);
        let response_completion_state = ResponseCompletionState::compile(
            &request.sampling_params.response_completion_boundary,
            tokenizer.as_deref(),
            &model_eos_token_ids,
            request.sampling_params.max_tokens,
        )?;
        let completion_boundary_open = !response_completion_state.allows_model_eos();
        let structured_output_processor = shared_structured_factory
            .or(local_structured_factory.as_ref())
            .map(|factory| {
                factory.create_processor_with_chat_contract(
                    &request.sampling_params.response_format,
                    &request.sampling_params.structured_output_start,
                    request.sampling_params.max_tokens,
                    &stop_token_ids,
                    &stop_text_seqs,
                    request
                        .api_request
                        .as_ref()
                        .and_then(|request| match request {
                            ferrum_types::ApiRequest::Chat(chat) => Some(chat),
                            ferrum_types::ApiRequest::Completion(_) => None,
                        }),
                    request.sampling_params.model_output_protocol,
                )
            })
            .transpose()?
            .flatten();
        let mut request_generated_control_token_texts = request
            .api_request
            .as_ref()
            .map(ferrum_types::ApiRequest::generated_control_token_texts)
            .unwrap_or_default()
            .to_vec();
        request_generated_control_token_texts.extend_from_slice(
            request
                .sampling_params
                .model_output_protocol
                .generated_control_token_texts(),
        );
        request_generated_control_token_texts.sort_unstable();
        request_generated_control_token_texts.dedup();
        let (
            forbidden_token_ids,
            model_greedy_forbidden_token_ids,
            tokenizer_base_vocab_size,
            allowed_extended_token_ids,
        ) = resolve_sampling_token_constraints(
            tokenizer.as_ref(),
            &stop_token_ids,
            &request_generated_control_token_texts,
        );
        let mut initial_forbidden_token_ids = HashSet::new();
        let initial_forbidden_token_texts = request
            .metadata
            .get("ferrum_initial_forbidden_token_texts")
            .and_then(|value| value.as_array());
        if let (Some(texts), Some(tok)) = (initial_forbidden_token_texts, tokenizer.as_deref()) {
            for text in texts.iter().filter_map(|value| value.as_str()) {
                if let Some(token) = tok.token_id(text) {
                    initial_forbidden_token_ids.insert(token.get());
                }
            }
        }
        let empty_initial_forbidden = HashSet::new();
        let mut argmax_token_mask = tokenizer.as_deref().map(|tok| {
            build_argmax_token_mask(
                tok,
                model_vocab_size,
                &forbidden_token_ids,
                &model_greedy_forbidden_token_ids,
                &empty_initial_forbidden,
                &stop_token_ids,
                &allowed_extended_token_ids,
            )
        });
        let mut initial_argmax_token_mask = if initial_forbidden_token_ids.is_empty() {
            None
        } else {
            tokenizer.as_deref().map(|tok| {
                build_argmax_token_mask(
                    tok,
                    model_vocab_size,
                    &forbidden_token_ids,
                    &model_greedy_forbidden_token_ids,
                    &initial_forbidden_token_ids,
                    &stop_token_ids,
                    &allowed_extended_token_ids,
                )
            })
        };
        if completion_boundary_open {
            if let Some(mask) = &mut argmax_token_mask {
                mask.set_tokens_validity(&model_eos_token_ids, false);
            }
            if let Some(mask) = &mut initial_argmax_token_mask {
                mask.set_tokens_validity(&model_eos_token_ids, false);
            }
        }
        let (start_time, token_timing) = if request.evidence_request.capture_engine_token_timing {
            let (start_time, timing) = SequenceTokenTiming::capture_anchor()?;
            (start_time, Some(timing))
        } else {
            (Instant::now(), None)
        };
        Ok(Self {
            request_id: request.id.clone(),
            original_request: request.clone(),
            input_tokens,
            generated_tokens: Vec::new(),
            pending_token_admission: None,
            execution_frontier: None,
            execution_close: None,
            sampling_params: request.sampling_params,
            sampling_plan,
            phase: RequestPhase::Waiting,
            rng,
            prefill_complete: false,
            prefill_tokens_processed: 0,
            stream_sender: None,
            response_sender: None,
            request_slot: None,
            start_time,
            token_timing,
            first_emit_at: None,
            last_emit_at: None,
            emitted_chunks: 0,
            tokens_this_iteration: 0,
            preemption_count: 0,
            structured_output_processor,
            sampling_history: SequenceSamplingHistory::default(),
            stop_token_ids,
            model_eos_token_ids,
            user_stop_token_ids,
            response_completion_state,
            forbidden_token_ids,
            initial_forbidden_token_ids,
            tokenizer_base_vocab_size,
            allowed_extended_token_ids,
            stop_text_seqs,
            argmax_token_mask,
            initial_argmax_token_mask,
            pending_decoded_utf8_fragment: false,
            pending_decoded_utf8_bytes: Vec::new(),
            streamed_text_len: 0,
            decoded_text_len: 0,
        })
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    pub(super) fn record_generated_token_commit(&mut self) {
        if let Some(timing) = &mut self.token_timing {
            timing.record_commit(self.start_time, Instant::now());
        }
    }

    #[cfg(test)]
    pub(super) fn closes_pending_decode_scheduling(&self) -> bool {
        self.token_timing
            .as_ref()
            .is_some_and(|timing| timing.pending_decode_scheduling_started_at.is_some())
    }

    pub(super) fn close_decode_scheduling(&mut self, ended_at: Instant) {
        if let Some(timing) = &mut self.token_timing {
            timing.close_decode_scheduling(self.start_time, ended_at);
        }
    }

    pub(super) fn record_decode_execution(&mut self, started_at: Instant, ended_at: Instant) {
        if let Some(timing) = &mut self.token_timing {
            timing.record_decode_stage_interval(
                self.start_time,
                EngineDecodeStage::DecodeExecution,
                started_at,
                ended_at,
            );
        }
    }

    pub(super) fn record_generated_token_commit_with_decode_postprocess(
        &mut self,
        postprocess_started_at: Option<Instant>,
    ) {
        let Some(timing) = &mut self.token_timing else {
            return;
        };
        let committed_at = Instant::now();
        if let Some(started_at) = postprocess_started_at {
            timing.record_decode_stage_interval(
                self.start_time,
                EngineDecodeStage::DecodePostprocess,
                started_at,
                committed_at,
            );
        }
        timing.record_commit(self.start_time, committed_at);
    }

    pub(super) fn record_decode_ready(&mut self) {
        if let Some(timing) = &mut self.token_timing {
            timing.record_decode_ready(self.start_time);
        }
    }

    pub(super) fn take_execution_evidence(&mut self) -> Result<Option<InferenceExecutionEvidence>> {
        let capture_prompt = self
            .original_request
            .evidence_request
            .capture_prompt_token_ids;
        let timing = self
            .token_timing
            .take()
            .map(|timing| timing.into_evidence(self.generated_tokens.len()))
            .transpose()?;
        Ok(
            (capture_prompt || timing.is_some()).then(|| InferenceExecutionEvidence {
                prompt_token_ids: if capture_prompt {
                    std::mem::take(&mut self.input_tokens)
                } else {
                    Vec::new()
                },
                output_token_ids: self.generated_tokens.clone(),
                engine_token_timing: timing,
            }),
        )
    }

    /// Original immutable sampling parameters used to prepare this request.
    pub fn sampling_params(&self) -> &SamplingParams {
        &self.sampling_params
    }

    pub fn prefill_context_tokens(&self) -> Vec<TokenId> {
        if self.generated_tokens.is_empty() {
            return self.input_tokens.clone();
        }
        let mut tokens = Vec::with_capacity(self.input_tokens.len() + self.generated_tokens.len());
        tokens.extend_from_slice(&self.input_tokens);
        tokens.extend_from_slice(&self.generated_tokens);
        tokens
    }

    pub fn prefill_context_len(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    pub fn model_decode_metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = self.original_request.metadata.clone();
        if self.requires_engine_full_logits_for_sampling() {
            metadata.insert(
                "ferrum_require_full_logits".to_string(),
                serde_json::json!(true),
            );
        }
        metadata.insert(
            "ferrum_kv_capacity_hint".to_string(),
            serde_json::json!(self
                .prefill_context_len()
                .max(self.input_tokens.len() + self.sampling_params.max_tokens.saturating_sub(1))),
        );
        metadata.insert(
            KV_ADMISSION_TARGET_LEN_METADATA_KEY.to_string(),
            serde_json::json!(self.prefill_context_len()),
        );
        metadata
    }

    pub(super) fn model_maximum_sequence_tokens(&self) -> usize {
        self.prefill_context_len().max(
            self.input_tokens
                .len()
                .saturating_add(self.sampling_params.max_tokens.saturating_sub(1)),
        )
    }

    pub(super) fn take_completion_resources(&mut self) -> SequenceCompletionResources {
        SequenceCompletionResources {
            request_slot: self.request_slot.take(),
            execution_frontier: self.execution_frontier.take(),
            execution_close: self
                .execution_close
                .take()
                .map(|close| (*close).into_inner()),
        }
    }

    pub fn model_decode_logits_policy(&self) -> LogitsReturnPolicy {
        if !self.can_use_model_greedy_argmax() {
            return LogitsReturnPolicy::FullLogits;
        }
        let token_mask =
            if self.generated_tokens.is_empty() && self.initial_argmax_token_mask.is_some() {
                self.initial_argmax_token_mask.clone()
            } else {
                self.argmax_token_mask.clone()
            };
        LogitsReturnPolicy::GreedyArgmax {
            token_mask,
            repetition_penalty: self.model_decode_repetition_penalty(),
        }
    }

    pub(super) fn can_use_model_greedy_argmax(&self) -> bool {
        use ferrum_types::ResponseFormat;

        let params = &self.sampling_params;
        params.temperature == 0.0
            && params.top_p == 1.0
            && params.top_k.is_none()
            && params.repetition_penalty > 0.0
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0
            && params.min_p.is_none()
            && params.tfs.is_none()
            && params.typical_p.is_none()
            && params.mirostat.is_none()
            && !self.pending_decoded_utf8_fragment
            && self.structured_output_processor.is_none()
            && matches!(params.response_format, ResponseFormat::Text)
    }

    pub(super) fn model_decode_repetition_penalty(&self) -> Option<GreedyRepetitionPenalty> {
        let penalty = self.sampling_params.repetition_penalty;
        if penalty == 1.0 || self.generated_tokens.is_empty() {
            return None;
        }
        let visible_tokens = self
            .sampling_history
            .previous_tokens(&self.generated_tokens)
            .ok()?;
        let token_ids: Vec<u32> = visible_tokens.iter().map(|token| token.get()).collect();
        if token_ids.is_empty() {
            None
        } else {
            Some(GreedyRepetitionPenalty::new(penalty, token_ids))
        }
    }

    pub(super) fn accept_response_completion_token(
        &mut self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        token: TokenId,
    ) -> Result<()> {
        if !self.response_completion_state.allows_model_eos()
            && self.model_eos_token_ids.contains(&token.get())
        {
            return Err(FerrumError::model(format!(
                "model selected EOS token {} before satisfying the response completion boundary",
                token.get()
            )));
        }
        let model_eos_availability =
            self.response_completion_state
                .observe(&self.generated_tokens, token, tokenizer)?;
        if let Some(allows_model_eos) = model_eos_availability {
            if let Some(mask) = &mut self.argmax_token_mask {
                mask.set_tokens_validity(&self.model_eos_token_ids, allows_model_eos);
            }
            if let Some(mask) = &mut self.initial_argmax_token_mask {
                mask.set_tokens_validity(&self.model_eos_token_ids, allows_model_eos);
            }
        }
        Ok(())
    }

    pub fn validate_and_commit_model_greedy_argmax_token(
        &mut self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        token: TokenId,
    ) -> Result<()> {
        let token_detail = || self.describe_model_greedy_argmax_token(tokenizer, token);
        if !self.can_use_model_greedy_argmax() {
            return Err(FerrumError::model(format!(
                "model returned a greedy token for request requiring full logits ({})",
                token_detail()
            )));
        }

        let token_id = token.get();
        if self.forbidden_token_ids.contains(&token_id) {
            return Err(FerrumError::model(format!(
                "model greedy argmax returned a forbidden token ({})",
                token_detail()
            )));
        }
        if self.generated_tokens.is_empty() && self.initial_forbidden_token_ids.contains(&token_id)
        {
            return Err(FerrumError::model(format!(
                "model greedy argmax returned an initially forbidden token ({})",
                token_detail()
            )));
        }
        if self
            .tokenizer_base_vocab_size
            .is_some_and(|base| token_id as usize >= base)
            && !self.allowed_extended_token_ids.contains(&token_id)
        {
            return Err(FerrumError::model(format!(
                "model greedy argmax returned a disallowed extended-vocab token ({})",
                token_detail()
            )));
        }
        if self.sample_candidate_decodes_to_forbidden_output(
            tokenizer,
            self.decoded_text_len,
            token,
            None,
        ) {
            return Err(FerrumError::model(format!(
                "model greedy argmax token decoded to forbidden output ({})",
                token_detail()
            )));
        }

        self.commit_generated_token(tokenizer, token)
    }

    pub(super) fn commit_generated_token(
        &mut self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        token: TokenId,
    ) -> Result<()> {
        let next_pending_utf8_bytes = tokenizer
            .and_then(|tokenizer| tokenizer.token_bytes(token))
            .map(|bytes| advance_pending_utf8_fragment(&self.pending_decoded_utf8_bytes, &bytes));
        if next_pending_utf8_bytes
            .as_ref()
            .is_some_and(|result| result.is_err())
        {
            return Err(FerrumError::model(format!(
                "token {} violates the pending UTF-8 byte fragment",
                token.get()
            )));
        }
        self.accept_response_completion_token(tokenizer, token)?;
        self.generated_tokens.push(token);
        self.sampling_history.record(token);
        match next_pending_utf8_bytes {
            Some(Ok(bytes)) => {
                self.pending_decoded_utf8_fragment = !bytes.is_empty();
                self.pending_decoded_utf8_bytes = bytes;
            }
            Some(Err(())) => unreachable!("invalid UTF-8 fragment was rejected before commit"),
            None => {
                self.pending_decoded_utf8_bytes.clear();
                self.pending_decoded_utf8_fragment = tokenizer.is_some_and(|tokenizer| {
                    tokenizer
                        .decode(&self.generated_tokens, true)
                        .map(|text| {
                            self.decoded_text_len <= text.len()
                                && text.is_char_boundary(self.decoded_text_len)
                                && text[self.decoded_text_len..].contains('\u{FFFD}')
                                && text.ends_with('\u{FFFD}')
                        })
                        .unwrap_or(true)
                });
            }
        }
        Ok(())
    }

    pub(super) fn describe_model_greedy_argmax_token(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        token: TokenId,
    ) -> String {
        let token_text = tokenizer
            .and_then(|tokenizer| tokenizer.token_text(token))
            .map(|text| format!("{text:?}"))
            .unwrap_or_else(|| "None".to_string());
        let decoded_delta = tokenizer
            .map(|tokenizer| tokenizer.decode_incremental(&self.generated_tokens, token))
            .map(|result| match result {
                Ok(text) => format!("{text:?}"),
                Err(err) => format!("decode_error:{err}"),
            })
            .unwrap_or_else(|| "None".to_string());
        format!(
            "token_id={}, token_text={}, decoded_delta={}, generated_tokens={}, \
             forbidden_count={}, initial_forbidden_count={}, base_vocab_size={:?}, \
             allowed_extended_count={}, argmax_mask={}, initial_argmax_mask={}",
            token.get(),
            token_text,
            decoded_delta,
            self.generated_tokens.len(),
            self.forbidden_token_ids.len(),
            self.initial_forbidden_token_ids.len(),
            self.tokenizer_base_vocab_size,
            self.allowed_extended_token_ids.len(),
            Self::describe_argmax_mask_value(self.argmax_token_mask.as_ref(), token),
            Self::describe_argmax_mask_value(self.initial_argmax_token_mask.as_ref(), token)
        )
    }

    pub(super) fn describe_argmax_mask_value(
        mask: Option<&TokenSelectionMask>,
        token: TokenId,
    ) -> String {
        match mask {
            Some(mask) => {
                let value = mask
                    .valid_token_mask
                    .get(token.get() as usize)
                    .copied()
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "out_of_range".to_string());
                format!(
                    "fingerprint={},len={},value={}",
                    mask.fingerprint,
                    mask.len(),
                    value
                )
            }
            None => "none".to_string(),
        }
    }

    pub fn requires_engine_full_logits_for_sampling(&self) -> bool {
        use ferrum_types::ResponseFormat;

        self.structured_output_processor.is_some()
            || matches!(
                self.sampling_params.response_format,
                ResponseFormat::JsonSchema(_)
            )
    }

    pub fn has_structured_output_constraint(&self) -> bool {
        self.structured_output_processor.is_some()
    }

    pub fn requires_full_logits_for_sampling(&self) -> bool {
        self.requires_engine_full_logits_for_sampling()
            || self.argmax_token_mask.is_some()
            || (self.generated_tokens.is_empty() && self.initial_argmax_token_mask.is_some())
    }

    pub fn reset_guided_processors(&mut self) -> Result<()> {
        if let Some(processor) = &self.structured_output_processor {
            processor.reset()?;
            // Replay has not resolved the hidden/visible boundary yet. Keep
            // penalties disabled until the grammar mask reports the exact
            // visible-output start, then rebuild that suffix once.
            self.sampling_history.enter_hidden_structured_output();
        }
        Ok(())
    }

    /// Return the reason this sequence should stop, if any.
    ///
    /// Explicit user IDs or decoded text stops take precedence over automatic
    /// model EOS, followed by the token budget. Text-stop decoding only runs
    /// for requests that supplied a stop string, so the common EOS path stays
    /// cheap. Keep the actual reason distinct until protocol serialization.
    pub fn stop_reason(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ) -> Option<FinishReason> {
        if let Some(&last_token) = self.generated_tokens.last() {
            if self.user_stop_token_ids.contains(&last_token.get()) {
                return Some(FinishReason::Stop);
            }
        }
        if !self.stop_text_seqs.is_empty() {
            if let Some(tok) = tokenizer {
                if let Ok(text) = tok.decode(&self.generated_tokens, true) {
                    if self
                        .stop_text_seqs
                        .iter()
                        .any(|stop| !stop.is_empty() && text.contains(stop))
                    {
                        return Some(FinishReason::Stop);
                    }
                }
            }
        }
        if let Some(processor) = &self.structured_output_processor {
            match processor
                .is_protocol_complete_with_terminals(&self.generated_tokens, &self.stop_token_ids)
            {
                Ok(true) => return Some(FinishReason::EOS),
                Ok(false) => {}
                Err(_) => return Some(FinishReason::Error),
            }
        }
        if let Some(last_token) = self.generated_tokens.last() {
            if self.model_eos_token_ids.contains(&last_token.get()) {
                return Some(FinishReason::EOS);
            }
        }
        if self.generated_tokens.len() >= self.sampling_params.max_tokens {
            return Some(FinishReason::Length);
        }
        None
    }

    /// Return whether the just-committed token belongs in the incremental
    /// response stream.
    ///
    /// Ordinary stop/EOS tokens remain engine-owned and are never exposed.
    /// Harmony is different: `<|call|>` and `<|return|>` are both model EOS
    /// tokens and required wire-protocol terminators, so its typed response
    /// parser must observe the decoded marker before completion is published.
    pub(in crate::continuous_engine) fn should_stream_generated_token(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        token: TokenId,
        stop_reason: Option<FinishReason>,
    ) -> bool {
        match stop_reason {
            Some(FinishReason::Error) => false,
            Some(FinishReason::Stop | FinishReason::EOS) => {
                self.sampling_params.model_output_protocol
                    == ferrum_types::ModelOutputProtocol::HarmonyGptOss
                    && (tokenizer.is_some_and(|tokenizer| {
                        ["<|call|>", "<|return|>"]
                            .into_iter()
                            .filter_map(|marker| tokenizer.token_id(marker))
                            .any(|terminal| terminal == token)
                    }) || (stop_reason == Some(FinishReason::EOS)
                        && !self.stop_token_ids.contains(&token.get())
                        && self
                            .structured_output_processor
                            .as_ref()
                            .is_some_and(|processor| {
                                processor
                                    .is_protocol_complete_with_terminals(
                                        &self.generated_tokens,
                                        &self.stop_token_ids,
                                    )
                                    .unwrap_or(false)
                            })))
            }
            _ => true,
        }
    }

    /// Cheap stop check for tests and callers that do not have tokenizer
    /// access. Engine hot paths use `stop_reason` through `EngineInner`.
    pub fn should_stop(&self) -> bool {
        self.stop_reason(None).is_some()
    }

    /// Sample and commit the next token with the full processor chain.
    pub fn sample_and_commit_with_processors(&mut self, logits: &mut [f32]) -> Result<TokenId> {
        self.sample_and_commit_with_processors_and_tokenizer(logits, None)
    }

    pub fn sample_and_commit_with_processors_and_tokenizer(
        &mut self,
        logits: &mut [f32],
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ) -> Result<TokenId> {
        use ferrum_interfaces::sampler::SamplingContext;

        // The grammar mask runs first. There is deliberately no invalid-token
        // fallback: a dead grammar state is a request error, not permission to
        // return malformed JSON.
        let mut required_structured_delimiter_token_id = None;
        let mut grammar_start_token_index = None;
        let has_structured_output = self.structured_output_processor.is_some();
        if let Some(processor) = &self.structured_output_processor {
            let constraint = processor.mask_logits_with_terminals(
                logits,
                &self.generated_tokens,
                &self.stop_token_ids,
                &self.allowed_extended_token_ids,
            )?;
            required_structured_delimiter_token_id = constraint.required_delimiter_token_id;
            grammar_start_token_index = constraint.grammar_start_token_index;
            if constraint.accepting
                && processor
                    .complete_root_text_len_with_terminals(
                        &self.generated_tokens,
                        &self.stop_token_ids,
                    )?
                    .is_some()
            {
                // A single token may contain both the reasoning delimiter and
                // its complete result. The whole-root grammar has checked
                // those bytes even when the token-sequence matcher cannot
                // observe a separately encoded delimiter.
                self.response_completion_state = ResponseCompletionState::Satisfied;
                if let Some(mask) = &mut self.argmax_token_mask {
                    mask.set_tokens_validity(&self.model_eos_token_ids, true);
                }
                if let Some(mask) = &mut self.initial_argmax_token_mask {
                    mask.set_tokens_validity(&self.model_eos_token_ids, true);
                }
            }
            if !constraint.accepting {
                // Native terminators belong to the grammar itself. The root
                // becomes accepting only after this exact allowed token is
                // consumed; other early stop IDs must remain masked.
                if constraint.grammar_owned_terminal_token_id.is_some() {
                    for &token_id in &self.stop_token_ids {
                        if constraint.grammar_owned_terminal_token_id != Some(token_id) {
                            if let Some(logit) = logits.get_mut(token_id as usize) {
                                *logit = f32::NEG_INFINITY;
                            }
                        }
                    }
                } else {
                    mask_stop_token_logits(logits, &self.stop_token_ids);
                }
            }
        }
        match (has_structured_output, grammar_start_token_index) {
            (true, Some(start_token_index)) => {
                self.sampling_history.activate_visible_structured_output(
                    start_token_index,
                    &self.generated_tokens,
                )?;
            }
            (true, None) => {
                self.sampling_history.enter_hidden_structured_output();
            }
            (false, _) => {}
        }
        if !self.response_completion_state.allows_model_eos() {
            for &token_id in &self.model_eos_token_ids {
                if let Some(logit) = logits.get_mut(token_id as usize) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        for &token_id in &self.forbidden_token_ids {
            if required_structured_delimiter_token_id == Some(token_id) {
                continue;
            }
            if let Some(logit) = logits.get_mut(token_id as usize) {
                *logit = f32::NEG_INFINITY;
            }
        }
        if self.generated_tokens.is_empty() {
            for &token_id in &self.initial_forbidden_token_ids {
                if required_structured_delimiter_token_id == Some(token_id) {
                    continue;
                }
                if let Some(logit) = logits.get_mut(token_id as usize) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(base_vocab_size) = self.tokenizer_base_vocab_size {
            if logits.len() > base_vocab_size {
                for (token_id, logit) in logits.iter_mut().enumerate().skip(base_vocab_size) {
                    if required_structured_delimiter_token_id != Some(token_id as u32)
                        && !self.allowed_extended_token_ids.contains(&(token_id as u32))
                    {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }

        let step = self.generated_tokens.len();
        let vocab_size = logits.len();
        let previous_streamed_text_len = self.decoded_text_len;
        let token = {
            let previous_tokens = self
                .sampling_history
                .previous_tokens(&self.generated_tokens)?;
            let mut ctx = SamplingContext::new(
                step,
                &self.sampling_params,
                logits,
                previous_tokens,
                self.sampling_history.token_frequencies(),
                vocab_size,
            );
            self.sampling_plan.processor_chain.process(&mut ctx)?;
            if !ctx.logits.iter().any(|logit| logit.is_finite()) {
                let message = if self.structured_output_processor.is_some() {
                    "structured-output constraints and engine sampling policies have no finite token"
                } else {
                    "engine sampling policies have no finite token"
                };
                return Err(FerrumError::model(message));
            }
            let mut attempts = 0usize;
            let mut rejected_tokens = Vec::new();
            loop {
                let token = self
                    .sampling_plan
                    .sampler
                    .sample_with_context(&ctx, &mut self.rng)?;
                if !self.sample_candidate_decodes_to_forbidden_output(
                    tokenizer,
                    previous_streamed_text_len,
                    token,
                    required_structured_delimiter_token_id,
                ) {
                    break token;
                }
                if rejected_tokens.len() < 8 {
                    rejected_tokens.push(token);
                }
                if let Some(logit) = ctx.logits.get_mut(usize::from(token)) {
                    *logit = f32::NEG_INFINITY;
                }
                attempts += 1;
                if attempts >= FORBIDDEN_DECODE_RESAMPLE_LIMIT {
                    self.log_forbidden_decode_resample_failure(
                        tokenizer,
                        previous_streamed_text_len,
                        &rejected_tokens,
                    );
                    return Err(FerrumError::model(
                        "sampling candidates decoded to forbidden output",
                    ));
                }
            }
        };

        self.commit_generated_token(tokenizer, token)?;

        Ok(token)
    }

    pub(super) fn sample_candidate_decodes_to_forbidden_output(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        previous_streamed_text_len: usize,
        token: TokenId,
        required_structured_delimiter_token_id: Option<u32>,
    ) -> bool {
        if required_structured_delimiter_token_id == Some(token.get()) {
            return false;
        }
        let Some(tokenizer) = tokenizer else {
            return false;
        };
        let mut tokens = Vec::with_capacity(self.generated_tokens.len() + 1);
        tokens.extend_from_slice(&self.generated_tokens);
        tokens.push(token);
        let candidate_is_stop = self.stop_token_ids.contains(&token.get());
        let candidate_is_non_stop_control =
            self.allowed_extended_token_ids.contains(&token.get()) && !candidate_is_stop;
        if let Some(bytes) = tokenizer.token_bytes(token) {
            match advance_pending_utf8_fragment(&self.pending_decoded_utf8_bytes, &bytes) {
                Ok(pending) if candidate_is_stop && !pending.is_empty() => return true,
                Ok(_) => {}
                Err(()) => return true,
            }
        }
        tokenizer
            .decode(&tokens, true)
            .map(|text| {
                decoded_delta_has_forbidden_quality(
                    &text,
                    previous_streamed_text_len,
                    candidate_is_stop,
                    candidate_is_non_stop_control,
                )
            })
            .unwrap_or(true)
    }

    pub(super) fn log_forbidden_decode_resample_failure(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        previous_streamed_text_len: usize,
        rejected_tokens: &[TokenId],
    ) {
        let generated_tail: Vec<String> = self
            .generated_tokens
            .iter()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|token| describe_token_for_log(tokenizer, *token))
            .collect();
        let rejected: Vec<String> = rejected_tokens
            .iter()
            .map(|token| describe_token_for_log(tokenizer, *token))
            .collect();
        warn!(
            request_id = %self.request_id,
            generated_len = self.generated_tokens.len(),
            previous_streamed_text_len,
            generated_tail = ?generated_tail,
            rejected_candidates = ?rejected,
            "sampling candidates decoded to forbidden output"
        );
    }
}

impl Drop for SequenceState {
    fn drop(&mut self) {
        if self.request_slot.is_some() {
            let message = "sequence state dropped with owned request slot";
            warn!(
                request_id = %self.request_id,
                execution_frontier = ?self.execution_frontier,
                has_close_observation = self.execution_close.is_some(),
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}

pub(super) fn describe_token_for_log(
    tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    token: TokenId,
) -> String {
    let Some(tokenizer) = tokenizer else {
        return token.get().to_string();
    };
    let raw = tokenizer.token_text(token).unwrap_or("<missing>");
    let decoded = tokenizer
        .decode(&[token], true)
        .unwrap_or_else(|_| "<decode-error>".to_string());
    format!("{} raw={:?} decoded={:?}", token.get(), raw, decoded)
}

pub(super) fn mask_stop_token_logits(logits: &mut [f32], stop_token_ids: &HashSet<u32>) {
    for &token_id in stop_token_ids {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Engine inner – shared via Arc so we can spawn tasks
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
#[must_use = "request slot leases must be consumed by reject() or close()"]
pub(super) struct RequestSlotLease {
    pub(super) request_id: RequestId,
    pub(super) admitted: bool,
    pub(super) armed: bool,
}

impl RequestSlotLease {
    pub(super) fn open(engine: &EngineInner, request_id: RequestId) -> Self {
        engine.trace_request_open(&request_id);
        Self {
            request_id,
            admitted: false,
            armed: true,
        }
    }

    pub(super) fn admit(&mut self, engine: &EngineInner) {
        if !self.admitted {
            engine.trace_request_admitted(&self.request_id);
            self.admitted = true;
        }
    }

    pub(super) fn reject(mut self, engine: &EngineInner, reason: String) {
        engine.trace_request_rejected(&self.request_id, reason);
        self.armed = false;
    }

    pub(super) fn close(mut self, engine: &EngineInner) {
        if self.admitted {
            engine.trace_request_close(&self.request_id);
        } else {
            engine.trace_request_owner_close(&self.request_id);
        }
        self.armed = false;
    }
}

impl Drop for RequestSlotLease {
    fn drop(&mut self) {
        if self.armed {
            let message = "request slot lease dropped without explicit reject or close";
            warn!(
                request_id = %self.request_id,
                admitted = self.admitted,
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}
