use super::*;

#[derive(Debug)]
struct SequenceTokenTiming {
    wall_anchor_unix_nanos: i64,
    wall_anchor_max_error_nanos: u64,
    decode_ready_nanos_since_request_start: Option<u64>,
    token_commit_nanos_since_request_start: Vec<u64>,
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
            },
        ))
    }

    fn record_commit(&mut self, request_start: Instant) {
        let elapsed = u64::try_from(request_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let monotonic = self
            .token_commit_nanos_since_request_start
            .last()
            .copied()
            .map_or(elapsed, |previous| previous.max(elapsed));
        self.token_commit_nanos_since_request_start.push(monotonic);
    }

    fn record_decode_ready(&mut self, request_start: Instant) {
        self.decode_ready_nanos_since_request_start
            .get_or_insert_with(|| {
                u64::try_from(request_start.elapsed().as_nanos()).unwrap_or(u64::MAX)
            });
    }

    fn into_evidence(self, output_tokens: usize) -> Result<EngineTokenTimingEvidence> {
        let evidence = EngineTokenTimingEvidence {
            clock_source: "rust_std_instant".to_string(),
            wall_anchor_unix_nanos: self.wall_anchor_unix_nanos,
            wall_anchor_max_error_nanos: self.wall_anchor_max_error_nanos,
            decode_ready_nanos_since_request_start: self.decode_ready_nanos_since_request_start,
            token_commit_nanos_since_request_start: self.token_commit_nanos_since_request_start,
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

/// State of a running sequence in the continuous batch.
#[derive(Debug)]
pub struct SequenceState {
    pub request_id: RequestId,
    /// Original request — kept for re-submission after preemption.
    pub original_request: InferenceRequest,
    pub input_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    pub(super) model_kv: Option<SequenceModelKvState>,
    pub(super) recurrent_state: Option<SequenceRecurrentState>,
    pub(super) sampling_params: SamplingParams,
    /// Immutable logits-processing and sampler plan prepared once per request.
    pub(super) sampling_plan: TokenSamplingPlan,
    pub phase: RequestPhase,
    pub rng: SamplingRng,
    pub prefill_complete: bool,
    /// Number of prompt tokens already written into the model KV cache by
    /// opt-in unified chunked prefill. Zero for the normal full-prefill path.
    pub prefill_tokens_processed: usize,
    pub stream_sender: Option<mpsc::Sender<Result<StreamChunk>>>,
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
    pub(super) draft_kv: Option<SequenceDraftKvState>,
    /// Generated-token counts shared by repetition, presence, and frequency penalties.
    pub token_frequencies: HashMap<TokenId, usize>,
    /// Single-token stop ids: model's EOS + any `stop_sequences` that encode to
    /// exactly one token. Checked against the last generated token each step
    /// — replaces the old "token id near top of vocab = EOS" placeholder. Built
    /// from `tokenizer.eos_token`, a common-EOS fallback list (`</s>`,
    /// `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`), and one-token encodings of
    /// `sampling_params.stop_sequences`.
    pub stop_token_ids: HashSet<u32>,
    /// Model-owned EOS ids, kept separate from user stop conditions so a
    /// response-completion boundary can delay only model termination.
    pub model_eos_token_ids: Vec<u32>,
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
    /// Multi-token text stop sequences (`stop_sequences` entries that don't
    /// resolve to a single token). Checked via accumulated decoded text.
    pub stop_text_seqs: Vec<String>,
    /// Base token-validity mask for model-side greedy argmax.
    pub argmax_token_mask: Option<TokenSelectionMask>,
    /// First-token variant that also applies `initial_forbidden_token_ids`.
    pub initial_argmax_token_mask: Option<TokenSelectionMask>,
    /// Bytes of decoded `generated_tokens` already flushed via the stream
    /// channel. Used by `send_stream_update` to compute per-call delta from
    /// the full-history decode, so multi-byte UTF-8 sequences (Chinese chars,
    /// emoji) that span several BPE tokens don't get rendered as
    /// `\u{FFFD}` replacement chars when decoded one token at a time.
    pub streamed_text_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SequenceKvAllocation {
    pub(super) request_id: RequestId,
    pub(super) blocks: usize,
}

impl SequenceKvAllocation {
    pub(super) fn new(request_id: RequestId, blocks: usize) -> Self {
        Self {
            request_id,
            blocks: blocks.max(1),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum SequenceKvRelease {
    /// The model runtime is the only release authority. This covers vNext
    /// plan-runtime leases and cloned prefix-cache references; neither has a
    /// second allocation in the legacy engine KV manager.
    RuntimeManaged,
    /// Transitional legacy composition: release the model cache reference and
    /// the exact engine KV-manager allocation together.
    LegacyAllocated(SequenceKvAllocation),
}

#[derive(Debug, Clone)]
pub(super) struct SequenceModelKvState {
    pub(super) cache: Arc<dyn KvCacheHandle>,
    pub(super) model_cache_id: String,
    pub(super) release: SequenceKvRelease,
}

impl SequenceModelKvState {
    pub(super) fn runtime_managed(cache: Arc<dyn KvCacheHandle>) -> Self {
        let model_cache_id = cache.cache_id();
        Self {
            cache,
            model_cache_id,
            release: SequenceKvRelease::RuntimeManaged,
        }
    }

    pub(super) fn legacy_allocated(
        cache: Arc<dyn KvCacheHandle>,
        allocation: SequenceKvAllocation,
    ) -> Self {
        let model_cache_id = cache.cache_id();
        Self {
            cache,
            model_cache_id,
            release: SequenceKvRelease::LegacyAllocated(allocation),
        }
    }

    pub(super) fn handle(&self) -> Arc<dyn KvCacheHandle> {
        self.cache.clone()
    }

    pub(super) fn legacy_allocation(&self) -> Option<&SequenceKvAllocation> {
        match &self.release {
            SequenceKvRelease::RuntimeManaged => None,
            SequenceKvRelease::LegacyAllocated(allocation) => Some(allocation),
        }
    }

    pub(super) fn model_cache_id(&self) -> &str {
        &self.model_cache_id
    }

    pub(super) fn validate_replacement_cache(&self, cache: &Arc<dyn KvCacheHandle>) -> Result<()> {
        let replacement_cache_id = cache.cache_id();
        if replacement_cache_id != self.model_cache_id() {
            return Err(FerrumError::internal(format!(
                "decode replaced model cache authority {} with {}",
                self.model_cache_id(),
                replacement_cache_id
            )));
        }
        Ok(())
    }

    pub(super) fn replace_cache_handle(&mut self, cache: Arc<dyn KvCacheHandle>) -> Result<()> {
        self.validate_replacement_cache(&cache)?;
        self.cache = cache;
        Ok(())
    }

    pub(super) fn into_physical_resources(self) -> (Option<SequenceKvAllocation>, String) {
        let legacy_allocation = match self.release {
            SequenceKvRelease::RuntimeManaged => None,
            SequenceKvRelease::LegacyAllocated(allocation) => Some(allocation),
        };
        (legacy_allocation, self.model_cache_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SequenceRecurrentAllocation {
    pub(super) slots: Option<usize>,
}

impl SequenceRecurrentAllocation {
    pub(super) fn new(slots: Option<usize>) -> Self {
        Self {
            slots: slots.map(|slots| slots.max(1)),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SequenceRecurrentState {
    pub(super) handle: Arc<dyn RecurrentStateHandle>,
    pub(super) slots: Option<usize>,
}

impl SequenceRecurrentState {
    pub(super) fn new(handle: Arc<dyn RecurrentStateHandle>, slots: Option<usize>) -> Self {
        Self {
            handle,
            slots: slots.map(|slots| slots.max(1)),
        }
    }

    pub(super) fn handle(&self) -> Arc<dyn RecurrentStateHandle> {
        self.handle.clone()
    }

    pub(super) fn allocation(self) -> SequenceRecurrentAllocation {
        SequenceRecurrentAllocation::new(self.slots)
    }
}

#[derive(Debug, Clone)]
pub(super) struct SequenceDraftKvState {
    pub(super) cache: Arc<dyn KvCacheHandle>,
    pub(super) request_id: RequestId,
    pub(super) resource_blocks: usize,
}

impl SequenceDraftKvState {
    pub(super) fn new(
        cache: Arc<dyn KvCacheHandle>,
        request_id: RequestId,
        resource_blocks: usize,
    ) -> Self {
        Self {
            cache,
            request_id,
            resource_blocks: resource_blocks.max(1),
        }
    }

    pub(super) fn allocation(self) -> SequenceKvAllocation {
        SequenceKvAllocation::new(self.request_id, self.resource_blocks)
    }
}

#[derive(Debug, Default)]
pub(super) struct SequencePhysicalResources {
    pub(super) legacy_kv_allocation: Option<SequenceKvAllocation>,
    pub(super) legacy_draft_kv_allocation: Option<SequenceKvAllocation>,
    pub(super) recurrent_state_allocation: Option<SequenceRecurrentAllocation>,
    pub(super) model_cache_id: Option<String>,
}

#[cfg(test)]
impl SequencePhysicalResources {
    pub(super) fn model_cache_id(&self) -> Option<&str> {
        self.model_cache_id.as_deref()
    }
}

#[derive(Debug, Default)]
pub(super) struct SequenceCompletionResources {
    pub(super) physical: SequencePhysicalResources,
    pub(super) request_slot: Option<RequestSlotLease>,
}

#[derive(Debug, Default)]
#[must_use = "unified prefill owned resources must be released or committed"]
pub(super) struct UnifiedPrefillOwnedResources {
    pub(super) legacy_kv_allocation: Option<SequenceKvAllocation>,
    pub(super) recurrent_state_allocation: Option<SequenceRecurrentAllocation>,
}

impl UnifiedPrefillOwnedResources {
    pub(super) fn with_fresh_kv(mut self, allocation: SequenceKvAllocation) -> Self {
        self.legacy_kv_allocation = Some(allocation);
        self
    }

    pub(super) fn with_fresh_recurrent_state(mut self, slots: usize) -> Self {
        self.recurrent_state_allocation = Some(SequenceRecurrentAllocation::new(Some(slots)));
        self
    }

    pub(super) fn commit(mut self) {
        self.legacy_kv_allocation = None;
        self.recurrent_state_allocation = None;
    }

    pub(super) fn is_empty(&self) -> bool {
        self.legacy_kv_allocation.is_none() && self.recurrent_state_allocation.is_none()
    }

    pub(super) async fn release(mut self, engine: &EngineInner, owner_request_id: &RequestId) {
        if let Some(kv_allocation) = self.legacy_kv_allocation.take() {
            engine
                .release_kv_allocation(
                    owner_request_id,
                    kv_allocation.request_id,
                    kv_allocation.blocks,
                )
                .await;
        }
        if let Some(recurrent_allocation) = self.recurrent_state_allocation.take() {
            let sequence_slots = engine
                .sequences
                .write()
                .get_mut(owner_request_id)
                .and_then(SequenceState::take_recurrent_state_allocation);
            if sequence_slots != recurrent_allocation.slots {
                warn!(
                    request_id = %owner_request_id,
                    sequence_slots = ?sequence_slots,
                    owned_slots = ?recurrent_allocation.slots,
                    "unified prefill recurrent ownership metadata differed from sequence state"
                );
            }
            engine
                .release_recurrent_allocation(
                    owner_request_id,
                    recurrent_allocation.slots.or(sequence_slots),
                )
                .await;
        }
    }
}

impl Drop for UnifiedPrefillOwnedResources {
    fn drop(&mut self) {
        if self.is_empty() {
            return;
        }
        let message = "unified prefill resources dropped without explicit release or commit";
        warn!(
            legacy_kv_allocation = ?self.legacy_kv_allocation,
            recurrent_state_allocation = ?self.recurrent_state_allocation,
            "{message}"
        );
        #[cfg(test)]
        if !std::thread::panicking() {
            panic!("{message}");
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SequenceDecodeResources {
    pub(super) seq_id: String,
    pub(super) kv_cache: Arc<dyn KvCacheHandle>,
    pub(super) recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    pub(super) last_token: TokenId,
    pub(super) pos_offset: usize,
}

#[derive(Debug, Clone)]
pub(super) struct SequencePrefillResources {
    pub(super) kv_cache: Option<Arc<dyn KvCacheHandle>>,
    pub(super) legacy_kv_allocation: Option<SequenceKvAllocation>,
    pub(super) recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    pub(super) prefill_tokens_processed: usize,
}

#[cfg(test)]
impl SequencePrefillResources {
    pub(super) fn kv_cache_handle(&self) -> Option<Arc<dyn KvCacheHandle>> {
        self.kv_cache.clone()
    }

    pub(super) fn kv_resource_blocks(&self) -> Option<usize> {
        self.legacy_kv_allocation
            .as_ref()
            .map(|allocation| allocation.blocks)
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct ModelCacheRefUpdate {
    pub(super) released: Option<String>,
    pub(super) acquired: Option<String>,
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
        use ferrum_types::ResponseFormat;
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
        let needs_structured_output = !matches!(
            request.sampling_params.response_format,
            ResponseFormat::Text
        );
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
        let (model_eos_token_ids, stop_token_ids, stop_text_seqs) =
            resolve_stop_conditions(&request.sampling_params, tokenizer.as_deref(), ignore_eos);
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
                factory.create_processor(
                    &request.sampling_params.response_format,
                    &request.sampling_params.structured_output_start,
                    request.sampling_params.max_tokens,
                    &stop_token_ids,
                    &stop_text_seqs,
                )
            })
            .transpose()?
            .flatten();
        let request_generated_control_token_texts = request
            .api_request
            .as_ref()
            .map(ferrum_types::ApiRequest::generated_control_token_texts)
            .unwrap_or_default();
        let (forbidden_token_ids, tokenizer_base_vocab_size, allowed_extended_token_ids) =
            resolve_sampling_token_constraints(
                tokenizer.as_ref(),
                &stop_token_ids,
                request_generated_control_token_texts,
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
            model_kv: None,
            recurrent_state: None,
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
            draft_kv: None,
            token_frequencies: HashMap::new(),
            stop_token_ids,
            model_eos_token_ids,
            response_completion_state,
            forbidden_token_ids,
            initial_forbidden_token_ids,
            tokenizer_base_vocab_size,
            allowed_extended_token_ids,
            stop_text_seqs,
            argmax_token_mask,
            initial_argmax_token_mask,
            streamed_text_len: 0,
        })
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    pub(super) fn record_generated_token_commit(&mut self) {
        if let Some(timing) = &mut self.token_timing {
            timing.record_commit(self.start_time);
        }
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

    pub(super) fn take_physical_resources(&mut self) -> SequencePhysicalResources {
        let (legacy_kv_allocation, model_cache_id) = self
            .model_kv
            .take()
            .map(|state| {
                let (legacy_kv_allocation, model_cache_id) = state.into_physical_resources();
                (legacy_kv_allocation, Some(model_cache_id))
            })
            .unwrap_or((None, None));
        let legacy_draft_kv_allocation = self.draft_kv.take().map(SequenceDraftKvState::allocation);
        let resources = SequencePhysicalResources {
            legacy_kv_allocation,
            legacy_draft_kv_allocation,
            recurrent_state_allocation: self
                .recurrent_state
                .take()
                .map(SequenceRecurrentState::allocation),
            model_cache_id,
        };
        resources
    }

    pub(super) fn take_physical_resources_for_recompute(&mut self) -> SequencePhysicalResources {
        let resources = self.take_physical_resources();
        self.prefill_complete = false;
        self.prefill_tokens_processed = 0;
        self.phase = RequestPhase::Waiting;
        self.tokens_this_iteration = 0;
        resources
    }

    pub(super) fn take_completion_resources(&mut self) -> SequenceCompletionResources {
        SequenceCompletionResources {
            physical: self.take_physical_resources(),
            request_slot: self.request_slot.take(),
        }
    }

    pub(super) fn model_cache_ref_update_for(&self, cache_id: &str) -> ModelCacheRefUpdate {
        if self
            .model_kv
            .as_ref()
            .is_some_and(|state| state.model_cache_id() == cache_id)
        {
            return ModelCacheRefUpdate::default();
        }
        let released = self
            .model_kv
            .as_ref()
            .map(|state| state.model_cache_id().to_string());
        ModelCacheRefUpdate {
            released,
            acquired: Some(cache_id.to_string()),
        }
    }

    pub(super) fn install_model_kv_state(
        &mut self,
        state: SequenceModelKvState,
    ) -> ModelCacheRefUpdate {
        let model_cache_id = state.model_cache_id().to_string();
        let model_cache_update = self.model_cache_ref_update_for(&model_cache_id);
        self.model_kv = Some(state);
        model_cache_update
    }

    pub(super) fn install_runtime_managed_model_kv(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
    ) -> ModelCacheRefUpdate {
        self.install_model_kv_state(SequenceModelKvState::runtime_managed(kv_cache))
    }

    pub(super) fn install_legacy_allocated_model_kv(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        allocation: SequenceKvAllocation,
    ) -> ModelCacheRefUpdate {
        self.install_model_kv_state(SequenceModelKvState::legacy_allocated(kv_cache, allocation))
    }

    pub(super) fn commit_cached_prefill_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        prefill_tokens_processed: usize,
    ) -> ModelCacheRefUpdate {
        let model_cache_update = self.install_runtime_managed_model_kv(kv_cache);
        self.prefill_tokens_processed = prefill_tokens_processed;
        self.prefill_complete = true;
        self.phase = RequestPhase::Decoding;
        self.record_decode_ready();
        model_cache_update
    }

    pub(super) fn commit_prefill_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        kv_resource_blocks: usize,
        recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
        recurrent_state_slots: Option<usize>,
    ) -> ModelCacheRefUpdate {
        let allocation = SequenceKvAllocation::new(self.request_id.clone(), kv_resource_blocks);
        let model_cache_update = self.install_legacy_allocated_model_kv(kv_cache, allocation);
        self.recurrent_state =
            recurrent_state.map(|state| SequenceRecurrentState::new(state, recurrent_state_slots));
        self.prefill_complete = true;
        self.phase = RequestPhase::Decoding;
        self.record_decode_ready();
        model_cache_update
    }

    pub(super) fn commit_plan_runtime_prefill_chunk_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        prefill_tokens_processed: usize,
        is_final_chunk: bool,
    ) -> ModelCacheRefUpdate {
        let model_cache_update = self.install_runtime_managed_model_kv(kv_cache);
        self.recurrent_state = None;
        self.prefill_tokens_processed = prefill_tokens_processed;
        self.prefill_complete = is_final_chunk;
        self.phase = if is_final_chunk {
            RequestPhase::Decoding
        } else {
            RequestPhase::Prefilling
        };
        if is_final_chunk {
            self.record_decode_ready();
        }
        model_cache_update
    }

    pub(super) fn commit_prefill_chunk_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        legacy_kv_allocation: SequenceKvAllocation,
        recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
        prefill_tokens_processed: usize,
        is_final_chunk: bool,
    ) -> ModelCacheRefUpdate {
        let model_cache_update =
            self.install_legacy_allocated_model_kv(kv_cache, legacy_kv_allocation);
        let existing_slots = self.recurrent_state.as_ref().and_then(|state| state.slots);
        self.recurrent_state =
            recurrent_state.map(|state| SequenceRecurrentState::new(state, existing_slots));
        self.prefill_tokens_processed = prefill_tokens_processed;
        self.prefill_complete = is_final_chunk;
        self.phase = if is_final_chunk {
            RequestPhase::Decoding
        } else {
            RequestPhase::Prefilling
        };
        if is_final_chunk {
            self.record_decode_ready();
        }
        model_cache_update
    }

    pub(super) fn decode_model_cache_id_or_request_id(&self, request_id: &RequestId) -> String {
        self.model_cache_id()
            .map(str::to_string)
            .unwrap_or_else(|| request_id.to_string())
    }

    pub(super) fn decode_model_kv_len_after_last_generated_token(&self) -> usize {
        self.input_tokens
            .len()
            .saturating_add(self.generated_tokens.len())
            .saturating_sub(1)
    }

    pub(super) fn decode_resources(
        &self,
        request_id: &RequestId,
    ) -> Option<SequenceDecodeResources> {
        Some(SequenceDecodeResources {
            seq_id: self.decode_model_cache_id_or_request_id(request_id),
            kv_cache: self.model_kv.as_ref()?.handle(),
            recurrent_state: self
                .recurrent_state
                .as_ref()
                .map(SequenceRecurrentState::handle),
            last_token: self
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(TokenId::new(0)),
            pos_offset: self.decode_model_kv_len_after_last_generated_token(),
        })
    }

    pub(super) fn ready_decode_resources(
        &self,
        request_id: &RequestId,
    ) -> Option<SequenceDecodeResources> {
        if !self.prefill_complete || self.generated_tokens.is_empty() {
            return None;
        }
        self.decode_resources(request_id)
    }

    pub(super) fn is_preemptible_decode_candidate(&self) -> bool {
        self.prefill_complete && self.model_kv.is_some()
    }

    pub(super) fn prefill_resources(&self) -> SequencePrefillResources {
        SequencePrefillResources {
            kv_cache: self.model_kv.as_ref().map(SequenceModelKvState::handle),
            legacy_kv_allocation: self
                .model_kv
                .as_ref()
                .and_then(SequenceModelKvState::legacy_allocation)
                .cloned(),
            recurrent_state: self
                .recurrent_state
                .as_ref()
                .map(SequenceRecurrentState::handle),
            prefill_tokens_processed: self.prefill_tokens_processed,
        }
    }

    pub(super) fn recurrent_state_handle(&self) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.recurrent_state
            .as_ref()
            .map(SequenceRecurrentState::handle)
    }

    pub(super) fn recurrent_state_slots(&self) -> Option<usize> {
        self.recurrent_state.as_ref().and_then(|state| state.slots)
    }

    pub(super) fn draft_kv_cache_handle(&self) -> Option<Arc<dyn KvCacheHandle>> {
        self.draft_kv.as_ref().map(|draft| draft.cache.clone())
    }

    pub(super) fn kv_cache_handle(&self) -> Option<Arc<dyn KvCacheHandle>> {
        self.model_kv.as_ref().map(SequenceModelKvState::handle)
    }

    pub(super) fn kv_resource_blocks(&self) -> Option<usize> {
        self.model_kv
            .as_ref()
            .and_then(SequenceModelKvState::legacy_allocation)
            .map(|allocation| allocation.blocks)
    }

    pub(super) fn model_cache_id(&self) -> Option<&str> {
        self.model_kv
            .as_ref()
            .map(SequenceModelKvState::model_cache_id)
    }

    #[cfg(test)]
    pub(super) fn clear_model_kv_for_test(&mut self) {
        self.model_kv = None;
    }

    pub(super) fn commit_decode_step_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
    ) -> Result<()> {
        self.model_kv
            .as_mut()
            .ok_or_else(|| {
                FerrumError::internal("decode completed without an active model KV lease")
            })?
            .replace_cache_handle(kv_cache)?;
        self.tokens_this_iteration += 1;
        Ok(())
    }

    pub(super) fn commit_decode_recurrent_state(
        &mut self,
        recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    ) {
        let existing_slots = self.recurrent_state.as_ref().and_then(|state| state.slots);
        self.recurrent_state =
            recurrent_state.map(|state| SequenceRecurrentState::new(state, existing_slots));
    }

    pub(super) fn commit_recurrent_state_admission(
        &mut self,
        recurrent_state: Arc<dyn RecurrentStateHandle>,
        slots: usize,
    ) {
        self.recurrent_state = Some(SequenceRecurrentState::new(recurrent_state, Some(slots)));
    }

    pub(super) fn take_recurrent_state_allocation(&mut self) -> Option<usize> {
        self.recurrent_state.take().and_then(|state| state.slots)
    }

    pub(super) fn commit_speculative_decode_physical_resources(
        &mut self,
        target_kv_cache: Arc<dyn KvCacheHandle>,
        draft_kv_cache: Arc<dyn KvCacheHandle>,
    ) -> Result<()> {
        self.model_kv
            .as_ref()
            .ok_or_else(|| {
                FerrumError::internal(
                    "speculative decode completed without an active target KV lease",
                )
            })?
            .validate_replacement_cache(&target_kv_cache)?;
        if let Some(draft) = &self.draft_kv {
            let replacement_cache_id = draft_kv_cache.cache_id();
            if replacement_cache_id != draft.cache.cache_id() {
                return Err(FerrumError::internal(format!(
                    "speculative decode replaced draft cache authority {} with {}",
                    draft.cache.cache_id(),
                    replacement_cache_id
                )));
            }
        } else {
            return Err(FerrumError::internal(
                "draft KV cache updated without owned allocation metadata",
            ));
        }
        self.model_kv
            .as_mut()
            .expect("validated target KV lease remains installed")
            .replace_cache_handle(target_kv_cache)?;
        self.draft_kv
            .as_mut()
            .expect("validated draft KV lease remains installed")
            .cache = draft_kv_cache;
        Ok(())
    }

    pub(super) fn commit_draft_kv_allocation(
        &mut self,
        draft_kv_cache: Arc<dyn KvCacheHandle>,
        draft_request_id: RequestId,
        draft_resource_blocks: usize,
    ) {
        self.draft_kv = Some(SequenceDraftKvState::new(
            draft_kv_cache,
            draft_request_id,
            draft_resource_blocks,
        ));
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
            && self.structured_output_processor.is_none()
            && matches!(params.response_format, ResponseFormat::Text)
    }

    pub(super) fn supports_raw_speculative_decode(&self) -> bool {
        self.structured_output_processor.is_none()
            && self.sampling_plan.supports_raw_greedy_speculation()
            && self
                .argmax_token_mask
                .as_ref()
                .is_none_or(|mask| mask.valid_token_mask.iter().all(|&value| value != 0))
    }

    pub(super) fn model_decode_repetition_penalty(&self) -> Option<GreedyRepetitionPenalty> {
        let penalty = self.sampling_params.repetition_penalty;
        if penalty == 1.0 || self.generated_tokens.is_empty() {
            return None;
        }
        let token_ids: Vec<u32> = self
            .generated_tokens
            .iter()
            .map(|token| token.get())
            .collect();
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

    pub fn accept_model_greedy_argmax_token(
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
            self.streamed_text_len,
            token,
            None,
        ) {
            return Err(FerrumError::model(format!(
                "model greedy argmax token decoded to forbidden output ({})",
                token_detail()
            )));
        }

        self.accept_response_completion_token(tokenizer, token)?;
        *self.token_frequencies.entry(token).or_insert(0) += 1;
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

    pub fn reset_guided_processors(&self) -> Result<()> {
        if let Some(processor) = &self.structured_output_processor {
            processor.reset()?;
        }
        Ok(())
    }

    /// Return the reason this sequence should stop, if any.
    ///
    /// Checks: (1) last generated token is in the resolved `stop_token_ids`
    /// set (model EOS + any single-token `stop_sequences`), (2) decoded text
    /// contains a multi-token user stop sequence, (3) max-tokens budget is
    /// exhausted. Text-stop decoding only runs for requests that supplied a
    /// multi-token stop string, so the common EOS path stays cheap.
    pub fn stop_reason(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ) -> Option<FinishReason> {
        if let Some(&last_token) = self.generated_tokens.last() {
            if self.stop_token_ids.contains(&last_token.get()) {
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
        if self.generated_tokens.len() >= self.sampling_params.max_tokens {
            return Some(FinishReason::Length);
        }
        None
    }

    /// Cheap stop check for tests and callers that do not have tokenizer
    /// access. Engine hot paths use `stop_reason` through `EngineInner`.
    pub fn should_stop(&self) -> bool {
        self.stop_reason(None).is_some()
    }

    /// Sample next token with full processor chain (temperature, top-k/p,
    /// repetition penalty and tokenizer-aware structured-output mask).
    pub fn sample_with_processors(&mut self, logits: &mut [f32]) -> Result<TokenId> {
        self.sample_with_processors_with_tokenizer(logits, None)
    }

    pub fn sample_with_processors_with_tokenizer(
        &mut self,
        logits: &mut [f32],
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ) -> Result<TokenId> {
        use ferrum_interfaces::sampler::SamplingContext;

        // The grammar mask runs first. There is deliberately no invalid-token
        // fallback: a dead grammar state is a request error, not permission to
        // return malformed JSON.
        let mut required_structured_delimiter_token_id = None;
        if let Some(processor) = &self.structured_output_processor {
            let constraint = processor.mask_logits_with_terminals(
                logits,
                &self.generated_tokens,
                &self.stop_token_ids,
                &self.allowed_extended_token_ids,
            )?;
            required_structured_delimiter_token_id = constraint.required_delimiter_token_id;
            if !constraint.accepting {
                mask_stop_token_logits(logits, &self.stop_token_ids);
            }
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
        let previous_streamed_text_len = self.streamed_text_len;
        let token = {
            let mut ctx = SamplingContext::new(
                step,
                &self.sampling_params,
                logits,
                &self.generated_tokens,
                &self.token_frequencies,
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

        self.accept_response_completion_token(tokenizer, token)?;
        // Update frequency tracking
        *self.token_frequencies.entry(token).or_insert(0) += 1;

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
                has_kv_cache = self.model_kv.is_some(),
                kv_resource_blocks = ?self.kv_resource_blocks(),
                has_recurrent_state = self.recurrent_state.is_some(),
                recurrent_state_slots = ?self.recurrent_state_slots(),
                has_draft_kv = self.draft_kv.is_some(),
                draft_kv_resource_blocks = ?self.draft_kv.as_ref().map(|draft| draft.resource_blocks),
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

#[must_use = "KV allocation leases must be consumed by release().await or into_committed_parts()"]
pub(super) struct KvAllocationLease {
    pub(super) owner_request_id: RequestId,
    pub(super) allocation_request_id: RequestId,
    pub(super) handle: Arc<dyn KvCacheHandle>,
    pub(super) blocks: usize,
    pub(super) armed: bool,
}

impl KvAllocationLease {
    pub(super) fn new(
        owner_request_id: RequestId,
        allocation_request_id: RequestId,
        handle: Arc<dyn KvCacheHandle>,
        blocks: usize,
    ) -> Self {
        Self {
            owner_request_id,
            allocation_request_id,
            handle,
            blocks,
            armed: true,
        }
    }

    pub(super) fn handle(&self) -> Arc<dyn KvCacheHandle> {
        self.handle.clone()
    }

    pub(super) fn blocks(&self) -> usize {
        self.blocks
    }

    pub(super) async fn release(mut self, engine: &EngineInner) {
        engine
            .release_kv_allocation(
                &self.owner_request_id,
                self.allocation_request_id.clone(),
                self.blocks,
            )
            .await;
        self.armed = false;
    }

    pub(super) fn into_committed_parts(mut self) -> (RequestId, usize) {
        self.armed = false;
        (self.allocation_request_id.clone(), self.blocks)
    }
}

impl Drop for KvAllocationLease {
    fn drop(&mut self) {
        if self.armed {
            let message = "KV allocation lease dropped without explicit commit or async release";
            warn!(
                owner_request_id = %self.owner_request_id,
                allocation_request_id = %self.allocation_request_id,
                blocks = self.blocks,
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}

#[must_use = "recurrent-state leases must be consumed by release().await or commit()"]
pub(super) struct RecurrentStateLease {
    pub(super) request_id: RequestId,
    pub(super) handle: Arc<dyn RecurrentStateHandle>,
    pub(super) slots: usize,
    pub(super) capacity: Option<usize>,
    pub(super) armed: bool,
}

impl RecurrentStateLease {
    pub(super) fn new(
        request_id: RequestId,
        handle: Arc<dyn RecurrentStateHandle>,
        slots: usize,
        capacity: Option<usize>,
    ) -> Self {
        Self {
            request_id,
            handle,
            slots,
            capacity,
            armed: true,
        }
    }

    pub(super) fn handle(&self) -> Arc<dyn RecurrentStateHandle> {
        self.handle.clone()
    }

    pub(super) fn slots(&self) -> usize {
        self.slots
    }

    pub(super) async fn release(mut self, engine: &EngineInner) {
        engine
            .release_recurrent_allocation(&self.request_id, Some(self.slots))
            .await;
        self.armed = false;
    }

    pub(super) fn commit(mut self) -> usize {
        self.armed = false;
        self.slots
    }
}

impl Drop for RecurrentStateLease {
    fn drop(&mut self) {
        if self.armed {
            let message = "recurrent-state lease dropped without explicit commit or async release";
            warn!(
                request_id = %self.request_id,
                slots = self.slots,
                capacity = ?self.capacity,
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}

pub(super) struct RecurrentStateAdmission {
    pub(super) handle: Option<Arc<dyn RecurrentStateHandle>>,
    pub(super) lease: Option<RecurrentStateLease>,
}

impl RecurrentStateAdmission {
    pub(super) fn none() -> Self {
        Self {
            handle: None,
            lease: None,
        }
    }

    pub(super) fn existing(handle: Arc<dyn RecurrentStateHandle>) -> Self {
        Self {
            handle: Some(handle),
            lease: None,
        }
    }

    pub(super) fn fresh(lease: RecurrentStateLease) -> Self {
        Self {
            handle: Some(lease.handle()),
            lease: Some(lease),
        }
    }

    pub(super) fn handle(&self) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.handle.clone()
    }

    pub(super) fn fresh_slots(&self) -> Option<usize> {
        self.lease.as_ref().map(RecurrentStateLease::slots)
    }

    pub(super) fn commit_fresh(&mut self) -> Option<usize> {
        self.lease.take().map(RecurrentStateLease::commit)
    }

    pub(super) async fn release_fresh(&mut self, engine: &EngineInner) {
        if let Some(lease) = self.lease.take() {
            lease.release(engine).await;
        }
    }
}

#[must_use = "legacy backend workspace trace leases must be released or dropped"]
pub(super) struct LegacyBackendWorkspaceTraceLease<'a> {
    pub(super) engine: &'a EngineInner,
    pub(super) request_ids: Vec<RequestId>,
    pub(super) release_phase: &'static str,
    pub(super) armed: bool,
}

impl<'a> LegacyBackendWorkspaceTraceLease<'a> {
    pub(super) fn new(
        engine: &'a EngineInner,
        request_ids: Vec<RequestId>,
        phase_prefix: &'static str,
        release_phase: &'static str,
    ) -> Self {
        engine.trace_legacy_backend_workspace_acquire_many(&request_ids, phase_prefix);
        Self {
            engine,
            request_ids,
            release_phase,
            armed: true,
        }
    }

    pub(super) fn release(mut self) {
        self.release_now();
        self.armed = false;
    }

    pub(super) fn release_now(&self) {
        self.engine
            .trace_legacy_backend_workspace_release_many(&self.request_ids, self.release_phase);
    }
}

impl Drop for LegacyBackendWorkspaceTraceLease<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.release_now();
        }
    }
}

pub(super) struct PendingBatchPrefill {
    pub(super) request_id: RequestId,
    pub(super) input_tokens: Vec<TokenId>,
    pub(super) kv_lease: Option<KvAllocationLease>,
    pub(super) recurrent_state: RecurrentStateAdmission,
    pub(super) metadata: HashMap<String, serde_json::Value>,
    pub(super) can_use_prefix_cache: bool,
}

impl PendingBatchPrefill {
    pub(super) fn new(
        request_id: RequestId,
        input_tokens: Vec<TokenId>,
        kv_lease: KvAllocationLease,
        recurrent_state: RecurrentStateAdmission,
        metadata: HashMap<String, serde_json::Value>,
        can_use_prefix_cache: bool,
    ) -> Self {
        Self {
            request_id,
            input_tokens,
            kv_lease: Some(kv_lease),
            recurrent_state,
            metadata,
            can_use_prefix_cache,
        }
    }

    pub(super) fn kv_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        self.kv_lease
            .as_ref()
            .map(KvAllocationLease::handle)
            .ok_or_else(|| FerrumError::internal("batch prefill KV lease already consumed"))
    }

    pub(super) fn kv_resource_blocks(&self) -> Result<usize> {
        self.kv_lease
            .as_ref()
            .map(KvAllocationLease::blocks)
            .ok_or_else(|| FerrumError::internal("batch prefill KV lease already consumed"))
    }

    pub(super) fn commit_kv(&mut self) -> Result<usize> {
        let lease = self
            .kv_lease
            .take()
            .ok_or_else(|| FerrumError::internal("batch prefill KV lease already consumed"))?;
        let (_allocation_request_id, blocks) = lease.into_committed_parts();
        Ok(blocks)
    }

    pub(super) async fn release_resources(&mut self, engine: &EngineInner) {
        if let Some(lease) = self.kv_lease.take() {
            lease.release(engine).await;
        }
        self.recurrent_state.release_fresh(engine).await;
    }
}
