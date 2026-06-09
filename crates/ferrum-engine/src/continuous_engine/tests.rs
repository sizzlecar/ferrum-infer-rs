use super::*;
use ferrum_interfaces::tokenizer::{TokenizerInfo, TokenizerType};

struct PolicyTokenizer {
    vocab_size: usize,
    special: ferrum_types::SpecialTokens,
    ids: HashMap<String, TokenId>,
    texts: Vec<Option<String>>,
}

impl PolicyTokenizer {
    fn new(vocab_size: usize, pairs: &[(&str, u32)]) -> Self {
        let max_id = pairs.iter().map(|(_, id)| *id as usize).max().unwrap_or(0);
        let mut texts = vec![None; max_id + 1];
        let mut ids = HashMap::new();
        for (text, id) in pairs {
            ids.insert((*text).to_string(), TokenId::new(*id));
            texts[*id as usize] = Some((*text).to_string());
        }
        Self {
            vocab_size,
            special: ferrum_types::SpecialTokens {
                bos_token: Some(TokenId::new(1)),
                eos_token: Some(TokenId::new(3)),
                unk_token: Some(TokenId::new(2)),
                pad_token: Some(TokenId::new(4)),
                sep_token: None,
                cls_token: None,
                mask_token: None,
            },
            ids,
            texts,
        }
    }
}

impl Tokenizer for PolicyTokenizer {
    fn encode(&self, _text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
        Ok(vec![TokenId::new(0)])
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let mut output = String::new();
        let mut pending_bad_byte = false;
        for token in tokens {
            let Some(text) = self.token_text(*token) else {
                continue;
            };
            if skip_special && matches!(text, "<think>") {
                continue;
            }
            match text {
                "byte-fallback" => output.push('\u{FFFD}'),
                "bad-byte-lead" => pending_bad_byte = true,
                "valid-byte-cont" if pending_bad_byte => {
                    output.push('好');
                    pending_bad_byte = false;
                }
                text => {
                    if pending_bad_byte {
                        output.push('\u{FFFD}');
                        pending_bad_byte = false;
                    }
                    output.push_str(text);
                }
            }
        }
        Ok(output)
    }

    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        Ok(self.token_text(next).unwrap_or_default().to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        &self.special
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.ids.get(text).copied()
    }

    fn token_text(&self, token_id: TokenId) -> Option<&str> {
        self.texts
            .get(token_id.get() as usize)
            .and_then(|text| text.as_deref())
    }

    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::Custom,
            vocab_size: self.vocab_size,
            special_tokens: self.special.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: None,
            model_name: Some("policy-tokenizer-test".to_string()),
        }
    }
}

fn policy_request() -> InferenceRequest {
    InferenceRequest {
        id: RequestId::new(),
        prompt: "test".to_string(),
        model_id: ferrum_types::ModelId::new("test"),
        sampling_params: SamplingParams::greedy(),
        stream: false,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: None,
        metadata: HashMap::new(),
    }
}

#[test]
fn continuous_engine_runtime_config_parses_env_snapshot() {
    let cfg = ContinuousEngineRuntimeConfig::from_env_vars(
        Some(64),
        [
            (BATCH_DECODE_PROF_ENV, "1"),
            (CHUNKED_PREFILL_ENV, "128"),
            (KV_CAPACITY_ENV, "2048"),
            (MAX_MODEL_LEN_ENV, "4096"),
            (NEXT_BATCH_PROF_ENV, "1"),
            (WHOLE_PROMPT_PREFIX_CACHE_ENV, "1"),
            (RBD_PROF_ENV, "1"),
            (UNIFIED_POST_PROF_ENV, "1"),
        ],
    );

    assert_eq!(cfg.active_decode_prefill_chunk, Some(64));
    assert!(cfg.batch_decode_prof);
    assert!(cfg.chunked_prefill_present);
    assert_eq!(cfg.chunked_prefill_size, Some(128));
    assert_eq!(cfg.chunked_prefill_size_for(200), Some(128));
    assert_eq!(cfg.chunked_prefill_size_for(128), None);
    assert_eq!(cfg.kv_capacity, Some(2048));
    assert_eq!(cfg.max_model_len, Some(4096));
    assert!(cfg.next_batch_prof);
    assert!(cfg.prefix_cache_enabled);
    assert!(cfg.rbd_prof);
    assert!(cfg.unified_post_prof);
}

#[test]
fn continuous_engine_runtime_config_keeps_invalid_chunk_presence() {
    let cfg = ContinuousEngineRuntimeConfig::from_env_vars(
        None,
        [
            (CHUNKED_PREFILL_ENV, "invalid"),
            (WHOLE_PROMPT_PREFIX_CACHE_ENV, "0"),
        ],
    );

    assert!(cfg.chunked_prefill_present);
    assert_eq!(cfg.chunked_prefill_size, None);
    assert_eq!(cfg.chunked_prefill_size_for(200), None);
    assert!(!cfg.prefix_cache_enabled);
}

#[test]
fn performance_breakdown_reports_engine_timing_counters() {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
        Arc::new(ferrum_testkit::MockKvCacheManager::new(256));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(ferrum_testkit::MockTensorFactory);
    let model_executor: Arc<dyn ModelExecutor + Send + Sync> =
        Arc::new(ferrum_testkit::MockModelExecutor::instant(128));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        model_executor,
        tensor_factory,
    );

    engine
        .inner
        .record_scheduling_time(Duration::from_micros(1500));
    engine
        .inner
        .record_scheduling_time(Duration::from_micros(2500));
    engine
        .inner
        .record_model_execution_time(Duration::from_micros(10_000));
    engine
        .inner
        .record_model_execution_time(Duration::from_micros(14_000));
    engine
        .inner
        .record_iteration_lock_wait(Duration::from_micros(300));
    engine
        .inner
        .record_iteration_lock_wait(Duration::from_micros(700));

    let breakdown = engine.metrics().performance_breakdown;
    assert_eq!(breakdown.scheduling_time_ms, 2.0);
    assert_eq!(breakdown.model_execution_time_ms, 12.0);
    assert_eq!(breakdown.other_overhead_time_ms, 0.5);
}

#[test]
fn request_context_capacity_uses_executor_kv_capacity_when_smaller() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 2048;
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());

    assert_eq!(
        effective_request_context_capacity(&config, &runtime, Some(512)),
        Some(512)
    );
}

#[test]
fn test_sequence_state() {
    let request = InferenceRequest {
        id: RequestId::new(),
        prompt: "test".to_string(),
        model_id: ferrum_types::ModelId::new("test"),
        sampling_params: SamplingParams::default(),
        stream: false,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: None,
        metadata: HashMap::new(),
    };

    let tokens = vec![TokenId::new(1), TokenId::new(2)];
    let state = SequenceState::new(request, tokens);

    assert_eq!(state.phase, RequestPhase::Waiting);
    assert_eq!(state.total_tokens(), 2);
    assert!(!state.prefill_complete);
}

#[test]
fn sequence_state_detects_text_stop_before_length() {
    let tokenizer = PolicyTokenizer::new(8, &[("OK", 5), ("<END>", 6), ("TAIL", 7)]);
    let mut request = policy_request();
    request.sampling_params.max_tokens = 3;
    let mut state = SequenceState::new(request, vec![TokenId::new(0)]);
    state.generated_tokens = vec![TokenId::new(5), TokenId::new(6), TokenId::new(7)];
    state.stop_text_seqs = vec!["<END>".to_string()];

    assert_eq!(
        state.stop_reason(Some(&tokenizer)),
        Some(FinishReason::Stop)
    );
}

#[test]
fn model_decode_metadata_marks_structured_requests_for_full_logits() {
    let plain = SequenceState::new(policy_request(), vec![TokenId::new(0)]);
    assert_eq!(
        plain
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        None
    );

    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    let structured = SequenceState::new(request, vec![TokenId::new(0)]);
    assert_eq!(
        structured
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        Some(true)
    );

    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let json_schema_without_tokenizer = SequenceState::new(request, vec![TokenId::new(0)]);
    assert_eq!(
        json_schema_without_tokenizer
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn model_decode_metadata_marks_sampling_masks_for_full_logits() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));

    assert_eq!(
        state
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn schema_guided_sampling_masks_extended_stop_tokens_before_accept() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<|eot_id|>", 8),
        ],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

    assert!(state.regex_processor.is_some());
    assert!(
        state.stop_token_ids.contains(&8),
        "common eot token should be a resolved stop token"
    );

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[0] = 1.0;
    logits[1] = 0.5;
    logits[8] = 100.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 0);
    assert!(
        logits[8].is_infinite() && logits[8].is_sign_negative(),
        "schema-guided generation must not sample eot before the schema accepts"
    );
}

#[test]
fn schema_guided_sampling_masks_extended_control_tokens_before_accept() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<think>", 7),
            ("<|eot_id|>", 8),
        ],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

    assert!(state.regex_processor.is_some());
    assert!(
        state.allowed_extended_token_ids.contains(&7),
        "think token should be an allowed generated control token outside base vocab"
    );
    assert!(
        !state.stop_token_ids.contains(&7),
        "think token should not be treated as a terminator"
    );

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[0] = 1.0;
    logits[1] = 0.5;
    logits[7] = 100.0;
    logits[8] = 90.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 0);
    assert!(
        logits[7].is_infinite() && logits[7].is_sign_negative(),
        "schema-guided generation must not sample invisible control tokens before accept"
    );
    assert!(
        logits[8].is_infinite() && logits[8].is_sign_negative(),
        "schema-guided generation must not sample stop tokens before accept"
    );
}

#[test]
fn schema_guided_sampling_allows_extended_stop_after_accept() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<think>", 7),
            ("<|eot_id|>", 8),
        ],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format =
        ferrum_types::ResponseFormat::JsonSchema(r#"{"enum":["x"]}"#.to_string());
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    state.generated_tokens = vec![TokenId::new(5), TokenId::new(2), TokenId::new(5)];

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[1] = 80.0;
    logits[7] = 100.0;
    logits[8] = 90.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 8);
    assert!(
        logits[7].is_infinite() && logits[7].is_sign_negative(),
        "completed schema output should still reject non-stop control tokens"
    );
    assert!(
        logits[8].is_finite(),
        "completed schema output should allow the resolved stop token"
    );
}

#[test]
fn sample_masks_unknown_pad_reserved_and_bos_tokens() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        10,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("[PAD151935]", 4),
            ("<|reserved_special_token_0|>", 5),
            ("ok", 6),
            ("other", 7),
            ("byte-fallback", 8),
            ("\u{00ef}\u{00bf}\u{00bd}", 9),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    let mut logits = vec![0.0f32; 10];
    logits[1] = 100.0;
    logits[2] = 99.0;
    logits[4] = 98.0;
    logits[5] = 97.0;
    logits[8] = 96.0;
    logits[9] = 95.0;
    logits[6] = 1.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 6);
    for token_id in [1usize, 2, 4, 5, 8, 9] {
        assert_eq!(logits[token_id], f32::NEG_INFINITY);
    }
}

#[test]
fn sample_resamples_candidate_that_would_flush_replacement_char() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        8,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("bad-byte-lead", 5),
            ("ok", 6),
            ("valid-byte-cont", 7),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(5));

    let mut logits = vec![0.0f32; 8];
    logits[6] = 100.0;
    logits[7] = 1.0;

    let token = state
        .sample_with_processors_with_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(token.get(), 7);
    assert_eq!(logits[6], f32::NEG_INFINITY);
}

#[test]
fn sample_candidate_checks_from_streamed_text_boundary() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        7,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("byte-fallback", 5),
            ("ok", 6),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(5));
    state.streamed_text_len = 0;

    assert!(state.sample_candidate_decodes_to_forbidden_output(
        Some(tokenizer.as_ref()),
        state.streamed_text_len,
        TokenId::new(6),
    ));
}

#[test]
fn sample_allows_generated_control_tokens_above_base_vocab() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("</think>", 5),
            ("[PAD151935]", 6),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    let mut logits = vec![0.0f32; 7];
    logits[4] = 1.0;
    logits[5] = 90.0;
    logits[6] = 100.0;

    let token = state.sample_with_processors(&mut logits).unwrap();

    assert_eq!(token.get(), 5);
    assert_eq!(logits[5], 90.0);
    assert_eq!(logits[6], f32::NEG_INFINITY);
}

#[test]
fn sample_resamples_hidden_non_stop_control_tokens_above_base_vocab() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("x", 5),
            ("<think>", 7),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(4));
    state.streamed_text_len = tokenizer
        .decode(&state.generated_tokens, true)
        .expect("generated prefix decodes")
        .len();

    assert!(
        state.allowed_extended_token_ids.contains(&7),
        "think token should be whitelisted as a generated control token"
    );
    assert!(
        !state.stop_token_ids.contains(&7),
        "think token should not be treated as a stop token"
    );

    let mut logits = vec![f32::NEG_INFINITY; 8];
    logits[5] = 1.0;
    logits[7] = 100.0;

    let token = state
        .sample_with_processors_with_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(token.get(), 5);
    assert_eq!(logits[7], f32::NEG_INFINITY);
}

#[test]
fn sample_masks_metadata_initial_token_text_only_before_first_generation() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("</think>", 5),
        ],
    ));
    let mut request = policy_request();
    request.metadata.insert(
        "ferrum_initial_forbidden_token_texts".to_string(),
        serde_json::json!(["</think>"]),
    );
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

    let mut first_logits = vec![0.0f32; 6];
    first_logits[0] = 1.0;
    first_logits[5] = 100.0;
    let first = state.sample_with_processors(&mut first_logits).unwrap();
    assert_eq!(first.get(), 0);
    assert_eq!(first_logits[5], f32::NEG_INFINITY);

    state.generated_tokens.push(first);
    let mut next_logits = vec![0.0f32; 6];
    next_logits[0] = 1.0;
    next_logits[5] = 100.0;
    let next = state.sample_with_processors(&mut next_logits).unwrap();
    assert_eq!(next.get(), 5);
    assert_eq!(next_logits[5], 100.0);
}
