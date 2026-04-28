//! JSON-Schema guided decoding — end-to-end through the engine with a
//! real byte-level tokenizer.
//!
//! Uses OpenAI's `response_format = json_schema` (the only public surface
//! — we translate the schema to a regex internally and feed it into
//! `RegexGuidedProcessor`). The `ByteTokenizer` here gives every byte its
//! own token, so the DFA mask has real per-token byte transitions (the
//! default `MockTokenizer` returns `None` from `token_text`, which would
//! defeat the mask).
//!
//! We bias the executor toward a non-matching token ('*' = 42) and let
//! the schema-derived mask force the sampler into schema-valid tokens.

use std::sync::Arc;

use async_trait::async_trait;
use ferrum_engine::{ContinuousBatchEngine, InferenceEngineInterface};
use ferrum_interfaces::tokenizer::{ChatMessage, Tokenizer, TokenizerInfo, TokenizerType};
use ferrum_scheduler::implementations::ContinuousBatchScheduler;
use ferrum_testkit::{
    ConfigurableModelExecutor, MockKvCacheManager, MockSampler, MockTensorFactory,
};
use ferrum_types::{
    InferenceRequest, ResponseFormat, Result, SchedulerConfig, SpecialTokens, TokenId,
};

/// One token per byte 0..=126, plus a "stop-like" token at 127. Engine's
/// `should_stop` considers tokens >= (vocab_size - 10) as terminal, so
/// anything 118..=127 ends the sequence.
const VOCAB: usize = 128;
const STOP_TOKEN: u32 = 122; // 'z'

struct ByteTokenizer {
    byte_strings: Vec<String>,
    special: SpecialTokens,
}

impl ByteTokenizer {
    fn new() -> Self {
        let mut byte_strings = Vec::with_capacity(VOCAB);
        for b in 0..VOCAB {
            // For each byte, produce the UTF-8 character if printable; else
            // an empty string. The regex DFA walks by bytes regardless of
            // what the Rust `str` representation looks like — using the
            // single byte converted to a char via UTF-8 covers the ASCII
            // range we need for `[0-9]+`.
            let ch = (b as u8) as char;
            byte_strings.push(ch.to_string());
        }
        Self {
            byte_strings,
            special: SpecialTokens {
                bos_token: Some(TokenId::new(1)),
                eos_token: Some(TokenId::new(STOP_TOKEN)),
                unk_token: Some(TokenId::new(0)),
                pad_token: None,
                sep_token: None,
                cls_token: None,
                mask_token: None,
            },
        }
    }
}

#[async_trait]
impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        if add_special {
            if let Some(bos) = self.special.bos_token {
                out.push(bos);
            }
        }
        for b in text.bytes() {
            if (b as usize) < VOCAB {
                out.push(TokenId::new(b as u32));
            }
        }
        if out.is_empty() {
            out.push(TokenId::new(b' ' as u32));
        }
        Ok(out)
    }
    fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|t| {
                let v = t.get();
                if (v as usize) < VOCAB {
                    Some(v as u8)
                } else {
                    None
                }
            })
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        self.decode(&[next], false)
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        if text.len() == 1 {
            Some(TokenId::new(text.bytes().next().unwrap() as u32))
        } else {
            None
        }
    }
    fn token_text(&self, token_id: TokenId) -> Option<&str> {
        self.byte_strings
            .get(token_id.get() as usize)
            .map(|s| s.as_str())
    }
    fn apply_chat_template(&self, _messages: &[ChatMessage]) -> Result<String> {
        Ok(String::new())
    }
    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::Custom,
            vocab_size: VOCAB,
            special_tokens: self.special.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: Some(1),
            model_name: Some("byte-test".into()),
        }
    }
}

fn make_engine() -> ContinuousBatchEngine {
    let config = ferrum_types::EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(SchedulerConfig::default()));
    let tokenizer = Arc::new(ByteTokenizer::new());
    let sampler = Arc::new(MockSampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(1024));
    let tensor_factory = Arc::new(MockTensorFactory);
    // Bias '*' (42) heavily; emit STOP_TOKEN ('z' = 122) after 3 decode
    // steps. With the regex mask `[0-9]+`, '*' gets -INF and the greedy
    // sampler falls through to the smallest unmasked token = '0' (48).
    let executor = Arc::new(ConfigurableModelExecutor::with_token_sequence(
        VOCAB,
        vec![42, 42, 42, STOP_TOKEN],
    ));
    ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    )
}

fn make_request(prompt: &str, schema_json: &str, max_tokens: usize) -> InferenceRequest {
    let mut req = InferenceRequest::new(prompt, "mock-model");
    req.sampling_params.max_tokens = max_tokens;
    req.sampling_params.temperature = 0.0; // greedy so output is deterministic
    req.sampling_params.response_format = ResponseFormat::JsonSchema(schema_json.to_string());
    req
}

/// `{"type": "integer"}` compiles to `-?\d+`. The executor wants to emit
/// '*' but the mask forbids it; greedy falls through to the smallest
/// unmasked byte which is a digit (48-57).
#[tokio::test]
async fn integer_schema_forces_digit_tokens() {
    let engine = make_engine();
    let req = make_request("hi", r#"{"type":"integer"}"#, 10);
    let resp = engine.infer(req).await.expect("infer");

    // `text` is decoded via ByteTokenizer, so each generated byte is one
    // char in the string. With stop-after-3, we expect 3 digits + 1 stop.
    let generated = resp.text;
    assert!(
        !generated.is_empty(),
        "engine should have generated at least one token"
    );

    // Strip the trailing stop char if present.
    let trimmed: String = generated.chars().filter(|c| c.is_ascii_digit()).collect();
    assert!(
        trimmed.chars().count() >= 1,
        "expected at least one digit in output, got {generated:?}"
    );

    // No letters or '*' allowed — the integer schema's regex `-?\d+`
    // would have rejected them.
    for c in generated.chars() {
        assert!(
            c.is_ascii_digit() || c == '-' || c == (STOP_TOKEN as u8) as char,
            "unexpected char {c:?} (byte {}) in schema-guided output {generated:?}",
            c as u32,
        );
    }
}

/// Sanity check: without response_format, the same executor produces
/// the '*' tokens it was biased toward — confirms the guided path is
/// what forces the switch, not some incidental engine behaviour.
#[tokio::test]
async fn unconstrained_decode_emits_biased_token() {
    let engine = make_engine();
    let mut req = InferenceRequest::new("hi", "mock-model");
    req.sampling_params.max_tokens = 10;
    req.sampling_params.temperature = 0.0;
    // response_format = Text (default)
    let resp = engine.infer(req).await.expect("infer");

    assert!(
        resp.text.contains('*'),
        "without regex, executor should emit '*' tokens, got {:?}",
        resp.text
    );
}
