//! Tokenizer-aware hard constraints for structured output.
//!
//! A factory owns the tokenizer trie and grammar compiler and is shared by an
//! engine. Each request gets an independent matcher with no shared mutable
//! grammar state.

use std::{
    collections::{HashMap, HashSet},
    str,
    sync::Arc,
};

use ferrum_interfaces::tokenizer::Tokenizer;
use ferrum_types::{FerrumError, ResponseFormat, Result, StructuredOutputStart, TokenId};
use llguidance::{
    api::TopLevelGrammar,
    toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenizerEnv},
    Matcher, ParserFactory,
};
use parking_lot::Mutex;
use serde_json::json;

const MAX_CACHED_GRAMMARS: usize = 64;

/// Shared, immutable tokenizer and grammar compilation state.
pub struct StructuredOutputFactory {
    parser_factory: ParserFactory,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    vocab_size: usize,
    grammar_templates: Mutex<HashMap<String, Matcher>>,
}

impl std::fmt::Debug for StructuredOutputFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuredOutputFactory")
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl StructuredOutputFactory {
    /// Build the tokenizer trie once for this engine.
    pub fn new(tokenizer: Arc<dyn Tokenizer + Send + Sync>) -> Result<Self> {
        Self::new_with_model_vocab_size(tokenizer, None)
    }

    /// Build against the executor's logits width when it is larger than the
    /// tokenizer base vocabulary (for example added EOS/control tokens).
    pub fn new_with_model_vocab_size(
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        model_vocab_size: Option<usize>,
    ) -> Result<Self> {
        let eos = tokenizer.special_tokens().eos_token.ok_or_else(|| {
            FerrumError::config("structured output requires a tokenizer EOS token")
        })?;
        let vocab_size = model_vocab_size
            .unwrap_or_else(|| tokenizer.vocab_size())
            .max(tokenizer.vocab_size());
        if vocab_size == 0 || eos.get() as usize >= vocab_size {
            return Err(FerrumError::config(format!(
                "structured output tokenizer has invalid vocab/EOS: vocab_size={vocab_size}, eos={}",
                eos.get()
            )));
        }

        let special_ids = tokenizer_special_ids(tokenizer.as_ref());
        let token_bytes = (0..vocab_size)
            .map(|idx| {
                let token = TokenId::new(idx as u32);
                if special_ids.contains(&token.get()) {
                    special_token_marker(token)
                } else {
                    tokenizer
                        .decode(&[token], false)
                        .ok()
                        .or_else(|| tokenizer.token_text(token).map(str::to_owned))
                        .unwrap_or_default()
                        .into_bytes()
                }
            })
            .collect::<Vec<_>>();

        let mut eos_tokens = vec![eos.get()];
        eos_tokens.extend(
            tokenizer
                .special_tokens()
                .extra_eos_tokens
                .iter()
                .map(|token| token.get())
                .filter(|token| *token < vocab_size as u32),
        );
        eos_tokens.sort_unstable();
        eos_tokens.dedup();
        if let Some(position) = eos_tokens.iter().position(|token| *token == eos.get()) {
            eos_tokens.swap(0, position);
        }

        let info = TokRxInfo::new(vocab_size as u32, eos.get());
        let trie = TokTrie::from(&info, &token_bytes).with_eos_tokens(&eos_tokens);
        let tok_env: TokEnv = Arc::new(FerrumTokenizerEnv {
            tokenizer: Arc::clone(&tokenizer),
            trie,
        });
        let mut parser_factory = ParserFactory::new(
            &tok_env,
            InferenceCapabilities {
                ff_tokens: false,
                conditional_ff_tokens: false,
                backtrack: false,
                fork: false,
            },
            &llguidance::earley::SlicedBiasComputer::general_slices(),
        )
        .map_err(|error| {
            FerrumError::config(format!("build structured-output parser factory: {error}"))
        })?;
        parser_factory.quiet();

        Ok(Self {
            parser_factory,
            tokenizer,
            vocab_size,
            grammar_templates: Mutex::new(HashMap::new()),
        })
    }

    /// Compile one request's grammar while reusing the tokenizer trie.
    pub fn create_processor(
        &self,
        response_format: &ResponseFormat,
        start: &StructuredOutputStart,
    ) -> Result<Option<StructuredOutputProcessor>> {
        let schema = match response_format {
            ResponseFormat::Text => return Ok(None),
            ResponseFormat::JsonObject => json!({"type": "object"}),
            ResponseFormat::JsonSchema(schema) => {
                serde_json::from_str(schema).map_err(|error| {
                    FerrumError::invalid_request(format!(
                        "response_format.schema is not valid JSON: {error}"
                    ))
                })?
            }
        };
        let grammar_key = serde_json::to_string(&schema).map_err(|error| {
            FerrumError::invalid_request(format!("serialize structured-output schema: {error}"))
        })?;
        let matcher = {
            let mut templates = self.grammar_templates.lock();
            if let Some(template) = templates.get(&grammar_key) {
                template.deep_clone()
            } else {
                let grammar = TopLevelGrammar::from_json_schema(schema);
                let parser = self
                    .parser_factory
                    .create_parser(grammar)
                    .map_err(|error| {
                        FerrumError::invalid_request(format!(
                            "unsupported structured-output grammar: {error}"
                        ))
                    })?;
                let matcher = Matcher::new(Ok(parser));
                if templates.len() >= MAX_CACHED_GRAMMARS {
                    templates.clear();
                }
                templates.insert(grammar_key, matcher.deep_clone());
                matcher
            }
        };
        let activation = match start {
            StructuredOutputStart::Immediate => Activation::Active,
            StructuredOutputStart::AfterDelimiter(delimiter) => {
                if delimiter.is_empty() {
                    return Err(FerrumError::invalid_request(
                        "structured-output delimiter must not be empty",
                    ));
                }
                let delimiter_tokens = if let Some(token) = self.tokenizer.token_id(delimiter) {
                    vec![token.get()]
                } else {
                    self.tokenizer
                        .encode(delimiter, false)?
                        .into_iter()
                        .map(|token| token.get())
                        .collect::<Vec<_>>()
                };
                if delimiter_tokens.is_empty() {
                    return Err(FerrumError::invalid_request(format!(
                        "structured-output delimiter {delimiter:?} did not tokenize"
                    )));
                }
                Activation::WaitingForDelimiter { delimiter_tokens }
            }
        };

        Ok(Some(StructuredOutputProcessor {
            state: Mutex::new(ProcessorState {
                matcher,
                activation: activation.clone(),
                initial_activation: activation,
                consumed: 0,
            }),
            vocab_size: self.vocab_size,
        }))
    }
}

/// Per-request structured-output parser state.
pub struct StructuredOutputProcessor {
    state: Mutex<ProcessorState>,
    vocab_size: usize,
}

impl std::fmt::Debug for StructuredOutputProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();
        f.debug_struct("StructuredOutputProcessor")
            .field("vocab_size", &self.vocab_size)
            .field("consumed", &state.consumed)
            .field("active", &matches!(state.activation, Activation::Active))
            .finish()
    }
}

struct ProcessorState {
    matcher: Matcher,
    activation: Activation,
    initial_activation: Activation,
    consumed: usize,
}

#[derive(Clone)]
enum Activation {
    Active,
    WaitingForDelimiter { delimiter_tokens: Vec<u32> },
}

impl StructuredOutputProcessor {
    /// Consume newly generated tokens and hard-mask every illegal next token.
    /// Waiting-for-reasoning mode deliberately leaves logits untouched until
    /// the typed delimiter has been observed.
    pub fn mask_logits(&self, logits: &mut [f32], generated: &[TokenId]) -> Result<()> {
        self.mask_logits_inner(logits, generated, None).map(|_| ())
    }

    /// Apply the grammar mask while allowing engine-resolved stop tokens once
    /// the grammar accepts. Some model templates use an end-of-turn token that
    /// is not the tokenizer's primary EOS, so it cannot be represented by the
    /// grammar parser's EOS set alone.
    pub fn mask_logits_with_terminals(
        &self,
        logits: &mut [f32],
        generated: &[TokenId],
        terminal_token_ids: &HashSet<u32>,
    ) -> Result<bool> {
        self.mask_logits_inner(logits, generated, Some(terminal_token_ids))
    }

    fn mask_logits_inner(
        &self,
        logits: &mut [f32],
        generated: &[TokenId],
        terminal_token_ids: Option<&HashSet<u32>>,
    ) -> Result<bool> {
        let mut state = self.state.lock();
        advance_state(&mut state, generated, terminal_token_ids)?;
        if !matches!(state.activation, Activation::Active) {
            return Ok(false);
        }

        let accepting = state.matcher.is_accepting().map_err(|error| {
            FerrumError::model(format!(
                "structured-output acceptance check failed: {error}"
            ))
        })?;
        let mask = state.matcher.compute_mask_or_eos().map_err(|error| {
            FerrumError::model(format!("structured-output mask failed: {error}"))
        })?;
        let mut finite_allowed = 0usize;
        for (idx, logit) in logits.iter_mut().enumerate() {
            let token = idx as u32;
            let allowed = idx < self.vocab_size
                && (mask.is_allowed(token)
                    || (accepting
                        && terminal_token_ids.is_some_and(|terminals| terminals.contains(&token))));
            if !allowed {
                *logit = f32::NEG_INFINITY;
            } else if logit.is_finite() {
                finite_allowed += 1;
            }
        }
        if finite_allowed == 0 {
            return Err(FerrumError::model(
                "structured-output grammar has no legal finite token",
            ));
        }
        Ok(accepting)
    }

    /// True only when reasoning has closed and the grammar accepts the full
    /// generated structured value.
    pub fn is_accepting(&self, generated: &[TokenId]) -> Result<bool> {
        self.is_accepting_inner(generated, None)
    }

    /// Completion check that treats an engine-resolved terminal sampled after
    /// grammar acceptance as framing rather than part of the JSON value.
    pub fn is_accepting_with_terminals(
        &self,
        generated: &[TokenId],
        terminal_token_ids: &HashSet<u32>,
    ) -> Result<bool> {
        self.is_accepting_inner(generated, Some(terminal_token_ids))
    }

    fn is_accepting_inner(
        &self,
        generated: &[TokenId],
        terminal_token_ids: Option<&HashSet<u32>>,
    ) -> Result<bool> {
        let mut state = self.state.lock();
        advance_state(&mut state, generated, terminal_token_ids)?;
        if !matches!(state.activation, Activation::Active) {
            return Ok(false);
        }
        state.matcher.is_accepting().map_err(|error| {
            FerrumError::model(format!(
                "structured-output acceptance check failed: {error}"
            ))
        })
    }

    pub fn reset(&self) -> Result<()> {
        let mut state = self.state.lock();
        state
            .matcher
            .reset()
            .map_err(|error| FerrumError::internal(format!("reset structured output: {error}")))?;
        state.activation = state.initial_activation.clone();
        state.consumed = 0;
        Ok(())
    }
}

fn advance_state(
    state: &mut ProcessorState,
    generated: &[TokenId],
    terminal_token_ids: Option<&HashSet<u32>>,
) -> Result<()> {
    if state.consumed > generated.len() {
        return Err(FerrumError::internal(
            "structured-output token history moved backwards without reset",
        ));
    }

    if let Activation::WaitingForDelimiter { delimiter_tokens } = &state.activation {
        let search_from = state.consumed.saturating_sub(delimiter_tokens.len());
        if let Some(offset) = generated[search_from..]
            .windows(delimiter_tokens.len())
            .position(|window| {
                window
                    .iter()
                    .zip(delimiter_tokens)
                    .all(|(token, expected)| token.get() == *expected)
            })
        {
            let grammar_start = search_from + offset + delimiter_tokens.len();
            state.activation = Activation::Active;
            state.consumed = grammar_start;
        } else {
            state.consumed = generated.len();
            return Ok(());
        }
    }

    for token in &generated[state.consumed..] {
        if terminal_token_ids.is_some_and(|terminals| terminals.contains(&token.get()))
            && state.matcher.is_accepting().map_err(|error| {
                FerrumError::model(format!(
                    "structured-output acceptance check failed: {error}"
                ))
            })?
        {
            continue;
        }
        state.matcher.consume_token(token.get()).map_err(|error| {
            FerrumError::model(format!(
                "structured-output token {} violated the grammar: {error}",
                token.get()
            ))
        })?;
    }
    state.consumed = generated.len();
    Ok(())
}

struct FerrumTokenizerEnv {
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    trie: TokTrie,
}

impl TokenizerEnv for FerrumTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        str::from_utf8(bytes)
            .ok()
            .and_then(|text| self.tokenizer.encode(text, false).ok())
            .map(|tokens| tokens.into_iter().map(|token| token.get()).collect())
            .unwrap_or_else(|| self.trie.greedy_tokenize(bytes))
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

fn tokenizer_special_ids(tokenizer: &(dyn Tokenizer + Send + Sync)) -> HashSet<u32> {
    let special = tokenizer.special_tokens();
    [
        special.bos_token,
        special.eos_token,
        special.unk_token,
        special.pad_token,
        special.sep_token,
        special.cls_token,
        special.mask_token,
    ]
    .into_iter()
    .flatten()
    .chain(special.extra_eos_tokens.iter().copied())
    .map(|token| token.get())
    .collect()
}

fn special_token_marker(token: TokenId) -> Vec<u8> {
    let mut marker = vec![TokTrie::SPECIAL_TOKEN_MARKER];
    marker.extend_from_slice(format!("[{}]", token.get()).as_bytes());
    marker
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::tokenizer::{ChatMessage, TokenizerInfo, TokenizerType};
    use ferrum_types::SpecialTokens;

    const EOS: u32 = 256;

    struct ByteTokenizer {
        special: SpecialTokens,
        token_text: Vec<String>,
    }

    impl ByteTokenizer {
        fn new() -> Self {
            let mut token_text = (0u16..=255)
                .map(|byte| char::from_u32(byte as u32).unwrap().to_string())
                .collect::<Vec<_>>();
            token_text.push("<eos>".to_string());
            Self {
                special: SpecialTokens {
                    eos_token: Some(TokenId::new(EOS)),
                    ..SpecialTokens::default()
                },
                token_text,
            }
        }
    }

    impl Tokenizer for ByteTokenizer {
        fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
            Ok(text
                .as_bytes()
                .iter()
                .map(|byte| TokenId::new(*byte as u32))
                .collect())
        }

        fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
            Ok(tokens
                .iter()
                .filter(|token| token.get() < 256)
                .map(|token| token.get() as u8 as char)
                .collect())
        }

        fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
            self.decode(&[next], true)
        }

        fn vocab_size(&self) -> usize {
            self.token_text.len()
        }

        fn special_tokens(&self) -> &SpecialTokens {
            &self.special
        }

        fn token_id(&self, text: &str) -> Option<TokenId> {
            (text.len() == 1).then(|| TokenId::new(text.as_bytes()[0] as u32))
        }

        fn token_text(&self, token_id: TokenId) -> Option<&str> {
            self.token_text
                .get(token_id.get() as usize)
                .map(String::as_str)
        }

        fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
            Ok(messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n"))
        }

        fn info(&self) -> TokenizerInfo {
            TokenizerInfo {
                tokenizer_type: TokenizerType::BPE,
                vocab_size: self.vocab_size(),
                special_tokens: self.special.clone(),
                supports_incremental: true,
                supports_chat_template: false,
                max_token_length: Some(1),
                model_name: Some("byte-test".to_string()),
            }
        }
    }

    fn factory() -> StructuredOutputFactory {
        StructuredOutputFactory::new(Arc::new(ByteTokenizer::new())).unwrap()
    }

    fn assert_and_append(
        processor: &StructuredOutputProcessor,
        generated: &mut Vec<TokenId>,
        text: &str,
    ) {
        for byte in text.bytes() {
            let mut logits = vec![0.0; EOS as usize + 1];
            processor.mask_logits(&mut logits, generated).unwrap();
            assert!(
                logits[byte as usize].is_finite(),
                "byte {byte:?} rejected after {:?}",
                generated
            );
            generated.push(TokenId::new(byte as u32));
        }
    }

    #[test]
    fn json_object_hard_masks_non_object_roots() {
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::Immediate,
            )
            .unwrap()
            .unwrap();
        let mut logits = vec![0.0; EOS as usize + 1];
        processor.mask_logits(&mut logits, &[]).unwrap();
        assert!(logits[b'{' as usize].is_finite());
        assert!(!logits[b'[' as usize].is_finite());
        assert!(!logits[b'`' as usize].is_finite());
        assert!(!logits[EOS as usize].is_finite());
    }

    #[test]
    fn json_object_accepts_nested_unicode_escape_and_eos_only_after_close() {
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::Immediate,
            )
            .unwrap()
            .unwrap();
        let mut generated = Vec::new();
        assert_and_append(
            &processor,
            &mut generated,
            r#"{"items":[true,null,{"name":"line\u000A"}],"n":-1.2e+3}"#,
        );
        assert!(processor.is_accepting(&generated).unwrap());
        let mut logits = vec![0.0; EOS as usize + 1];
        processor.mask_logits(&mut logits, &generated).unwrap();
        assert!(logits[EOS as usize].is_finite());
        assert!(!logits[b'x' as usize].is_finite());
    }

    #[test]
    fn strict_schema_rejects_wrong_property_and_accepts_required_value() {
        let schema = r#"{
            "type":"object",
            "properties":{"answer":{"const":42}},
            "required":["answer"],
            "additionalProperties":false
        }"#;
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonSchema(schema.to_string()),
                &StructuredOutputStart::Immediate,
            )
            .unwrap()
            .unwrap();
        let mut generated = Vec::new();
        assert_and_append(&processor, &mut generated, r#"{"answer":42}"#);
        assert!(processor.is_accepting(&generated).unwrap());
    }

    #[test]
    fn reasoning_delimiter_defers_then_activates_the_grammar() {
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::AfterDelimiter("</think>".to_string()),
            )
            .unwrap()
            .unwrap();
        let mut generated = Vec::new();
        assert_and_append(&processor, &mut generated, "reasoning [is free]</think>");
        let mut logits = vec![0.0; EOS as usize + 1];
        processor.mask_logits(&mut logits, &generated).unwrap();
        assert!(logits[b'{' as usize].is_finite());
        assert!(!logits[b'[' as usize].is_finite());
        assert!(!logits[EOS as usize].is_finite());

        assert_and_append(&processor, &mut generated, r#"{"ok":true}"#);
        assert!(processor.is_accepting(&generated).unwrap());
    }
}
