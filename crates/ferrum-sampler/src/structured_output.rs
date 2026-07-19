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
const AUTO_STRUCTURED_RESERVE_DIVISOR: usize = 4;
const MIN_AUTO_STRUCTURED_RESERVE_TOKENS: usize = 32;
const MAX_AUTO_STRUCTURED_RESERVE_TOKENS: usize = 1024;

/// Immutable per-request output budget used when a structured grammar starts
/// after a reasoning delimiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredOutputBudgetPlan {
    pub total_output_tokens: usize,
    pub reasoning_token_limit: usize,
    pub boundary_token_count: usize,
    pub structured_reserve_tokens: usize,
}

impl StructuredOutputBudgetPlan {
    fn automatic(total_output_tokens: usize, boundary_token_count: usize) -> Result<Self> {
        if boundary_token_count == 0 || total_output_tokens <= boundary_token_count {
            return Err(FerrumError::invalid_request(format!(
                "structured output requires max_tokens greater than its {boundary_token_count}-token delimiter"
            )));
        }
        let available_after_boundary = total_output_tokens - boundary_token_count;
        let proportional_reserve = total_output_tokens.div_ceil(AUTO_STRUCTURED_RESERVE_DIVISOR);
        let structured_reserve_tokens = proportional_reserve
            .clamp(
                MIN_AUTO_STRUCTURED_RESERVE_TOKENS,
                MAX_AUTO_STRUCTURED_RESERVE_TOKENS,
            )
            .min(available_after_boundary);
        Ok(Self {
            total_output_tokens,
            reasoning_token_limit: total_output_tokens
                - boundary_token_count
                - structured_reserve_tokens,
            boundary_token_count,
            structured_reserve_tokens,
        })
    }
}

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
        max_output_tokens: usize,
        stop_token_ids: &HashSet<u32>,
        stop_text_sequences: &[String],
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
        let (activation, budget) = match start {
            StructuredOutputStart::Immediate => (Activation::Active, None),
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
                if let Some(token) = delimiter_tokens
                    .iter()
                    .find(|token| stop_token_ids.contains(token))
                {
                    return Err(FerrumError::invalid_request(format!(
                        "structured-output delimiter token {token} conflicts with a stop token"
                    )));
                }
                if let Some(stop) = stop_text_sequences
                    .iter()
                    .find(|stop| !stop.is_empty() && delimiter.contains(stop.as_str()))
                {
                    return Err(FerrumError::invalid_request(format!(
                        "structured-output delimiter {delimiter:?} conflicts with stop sequence {stop:?}"
                    )));
                }
                let budget = StructuredOutputBudgetPlan::automatic(
                    max_output_tokens,
                    delimiter_tokens.len(),
                )?;
                (
                    Activation::Boundary {
                        delimiter_tokens,
                        forcing: false,
                    },
                    Some(budget),
                )
            }
        };

        Ok(Some(StructuredOutputProcessor {
            state: Mutex::new(ProcessorState {
                matcher,
                activation: activation.clone(),
                initial_activation: activation,
                consumed: 0,
                boundary_forced: false,
                boundary_start: None,
            }),
            vocab_size: self.vocab_size,
            budget,
        }))
    }
}

/// Per-request structured-output parser state.
pub struct StructuredOutputProcessor {
    state: Mutex<ProcessorState>,
    vocab_size: usize,
    budget: Option<StructuredOutputBudgetPlan>,
}

/// Typed phase returned after applying a structured-output constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputPhase {
    WaitingForDelimiter,
    ForcingDelimiter,
    EnforcingGrammar,
}

/// Allocation-free hot-path result of one structured-output mask operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredOutputMaskOutcome {
    pub phase: StructuredOutputPhase,
    pub accepting: bool,
    /// Exact delimiter token authorized for the next sampling step while the
    /// processor is waiting to activate. The engine uses this typed grant to
    /// avoid rejecting an intentionally hidden special token during output
    /// quality filtering.
    pub required_delimiter_token_id: Option<u32>,
}

/// Terminal/debug snapshot that distinguishes activation failures from an
/// incomplete grammar without retaining generated text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredOutputProgress {
    pub phase: StructuredOutputPhase,
    pub generated_token_count: usize,
    pub consumed_token_count: usize,
    pub delimiter_token_count: Option<usize>,
    pub delimiter_prefix_token_count: usize,
    pub reasoning_token_count: Option<usize>,
    pub boundary_forced: bool,
    pub budget: Option<StructuredOutputBudgetPlan>,
    pub accepting: bool,
}

impl std::fmt::Debug for StructuredOutputProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();
        f.debug_struct("StructuredOutputProcessor")
            .field("vocab_size", &self.vocab_size)
            .field("consumed", &state.consumed)
            .field("active", &matches!(state.activation, Activation::Active))
            .field("budget", &self.budget)
            .finish()
    }
}

struct ProcessorState {
    matcher: Matcher,
    activation: Activation,
    initial_activation: Activation,
    consumed: usize,
    boundary_forced: bool,
    boundary_start: Option<usize>,
}

#[derive(Clone)]
enum Activation {
    Active,
    Boundary {
        delimiter_tokens: Vec<u32>,
        forcing: bool,
    },
}

impl StructuredOutputProcessor {
    /// Consume newly generated tokens and hard-mask every illegal next token.
    /// Waiting-for-reasoning mode leaves normal logits untouched until the
    /// typed delimiter has been observed.
    pub fn mask_logits(&self, logits: &mut [f32], generated: &[TokenId]) -> Result<()> {
        self.mask_logits_inner(logits, generated, None, None)
            .map(|_| ())
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
        hidden_control_token_ids: &HashSet<u32>,
    ) -> Result<StructuredOutputMaskOutcome> {
        self.mask_logits_inner(
            logits,
            generated,
            Some(terminal_token_ids),
            Some(hidden_control_token_ids),
        )
    }

    fn mask_logits_inner(
        &self,
        logits: &mut [f32],
        generated: &[TokenId],
        terminal_token_ids: Option<&HashSet<u32>>,
        hidden_control_token_ids: Option<&HashSet<u32>>,
    ) -> Result<StructuredOutputMaskOutcome> {
        let mut state = self.state.lock();
        advance_state(&mut state, generated, terminal_token_ids)?;
        activate_forcing_if_due(&mut state, generated, self.budget);
        if let Activation::Boundary {
            delimiter_tokens,
            forcing,
        } = &state.activation
        {
            let delimiter_prefix_token_count =
                delimiter_prefix_token_count(generated, delimiter_tokens);
            let required_delimiter_token = delimiter_tokens
                .get(delimiter_prefix_token_count)
                .copied()
                .ok_or_else(|| {
                    FerrumError::internal("structured-output delimiter state has no next token")
                })?;
            if *forcing {
                force_exact_token(logits, required_delimiter_token)?;
            } else if let Some(hidden_control_token_ids) = hidden_control_token_ids {
                for token_id in hidden_control_token_ids {
                    if required_delimiter_token == *token_id {
                        continue;
                    }
                    if let Some(logit) = logits.get_mut(*token_id as usize) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
            return Ok(StructuredOutputMaskOutcome {
                phase: if *forcing {
                    StructuredOutputPhase::ForcingDelimiter
                } else {
                    StructuredOutputPhase::WaitingForDelimiter
                },
                accepting: false,
                required_delimiter_token_id: Some(required_delimiter_token),
            });
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
            let hidden_non_terminal_control = hidden_control_token_ids
                .is_some_and(|controls| controls.contains(&token))
                && !terminal_token_ids.is_some_and(|terminals| terminals.contains(&token));
            let allowed = idx < self.vocab_size
                && !hidden_non_terminal_control
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
        Ok(StructuredOutputMaskOutcome {
            phase: StructuredOutputPhase::EnforcingGrammar,
            accepting,
            required_delimiter_token_id: None,
        })
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
        Ok(self
            .progress_inner(generated, terminal_token_ids)?
            .accepting)
    }

    /// Inspect the typed activation/grammar state after consuming `generated`.
    pub fn progress_with_terminals(
        &self,
        generated: &[TokenId],
        terminal_token_ids: &HashSet<u32>,
    ) -> Result<StructuredOutputProgress> {
        self.progress_inner(generated, Some(terminal_token_ids))
    }

    fn progress_inner(
        &self,
        generated: &[TokenId],
        terminal_token_ids: Option<&HashSet<u32>>,
    ) -> Result<StructuredOutputProgress> {
        let mut state = self.state.lock();
        advance_state(&mut state, generated, terminal_token_ids)?;
        activate_forcing_if_due(&mut state, generated, self.budget);
        let (phase, delimiter_token_count, delimiter_prefix_token_count, accepting) =
            match &state.activation {
                Activation::Boundary {
                    delimiter_tokens,
                    forcing,
                } => (
                    if *forcing {
                        StructuredOutputPhase::ForcingDelimiter
                    } else {
                        StructuredOutputPhase::WaitingForDelimiter
                    },
                    Some(delimiter_tokens.len()),
                    delimiter_prefix_token_count(generated, delimiter_tokens),
                    false,
                ),
                Activation::Active => (
                    StructuredOutputPhase::EnforcingGrammar,
                    None,
                    0,
                    state.matcher.is_accepting().map_err(|error| {
                        FerrumError::model(format!(
                            "structured-output acceptance check failed: {error}"
                        ))
                    })?,
                ),
            };
        Ok(StructuredOutputProgress {
            phase,
            generated_token_count: generated.len(),
            consumed_token_count: state.consumed,
            delimiter_token_count: delimiter_token_count
                .or(self.budget.map(|budget| budget.boundary_token_count)),
            delimiter_prefix_token_count,
            reasoning_token_count: self
                .budget
                .map(|_| state.boundary_start.unwrap_or(generated.len())),
            boundary_forced: state.boundary_forced,
            budget: self.budget,
            accepting,
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
        state.boundary_forced = false;
        state.boundary_start = None;
        Ok(())
    }
}

fn activate_forcing_if_due(
    state: &mut ProcessorState,
    generated: &[TokenId],
    budget: Option<StructuredOutputBudgetPlan>,
) {
    let Some(budget) = budget else {
        return;
    };
    let should_force = matches!(
        state.activation,
        Activation::Boundary { forcing: false, .. }
    ) && generated.len() >= budget.reasoning_token_limit;
    if should_force {
        let delimiter_prefix_token_count = match &state.activation {
            Activation::Boundary {
                delimiter_tokens, ..
            } => delimiter_prefix_token_count(generated, delimiter_tokens),
            Activation::Active => 0,
        };
        if let Activation::Boundary { forcing, .. } = &mut state.activation {
            *forcing = true;
        }
        state.boundary_forced = true;
        state.boundary_start = Some(generated.len() - delimiter_prefix_token_count);
    }
}

fn force_exact_token(logits: &mut [f32], required_token: u32) -> Result<()> {
    let required_index = required_token as usize;
    if required_index >= logits.len() {
        return Err(FerrumError::model(format!(
            "structured-output delimiter token {required_token} is outside logits width {}",
            logits.len()
        )));
    }
    logits.fill(f32::NEG_INFINITY);
    logits[required_index] = 0.0;
    Ok(())
}

fn delimiter_prefix_token_count(generated: &[TokenId], delimiter_tokens: &[u32]) -> usize {
    let max_prefix = generated
        .len()
        .min(delimiter_tokens.len().saturating_sub(1));
    (1..=max_prefix)
        .rev()
        .find(|prefix_len| {
            generated[generated.len() - prefix_len..]
                .iter()
                .zip(&delimiter_tokens[..*prefix_len])
                .all(|(token, expected)| token.get() == *expected)
        })
        .unwrap_or(0)
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

    if let Activation::Boundary {
        delimiter_tokens, ..
    } = &state.activation
    {
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
            state.boundary_start = Some(grammar_start - delimiter_tokens.len());
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
    const TEST_MAX_OUTPUT_TOKENS: usize = 128;

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
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
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
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
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
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
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
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let mut generated = Vec::new();
        let controls = HashSet::from([b'<' as u32, b'>' as u32]);
        let mut waiting_logits = vec![0.0; EOS as usize + 1];
        let waiting = processor
            .mask_logits_with_terminals(&mut waiting_logits, &generated, &HashSet::new(), &controls)
            .unwrap();
        assert_eq!(waiting.phase, StructuredOutputPhase::WaitingForDelimiter);
        assert!(!waiting.accepting);
        assert_eq!(waiting.required_delimiter_token_id, Some(b'<' as u32));
        assert!(waiting_logits[b'<' as usize].is_finite());
        assert!(!waiting_logits[b'>' as usize].is_finite());

        let delimiter_prefix = "</think"
            .bytes()
            .map(|byte| TokenId::new(byte as u32))
            .collect::<Vec<_>>();
        let mut partial_logits = vec![0.0; EOS as usize + 1];
        let partial = processor
            .mask_logits_with_terminals(
                &mut partial_logits,
                &delimiter_prefix,
                &HashSet::new(),
                &controls,
            )
            .unwrap();
        assert_eq!(partial.phase, StructuredOutputPhase::WaitingForDelimiter);
        assert_eq!(partial.required_delimiter_token_id, Some(b'>' as u32));
        assert!(!partial_logits[b'<' as usize].is_finite());
        assert!(partial_logits[b'>' as usize].is_finite());
        let partial_progress = processor
            .progress_with_terminals(&delimiter_prefix, &HashSet::new())
            .unwrap();
        assert_eq!(partial_progress.delimiter_token_count, Some(8));
        assert_eq!(partial_progress.delimiter_prefix_token_count, 7);

        processor.reset().unwrap();

        assert_and_append(&processor, &mut generated, "reasoning [is free]</think>");
        let mut logits = vec![0.0; EOS as usize + 1];
        processor.mask_logits(&mut logits, &generated).unwrap();
        assert!(logits[b'{' as usize].is_finite());
        assert!(!logits[b'[' as usize].is_finite());
        assert!(!logits[EOS as usize].is_finite());

        assert_and_append(&processor, &mut generated, r#"{"ok":true}"#);
        assert!(processor.is_accepting(&generated).unwrap());
        let progress = processor
            .progress_with_terminals(&generated, &HashSet::new())
            .unwrap();
        assert_eq!(progress.phase, StructuredOutputPhase::EnforcingGrammar);
        assert!(progress.accepting);
        assert_eq!(progress.generated_token_count, generated.len());
        assert!(!progress.boundary_forced);
        assert_eq!(progress.reasoning_token_count, Some(19));
    }

    #[test]
    fn reasoning_budget_forces_exact_delimiter_and_preserves_structured_reserve() {
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::AfterDelimiter("</think>".to_string()),
                48,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let mut generated = "reason!!"
            .bytes()
            .map(|byte| TokenId::new(byte as u32))
            .collect::<Vec<_>>();

        for expected in "</think>".bytes() {
            let mut logits = vec![f32::NEG_INFINITY; EOS as usize + 1];
            let outcome = processor
                .mask_logits_with_terminals(
                    &mut logits,
                    &generated,
                    &HashSet::new(),
                    &HashSet::new(),
                )
                .unwrap();
            assert_eq!(outcome.phase, StructuredOutputPhase::ForcingDelimiter);
            assert_eq!(outcome.required_delimiter_token_id, Some(expected as u32));
            assert_eq!(
                logits
                    .iter()
                    .enumerate()
                    .filter(|(_, logit)| logit.is_finite())
                    .map(|(token, _)| token)
                    .collect::<Vec<_>>(),
                vec![expected as usize]
            );
            generated.push(TokenId::new(expected as u32));
        }

        let mut grammar_logits = vec![0.0; EOS as usize + 1];
        let outcome = processor
            .mask_logits_with_terminals(
                &mut grammar_logits,
                &generated,
                &HashSet::new(),
                &HashSet::new(),
            )
            .unwrap();
        assert_eq!(outcome.phase, StructuredOutputPhase::EnforcingGrammar);
        assert!(grammar_logits[b'{' as usize].is_finite());
        assert!(!grammar_logits[b'[' as usize].is_finite());

        let progress = processor
            .progress_with_terminals(&generated, &HashSet::new())
            .unwrap();
        assert_eq!(progress.reasoning_token_count, Some(8));
        assert!(progress.boundary_forced);
        assert_eq!(
            progress.budget,
            Some(StructuredOutputBudgetPlan {
                total_output_tokens: 48,
                reasoning_token_limit: 8,
                boundary_token_count: 8,
                structured_reserve_tokens: 32,
            })
        );

        processor.reset().unwrap();
        let reset_progress = processor
            .progress_with_terminals(&[], &HashSet::new())
            .unwrap();
        assert_eq!(
            reset_progress.phase,
            StructuredOutputPhase::WaitingForDelimiter
        );
        assert!(!reset_progress.boundary_forced);
        assert_eq!(reset_progress.reasoning_token_count, Some(0));
    }

    #[test]
    fn reasoning_delimiter_requires_room_beyond_the_boundary() {
        let error = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::AfterDelimiter("</think>".to_string()),
                8,
                &HashSet::new(),
                &[],
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("max_tokens greater than its 8-token delimiter"));
    }

    #[test]
    fn forcing_accounts_for_an_existing_delimiter_prefix() {
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::AfterDelimiter("</think>".to_string()),
                48,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let generated = "reason</"
            .bytes()
            .map(|byte| TokenId::new(byte as u32))
            .collect::<Vec<_>>();
        let mut logits = vec![0.0; EOS as usize + 1];

        let outcome = processor
            .mask_logits_with_terminals(&mut logits, &generated, &HashSet::new(), &HashSet::new())
            .unwrap();
        assert_eq!(outcome.phase, StructuredOutputPhase::ForcingDelimiter);
        assert_eq!(outcome.required_delimiter_token_id, Some(b't' as u32));
        let progress = processor
            .progress_with_terminals(&generated, &HashSet::new())
            .unwrap();
        assert_eq!(progress.delimiter_prefix_token_count, 2);
        assert_eq!(progress.reasoning_token_count, Some(6));
    }

    #[test]
    fn delimiter_rejects_any_conflicting_stop_condition_up_front() {
        let token_error = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::AfterDelimiter("</think>".to_string()),
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::from([b'/' as u32]),
                &[],
            )
            .unwrap_err();
        assert!(token_error
            .to_string()
            .contains("conflicts with a stop token"));

        let text_error = factory()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::AfterDelimiter("</think>".to_string()),
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &["think".to_string()],
            )
            .unwrap_err();
        assert!(text_error
            .to_string()
            .contains("conflicts with stop sequence"));
    }
}
