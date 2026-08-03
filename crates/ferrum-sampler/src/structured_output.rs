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
    JsonCompileOptions, Matcher, ParserFactory,
};
use parking_lot::Mutex;
use serde_json::json;

const MAX_CACHED_GRAMMARS: usize = 64;
// Structured output is the requested product result; hidden reasoning may use
// at most the other half of a normal-sized completion budget.
const AUTO_STRUCTURED_RESERVE_DIVISOR: usize = 2;
const MIN_AUTO_STRUCTURED_RESERVE_TOKENS: usize = 32;
const MAX_AUTO_STRUCTURED_RESERVE_TOKENS: usize = 1024;
const MAX_IDENTICAL_TOKEN_RUN: usize = MAX_AUTO_STRUCTURED_RESERVE_TOKENS / 2;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StructuredOutputLivenessPolicy {
    max_identical_token_run: usize,
}

impl StructuredOutputLivenessPolicy {
    fn for_request(max_output_tokens: usize, budget: Option<StructuredOutputBudgetPlan>) -> Self {
        let guaranteed_structured_tokens = budget
            .map(|plan| plan.structured_reserve_tokens)
            .unwrap_or(max_output_tokens);
        Self {
            max_identical_token_run: guaranteed_structured_tokens
                .div_ceil(2)
                .clamp(1, MAX_IDENTICAL_TOKEN_RUN),
        }
    }
}

/// Shared, immutable tokenizer and grammar compilation state.
pub struct StructuredOutputFactory {
    parser_factory: ParserFactory,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    vocab_size: usize,
    defined_token_ids: Arc<[bool]>,
    json_token_classes: Arc<[StructuredOutputTokenClass]>,
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
        let mut defined_token_ids = Vec::with_capacity(vocab_size);
        let mut json_token_classes = Vec::with_capacity(vocab_size);
        let token_bytes = (0..vocab_size)
            .map(|idx| {
                let token = TokenId::new(idx as u32);
                if special_ids.contains(&token.get()) {
                    defined_token_ids.push(true);
                    json_token_classes.push(StructuredOutputTokenClass::Control);
                    special_token_marker(token)
                } else if let Some(bytes) = tokenizer
                    .token_bytes(token)
                    .filter(|bytes| !bytes.is_empty())
                {
                    defined_token_ids.push(true);
                    json_token_classes.push(classify_json_token_bytes(&bytes));
                    bytes
                } else {
                    // Keep vocabulary holes out of the trie. The explicit
                    // eligibility mask below is still required because
                    // llguidance's wildcard slice represents its root as an
                    // all-token bitset, including IDs with no trie node.
                    defined_token_ids.push(false);
                    json_token_classes.push(StructuredOutputTokenClass::Undefined);
                    Vec::new()
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
            defined_token_ids: defined_token_ids.into(),
            json_token_classes: json_token_classes.into(),
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
        let schema = compact_json_schema(schema)?;
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

        let grammar_start = matches!(activation, Activation::Active).then_some(0);
        let liveness = StructuredOutputLivenessPolicy::for_request(max_output_tokens, budget);
        Ok(Some(StructuredOutputProcessor {
            state: Mutex::new(ProcessorState {
                matcher,
                activation: activation.clone(),
                initial_activation: activation,
                consumed: 0,
                boundary_forced: false,
                boundary_start: None,
                grammar_start,
                trailing_grammar_token_id: None,
                trailing_identical_token_count: 0,
                liveness_intervention_count: 0,
                last_liveness_intervention_at: None,
            }),
            vocab_size: self.vocab_size,
            defined_token_ids: Arc::clone(&self.defined_token_ids),
            json_token_classes: Arc::clone(&self.json_token_classes),
            budget,
            liveness,
        }))
    }
}

fn compact_json_schema(schema: serde_json::Value) -> Result<serde_json::Value> {
    let mut schema = match schema {
        schema @ serde_json::Value::Object(_) => schema,
        schema @ serde_json::Value::Bool(_) => json!({"allOf": [schema]}),
        _ => {
            return Err(FerrumError::invalid_request(
                "response_format.schema must be a JSON Schema object or boolean",
            ));
        }
    };

    // x-guidance is an llguidance compiler extension, not a JSON Schema
    // constraint. Keep compiler policy owned by Ferrum so a request cannot
    // re-enable an unbounded whitespace loop or change JSON separators.
    JsonCompileOptions {
        whitespace_flexible: false,
        ..JsonCompileOptions::default()
    }
    .apply_to(&mut schema);
    Ok(schema)
}

/// Per-request structured-output parser state.
pub struct StructuredOutputProcessor {
    state: Mutex<ProcessorState>,
    vocab_size: usize,
    defined_token_ids: Arc<[bool]>,
    json_token_classes: Arc<[StructuredOutputTokenClass]>,
    budget: Option<StructuredOutputBudgetPlan>,
    liveness: StructuredOutputLivenessPolicy,
}

/// Typed phase returned after applying a structured-output constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputPhase {
    WaitingForDelimiter,
    ForcingDelimiter,
    EnforcingGrammar,
}

/// Privacy-safe lexical class for the tail of an incomplete grammar.
///
/// This deliberately exposes neither decoded text nor a token history. It is
/// used only by terminal diagnostics to distinguish an unbounded whitespace,
/// number, string, or control-token run from a parser activation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StructuredOutputTokenClass {
    Whitespace,
    Number,
    Structural,
    StringBoundary,
    Literal,
    Other,
    Control,
    Undefined,
}

/// Allocation-free hot-path result of one structured-output mask operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredOutputMaskOutcome {
    pub phase: StructuredOutputPhase,
    pub accepting: bool,
    pub liveness_intervention: bool,
    /// Generated-token index at which the visible grammar-owned output starts.
    /// `None` means the processor is still in the hidden pre-grammar domain.
    /// The execution engine uses this boundary to scope request-local sampling
    /// history without teaching the grammar about penalty policy.
    pub grammar_start_token_index: Option<usize>,
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
    pub grammar_token_count: usize,
    pub trailing_token_class: Option<StructuredOutputTokenClass>,
    pub trailing_token_class_count: usize,
    pub trailing_token_id: Option<u32>,
    pub trailing_identical_token_count: usize,
    pub liveness_identical_token_limit: usize,
    pub liveness_intervention_count: usize,
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
    grammar_start: Option<usize>,
    trailing_grammar_token_id: Option<u32>,
    trailing_identical_token_count: usize,
    liveness_intervention_count: usize,
    last_liveness_intervention_at: Option<usize>,
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
            self.mask_undefined_token_ids(logits);
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
                liveness_intervention: false,
                grammar_start_token_index: None,
                required_delimiter_token_id: Some(required_delimiter_token),
            });
        }

        let grammar_start_token_index = state.grammar_start.ok_or_else(|| {
            FerrumError::internal(
                "active structured-output processor has no grammar start token index",
            )
        })?;

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
                && self.defined_token_ids.get(idx).copied().unwrap_or(false)
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
        let liveness_intervention = if !accepting
            && state.trailing_identical_token_count >= self.liveness.max_identical_token_run
        {
            state
                .trailing_grammar_token_id
                .and_then(|token| logits.get_mut(token as usize))
                .is_some_and(|logit| {
                    if logit.is_finite() && finite_allowed > 1 {
                        *logit = f32::NEG_INFINITY;
                        if state.last_liveness_intervention_at != Some(generated.len()) {
                            state.liveness_intervention_count += 1;
                            state.last_liveness_intervention_at = Some(generated.len());
                        }
                        true
                    } else {
                        false
                    }
                })
        } else {
            false
        };
        Ok(StructuredOutputMaskOutcome {
            phase: StructuredOutputPhase::EnforcingGrammar,
            accepting,
            liveness_intervention,
            grammar_start_token_index: Some(grammar_start_token_index),
            required_delimiter_token_id: None,
        })
    }

    fn mask_undefined_token_ids(&self, logits: &mut [f32]) {
        for (idx, logit) in logits.iter_mut().enumerate() {
            if !self.defined_token_ids.get(idx).copied().unwrap_or(false) {
                *logit = f32::NEG_INFINITY;
            }
        }
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
        let grammar_tokens = state
            .grammar_start
            .and_then(|start| generated.get(start..))
            .unwrap_or_default();
        let trailing_token_id = grammar_tokens.last().map(|token| token.get());
        let trailing_token_class = trailing_token_id.map(|token| {
            self.json_token_classes
                .get(token as usize)
                .copied()
                .unwrap_or(StructuredOutputTokenClass::Undefined)
        });
        let trailing_token_class_count = trailing_token_class.map_or(0, |class| {
            grammar_tokens
                .iter()
                .rev()
                .take_while(|token| {
                    self.json_token_classes
                        .get(token.get() as usize)
                        .copied()
                        .unwrap_or(StructuredOutputTokenClass::Undefined)
                        == class
                })
                .count()
        });
        let trailing_identical_token_count = trailing_token_id.map_or(0, |token_id| {
            grammar_tokens
                .iter()
                .rev()
                .take_while(|token| token.get() == token_id)
                .count()
        });
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
            grammar_token_count: grammar_tokens.len(),
            trailing_token_class,
            trailing_token_class_count,
            trailing_token_id,
            trailing_identical_token_count,
            liveness_identical_token_limit: self.liveness.max_identical_token_run,
            liveness_intervention_count: state.liveness_intervention_count,
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
        state.grammar_start = matches!(state.initial_activation, Activation::Active).then_some(0);
        state.trailing_grammar_token_id = None;
        state.trailing_identical_token_count = 0;
        state.liveness_intervention_count = 0;
        state.last_liveness_intervention_at = None;
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
            state.grammar_start = Some(grammar_start);
            state.activation = Activation::Active;
            state.consumed = grammar_start;
            state.trailing_grammar_token_id = None;
            state.trailing_identical_token_count = 0;
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
        if state.trailing_grammar_token_id == Some(token.get()) {
            state.trailing_identical_token_count += 1;
        } else {
            state.trailing_grammar_token_id = Some(token.get());
            state.trailing_identical_token_count = 1;
        }
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

fn classify_json_token_bytes(bytes: &[u8]) -> StructuredOutputTokenClass {
    if bytes.is_empty() {
        StructuredOutputTokenClass::Undefined
    } else if bytes
        .iter()
        .all(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
    {
        StructuredOutputTokenClass::Whitespace
    } else if bytes
        .iter()
        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E'))
    {
        StructuredOutputTokenClass::Number
    } else if bytes
        .iter()
        .all(|byte| matches!(byte, b'{' | b'}' | b'[' | b']' | b',' | b':'))
    {
        StructuredOutputTokenClass::Structural
    } else if bytes.iter().all(|byte| matches!(byte, b'"' | b'\\')) {
        StructuredOutputTokenClass::StringBoundary
    } else if bytes.iter().all(u8::is_ascii_alphabetic) {
        StructuredOutputTokenClass::Literal
    } else {
        StructuredOutputTokenClass::Other
    }
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

    struct MergedObjectTokenizer {
        inner: ByteTokenizer,
    }

    impl MergedObjectTokenizer {
        const OBJECT: u32 = 256;
        const EOS: u32 = 257;

        fn new() -> Self {
            let mut inner = ByteTokenizer::new();
            inner.token_text[EOS as usize] = "{}".to_string();
            inner.token_text.push("<eos>".to_string());
            inner.special.eos_token = Some(TokenId::new(Self::EOS));
            Self { inner }
        }
    }

    impl Tokenizer for MergedObjectTokenizer {
        fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
            self.inner.encode(text, add_special)
        }

        fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
            let mut decoded = String::new();
            for token in tokens {
                match token.get() {
                    Self::OBJECT => decoded.push_str("{}"),
                    Self::EOS if skip_special => {}
                    Self::EOS => decoded.push_str("<eos>"),
                    _ => decoded.push_str(&self.inner.decode(&[*token], skip_special)?),
                }
            }
            Ok(decoded)
        }

        fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
            self.decode(&[next], true)
        }

        fn vocab_size(&self) -> usize {
            self.inner.token_text.len()
        }

        fn special_tokens(&self) -> &SpecialTokens {
            &self.inner.special
        }

        fn token_id(&self, text: &str) -> Option<TokenId> {
            (text == "{}")
                .then(|| TokenId::new(Self::OBJECT))
                .or_else(|| self.inner.token_id(text))
        }

        fn token_text(&self, token_id: TokenId) -> Option<&str> {
            self.inner.token_text(token_id)
        }

        fn info(&self) -> TokenizerInfo {
            TokenizerInfo {
                vocab_size: self.vocab_size(),
                max_token_length: Some(2),
                model_name: Some("merged-object-test".to_string()),
                ..self.inner.info()
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
    fn json_object_uses_compact_separators_without_unbounded_whitespace() {
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
        let generated = vec![TokenId::new(b'{' as u32)];
        let mut logits = vec![0.0; EOS as usize + 1];
        processor.mask_logits(&mut logits, &generated).unwrap();

        assert!(!logits[b' ' as usize].is_finite());
        assert!(logits[b'}' as usize].is_finite());
        assert!(logits[b'"' as usize].is_finite());
    }

    #[test]
    fn undefined_model_vocab_ids_are_masked_inside_wildcard_strings() {
        let tokenizer = Arc::new(ByteTokenizer::new());
        let undefined_token = tokenizer.vocab_size() as u32;
        assert_eq!(
            tokenizer.token_bytes(TokenId::new(undefined_token)),
            Some(Vec::new()),
            "the test tokenizer must reproduce a decoder that returns empty text for an unknown id"
        );
        let processor = StructuredOutputFactory::new_with_model_vocab_size(
            tokenizer,
            Some(undefined_token as usize + 1),
        )
        .unwrap()
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
        assert_and_append(&processor, &mut generated, r#"{"value":"Ferrum "#);
        let mut logits = vec![0.0; undefined_token as usize + 1];
        processor.mask_logits(&mut logits, &generated).unwrap();
        assert!(logits[b'x' as usize].is_finite());
        assert!(!logits[undefined_token as usize].is_finite());
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
    fn json_object_accepts_a_complete_root_from_one_merged_token() {
        let processor = StructuredOutputFactory::new(Arc::new(MergedObjectTokenizer::new()))
            .unwrap()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::Immediate,
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let generated = vec![TokenId::new(MergedObjectTokenizer::OBJECT)];
        let terminals = HashSet::from([MergedObjectTokenizer::EOS]);

        let progress = processor
            .progress_with_terminals(&generated, &terminals)
            .unwrap();
        assert!(progress.accepting);

        let mut logits = vec![0.0; MergedObjectTokenizer::EOS as usize + 1];
        let outcome = processor
            .mask_logits_with_terminals(&mut logits, &generated, &terminals, &HashSet::new())
            .unwrap();
        assert!(outcome.accepting);
        assert_eq!(outcome.grammar_start_token_index, Some(0));
        assert!(!logits[MergedObjectTokenizer::OBJECT as usize].is_finite());
        assert!(logits[MergedObjectTokenizer::EOS as usize].is_finite());
    }

    #[test]
    fn json_object_breaks_an_unbounded_identical_token_run_when_closure_is_legal() {
        let processor = StructuredOutputFactory::new(Arc::new(MergedObjectTokenizer::new()))
            .unwrap()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::Immediate,
                64,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let mut generated =
            br#"{"marker":""#.iter().map(|byte| TokenId::new(*byte as u32)).collect::<Vec<_>>();
        generated.extend(std::iter::repeat_n(
            TokenId::new(MergedObjectTokenizer::OBJECT),
            32,
        ));

        let mut logits = vec![0.0; MergedObjectTokenizer::EOS as usize + 1];
        let outcome = processor
            .mask_logits_with_terminals(
                &mut logits,
                &generated,
                &HashSet::from([MergedObjectTokenizer::EOS]),
                &HashSet::new(),
            )
            .unwrap();

        assert!(!outcome.accepting);
        assert!(outcome.liveness_intervention);
        assert!(!logits[MergedObjectTokenizer::OBJECT as usize].is_finite());
        assert!(logits[b'"' as usize].is_finite());

        generated.extend([TokenId::new(b'"' as u32), TokenId::new(b'}' as u32)]);
        let progress = processor
            .progress_with_terminals(&generated, &HashSet::from([MergedObjectTokenizer::EOS]))
            .unwrap();
        assert!(progress.accepting);
        assert_eq!(progress.liveness_identical_token_limit, 32);
        assert_eq!(progress.liveness_intervention_count, 1);
    }

    #[test]
    fn structured_liveness_guard_preserves_the_only_finite_grammar_candidate() {
        let processor = StructuredOutputFactory::new(Arc::new(MergedObjectTokenizer::new()))
            .unwrap()
            .create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::Immediate,
                64,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let mut generated =
            br#"{"marker":""#.iter().map(|byte| TokenId::new(*byte as u32)).collect::<Vec<_>>();
        generated.extend(std::iter::repeat_n(
            TokenId::new(MergedObjectTokenizer::OBJECT),
            32,
        ));
        let mut logits = vec![f32::NEG_INFINITY; MergedObjectTokenizer::EOS as usize + 1];
        logits[MergedObjectTokenizer::OBJECT as usize] = 0.0;

        let outcome = processor
            .mask_logits_with_terminals(
                &mut logits,
                &generated,
                &HashSet::from([MergedObjectTokenizer::EOS]),
                &HashSet::new(),
            )
            .unwrap();

        assert!(!outcome.liveness_intervention);
        assert!(logits[MergedObjectTokenizer::OBJECT as usize].is_finite());
    }

    #[test]
    fn structured_liveness_limit_is_derived_from_the_guaranteed_result_budget() {
        let budget = StructuredOutputBudgetPlan::automatic(4096, 1).unwrap();
        assert_eq!(budget.structured_reserve_tokens, 1024);
        assert_eq!(
            StructuredOutputLivenessPolicy::for_request(4096, Some(budget)).max_identical_token_run,
            512
        );
        assert_eq!(
            StructuredOutputLivenessPolicy::for_request(64, None).max_identical_token_run,
            32
        );
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
    fn request_cannot_override_compact_json_compiler_policy() {
        let schema = r#"{
            "type":"object",
            "properties":{"answer":{"const":42}},
            "required":["answer"],
            "additionalProperties":false,
            "x-guidance":{
                "item_separator":", ",
                "key_separator":": ",
                "whitespace_flexible":true
            }
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
        assert_and_append(&processor, &mut generated, r#"{"answer":"#);
        let mut logits = vec![0.0; EOS as usize + 1];
        processor.mask_logits(&mut logits, &generated).unwrap();

        assert!(!logits[b' ' as usize].is_finite());
        assert!(logits[b'4' as usize].is_finite());
        assert_and_append(&processor, &mut generated, "42}");
        assert!(processor.is_accepting(&generated).unwrap());
    }

    #[test]
    fn boolean_json_schema_keeps_its_semantics_under_compact_policy() {
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonSchema("true".to_string()),
                &StructuredOutputStart::Immediate,
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let mut generated = Vec::new();
        assert_and_append(&processor, &mut generated, "true");
        assert!(processor.is_accepting(&generated).unwrap());
    }

    #[test]
    fn terminal_progress_classifies_an_unclosed_number_without_retaining_text() {
        let processor = factory()
            .create_processor(
                &ResponseFormat::JsonSchema(
                    r#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}"#
                        .to_string(),
                ),
                &StructuredOutputStart::Immediate,
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();
        let generated =
            r#"{"value":123777"#.bytes().map(|byte| TokenId::new(byte as u32)).collect::<Vec<_>>();
        let progress = processor
            .progress_with_terminals(&generated, &HashSet::new())
            .unwrap();

        assert_eq!(progress.phase, StructuredOutputPhase::EnforcingGrammar);
        assert_eq!(progress.grammar_token_count, generated.len());
        assert_eq!(
            progress.trailing_token_class,
            Some(StructuredOutputTokenClass::Number)
        );
        assert_eq!(progress.trailing_token_class_count, 6);
        assert_eq!(progress.trailing_token_id, Some(b'7' as u32));
        assert_eq!(progress.trailing_identical_token_count, 3);
        assert!(!progress.accepting);
    }

    #[test]
    fn lexical_diagnostics_classify_json_bytes_without_decoding_content() {
        assert_eq!(
            classify_json_token_bytes(b" \n\t"),
            StructuredOutputTokenClass::Whitespace
        );
        assert_eq!(
            classify_json_token_bytes(b"-12.5e+3"),
            StructuredOutputTokenClass::Number
        );
        assert_eq!(
            classify_json_token_bytes(br#"{}[],:"#),
            StructuredOutputTokenClass::Structural
        );
        assert_eq!(
            classify_json_token_bytes(br#"\""#),
            StructuredOutputTokenClass::StringBoundary
        );
        assert_eq!(
            classify_json_token_bytes(b"true"),
            StructuredOutputTokenClass::Literal
        );
        assert_eq!(
            classify_json_token_bytes(br#""value":"#),
            StructuredOutputTokenClass::Other
        );
    }

    struct FragmentedUtf8Tokenizer {
        special: SpecialTokens,
        token_text: Vec<String>,
    }

    impl FragmentedUtf8Tokenizer {
        const FIRE_HEAD: u32 = 128;
        const FIRE_TAIL: u32 = 129;
        const EOS: u32 = 130;

        fn new() -> Self {
            let mut token_text = (0u8..=127)
                .map(|byte| (byte as char).to_string())
                .collect::<Vec<_>>();
            token_text.extend(["\u{fffd}".to_string(), "\u{fffd}".to_string()]);
            token_text.push("<eos>".to_string());
            Self {
                special: SpecialTokens {
                    eos_token: Some(TokenId::new(Self::EOS)),
                    ..SpecialTokens::default()
                },
                token_text,
            }
        }

        fn raw_bytes(token: TokenId) -> Option<Vec<u8>> {
            match token.get() {
                byte @ 0..=127 => Some(vec![byte as u8]),
                Self::FIRE_HEAD => Some(vec![0xf0, 0x9f]),
                Self::FIRE_TAIL => Some(vec![0x94, 0xa5]),
                _ => None,
            }
        }
    }

    impl Tokenizer for FragmentedUtf8Tokenizer {
        fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
            let mut tokens = Vec::new();
            let mut bytes = text.as_bytes();
            while let Some((&byte, remaining)) = bytes.split_first() {
                if bytes.starts_with(&[0xf0, 0x9f, 0x94, 0xa5]) {
                    tokens.push(TokenId::new(Self::FIRE_HEAD));
                    tokens.push(TokenId::new(Self::FIRE_TAIL));
                    bytes = &bytes[4..];
                } else if byte <= 127 {
                    tokens.push(TokenId::new(byte as u32));
                    bytes = remaining;
                } else {
                    return Err(FerrumError::tokenizer(
                        "fragmented UTF-8 test tokenizer received unsupported input",
                    ));
                }
            }
            Ok(tokens)
        }

        fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
            let bytes = tokens
                .iter()
                .filter_map(|token| Self::raw_bytes(*token))
                .flatten()
                .collect::<Vec<_>>();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
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
            (text.len() == 1 && text.is_ascii()).then(|| TokenId::new(text.as_bytes()[0] as u32))
        }

        fn token_text(&self, token_id: TokenId) -> Option<&str> {
            self.token_text
                .get(token_id.get() as usize)
                .map(String::as_str)
        }

        fn token_bytes(&self, token_id: TokenId) -> Option<Vec<u8>> {
            Self::raw_bytes(token_id)
        }

        fn info(&self) -> TokenizerInfo {
            TokenizerInfo {
                tokenizer_type: TokenizerType::BPE,
                vocab_size: self.vocab_size(),
                special_tokens: self.special.clone(),
                supports_incremental: true,
                supports_chat_template: false,
                max_token_length: Some(2),
                model_name: Some("fragmented-utf8-test".to_string()),
            }
        }
    }

    #[test]
    fn strict_schema_accepts_utf8_split_across_byte_level_tokens() {
        let tokenizer = Arc::new(FragmentedUtf8Tokenizer::new());
        assert!(tokenizer
            .decode(&[TokenId::new(FragmentedUtf8Tokenizer::FIRE_HEAD)], false)
            .unwrap()
            .contains('\u{fffd}'));
        let processor = StructuredOutputFactory::new(tokenizer)
            .unwrap()
            .create_processor(
                &ResponseFormat::JsonSchema(
                    r#"{"type":"object","properties":{"value":{"const":"\ud83d\udd25"}},"required":["value"],"additionalProperties":false}"#
                        .to_string(),
                ),
                &StructuredOutputStart::Immediate,
                TEST_MAX_OUTPUT_TOKENS,
                &HashSet::new(),
                &[],
            )
            .unwrap()
            .unwrap();

        let mut generated = Vec::new();
        for token in r#"{"value":""#
            .bytes()
            .map(|byte| TokenId::new(byte as u32))
            .chain([
                TokenId::new(FragmentedUtf8Tokenizer::FIRE_HEAD),
                TokenId::new(FragmentedUtf8Tokenizer::FIRE_TAIL),
            ])
            .chain(r#""}"#.bytes().map(|byte| TokenId::new(byte as u32)))
        {
            let mut logits = vec![0.0; FragmentedUtf8Tokenizer::EOS as usize + 1];
            processor.mask_logits(&mut logits, &generated).unwrap();
            assert!(
                logits[token.get() as usize].is_finite(),
                "token {} rejected after {:?}",
                token.get(),
                generated
            );
            generated.push(token);
        }
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
        assert_eq!(waiting.grammar_start_token_index, None);
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
        assert_eq!(partial.grammar_start_token_index, None);
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
        let active = processor
            .mask_logits_with_terminals(&mut logits, &generated, &HashSet::new(), &HashSet::new())
            .unwrap();
        assert_eq!(active.phase, StructuredOutputPhase::EnforcingGrammar);
        assert_eq!(active.grammar_start_token_index, Some(27));
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
            assert_eq!(outcome.grammar_start_token_index, None);
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
    fn reasoning_budget_reserves_half_of_a_normal_completion_for_structure() {
        assert_eq!(
            StructuredOutputBudgetPlan::automatic(1024, 1).unwrap(),
            StructuredOutputBudgetPlan {
                total_output_tokens: 1024,
                reasoning_token_limit: 511,
                boundary_token_count: 1,
                structured_reserve_tokens: 512,
            }
        );
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
