//! A single grammar owns reasoning, protocol framing, and either a final
//! schema value or an explicit automatic tool call. Tokens may span rules.
use super::*;
use ferrum_types::{ApiChatRequest, ModelOutputProtocol};

mod compiler;
#[cfg(test)]
mod tests;

#[derive(Clone)]
struct Header {
    bytes: Vec<u8>,
    reasoning: bool,
}

impl Header {
    fn result(text: &str) -> Self {
        Self {
            bytes: text.as_bytes().to_vec(),
            reasoning: false,
        }
    }
    fn reasoning(text: &str) -> Self {
        Self {
            bytes: text.as_bytes().to_vec(),
            reasoning: true,
        }
    }
}

struct Plan {
    headers: Vec<Header>,
    following_headers: Vec<Header>,
    reasoning_close: Vec<u8>,
    wire_token_bytes: Arc<[Vec<u8>]>,
    marker_bytes: HashMap<u32, Vec<u8>>,
    boundary_controls: HashSet<u32>,
    native_terminals: HashSet<u32>,
    native: bool,
}

#[derive(Clone, Copy)]
enum Domain {
    Header { offset: usize, following: bool },
    Reasoning { offset: usize, search_from: usize },
    Result { offset: usize },
}

pub(super) struct ComposedState {
    plan: Arc<Plan>,
    initial_matcher: Matcher,
    final_validator: Arc<jsonschema::Validator>,
    wire: Vec<u8>,
    token_ends: Vec<usize>,
    domain: Domain,
    reasoning_start: Option<usize>,
    reasoning_end: Option<usize>,
    terminal_seen: bool,
}

impl ComposedState {
    pub(super) fn reset(&mut self) {
        self.wire.clear();
        self.token_ends.clear();
        self.domain = Domain::Header {
            offset: 0,
            following: false,
        };
        self.reasoning_start = None;
        self.reasoning_end = None;
        self.terminal_seen = false;
        self.observe_domain();
    }

    pub(super) fn is_native(&self) -> bool {
        self.plan.native
    }

    pub(super) fn fresh_matcher(&self) -> Matcher {
        self.initial_matcher.deep_clone()
    }

    pub(super) fn complete_text_len(&self) -> usize {
        self.wire.len()
            - self
                .wire
                .iter()
                .rev()
                .take_while(|byte| byte.is_ascii_whitespace())
                .count()
    }

    fn bytes(&self, token: u32) -> &[u8] {
        self.plan
            .marker_bytes
            .get(&token)
            .map(Vec::as_slice)
            .or_else(|| {
                self.plan
                    .wire_token_bytes
                    .get(token as usize)
                    .map(Vec::as_slice)
            })
            .unwrap_or_default()
    }

    fn token_at(&self, byte_offset: usize) -> usize {
        self.token_ends.partition_point(|end| *end <= byte_offset)
    }

    fn observe(&mut self, token: u32) {
        let bytes = self.bytes(token).to_vec();
        self.wire.extend_from_slice(&bytes);
        self.token_ends.push(self.wire.len());
        self.observe_domain();
    }

    fn observe_domain(&mut self) {
        loop {
            match self.domain {
                Domain::Header { offset, following } => {
                    let candidates = if following {
                        &self.plan.following_headers
                    } else {
                        &self.plan.headers
                    };
                    let remaining = &self.wire[offset..];
                    let Some(header) = candidates
                        .iter()
                        .find(|header| remaining.starts_with(&header.bytes))
                    else {
                        return;
                    };
                    let end = offset + header.bytes.len();
                    if header.reasoning {
                        self.reasoning_start = Some(end);
                        self.domain = Domain::Reasoning {
                            offset: end,
                            search_from: end,
                        };
                    } else {
                        self.domain = Domain::Result { offset: end };
                    }
                }
                Domain::Reasoning {
                    offset,
                    search_from,
                } => {
                    let close = &self.plan.reasoning_close;
                    if close.is_empty() {
                        return;
                    }
                    if let Some(position) = self.wire[search_from..]
                        .windows(close.len())
                        .position(|bytes| bytes == close)
                    {
                        let start = search_from + position;
                        let end = start + close.len();
                        self.reasoning_end = Some(start);
                        self.domain = if self.plan.native {
                            Domain::Header {
                                offset: end,
                                following: true,
                            }
                        } else {
                            Domain::Result { offset: end }
                        };
                    } else {
                        self.domain = Domain::Reasoning {
                            offset,
                            search_from: self
                                .wire
                                .len()
                                .saturating_sub(close.len().saturating_sub(1))
                                .max(offset),
                        };
                        return;
                    }
                }
                Domain::Result { .. } => return,
            }
        }
    }

    pub(super) fn result_start(&self) -> Option<usize> {
        match self.domain {
            Domain::Result { offset } => Some(self.token_at(offset)),
            _ => None,
        }
    }

    fn reasoning_tokens(&self) -> usize {
        self.reasoning_start.map_or(0, |start| {
            let end = self
                .reasoning_end
                .map(|end| self.token_at(end))
                .unwrap_or(self.token_ends.len());
            end.saturating_sub(self.token_at(start))
        })
    }

    fn forcing_suffix(&self, budget: Option<StructuredOutputBudgetPlan>) -> Option<&[u8]> {
        let Domain::Reasoning { offset, .. } = self.domain else {
            return None;
        };
        if !budget.is_some_and(|budget| self.reasoning_tokens() >= budget.reasoning_token_limit) {
            return None;
        }
        let close = &self.plan.reasoning_close;
        let tail = &self.wire[offset..];
        let prefix = (1..close.len().min(tail.len() + 1))
            .rev()
            .find(|length| tail.ends_with(&close[..*length]))
            .unwrap_or(0);
        Some(&close[prefix..])
    }
}

pub(super) fn compile(
    factory: &StructuredOutputFactory,
    response_format: &ResponseFormat,
    start: &StructuredOutputStart,
    max_tokens: usize,
    stop_ids: &HashSet<u32>,
    stop_texts: &[String],
    chat: &ApiChatRequest,
    protocol: ModelOutputProtocol,
) -> Result<StructuredOutputProcessor> {
    let compiled = compiler::build(
        factory,
        response_format,
        start,
        max_tokens,
        stop_ids,
        stop_texts,
        chat,
        protocol,
    )?;
    let matcher = cached_matcher(factory, compiled.grammar)?;
    let final_validator = cached_final_validator(factory, response_format)?;
    let mut composed = ComposedState {
        plan: Arc::new(compiled.plan),
        initial_matcher: matcher.deep_clone(),
        final_validator,
        wire: Vec::new(),
        token_ends: Vec::new(),
        domain: Domain::Header {
            offset: 0,
            following: false,
        },
        reasoning_start: None,
        reasoning_end: None,
        terminal_seen: false,
    };
    composed.observe_domain();
    let grammar_start = composed.result_start();
    Ok(StructuredOutputProcessor {
        state: Mutex::new(ProcessorState {
            matcher,
            activation: Activation::Active,
            initial_activation: Activation::Active,
            consumed: 0,
            boundary_forced: false,
            boundary_start: None,
            grammar_start,
            trailing_grammar_token_id: None,
            trailing_identical_token_count: 0,
            liveness_intervention_count: 0,
            last_liveness_intervention_at: None,
            composed: Some(composed),
        }),
        vocab_size: factory.vocab_size,
        defined_token_ids: Arc::clone(&factory.defined_token_ids),
        json_token_classes: Arc::clone(&factory.json_token_classes),
        budget: compiled.budget,
        liveness: StructuredOutputLivenessPolicy::for_request(max_tokens, compiled.budget),
    })
}

fn cached_matcher(factory: &StructuredOutputFactory, grammar: TopLevelGrammar) -> Result<Matcher> {
    let key = format!(
        "automatic-tools:{}",
        serde_json::to_string(&grammar).map_err(|error| {
            FerrumError::invalid_request(format!("serialize structured tool grammar: {error}"))
        })?
    );
    let mut cache = factory.grammar_templates.lock();
    if let Some(matcher) = cache.get(&key) {
        return Ok(matcher.deep_clone());
    }
    let parser = factory
        .parser_factory
        .create_parser(grammar)
        .map_err(|error| {
            FerrumError::invalid_request(format!("unsupported structured tool grammar: {error}"))
        })?;
    let matcher = Matcher::new(Ok(parser));
    if cache.len() >= MAX_CACHED_GRAMMARS {
        cache.clear();
    }
    cache.insert(key, matcher.deep_clone());
    Ok(matcher)
}

fn acceptance(matcher: &mut Matcher) -> Result<bool> {
    matcher
        .is_accepting()
        .map_err(|error| FerrumError::model(format!("structured tool acceptance failed: {error}")))
}

fn cached_final_validator(
    factory: &StructuredOutputFactory,
    response_format: &ResponseFormat,
) -> Result<Arc<jsonschema::Validator>> {
    let schema = match response_format {
        ResponseFormat::JsonObject => serde_json::json!({"type": "object"}),
        ResponseFormat::JsonSchema(schema) => serde_json::from_str(schema).map_err(|error| {
            FerrumError::invalid_request(format!(
                "response_format.schema is not valid JSON: {error}"
            ))
        })?,
        ResponseFormat::Text => {
            return Err(FerrumError::internal(
                "automatic tool grammar requires a structured response format",
            ));
        }
    };
    let key = serde_json::to_string(&schema).map_err(|error| {
        FerrumError::invalid_request(format!("serialize structured final schema: {error}"))
    })?;
    let mut cache = factory.schema_validators.lock();
    if let Some(validator) = cache.get(&key) {
        return Ok(Arc::clone(validator));
    }
    // Match the server's full-schema validator. The sampling compiler's ordered
    // properties and whitespace policy must not change a value's final/tool role.
    let validator = Arc::new(jsonschema::validator_for(&schema).map_err(|error| {
        FerrumError::invalid_request(format!("invalid structured final schema: {error}"))
    })?);
    if cache.len() >= MAX_CACHED_GRAMMARS {
        cache.clear();
    }
    cache.insert(key, Arc::clone(&validator));
    Ok(validator)
}

pub(super) fn classify(
    state: &mut ProcessorState,
    generated: &[TokenId],
    terminals: &HashSet<u32>,
) -> Result<Option<(ferrum_types::StructuredOutputBranch, String)>> {
    advance(state, generated, Some(terminals))?;
    if !acceptance(&mut state.matcher)? {
        return Ok(None);
    }
    let composed = state.composed.as_ref().expect("composed processor");
    if composed.is_native() {
        return Ok(None);
    }
    let Domain::Result { offset } = composed.domain else {
        return Err(FerrumError::internal(
            "complete structured tool grammar has no result boundary",
        ));
    };
    let payload = std::str::from_utf8(&composed.wire[offset..composed.complete_text_len()])
        .map_err(|error| FerrumError::model(format!("structured result is not UTF-8: {error}")))?
        .to_string();
    // The union has already accepted a complete root. Prefer semantic final
    // schema membership even when the tool branch supplied its generation path.
    let branch = if serde_json::from_str::<serde_json::Value>(&payload)
        .is_ok_and(|value| composed.final_validator.is_valid(&value))
    {
        ferrum_types::StructuredOutputBranch::Final
    } else {
        ferrum_types::StructuredOutputBranch::ToolCall
    };
    Ok(Some((branch, payload)))
}

pub(super) fn advance(
    state: &mut ProcessorState,
    generated: &[TokenId],
    terminals: Option<&HashSet<u32>>,
) -> Result<()> {
    if state.consumed > generated.len() {
        return Err(FerrumError::internal(
            "structured-output token history moved backwards without reset",
        ));
    }
    let composed = state.composed.as_mut().expect("composed processor");
    for token in &generated[state.consumed..] {
        if composed.terminal_seen {
            return Err(FerrumError::model(
                "structured tool result continued after its terminal",
            ));
        }
        // Ordinary EOS is external framing after a complete text/XML root.
        // Harmony handoff/return are grammar-owned and must be consumed.
        if !composed.plan.native
            && terminals.is_some_and(|ids| ids.contains(&token.get()))
            && acceptance(&mut state.matcher)?
        {
            composed.terminal_seen = true;
            composed.token_ends.push(composed.wire.len());
        } else {
            state.matcher.consume_token(token.get()).map_err(|error| {
                FerrumError::model(format!(
                    "structured-output token {} violated the tool/final grammar: {error}",
                    token.get()
                ))
            })?;
            composed.observe(token.get());
        }
        let result_start = composed.result_start();
        if state.grammar_start != result_start {
            state.trailing_grammar_token_id = None;
            state.trailing_identical_token_count = 0;
        }
        state.grammar_start = result_start;
        if state.grammar_start.is_some() {
            if state.trailing_grammar_token_id == Some(token.get()) {
                state.trailing_identical_token_count += 1;
            } else {
                state.trailing_grammar_token_id = Some(token.get());
                state.trailing_identical_token_count = 1;
            }
        }
        state.boundary_start = composed
            .reasoning_end
            .map(|offset| composed.token_at(offset));
    }
    state.consumed = generated.len();
    Ok(())
}

pub(super) fn mask(
    processor: &StructuredOutputProcessor,
    state: &mut ProcessorState,
    logits: &mut [f32],
    generated: &[TokenId],
    terminals: Option<&HashSet<u32>>,
    hidden_controls: Option<&HashSet<u32>>,
) -> Result<StructuredOutputMaskOutcome> {
    advance(state, generated, terminals)?;
    let accepting = acceptance(&mut state.matcher)?;
    let grammar_mask = state
        .matcher
        .compute_mask_or_eos()
        .map_err(|error| FerrumError::model(format!("structured tool mask failed: {error}")))?;
    let composed = state.composed.as_ref().expect("composed processor");
    let forcing = composed.forcing_suffix(processor.budget);
    state.boundary_forced |= forcing.is_some();
    let mut finite = 0usize;
    let mut forced_candidate = None;
    let mut native_terminal = None;
    let mut delimiter_candidates = Vec::new();
    for (index, logit) in logits.iter_mut().enumerate() {
        let id = index as u32;
        let external_terminal =
            !composed.plan.native && accepting && terminals.is_some_and(|ids| ids.contains(&id));
        let mut allowed = processor
            .defined_token_ids
            .get(index)
            .copied()
            .unwrap_or(false)
            && (grammar_mask.is_allowed(id) || external_terminal);
        if hidden_controls.is_some_and(|controls| controls.contains(&id))
            && !composed.plan.boundary_controls.contains(&id)
            && !external_terminal
        {
            allowed = false;
        }
        if let Some(suffix) = forcing {
            let bytes = composed.bytes(id);
            allowed &=
                !bytes.is_empty() && (suffix.starts_with(bytes) || bytes.starts_with(suffix));
        }
        if allowed {
            if composed.plan.native_terminals.contains(&id) && !accepting {
                if native_terminal.replace(id).is_some() {
                    return Err(FerrumError::internal(
                        "structured tool grammar allowed ambiguous native terminals",
                    ));
                }
            }
            if composed.plan.boundary_controls.contains(&id)
                && !composed.plan.native_terminals.contains(&id)
            {
                delimiter_candidates.push(id);
            }
            if forcing.is_some() && forced_candidate.is_none() {
                forced_candidate = Some(index);
            }
            if logit.is_finite() {
                finite += 1;
            }
        } else {
            *logit = f32::NEG_INFINITY;
        }
    }
    if finite == 0 {
        if let Some(index) = forced_candidate {
            logits[index] = 0.0;
            finite = 1;
        } else {
            return Err(FerrumError::model(
                "structured tool/final grammar has no legal finite token",
            ));
        }
    }
    let liveness_intervention = if state.grammar_start.is_some()
        && !accepting
        && state.trailing_identical_token_count >= processor.liveness.max_identical_token_run
    {
        state
            .trailing_grammar_token_id
            .and_then(|id| logits.get_mut(id as usize))
            .is_some_and(|logit| {
                if logit.is_finite() && finite > 1 {
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
        phase: if state.grammar_start.is_some() {
            StructuredOutputPhase::EnforcingGrammar
        } else if forcing.is_some() {
            StructuredOutputPhase::ForcingDelimiter
        } else {
            StructuredOutputPhase::WaitingForDelimiter
        },
        accepting,
        liveness_intervention,
        grammar_start_token_index: state.grammar_start,
        required_delimiter_token_id: (delimiter_candidates.len() == 1)
            .then(|| delimiter_candidates[0]),
        grammar_owned_terminal_token_id: native_terminal,
    })
}

pub(super) fn progress(
    processor: &StructuredOutputProcessor,
    state: &mut ProcessorState,
    generated: &[TokenId],
    terminals: Option<&HashSet<u32>>,
) -> Result<StructuredOutputProgress> {
    advance(state, generated, terminals)?;
    let accepting = acceptance(&mut state.matcher)?;
    let composed = state.composed.as_ref().expect("composed processor");
    let forcing = composed.forcing_suffix(processor.budget).is_some();
    state.boundary_forced |= forcing;
    let grammar_tokens = state
        .grammar_start
        .and_then(|index| generated.get(index..))
        .unwrap_or_default();
    let trailing = grammar_tokens.last().map(|token| token.get());
    let class_of = |token: u32| {
        processor
            .json_token_classes
            .get(token as usize)
            .copied()
            .unwrap_or(StructuredOutputTokenClass::Undefined)
    };
    let class = trailing.map(class_of);
    let class_count = class.map_or(0, |class| {
        grammar_tokens
            .iter()
            .rev()
            .take_while(|token| class_of(token.get()) == class)
            .count()
    });
    Ok(StructuredOutputProgress {
        phase: if state.grammar_start.is_some() {
            StructuredOutputPhase::EnforcingGrammar
        } else if forcing {
            StructuredOutputPhase::ForcingDelimiter
        } else {
            StructuredOutputPhase::WaitingForDelimiter
        },
        generated_token_count: generated.len(),
        consumed_token_count: state.consumed,
        delimiter_token_count: processor.budget.map(|budget| budget.boundary_token_count),
        delimiter_prefix_token_count: match composed.domain {
            Domain::Header { offset, .. } => composed.token_ends.len() - composed.token_at(offset),
            Domain::Reasoning { offset, .. } => {
                let close = &composed.plan.reasoning_close;
                let tail = &composed.wire[offset..];
                let prefix_bytes = (1..close.len().min(tail.len() + 1))
                    .rev()
                    .find(|length| tail.ends_with(&close[..*length]))
                    .unwrap_or(0);
                composed.token_ends.len() - composed.token_at(composed.wire.len() - prefix_bytes)
            }
            Domain::Result { .. } => 0,
        },
        reasoning_token_count: processor.budget.map(|_| composed.reasoning_tokens()),
        boundary_forced: state.boundary_forced,
        budget: processor.budget,
        grammar_token_count: grammar_tokens.len(),
        trailing_token_class: class,
        trailing_token_class_count: class_count,
        trailing_token_id: trailing,
        trailing_identical_token_count: trailing.map_or(0, |id| {
            grammar_tokens
                .iter()
                .rev()
                .take_while(|token| token.get() == id)
                .count()
        }),
        liveness_identical_token_limit: processor.liveness.max_identical_token_run,
        liveness_intervention_count: state.liveness_intervention_count,
        accepting,
    })
}
