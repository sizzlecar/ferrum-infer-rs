//! The fixed Harmony envelope around a grammar-owned final JSON value.
//! This handles neither tool calls nor JSON parsing; the parent owns the matcher.

use super::{force_exact_token, StructuredOutputBudgetPlan, StructuredOutputPhase};
use ferrum_interfaces::tokenizer::Tokenizer;
use ferrum_types::{FerrumError, ModelOutputProtocol, Result, TokenId};
use std::{collections::HashSet, sync::Arc};

const ANALYSIS: u8 = 1;
const FINAL: u8 = 2;

#[derive(Clone)]
struct HarmonyTokens {
    analysis_header: Vec<u32>,
    final_header: Vec<u32>,
    transition: Vec<u32>,
    controls: HashSet<u32>,
    return_token: u32,
}

#[derive(Clone, Copy)]
enum Phase {
    InitialHeader { candidates: u8, cursor: usize },
    AnalysisBody,
    FinalHeader { cursor: usize },
    Payload,
    Finished,
}

#[derive(Clone)]
pub(super) struct HarmonyBoundary {
    tokens: Arc<HarmonyTokens>,
    phase: Phase,
    budget: StructuredOutputBudgetPlan,
    forced: bool,
    boundary_start: usize,
}

impl HarmonyBoundary {
    pub(super) fn compile(
        tokenizer: &(dyn Tokenizer + Send + Sync),
        vocab_size: usize,
        max_tokens: usize,
        stop_ids: &HashSet<u32>,
        stop_texts: &[String],
    ) -> Result<Self> {
        let marker = |text: &str| -> Result<u32> {
            let id = tokenizer
                .token_id(text)
                .ok_or_else(|| {
                    FerrumError::config(format!(
                        "Harmony structured output requires token {text:?}"
                    ))
                })?
                .get();
            if id as usize >= vocab_size {
                return Err(FerrumError::config(format!(
                    "Harmony token {text:?} is outside the model vocabulary"
                )));
            }
            Ok(id)
        };
        let channel = marker("<|channel|>")?;
        let message = marker("<|message|>")?;
        let start = marker("<|start|>")?;
        let end = marker("<|end|>")?;
        let return_token = marker("<|return|>")?;
        if !stop_ids.contains(&return_token) {
            return Err(FerrumError::invalid_request(
                "Harmony structured output requires <|return|> as a resolved terminal token",
            ));
        }
        let word = |text: &str| -> Result<Vec<u32>> {
            let tokens = tokenizer
                .encode(text, false)?
                .into_iter()
                .map(TokenId::get)
                .collect::<Vec<_>>();
            if tokens.is_empty() || tokens.iter().any(|id| *id as usize >= vocab_size) {
                return Err(FerrumError::config(format!(
                    "Harmony header word {text:?} did not tokenize within the model vocabulary"
                )));
            }
            Ok(tokens)
        };
        let mut analysis_header = vec![channel];
        analysis_header.extend(word("analysis")?);
        analysis_header.push(message);
        let mut final_header = vec![channel];
        final_header.extend(word("final")?);
        final_header.push(message);
        let mut transition = vec![end, start];
        transition.extend(word("assistant")?);
        transition.extend_from_slice(&final_header);
        for sequence in [&analysis_header, &final_header, &transition] {
            if let Some(token) = sequence.iter().find(|id| stop_ids.contains(id)) {
                return Err(FerrumError::invalid_request(format!(
                    "Harmony envelope token {token} conflicts with a stop token"
                )));
            }
        }
        for envelope in [
            "<|channel|>analysis<|message|>",
            "<|channel|>final<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
            "<|return|>",
        ] {
            if let Some(stop) = stop_texts
                .iter()
                .find(|stop| !stop.is_empty() && envelope.contains(stop.as_str()))
            {
                return Err(FerrumError::invalid_request(format!(
                    "Harmony envelope conflicts with stop sequence {stop:?}"
                )));
            }
        }

        // The reserve includes only JSON tokens. Keep one further token for
        // <|return|>, even when the completion budget is too small for analysis.
        let mut budget = StructuredOutputBudgetPlan::automatic(max_tokens, final_header.len() + 1)?;
        let analysis_fits =
            analysis_header.len() + transition.len() + budget.structured_reserve_tokens + 1
                <= max_tokens;
        if analysis_fits {
            budget.boundary_token_count = transition.len() + 1;
            budget.reasoning_token_limit =
                max_tokens - budget.boundary_token_count - budget.structured_reserve_tokens;
        }
        let mut controls = stop_ids.clone();
        controls.extend(
            ModelOutputProtocol::HarmonyGptOss
                .preserved_special_token_texts()
                .iter()
                .filter_map(|text| tokenizer.token_id(text))
                .map(TokenId::get),
        );
        controls.extend(tokenizer.special_tokens().eos_token.map(TokenId::get));
        controls.extend(
            tokenizer
                .special_tokens()
                .extra_eos_tokens
                .iter()
                .map(|token| token.get()),
        );
        Ok(Self {
            tokens: Arc::new(HarmonyTokens {
                analysis_header,
                final_header,
                transition,
                controls,
                return_token,
            }),
            phase: Phase::InitialHeader {
                candidates: FINAL | if analysis_fits { ANALYSIS } else { 0 },
                cursor: 0,
            },
            budget,
            forced: false,
            boundary_start: 0,
        })
    }

    pub(super) fn budget(&self) -> StructuredOutputBudgetPlan {
        self.budget
    }
    pub(super) fn in_payload(&self) -> bool {
        matches!(self.phase, Phase::Payload | Phase::Finished)
    }
    pub(super) fn is_forced(&self) -> bool {
        self.forced
    }
    pub(super) fn boundary_start(&self) -> usize {
        self.boundary_start
    }
    pub(super) fn is_terminal(&self, token: u32) -> bool {
        token == self.tokens.return_token
    }
    pub(super) fn is_control(&self, token: u32) -> bool {
        self.tokens.controls.contains(&token)
    }

    pub(super) fn activate_forcing_if_due(&mut self, generated_len: usize) {
        if matches!(self.phase, Phase::AnalysisBody)
            && generated_len >= self.budget.reasoning_token_limit
        {
            self.forced = true;
        }
    }

    /// Observe only framing/reasoning tokens. The returned index is the first
    /// token the parent's JSON matcher must consume, including batched replay.
    pub(super) fn observe(&mut self, token: u32, index: usize) -> Result<Option<usize>> {
        self.activate_forcing_if_due(index);
        match self.phase {
            Phase::InitialHeader {
                mut candidates,
                cursor,
            } => {
                if self.tokens.analysis_header.get(cursor) != Some(&token) {
                    candidates &= !ANALYSIS;
                }
                if self.tokens.final_header.get(cursor) != Some(&token) {
                    candidates &= !FINAL;
                }
                if candidates == 0 {
                    return Err(invalid_envelope());
                }
                let cursor = cursor + 1;
                if candidates & ANALYSIS != 0 && cursor == self.tokens.analysis_header.len() {
                    self.phase = Phase::AnalysisBody;
                } else if candidates & FINAL != 0 && cursor == self.tokens.final_header.len() {
                    self.phase = Phase::Payload;
                    return Ok(Some(index + 1));
                } else {
                    self.phase = Phase::InitialHeader { candidates, cursor };
                }
            }
            Phase::AnalysisBody => {
                if token == self.tokens.transition[0] {
                    self.boundary_start = index;
                    self.phase = Phase::FinalHeader { cursor: 1 };
                } else if self.forced || self.is_control(token) {
                    return Err(invalid_envelope());
                }
            }
            Phase::FinalHeader { cursor } => {
                if self.tokens.transition.get(cursor) != Some(&token) {
                    return Err(invalid_envelope());
                }
                if cursor + 1 == self.tokens.transition.len() {
                    self.phase = Phase::Payload;
                    return Ok(Some(index + 1));
                }
                self.phase = Phase::FinalHeader { cursor: cursor + 1 };
            }
            Phase::Payload | Phase::Finished => {
                return Err(FerrumError::internal(
                    "Harmony framing observer reached the JSON payload",
                ))
            }
        }
        Ok(None)
    }

    pub(super) fn mask_before_payload(
        &self,
        logits: &mut [f32],
        hidden_controls: Option<&HashSet<u32>>,
    ) -> Result<Option<u32>> {
        let next = match self.phase {
            Phase::InitialHeader { candidates, cursor } => [
                (candidates & ANALYSIS != 0).then(|| self.tokens.analysis_header[cursor]),
                (candidates & FINAL != 0).then(|| self.tokens.final_header[cursor]),
            ],
            Phase::FinalHeader { cursor } => [Some(self.tokens.transition[cursor]), None],
            Phase::AnalysisBody => {
                let end = self.tokens.transition[0];
                if self.forced {
                    force_exact_token(logits, end)?;
                } else {
                    for token in self
                        .tokens
                        .controls
                        .iter()
                        .chain(hidden_controls.into_iter().flatten())
                    {
                        if *token != end {
                            if let Some(logit) = logits.get_mut(*token as usize) {
                                *logit = f32::NEG_INFINITY;
                            }
                        }
                    }
                    require_finite(logits)?;
                }
                return Ok(Some(end));
            }
            Phase::Payload => return Ok(None),
            Phase::Finished => return Err(invalid_envelope()),
        };
        if self.forced {
            force_exact_token(logits, next[0].or(next[1]).ok_or_else(invalid_envelope)?)?;
        } else {
            for (index, logit) in logits.iter_mut().enumerate() {
                if !next.contains(&Some(index as u32)) {
                    *logit = f32::NEG_INFINITY;
                }
            }
            require_finite(logits)?;
        }
        Ok(match next {
            [Some(a), Some(b)] if a == b => Some(a),
            [Some(a), None] | [None, Some(a)] => Some(a),
            _ => None,
        })
    }

    /// A model terminal is framing only after a complete value. Other Harmony
    /// channels and EOS tokens cannot terminate a final JSON payload.
    pub(super) fn observe_payload_control(&mut self, token: u32, accepting: bool) -> Result<bool> {
        if matches!(self.phase, Phase::Finished) {
            return Err(invalid_envelope());
        }
        if self.is_terminal(token) && accepting {
            self.phase = Phase::Finished;
            Ok(true)
        } else if self.is_control(token) {
            Err(invalid_envelope())
        } else {
            Ok(false)
        }
    }

    pub(super) fn progress(&self) -> (StructuredOutputPhase, usize, usize) {
        let phase = if self.in_payload() {
            StructuredOutputPhase::EnforcingGrammar
        } else if self.forced {
            StructuredOutputPhase::ForcingDelimiter
        } else {
            StructuredOutputPhase::WaitingForDelimiter
        };
        let (length, prefix) = match self.phase {
            Phase::InitialHeader { candidates, cursor } => (
                if candidates == ANALYSIS {
                    self.tokens.analysis_header.len()
                } else {
                    self.tokens.final_header.len()
                },
                cursor,
            ),
            Phase::FinalHeader { cursor } => (self.tokens.transition.len(), cursor),
            Phase::AnalysisBody => (self.tokens.transition.len(), 0),
            Phase::Payload | Phase::Finished => (
                if self.boundary_start == 0 {
                    self.tokens.final_header.len()
                } else {
                    self.tokens.transition.len()
                },
                0,
            ),
        };
        (phase, length, prefix)
    }
}

fn invalid_envelope() -> FerrumError {
    FerrumError::model("generated tokens violated the Harmony final-response envelope")
}

fn require_finite(logits: &[f32]) -> Result<()> {
    if logits.iter().any(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(FerrumError::model(
            "Harmony envelope has no legal finite token",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use ferrum_interfaces::tokenizer::TokenizerInfo;
    use ferrum_types::{parse_harmony_response, SpecialTokens};

    struct HarmonyTokenizer {
        bytes: super::super::tests::ByteTokenizer,
        special: SpecialTokens,
    }

    impl HarmonyTokenizer {
        const EOS: u32 = 256;
        const CHANNEL: u32 = 257;
        const MESSAGE: u32 = 258;
        const START: u32 = 259;
        const END: u32 = 260;
        const RETURN: u32 = 261;
        const CALL: u32 = 262;
        const CONSTRAIN: u32 = 263;
        const VOCAB: usize = 264;
        const MARKERS: [&'static str; 7] = [
            "<|channel|>",
            "<|message|>",
            "<|start|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|constrain|>",
        ];

        fn new() -> Self {
            Self {
                bytes: super::super::tests::ByteTokenizer::new(),
                special: SpecialTokens {
                    eos_token: Some(TokenId::new(Self::EOS)),
                    extra_eos_tokens: vec![TokenId::new(Self::RETURN), TokenId::new(Self::CALL)],
                    ..SpecialTokens::default()
                },
            }
        }

        fn terminals() -> HashSet<u32> {
            HashSet::from([Self::EOS, Self::RETURN, Self::CALL])
        }

        fn controls() -> HashSet<u32> {
            (Self::EOS..Self::VOCAB as u32).collect()
        }

        fn factory() -> StructuredOutputFactory {
            StructuredOutputFactory::new_with_model_vocab_size(
                Arc::new(Self::new()),
                Some(Self::VOCAB),
            )
            .unwrap()
        }

        fn processor(max_tokens: usize) -> StructuredOutputProcessor {
            Self::factory().create_processor(
                &ResponseFormat::JsonSchema(
                    r#"{"type":"object","properties":{"answer":{"const":42}},"required":["answer"],"additionalProperties":false}"#.to_string(),
                ),
                &StructuredOutputStart::HarmonyFinal,
                max_tokens,
                &Self::terminals(),
                &[],
            ).unwrap().unwrap()
        }
    }

    impl Tokenizer for HarmonyTokenizer {
        fn encode(&self, mut text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
            let mut tokens = Vec::new();
            while !text.is_empty() {
                if let Some((index, marker)) = Self::MARKERS
                    .iter()
                    .enumerate()
                    .find(|(_, marker)| text.starts_with(**marker))
                {
                    tokens.push(TokenId::new(Self::CHANNEL + index as u32));
                    text = &text[marker.len()..];
                } else {
                    let next = text.chars().next().unwrap();
                    tokens.extend(
                        next.to_string()
                            .bytes()
                            .map(|byte| TokenId::new(byte as u32)),
                    );
                    text = &text[next.len_utf8()..];
                }
            }
            Ok(tokens)
        }

        fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
            let mut text = String::new();
            for token in tokens {
                if let Some(marker) = token
                    .get()
                    .checked_sub(Self::CHANNEL)
                    .and_then(|index| Self::MARKERS.get(index as usize))
                {
                    text.push_str(marker);
                } else {
                    text.push_str(&self.bytes.decode(&[*token], skip_special)?);
                }
            }
            Ok(text)
        }

        fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
            self.decode(&[next], true)
        }
        fn vocab_size(&self) -> usize {
            Self::EOS as usize + 1
        }
        fn special_tokens(&self) -> &SpecialTokens {
            &self.special
        }
        fn token_id(&self, text: &str) -> Option<TokenId> {
            Self::MARKERS
                .iter()
                .position(|marker| *marker == text)
                .map(|index| TokenId::new(Self::CHANNEL + index as u32))
                .or_else(|| self.bytes.token_id(text))
        }
        fn token_text(&self, token: TokenId) -> Option<&str> {
            token
                .get()
                .checked_sub(Self::CHANNEL)
                .and_then(|index| Self::MARKERS.get(index as usize).copied())
                .or_else(|| self.bytes.token_text(token))
        }
        fn info(&self) -> TokenizerInfo {
            self.bytes.info()
        }
    }

    fn mask(
        processor: &StructuredOutputProcessor,
        generated: &[TokenId],
    ) -> (Vec<f32>, StructuredOutputMaskOutcome) {
        let mut logits = vec![0.0; HarmonyTokenizer::VOCAB];
        let outcome = processor
            .mask_logits_with_terminals(
                &mut logits,
                generated,
                &HarmonyTokenizer::terminals(),
                &HarmonyTokenizer::controls(),
            )
            .unwrap();
        (logits, outcome)
    }

    fn append(processor: &StructuredOutputProcessor, generated: &mut Vec<TokenId>, text: &str) {
        for token in HarmonyTokenizer::new().encode(text, false).unwrap() {
            let (logits, _) = mask(processor, generated);
            assert!(
                logits[token.get() as usize].is_finite(),
                "token {token:?} rejected after {:?}",
                HarmonyTokenizer::new().decode(generated, true)
            );
            generated.push(token);
        }
    }

    #[test]
    fn harmony_final_preserves_multitoken_headers_and_constrains_only_payload() {
        for prefix in [
            "<|channel|>final<|message|>",
            "<|channel|>analysis<|message|>Free [reasoning].<|end|><|start|>assistant<|channel|>final<|message|>",
        ] {
            let processor = HarmonyTokenizer::processor(128);
            let mut generated = Vec::new();
            let (logits, outcome) = mask(&processor, &generated);
            assert!(logits[HarmonyTokenizer::CHANNEL as usize].is_finite(),
                "Harmony channel must survive before the JSON grammar starts");
            assert!(!logits[b'{' as usize].is_finite());
            assert_eq!(outcome.grammar_start_token_index, None);
            append(&processor, &mut generated, prefix);
            let (logits, outcome) = mask(&processor, &generated);
            assert_eq!(outcome.grammar_start_token_index, Some(generated.len()));
            assert!(logits[b'{' as usize].is_finite());
            assert!(!logits[b'[' as usize].is_finite());
            append(&processor, &mut generated, r#"{"answer":42}<|return|>"#);
            let parsed = parse_harmony_response(&HarmonyTokenizer::new().decode(&generated, true).unwrap()).unwrap();
            assert_eq!(parsed.content, r#"{"answer":42}"#);
            assert_eq!(parsed.reasoning_content.as_deref(), prefix.contains("analysis").then_some("Free [reasoning]."));
        }
    }

    #[test]
    fn harmony_final_forces_a_complete_transition_before_the_payload_reserve() {
        let processor = HarmonyTokenizer::processor(128);
        let mut generated = Vec::new();
        append(&processor, &mut generated, "<|channel|>analysis<|message|>");
        loop {
            let (logits, outcome) = mask(&processor, &generated);
            if outcome.grammar_start_token_index.is_some() {
                break;
            }
            assert!(
                generated.len() < 128,
                "reasoning exhausted the payload budget"
            );
            let token = if outcome.phase == StructuredOutputPhase::ForcingDelimiter {
                let required = outcome
                    .required_delimiter_token_id
                    .expect("forced transition token");
                assert_eq!(logits.iter().filter(|value| value.is_finite()).count(), 1);
                required
            } else {
                assert!(logits[b'x' as usize].is_finite());
                b'x' as u32
            };
            generated.push(TokenId::new(token));
        }
        assert!(HarmonyTokenizer::new()
            .decode(&generated, true)
            .unwrap()
            .ends_with("<|end|><|start|>assistant<|channel|>final<|message|>"));
        let progress = processor
            .progress_with_terminals(&generated, &HarmonyTokenizer::terminals())
            .unwrap();
        assert!(progress.boundary_forced);
        append(&processor, &mut generated, r#"{"answer":42}<|return|>"#);
        assert!(generated.len() <= 128);
        assert!(
            parse_harmony_response(&HarmonyTokenizer::new().decode(&generated, true).unwrap())
                .is_ok()
        );
    }

    #[test]
    fn harmony_final_rejects_incomplete_values_and_non_return_terminals() {
        let processor = HarmonyTokenizer::processor(128);
        let mut generated = Vec::new();
        append(&processor, &mut generated, "<|channel|>final<|message|>{");
        assert!(!processor
            .is_accepting_with_terminals(&generated, &HarmonyTokenizer::terminals())
            .unwrap());
        let (logits, _) = mask(&processor, &generated);
        assert!(HarmonyTokenizer::terminals()
            .iter()
            .all(|id| !logits[*id as usize].is_finite()));
        append(&processor, &mut generated, r#""answer":42}"#);
        let (logits, _) = mask(&processor, &generated);
        assert!(logits[HarmonyTokenizer::RETURN as usize].is_finite());
        for id in [
            HarmonyTokenizer::EOS,
            HarmonyTokenizer::CALL,
            HarmonyTokenizer::CHANNEL,
            HarmonyTokenizer::START,
            HarmonyTokenizer::END,
            HarmonyTokenizer::CONSTRAIN,
        ] {
            assert!(
                !logits[id as usize].is_finite(),
                "invalid final terminal/control {id}"
            );
        }
        let mut invalid = generated.clone();
        invalid.push(TokenId::new(HarmonyTokenizer::CALL));
        assert!(processor
            .progress_with_terminals(&invalid, &HarmonyTokenizer::terminals())
            .is_err());
        processor.reset().unwrap();
        generated.push(TokenId::new(HarmonyTokenizer::RETURN));
        assert!(processor
            .is_accepting_with_terminals(&generated, &HarmonyTokenizer::terminals())
            .unwrap());
        generated.push(TokenId::new(b'x' as u32));
        assert!(processor
            .progress_with_terminals(&generated, &HarmonyTokenizer::terminals())
            .is_err());
    }

    #[test]
    fn harmony_final_fails_closed_for_insufficient_budget_and_stop_conflicts() {
        let factory = HarmonyTokenizer::factory();
        let create = |max_tokens, stops: &HashSet<u32>, texts: &[String]| {
            factory.create_processor(
                &ResponseFormat::JsonObject,
                &StructuredOutputStart::HarmonyFinal,
                max_tokens,
                stops,
                texts,
            )
        };
        let direct_header = HarmonyTokenizer::new()
            .encode("<|channel|>final<|message|>", false)
            .unwrap()
            .len();
        assert!(create(direct_header + 1, &HarmonyTokenizer::terminals(), &[]).is_err());
        assert!(create(128, &HarmonyTokenizer::terminals(), &["final".to_string()]).is_err());
        let mut stops = HarmonyTokenizer::terminals();
        stops.insert(HarmonyTokenizer::MESSAGE);
        assert!(create(128, &stops, &[]).is_err());
        assert!(
            create(128, &HashSet::new(), &[]).is_err(),
            "return must be an engine-resolved terminal"
        );
    }

    #[test]
    fn harmony_final_progress_and_reset_replay_the_payload_boundary_once() {
        let processor = HarmonyTokenizer::processor(128);
        let tokenizer = HarmonyTokenizer::new();
        let prefix = "<|channel|>analysis<|message|>Reason.<|end|><|start|>assistant<|channel|>final<|message|>";
        let payload = r#"{"answer":42}<|return|>"#;
        let generated = tokenizer
            .encode(&format!("{prefix}{payload}"), false)
            .unwrap();
        let first = processor
            .progress_with_terminals(&generated, &HarmonyTokenizer::terminals())
            .unwrap();
        assert!(first.accepting);
        assert_eq!(
            first.grammar_token_count,
            tokenizer.encode(payload, false).unwrap().len()
        );
        let second = processor
            .progress_with_terminals(&generated, &HarmonyTokenizer::terminals())
            .unwrap();
        assert_eq!(first, second);
        processor.reset().unwrap();
        let replayed = processor
            .progress_with_terminals(&generated, &HarmonyTokenizer::terminals())
            .unwrap();
        assert_eq!(first, replayed);
        assert_eq!(replayed.consumed_token_count, generated.len());
    }
}
