use super::*;
use ferrum_interfaces::tokenizer::TokenizerInfo;

// Channel controls deliberately live inside the base vocabulary. The header
// word is ordinary multi-token text, as it can be in a real model tokenizer.
struct EnvelopeTokenizer(super::tests::ByteTokenizer);

impl EnvelopeTokenizer {
    const OPEN: u32 = 2;
    const CLOSE: u32 = 3;
    const EOS: u32 = 256;
    const OPENING: &'static str = "<|channel>thought\n";
    const CLOSING: &'static str = "<channel|>";

    fn new() -> Self {
        Self(super::tests::ByteTokenizer::new())
    }

    fn controls() -> HashSet<u32> {
        HashSet::from([Self::OPEN, Self::CLOSE, Self::EOS])
    }

    fn start(allow_reasoning: bool) -> StructuredOutputStart {
        StructuredOutputStart::AfterReasoningEnvelope {
            opening: Self::OPENING.to_string(),
            closing: Self::CLOSING.to_string(),
            allow_reasoning,
        }
    }

    fn processor(allow_reasoning: bool, max_tokens: usize) -> StructuredOutputProcessor {
        StructuredOutputFactory::new(Arc::new(Self::new()))
            .unwrap()
            .create_processor(
                &ResponseFormat::JsonObject,
                &Self::start(allow_reasoning),
                max_tokens,
                &HashSet::from([Self::EOS]),
                &[],
            )
            .unwrap()
            .unwrap()
    }
}

impl Tokenizer for EnvelopeTokenizer {
    fn encode(&self, mut text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        while !text.is_empty() {
            if let Some((marker, token)) = [("<|channel>", Self::OPEN), ("<channel|>", Self::CLOSE)]
                .into_iter()
                .find(|(marker, _)| text.starts_with(*marker))
            {
                tokens.push(TokenId::new(token));
                text = &text[marker.len()..];
            } else {
                let next = text.chars().next().unwrap();
                tokens.extend(self.0.encode(&next.to_string(), false)?);
                text = &text[next.len_utf8()..];
            }
        }
        Ok(tokens)
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let mut text = String::new();
        for token in tokens {
            match token.get() {
                Self::OPEN => text.push_str("<|channel>"),
                Self::CLOSE => text.push_str("<channel|>"),
                _ => text.push_str(&self.0.decode(&[*token], skip_special)?),
            }
        }
        Ok(text)
    }

    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        self.decode(&[next], true)
    }

    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        self.0.special_tokens()
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        match text {
            "<|channel>" => Some(TokenId::new(Self::OPEN)),
            "<channel|>" => Some(TokenId::new(Self::CLOSE)),
            _ => self.0.token_id(text),
        }
    }

    fn token_text(&self, token: TokenId) -> Option<&str> {
        match token.get() {
            Self::OPEN => Some("<|channel>"),
            Self::CLOSE => Some("<channel|>"),
            _ => self.0.token_text(token),
        }
    }

    fn info(&self) -> TokenizerInfo {
        self.0.info()
    }
}

fn mask(
    processor: &StructuredOutputProcessor,
    generated: &[TokenId],
) -> (Vec<f32>, StructuredOutputMaskOutcome) {
    let mut logits = vec![0.0; EnvelopeTokenizer::EOS as usize + 1];
    // Reproduce the engine's pre-existing rejection of hidden control IDs.
    logits[EnvelopeTokenizer::OPEN as usize] = f32::NEG_INFINITY;
    logits[EnvelopeTokenizer::CLOSE as usize] = f32::NEG_INFINITY;
    let outcome = processor
        .mask_logits_with_terminals(
            &mut logits,
            generated,
            &HashSet::from([EnvelopeTokenizer::EOS]),
            &EnvelopeTokenizer::controls(),
        )
        .unwrap();
    (logits, outcome)
}

#[test]
fn reasoning_envelope_forces_empty_native_header_before_json() {
    let tokenizer = EnvelopeTokenizer::new();
    let processor = EnvelopeTokenizer::processor(false, 64);
    let mut generated = Vec::new();
    let envelope = tokenizer
        .encode(
            &format!(
                "{}{}",
                EnvelopeTokenizer::OPENING,
                EnvelopeTokenizer::CLOSING
            ),
            false,
        )
        .unwrap();
    for expected in &envelope {
        let (logits, outcome) = mask(&processor, &generated);
        assert_eq!(outcome.required_delimiter_token_id, Some(expected.get()));
        assert_eq!(outcome.phase, StructuredOutputPhase::ForcingDelimiter);
        assert_eq!(outcome.grammar_start_token_index, None);
        assert_eq!(
            logits
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(token, _)| token as u32)
                .collect::<Vec<_>>(),
            vec![expected.get()]
        );
        generated.push(*expected);
    }
    let (_, outcome) = mask(&processor, &generated);
    assert_eq!(outcome.grammar_start_token_index, Some(envelope.len()));
    for token in tokenizer.encode(r#"{"ok":1}"#, false).unwrap() {
        let (logits, _) = mask(&processor, &generated);
        assert!(logits[token.get() as usize].is_finite());
        assert!(!logits[EnvelopeTokenizer::OPEN as usize].is_finite());
        assert!(!logits[EnvelopeTokenizer::CLOSE as usize].is_finite());
        generated.push(token);
    }
    let terminals = HashSet::from([EnvelopeTokenizer::EOS]);
    let progress = processor
        .progress_with_terminals(&generated, &terminals)
        .unwrap();
    assert!(progress.accepting);
    assert_eq!(progress.reasoning_token_count, Some(0));
    assert_eq!(progress.budget.unwrap().reasoning_token_limit, 0);
    let (logits, _) = mask(&processor, &generated);
    assert!(logits[EnvelopeTokenizer::EOS as usize].is_finite());
    processor.reset().unwrap();
    let replayed = processor
        .progress_with_terminals(&generated, &terminals)
        .unwrap();
    assert!(replayed.accepting);
    assert_eq!(replayed, progress);
    assert_eq!(
        processor
            .progress_with_terminals(&generated, &terminals)
            .unwrap(),
        replayed
    );
}

#[test]
fn reasoning_envelope_budget_reserves_both_boundaries_and_allows_reasoning() {
    let tokenizer = EnvelopeTokenizer::new();
    let processor = EnvelopeTokenizer::processor(true, 64);
    let mut generated = tokenizer.encode(EnvelopeTokenizer::OPENING, false).unwrap();
    let opening_len = generated.len();
    let progress = processor
        .progress_with_terminals(&generated, &HashSet::new())
        .unwrap();
    let budget = progress.budget.unwrap();
    assert_eq!(budget.boundary_token_count, opening_len + 1);
    assert_eq!(
        budget.reasoning_token_limit
            + budget.boundary_token_count
            + budget.structured_reserve_tokens,
        64
    );
    assert!(budget.reasoning_token_limit > 0);
    for _ in 0..budget.reasoning_token_limit {
        let (logits, outcome) = mask(&processor, &generated);
        assert_eq!(outcome.phase, StructuredOutputPhase::WaitingForDelimiter);
        assert!(logits[b'x' as usize].is_finite());
        assert!(!logits[EnvelopeTokenizer::OPEN as usize].is_finite());
        generated.push(TokenId::new(b'x' as u32));
    }
    let (logits, outcome) = mask(&processor, &generated);
    assert_eq!(outcome.phase, StructuredOutputPhase::ForcingDelimiter);
    assert_eq!(
        outcome.required_delimiter_token_id,
        Some(EnvelopeTokenizer::CLOSE)
    );
    assert!(logits[EnvelopeTokenizer::CLOSE as usize].is_finite());
    assert!(!logits[b'x' as usize].is_finite());
    generated.push(TokenId::new(EnvelopeTokenizer::CLOSE));
    let (_, outcome) = mask(&processor, &generated);
    assert_eq!(outcome.grammar_start_token_index, Some(generated.len()));
    let progress = processor
        .progress_with_terminals(&generated, &HashSet::new())
        .unwrap();
    assert_eq!(
        progress.reasoning_token_count,
        Some(budget.reasoning_token_limit)
    );
    assert!(progress.boundary_forced);
    processor.reset().unwrap();
    assert_eq!(
        processor
            .progress_with_terminals(&generated, &HashSet::new())
            .unwrap(),
        progress
    );

    processor.reset().unwrap();
    let mut early = tokenizer.encode(EnvelopeTokenizer::OPENING, false).unwrap();
    early.push(TokenId::new(b'x' as u32));
    let mut logits = vec![0.0; EnvelopeTokenizer::EOS as usize + 1];
    let outcome = processor
        .mask_logits_with_terminals(
            &mut logits,
            &early,
            &HashSet::from([EnvelopeTokenizer::EOS]),
            &EnvelopeTokenizer::controls(),
        )
        .unwrap();
    assert_eq!(outcome.phase, StructuredOutputPhase::WaitingForDelimiter);
    assert!(logits[EnvelopeTokenizer::CLOSE as usize].is_finite());
    early.extend(tokenizer.encode("<channel|>{}", false).unwrap());
    let progress = processor
        .progress_with_terminals(&early, &HashSet::new())
        .unwrap();
    assert!(progress.accepting);
    assert_eq!(progress.reasoning_token_count, Some(1));
    assert!(!progress.boundary_forced);
}

#[test]
fn reasoning_envelope_closing_prefix_excludes_header_and_replays_budget_forcing() {
    let tokenizer = super::tests::ByteTokenizer::new();
    let processor = StructuredOutputFactory::new(Arc::new(tokenizer))
        .unwrap()
        .create_processor(
            &ResponseFormat::JsonObject,
            &StructuredOutputStart::AfterReasoningEnvelope {
                opening: "ABX".to_string(),
                closing: "XY".to_string(),
                allow_reasoning: true,
            },
            40,
            &HashSet::new(),
            &[],
        )
        .unwrap()
        .unwrap();
    let mut generated = "ABX"
        .bytes()
        .map(|byte| TokenId::new(byte as u32))
        .collect::<Vec<_>>();
    let (_, outcome) = mask(&processor, &generated);
    assert_eq!(outcome.required_delimiter_token_id, Some(b'X' as u32));
    let budget = processor
        .progress_with_terminals(&generated, &HashSet::new())
        .unwrap()
        .budget
        .unwrap();
    // The first closing token reaches the budget. The final token must then
    // be forced, including after a reset and whole-history replay.
    generated.extend(std::iter::repeat_n(
        TokenId::new(b'r' as u32),
        budget.reasoning_token_limit - 1,
    ));
    generated.push(TokenId::new(b'X' as u32));
    let (_, outcome) = mask(&processor, &generated);
    assert_eq!(outcome.required_delimiter_token_id, Some(b'Y' as u32));
    assert_eq!(outcome.phase, StructuredOutputPhase::ForcingDelimiter);
    generated.push(TokenId::new(b'Y' as u32));
    let progress = processor
        .progress_with_terminals(&generated, &HashSet::new())
        .unwrap();
    assert!(progress.boundary_forced);
    assert_eq!(
        progress.reasoning_token_count,
        Some(budget.reasoning_token_limit - 1)
    );
    processor.reset().unwrap();
    assert_eq!(
        processor
            .progress_with_terminals(&generated, &HashSet::new())
            .unwrap(),
        progress
    );
}

#[test]
fn reasoning_envelope_replay_rejects_missing_or_malformed_opening() {
    let tokenizer = EnvelopeTokenizer::new();
    for text in [
        "<channel|>{}",
        "<|channel>wrong\n<channel|>{}",
        "<|channel>thought\nx<channel|>{}",
    ] {
        let processor = EnvelopeTokenizer::processor(false, 64);
        let generated = tokenizer.encode(text, false).unwrap();
        assert!(
            processor
                .progress_with_terminals(&generated, &HashSet::new())
                .is_err(),
            "accepted {text:?}"
        );
    }
    let processor = EnvelopeTokenizer::processor(true, 64);
    let partial = tokenizer.encode("<|channel>th", false).unwrap();
    assert!(
        !processor
            .progress_with_terminals(&partial, &HashSet::new())
            .unwrap()
            .accepting
    );
    let (_, outcome) = mask(&processor, &partial);
    assert_eq!(outcome.required_delimiter_token_id, Some(b'o' as u32));
}

#[test]
fn reasoning_envelope_rejects_insufficient_budget_and_either_boundary_stop() {
    let tokenizer = EnvelopeTokenizer::new();
    let opening_len = tokenizer
        .encode(EnvelopeTokenizer::OPENING, false)
        .unwrap()
        .len();
    let factory = StructuredOutputFactory::new(Arc::new(tokenizer)).unwrap();
    let make =
        |start: &StructuredOutputStart, max_tokens, stops: &HashSet<u32>, text_stops: &[String]| {
            factory.create_processor(
                &ResponseFormat::JsonObject,
                start,
                max_tokens,
                stops,
                text_stops,
            )
        };
    for allow in [false, true] {
        let start = EnvelopeTokenizer::start(allow);
        assert!(make(&start, opening_len + 1, &HashSet::new(), &[]).is_err());
        for stop in [EnvelopeTokenizer::OPEN, EnvelopeTokenizer::CLOSE] {
            assert!(make(&start, 64, &HashSet::from([stop]), &[]).is_err());
        }
        for stop in ["thought", EnvelopeTokenizer::CLOSING, "thought\n<channel|>"] {
            assert!(make(&start, 64, &HashSet::new(), &[stop.to_string()]).is_err());
        }
    }
}
