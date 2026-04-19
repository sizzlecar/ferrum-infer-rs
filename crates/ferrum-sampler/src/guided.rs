//! Regex-guided decoding — hard token masking via DFA.
//!
//! Given a regex pattern and a tokenizer vocab, build a DFA and at each
//! sampling step compute which tokens can extend the currently accepted
//! prefix without leaving the language. Invalid tokens get `-INFINITY` so
//! the downstream sampler cannot pick them, regardless of temperature or
//! top-k/top-p.
//!
//! This is the "outlines"-style approach: convert the constraint to a
//! finite automaton, walk it byte-by-byte per token to decide validity.
//! No schema → regex transformation here — that belongs a layer up (see
//! `ResponseFormat::JsonSchema` handling).
//!
//! # Design notes
//!
//! * The DFA is built once at request admission — regex compilation is the
//!   expensive step (~1-5 ms for short patterns, scales with ambiguity).
//! * Per-step cost is O(vocab_size · avg_token_bytes). For a 150k vocab
//!   with ~5 byte tokens that's ~750k state transitions per sampling step.
//!   Fine for single requests; we'll add a cached (state, token) → (valid,
//!   next_state) transition table if this becomes a bottleneck.
//! * End-of-string: once the DFA can accept, EOS becomes a valid choice.
//!   If the pattern is "open" (e.g. `.*`) EOS is always allowed.

use std::sync::Arc;

use ferrum_interfaces::sampler::{LogitsProcessor, ProcessorPriority, SamplingContext};
use ferrum_interfaces::tokenizer::Tokenizer;
use ferrum_types::{FerrumError, Result, TokenId};
use parking_lot::Mutex;
use regex_automata::{
    dfa::{dense::DFA, Automaton, StartKind},
    util::{primitives::StateID, start::Config as StartConfig},
    Anchored,
};

/// Hard-mask regex constraint processor.
///
/// Build with `RegexGuidedProcessor::new(pattern, tokenizer, eos_token)`;
/// use by adding as a high-priority logits processor before temperature /
/// top-k / top-p.
pub struct RegexGuidedProcessor {
    dfa: DFA<Vec<u32>>,
    /// Current DFA state — advanced lazily per `process()` call from the
    /// generated tokens accumulated so far.
    state: Mutex<DfaPosition>,
    /// Precomputed per-token byte sequences. Indexed by token id.
    token_bytes: Vec<Vec<u8>>,
    /// Optional EOS id — always allowed once the DFA can accept.
    eos_token: Option<TokenId>,
    /// Number of tokens already consumed into `state`.
    consumed: Mutex<usize>,
}

impl std::fmt::Debug for RegexGuidedProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegexGuidedProcessor")
            .field("vocab_size", &self.token_bytes.len())
            .field("consumed", &*self.consumed.lock())
            .finish()
    }
}

#[derive(Copy, Clone, Debug)]
struct DfaPosition {
    state: StateID,
    /// Set once the DFA enters a dead (non-accepting, non-escapable) state.
    /// At that point no token is valid and we must rely on the sampler's
    /// fallback (we emit EOS to terminate the sequence gracefully).
    dead: bool,
}

impl RegexGuidedProcessor {
    /// Build a guided-decoding processor for `pattern`. `tokenizer` is used
    /// to map each vocab entry to its byte representation.
    pub fn new(
        pattern: &str,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        eos_token: Option<TokenId>,
    ) -> Result<Self> {
        // `Anchored::Yes` only pins the match start; without `\z` the DFA
        // happily keeps accepting after the match completes, and tokens that
        // would produce "valid match + garbage" wouldn't get masked. Wrap
        // the user pattern so the full generated output must match.
        let wrapped = format!(r"(?:{pattern})\z");
        let dfa = DFA::builder()
            .configure(DFA::config().start_kind(StartKind::Anchored))
            .build(&wrapped)
            .map_err(|e| FerrumError::invalid_request(format!("regex compile: {e}")))?;

        let start = dfa
            .start_state(&StartConfig::new().anchored(Anchored::Yes))
            .map_err(|e| FerrumError::invalid_request(format!("regex start state: {e}")))?;

        // Decode every token in the vocab once. If a token has no text
        // representation (common for byte-level BPE control tokens), treat
        // it as empty bytes so the DFA walk trivially accepts it — the EOS
        // check below covers the termination case.
        let vocab_size = tokenizer.vocab_size();
        let mut token_bytes = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let id = TokenId::new(i as u32);
            let bytes = tokenizer
                .token_text(id)
                .map(|s| s.as_bytes().to_vec())
                .unwrap_or_default();
            token_bytes.push(bytes);
        }

        Ok(Self {
            dfa,
            state: Mutex::new(DfaPosition { state: start, dead: false }),
            token_bytes,
            eos_token,
            consumed: Mutex::new(0),
        })
    }

    /// Reset for a new generation.
    pub fn reset(&self) -> Result<()> {
        let start = self
            .dfa
            .start_state(&StartConfig::new().anchored(Anchored::Yes))
            .map_err(|e| FerrumError::internal(format!("regex start state: {e}")))?;
        *self.state.lock() = DfaPosition { state: start, dead: false };
        *self.consumed.lock() = 0;
        Ok(())
    }

    /// Check if the pattern can currently accept (i.e. EOS is valid here).
    pub fn can_accept(&self) -> bool {
        let pos = *self.state.lock();
        !pos.dead && self.dfa.is_match_state(self.dfa.next_eoi_state(pos.state))
    }

    /// Walk the DFA over `bytes` starting from `state`. Returns the new
    /// state, or `None` if a dead state is reached partway through — i.e.
    /// this byte sequence cannot extend the current match.
    fn advance(&self, mut state: StateID, bytes: &[u8]) -> Option<StateID> {
        for &b in bytes {
            state = self.dfa.next_state(state, b);
            if self.dfa.is_dead_state(state) {
                return None;
            }
        }
        Some(state)
    }

    /// Apply the hard mask to `logits` given the current DFA state.
    pub fn mask_logits(&self, logits: &mut [f32]) {
        let pos = *self.state.lock();
        if pos.dead {
            // Give up and force EOS — no other token can recover.
            if let Some(eos) = self.eos_token {
                let eos_idx = eos.get() as usize;
                for (i, l) in logits.iter_mut().enumerate() {
                    *l = if i == eos_idx { 0.0 } else { f32::NEG_INFINITY };
                }
            }
            return;
        }

        let pattern_done = self.dfa.is_match_state(self.dfa.next_eoi_state(pos.state));

        // Mask tokens individually. `&self.token_bytes` is O(vocab) once; a
        // cached per-state transition table would amortise further, but this
        // keeps the hot path allocation-free for now.
        let vocab = logits.len().min(self.token_bytes.len());
        for idx in 0..vocab {
            let is_eos = self.eos_token.map_or(false, |e| e.get() as usize == idx);
            let bytes = &self.token_bytes[idx];

            let allowed = if is_eos {
                pattern_done
            } else if bytes.is_empty() {
                // Unknown / special token outside the regex alphabet — only
                // let it through if pattern can already accept (so it can't
                // block a valid termination).
                pattern_done
            } else {
                self.advance(pos.state, bytes).is_some()
            };

            if !allowed {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }

    /// Public wrapper around `advance_with_tokens` for direct callers
    /// (the engine applies the mask inline rather than via
    /// `LogitsProcessor::process`).
    pub fn advance_with_tokens_public(&self, tokens: &[TokenId]) {
        self.advance_with_tokens(tokens);
    }

    /// Advance the stored state by consuming tokens that were decided after
    /// the last `process()` call. The engine calls this in `process()` with
    /// the full generated-tokens list; we skip the prefix we've already seen.
    fn advance_with_tokens(&self, tokens: &[TokenId]) {
        let mut consumed = self.consumed.lock();
        if *consumed >= tokens.len() {
            return;
        }
        let mut pos = self.state.lock();
        for &tok in &tokens[*consumed..] {
            if pos.dead {
                break;
            }
            let idx = tok.get() as usize;
            if idx >= self.token_bytes.len() {
                continue;
            }
            let bytes = &self.token_bytes[idx];
            // EOS terminates cleanly — leave state where it is.
            if self.eos_token.map_or(false, |e| e == tok) {
                continue;
            }
            if let Some(next) = self.advance(pos.state, bytes) {
                pos.state = next;
            } else {
                pos.dead = true;
            }
        }
        *consumed = tokens.len();
    }
}

impl LogitsProcessor for RegexGuidedProcessor {
    fn process(&self, ctx: &mut SamplingContext) -> Result<()> {
        self.advance_with_tokens(ctx.previous_tokens);
        self.mask_logits(ctx.logits);
        Ok(())
    }

    fn name(&self) -> &str {
        "regex_guided"
    }

    fn priority(&self) -> ProcessorPriority {
        // Apply before temperature / top-k / top-p — those should only see
        // logits for *valid* tokens.
        ProcessorPriority::High
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::tokenizer::{ChatMessage, TokenizerInfo, TokenizerType};
    use ferrum_types::{SpecialTokens, TokenId};

    /// Tiny tokenizer: each ASCII character 0..=255 is a single token, plus
    /// an EOS token at 256. Matches how byte-level BPEs decompose in the
    /// worst case, so the test is a lower bound on the real-world case.
    struct ByteTokenizer {
        special: SpecialTokens,
        byte_strings: Vec<String>,
    }

    impl ByteTokenizer {
        fn new() -> Self {
            let mut byte_strings = Vec::with_capacity(257);
            for b in 0u8..=255 {
                byte_strings.push(String::from_utf8(vec![b]).unwrap_or_default());
            }
            byte_strings.push("</s>".to_string());
            Self {
                special: SpecialTokens {
                    bos_token: None,
                    eos_token: Some(TokenId::new(256)),
                    unk_token: None,
                    pad_token: None,
                    sep_token: None,
                    cls_token: None,
                    mask_token: None,
                },
                byte_strings,
            }
        }
    }

    impl Tokenizer for ByteTokenizer {
        fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
            Ok(text.bytes().map(|b| TokenId::new(b as u32)).collect())
        }
        fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
            let mut out = String::new();
            for t in tokens {
                let idx = t.get() as usize;
                if idx < 256 {
                    out.push(idx as u8 as char);
                }
            }
            Ok(out)
        }
        fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
            self.decode(&[next], false)
        }
        fn vocab_size(&self) -> usize {
            257
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
            self.byte_strings.get(token_id.get() as usize).map(|s| s.as_str())
        }
        fn apply_chat_template(&self, _messages: &[ChatMessage]) -> Result<String> {
            Ok(String::new())
        }
        fn info(&self) -> TokenizerInfo {
            TokenizerInfo {
                tokenizer_type: TokenizerType::Custom,
                vocab_size: 257,
                special_tokens: self.special.clone(),
                supports_incremental: true,
                supports_chat_template: false,
                max_token_length: Some(1),
                model_name: Some("byte-tokenizer-test".into()),
            }
        }
    }

    fn processor(pattern: &str) -> RegexGuidedProcessor {
        let tok: Arc<dyn Tokenizer> = Arc::new(ByteTokenizer::new());
        RegexGuidedProcessor::new(pattern, tok, Some(TokenId::new(256))).unwrap()
    }

    #[test]
    fn digits_only_allows_digits_at_start() {
        let p = processor(r"[0-9]+");
        let mut logits = vec![0.0f32; 257];
        p.mask_logits(&mut logits);
        for b in 0u8..=255 {
            let expected_allowed = b.is_ascii_digit();
            let got = logits[b as usize].is_finite();
            assert_eq!(
                got, expected_allowed,
                "byte {b:?} ({}): expected allowed={expected_allowed}, got={got}",
                b as char
            );
        }
        // EOS is NOT yet allowed (pattern requires >=1 digit).
        assert!(logits[256].is_infinite() && logits[256].is_sign_negative());
    }

    #[test]
    fn digits_only_allows_eos_after_a_digit() {
        let p = processor(r"[0-9]+");
        p.advance_with_tokens(&[TokenId::new(b'3' as u32)]);
        let mut logits = vec![0.0f32; 257];
        p.mask_logits(&mut logits);
        assert!(logits[256].is_finite(), "EOS should be allowed after a digit");
        assert!(logits[b'7' as usize].is_finite(), "another digit still allowed");
        assert!(logits[b'a' as usize].is_infinite(), "alpha still forbidden");
    }

    #[test]
    fn dead_state_forces_eos() {
        let p = processor(r"[0-9]+");
        // Feed an invalid token ("a") — DFA dies.
        p.advance_with_tokens(&[TokenId::new(b'a' as u32)]);
        let mut logits = vec![0.0f32; 257];
        p.mask_logits(&mut logits);
        assert!(logits[256].is_finite(), "EOS forced as fallback");
        for b in 0u8..=255 {
            assert!(
                logits[b as usize].is_infinite(),
                "byte {b} should be masked when DFA dead"
            );
        }
    }

    #[test]
    fn reset_restores_initial_state() {
        let p = processor(r"[0-9]+");
        p.advance_with_tokens(&[TokenId::new(b'3' as u32)]);
        assert!(p.can_accept());
        p.reset().unwrap();
        assert!(!p.can_accept(), "fresh state should not accept empty input");
    }

    #[test]
    fn hex_prefix_pattern() {
        let p = processor(r"0x[0-9a-f]+");
        let mut logits = vec![0.0f32; 257];
        p.mask_logits(&mut logits);
        assert!(logits[b'0' as usize].is_finite(), "'0' starts the pattern");
        assert!(logits[b'1' as usize].is_infinite(), "'1' can't start");
        p.advance_with_tokens(&[TokenId::new(b'0' as u32), TokenId::new(b'x' as u32)]);
        let mut logits = vec![0.0f32; 257];
        p.mask_logits(&mut logits);
        assert!(logits[b'a' as usize].is_finite());
        assert!(logits[b'f' as usize].is_finite());
        assert!(logits[b'g' as usize].is_infinite());
        assert!(logits[b'9' as usize].is_finite());
    }
}
