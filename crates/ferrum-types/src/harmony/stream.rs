//! Incremental framing over an append-only, caller-owned decoded buffer.
use super::*;

#[derive(Debug, Clone, Copy)]
enum Phase {
    Header {
        start: usize,
        scan: usize,
        first: bool,
    },
    Body {
        channel: HarmonyChannel,
        scan: usize,
    },
    Terminal {
        end: usize,
    },
}

/// Projects only complete-header analysis/final payload prefixes. The caller
/// owns and accounts for the raw buffer; this state stores only byte offsets.
/// Tool arguments stay private until the strict terminal parser validates them.
#[derive(Debug, Clone, Copy)]
pub struct HarmonyStreamProjection {
    phase: Phase,
    reasoning: Option<(usize, usize)>,
    content: Option<(usize, usize)>,
}

impl Default for HarmonyStreamProjection {
    fn default() -> Self {
        Self {
            phase: Phase::Header {
                start: 0,
                scan: 0,
                first: true,
            },
            reasoning: None,
            content: None,
        }
    }
}

impl HarmonyStreamProjection {
    /// Advance after appending decoded text to the same raw buffer. Failed
    /// advances leave the previously published prefix and parser state intact.
    pub fn advance(&mut self, raw: &str) -> Result<()> {
        let mut next = *self;
        next.advance_inner(raw)?;
        *self = next;
        Ok(())
    }

    fn advance_inner(&mut self, raw: &str) -> Result<()> {
        loop {
            match self.phase {
                Phase::Header {
                    start,
                    mut scan,
                    first,
                } => {
                    while scan < raw.len() {
                        let tail = raw
                            .get(scan..)
                            .ok_or_else(|| invalid_harmony("stream buffer changed"))?;
                        if tail.starts_with(MESSAGE) {
                            break;
                        }
                        if MESSAGE.starts_with(tail) {
                            return Ok(());
                        }
                        scan += tail.chars().next().expect("nonempty suffix").len_utf8();
                        self.phase = Phase::Header { start, scan, first };
                    }
                    if scan == raw.len() {
                        return Ok(());
                    }
                    let end = scan + MESSAGE.len();
                    // Reuse the terminal parser's role, recipient and channel
                    // grammar, without pretending an incomplete payload ended.
                    let message = parse_message(&raw[start..end], first, true)?;
                    match message.channel {
                        HarmonyChannel::Analysis => {
                            if !first {
                                return Err(invalid_harmony("only one analysis message is supported before the terminal response"));
                            }
                            validate_plain_message(&message, "analysis")?;
                            self.reasoning = Some((end, end));
                        }
                        HarmonyChannel::Final => {
                            validate_plain_message(&message, "final")?;
                            self.content = Some((end, end));
                        }
                        HarmonyChannel::Commentary => {
                            // The strict terminal parser owns recipient/JSON
                            // validation. No commentary bytes become text.
                        }
                    }
                    self.phase = Phase::Body {
                        channel: message.channel,
                        scan: end,
                    };
                }
                Phase::Body { channel, mut scan } => {
                    while scan < raw.len() {
                        let tail = raw
                            .get(scan..)
                            .ok_or_else(|| invalid_harmony("stream buffer changed"))?;
                        if tail.starts_with("<|") {
                            let terminal = match channel {
                                HarmonyChannel::Analysis => END,
                                HarmonyChannel::Final => RETURN,
                                HarmonyChannel::Commentary => CALL,
                            };
                            if tail.starts_with(terminal) {
                                let end = scan + terminal.len();
                                self.phase = if channel == HarmonyChannel::Analysis {
                                    Phase::Header {
                                        start: end,
                                        scan: end,
                                        first: false,
                                    }
                                } else {
                                    Phase::Terminal { end }
                                };
                                break;
                            }
                            if terminal.starts_with(tail) {
                                return Ok(());
                            }
                            return Err(invalid_harmony("message payload contains an unknown or misplaced raw control marker"));
                        }
                        if tail.starts_with("|>") {
                            return Err(invalid_harmony(
                                "message payload contains a raw or incomplete control marker",
                            ));
                        }
                        // A marker may straddle decoded chunks. Its first byte
                        // must not be exposed before its meaning is known.
                        if tail == "<" || tail == "|" {
                            return Ok(());
                        }
                        scan += tail.chars().next().expect("nonempty suffix").len_utf8();
                        match channel {
                            HarmonyChannel::Analysis => self.reasoning.as_mut().unwrap().1 = scan,
                            HarmonyChannel::Final => self.content.as_mut().unwrap().1 = scan,
                            HarmonyChannel::Commentary => {}
                        }
                        self.phase = Phase::Body { channel, scan };
                    }
                    if matches!(self.phase, Phase::Body { .. }) {
                        return Ok(());
                    }
                }
                Phase::Terminal { end } => {
                    if raw.len() != end {
                        return Err(invalid_harmony("terminal control token was followed by duplicate terminal data or trailing garbage"));
                    }
                    return Ok(());
                }
            }
        }
    }

    pub fn content<'a>(&self, raw: &'a str) -> &'a str {
        self.content
            .map(|(start, end)| &raw[start..end])
            .unwrap_or("")
    }

    pub fn reasoning<'a>(&self, raw: &'a str) -> Option<&'a str> {
        self.reasoning.map(|(start, end)| &raw[start..end])
    }

    /// Streaming never weakens terminal validation or authorizes an incomplete
    /// tool call. Stop/Length retain the existing text-truncation semantics.
    pub fn finish(&self, raw: &str, reason: Option<FinishReason>) -> Result<ParsedHarmonyResponse> {
        parse_harmony_response_for_finish_reason(raw, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmony_stream_prefixes_are_monotonic_at_every_utf8_split() {
        for raw in [
            "<|channel|>final<|message|>答案 < 4 | 5<|return|>",
            "<|channel|>analysis<|message|>推理🙂<|end|><|start|>assistant<|channel|>final<|message|>答复<|return|>",
            "<|channel|>analysis<|message|>Reason.<|end|><|start|>assistant to=functions.weather<|channel|>commentary<|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>",
        ] {
            let expected = parse_harmony_response(raw).unwrap();
            let mut state = HarmonyStreamProjection::default();
            let mut content = String::new();
            let mut reasoning = String::new();
            for end in raw.char_indices().map(|(offset, ch)| offset + ch.len_utf8()) {
                let prefix = &raw[..end];
                state.advance(prefix).unwrap();
                let visible = state.content(prefix);
                let analysis = state.reasoning(prefix).unwrap_or("");
                assert!(visible.starts_with(&content));
                assert!(analysis.starts_with(&reasoning));
                assert!(expected.content.starts_with(visible));
                assert!(expected.reasoning_content.as_deref().unwrap_or("").starts_with(analysis));
                content = visible.to_owned();
                reasoning = analysis.to_owned();
            }
            assert_eq!(state.finish(raw, Some(FinishReason::EOS)).unwrap(), expected);
            assert_eq!(content, expected.content);
            assert_eq!(reasoning, expected.reasoning_content.unwrap_or_default());
        }
    }

    #[test]
    fn harmony_stream_holds_partial_controls_and_rejects_invalid_completion() {
        let mut state = HarmonyStreamProjection::default();
        let first = "<|channel|>final<|message|>Answer<";
        state.advance(first).unwrap();
        assert_eq!(state.content(first), "Answer");
        assert!(state.advance(&format!("{first}|call|>")).is_err());
        assert_eq!(state.content(first), "Answer");
        assert!(state.finish(first, Some(FinishReason::EOS)).is_err());
        assert_eq!(
            state
                .finish(first, Some(FinishReason::Length))
                .unwrap()
                .content,
            "Answer<"
        );
        let full = format!("{first}|return|>");
        state.advance(&full).unwrap();
        assert!(state.advance(&format!("{full}garbage")).is_err());

        let mut tool = HarmonyStreamProjection::default();
        let partial = "<|channel|>commentary to=functions.weather<|message|>{\"city\":\"Paris\"}";
        tool.advance(partial).unwrap();
        assert_eq!(tool.content(partial), "");
        assert!(tool.finish(partial, Some(FinishReason::Length)).is_err());
        assert!(tool
            .finish(&format!("{partial}<|call|>"), Some(FinishReason::EOS))
            .unwrap()
            .tool_call
            .is_some());
    }
}
