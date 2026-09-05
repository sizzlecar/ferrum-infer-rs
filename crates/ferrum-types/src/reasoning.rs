//! Shared parsing for declared model reasoning protocols.

use crate::{FerrumError, ModelOutputProtocol, Result};

mod gemma;

pub const THINK_START_TAG: &str = "<think>";
pub const THINK_END_TAG: &str = "</think>";
pub const GEMMA_THOUGHT_START_TAG: &str = "<|channel>thought\n";
pub const GEMMA_THOUGHT_END_TAG: &str = "<channel|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedReasoningResponse {
    pub content: String,
    pub reasoning: Option<String>,
}

/// The full generated opening header and closing marker of a reasoning block.
/// Harmony owns a separate message protocol rather than a delimited block.
pub const fn model_reasoning_markers(
    protocol: ModelOutputProtocol,
) -> Option<(&'static str, &'static str)> {
    match protocol {
        ModelOutputProtocol::Text => Some((THINK_START_TAG, THINK_END_TAG)),
        ModelOutputProtocol::GemmaThought => Some((GEMMA_THOUGHT_START_TAG, GEMMA_THOUGHT_END_TAG)),
        ModelOutputProtocol::HarmonyGptOss => None,
    }
}

pub fn has_unclosed_model_reasoning_block(protocol: ModelOutputProtocol, prompt: &str) -> bool {
    match protocol {
        ModelOutputProtocol::Text => has_unclosed_thinking_block(prompt),
        ModelOutputProtocol::HarmonyGptOss => false,
        ModelOutputProtocol::GemmaThought => {
            // The generated turn, including a tool-response continuation, owns
            // the prefill state. A marker mentioned in an earlier user turn
            // must not make an ordinary model turn start inside reasoning.
            let turn = prompt
                .rsplit_once("<|turn>")
                .map_or(prompt, |(_, turn)| turn);
            match (
                turn.rfind(GEMMA_THOUGHT_START_TAG),
                turn.rfind(GEMMA_THOUGHT_END_TAG),
            ) {
                (Some(start), Some(end)) => start > end,
                (Some(_), None) => true,
                _ => false,
            }
        }
    }
}

/// Parse a cumulative generated prefix without exposing partial protocol
/// markers. Gemma accepts only the declared thought channel; ordinary words
/// such as "thought" remain ordinary content outside that channel.
pub fn parse_model_reasoning_response(
    protocol: ModelOutputProtocol,
    text: &str,
    prompt_opened_thinking: bool,
) -> Result<ParsedReasoningResponse> {
    match protocol {
        ModelOutputProtocol::Text => Ok(parse_reasoning_response_for_prompt(
            text,
            prompt_opened_thinking,
        )),
        ModelOutputProtocol::GemmaThought => gemma::parse(text, prompt_opened_thinking),
        ModelOutputProtocol::HarmonyGptOss => Err(FerrumError::invalid_request(
            "Harmony output requires its message-protocol parser",
        )),
    }
}

/// Hold only prefixes that can still become framing, not complete responses.
/// Callers retain the cumulative input and parse it again when more arrives.
pub fn should_defer_model_reasoning_stream_delta(
    protocol: ModelOutputProtocol,
    text: &str,
) -> bool {
    match protocol {
        ModelOutputProtocol::Text => {
            let candidate = text.trim_start_matches(['\r', '\n']);
            candidate.is_empty()
                || THINK_START_TAG.starts_with(candidate)
                || THINK_END_TAG.starts_with(candidate)
        }
        ModelOutputProtocol::GemmaThought => {
            text.is_empty() || gemma::has_partial_marker_suffix(text)
        }
        ModelOutputProtocol::HarmonyGptOss => false,
    }
}

pub fn has_unclosed_thinking_block(prompt: &str) -> bool {
    match (prompt.rfind(THINK_START_TAG), prompt.rfind(THINK_END_TAG)) {
        (Some(start), Some(end)) => start > end,
        (Some(_), None) => true,
        _ => false,
    }
}

/// Parse generated text when the rendered prompt already opened `<think>`.
/// Some reasoning templates emit only the closing tag in generated text.
pub fn parse_reasoning_response_started_in_think(text: &str) -> ParsedReasoningResponse {
    let end = text.find(THINK_END_TAG);
    // A generated opener before the closer can repeat the template's opener.
    // Once the prompt-opened block closes, further tags belong to the content.
    if text
        .find(THINK_START_TAG)
        .is_some_and(|start| end.is_none_or(|end| start < end))
    {
        return parse_reasoning_response(text);
    }
    let Some(end) = end else {
        return ParsedReasoningResponse {
            content: String::new(),
            reasoning: (!text.is_empty()).then(|| text.to_string()),
        };
    };
    let reasoning = text[..end].to_string();
    let content = text[end + THINK_END_TAG.len()..]
        .trim_start_matches(['\r', '\n'])
        .to_string();
    ParsedReasoningResponse {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    }
}

pub fn parse_reasoning_response(text: &str) -> ParsedReasoningResponse {
    let Some(start) = text.find(THINK_START_TAG) else {
        if let Some(end) = text.find(THINK_END_TAG) {
            let reasoning = text[..end].to_string();
            let content = text[end + THINK_END_TAG.len()..]
                .trim_start_matches(['\r', '\n'])
                .to_string();
            return ParsedReasoningResponse {
                content,
                reasoning: (!reasoning.is_empty()).then_some(reasoning),
            };
        }
        return ParsedReasoningResponse {
            content: text.to_string(),
            reasoning: None,
        };
    };

    let before = &text[..start];
    let after_start = &text[start + THINK_START_TAG.len()..];
    let Some(end) = after_start.find(THINK_END_TAG) else {
        return ParsedReasoningResponse {
            content: before.to_string(),
            reasoning: Some(after_start.to_string()),
        };
    };

    let reasoning = after_start[..end].to_string();
    let after_end = &after_start[end + THINK_END_TAG.len()..];
    let mut content = String::with_capacity(before.len() + after_end.len());
    content.push_str(before);
    content.push_str(after_end.trim_start_matches(['\r', '\n']));
    ParsedReasoningResponse {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    }
}

pub fn parse_reasoning_response_for_prompt(
    text: &str,
    prompt_opened_thinking: bool,
) -> ParsedReasoningResponse {
    if prompt_opened_thinking {
        parse_reasoning_response_started_in_think(text)
    } else {
        parse_reasoning_response(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_reasoning_block() {
        let parsed = parse_reasoning_response("<think>reason</think>\nanswer");
        assert_eq!(parsed.reasoning.as_deref(), Some("reason"));
        assert_eq!(parsed.content, "answer");
    }

    #[test]
    fn parses_prompt_opened_reasoning_block() {
        let parsed = parse_reasoning_response_started_in_think("reason</think>\nanswer");
        assert_eq!(parsed.reasoning.as_deref(), Some("reason"));
        assert_eq!(parsed.content, "answer");
    }

    #[test]
    fn prompt_opened_thinking_preserves_literal_tags_in_final_content() {
        for content in [
            r#"{"text":"<think>literal</think>"}"#,
            r#"{"text":"an unpaired <think> marker"}"#,
            "The tags `<think>` and `</think>` are ordinary text here.",
        ] {
            let raw = format!("reason</think>\r\n{content}");
            let parsed = parse_model_reasoning_response(ModelOutputProtocol::Text, &raw, true)
                .expect("declared text reasoning protocol");
            assert_eq!(parsed.reasoning.as_deref(), Some("reason"), "{content}");
            assert_eq!(parsed.content, content);
        }
    }

    #[test]
    fn empty_prompt_opened_thinking_preserves_final_literal_tags() {
        let parsed = parse_reasoning_response_started_in_think(
            "</think>\n{\"text\":\"<think>literal</think>\"}",
        );
        assert_eq!(parsed.reasoning, None);
        assert_eq!(parsed.content, r#"{"text":"<think>literal</think>"}"#);
    }

    #[test]
    fn prompt_opened_thinking_keeps_closed_boundary_across_stream_prefixes() {
        let content = r#"{"text":"答案 🦀 <think>literal</think>"}"#;
        for end in content
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(content.len()))
        {
            let prefix = &content[..end];
            let raw = format!("推理</think>\n{prefix}");
            let parsed = parse_model_reasoning_response(ModelOutputProtocol::Text, &raw, true)
                .expect("declared text reasoning protocol");
            assert_eq!(parsed.reasoning.as_deref(), Some("推理"), "{prefix}");
            assert_eq!(parsed.content, prefix);
        }
    }

    #[test]
    fn prompt_opened_thinking_accepts_repeated_opening_and_unfinished_reasoning() {
        let parsed = parse_reasoning_response_started_in_think(
            "<think>reason</think>\nThe literal <think> tag remains content.",
        );
        assert_eq!(parsed.reasoning.as_deref(), Some("reason"));
        assert_eq!(parsed.content, "The literal <think> tag remains content.");

        for raw in ["reason", "<think>reason"] {
            let parsed = parse_reasoning_response_started_in_think(raw);
            assert_eq!(parsed.reasoning.as_deref(), Some("reason"));
            assert!(parsed.content.is_empty());
        }
    }

    #[test]
    fn preserves_plain_content() {
        let parsed = parse_reasoning_response("answer");
        assert_eq!(parsed.reasoning, None);
        assert_eq!(parsed.content, "answer");
    }

    #[test]
    fn detects_prompt_opened_reasoning() {
        assert!(has_unclosed_thinking_block("assistant:<think>\n"));
        assert!(!has_unclosed_thinking_block(
            "assistant:<think>reason</think>\n"
        ));
    }
}
