//! Shared parsing for model outputs that carry `<think>` reasoning blocks.

pub const THINK_START_TAG: &str = "<think>";
pub const THINK_END_TAG: &str = "</think>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedReasoningResponse {
    pub content: String,
    pub reasoning: Option<String>,
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
    if text.contains(THINK_START_TAG) {
        return parse_reasoning_response(text);
    }
    let Some(end) = text.find(THINK_END_TAG) else {
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
