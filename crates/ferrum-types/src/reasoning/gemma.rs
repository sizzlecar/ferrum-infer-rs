use super::{ParsedReasoningResponse, GEMMA_THOUGHT_END_TAG, GEMMA_THOUGHT_START_TAG};
use crate::{FerrumError, Result};

pub(super) fn has_partial_marker_suffix(text: &str) -> bool {
    [GEMMA_THOUGHT_START_TAG, GEMMA_THOUGHT_END_TAG]
        .iter()
        .any(|marker| (1..marker.len()).any(|length| text.ends_with(&marker[..length])))
}

/// A cumulative-prefix parser keeps already-visible content monotonic across
/// arbitrary chunk boundaries. Only an unfinished marker suffix is withheld;
/// an unfinished thought body remains reasoning rather than visible content.
pub(super) fn parse(text: &str, prompt_opened: bool) -> Result<ParsedReasoningResponse> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut inside_thought = prompt_opened;
    let mut remaining = text;
    while !remaining.is_empty() {
        if let Some(after) = remaining.strip_prefix(GEMMA_THOUGHT_START_TAG) {
            if inside_thought {
                return Err(invalid_channel());
            }
            inside_thought = true;
            remaining = after;
        } else if let Some(after) = remaining.strip_prefix(GEMMA_THOUGHT_END_TAG) {
            if !inside_thought {
                return Err(invalid_channel());
            }
            inside_thought = false;
            remaining = after;
        } else if GEMMA_THOUGHT_START_TAG.starts_with(remaining)
            || GEMMA_THOUGHT_END_TAG.starts_with(remaining)
        {
            // Includes a length-truncated channel header. The channel name is
            // framing, even if the newline has not arrived yet.
            break;
        } else if remaining.starts_with("<|channel>")
            || remaining.starts_with("<|channel|")
            || remaining.starts_with("<channel|")
        {
            // Only the declared channel syntax is reserved. `<channel>` is
            // ordinary XML and `<|` can be an F# operator; neither is a native
            // thought marker once subsequent text disambiguates the prefix.
            return Err(invalid_channel());
        } else {
            let next = remaining.chars().next().expect("nonempty remainder");
            if inside_thought {
                reasoning.push(next);
            } else {
                content.push(next);
            }
            remaining = &remaining[next.len_utf8()..];
        }
    }
    Ok(ParsedReasoningResponse {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    })
}

fn invalid_channel() -> FerrumError {
    // Do not echo model-generated control text into a downstream TTS error.
    FerrumError::invalid_format("model output violated the declared Gemma thought-channel protocol")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        has_unclosed_model_reasoning_block, parse_model_reasoning_response,
        should_defer_model_reasoning_stream_delta, ModelOutputProtocol,
    };

    const PROTOCOL: ModelOutputProtocol = ModelOutputProtocol::GemmaThought;

    #[test]
    fn separates_empty_and_nonempty_native_thought_channels() {
        for (text, content, reasoning) in [
            ("thought\n579", "thought\n579", None),
            ("579", "579", None),
            ("<|channel>thought\n<channel|>579", "579", None),
            (
                "<|channel>thought\nUse the tool result.<channel|>579",
                "579",
                Some("Use the tool result."),
            ),
            (
                "Before.<|channel>thought\nFirst.<channel|>Between.\
                 <|channel>thought\nSecond.<channel|>After.",
                "Before.Between.After.",
                Some("First.Second."),
            ),
        ] {
            let parsed = parse_model_reasoning_response(PROTOCOL, text, false).unwrap();
            assert_eq!(parsed.content, content);
            assert_eq!(parsed.reasoning.as_deref(), reasoning);
        }
    }

    #[test]
    fn preserves_ordinary_markup_and_code_inside_a_json_payload() {
        let json = r#"{"rss":"<channel><title>News</title></channel>","code":"f <| x + 1","text":"<|other>"}"#;
        let output = format!("{GEMMA_THOUGHT_START_TAG}{GEMMA_THOUGHT_END_TAG}{json}");
        let parsed = parse_model_reasoning_response(PROTOCOL, &output, false).unwrap();
        assert_eq!(parsed.content, json);
        assert!(parsed.reasoning.is_none());
        let value: serde_json::Value = serde_json::from_str(&parsed.content).unwrap();
        assert_eq!(value["rss"], "<channel><title>News</title></channel>");
        assert_eq!(value["code"], "f <| x + 1");
        assert_eq!(value["text"], "<|other>");
    }

    #[test]
    fn rendered_generation_turn_controls_prompt_opened_state() {
        for (prompt, generated, opened, reasoning) in [
            (
                "<|turn>model\n<|channel>thought\n<channel|>",
                "579",
                false,
                None,
            ),
            (
                "<|turn>system\n<|think|>\n<turn|>\n<|turn>model\n",
                "<|channel>thought\nCompute.<channel|>579",
                false,
                Some("Compute."),
            ),
            (
                "<|turn>model\n<|tool_response>579<tool_response|><|channel>thought\n",
                "Compute.<channel|>579",
                true,
                Some("Compute."),
            ),
            (
                "<|turn>model\n<|tool_response>579<tool_response|>",
                "<|channel>thought\n<channel|>579",
                false,
                None,
            ),
            (
                "<|turn>user\nExplain <|channel>thought\n<turn|>\n<|turn>model\n",
                "579",
                false,
                None,
            ),
        ] {
            assert_eq!(has_unclosed_model_reasoning_block(PROTOCOL, prompt), opened);
            let parsed = parse_model_reasoning_response(PROTOCOL, generated, opened).unwrap();
            assert_eq!(parsed.content, "579");
            assert_eq!(parsed.reasoning.as_deref(), reasoning);
        }
    }

    #[test]
    fn every_chunk_split_preserves_visible_and_reasoning_prefixes() {
        for (text, opened, expected_content, expected_reasoning) in [
            (
                "Before.<|channel>thought\nReason 🧠.<channel|>After.",
                false,
                "Before.After.",
                "Reason 🧠.",
            ),
            (
                "Reason.<channel|>After.<|channel>thought\nMore.<channel|>Done.",
                true,
                "After.Done.",
                "Reason.More.",
            ),
            ("<|channel>thought\n<channel|>579", false, "579", ""),
            (
                "RSS uses <channel><title>News</title></channel>.",
                false,
                "RSS uses <channel><title>News</title></channel>.",
                "",
            ),
            ("f <| x + 1", false, "f <| x + 1", ""),
            (
                r#"{"rss":"<channel><title>News</title></channel>","code":"f <| x + 1","text":"<|other>"}"#,
                false,
                r#"{"rss":"<channel><title>News</title></channel>","code":"f <| x + 1","text":"<|other>"}"#,
                "",
            ),
            (
                "<|channel>thought\nExplain f <| x + 1 and <channel>.<channel|>Use f <| x + 1.",
                false,
                "Use f <| x + 1.",
                "Explain f <| x + 1 and <channel>.",
            ),
        ] {
            let mut previous_content = String::new();
            let mut previous_reasoning = String::new();
            for (length, _) in text
                .char_indices()
                .chain(std::iter::once((text.len(), '\0')))
            {
                let prefix = &text[..length];
                let parsed = parse_model_reasoning_response(PROTOCOL, prefix, opened).unwrap();
                let reasoning = parsed.reasoning.unwrap_or_default();
                assert!(parsed.content.starts_with(&previous_content));
                assert!(reasoning.starts_with(&previous_reasoning));
                assert!(expected_content.starts_with(&parsed.content));
                assert!(expected_reasoning.starts_with(&reasoning));
                previous_content = parsed.content;
                previous_reasoning = reasoning;
            }
            assert_eq!(previous_content, expected_content);
            assert_eq!(previous_reasoning, expected_reasoning);
        }
    }

    #[test]
    fn incomplete_headers_and_closing_markers_never_become_content() {
        for marker in [GEMMA_THOUGHT_START_TAG, GEMMA_THOUGHT_END_TAG] {
            for length in 1..marker.len() {
                let prefix = &marker[..length];
                assert!(should_defer_model_reasoning_stream_delta(PROTOCOL, prefix));
                let parsed = parse_model_reasoning_response(
                    PROTOCOL,
                    prefix,
                    marker == GEMMA_THOUGHT_END_TAG,
                )
                .unwrap();
                assert!(parsed.content.is_empty());
                assert!(parsed.reasoning.is_none());
            }
        }
        let parsed = parse_model_reasoning_response(
            PROTOCOL,
            "<|channel>thought\nUnfinished reasoning.<chan",
            false,
        )
        .unwrap();
        assert!(parsed.content.is_empty());
        assert_eq!(parsed.reasoning.as_deref(), Some("Unfinished reasoning."));
        assert!(!should_defer_model_reasoning_stream_delta(PROTOCOL, "579"));
        assert!(!should_defer_model_reasoning_stream_delta(
            PROTOCOL, "thought"
        ));
    }

    #[test]
    fn rejects_undeclared_or_malformed_channels_without_echoing_them() {
        for malformed in [
            "<|channel>final\n579<channel|>",
            "<|channel>thoughtful\n579<channel|>",
            "<|channel|>thought\n579<channel|>",
            "<|channel>thought\r\n579<channel|>",
            "<channel|>579",
            "<|channel>thought\n<|channel>thought\nsecret<channel|>",
            "<|channel>thought\nsecret<channel|x>579",
        ] {
            let error = parse_model_reasoning_response(PROTOCOL, malformed, false).unwrap_err();
            assert!(!error.to_string().contains("<|"));
            assert!(!error.to_string().contains("secret"));
        }
    }
}
