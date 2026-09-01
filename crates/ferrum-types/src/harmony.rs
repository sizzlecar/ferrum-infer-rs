//! Strict terminal-output parsing for the GPT-OSS Harmony wire protocol.
//!
//! This intentionally supports only Ferrum's first product slice: a direct
//! final answer, one analysis message followed by a final answer, or one
//! analysis message followed by a function call on the commentary channel.

use serde::{Deserialize, Serialize};

use crate::{FerrumError, Result};

const START: &str = "<|start|>";
const END: &str = "<|end|>";
const MESSAGE: &str = "<|message|>";
const CHANNEL: &str = "<|channel|>";
const CONSTRAIN: &str = "<|constrain|>";
const CALL: &str = "<|call|>";
const RETURN: &str = "<|return|>";
const FUNCTION_RECIPIENT_PREFIX: &str = "functions.";

/// A validated function call emitted through the Harmony commentary channel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HarmonyToolCall {
    pub name: String,
    /// The original JSON object text, with only surrounding whitespace removed.
    pub arguments_json: String,
}

/// Product-facing terminal result of a complete Harmony model output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedHarmonyResponse {
    pub reasoning_content: Option<String>,
    pub content: String,
    pub tool_call: Option<HarmonyToolCall>,
}

/// Parse one complete decoded GPT-OSS Harmony response.
///
/// Accepted shapes are deliberately narrow:
///
/// - `final<|return|>`
/// - `analysis<|end|> -> final<|return|>`
/// - `analysis<|end|> -> commentary to=functions.NAME ... <|call|>`
///
/// The first message may omit `<|start|>assistant` because that prefix is
/// normally already present at the end of the rendered generation prompt.
pub fn parse_harmony_response(output: &str) -> Result<ParsedHarmonyResponse> {
    parse_harmony_response_internal(output, false)
}

/// Parse a Harmony response that the engine stopped at its token limit.
///
/// A length stop may legitimately omit the terminal token from an analysis or
/// final text message. Tool calls remain fail-closed and still require a
/// complete `<|call|>` envelope.
pub fn parse_length_truncated_harmony_response(output: &str) -> Result<ParsedHarmonyResponse> {
    parse_harmony_response_internal(output, true)
}

fn parse_harmony_response_internal(
    output: &str,
    allow_missing_text_terminal: bool,
) -> Result<ParsedHarmonyResponse> {
    let first = parse_message(output, true, allow_missing_text_terminal)?;
    match first.channel {
        HarmonyChannel::Final => {
            validate_plain_message(&first, "final")?;
            require_text_terminal(&first, HarmonyTerminal::Return, allow_missing_text_terminal)?;
            require_no_trailing_output(&first)?;
            Ok(ParsedHarmonyResponse {
                reasoning_content: None,
                content: first.payload.to_string(),
                tool_call: None,
            })
        }
        HarmonyChannel::Analysis => {
            validate_plain_message(&first, "analysis")?;
            if first.terminal.is_none() && allow_missing_text_terminal {
                return Ok(ParsedHarmonyResponse {
                    reasoning_content: Some(first.payload.to_string()),
                    content: String::new(),
                    tool_call: None,
                });
            }
            require_terminal(&first, HarmonyTerminal::End)?;
            if first.remaining.is_empty() {
                if allow_missing_text_terminal {
                    return Ok(ParsedHarmonyResponse {
                        reasoning_content: Some(first.payload.to_string()),
                        content: String::new(),
                        tool_call: None,
                    });
                }
                return Err(invalid_harmony(
                    "analysis message was not followed by a terminal final answer or tool call",
                ));
            }

            if allow_missing_text_terminal && is_length_truncated_followup_envelope(first.remaining)
            {
                return Ok(ParsedHarmonyResponse {
                    reasoning_content: Some(first.payload.to_string()),
                    content: String::new(),
                    tool_call: None,
                });
            }

            let second = parse_message(first.remaining, false, allow_missing_text_terminal)?;
            match second.channel {
                HarmonyChannel::Final => {
                    validate_plain_message(&second, "final")?;
                    require_text_terminal(
                        &second,
                        HarmonyTerminal::Return,
                        allow_missing_text_terminal,
                    )?;
                    require_no_trailing_output(&second)?;
                    Ok(ParsedHarmonyResponse {
                        reasoning_content: Some(first.payload.to_string()),
                        content: second.payload.to_string(),
                        tool_call: None,
                    })
                }
                HarmonyChannel::Commentary => {
                    let tool_call = parse_tool_call(&second)?;
                    require_terminal(&second, HarmonyTerminal::Call)?;
                    require_no_trailing_output(&second)?;
                    Ok(ParsedHarmonyResponse {
                        reasoning_content: Some(first.payload.to_string()),
                        content: String::new(),
                        tool_call: Some(tool_call),
                    })
                }
                HarmonyChannel::Analysis => Err(invalid_harmony(
                    "only one analysis message is supported before the terminal response",
                )),
            }
        }
        HarmonyChannel::Commentary => Err(invalid_harmony(
            "a commentary tool call must follow one complete analysis message",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyChannel {
    Analysis,
    Commentary,
    Final,
}

impl HarmonyChannel {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "analysis" => Ok(Self::Analysis),
            "commentary" => Ok(Self::Commentary),
            "final" => Ok(Self::Final),
            _ => Err(invalid_harmony(format!(
                "unsupported Harmony channel {value:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyTerminal {
    End,
    Call,
    Return,
}

impl HarmonyTerminal {
    const fn text(self) -> &'static str {
        match self {
            Self::End => END,
            Self::Call => CALL,
            Self::Return => RETURN,
        }
    }
}

#[derive(Debug)]
struct ParsedMessage<'a> {
    channel: HarmonyChannel,
    recipient: Option<&'a str>,
    content_type: Option<&'a str>,
    payload: &'a str,
    terminal: Option<HarmonyTerminal>,
    remaining: &'a str,
}

fn parse_message(
    output: &str,
    first: bool,
    allow_missing_terminal: bool,
) -> Result<ParsedMessage<'_>> {
    if output.is_empty() {
        return Err(invalid_harmony("Harmony output is empty"));
    }

    let (role_recipient, after_channel_marker) =
        if let Some(after_start) = output.strip_prefix(START) {
            let channel_offset = after_start.find(CHANNEL).ok_or_else(|| {
                invalid_harmony("message start was not followed by a channel marker")
            })?;
            let role_header = &after_start[..channel_offset];
            reject_raw_marker(role_header, "assistant role header")?;
            let recipient = parse_role_header(role_header)?;
            (recipient, &after_start[channel_offset + CHANNEL.len()..])
        } else if first {
            let after_channel = output
                .strip_prefix(CHANNEL)
                .ok_or_else(|| invalid_harmony("first message must begin with a channel marker"))?;
            (None, after_channel)
        } else {
            return Err(invalid_harmony(
                "a message after analysis must begin with <|start|>assistant",
            ));
        };

    let message_offset = after_channel_marker.find(MESSAGE);
    let channel_header = message_offset
        .map(|offset| &after_channel_marker[..offset])
        .unwrap_or(after_channel_marker);
    let (channel, channel_recipient, content_type) = parse_channel_header(channel_header)?;
    let recipient = match (role_recipient, channel_recipient) {
        (Some(_), Some(_)) => {
            return Err(invalid_harmony(
                "recipient was repeated in both role and channel headers",
            ));
        }
        (Some(recipient), None) | (None, Some(recipient)) => Some(recipient),
        (None, None) => None,
    };

    let Some(message_offset) = message_offset else {
        if allow_missing_terminal
            && matches!(channel, HarmonyChannel::Analysis | HarmonyChannel::Final)
            && recipient.is_none()
            && content_type.is_none()
        {
            return Ok(ParsedMessage {
                channel,
                recipient,
                content_type,
                payload: "",
                terminal: None,
                remaining: "",
            });
        }
        return Err(invalid_harmony(
            "channel header was not followed by a message marker",
        ));
    };

    let after_message = &after_channel_marker[message_offset + MESSAGE.len()..];
    let Some(terminal_offset) = after_message.find("<|") else {
        reject_raw_marker(after_message, "message payload")?;
        if !allow_missing_terminal {
            return Err(invalid_harmony(
                "message is missing a terminal control token",
            ));
        }
        return Ok(ParsedMessage {
            channel,
            recipient,
            content_type,
            payload: after_message,
            terminal: None,
            remaining: "",
        });
    };
    let payload = &after_message[..terminal_offset];
    reject_raw_marker(payload, "message payload")?;
    let terminal_and_remaining = &after_message[terminal_offset..];
    let (terminal, remaining) = if let Some(remaining) = terminal_and_remaining.strip_prefix(END) {
        (Some(HarmonyTerminal::End), remaining)
    } else if let Some(remaining) = terminal_and_remaining.strip_prefix(CALL) {
        (Some(HarmonyTerminal::Call), remaining)
    } else if let Some(remaining) = terminal_and_remaining.strip_prefix(RETURN) {
        (Some(HarmonyTerminal::Return), remaining)
    } else {
        return Err(invalid_harmony(
            "message payload contains an unknown or misplaced raw control marker",
        ));
    };

    Ok(ParsedMessage {
        channel,
        recipient,
        content_type,
        payload,
        terminal,
        remaining,
    })
}

fn is_length_truncated_followup_envelope(output: &str) -> bool {
    output == concat!("<|start|>", "assistant")
        || output == concat!("<|start|>", "assistant", "<|channel|>")
}

fn parse_role_header(header: &str) -> Result<Option<&str>> {
    let mut parts = header.split_ascii_whitespace();
    if parts.next() != Some("assistant") {
        return Err(invalid_harmony(
            "generated Harmony messages must have the assistant role",
        ));
    }
    let recipient = parts.next().map(parse_recipient_token).transpose()?;
    if parts.next().is_some() {
        return Err(invalid_harmony(
            "assistant role header contains unsupported metadata",
        ));
    }
    Ok(recipient)
}

fn parse_channel_header(header: &str) -> Result<(HarmonyChannel, Option<&str>, Option<&str>)> {
    let mut constrain_parts = header.split(CONSTRAIN);
    let channel_part = constrain_parts.next().unwrap_or_default();
    let content_type = constrain_parts.next().map(str::trim);
    if constrain_parts.next().is_some() {
        return Err(invalid_harmony(
            "channel header contains repeated constrain markers",
        ));
    }
    reject_raw_marker(channel_part, "channel header")?;
    if let Some(content_type) = content_type {
        reject_raw_marker(content_type, "content type")?;
        if content_type.is_empty() {
            return Err(invalid_harmony("constrain marker has no content type"));
        }
    }

    let mut parts = channel_part.split_ascii_whitespace();
    let channel = parts
        .next()
        .ok_or_else(|| invalid_harmony("channel marker has no channel value"))?;
    let channel = HarmonyChannel::parse(channel)?;
    let recipient = parts.next().map(parse_recipient_token).transpose()?;
    if parts.next().is_some() {
        return Err(invalid_harmony(
            "channel header contains unsupported metadata",
        ));
    }
    Ok((channel, recipient, content_type))
}

fn parse_recipient_token(token: &str) -> Result<&str> {
    let recipient = token
        .strip_prefix("to=")
        .ok_or_else(|| invalid_harmony("recipient metadata must use the to= form"))?;
    if recipient.is_empty() {
        return Err(invalid_harmony("recipient must not be empty"));
    }
    Ok(recipient)
}

fn parse_tool_call(message: &ParsedMessage<'_>) -> Result<HarmonyToolCall> {
    let recipient = message
        .recipient
        .ok_or_else(|| invalid_harmony("commentary tool call has no recipient"))?;
    let name = recipient
        .strip_prefix(FUNCTION_RECIPIENT_PREFIX)
        .ok_or_else(|| invalid_harmony(format!("unknown tool recipient {recipient:?}")))?;
    if name.is_empty() {
        return Err(invalid_harmony("tool name must not be empty"));
    }
    if let Some(content_type) = message.content_type {
        if content_type != "json" {
            return Err(invalid_harmony(format!(
                "tool arguments must use the json content type, got {content_type:?}"
            )));
        }
    }

    let arguments_json = message.payload.trim();
    let arguments: serde_json::Value = serde_json::from_str(arguments_json)
        .map_err(|error| invalid_harmony(format!("tool arguments are not valid JSON: {error}")))?;
    if !arguments.is_object() {
        return Err(invalid_harmony("tool arguments must be a JSON object"));
    }

    Ok(HarmonyToolCall {
        name: name.to_string(),
        arguments_json: arguments_json.to_string(),
    })
}

fn validate_plain_message(message: &ParsedMessage<'_>, channel: &str) -> Result<()> {
    if message.recipient.is_some() {
        return Err(invalid_harmony(format!(
            "{channel} channel must not contain a recipient"
        )));
    }
    if message.content_type.is_some() {
        return Err(invalid_harmony(format!(
            "{channel} channel must not contain a constrain marker"
        )));
    }
    Ok(())
}

fn require_terminal(message: &ParsedMessage<'_>, expected: HarmonyTerminal) -> Result<()> {
    if message.terminal != Some(expected) {
        return Err(invalid_harmony(format!(
            "{} channel must end with {}, got {}",
            match message.channel {
                HarmonyChannel::Analysis => "analysis",
                HarmonyChannel::Commentary => "commentary",
                HarmonyChannel::Final => "final",
            },
            expected.text(),
            message
                .terminal
                .map(HarmonyTerminal::text)
                .unwrap_or("no terminal token"),
        )));
    }
    Ok(())
}

fn require_text_terminal(
    message: &ParsedMessage<'_>,
    expected: HarmonyTerminal,
    allow_missing: bool,
) -> Result<()> {
    if allow_missing && message.terminal.is_none() {
        return Ok(());
    }
    require_terminal(message, expected)
}

fn require_no_trailing_output(message: &ParsedMessage<'_>) -> Result<()> {
    if !message.remaining.is_empty() {
        return Err(invalid_harmony(
            "terminal control token was followed by duplicate terminal data or trailing garbage",
        ));
    }
    Ok(())
}

fn reject_raw_marker(value: &str, location: &str) -> Result<()> {
    if value.contains("<|") || value.contains("|>") {
        return Err(invalid_harmony(format!(
            "{location} contains a raw or incomplete control marker"
        )));
    }
    Ok(())
}

fn invalid_harmony(message: impl Into<String>) -> FerrumError {
    FerrumError::invalid_format(format!(
        "invalid GPT-OSS Harmony output: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_direct_final_response() {
        let parsed =
            parse_harmony_response("<|channel|>final<|message|>The capital is Paris.<|return|>")
                .unwrap();

        assert_eq!(
            parsed,
            ParsedHarmonyResponse {
                reasoning_content: None,
                content: "The capital is Paris.".to_string(),
                tool_call: None,
            }
        );
    }

    #[test]
    fn parses_analysis_then_final_response() {
        let parsed = parse_harmony_response(
            "<|channel|>analysis<|message|>Need the capital.<|end|>\
             <|start|>assistant<|channel|>final<|message|>Paris.<|return|>",
        )
        .unwrap();

        assert_eq!(
            parsed.reasoning_content.as_deref(),
            Some("Need the capital.")
        );
        assert_eq!(parsed.content, "Paris.");
        assert_eq!(parsed.tool_call, None);
    }

    #[test]
    fn parses_length_truncated_text_messages_without_weakening_strict_parser() {
        for (output, reasoning, content) in [
            ("<|channel|>final", None, ""),
            ("<|channel|>analysis", Some(""), ""),
            (
                "<|channel|>final<|message|>Partial answer",
                None,
                "Partial answer",
            ),
            (
                "<|channel|>analysis<|message|>Partial reasoning",
                Some("Partial reasoning"),
                "",
            ),
            (
                "<|channel|>analysis<|message|>Reason.<|end|>",
                Some("Reason."),
                "",
            ),
            (
                "<|channel|>analysis<|message|>Reason.<|end|>\
                 <|start|>assistant<|channel|>final",
                Some("Reason."),
                "",
            ),
            (
                "<|channel|>analysis<|message|>Reason.<|end|>\
                 <|start|>assistant<|channel|>final<|message|>Partial answer",
                Some("Reason."),
                "Partial answer",
            ),
        ] {
            assert!(parse_harmony_response(output).is_err());
            let parsed = parse_length_truncated_harmony_response(output).unwrap();
            assert_eq!(parsed.reasoning_content.as_deref(), reasoning);
            assert_eq!(parsed.content, content);
            assert!(parsed.tool_call.is_none());
        }
    }

    #[test]
    fn parses_length_truncation_between_followup_envelope_markers() {
        for output in [
            "<|channel|>analysis<|message|>Reason.<|end|>\
             <|start|>assistant",
            "<|channel|>analysis<|message|>Reason.<|end|>\
             <|start|>assistant<|channel|>",
        ] {
            assert!(parse_harmony_response(output).is_err());
            let parsed = parse_length_truncated_harmony_response(output).unwrap();
            assert_eq!(parsed.reasoning_content.as_deref(), Some("Reason."));
            assert!(parsed.content.is_empty());
            assert!(parsed.tool_call.is_none());
        }
    }

    #[test]
    fn length_truncation_keeps_tool_calls_and_control_markers_fail_closed() {
        for output in [
            "<|channel|>analysis<|message|>Reason.<|end|>\
             <|start|>assistant<|channel|>commentary to=functions.weather\
             <|constrain|>json<|message|>{\"city\":\"Paris\"}",
            "<|channel|>final<|message|>leak <|bogus|>",
            "<|channel|>final<|message|>incomplete <|",
            "<|channel|>analysis<|message|>Reason.<|end|>\
             <|start|>assistant<|channel|>commentary to=functions.weather\
             <|constrain|>json",
            "<|channel|>analysis<|message|>Reason.<|end|>\
             <|start|>assistant<|channel|>final<|mess",
            "<|channel|>analysis<|message|>Reason.<|end|>\
             <|start|>assistant<|channel|>fina",
            "<|channel|>analysis<|message|>Reason.<|end|>\
             <|start|>assistant to=functions.weather<|channel|>final",
        ] {
            assert!(
                parse_length_truncated_harmony_response(output).is_err(),
                "accepted {output:?}"
            );
        }
    }

    #[test]
    fn parses_analysis_then_function_tool_call_in_both_header_orders() {
        for output in [
            "<|channel|>analysis<|message|>Need weather.<|end|>\
             <|start|>assistant<|channel|>commentary to=functions.weather<|constrain|>json\
             <|message|>{\"city\":\"Paris\"}<|call|>",
            "<|channel|>analysis<|message|>Need weather.<|end|>\
             <|start|>assistant to=functions.weather<|channel|>commentary<|constrain|>json\
             <|message|>{\"city\":\"Paris\"}<|call|>",
        ] {
            let parsed = parse_harmony_response(output).unwrap();
            assert_eq!(parsed.reasoning_content.as_deref(), Some("Need weather."));
            assert!(parsed.content.is_empty());
            assert_eq!(
                parsed.tool_call,
                Some(HarmonyToolCall {
                    name: "weather".to_string(),
                    arguments_json: "{\"city\":\"Paris\"}".to_string(),
                })
            );
        }
    }

    #[test]
    fn rejects_invalid_or_non_object_tool_json() {
        for arguments in ["{", "[]", "null", "\"Paris\""] {
            let output = format!(
                "<|channel|>analysis<|message|>Need weather.<|end|>\
                 <|start|>assistant<|channel|>commentary to=functions.weather<|constrain|>json\
                 <|message|>{arguments}<|call|>"
            );
            assert!(
                parse_harmony_response(&output).is_err(),
                "accepted {arguments:?}"
            );
        }
    }

    #[test]
    fn rejects_raw_marker_leakage_and_incomplete_envelopes() {
        for output in [
            "<|channel|>final<|message|>leak <|bogus|> marker<|return|>",
            "<|channel|>final answer<|return|>",
            "<|channel|>final<|message|>incomplete <| marker",
        ] {
            assert!(
                parse_harmony_response(output).is_err(),
                "accepted {output:?}"
            );
        }
    }

    #[test]
    fn rejects_wrong_missing_or_repeated_terminal_tokens() {
        for output in [
            "<|channel|>final<|message|>Paris<|end|>",
            "<|channel|>final<|message|>Paris<|call|>",
            "<|channel|>final<|message|>Paris",
            "<|channel|>final<|message|>Paris<|return|><|return|>",
            "<|channel|>final<|message|>Paris<|return|>garbage",
            "<|channel|>analysis<|message|>Need weather.<|end|>\
             <|start|>assistant<|channel|>commentary to=functions.weather<|message|>{}<|return|>",
        ] {
            assert!(
                parse_harmony_response(output).is_err(),
                "accepted {output:?}"
            );
        }
    }

    #[test]
    fn rejects_unknown_or_empty_tool_recipients() {
        for recipient in ["browser.search", "python", "functions."] {
            let output = format!(
                "<|channel|>analysis<|message|>Need a tool.<|end|>\
                 <|start|>assistant<|channel|>commentary to={recipient}<|constrain|>json\
                 <|message|>{{}}<|call|>"
            );
            assert!(
                parse_harmony_response(&output).is_err(),
                "accepted {recipient:?}"
            );
        }
    }

    #[test]
    fn rejects_invalid_channel_or_extra_message() {
        for output in [
            "<|channel|>developer<|message|>no<|return|>",
            "<|channel|>analysis<|message|>one<|end|>\
             <|start|>assistant<|channel|>analysis<|message|>two<|end|>\
             <|start|>assistant<|channel|>final<|message|>answer<|return|>",
        ] {
            assert!(
                parse_harmony_response(output).is_err(),
                "accepted {output:?}"
            );
        }
    }
}
