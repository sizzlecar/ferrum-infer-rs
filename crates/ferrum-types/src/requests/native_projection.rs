//! Monotonic output projection for declared automatic native XML tools.
//!
//! Only the initial reasoning envelope owns a reasoning channel. Once the
//! response enters its body, later reasoning tags are literal text. Tool
//! parameters are parsed by the native framing owner, never by a tag regex.

use super::{
    xml_tool_calls, ApiChatMessage, ApiChatRequest, ApiChatResponse, ApiMessageRole,
    ApiToolCallProtocol, ApiToolChoice,
};
use crate::{
    FinishReason, ModelOutputProtocol, ParsedReasoningResponse, THINK_END_TAG, THINK_START_TAG,
};

const TOOL_START: &str = "<tool_call>";

/// Final projection shared by engine completion and both HTTP response modes.
#[derive(Debug, Clone)]
pub struct NativeChatOutputProjection {
    pub visible: ParsedReasoningResponse,
    pub api_response: Option<ApiChatResponse>,
}

#[derive(Debug, Clone, Copy)]
pub enum NativeInitialBoundary {
    Pending,
    Body,
    Reasoning { start: usize, prefix_end: usize },
}

/// Pure eligibility; contains no mutable output or budget owner.
pub fn native_chat_projection_eligible(
    request: &ApiChatRequest,
    protocol: ModelOutputProtocol,
) -> bool {
    let automatic = request.tool_choice.as_ref().is_none_or(
        |choice| matches!(choice, ApiToolChoice::Mode(mode) if mode.eq_ignore_ascii_case("auto")),
    );
    let plain = request
        .response_format
        .as_ref()
        .is_none_or(|format| format.format_type == "text");
    if request.tool_call_protocol != ApiToolCallProtocol::FunctionParameterXml
        || protocol != ModelOutputProtocol::Text
        || request.tools.is_empty()
        || !request.legacy_functions.is_empty()
        || request.legacy_function_call.is_some()
        || !automatic
        || !plain
    {
        return false;
    }
    true
}

pub fn project_native_chat_output(
    request: &ApiChatRequest,
    protocol: ModelOutputProtocol,
    started_in_think: bool,
    raw: &str,
    finish_reason: FinishReason,
) -> Option<NativeChatOutputProjection> {
    if !native_chat_projection_eligible(request, protocol) {
        return None;
    }
    let (mut content, reasoning) = split_initial_reasoning(raw, started_in_think, request);
    let api_response = if matches!(finish_reason, FinishReason::Stop | FinishReason::EOS) {
        let reasoning_calls = reasoning
            .as_deref()
            .and_then(|text| xml_tool_calls::parse_with_content(text, request, false));
        let parsed = if let Some(parsed) = reasoning_calls {
            content.clear();
            Some((String::new(), parsed.calls))
        } else {
            xml_tool_calls::parse_with_content(&content, request, false)
                .map(|parsed| (parsed.content, parsed.calls))
        };
        parsed.map(|(outside, calls)| {
            content = outside.clone();
            ApiChatResponse {
                message: ApiChatMessage {
                    role: ApiMessageRole::Assistant,
                    content: outside,
                    name: None,
                    tool_calls: calls,
                    tool_call_id: None,
                    function_call: None,
                },
                finish_reason: Some("tool_calls".to_owned()),
            }
        })
    } else {
        // In particular, Length never turns a partial envelope into an
        // executable call. Its text/usage retain the terminal fallback.
        None
    };
    Some(NativeChatOutputProjection {
        visible: ParsedReasoningResponse { content, reasoning },
        api_response,
    })
}

pub fn native_initial_boundary(raw: &str, started: bool) -> NativeInitialBoundary {
    if started {
        if raw.len() < THINK_START_TAG.len() && THINK_START_TAG.starts_with(raw) {
            return NativeInitialBoundary::Pending;
        }
        let start = if raw.starts_with(THINK_START_TAG) {
            THINK_START_TAG.len()
        } else {
            0
        };
        return NativeInitialBoundary::Reasoning {
            start,
            prefix_end: 0,
        };
    }
    let trimmed = raw.trim_start();
    if trimmed.len() < THINK_START_TAG.len() && THINK_START_TAG.starts_with(trimmed) {
        return NativeInitialBoundary::Pending;
    }
    if trimmed.starts_with(THINK_START_TAG) {
        let prefix_end = raw.len() - trimmed.len();
        NativeInitialBoundary::Reasoning {
            start: prefix_end + THINK_START_TAG.len(),
            prefix_end,
        }
    } else {
        NativeInitialBoundary::Body
    }
}

fn split_initial_reasoning(
    raw: &str,
    started: bool,
    request: &ApiChatRequest,
) -> (String, Option<String>) {
    let (start, prefix_end) = match native_initial_boundary(raw, started) {
        NativeInitialBoundary::Body => return (raw.to_owned(), None),
        NativeInitialBoundary::Pending if !started => return (raw.to_owned(), None),
        NativeInitialBoundary::Pending => {
            return (String::new(), (!raw.is_empty()).then(|| raw.to_owned()))
        }
        NativeInitialBoundary::Reasoning { start, prefix_end } => (start, prefix_end),
    };
    let mut scan = start;
    while scan < raw.len() {
        let tail = &raw[scan..];
        if tail.starts_with(THINK_END_TAG) {
            let reasoning = &raw[start..scan];
            let body = tail[THINK_END_TAG.len()..].trim_start_matches(['\r', '\n']);
            return (
                format!("{}{body}", &raw[..prefix_end]),
                (!reasoning.is_empty()).then(|| reasoning.to_owned()),
            );
        }
        if tail.starts_with(TOOL_START) && xml_tool_calls::has_native_envelope(tail) {
            // The structural parser owns parameter boundaries, including tag
            // literals. An incomplete native call cannot expose its tail as body.
            let Some((_, rest)) = xml_tool_calls::parse_one(tail, request, 0) else {
                break;
            };
            scan = raw.len() - rest.len();
        } else {
            scan += tail.chars().next().expect("nonempty suffix").len_utf8();
        }
    }
    let reasoning = &raw[start..];
    (
        raw[..prefix_end].to_owned(),
        (!reasoning.is_empty()).then(|| reasoning.to_owned()),
    )
}
