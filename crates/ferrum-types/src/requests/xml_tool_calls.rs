//! Framing for native function/parameter tool calls with verbatim text values.
//!
//! This is a raw-text protocol, not general XML: parameter values may contain
//! tag literals. A closing parameter marker is structural when followed by a
//! parameter opener or a function closer. Other occurrences remain payload.
//! Once that continuation establishes a boundary we never backtrack over it.
//! Fully formed framing bytes within a raw value are inherently ambiguous;
//! this parser cannot infer whether their author intended text or structure.

use super::{
    api_tool_name_allowed, validate_parsed_tool_calls,
    xml_parameter_schema::XmlParameterSchemaProbe, ApiChatRequest, ApiFunctionCall, ApiToolCall,
    MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE,
};
use serde_json::{Map, Value};

const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

pub(super) fn has_native_envelope(text: &str) -> bool {
    text.find(TOOL_START).is_some_and(|start| {
        text[start + TOOL_START.len()..]
            .trim_start()
            .starts_with(FUNCTION_START)
    })
}

pub(super) fn parse(
    text: &str,
    request: &ApiChatRequest,
    require_complete_envelopes: bool,
) -> Option<Vec<ApiToolCall>> {
    parse_with_content(text, request, require_complete_envelopes).map(|parsed| parsed.calls)
}

pub(super) struct Parsed {
    pub(super) calls: Vec<ApiToolCall>,
    pub(super) content: String,
}

pub(super) fn parse_with_content(
    text: &str,
    request: &ApiChatRequest,
    require_complete_envelopes: bool,
) -> Option<Parsed> {
    let mut remaining = text;
    let mut calls = Vec::new();
    let mut content = String::new();
    loop {
        let Some(start) = remaining.find(TOOL_START) else {
            valid_outside_text(remaining, require_complete_envelopes).then_some(())?;
            content.push_str(remaining);
            break;
        };
        if calls.len() == MAX_PARALLEL_TOOL_CALLS_PER_RESPONSE
            || !valid_outside_text(&remaining[..start], require_complete_envelopes)
        {
            return None;
        }
        let (call, rest) = parse_one(&remaining[start..], request, calls.len())?;
        // Only the structural parser knows where parameter payload ends.
        // Preserve its outside spans directly; marker literals inside a value
        // must never be recovered as assistant prose by another text scan.
        content.push_str(&remaining[..start]);
        calls.push(call);
        remaining = rest;
    }
    let calls = validate_parsed_tool_calls(calls)?;
    // Keep the existing empty-content behavior for tools separated only by
    // framing whitespace. Otherwise retain every outside byte in order.
    if content.trim().is_empty() {
        content.clear();
    }
    Some(Parsed { calls, content })
}

fn valid_outside_text(text: &str, require_complete_envelopes: bool) -> bool {
    if require_complete_envelopes {
        return text.trim().is_empty();
    }
    // Ordinary auto calls can accompany prose. Stray framing cannot: it may
    // be the unconsumed outer tail of an apparent call inside a raw value.
    ![
        TOOL_END,
        FUNCTION_START,
        FUNCTION_END,
        PARAMETER_START,
        PARAMETER_END,
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

pub(super) fn parse_one<'a>(
    text: &'a str,
    request: &ApiChatRequest,
    index: usize,
) -> Option<(ApiToolCall, &'a str)> {
    let body = text.strip_prefix(TOOL_START)?.trim_start();
    let (name, mut remaining) = named_tag(body, FUNCTION_START)?;
    if !api_tool_name_allowed(request, name) {
        return None;
    }
    let schema = request
        .tools
        .iter()
        .find(|tool| tool.tool_type == "function" && tool.function.name == name)
        .and_then(|tool| tool.function.parameters.as_ref());
    let mut probe = schema.map(XmlParameterSchemaProbe::new);
    let mut arguments = Map::new();
    loop {
        remaining = remaining.trim_start();
        if let Some(after_function) = remaining.strip_prefix(FUNCTION_END) {
            remaining = after_function.trim_start().strip_prefix(TOOL_END)?;
            break;
        }
        let (parameter, value_start) = named_tag(remaining, PARAMETER_START)?;
        if arguments.contains_key(parameter) {
            return None;
        }
        let (raw_value, rest) = parameter_value(value_start)?;
        let value = strip_wrapper_newlines(raw_value);
        let value = probe.as_mut().map_or_else(
            || Value::String(value.to_owned()),
            |probe| probe.decode(parameter, value),
        );
        arguments.insert(parameter.to_owned(), value);
        remaining = rest;
    }
    Some((
        ApiToolCall {
            id: format!("call_{index}"),
            tool_type: "function".to_owned(),
            function: ApiFunctionCall {
                name: name.to_owned(),
                arguments: serde_json::to_string(&arguments).ok()?,
            },
        },
        remaining,
    ))
}

fn named_tag<'a>(text: &'a str, prefix: &str) -> Option<(&'a str, &'a str)> {
    let rest = text.strip_prefix(prefix)?;
    let end = rest.find('>')?;
    let name = rest[..end].trim();
    if name.is_empty() || name.contains('<') {
        return None;
    }
    Some((name, &rest[end + 1..]))
}

fn parameter_value(text: &str) -> Option<(&str, &str)> {
    // Each candidate and the whitespace immediately following it are visited
    // once. There is no recursion, speculative parse tree or suffix retry, so
    // tag-looking payload cannot cause exponential or quadratic backtracking.
    for (end, _) in text.match_indices('<') {
        let marker = &text[end..];
        if marker
            .strip_prefix(TOOL_START)
            .is_some_and(|after| after.trim_start().starts_with(FUNCTION_START))
        {
            // A nested native envelope is ambiguous with a new call after a
            // truncated value. Never consume that call as this one's payload.
            return None;
        }
        if !marker.starts_with(PARAMETER_END) {
            continue;
        }
        let after = &text[end + PARAMETER_END.len()..];
        let continuation = after.trim_start();
        if continuation.starts_with(PARAMETER_START) || continuation.starts_with(FUNCTION_END) {
            return Some((&text[..end], after));
        }
        if [PARAMETER_END, TOOL_START, TOOL_END, FUNCTION_START]
            .iter()
            .any(|marker| continuation.starts_with(marker))
        {
            // A misplaced structural marker cannot close this parameter.
            // Do not swallow it and search for a later call's valid ending.
            return None;
        }
    }
    None
}

/// Remove only the template's one matching framing newline. All payload
/// indentation, escaping, line endings and trailing whitespace are retained.
fn strip_wrapper_newlines(value: &str) -> &str {
    if let Some(value) = value.strip_prefix("\r\n") {
        return value.strip_suffix("\r\n").unwrap_or(value);
    }
    if let Some(value) = value.strip_prefix('\n') {
        if value.ends_with("\r\n") {
            return value;
        }
        return value.strip_suffix('\n').unwrap_or(value);
    }
    value
}

#[cfg(test)]
mod tests;
