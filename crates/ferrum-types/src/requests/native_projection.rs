//! Monotonic output projection for declared automatic native XML tools.
//!
//! Only the initial reasoning envelope owns a reasoning channel. Once the
//! response enters its body, later reasoning tags are literal text. Tool
//! parameters are parsed by the native framing owner, never by a tag regex.

use super::{
    xml_tool_calls, ApiChatMessage, ApiChatRequest, ApiChatResponse, ApiMessageRole, ApiRequest,
    ApiToolCallProtocol, ApiToolChoice, InferenceRequest, PROMPT_OPENED_REASONING_METADATA_KEY,
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
enum InitialBoundary {
    Pending,
    Body,
    Reasoning { start: usize, prefix_end: usize },
}

#[derive(Debug, Clone, Copy)]
enum PrefixPhase {
    Initial,
    Reasoning {
        start: usize,
        scan: usize,
        prefix_end: usize,
    },
    Body {
        scan: usize,
        skip_newlines: bool,
    },
    Held,
}

/// Holds uncertain framing and exposes only an irrevocable ordinary-text prefix.
/// Calls and their arguments are withheld until the original terminal contract
/// permits them. Unknown generation state and other output protocols opt out.
pub struct NativeChatOutputProjector {
    request: ApiChatRequest,
    started_in_think: bool,
    raw: String,
    phase: PrefixPhase,
    visible_prefix: String,
    closed_reasoning: Option<(usize, usize)>,
}

impl NativeChatOutputProjector {
    pub fn for_request(request: &InferenceRequest) -> Option<Self> {
        let ApiRequest::Chat(chat) = request.api_request.as_ref()? else {
            return None;
        };
        let started = request
            .metadata
            .get(PROMPT_OPENED_REASONING_METADATA_KEY)?
            .as_bool()?;
        Self::new(chat, request.sampling_params.model_output_protocol, started)
    }

    pub fn new(
        request: &ApiChatRequest,
        protocol: ModelOutputProtocol,
        started_in_think: bool,
    ) -> Option<Self> {
        let automatic = request.tool_choice.as_ref().is_none_or(|choice| {
            matches!(choice, ApiToolChoice::Mode(mode) if mode.eq_ignore_ascii_case("auto"))
        });
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
            return None;
        }
        Some(Self {
            request: request.clone(),
            started_in_think,
            raw: String::new(),
            phase: PrefixPhase::Initial,
            visible_prefix: String::new(),
            closed_reasoning: None,
        })
    }

    pub fn push(&mut self, text: &str) {
        self.raw.push_str(text);
        loop {
            match self.phase {
                PrefixPhase::Initial => match initial_boundary(&self.raw, self.started_in_think) {
                    InitialBoundary::Pending => return,
                    InitialBoundary::Body => {
                        self.phase = PrefixPhase::Body {
                            scan: 0,
                            skip_newlines: false,
                        }
                    }
                    InitialBoundary::Reasoning { start, prefix_end } => {
                        self.phase = PrefixPhase::Reasoning {
                            start,
                            scan: start,
                            prefix_end,
                        };
                    }
                },
                PrefixPhase::Reasoning {
                    start,
                    mut scan,
                    prefix_end,
                } => {
                    while scan < self.raw.len() {
                        let tail = &self.raw[scan..];
                        if tail.starts_with(TOOL_START) {
                            // A reasoning-owned call may take precedence over later
                            // body text. Its payload can contain a literal closer.
                            self.phase = PrefixPhase::Held;
                            return;
                        }
                        if tail.starts_with(THINK_END_TAG) {
                            self.closed_reasoning = (scan > start).then_some((start, scan));
                            self.visible_prefix.push_str(&self.raw[..prefix_end]);
                            self.phase = PrefixPhase::Body {
                                scan: scan + THINK_END_TAG.len(),
                                skip_newlines: true,
                            };
                            break;
                        }
                        if TOOL_START.starts_with(tail) || THINK_END_TAG.starts_with(tail) {
                            self.phase = PrefixPhase::Reasoning {
                                start,
                                scan,
                                prefix_end,
                            };
                            return;
                        }
                        scan += tail.chars().next().expect("nonempty suffix").len_utf8();
                    }
                    if matches!(self.phase, PrefixPhase::Reasoning { .. }) {
                        self.phase = PrefixPhase::Reasoning {
                            start,
                            scan,
                            prefix_end,
                        };
                        return;
                    }
                }
                PrefixPhase::Body {
                    mut scan,
                    mut skip_newlines,
                } => {
                    // Match terminal projection's removal of framing newlines
                    // directly following the first reasoning closer.
                    if skip_newlines {
                        while self
                            .raw
                            .as_bytes()
                            .get(scan)
                            .is_some_and(|byte| matches!(byte, b'\r' | b'\n'))
                        {
                            scan += 1;
                        }
                        skip_newlines = scan == self.raw.len();
                    }
                    let tail = &self.raw[scan..];
                    if let Some(marker) = tail.find(TOOL_START) {
                        self.visible_prefix.push_str(&tail[..marker]);
                        self.phase = PrefixPhase::Held;
                        return;
                    }
                    let held = partial_suffix_len(tail, TOOL_START);
                    let end = self.raw.len() - held;
                    self.visible_prefix.push_str(&self.raw[scan..end]);
                    self.phase = PrefixPhase::Body {
                        scan: end,
                        skip_newlines,
                    };
                    return;
                }
                PrefixPhase::Held => return,
            }
        }
    }

    /// A cumulative prefix, suitable for the existing sent-length bookkeeping.
    /// Whitespace alone remains pending because a tools-only result drops it.
    pub fn visible_prefix(&self) -> &str {
        if self.visible_prefix.trim().is_empty() {
            ""
        } else {
            &self.visible_prefix
        }
    }

    /// The completed initial reasoning channel can precede an irrevocable body.
    /// A block with unresolved tool intent remains held until completion.
    pub fn reasoning_prefix(&self) -> Option<&str> {
        self.closed_reasoning
            .map(|(start, end)| &self.raw[start..end])
    }

    pub fn finish(self, finish_reason: FinishReason) -> NativeChatOutputProjection {
        let (mut content, reasoning) =
            split_initial_reasoning(&self.raw, self.started_in_think, &self.request);
        let api_response = if matches!(finish_reason, FinishReason::Stop | FinishReason::EOS) {
            let reasoning_calls = reasoning
                .as_deref()
                .and_then(|text| xml_tool_calls::parse_with_content(text, &self.request, false));
            let parsed = if let Some(parsed) = reasoning_calls {
                content.clear();
                Some((String::new(), parsed.calls))
            } else {
                xml_tool_calls::parse_with_content(&content, &self.request, false)
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
        NativeChatOutputProjection {
            visible: ParsedReasoningResponse { content, reasoning },
            api_response,
        }
    }
}

fn initial_boundary(raw: &str, started: bool) -> InitialBoundary {
    if started {
        if raw.len() < THINK_START_TAG.len() && THINK_START_TAG.starts_with(raw) {
            return InitialBoundary::Pending;
        }
        let start = if raw.starts_with(THINK_START_TAG) {
            THINK_START_TAG.len()
        } else {
            0
        };
        return InitialBoundary::Reasoning {
            start,
            prefix_end: 0,
        };
    }
    let trimmed = raw.trim_start();
    if trimmed.len() < THINK_START_TAG.len() && THINK_START_TAG.starts_with(trimmed) {
        return InitialBoundary::Pending;
    }
    if trimmed.starts_with(THINK_START_TAG) {
        let prefix_end = raw.len() - trimmed.len();
        InitialBoundary::Reasoning {
            start: prefix_end + THINK_START_TAG.len(),
            prefix_end,
        }
    } else {
        InitialBoundary::Body
    }
}

fn partial_suffix_len(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|length| text.ends_with(&marker[..*length]))
        .unwrap_or(0)
}

fn split_initial_reasoning(
    raw: &str,
    started: bool,
    request: &ApiChatRequest,
) -> (String, Option<String>) {
    let (start, prefix_end) = match initial_boundary(raw, started) {
        InitialBoundary::Body => return (raw.to_owned(), None),
        InitialBoundary::Pending if !started => return (raw.to_owned(), None),
        InitialBoundary::Pending => {
            return (String::new(), (!raw.is_empty()).then(|| raw.to_owned()))
        }
        InitialBoundary::Reasoning { start, prefix_end } => (start, prefix_end),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn request() -> ApiChatRequest {
        serde_json::from_value(json!({
            "messages": [], "tool_call_protocol": "function_parameter_xml",
            "tools": [{"type": "function", "function": {
                "name": "write", "parameters": {
                    "type": "object", "properties": {"content": {"type": "string"}}
                }
            }}]
        }))
        .unwrap()
    }

    fn call(content: &str) -> String {
        format!("<tool_call><function=write><parameter=content>{content}</parameter></function></tool_call>")
    }

    fn project(raw: &str, started: bool, finish: FinishReason) -> NativeChatOutputProjection {
        let mut projector =
            NativeChatOutputProjector::new(&request(), ModelOutputProtocol::Text, started).unwrap();
        projector.push(raw);
        projector.finish(finish)
    }

    #[test]
    fn native_projection_keeps_every_published_prefix_at_every_chunk_boundary() {
        let cases = [
            (
                false,
                "A Unicode body: 字节. Later </think> and <think> are literal.".into(),
            ),
            (false, " \n<think>Private.</think>\r\nPublic body.".into()),
            (true, "Private.</think>\r\nPublic body.".into()),
            (true, "<think>Private.</think>\r\nPublic body.".into()),
            (
                false,
                format!(
                    "Before.\n{}\nAfter.",
                    call("literal </think> and </tool_call>")
                ),
            ),
            (false, format!(" \n{}\n ", call("only payload"))),
            (
                false,
                format!(
                    "<think>Private. {}</think>Discarded body. {}",
                    call("reasoning choice"),
                    call("body choice")
                ),
            ),
            (
                true,
                format!(
                    "Private. {}</think>Discarded body.",
                    call("<think>literal</think> </tool_call>")
                ),
            ),
            (
                false,
                format!(
                    "Before. {}<tool_call><function=write><parameter=content>unfinished",
                    call("first")
                ),
            ),
            (false, "<thi".into()),
            (true, "Unfinished private reasoning <tool_".into()),
        ];
        for (started, raw) in cases {
            for finish in [FinishReason::Stop, FinishReason::EOS, FinishReason::Length] {
                let final_projection = project(&raw, started, finish);
                // Exercise both arbitrary two-chunk splits and one character
                // per chunk, including every framing and UTF-8 boundary.
                for split in raw
                    .char_indices()
                    .map(|(index, _)| index)
                    .chain([raw.len()])
                {
                    let mut projector = NativeChatOutputProjector::new(
                        &request(),
                        ModelOutputProtocol::Text,
                        started,
                    )
                    .unwrap();
                    projector.push(&raw[..split]);
                    assert!(
                        final_projection
                            .visible
                            .content
                            .starts_with(projector.visible_prefix()),
                        "split {split}: {raw}"
                    );
                    projector.push(&raw[split..]);
                    assert!(
                        final_projection
                            .visible
                            .content
                            .starts_with(projector.visible_prefix()),
                        "final prefix: {raw}"
                    );
                    let chunked = projector.finish(finish);
                    assert_eq!(chunked.visible, final_projection.visible);
                    assert_eq!(chunked.api_response, final_projection.api_response);
                }
                let mut projector =
                    NativeChatOutputProjector::new(&request(), ModelOutputProtocol::Text, started)
                        .unwrap();
                for character in raw.chars() {
                    projector.push(character.encode_utf8(&mut [0; 4]));
                    assert!(
                        final_projection
                            .visible
                            .content
                            .starts_with(projector.visible_prefix()),
                        "character prefix: {raw}"
                    );
                }
                assert_eq!(projector.finish(finish).visible, final_projection.visible);
            }
        }
    }

    #[test]
    fn native_projection_never_reopens_body_or_guesses_json_calls() {
        let text = "Explanation </think> <think>literal</think> {\"name\":\"write\",\"arguments\":{\"content\":\"x\"}}";
        let projected = project(text, false, FinishReason::EOS);
        assert_eq!(projected.visible.content, text);
        assert_eq!(projected.visible.reasoning, None);
        assert_eq!(projected.api_response, None);
        let projected = project(
            "<think>Private.</think>\r\nVisible <think>literal</think>.",
            false,
            FinishReason::Stop,
        );
        assert_eq!(projected.visible.content, "Visible <think>literal</think>.");
        assert_eq!(projected.visible.reasoning.as_deref(), Some("Private."));
    }

    #[test]
    fn native_projection_preserves_reasoning_owned_call_and_literal_parameters() {
        let payload = "const MARK: &str = \"<think>literal</think> </tool_call>\";\r\n";
        let reasoning = format!("Private. {}", call(payload));
        let raw = format!("{reasoning}</think>Body call {}", call("must not win"));
        let projected = project(&raw, true, FinishReason::Stop);
        assert!(projected.visible.content.is_empty());
        assert_eq!(
            projected.visible.reasoning.as_deref(),
            Some(reasoning.as_str())
        );
        let response = projected.api_response.unwrap();
        assert_eq!(response.message.tool_calls.len(), 1);
        let arguments: Value =
            serde_json::from_str(&response.message.tool_calls[0].function.arguments).unwrap();
        assert_eq!(arguments["content"], payload);
        let limited = project(&raw, true, FinishReason::Length);
        assert!(limited.api_response.is_none());
        assert_eq!(
            limited.visible.content,
            format!("Body call {}", call("must not win"))
        );
        let unfinished = project(
            "Private. <tool_call><function=write><parameter=content>literal </think> still payload",
            true,
            FinishReason::Length,
        );
        assert!(unfinished.visible.content.is_empty());
        assert!(unfinished.api_response.is_none());
    }

    #[test]
    fn native_projection_requires_known_render_state_and_keeps_other_contracts_buffered() {
        let mut inference = InferenceRequest::new("prompt", "fixture");
        inference.api_request = Some(ApiRequest::Chat(request()));
        assert!(NativeChatOutputProjector::for_request(&inference).is_none());
        inference
            .metadata
            .insert(PROMPT_OPENED_REASONING_METADATA_KEY.into(), json!(false));
        assert!(NativeChatOutputProjector::for_request(&inference).is_some());
        let mut required = request();
        required.tool_choice = Some(ApiToolChoice::Mode("required".into()));
        let mut named = request();
        named.tool_choice = Some(
            serde_json::from_value(json!({"type": "function", "function": {"name": "write"}}))
                .unwrap(),
        );
        let mut hard = request();
        hard.response_format =
            Some(serde_json::from_value(json!({"type": "json_object"})).unwrap());
        let mut legacy = request();
        legacy.legacy_functions = vec![legacy.tools[0].function.clone()];
        let mut json_request = request();
        json_request.tool_call_protocol = ApiToolCallProtocol::Json;
        for other in [required, named, hard, legacy, json_request] {
            assert!(
                NativeChatOutputProjector::new(&other, ModelOutputProtocol::Text, false).is_none()
            );
        }
        for protocol in [
            ModelOutputProtocol::HarmonyGptOss,
            ModelOutputProtocol::GemmaThought,
        ] {
            assert!(NativeChatOutputProjector::new(&request(), protocol, false).is_none());
        }
    }
}
