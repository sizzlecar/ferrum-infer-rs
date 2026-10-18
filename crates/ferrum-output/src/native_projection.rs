//! Charged incremental native projection. Static protocol parsing stays in types.
use crate::{semantic_cost, OutputSemanticLease, OutputSession};
use ferrum_types::{
    native_chat_projection_eligible, native_initial_boundary, project_native_chat_output,
    ApiChatRequest, ApiRequest, FerrumError, FinishReason, InferenceRequest, ModelOutputProtocol,
    NativeChatOutputProjection, NativeInitialBoundary, Result,
    PROMPT_OPENED_REASONING_METADATA_KEY, THINK_END_TAG,
};
const TOOL_START: &str = "<tool_call>";
#[cfg(test)]
mod semantic_tests;
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
    lease: OutputSemanticLease,
    request: ApiChatRequest,
    started_in_think: bool,
    raw: String,
    phase: PrefixPhase,
    visible_prefix: String,
    closed_reasoning: Option<(usize, usize)>,
}

pub struct OwnedNativeChatOutputProjection {
    projection: NativeChatOutputProjection,
    lease: OutputSemanticLease,
}
impl OwnedNativeChatOutputProjection {
    pub fn projection(&self) -> &NativeChatOutputProjection {
        &self.projection
    }
    pub fn output_session(&self) -> &OutputSession {
        self.lease.output_session()
    }

    /// Transfer the projection and its existing content credit together.
    ///
    /// The caller must retain the lease for as long as it retains the moved
    /// content, and resize it before retaining any replacement or extra copy.
    pub fn into_parts(self) -> (NativeChatOutputProjection, OutputSemanticLease) {
        (self.projection, self.lease)
    }
}
impl NativeChatOutputProjector {
    pub fn for_request(
        session: &OutputSession,
        request: &InferenceRequest,
    ) -> Result<Option<Self>> {
        let Some(ApiRequest::Chat(chat)) = request.api_request.as_ref() else {
            return Ok(None);
        };
        let Some(started) = request
            .metadata
            .get(PROMPT_OPENED_REASONING_METADATA_KEY)
            .and_then(serde_json::Value::as_bool)
        else {
            return Ok(None);
        };
        Self::new(
            session,
            chat,
            request.sampling_params.model_output_protocol,
            started,
        )
    }
    pub fn new(
        session: &OutputSession,
        request: &ApiChatRequest,
        protocol: ModelOutputProtocol,
        started_in_think: bool,
    ) -> Result<Option<Self>> {
        if !native_chat_projection_eligible(request, protocol) {
            return Ok(None);
        }
        let request = ApiChatRequest {
            messages: Vec::new(),
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            tool_call_protocol: request.tool_call_protocol,
            legacy_functions: request.legacy_functions.clone(),
            legacy_function_call: request.legacy_function_call.clone(),
            response_format: request.response_format.clone(),
            stream_options: request.stream_options.clone(),
        };
        Ok(Some(Self {
            lease: session.try_reserve_semantic(0)?,
            request,
            started_in_think,
            raw: String::new(),
            phase: PrefixPhase::Initial,
            visible_prefix: String::new(),
            closed_reasoning: None,
        }))
    }
    pub fn push(&mut self, text: &str) -> Result<()> {
        let old = (
            self.raw.len(),
            self.visible_prefix.len(),
            self.phase,
            self.closed_reasoning,
            self.lease.bytes(),
        );
        if let Err(error) = self.push_inner(text) {
            self.raw.truncate(old.0);
            self.visible_prefix.truncate(old.1);
            self.phase = old.2;
            self.closed_reasoning = old.3;
            self.lease.shrink_to(old.4)?;
            return Err(error);
        }
        Ok(())
    }
    fn push_inner(&mut self, text: &str) -> Result<()> {
        self.lease.try_grow(text.len())?;
        self.raw.push_str(text);
        loop {
            match self.phase {
                PrefixPhase::Initial => {
                    match native_initial_boundary(&self.raw, self.started_in_think) {
                        NativeInitialBoundary::Pending => return Ok(()),
                        NativeInitialBoundary::Body => {
                            self.phase = PrefixPhase::Body {
                                scan: 0,
                                skip_newlines: false,
                            }
                        }
                        NativeInitialBoundary::Reasoning { start, prefix_end } => {
                            self.phase = PrefixPhase::Reasoning {
                                start,
                                scan: start,
                                prefix_end,
                            };
                        }
                    }
                }
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
                            return Ok(());
                        }
                        if tail.starts_with(THINK_END_TAG) {
                            self.closed_reasoning = (scan > start).then_some((start, scan));
                            self.lease.try_grow((&self.raw[..prefix_end]).len())?;
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
                            return Ok(());
                        }
                        scan += tail.chars().next().expect("nonempty suffix").len_utf8();
                    }
                    if matches!(self.phase, PrefixPhase::Reasoning { .. }) {
                        self.phase = PrefixPhase::Reasoning {
                            start,
                            scan,
                            prefix_end,
                        };
                        return Ok(());
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
                        self.lease.try_grow((&tail[..marker]).len())?;
                        self.visible_prefix.push_str(&tail[..marker]);
                        self.phase = PrefixPhase::Held;
                        return Ok(());
                    }
                    let held = partial_suffix_len(tail, TOOL_START);
                    let end = self.raw.len() - held;
                    self.lease.try_grow((&self.raw[scan..end]).len())?;
                    self.visible_prefix.push_str(&self.raw[scan..end]);
                    self.phase = PrefixPhase::Body {
                        scan: end,
                        skip_newlines,
                    };
                    return Ok(());
                }
                PrefixPhase::Held => return Ok(()),
            }
        }
    }

    pub fn raw(&self) -> &str {
        &self.raw
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

    pub fn finish(self, finish_reason: FinishReason) -> Result<OwnedNativeChatOutputProjection> {
        let projection = project_native_chat_output(
            &self.request,
            ModelOutputProtocol::Text,
            self.started_in_think,
            &self.raw,
            finish_reason,
        )
        .ok_or_else(|| FerrumError::internal("native projector lost its declared contract"))?;
        let api_bytes = semantic_cost(&projection.api_response)?;
        let bytes = semantic_cost(&(&projection.visible.content, &projection.visible.reasoning))?
            .checked_add(api_bytes)
            .ok_or_else(|| FerrumError::resource_exhausted("native projection size overflow"))?;
        let Self {
            mut lease,
            raw,
            visible_prefix,
            ..
        } = self;
        drop(raw);
        drop(visible_prefix);
        lease.try_resize(bytes)?;
        Ok(OwnedNativeChatOutputProjection { projection, lease })
    }
}
fn partial_suffix_len(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|length| text.ends_with(&marker[..*length]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{ApiToolCallProtocol, ApiToolChoice};
    fn session() -> OutputSession {
        crate::OutputBudget::new(Default::default())
            .unwrap()
            .try_open_session()
            .unwrap()
    }
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
        let mut projector = NativeChatOutputProjector::new(
            &session(),
            &request(),
            ModelOutputProtocol::Text,
            started,
        )
        .unwrap()
        .unwrap();
        projector.push(raw).unwrap();
        projector.finish(finish).unwrap().projection().clone()
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
                        &session(),
                        &request(),
                        ModelOutputProtocol::Text,
                        started,
                    )
                    .unwrap()
                    .unwrap();
                    projector.push(&raw[..split]).unwrap();
                    assert!(
                        final_projection
                            .visible
                            .content
                            .starts_with(projector.visible_prefix()),
                        "split {split}: {raw}"
                    );
                    projector.push(&raw[split..]).unwrap();
                    assert!(
                        final_projection
                            .visible
                            .content
                            .starts_with(projector.visible_prefix()),
                        "final prefix: {raw}"
                    );
                    let chunked = projector.finish(finish).unwrap().projection().clone();
                    assert_eq!(chunked.visible, final_projection.visible);
                    assert_eq!(chunked.api_response, final_projection.api_response);
                }
                let mut projector = NativeChatOutputProjector::new(
                    &session(),
                    &request(),
                    ModelOutputProtocol::Text,
                    started,
                )
                .unwrap()
                .unwrap();
                for character in raw.chars() {
                    projector.push(character.encode_utf8(&mut [0; 4])).unwrap();
                    assert!(
                        final_projection
                            .visible
                            .content
                            .starts_with(projector.visible_prefix()),
                        "character prefix: {raw}"
                    );
                }
                assert_eq!(
                    projector
                        .finish(finish)
                        .unwrap()
                        .projection()
                        .clone()
                        .visible,
                    final_projection.visible
                );
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
        assert!(
            NativeChatOutputProjector::for_request(&session(), &inference)
                .unwrap()
                .is_none()
        );
        inference
            .metadata
            .insert(PROMPT_OPENED_REASONING_METADATA_KEY.into(), json!(false));
        assert!(
            NativeChatOutputProjector::for_request(&session(), &inference)
                .unwrap()
                .is_some()
        );
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
            assert!(NativeChatOutputProjector::new(
                &session(),
                &other,
                ModelOutputProtocol::Text,
                false
            )
            .unwrap()
            .is_none());
        }
        for protocol in [
            ModelOutputProtocol::HarmonyGptOss,
            ModelOutputProtocol::GemmaThought,
        ] {
            assert!(
                NativeChatOutputProjector::new(&session(), &request(), protocol, false)
                    .unwrap()
                    .is_none()
            );
        }
    }
}
