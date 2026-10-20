use super::{wire, ResponseContext};
use crate::axum_server::{
    chat_stream::{ChatEventStream, ChatStreamEvent},
    output, stream_error_payload,
};
use crate::openai::AssistantMessagePhase;
use crate::openai::Usage;
use axum::response::Response;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use uuid::Uuid;

type ResponseEvents<'a> = std::pin::Pin<Box<dyn futures::Stream<Item = ResponseEvent> + Send + 'a>>;

#[cfg(test)]
mod semantic_tests;

enum ResponseEvent {
    Json {
        name: &'static str,
        value: output::Retained<Value>,
    },
    Encoded(ferrum_types::Result<axum::body::Bytes>),
    Failed(crate::openai::OpenAiError),
    Done,
}

pub(in crate::axum_server) fn adapt_chat_stream(
    mut chat: ChatEventStream,
    context: ResponseContext,
) -> Response {
    let session = chat.session.clone();
    // Identity is server-issued; do not clone the potentially large request
    // metadata or accumulated response into an encoding-failure envelope.
    let response_id = context.id.clone();
    let created_at = context.created_at;
    let frames = async_stream::try_stream! {
        let mut state = match ResponsesStreamState::new(context, chat.session.clone()) {
            Ok(state) => state,
            Err(error) => {
                let (kind, _) = output::encoding_error_details(&error);
                let bytes = encode_upstream_failure(&chat.session, &response_id, created_at, 0, &stream_error_payload(error.to_string(), kind, None))?;
                drop(chat); yield bytes; return;
            }
        };
        let mut next_sequence = 0;
        {
            let mut events = state.send_initial_events();
            while let Some(event) = events.next().await {
                let (bytes, terminal_error) = encode_event(&chat.session, event, &response_id, created_at, &mut next_sequence).await?;
                if terminal_error { drop(chat); yield bytes; return; }
                yield bytes;
            }
        }
        while let Some(event) = chat.events.next().await {
            {
                let mut events = state.consume_chat_event(event);
                while let Some(event) = events.next().await {
                    let (bytes, terminal_error) = encode_event(&chat.session, event, &response_id, created_at, &mut next_sequence).await?;
                    if terminal_error { drop(chat); yield bytes; return; }
                    yield bytes;
                }
            }
            if state.terminal { return; }
        }
        if !state.terminal {
            let mut events = state.fail(stream_error_payload("chat stream ended before [DONE]", "internal_server_error", None));
            while let Some(event) = events.next().await {
                let (bytes, terminal_error) = encode_event(&chat.session, event, &response_id, created_at, &mut next_sequence).await?;
                if terminal_error { drop(chat); yield bytes; return; }
                yield bytes;
            }
        }
    };
    output::sse_response(session, frames)
}

async fn encode_event(
    session: &ferrum_output::OutputSession,
    event: ResponseEvent,
    response_id: &str,
    created_at: u64,
    next_sequence: &mut u64,
) -> ferrum_types::Result<(axum::body::Bytes, bool)> {
    let numbered = matches!(
        &event,
        ResponseEvent::Json { .. } | ResponseEvent::Encoded(_)
    );
    let encoded = match event {
        ResponseEvent::Json { name, value } => {
            output::encode_json(session, Some(name), value.value()).await
        }
        ResponseEvent::Encoded(encoded) => encoded,
        ResponseEvent::Done => output::encode_done(session).await,
        ResponseEvent::Failed(error) => {
            let bytes =
                encode_upstream_failure(session, response_id, created_at, *next_sequence, &error)?;
            return Ok((bytes, true));
        }
    };
    match encoded {
        Ok(bytes) => {
            if numbered {
                *next_sequence += 1;
            }
            Ok((bytes, false))
        }
        Err(error) => {
            let (code, message) = output::encoding_error_details(&error);
            let failed = json!({
                "type": "response.failed",
                "sequence_number": *next_sequence,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "failed",
                    "error": {"code": code, "message": message},
                    "output": []
                }
            });
            Ok((
                output::encode_terminal_error(session, Some("response.failed"), &failed)?,
                true,
            ))
        }
    }
}

fn encode_upstream_failure(
    session: &ferrum_output::OutputSession,
    response_id: &str,
    created_at: u64,
    sequence: u64,
    error: &crate::openai::OpenAiError,
) -> ferrum_types::Result<axum::body::Bytes> {
    #[derive(serde::Serialize)]
    struct Failure<'a> {
        #[serde(rename = "type")]
        kind: &'static str,
        sequence_number: u64,
        response: FailedResponse<'a>,
    }
    #[derive(serde::Serialize)]
    struct FailedResponse<'a> {
        id: &'a str,
        object: &'static str,
        created_at: u64,
        status: &'static str,
        error: FailureDetail<'a>,
        output: [(); 0],
    }
    #[derive(serde::Serialize)]
    struct FailureDetail<'a> {
        code: &'a str,
        message: &'a str,
    }
    let error = output::upstream_error(&error.error);
    let make = |message| Failure {
        kind: "response.failed",
        sequence_number: sequence,
        response: FailedResponse {
            id: response_id,
            object: "response",
            created_at,
            status: "failed",
            error: FailureDetail {
                code: error.code.unwrap_or(error.error_type),
                message,
            },
            output: [],
        },
    };
    output::encode_upstream_failure(
        session,
        Some("response.failed"),
        &make(error.message),
        &make(output::OMITTED_UPSTREAM_DIAGNOSTIC),
    )
}

struct ResponsesStreamState {
    history: ferrum_output::OutputSemanticLease,
    context: ResponseContext,
    sequence: u64,
    next_output_index: usize,
    reasoning_item_id: String,
    reasoning: String,
    reasoning_output_index: Option<usize>,
    reasoning_done: bool,
    text_item_id: String,
    text: String,
    text_started: bool,
    text_output_index: Option<usize>,
    tool_calls: BTreeMap<u32, ToolCallAccumulator>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    terminal: bool,
}

#[derive(Default, serde::Serialize)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

impl ResponsesStreamState {
    fn new(
        context: ResponseContext,
        session: ferrum_output::OutputSession,
    ) -> ferrum_types::Result<Self> {
        let reasoning_item_id = format!("rs_{}", Uuid::new_v4().simple());
        let text_item_id = format!("msg_{}", Uuid::new_v4().simple());
        let history = session.try_reserve_semantic(ferrum_output::semantic_cost(&(
            &reasoning_item_id,
            &text_item_id,
            BTreeMap::<u32, ToolCallAccumulator>::new(),
            Option::<String>::None,
            Option::<Usage>::None,
        ))?)?;
        Ok(Self {
            history,
            context,
            sequence: 0,
            next_output_index: 0,
            reasoning_item_id,
            reasoning: String::new(),
            reasoning_output_index: None,
            reasoning_done: false,
            text_item_id,
            text: String::new(),
            text_started: false,
            text_output_index: None,
            tool_calls: BTreeMap::new(),
            finish_reason: None,
            usage: None,
            terminal: false,
        })
    }

    fn send_initial_events(&mut self) -> ResponseEvents<'_> {
        Box::pin(async_stream::stream! {
            let session = self.history.output_session().clone();
            for kind in ["response.created", "response.in_progress"] {
                yield borrowed_event(&session, &mut self.sequence, kind, wire::ResponseEvent { response: wire::ResponseView { context: &self.context, status: "in_progress", output: &[(); 0], usage: None, completed_at: None } }).await;
            }
        })
    }

    fn consume_chat_event(&mut self, event: ChatStreamEvent) -> ResponseEvents<'_> {
        Box::pin(async_stream::stream! {
            if self.terminal { return; }
            let owner = match event {
                ChatStreamEvent::Done => { let mut events = self.complete(); while let Some(event) = events.next().await { yield event; } return; }
                ChatStreamEvent::Error(error) => { let mut events = self.fail(error); while let Some(event) = events.next().await { yield event; } return; }
                ChatStreamEvent::Chunk(chunk) => chunk,
            };
            let chunk = owner.value();
            if let Some(usage) = &chunk.usage {
                if let Err(error) = self.replace_usage(usage) { yield self.quota_error(error); return; }
            }
            for choice in &chunk.choices {
                if let Some(delta) = &choice.delta {
                    if let Some(reasoning) = delta.reasoning.as_deref().filter(|value| !value.is_empty()) {
                        let mut events = self.push_reasoning(reasoning); while let Some(event) = events.next().await { yield event; }
                    }
                    if !delta.content.is_empty() {
                        let mut events = self.push_text(&delta.content); while let Some(event) = events.next().await { yield event; }
                    }
                    let calls = delta.tool_calls.as_deref().unwrap_or_default();
                    if !calls.is_empty() { let mut events = self.finish_reasoning("completed"); while let Some(event) = events.next().await { yield event; } }
                    for (position, call) in calls.iter().enumerate() {
                        if let Err(error) = self.append_call(call.index.unwrap_or(position as u32), &call.id, &call.function.name, &call.function.arguments) { yield self.quota_error(error); return; }
                    }
                    if let Some(function) = &delta.function_call {
                        { let mut events = self.finish_reasoning("completed"); while let Some(event) = events.next().await { yield event; } }
                        let call_id = self.tool_calls.get(&0).filter(|call| !call.id.is_empty()).map(|call| call.id.clone()).unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
                        if let Err(error) = self.append_call(0, &call_id, &function.name, &function.arguments) { yield self.quota_error(error); return; }
                    }
                }
                if let Some(reason) = &choice.finish_reason {
                    let old = self.finish_reason.as_ref().map_or(0, String::len);
                    if let Err(error) = self.history.try_grow(reason.len().saturating_sub(old)) { yield self.quota_error(error); return; }
                    self.finish_reason = Some(reason.clone());
                    if old > reason.len() { self.history.shrink_to(self.history.bytes() - (old - reason.len())).expect("shrinking retained finish reason"); }
                }
            }
        })
    }

    fn quota_error(&mut self, error: ferrum_types::FerrumError) -> ResponseEvent {
        self.terminal = true;
        let (kind, _) = output::encoding_error_details(&error);
        ResponseEvent::Failed(stream_error_payload(error.to_string(), kind, None))
    }

    fn replace_usage(&mut self, usage: &Usage) -> ferrum_types::Result<()> {
        let old = ferrum_output::semantic_cost(&self.usage)?;
        let new = ferrum_output::semantic_cost(&Some(usage))?;
        self.history.try_resize(self.history.bytes() - old + new)?;
        self.usage = Some(usage.clone());
        Ok(())
    }

    fn append_call(
        &mut self,
        index: u32,
        id: &str,
        name: &str,
        arguments: &str,
    ) -> ferrum_types::Result<()> {
        let old = self.tool_calls.get(&index);
        let old_id = old.map_or(0, |call| call.id.len());
        let old_name = old.map_or(0, |call| call.name.len());
        let new_id = if id.is_empty() { old_id } else { id.len() };
        let new_name = if name.is_empty() {
            old_name
        } else {
            name.len()
        };
        let entry = if old.is_none() {
            ferrum_output::semantic_cost(&index)?
                + ferrum_output::semantic_cost(&ToolCallAccumulator::default())?
        } else {
            0
        };
        let next = self
            .history
            .bytes()
            .checked_sub(old_id + old_name)
            .and_then(|n| n.checked_add(new_id))
            .and_then(|n| n.checked_add(new_name))
            .and_then(|n| n.checked_add(arguments.len()))
            .and_then(|n| n.checked_add(entry))
            .ok_or_else(|| {
                ferrum_types::FerrumError::resource_exhausted("response call history size overflow")
            })?;
        self.history.try_resize(next)?;
        let call = self.tool_calls.entry(index).or_default();
        if !id.is_empty() {
            call.id = id.to_owned();
        }
        if !name.is_empty() {
            call.name = name.to_owned();
        }
        call.arguments.push_str(arguments);
        Ok(())
    }

    fn allocate_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn push_reasoning<'a>(&'a mut self, delta: &'a str) -> ResponseEvents<'a> {
        Box::pin(async_stream::stream! {
        if self.reasoning_output_index.is_none() {
            let output_index = self.allocate_output_index();
            self.reasoning_output_index = Some(output_index);
            let mut item = json!({
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
                "content": []
            });
            if self.context.include_encrypted_reasoning {
                item.as_object_mut()
                    .expect("reasoning item is an object")
                    .insert("encrypted_content".to_string(), Value::Null);
            }
            yield self.event("response.output_item.added",
                json!({"output_index": output_index, "item": item}),
            );
            yield self.event("response.content_part.added",
                json!({
                    "item_id": self.reasoning_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "reasoning_text", "text": ""}
                }),
            );
        }
        if let Err(error) = self.history.try_grow(delta.len()) { yield self.quota_error(error); return; }
        self.reasoning.push_str(delta);
        let output_index = self
            .reasoning_output_index
            .expect("reasoning output index assigned");
        yield self.event("response.reasoning_text.delta",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": delta
            }),
        );
        })
    }

    fn finish_reasoning<'a>(&'a mut self, status: &'a str) -> ResponseEvents<'a> {
        Box::pin(async_stream::stream! {
            let Some(output_index) = self.reasoning_output_index else { return; };
            if self.reasoning_done { return; }
            let session = self.history.output_session().clone();
            yield borrowed_event(&session, &mut self.sequence, "response.reasoning_text.done", wire::TextEvent { item_id: &self.reasoning_item_id, output_index, content_index: 0, text: &self.reasoning, logprobs: None }).await;
            yield borrowed_event(&session, &mut self.sequence, "response.content_part.done", wire::PartEvent { item_id: &self.reasoning_item_id, output_index, content_index: 0, part: wire::ReasoningPart { kind: "reasoning_text", text: &self.reasoning } }).await;
            yield borrowed_event(&session, &mut self.sequence, "response.output_item.done", wire::ItemEvent { output_index, item: wire::Item::Reasoning { id: &self.reasoning_item_id, text: &self.reasoning, status, encrypted: self.context.include_encrypted_reasoning } }).await;
            self.reasoning_done = true;
        })
    }

    fn push_text<'a>(&'a mut self, delta: &'a str) -> ResponseEvents<'a> {
        Box::pin(async_stream::stream! {
        if !self.text_started {
            { let mut events = self.finish_reasoning("completed"); while let Some(event) = events.next().await { yield event; } }
            self.text_started = true;
            let output_index = self.allocate_output_index();
            self.text_output_index = Some(output_index);
            yield self.event("response.output_item.added",
                json!({
                    "output_index": output_index,
                    "item": {
                        "id": self.text_item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress"
                    }
                }),
            );
            yield self.event("response.content_part.added",
                json!({
                    "item_id": self.text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                }),
            );
        }
        if let Err(error) = self.history.try_grow(delta.len()) { yield self.quota_error(error); return; }
        self.text.push_str(delta);
        let output_index = self.text_output_index.expect("text output index assigned");
        yield self.event("response.output_text.delta",
            json!({
                "item_id": self.text_item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": delta,
                "logprobs": []
            }),
        );
        })
    }

    fn complete(&mut self) -> ResponseEvents<'_> {
        Box::pin(async_stream::stream! {
            if self.terminal { return; }
            if !self.context.parallel_tool_calls && self.tool_calls.len() > 1 {
                let mut events = self.fail(stream_error_payload("model emitted multiple function calls while parallel_tool_calls=false", "internal_server_error", None)); while let Some(event) = events.next().await { yield event; } return;
            }
            let status = if self.finish_reason.as_deref() == Some("length") { "incomplete" } else { "completed" };
            let reasoning_status = if status == "incomplete" && !self.text_started && self.tool_calls.is_empty() && self.reasoning_output_index.is_some() { "incomplete" } else { "completed" };
            { let mut events = self.finish_reasoning(reasoning_status); while let Some(event) = events.next().await { yield event; } }
            if !self.text_started && self.tool_calls.is_empty() && self.reasoning_output_index.is_none() {
                let mut events = self.push_text(""); while let Some(event) = events.next().await { yield event; }
            }
            let session = self.history.output_session().clone();
            let phase = if self.tool_calls.is_empty() { AssistantMessagePhase::FinalAnswer } else { AssistantMessagePhase::Commentary };
            let text_status = if status == "incomplete" && self.tool_calls.is_empty() { "incomplete" } else { "completed" };
            if let Some(output_index) = self.text_output_index {
                yield borrowed_event(&session, &mut self.sequence, "response.output_text.done", wire::TextEvent { item_id: &self.text_item_id, output_index, content_index: 0, text: &self.text, logprobs: Some([]) }).await;
                yield borrowed_event(&session, &mut self.sequence, "response.content_part.done", wire::PartEvent { item_id: &self.text_item_id, output_index, content_index: 0, part: wire::TextPart { kind: "output_text", text: &self.text, annotations: [], logprobs: [] } }).await;
                yield borrowed_event(&session, &mut self.sequence, "response.output_item.done", wire::ItemEvent { output_index, item: wire::Item::Message { id: &self.text_item_id, text: &self.text, status: text_status, phase } }).await;
            }
            // Only small wire identities are newly owned. Full text and arguments
            // below borrow the existing history for every terminal event.
            let call_metadata: Vec<_> = self.tool_calls.values().map(|call| (format!("fc_{}", Uuid::new_v4().simple()), if call.id.is_empty() { format!("call_{}", Uuid::new_v4().simple()) } else { call.id.clone() }, self.context.tool_names.response_name(&call.name))).collect();
            let _metadata = match ferrum_output::semantic_cost(&call_metadata).and_then(|n| session.try_reserve_semantic(n)) { Ok(lease) => lease, Err(error) => { yield self.quota_error(error); return; } };
            let call_base = self.next_output_index;
            self.next_output_index += self.tool_calls.len();
            for (position, (call, (item_id, call_id, name))) in self.tool_calls.values().zip(&call_metadata).enumerate() {
                let output_index = call_base + position;
                yield borrowed_event(&session, &mut self.sequence, "response.output_item.added", wire::ItemEvent { output_index, item: wire::Item::Function { id: item_id, call_id, name, arguments: "", status: "in_progress" } }).await;
                if !call.arguments.is_empty() {
                    yield borrowed_event(&session, &mut self.sequence, "response.function_call_arguments.delta", wire::ArgumentsDelta { item_id, output_index, delta: &call.arguments }).await;
                }
                yield borrowed_event(&session, &mut self.sequence, "response.function_call_arguments.done", wire::ArgumentsEvent { item_id, output_index, call_id, name: &name.name, arguments: &call.arguments, namespace: name.namespace.as_deref() }).await;
                let call_status = if status == "incomplete" && position + 1 == self.tool_calls.len() { "incomplete" } else { "completed" };
                yield borrowed_event(&session, &mut self.sequence, "response.output_item.done", wire::ItemEvent { output_index, item: wire::Item::Function { id: item_id, call_id, name, arguments: &call.arguments, status: call_status } }).await;
            }
            let mut items = BTreeMap::new();
            if let Some(index) = self.reasoning_output_index { items.insert(index, wire::Item::Reasoning { id: &self.reasoning_item_id, text: &self.reasoning, status: reasoning_status, encrypted: self.context.include_encrypted_reasoning }); }
            if let Some(index) = self.text_output_index { items.insert(index, wire::Item::Message { id: &self.text_item_id, text: &self.text, status: text_status, phase }); }
            for (position, (call, (id, call_id, name))) in self.tool_calls.values().zip(&call_metadata).enumerate() { items.insert(call_base + position, wire::Item::Function { id, call_id, name, arguments: &call.arguments, status: if status == "incomplete" && position + 1 == self.tool_calls.len() { "incomplete" } else { "completed" } }); }
            let ordered: Vec<_> = items.into_values().collect();
            let response = wire::ResponseView { context: &self.context, status, output: ordered, usage: self.usage.as_ref(), completed_at: Some(chrono::Utc::now().timestamp() as u64) };
            yield borrowed_event(&session, &mut self.sequence, if status == "incomplete" { "response.incomplete" } else { "response.completed" }, wire::ResponseEvent { response }).await;
            yield ResponseEvent::Done;
            self.terminal = true;
        })
    }

    fn fail(&mut self, error: crate::openai::OpenAiError) -> ResponseEvents<'_> {
        Box::pin(async_stream::stream! {
        if self.terminal {
            return;
        }
        self.terminal = true;
        yield ResponseEvent::Failed(error);
        })
    }

    fn event(&mut self, event_type: &'static str, mut event: Value) -> ResponseEvent {
        if let Some(object) = event.as_object_mut() {
            object.insert("type".to_string(), json!(event_type));
            object.insert("sequence_number".to_string(), json!(self.sequence));
        }
        self.sequence += 1;
        match output::Retained::new(self.history.output_session(), event) {
            Ok(value) => ResponseEvent::Json {
                name: event_type,
                value,
            },
            Err(error) => self.quota_error(error),
        }
    }
}

async fn borrowed_event<T: serde::Serialize + Send + Sync>(
    session: &ferrum_output::OutputSession,
    sequence: &mut u64,
    kind: &'static str,
    fields: T,
) -> ResponseEvent {
    let event = wire::Event {
        kind,
        sequence_number: *sequence,
        fields,
    };
    *sequence += 1;
    ResponseEvent::Encoded(output::encode_json(session, Some(kind), &event).await)
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct SseDecoder {
    buffer: Vec<u8>,
}

#[cfg(test)]
impl SseDecoder {
    pub(super) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(bytes);
        let mut data = Vec::new();
        while let Some((start, separator_len)) = find_sse_separator(&self.buffer) {
            let frame = self.buffer.drain(..start).collect::<Vec<_>>();
            self.buffer.drain(..separator_len);
            if let Some(value) = sse_data(&frame) {
                data.push(value);
            }
        }
        data
    }
}

#[cfg(test)]
fn find_sse_separator(bytes: &[u8]) -> Option<(usize, usize)> {
    let lf = bytes.windows(2).position(|window| window == b"\n\n");
    let crlf = bytes.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(left), Some(right)) if left <= right => Some((left, 2)),
        (Some(_), Some(right)) => Some((right, 4)),
        (Some(left), None) => Some((left, 2)),
        (None, Some(right)) => Some((right, 4)),
        (None, None) => None,
    }
}

#[cfg(test)]
fn sse_data(frame: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(frame);
    let lines = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect::<Vec<_>>();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axum_server::responses::{ResponseContext, ToolNameMap};
    use futures::{FutureExt, Stream};
    use serde_json::Map;
    use std::{
        pin::Pin,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        task::Poll,
    };
    use tokio::sync::oneshot;

    struct PendingBody {
        dropped: Option<oneshot::Sender<()>>,
        polled: Arc<AtomicUsize>,
    }

    impl Stream for PendingBody {
        type Item = ChatStreamEvent;

        fn poll_next(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            self.polled.fetch_add(1, Ordering::SeqCst);
            Poll::Pending
        }
    }

    impl Drop for PendingBody {
        fn drop(&mut self) {
            if let Some(dropped) = self.dropped.take() {
                let _ = dropped.send(());
            }
        }
    }

    fn response_context() -> ResponseContext {
        ResponseContext {
            id: "resp_test".to_string(),
            created_at: 0,
            model: "test-model".to_string(),
            instructions: None,
            max_output_tokens: 16,
            temperature: 0.0,
            top_p: 1.0,
            parallel_tool_calls: true,
            tools: Vec::new(),
            tool_choice: json!("auto"),
            metadata: Map::new(),
            text: json!({"format": {"type": "text"}}),
            reasoning: Value::Null,
            prompt_cache_key: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            include_encrypted_reasoning: false,
            tool_names: ToolNameMap::default(),
        }
    }

    #[tokio::test]
    async fn dropping_responses_body_drops_inner_chat_stream() {
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let budget =
            ferrum_output::OutputBudget::new(ferrum_types::OutputConfig::default()).unwrap();
        let inner = ChatEventStream {
            session: budget.try_open_session().unwrap(),
            events: Box::pin(PendingBody {
                dropped: Some(dropped_tx),
                polled: Arc::new(AtomicUsize::new(0)),
            }),
        };
        let outer = adapt_chat_stream(inner, response_context());

        tokio::task::yield_now().await;
        drop(outer);

        tokio::time::timeout(std::time::Duration::from_secs(1), dropped_rx)
            .await
            .expect("inner Chat body was not dropped after Responses disconnect")
            .expect("inner Chat body drop signal sender disappeared");
    }

    #[tokio::test]
    async fn responses_pull_does_not_drain_chat_while_client_stops_reading() {
        let budget =
            ferrum_output::OutputBudget::new(ferrum_types::OutputConfig::default()).unwrap();
        let polled = Arc::new(AtomicUsize::new(0));
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let chat = ChatEventStream {
            session: budget.try_open_session().unwrap(),
            events: Box::pin(PendingBody {
                dropped: Some(dropped_tx),
                polled: polled.clone(),
            }),
        };
        let response = adapt_chat_stream(chat, response_context());
        assert_eq!(polled.load(Ordering::SeqCst), 0);
        let mut body = response.into_body().into_data_stream();
        let first = body.next().await.unwrap().unwrap();
        let second = body.next().await.unwrap().unwrap();
        assert!(std::str::from_utf8(&first)
            .unwrap()
            .contains("event: response.created"));
        assert!(std::str::from_utf8(&second)
            .unwrap()
            .contains("event: response.in_progress"));
        assert_eq!(
            polled.load(Ordering::SeqCst),
            0,
            "initial events must not drain Chat"
        );
        assert!(body.next().now_or_never().is_none());
        assert_eq!(polled.load(Ordering::SeqCst), 1);
        drop(body);
        dropped_rx
            .await
            .expect("body Drop must drop the original Chat stream");
        drop(first);
        drop(second);
        assert_eq!(budget.usage().sessions, 0);
        assert_eq!(budget.usage().encoded_bytes, 0);
    }

    #[tokio::test]
    async fn responses_encoding_limit_emits_failed_with_full_data_pool_and_drops_source() {
        let budget = ferrum_output::OutputBudget::new(ferrum_types::OutputConfig {
            max_sessions: 1,
            encoded_bytes_total: 1,
            max_event_bytes: 1,
            ..Default::default()
        })
        .unwrap();
        let session = budget.try_open_session().unwrap();
        let normal = session.reserve_encoded(1).await.unwrap();
        let polled = Arc::new(AtomicUsize::new(0));
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let chat = ChatEventStream {
            session,
            events: Box::pin(PendingBody {
                dropped: Some(dropped_tx),
                polled: polled.clone(),
            }),
        };
        let mut context = response_context();
        context
            .metadata
            .insert("large".into(), json!("x".repeat(4096)));
        let mut body = adapt_chat_stream(chat, context)
            .into_body()
            .into_data_stream();
        let frame = body
            .next()
            .now_or_never()
            .expect("terminal control cannot wait for normal credit")
            .unwrap()
            .unwrap();
        assert_eq!(polled.load(Ordering::SeqCst), 0);
        dropped_rx
            .now_or_never()
            .expect("upstream drops before another body poll")
            .unwrap();
        let wire = std::str::from_utf8(&frame).unwrap();
        assert!(wire.starts_with("event: response.failed\n"));
        let event: Value = serde_json::from_str(&sse_data(&frame).unwrap()).unwrap();
        assert_eq!(event["sequence_number"], 0);
        assert_eq!(event["response"]["id"], "resp_test");
        assert_eq!(event["response"]["status"], "failed");
        assert_eq!(event["response"]["error"]["code"], "output_limit_exceeded");
        assert!(event["response"].get("metadata").is_none());
        assert!(!wire.contains("[DONE]"));
        assert!(frame.len() <= ferrum_output::TERMINAL_ERROR_BYTES);
        let last = frame.clone();
        assert!(body.next().now_or_never().unwrap().is_none());
        drop(body);
        drop(frame);
        drop(normal);
        assert_eq!(budget.usage().sessions, 1);
        assert_eq!(budget.usage().control_bytes, last.len());
        drop(last);
        assert_eq!(budget.usage().sessions, 0);
        assert_eq!(budget.usage().control_bytes, 0);
    }

    #[tokio::test]
    async fn responses_known_upstream_error_bypasses_full_encoded_pool() {
        let budget = ferrum_output::OutputBudget::new(ferrum_types::OutputConfig {
            encoded_bytes_total: 2048,
            max_event_bytes: 2048,
            ..Default::default()
        })
        .unwrap();
        let session = budget.try_open_session().unwrap();
        let chat = ChatEventStream {
            session: session.clone(),
            events: Box::pin(futures::stream::iter([ChatStreamEvent::Error(
                crate::openai::OpenAiError {
                    error: crate::openai::OpenAiErrorDetail {
                        message: "large diagnostic\n".repeat(1024),
                        error_type: "invalid_request_error".into(),
                        code: Some("invalid_schema".into()),
                        param: Some("response_format".into()),
                    },
                },
            )])),
        };
        let mut body = adapt_chat_stream(chat, response_context())
            .into_body()
            .into_data_stream();
        for expected in ["response.created", "response.in_progress"] {
            let frame = body.next().await.unwrap().unwrap();
            assert!(std::str::from_utf8(&frame).unwrap().contains(expected));
        }
        let held = session.reserve_encoded(2048).await.unwrap();
        drop(session);
        let frame = body
            .next()
            .now_or_never()
            .expect("failure bypasses full data pool")
            .unwrap()
            .unwrap();
        let wire = std::str::from_utf8(&frame).unwrap();
        assert!(wire.starts_with("event: response.failed\n"));
        let first = wire.split("\n\n").next().unwrap();
        let value: Value = serde_json::from_str(
            first
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["sequence_number"], 2);
        assert_eq!(value["response"]["id"], "resp_test");
        assert_eq!(value["response"]["status"], "failed");
        assert_eq!(value["response"]["error"]["code"], "invalid_schema");
        assert_eq!(
            value["response"]["error"]["message"],
            output::OMITTED_UPSTREAM_DIAGNOSTIC
        );
        assert!(value["response"].get("metadata").is_none());
        assert!(wire.ends_with("data: [DONE]\n\n"));
        assert_eq!(wire.matches("[DONE]").count(), 1);
        assert_eq!(budget.usage().encoded_bytes, 2048);
        assert_eq!(budget.usage().control_bytes, frame.len());
        let last = frame.clone();
        assert!(body.next().now_or_never().unwrap().is_none());
        drop(body);
        drop(frame);
        drop(held);
        assert_eq!(budget.usage().sessions, 1);
        drop(last);
        assert_eq!(budget.usage().sessions, 0);
        assert_eq!(budget.usage().control_bytes, 0);
    }

    #[tokio::test]
    async fn responses_encoding_limit_replaces_only_the_unpublished_sequence() {
        use crate::openai::{ChatChoice, ChatCompletionsResponse, ChatMessage, MessageRole};
        let budget = ferrum_output::OutputBudget::new(ferrum_types::OutputConfig {
            encoded_bytes_total: 2048,
            max_event_bytes: 2048,
            ..Default::default()
        })
        .unwrap();
        let session = budget.try_open_session().unwrap();
        let chat = ChatEventStream {
            session: session.clone(),
            events: Box::pin(futures::stream::iter([ChatStreamEvent::Chunk(
                output::Retained::new(
                    &session,
                    ChatCompletionsResponse {
                        id: "chat_test".into(),
                        object: "chat.completion.chunk".into(),
                        created: 0,
                        model: "test-model".into(),
                        usage: None,
                        choices: vec![ChatChoice {
                            index: 0,
                            message: None,
                            finish_reason: None,
                            delta: Some(ChatMessage {
                                role: MessageRole::Assistant,
                                content: "x".repeat(4096),
                                reasoning: None,
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                                function_call: None,
                            }),
                        }],
                    },
                )
                .unwrap(),
            )])),
        };
        let mut body = adapt_chat_stream(chat, response_context())
            .into_body()
            .into_data_stream();
        let mut sequence = 0;
        let mut failed = false;
        while let Some(frame) = body
            .next()
            .now_or_never()
            .expect("released prior frame permits progress")
        {
            let frame = frame.unwrap();
            let event: Value = serde_json::from_str(&sse_data(&frame).unwrap()).unwrap();
            assert_eq!(event["sequence_number"], sequence);
            sequence += 1;
            assert!(!failed, "no success or DONE may follow the failure");
            failed = event["type"] == "response.failed";
            if failed {
                assert_eq!(event["response"]["error"]["code"], "output_limit_exceeded");
            }
        }
        assert!(failed);
        assert!(
            sequence > 2,
            "failure must follow the two valid initial events"
        );
        drop(body);
        assert_eq!(budget.usage().semantic_bytes, 0);
        assert_eq!(budget.usage().encoded_bytes, 0);
        assert_eq!(budget.usage().control_bytes, 0);
        assert_eq!(budget.usage().sessions, 1);
        drop(session);
        assert_eq!(budget.usage().sessions, 0);
    }
}
