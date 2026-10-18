use super::{
    function_call_arguments_done, function_call_item, message_output_item, reasoning_output_item,
    ResponseContext,
};
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

enum ResponseEvent {
    Json { name: &'static str, value: Value },
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
        let mut state = ResponsesStreamState::new(context);
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
    let numbered = matches!(&event, ResponseEvent::Json { .. });
    let encoded = match event {
        ResponseEvent::Json { name, value } => {
            output::encode_json(session, Some(name), &value).await
        }
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

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

impl ResponsesStreamState {
    fn new(context: ResponseContext) -> Self {
        Self {
            context,
            sequence: 0,
            next_output_index: 0,
            reasoning_item_id: format!("rs_{}", Uuid::new_v4().simple()),
            reasoning: String::new(),
            reasoning_output_index: None,
            reasoning_done: false,
            text_item_id: format!("msg_{}", Uuid::new_v4().simple()),
            text: String::new(),
            text_started: false,
            text_output_index: None,
            tool_calls: BTreeMap::new(),
            finish_reason: None,
            usage: None,
            terminal: false,
        }
    }

    fn send_initial_events(&mut self) -> ResponseEvents<'_> {
        Box::pin(async_stream::stream! {
        let response = self.context.response("in_progress", vec![], None, None);
        yield self.event("response.created", json!({"response": response}));
        let response = self.context.response("in_progress", vec![], None, None);
        yield self.event("response.in_progress", json!({"response": response}));
        })
    }

    fn consume_chat_event(&mut self, event: ChatStreamEvent) -> ResponseEvents<'_> {
        Box::pin(async_stream::stream! {
        if self.terminal { return; }
        let chunk = match event {
            ChatStreamEvent::Done => {
                let mut events = self.complete();
                while let Some(event) = events.next().await { yield event; }
                return;
            }
            ChatStreamEvent::Error(error) => {
                let mut events = self.fail(error);
                while let Some(event) = events.next().await { yield event; }
                return;
            }
            ChatStreamEvent::Chunk(chunk) => chunk,
        };
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        for choice in chunk.choices {
            if let Some(delta) = choice.delta {
                if let Some(reasoning) = delta.reasoning.filter(|value| !value.is_empty()) {
                    { let mut events = self.push_reasoning(&reasoning); while let Some(event) = events.next().await { yield event; } }
                }
                if !delta.content.is_empty() {
                    { let mut events = self.push_text(&delta.content); while let Some(event) = events.next().await { yield event; } }
                }
                let tool_calls = delta.tool_calls.unwrap_or_default();
                if !tool_calls.is_empty() {
                    { let mut events = self.finish_reasoning("completed"); while let Some(event) = events.next().await { yield event; } }
                }
                for (position, call) in tool_calls.into_iter().enumerate() {
                    let index = call.index.unwrap_or(position as u32);
                    let accumulated = self.tool_calls.entry(index).or_default();
                    if !call.id.is_empty() {
                        accumulated.id = call.id;
                    }
                    if !call.function.name.is_empty() {
                        accumulated.name = call.function.name;
                    }
                    accumulated.arguments.push_str(&call.function.arguments);
                }
                if let Some(function) = delta.function_call {
                    { let mut events = self.finish_reasoning("completed"); while let Some(event) = events.next().await { yield event; } }
                    let accumulated = self.tool_calls.entry(0).or_default();
                    accumulated.id = accumulated
                        .id
                        .is_empty()
                        .then(|| format!("call_{}", Uuid::new_v4().simple()))
                        .unwrap_or_else(|| accumulated.id.clone());
                    if !function.name.is_empty() {
                        accumulated.name = function.name;
                    }
                    accumulated.arguments.push_str(&function.arguments);
                }
            }
            if choice.finish_reason.is_some() {
                self.finish_reason = choice.finish_reason;
            }
        }
        })
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
        let Some(output_index) = self.reasoning_output_index else {
            return;
        };
        if self.reasoning_done {
            return;
        }
        yield self.event("response.reasoning_text.done",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": self.reasoning
            }),
        );
        yield self.event("response.content_part.done",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "reasoning_text", "text": self.reasoning}
            }),
        );
        let item = reasoning_output_item(
            self.reasoning_item_id.clone(),
            self.reasoning.clone(),
            status,
            self.context.include_encrypted_reasoning,
        );
        yield self.event("response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        );
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
        if self.terminal {
            return;
        }
        if !self.context.parallel_tool_calls && self.tool_calls.len() > 1 {
            { let mut events = self.fail(stream_error_payload("model emitted multiple function calls while parallel_tool_calls=false", "internal_server_error", None)); while let Some(event) = events.next().await { yield event; } }
            return;
        }
        let status = if self.finish_reason.as_deref() == Some("length") {
            "incomplete"
        } else {
            "completed"
        };
        let reasoning_status = if status == "incomplete"
            && !self.text_started
            && self.tool_calls.is_empty()
            && self.reasoning_output_index.is_some()
        {
            "incomplete"
        } else {
            "completed"
        };
        { let mut events = self.finish_reasoning(reasoning_status); while let Some(event) = events.next().await { yield event; } }
        let mut output = BTreeMap::new();
        if self.text_started
            || (self.tool_calls.is_empty() && self.reasoning_output_index.is_none())
        {
            if !self.text_started {
                { let mut events = self.push_text(""); while let Some(event) = events.next().await { yield event; } }
            }
            let output_index = self.text_output_index.expect("text output index assigned");
            yield self.event("response.output_text.done",
                json!({
                    "item_id": self.text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": self.text,
                    "logprobs": []
                }),
            );
            let text_status = if status == "incomplete" && self.tool_calls.is_empty() {
                "incomplete"
            } else {
                "completed"
            };
            let phase = if self.tool_calls.is_empty() {
                AssistantMessagePhase::FinalAnswer
            } else {
                AssistantMessagePhase::Commentary
            };
            let item = message_output_item(
                self.text_item_id.clone(),
                self.text.clone(),
                text_status,
                phase,
            );
            yield self.event("response.content_part.done",
                json!({
                    "item_id": self.text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": item["content"][0].clone()
                }),
            );
            yield self.event("response.output_item.done",
                json!({"output_index": output_index, "item": item.clone()}),
            );
            output.insert(output_index, item);
        }
        if let Some(output_index) = self.reasoning_output_index {
            output.insert(
                output_index,
                reasoning_output_item(
                    self.reasoning_item_id.clone(),
                    self.reasoning.clone(),
                    reasoning_status,
                    self.context.include_encrypted_reasoning,
                ),
            );
        }

        let calls = std::mem::take(&mut self.tool_calls);
        let call_count = calls.len();
        for (position, (_, call)) in calls.into_iter().enumerate() {
            let output_index = self.allocate_output_index();
            let call_id = if call.id.is_empty() {
                format!("call_{}", Uuid::new_v4().simple())
            } else {
                call.id
            };
            let item_id = format!("fc_{}", Uuid::new_v4().simple());
            let response_name = self.context.tool_names.response_name(&call.name);
            let in_progress =
                function_call_item(item_id.clone(), "in_progress", &call_id, &response_name, "");
            yield self.event("response.output_item.added",
                json!({"output_index": output_index, "item": in_progress}),
            );
            if !call.arguments.is_empty() {
                yield self.event("response.function_call_arguments.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": call.arguments
                    }),
                );
            }
            let arguments_done = function_call_arguments_done(
                &item_id,
                output_index,
                &call_id,
                &response_name,
                &call.arguments,
            );
            yield self.event("response.function_call_arguments.done", arguments_done);
            let call_status = if status == "incomplete" && position + 1 == call_count {
                "incomplete"
            } else {
                "completed"
            };
            let item = function_call_item(
                item_id,
                call_status,
                &call_id,
                &response_name,
                &call.arguments,
            );
            yield self.event("response.output_item.done",
                json!({"output_index": output_index, "item": item.clone()}),
            );
            output.insert(output_index, item);
        }

        let response = self.context.response(
            status,
            output.into_values().collect(),
            self.usage.as_ref(),
            None,
        );
        let terminal_event = if status == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        yield self.event(terminal_event, json!({"response": response}));
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
        ResponseEvent::Json {
            name: event_type,
            value: event,
        }
    }
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
        let chat = ChatEventStream {
            session: budget.try_open_session().unwrap(),
            events: Box::pin(futures::stream::iter([ChatStreamEvent::Chunk(
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
        assert_eq!(budget.usage().sessions, 0);
    }
}
