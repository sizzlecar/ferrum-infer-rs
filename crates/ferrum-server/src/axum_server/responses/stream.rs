use super::{
    function_call_arguments_done, function_call_item, message_output_item, reasoning_output_item,
    ResponseContext,
};
use crate::openai::{ChatCompletionsResponse, Usage};
use axum::response::{sse::Event, IntoResponse, Response, Sse};
use futures::StreamExt;
use serde_json::{json, Value};
use std::{collections::BTreeMap, convert::Infallible};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

type EventSender = mpsc::UnboundedSender<std::result::Result<Event, Infallible>>;

pub(super) fn adapt_chat_stream(response: Response, context: ResponseContext) -> Response {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut state = ResponsesStreamState::new(context);
        state.send_initial_events(&tx);
        if state.terminal {
            return;
        }
        let mut decoder = SseDecoder::default();
        let mut body = response.into_body().into_data_stream();
        loop {
            let chunk = tokio::select! {
                biased;
                _ = tx.closed() => return,
                chunk = body.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            match chunk {
                Ok(bytes) => {
                    for data in decoder.push(&bytes) {
                        state.consume_chat_event(&data, &tx);
                        if state.terminal {
                            return;
                        }
                    }
                }
                Err(error) => {
                    state.fail(format!("failed to read chat stream: {error}"), &tx);
                    return;
                }
            }
        }
        for data in decoder.finish() {
            state.consume_chat_event(&data, &tx);
        }
        if !state.terminal {
            state.fail("chat stream ended before [DONE]".to_string(), &tx);
        }
    });
    Sse::new(UnboundedReceiverStream::new(rx)).into_response()
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

    fn send_initial_events(&mut self, tx: &EventSender) {
        let response = self.context.response("in_progress", vec![], None, None);
        self.send(tx, "response.created", json!({"response": response}));
        let response = self.context.response("in_progress", vec![], None, None);
        self.send(tx, "response.in_progress", json!({"response": response}));
    }

    fn consume_chat_event(&mut self, data: &str, tx: &EventSender) {
        if self.terminal {
            return;
        }
        if data == "[DONE]" {
            self.complete(tx);
            return;
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(error) => {
                self.fail(format!("invalid chat stream event: {error}"), tx);
                return;
            }
        };
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("chat stream failed")
                .to_string();
            self.fail(message, tx);
            return;
        }
        let chunk: ChatCompletionsResponse = match serde_json::from_value(value) {
            Ok(chunk) => chunk,
            Err(error) => {
                self.fail(format!("invalid chat completion chunk: {error}"), tx);
                return;
            }
        };
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        for choice in chunk.choices {
            if let Some(delta) = choice.delta {
                if let Some(reasoning) = delta.reasoning.filter(|value| !value.is_empty()) {
                    self.push_reasoning(&reasoning, tx);
                }
                if !delta.content.is_empty() {
                    self.push_text(&delta.content, tx);
                }
                let tool_calls = delta.tool_calls.unwrap_or_default();
                if !tool_calls.is_empty() {
                    self.finish_reasoning("completed", tx);
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
                    self.finish_reasoning("completed", tx);
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
    }

    fn allocate_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn push_reasoning(&mut self, delta: &str, tx: &EventSender) {
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
            self.send(
                tx,
                "response.output_item.added",
                json!({"output_index": output_index, "item": item}),
            );
            self.send(
                tx,
                "response.content_part.added",
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
        self.send(
            tx,
            "response.reasoning_text.delta",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": delta
            }),
        );
    }

    fn finish_reasoning(&mut self, status: &str, tx: &EventSender) {
        let Some(output_index) = self.reasoning_output_index else {
            return;
        };
        if self.reasoning_done {
            return;
        }
        self.send(
            tx,
            "response.reasoning_text.done",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": self.reasoning
            }),
        );
        self.send(
            tx,
            "response.content_part.done",
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
        self.send(
            tx,
            "response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        );
        self.reasoning_done = true;
    }

    fn push_text(&mut self, delta: &str, tx: &EventSender) {
        if !self.text_started {
            self.finish_reasoning("completed", tx);
            self.text_started = true;
            let output_index = self.allocate_output_index();
            self.text_output_index = Some(output_index);
            self.send(
                tx,
                "response.output_item.added",
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
            self.send(
                tx,
                "response.content_part.added",
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
        self.send(
            tx,
            "response.output_text.delta",
            json!({
                "item_id": self.text_item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": delta,
                "logprobs": []
            }),
        );
    }

    fn complete(&mut self, tx: &EventSender) {
        if self.terminal {
            return;
        }
        if !self.context.parallel_tool_calls && self.tool_calls.len() > 1 {
            self.fail(
                "model emitted multiple function calls while parallel_tool_calls=false".to_string(),
                tx,
            );
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
        self.finish_reasoning(reasoning_status, tx);
        let mut output = BTreeMap::new();
        if self.text_started
            || (self.tool_calls.is_empty() && self.reasoning_output_index.is_none())
        {
            if !self.text_started {
                self.push_text("", tx);
            }
            let output_index = self.text_output_index.expect("text output index assigned");
            self.send(
                tx,
                "response.output_text.done",
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
            let item =
                message_output_item(self.text_item_id.clone(), self.text.clone(), text_status);
            self.send(
                tx,
                "response.content_part.done",
                json!({
                    "item_id": self.text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": item["content"][0].clone()
                }),
            );
            self.send(
                tx,
                "response.output_item.done",
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
            self.send(
                tx,
                "response.output_item.added",
                json!({"output_index": output_index, "item": in_progress}),
            );
            if !call.arguments.is_empty() {
                self.send(
                    tx,
                    "response.function_call_arguments.delta",
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
            self.send(tx, "response.function_call_arguments.done", arguments_done);
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
            self.send(
                tx,
                "response.output_item.done",
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
        self.send(tx, terminal_event, json!({"response": response}));
        self.send_done_marker(tx);
        self.terminal = true;
    }

    fn fail(&mut self, message: String, tx: &EventSender) {
        if self.terminal {
            return;
        }
        let response = self.context.response(
            "failed",
            vec![],
            self.usage.as_ref(),
            Some(json!({"code": "internal_server_error", "message": message})),
        );
        self.send(tx, "response.failed", json!({"response": response}));
        self.send_done_marker(tx);
        self.terminal = true;
    }

    fn send(&mut self, tx: &EventSender, event_type: &str, mut event: Value) {
        if self.terminal {
            return;
        }
        if let Some(object) = event.as_object_mut() {
            object.insert("type".to_string(), json!(event_type));
            object.insert("sequence_number".to_string(), json!(self.sequence));
        }
        self.sequence += 1;
        if tx
            .send(Ok(Event::default()
                .event(event_type)
                .data(event.to_string())))
            .is_err()
        {
            self.terminal = true;
        }
    }

    fn send_done_marker(&self, tx: &EventSender) {
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    }
}

#[derive(Default)]
pub(super) struct SseDecoder {
    buffer: Vec<u8>,
}

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

    fn finish(&mut self) -> Vec<String> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let frame = std::mem::take(&mut self.buffer);
        sse_data(&frame).into_iter().collect()
    }
}

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
    use axum::body::{Body, Bytes};
    use futures::Stream;
    use serde_json::Map;
    use std::{pin::Pin, task::Poll};
    use tokio::sync::oneshot;

    struct PendingBody {
        dropped: Option<oneshot::Sender<()>>,
    }

    impl Stream for PendingBody {
        type Item = std::result::Result<Bytes, Infallible>;

        fn poll_next(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> Poll<Option<Self::Item>> {
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
        let inner = Response::new(Body::from_stream(PendingBody {
            dropped: Some(dropped_tx),
        }));
        let outer = adapt_chat_stream(inner, response_context());

        tokio::task::yield_now().await;
        drop(outer);

        tokio::time::timeout(std::time::Duration::from_secs(1), dropped_rx)
            .await
            .expect("inner Chat body was not dropped after Responses disconnect")
            .expect("inner Chat body drop signal sender disappeared");
    }
}
