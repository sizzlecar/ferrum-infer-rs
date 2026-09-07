//! Validate the observed Responses lifecycle before converting to the shared
//! semantic observation. Keep native output items intact for caller-owned replay.
use super::Chat;
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[derive(Debug)]
pub(crate) struct Responses {
    pub chat: Chat,
    pub response: Value,
}

fn nonempty<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value[field]
        .as_str()
        .filter(|text| !text.is_empty())
        .with_context(|| format!("Responses missing {field}"))
}

fn response_identity<'a>(response: &'a Value, status: &str) -> Result<&'a str> {
    ensure!(response["object"] == "response", "invalid Responses object");
    ensure!(response["status"] == status, "unexpected Responses status");
    ensure!(
        response["error"].is_null() && response["incomplete_details"].is_null(),
        "Responses returned error or incomplete details"
    );
    nonempty(response, "id")
}

fn native_response(response: Value) -> Result<Responses> {
    response_identity(&response, "completed")?;
    let output = response["output"]
        .as_array()
        .context("missing Responses output")?;
    ensure!(!output.is_empty(), "Responses output is empty");
    let mut ids = BTreeSet::new();
    let mut call_ids = BTreeSet::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    for item in output {
        ensure!(
            ids.insert(nonempty(item, "id")?),
            "duplicate Responses item id"
        );
        ensure!(
            item["status"] == "completed",
            "unfinished Responses output item"
        );
        match nonempty(item, "type")? {
            "message" | "reasoning" => {
                let is_reasoning = item["type"] == "reasoning";
                if !is_reasoning {
                    ensure!(
                        item["role"] == "assistant",
                        "non-assistant Responses message"
                    );
                } else {
                    ensure!(
                        item["summary"].as_array().is_some_and(Vec::is_empty),
                        "reasoning probe requires readable reasoning content, not a summary"
                    );
                }
                for part in item["content"]
                    .as_array()
                    .context("missing output content")?
                {
                    ensure!(
                        part["type"]
                            == if is_reasoning {
                                "reasoning_text"
                            } else {
                                "output_text"
                            },
                        "unexpected Responses text channel"
                    );
                    let text = part["text"].as_str().context("invalid Responses text")?;
                    if is_reasoning {
                        reasoning.push_str(text);
                    } else {
                        content.push_str(text);
                    }
                }
            }
            "function_call" => {
                let call_id = nonempty(item, "call_id")?;
                ensure!(call_ids.insert(call_id), "duplicate Responses call id");
                let name = nonempty(item, "name")?;
                let arguments = item["arguments"]
                    .as_str()
                    .context("invalid function arguments")?;
                let mut call = json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}});
                if let Some(namespace) = item.get("namespace") {
                    call["function"]["namespace"] = namespace.clone();
                }
                calls.push(call);
            }
            kind => bail!("unsupported Responses output item {kind}"),
        }
    }
    let finish = if calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    }
    .to_owned();
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    let usage = &response["usage"];
    let chat = Chat {
        message,
        finish,
        usage: json!({"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"], "total_tokens": usage["total_tokens"]}),
    }.validate()?;
    Ok(Responses { chat, response })
}

pub(crate) fn responses_sync(text: &str) -> Result<Responses> {
    native_response(serde_json::from_str(text).context("parse Responses JSON")?)
}

#[derive(Default)]
struct Part {
    added: Value,
    text: String,
    text_done: bool,
    done: Option<Value>,
}

struct Item {
    added: Value,
    parts: Vec<Part>,
    arguments: String,
    arguments_done: bool,
    done: Option<Value>,
}

/// Fields present at creation remain stable, apart from the lifecycle fields
/// explicitly assembled from subsequent events. New terminal metadata is kept.
fn stable_fields(before: &Value, after: &Value, mutable: &[&str]) -> Result<()> {
    for (field, value) in before.as_object().context("expected Responses object")? {
        if !mutable.contains(&field.as_str()) {
            ensure!(
                after.get(field) == Some(value),
                "Responses changed stable field {field}"
            );
        }
    }
    Ok(())
}

#[derive(Default)]
struct Stream {
    created: Option<Value>,
    in_progress: bool,
    sequence: u64,
    items: Vec<Item>,
    terminal: Option<Value>,
    done: bool,
}

impl Stream {
    fn item(&mut self, event: &Value) -> Result<&mut Item> {
        let index = event["output_index"]
            .as_u64()
            .context("missing output index")?;
        let index = usize::try_from(index).context("output index overflow")?;
        let item = self
            .items
            .get_mut(index)
            .context("event before output_item.added")?;
        ensure!(item.done.is_none(), "event after output_item.done");
        if let Some(id) = event.get("item_id") {
            ensure!(id == &item.added["id"], "Responses item id changed");
        } else {
            ensure!(
                event["type"] == "response.output_item.done",
                "missing item id"
            );
        }
        Ok(item)
    }

    fn apply(&mut self, event: Value) -> Result<()> {
        ensure!(
            self.terminal.is_none() && !self.done,
            "event after Responses terminal"
        );
        ensure!(event["error"].is_null(), "Responses SSE error");
        ensure!(
            event["sequence_number"].as_u64() == Some(self.sequence),
            "Responses sequence is missing, repeated or out of order"
        );
        self.sequence += 1;
        let kind = nonempty(&event, "type")?;
        if kind == "response.created" {
            ensure!(
                self.created.is_none() && self.sequence == 1,
                "duplicate or late response.created"
            );
            let response = &event["response"];
            response_identity(response, "in_progress")?;
            ensure!(
                response["output"].as_array().is_some_and(Vec::is_empty),
                "created response already has output"
            );
            self.created = Some(response.clone());
            return Ok(());
        }
        let created = self.created.as_ref().context("missing response.created")?;
        if kind == "response.in_progress" {
            ensure!(
                !self.in_progress && self.items.is_empty(),
                "duplicate or late response.in_progress"
            );
            response_identity(&event["response"], "in_progress")?;
            ensure!(
                &event["response"] == created,
                "in_progress response differs from created"
            );
            self.in_progress = true;
            return Ok(());
        }
        ensure!(self.in_progress, "output before response.in_progress");
        match kind {
            "response.output_item.added" => {
                ensure!(
                    event["output_index"].as_u64() == Some(self.items.len() as u64),
                    "noncontiguous or duplicate output index"
                );
                let added = &event["item"];
                let id = nonempty(added, "id")?;
                ensure!(
                    !self.items.iter().any(|item| item.added["id"] == id),
                    "duplicate output item id"
                );
                ensure!(
                    added["status"] == "in_progress",
                    "added output item is not in progress"
                );
                match nonempty(added, "type")? {
                    "message" | "reasoning" => ensure!(
                        added["content"].as_array().is_some_and(Vec::is_empty),
                        "added text item contains unobserved content"
                    ),
                    "function_call" => {
                        nonempty(added, "call_id")?;
                        nonempty(added, "name")?;
                        ensure!(
                            added["arguments"] == "",
                            "added function contains unobserved arguments"
                        );
                    }
                    other => bail!("unsupported added item {other}"),
                }
                self.items.push(Item {
                    added: added.clone(),
                    parts: Vec::new(),
                    arguments: String::new(),
                    arguments_done: false,
                    done: None,
                });
            }
            "response.content_part.added" => {
                let item = self.item(&event)?;
                ensure!(
                    event["content_index"].as_u64() == Some(item.parts.len() as u64),
                    "noncontiguous or duplicate content index"
                );
                let expected = match item.added["type"].as_str() {
                    Some("message") => "output_text",
                    Some("reasoning") => "reasoning_text",
                    _ => bail!("content part attached to non-text item"),
                };
                ensure!(
                    event["part"]["type"] == expected && event["part"]["text"] == "",
                    "invalid added content part"
                );
                item.parts.push(Part {
                    added: event["part"].clone(),
                    ..Part::default()
                });
            }
            "response.output_text.delta"
            | "response.reasoning_text.delta"
            | "response.output_text.done"
            | "response.reasoning_text.done"
            | "response.content_part.done" => {
                let item = self.item(&event)?;
                let index = usize::try_from(
                    event["content_index"]
                        .as_u64()
                        .context("missing content index")?,
                )?;
                let part = item
                    .parts
                    .get_mut(index)
                    .context("text event before content_part.added")?;
                ensure!(part.done.is_none(), "text event after content_part.done");
                if kind == "response.content_part.done" {
                    ensure!(part.text_done, "content_part.done before text.done");
                    stable_fields(&part.added, &event["part"], &["text"])?;
                    ensure!(
                        event["part"]["text"] == part.text,
                        "content part differs from streamed text"
                    );
                    part.done = Some(event["part"].clone());
                } else {
                    let channel = if kind.starts_with("response.reasoning_text.") {
                        "reasoning_text"
                    } else {
                        "output_text"
                    };
                    ensure!(
                        part.added["type"] == channel,
                        "Responses text channel changed"
                    );
                    ensure!(!part.text_done, "text event after text.done");
                    if kind.ends_with(".delta") {
                        part.text
                            .push_str(event["delta"].as_str().context("invalid text delta")?);
                    } else {
                        ensure!(
                            event["text"] == part.text,
                            "text.done differs from accumulated deltas"
                        );
                        part.text_done = true;
                    }
                }
            }
            "response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
                let item = self.item(&event)?;
                ensure!(
                    item.added["type"] == "function_call",
                    "function arguments on non-function item"
                );
                ensure!(!item.arguments_done, "arguments after arguments.done");
                if kind.ends_with(".delta") {
                    item.arguments
                        .push_str(event["delta"].as_str().context("invalid arguments delta")?);
                } else {
                    ensure!(
                        event["arguments"] == item.arguments,
                        "arguments.done differs from deltas"
                    );
                    for field in ["call_id", "name", "namespace"] {
                        if let Some(value) = event.get(field) {
                            ensure!(
                                item.added.get(field) == Some(value),
                                "function identity changed at arguments.done"
                            );
                        }
                    }
                    item.arguments_done = true;
                }
            }
            "response.output_item.done" => {
                let item = self.item(&event)?;
                let done = &event["item"];
                ensure!(
                    done["status"] == "completed",
                    "output item did not complete"
                );
                stable_fields(&item.added, done, &["status", "content", "arguments"])?;
                if item.added["type"] == "function_call" {
                    ensure!(
                        item.arguments_done && done["arguments"] == item.arguments,
                        "output item arguments differ or did not finish"
                    );
                } else {
                    ensure!(
                        !item.parts.is_empty(),
                        "text item has no observed content parts"
                    );
                    let parts = item
                        .parts
                        .iter()
                        .map(|part| {
                            part.done
                                .clone()
                                .context("output item before content_part.done")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    ensure!(
                        done["content"] == json!(parts),
                        "output item content differs from observed parts"
                    );
                }
                item.done = Some(done.clone());
            }
            "response.completed" => {
                let response = &event["response"];
                response_identity(response, "completed")?;
                stable_fields(
                    created,
                    response,
                    &["status", "output", "usage", "completed_at"],
                )?;
                let output = self
                    .items
                    .iter()
                    .map(|item| {
                        item.done
                            .clone()
                            .context("response completed before output_item.done")
                    })
                    .collect::<Result<Vec<_>>>()?;
                ensure!(
                    !output.is_empty() && response["output"] == json!(output),
                    "completed response differs from streamed output"
                );
                self.terminal = Some(response.clone());
            }
            other => bail!("unexpected or unsuccessful Responses event {other}"),
        }
        Ok(())
    }
}

pub(crate) fn responses_stream(text: &str) -> Result<Responses> {
    let normalized = text.replace("\r\n", "\n");
    let mut state = Stream::default();
    for frame in normalized.split_inclusive("\n\n") {
        if !frame.ends_with("\n\n") {
            ensure!(
                frame
                    .lines()
                    .all(|line| line.is_empty() || line.starts_with(':')),
                "Responses stream ended inside an event"
            );
            continue;
        }
        let mut event_type = None;
        let mut data = Vec::new();
        for line in frame.lines() {
            if let Some(value) = line.strip_prefix("data:") {
                data.push(value.strip_prefix(' ').unwrap_or(value));
            } else if let Some(value) = line.strip_prefix("event:") {
                ensure!(event_type.is_none(), "duplicate SSE event field");
                event_type = Some(value.strip_prefix(' ').unwrap_or(value));
            }
        }
        if data.is_empty() {
            continue;
        }
        ensure!(!state.done, "Responses data after DONE");
        let data = data.join("\n");
        if data == "[DONE]" {
            ensure!(
                state.terminal.is_some() && event_type.is_none(),
                "DONE before response.completed or typed DONE"
            );
            state.done = true;
            continue;
        }
        let event: Value = serde_json::from_str(&data).context("invalid Responses SSE JSON")?;
        if let Some(event_type) = event_type {
            ensure!(
                event["type"] == event_type,
                "SSE event name and payload type disagree"
            );
        }
        state.apply(event)?;
    }
    // response.completed is the Responses terminal. Ferrum additionally sends
    // one [DONE]; accept that transport marker without requiring it from others.
    native_response(
        state
            .terminal
            .context("Responses stream missing response.completed")?,
    )
}

#[cfg(test)]
#[path = "responses_protocol_tests.rs"]
mod tests;
