use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[path = "responses_protocol.rs"]
mod responses;
pub(super) use responses::{responses_stream, responses_sync, Responses};

#[derive(Debug)]
pub(super) struct Chat {
    pub message: Value,
    pub finish: String,
    pub usage: Value,
}

impl Chat {
    pub fn content(&self) -> &str {
        self.message["content"].as_str().unwrap_or("")
    }

    fn validate(self) -> Result<Self> {
        ensure!(
            matches!(self.finish.as_str(), "stop" | "length" | "tool_calls"),
            "invalid finish reason: {}",
            self.finish
        );
        ensure!(
            self.message["role"] == "assistant",
            "missing assistant role"
        );
        ensure!(
            self.message["content"].is_null() || self.message["content"].is_string(),
            "invalid content type"
        );
        ensure!(
            self.message.get("reasoning_content").is_none(),
            "response emitted noncanonical reasoning_content"
        );
        ensure!(
            self.message["reasoning"].is_null() || self.message["reasoning"].is_string(),
            "invalid reasoning type"
        );
        ensure!(
            self.message["tool_calls"].is_null() || self.message["tool_calls"].is_array(),
            "invalid tool_calls type"
        );
        let calls = self.message["tool_calls"].as_array();
        ensure!(
            (self.finish == "tool_calls") == calls.is_some_and(|calls| !calls.is_empty()),
            "tool calls and finish reason disagree"
        );
        if let Some(calls) = calls {
            for call in calls {
                ensure!(
                    call["type"] == "function"
                        && call["id"].as_str().is_some_and(|s| !s.is_empty()),
                    "invalid tool call identity"
                );
                ensure!(
                    call["function"]["name"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty())
                        && call["function"]["arguments"].is_string(),
                    "invalid tool call function"
                );
            }
        }
        validate_usage(&self.usage)?;
        Ok(self)
    }
}

fn validate_usage(usage: &Value) -> Result<()> {
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .context("missing prompt usage")?;
    let completion = usage["completion_tokens"]
        .as_u64()
        .context("missing completion usage")?;
    ensure!(prompt > 0 && completion > 0, "zero token usage: {usage}");
    ensure!(
        prompt.checked_add(completion) == usage["total_tokens"].as_u64(),
        "usage total mismatch: {usage}"
    );
    Ok(())
}

/// Machine-readable run output must remain JSONL during cold model downloads as
/// well as generation. Keep this parser on the actual process-output path.
pub(super) fn run_records(text: &str) -> Result<Vec<Value>> {
    let records: Vec<Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()
        .context("parse run JSONL")?;
    for event in ["ready", "exit"] {
        ensure!(
            records
                .iter()
                .filter(|record| record["event"] == event)
                .count()
                == 1,
            "run must emit one {event} event"
        );
    }
    ensure!(
        !records.iter().any(|record| record["event"] == "error"),
        "run emitted an error event"
    );
    Ok(records)
}

pub(super) fn sync(text: &str) -> Result<Chat> {
    let body: Value = serde_json::from_str(text).context("parse chat JSON")?;
    ensure!(body.get("error").is_none(), "chat error: {body}");
    let choices = body["choices"].as_array().context("missing choices")?;
    ensure!(
        choices.len() == 1 && choices[0]["index"] == 0,
        "expected the requested single choice"
    );
    Chat {
        message: choices[0]["message"].clone(),
        finish: choices[0]["finish_reason"]
            .as_str()
            .context("missing finish reason")?
            .into(),
        usage: body["usage"].clone(),
    }
    .validate()
}

/// Parse complete SSE framing, including fragmented tool arguments and
/// tokenless text tails. A successful stream has one terminal choice, one
/// requested usage event, and one DONE, in that order.
pub(super) fn stream(text: &str) -> Result<Chat> {
    let text = text.replace("\r\n", "\n");
    let mut done = false;
    let mut finish = None;
    let mut usage = None;
    let mut response_id = None;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut saw_reasoning = false;
    let mut calls = BTreeMap::<u64, Value>::new();
    for event in text.split_inclusive("\n\n") {
        if !event.ends_with("\n\n") {
            ensure!(
                !event.lines().any(|line| line.starts_with("data:")),
                "SSE ended inside a data event"
            );
            continue;
        }
        let data = event
            .lines()
            .filter_map(|line| {
                line.strip_prefix("data:")
                    .map(|s| s.strip_prefix(' ').unwrap_or(s))
            })
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() {
            continue;
        }
        ensure!(!done, "SSE data after DONE or duplicate DONE");
        if data == "[DONE]" {
            ensure!(
                finish.is_some() && usage.is_some(),
                "DONE before terminal choice or usage"
            );
            done = true;
            continue;
        }
        let chunk: Value = serde_json::from_str(&data).context("invalid SSE JSON")?;
        ensure!(chunk.get("error").is_none(), "SSE error: {chunk}");
        let id = chunk["id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .context("SSE missing response id")?;
        if let Some(previous) = &response_id {
            ensure!(previous == id, "SSE response id changed");
        } else {
            response_id = Some(id.to_owned());
        }
        if !chunk["usage"].is_null() {
            ensure!(
                usage.is_none() && finish.is_some(),
                "duplicate usage or usage before terminal"
            );
            ensure!(
                chunk["choices"].as_array().is_some_and(Vec::is_empty),
                "usage event must have empty choices"
            );
            usage = Some(chunk["usage"].clone());
            continue;
        }
        ensure!(
            finish.is_none(),
            "choice after terminal or duplicate terminal"
        );
        let choices = chunk["choices"].as_array().context("SSE missing choices")?;
        ensure!(
            choices.len() == 1 && choices[0]["index"] == 0,
            "SSE expected one choice at index zero"
        );
        let choice = &choices[0];
        let delta = &choice["delta"];
        ensure!(delta.is_object(), "SSE missing delta object");
        ensure!(
            delta["role"].is_null() || delta["role"] == "assistant",
            "invalid SSE role"
        );
        ensure!(
            delta.get("reasoning_content").is_none(),
            "SSE emitted noncanonical reasoning_content"
        );
        for (field, output) in [("content", &mut content), ("reasoning", &mut reasoning)] {
            if !delta[field].is_null() {
                output.push_str(delta[field].as_str().context("non-string SSE text delta")?);
                saw_reasoning |= field == "reasoning";
            }
        }
        if let Some(deltas) = delta.get("tool_calls") {
            for call in deltas
                .as_array()
                .context("tool_calls delta must be an array")?
            {
                let index = call["index"]
                    .as_u64()
                    .context("tool call delta missing index")?;
                let assembled = calls.entry(index).or_insert_with(
                    || json!({"id": "", "type": "", "function": {"name": "", "arguments": ""}}),
                );
                for field in ["id", "type"] {
                    if let Some(value) = call.get(field) {
                        let value = value.as_str().context("invalid tool identity delta")?;
                        ensure!(
                            assembled[field] == "" || assembled[field] == value,
                            "tool identity changed"
                        );
                        assembled[field] = json!(value);
                    }
                }
                for field in ["name", "arguments"] {
                    if let Some(value) = call["function"].get(field) {
                        let fragment = value.as_str().context("invalid tool function delta")?;
                        let mut combined = assembled["function"][field]
                            .as_str()
                            .unwrap_or("")
                            .to_owned();
                        combined.push_str(fragment);
                        assembled["function"][field] = json!(combined);
                    }
                }
            }
        }
        if !choice["finish_reason"].is_null() {
            finish = Some(
                choice["finish_reason"]
                    .as_str()
                    .context("invalid terminal reason")?
                    .to_owned(),
            );
        }
    }
    ensure!(done, "SSE missing DONE");
    let mut message = json!({"role": "assistant", "content": content});
    if saw_reasoning {
        message["reasoning"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls.into_values().collect::<Vec<_>>());
    }
    Chat {
        message,
        finish: finish.context("missing terminal choice")?,
        usage: usage.context("missing usage")?,
    }
    .validate()
}

/// The same fixture oracle runs after both actual HTTP parsers. A usable
/// handoff includes a terminal tool-call finish, an identity, and typed arguments;
/// a merely plausible function name is not enough to execute the fixture.
pub(super) fn calc_call(
    chat: &Chat,
    max_tokens: u32,
) -> Result<ferrum_bench_core::release_regression::model_tool::ValidatedCalcCall> {
    ferrum_bench_core::release_regression::model_tool::verify_calc_call(
        &json!({"message": chat.message, "finish_reason": chat.finish, "usage": chat.usage}),
        max_tokens,
    )
    .map_err(anyhow::Error::msg)
}

pub(super) fn answer(text: &str, expected: &str) -> Result<()> {
    ensure!(
        ferrum_bench_core::release_regression::model_tasks::probe_answer_matches(text, expected),
        "expected answer {expected:?}, received {text:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(choice: Value) -> String {
        format!("data: {}\n\n", json!({"id": "chat-1", "choices": [choice]}))
    }
    fn ending() -> String {
        format!(
            "{}data: {}\n\ndata: [DONE]\n\n",
            event(json!({"index": 0, "delta": {}, "finish_reason": "stop"})),
            json!({"id": "chat-1", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}})
        )
    }
    #[test]
    fn tokenless_unicode_tail_is_part_of_the_answer() {
        let text = event(json!({"index": 0, "delta": {"content": "hello "}}))
            + &event(json!({"index": 0, "delta": {"content": "尾"}}))
            + &ending();
        assert_eq!(
            stream(&text.replace('\n', "\r\n")).unwrap().content(),
            "hello 尾"
        );
    }
    #[test]
    fn incomplete_duplicate_and_error_streams_fail() {
        for text in [
            "data: [DONE]\n\n".to_owned(),
            ending() + "data: [DONE]\n\n",
            ending().replace("data: [DONE]\n\n", ""),
            "data: {\"error\":{\"message\":\"capacity exceeded\"}}\n\n".to_owned() + &ending(),
            ending().replace("\"completion_tokens\":2", "\"completion_tokens\":0"),
        ] {
            assert!(stream(&text).is_err(), "accepted {text}");
        }
    }
    #[test]
    fn final_sse_data_event_requires_its_ending_blank_line() {
        let complete = ending();
        for truncated in [complete.strip_suffix('\n').unwrap(), complete.trim_end()] {
            assert!(stream(truncated).is_err(), "accepted {truncated}");
        }
        stream(&complete).unwrap();
        stream(&(complete + ": trailing comment")).unwrap();
    }
    #[test]
    fn fragmented_tool_arguments_reassemble_before_validation() {
        let text = event(
            json!({"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call-1", "type": "function", "function": {"name": "calc", "arguments": "{\"expression\":"}}]}}),
        ) + &event(
            json!({"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"123+456\"}"}}]}}),
        ) + &ending().replace("\"stop\"", "\"tool_calls\"");
        let result = stream(&text).unwrap();
        assert_eq!(
            result.message["tool_calls"][0]["function"]["arguments"],
            "{\"expression\":\"123+456\"}"
        );
    }
    #[test]
    fn semantic_check_rejects_wrong_or_missing_answers() {
        answer("**42**", "42").unwrap();
        assert!(answer("41", "42").is_err());
        assert!(answer("", "42").is_err());
    }
    #[test]
    fn run_framing_rejects_download_noise_and_incomplete_or_failed_processes() {
        let valid = "{\"event\":\"ready\"}\n{\"event\":\"assistant\",\"content\":\"42\"}\n{\"event\":\"exit\"}\n";
        assert_eq!(run_records(valid).unwrap().len(), 3);
        for text in [
            format!("Downloading 25%\n{valid}"),
            valid.replace("{\"event\":\"exit\"}\n", ""),
            format!("{valid}{{\"event\":\"exit\"}}\n"),
            format!("{valid}{{\"event\":\"error\",\"message\":\"failed\"}}\n"),
            format!("{valid}{{\"event\":"),
        ] {
            assert!(run_records(&text).is_err(), "accepted {text}");
        }
    }

    fn tool_response(call: &Value, finish: &str, streamed: bool) -> String {
        if streamed {
            let mut delta = call.clone();
            delta["index"] = json!(0);
            event(json!({"index": 0, "delta": {"tool_calls": [delta]}}))
                + &ending().replace("\"stop\"", &format!("\"{finish}\""))
        } else {
            json!({
                "choices": [{"index": 0, "message": {"role": "assistant", "tool_calls": [call]}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            }).to_string()
        }
    }

    #[test]
    fn both_http_modes_share_named_tool_identity_arguments_and_finish_oracle() {
        let valid = json!({"id": "call-1", "type": "function", "function": {"name": "calc", "arguments": "{\"expression\":\"123 + 456\"}"}});
        for streamed in [false, true] {
            let parse = |call: &Value, finish: &str| {
                let text = tool_response(call, finish, streamed);
                if streamed { stream(&text) } else { sync(&text) }
                    .and_then(|chat| calc_call(&chat, 512).map(|_| ()))
            };
            parse(&valid, "tool_calls").unwrap();
            for (pointer, wrong) in [
                ("/id", json!("")),
                ("/type", json!("not-a-function")),
                ("/function/name", json!("lookup_weather")),
                ("/function/arguments", json!("invalid JSON")),
                ("/function/arguments", json!("{\"expression\":\"123-456\"}")),
                ("/function/arguments", json!("{\"expression\":579}")),
                (
                    "/function/arguments",
                    json!("{\"expression\":\"123+456\",\"extra\":true}"),
                ),
            ] {
                let mut wrong_call = valid.clone();
                *wrong_call.pointer_mut(pointer).unwrap() = wrong;
                assert!(
                    parse(&wrong_call, "tool_calls").is_err(),
                    "accepted {wrong_call} in stream={streamed}"
                );
            }
            for finish in ["stop", "length", "unknown"] {
                assert!(
                    parse(&valid, finish).is_err(),
                    "accepted finish {finish} in stream={streamed}"
                );
            }
        }
    }

    #[test]
    fn sync_framing_rejects_noncanonical_fields_choices_and_usage() {
        let valid = json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        });
        sync(&valid.to_string()).unwrap();
        for (pointer, wrong) in [
            ("/choices", json!([])),
            ("/choices/0/index", json!(1)),
            ("/choices/0/message/role", json!("user")),
            ("/choices/0/message/content", json!(42)),
            ("/choices/0/finish_reason", Value::Null),
            ("/usage/total_tokens", json!(7)),
        ] {
            let mut body = valid.clone();
            *body.pointer_mut(pointer).unwrap() = wrong;
            assert!(sync(&body.to_string()).is_err(), "accepted {body}");
        }
        for (field, wrong) in [
            ("reasoning_content", json!("thought")),
            ("tool_calls", json!({"id": "malformed"})),
        ] {
            let mut body = valid.clone();
            body["choices"][0]["message"][field] = wrong;
            assert!(sync(&body.to_string()).is_err(), "accepted {body}");
        }
    }
}
