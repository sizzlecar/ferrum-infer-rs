use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

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

pub(super) fn answer(text: &str, expected: &str) -> Result<()> {
    let actual = text
        .trim()
        .trim_matches(|c| matches!(c, '`' | '*' | '"' | '\'' | '.'))
        .trim();
    ensure!(
        actual == expected,
        "expected answer {expected:?}, received {text:?}"
    );
    Ok(())
}

pub(super) fn stop_from_baseline(content: &str, reasoning: &str) -> Result<(String, String)> {
    let boundaries: Vec<_> = content.char_indices().map(|(index, _)| index).collect();
    for start in boundaries.iter().skip((boundaries.len() / 2).max(1)) {
        let sentinel = &content[*start..];
        if content.find(sentinel) == Some(*start) && !reasoning.contains(sentinel) {
            return Ok((content[..*start].to_owned(), sentinel.to_owned()));
        }
    }
    bail!("baseline has no distinct visible stop suffix outside reasoning: {content:?}")
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
    fn stop_suffix_respects_unicode_and_existing_reasoning() {
        let content = "开头 alpha 结束";
        let (prefix, stop) = stop_from_baseline(content, "earlier reasoning").unwrap();
        assert_eq!(prefix + &stop, content);
        assert!(stop_from_baseline(content, content).is_err());
        assert!(stop_from_baseline("", "").is_err());
    }
    #[test]
    fn semantic_check_rejects_wrong_or_missing_answers() {
        answer("**42**", "42").unwrap();
        assert!(answer("41", "42").is_err());
        assert!(answer("", "42").is_err());
    }
}
