use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default, Debug, Serialize)]
pub(crate) struct Tool {
    pub call_id: String,
    pub assistant_turn: usize,
    pub origin_response_id: Option<String>,
    pub arguments: Value,
    pub name: String,
    pub requested: bool,
    pub started_ns: Option<u64>,
    pub ended_ns: Option<u64>,
    pub is_error: bool,
    pub returned_to_session: bool,
    pub returned_ns: Option<u64>,
    pub result_text: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Response {
    pub assistant_turn: usize,
    pub response_id: Option<String>,
    pub started_ns: Option<u64>,
    pub ended_ns: u64,
    pub elapsed_ns: Option<u64>,
    pub stop_reason: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Retry {
    pub observed_ns: u64,
    pub event: Value,
    pub next_assistant_start_ns: Option<u64>,
    pub next_assistant_end_ns: Option<u64>,
    pub observed_wait_ns: Option<u64>,
    pub observed_attempt_ns: Option<u64>,
}

#[derive(Default, Debug, Serialize)]
pub(crate) struct Events {
    pub session_id: Option<String>,
    pub assistant_messages: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub tools: BTreeMap<String, Tool>,
    #[serde(skip)]
    current_calls: BTreeMap<String, String>,
    pub settled: bool,
    pub last_stop_reason: Option<String>,
    pub last_assistant_ns: Option<u64>,
    pub model_errors: Vec<String>,
    pub protocol_errors: Vec<String>,
    pub observed_models: BTreeSet<String>,
    pub observed_providers: BTreeSet<String>,
    pub responses: Vec<Response>,
    pub retries: Vec<Retry>,
    #[serde(skip)]
    assistant_started_ns: Option<u64>,
}

impl Events {
    pub fn observe(&mut self, event: &Value, at: u64) {
        match event["type"].as_str() {
            Some("session") => self.session_id = event["id"].as_str().map(str::to_owned),
            Some("agent_start") => self.settled = false,
            Some("agent_settled") => self.settled = true,
            Some("message_start") if event["message"]["role"] == "assistant" => {
                self.assistant_started_ns = Some(at);
                if let Some(retry) = self.retries.last_mut().filter(|r| {
                    r.event["type"] == "auto_retry_start" && r.next_assistant_start_ns.is_none()
                }) {
                    retry.next_assistant_start_ns = Some(at);
                    retry.observed_wait_ns = Some(at.saturating_sub(retry.observed_ns));
                }
            }
            Some("auto_retry_start" | "auto_retry_end") => {
                self.retries.push(Retry {
                    observed_ns: at,
                    event: event.clone(),
                    next_assistant_start_ns: None,
                    next_assistant_end_ns: None,
                    observed_wait_ns: None,
                    observed_attempt_ns: None,
                });
            }
            Some("message_end") => {
                let message = &event["message"];
                if message["role"] == "assistant" {
                    self.assistant_messages += 1;
                    self.current_calls.clear();
                    self.last_assistant_ns = Some(at);
                    self.last_stop_reason = message["stopReason"].as_str().map(str::to_owned);
                    let started_ns = self.assistant_started_ns.take();
                    if let Some(retry) = self.retries.last_mut().filter(|r| {
                        r.event["type"] == "auto_retry_start"
                            && r.next_assistant_start_ns == started_ns
                            && started_ns.is_some()
                            && r.next_assistant_end_ns.is_none()
                    }) {
                        retry.next_assistant_end_ns = Some(at);
                        retry.observed_attempt_ns = Some(at.saturating_sub(retry.observed_ns));
                    }
                    self.responses.push(Response {
                        assistant_turn: self.assistant_messages,
                        response_id: message["responseId"].as_str().map(str::to_owned),
                        started_ns,
                        ended_ns: at,
                        elapsed_ns: started_ns.map(|start| at.saturating_sub(start)),
                        stop_reason: self.last_stop_reason.clone(),
                        error: message["errorMessage"].as_str().map(str::to_owned),
                    });
                    for (key, counter) in [
                        ("input", &mut self.input_tokens),
                        ("output", &mut self.output_tokens),
                        ("cacheRead", &mut self.cache_read_tokens),
                        ("cacheWrite", &mut self.cache_write_tokens),
                    ] {
                        *counter += message["usage"][key].as_u64().unwrap_or(0);
                    }
                    if let Some(model) = message["model"].as_str() {
                        self.observed_models.insert(model.to_owned());
                    }
                    if let Some(provider) = message["provider"].as_str() {
                        self.observed_providers.insert(provider.to_owned());
                    }
                    if matches!(self.last_stop_reason.as_deref(), Some("error" | "aborted")) {
                        self.model_errors.push(
                            message["errorMessage"]
                                .as_str()
                                .unwrap_or("model aborted/error")
                                .to_owned(),
                        );
                    }
                    if let Some(content) = message["content"].as_array() {
                        for call in content.iter().filter(|v| v["type"] == "toolCall") {
                            if let Some(id) = call["id"].as_str() {
                                // Call ids can repeat across assistant turns. Keep
                                // each actual invocation instead of overwriting it.
                                let key = format!("{}:{id}", self.assistant_messages);
                                if self.tools.contains_key(&key) {
                                    self.protocol_errors.push(format!(
                                        "duplicate call id in one assistant message: {id}"
                                    ));
                                }
                                self.current_calls.insert(id.into(), key.clone());
                                let tool = self.tools.entry(key).or_default();
                                tool.call_id = id.into();
                                tool.assistant_turn = self.assistant_messages;
                                tool.origin_response_id =
                                    message["responseId"].as_str().map(str::to_owned);
                                tool.arguments = call["arguments"].clone();
                                tool.requested = true;
                                tool.name = call["name"].as_str().unwrap_or("").into();
                            }
                        }
                    }
                } else if message["role"] == "toolResult" {
                    if let Some(id) = message["toolCallId"].as_str() {
                        if let Some(key) = self.current_calls.get(id) {
                            let tool = self.tools.get_mut(key).expect("known call");
                            if tool.ended_ns.is_none() || tool.returned_ns.is_some() {
                                self.protocol_errors
                                    .push(format!("tool result out of execution order: {id}"));
                            }
                            tool.returned_to_session = true;
                            tool.returned_ns = Some(at);
                            tool.result_text = result_text(message);
                        } else {
                            self.protocol_errors
                                .push(format!("unrequested tool result {id}"));
                        }
                    }
                }
            }
            Some("tool_execution_start" | "tool_execution_end") => {
                let Some(id) = event["toolCallId"].as_str() else {
                    self.protocol_errors
                        .push("tool event has no call id".into());
                    return;
                };
                let Some(key) = self.current_calls.get(id) else {
                    self.protocol_errors
                        .push(format!("unrequested tool execution {id}"));
                    return;
                };
                let tool = self.tools.get_mut(key).expect("known call");
                tool.name = event["toolName"].as_str().unwrap_or("").into();
                if event["type"] == "tool_execution_start" {
                    if tool.started_ns.is_some() {
                        self.protocol_errors
                            .push(format!("tool execution started twice: {id}"));
                    }
                    tool.started_ns = Some(at);
                } else {
                    if tool.started_ns.is_none() || tool.ended_ns.is_some() {
                        self.protocol_errors
                            .push(format!("tool execution end out of order: {id}"));
                    }
                    tool.ended_ns = Some(at);
                    tool.is_error = event["isError"].as_bool().unwrap_or(true);
                }
            }
            _ => {}
        }
    }

    pub fn completed_loop(&self, model: &str) -> bool {
        self.session_id.as_ref().is_some_and(|id| !id.is_empty())
            && self.settled
            && self.last_stop_reason.as_deref() == Some("stop")
            && self.protocol_errors.is_empty()
            && self.observed_models.len() == 1
            && self.observed_models.contains(model)
            && self.observed_providers.len() == 1
            && self.observed_providers.contains("ferrum-local")
            && !self.tools.is_empty()
            && self.tools.values().all(|t| {
                t.requested
                    && t.origin_response_id.is_some()
                    && t.started_ns.is_some()
                    && t.ended_ns.is_some()
                    && t.returned_to_session
                    && t.result_text.is_some()
                    && t.ended_ns < self.last_assistant_ns
            })
    }
}

/// Match pi's official OpenAI conversion, preserving whitespace and block order.
fn result_text(message: &Value) -> Option<String> {
    let content = message["content"].as_array()?;
    let text = content
        .iter()
        .filter(|v| v["type"] == "text")
        .map(|v| v["text"].as_str())
        .collect::<Option<Vec<_>>>()?
        .join("\n");
    Some(if !text.is_empty() {
        text
    } else if content.iter().any(|v| v["type"] == "image") {
        "(see attached image)".into()
    } else {
        "(no tool output)".into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn retry_wait_and_failed_request_time_are_recorded_separately() {
        let mut events = Events::default();
        events.observe(
            &json!({"type":"message_start","message":{"role":"assistant"}}),
            10,
        );
        events.observe(&json!({"type":"message_end","message":{"role":"assistant","stopReason":"error","errorMessage":"503"}}),20);
        events.observe(
            &json!({"type":"auto_retry_start","attempt":1,"maxAttempts":3,"delayMs":2000}),
            30,
        );
        events.observe(
            &json!({"type":"message_start","message":{"role":"assistant"}}),
            40,
        );
        events.observe(
            &json!({"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}),
            50,
        );
        events.observe(
            &json!({"type":"auto_retry_end","attempt":1,"success":true}),
            55,
        );
        assert_eq!(events.responses[0].elapsed_ns, Some(10));
        assert_eq!(events.responses[0].error.as_deref(), Some("503"));
        assert_eq!(events.retries[0].observed_wait_ns, Some(10));
        assert_eq!(events.retries[0].observed_attempt_ns, Some(20));
        assert_eq!(events.retries[1].event["success"], true);
    }
    #[test]
    fn repeated_call_id_in_later_turn_preserves_each_execution() {
        let mut events = Events::default();
        for at in [10, 20] {
            events.observe(
                &json!({"type":"message_end","message":{"role":"assistant","content":[
                {"type":"toolCall","id":"call_0","name":"read"}]}}),
                at,
            );
            events.observe(
                &json!({"type":"tool_execution_start","toolCallId":"call_0","toolName":"read"}),
                at + 1,
            );
            events.observe(&json!({"type":"tool_execution_end","toolCallId":"call_0","toolName":"read","isError":false}), at+2);
            events.observe(&json!({"type":"message_end","message":{"role":"toolResult","toolCallId":"call_0"}}), at+3);
        }
        assert_eq!(events.tools.len(), 2);
        assert_eq!(events.tools["1:call_0"].started_ns, Some(11));
        assert_eq!(events.tools["2:call_0"].started_ns, Some(21));
        assert!(events.tools.values().all(|t| t.returned_to_session));
    }
    #[test]
    fn exit_success_or_printed_tool_arguments_do_not_complete_a_loop() {
        let mut events = Events::default();
        events.observe(&json!({"type":"message_end","message":{"role":"assistant","provider":"ferrum-local",
            "model":"m","stopReason":"stop","content":[{"type":"text","text":"{\"tool\":\"write\"}"}]}}), 1);
        events.observe(&json!({"type":"agent_settled"}), 2);
        assert!(!events.completed_loop("m"));
    }
    #[test]
    fn tool_result_must_return_before_successful_final_assistant() {
        let mut events = Events::default();
        events.observe(&json!({"type":"session","id":"test-session"}), 0);
        events.observe(&json!({"type":"message_end","message":{"role":"assistant","provider":"ferrum-local",
            "model":"m","responseId":"origin-a","stopReason":"toolUse","content":[{"type":"toolCall","id":"a","name":"edit","arguments":{}}]}}), 1);
        events.observe(
            &json!({"type":"tool_execution_start","toolCallId":"a","toolName":"edit"}),
            2,
        );
        events.observe(&json!({"type":"tool_execution_end","toolCallId":"a","toolName":"edit","isError":false}), 3);
        events.observe(
            &json!({"type":"message_end","message":{"role":"toolResult","toolCallId":"a","content":[{"type":"text","text":"done"}]}}),
            4,
        );
        events.observe(
            &json!({"type":"message_end","message":{"role":"assistant","provider":"ferrum-local",
            "model":"m","stopReason":"stop"}}),
            5,
        );
        events.observe(&json!({"type":"agent_settled"}), 6);
        assert!(events.completed_loop("m"));
        events.tools.insert(
            "2:missing".into(),
            Tool {
                requested: true,
                ..Default::default()
            },
        );
        assert!(!events.completed_loop("m"));
        events.tools.remove("2:missing");
        events.last_stop_reason = Some("error".into());
        assert!(!events.completed_loop("m"));
    }
}
