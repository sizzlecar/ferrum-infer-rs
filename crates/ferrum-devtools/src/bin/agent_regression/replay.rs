//! Bind each invocation to its originating response and the appended wire turn.
//! Bare call ids are insufficient: local servers may reuse call_0 every turn.
use super::{
    events::{Events, Tool},
    proxy::RequestRecord,
};
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[path = "replay/compaction.rs"]
mod compaction;
pub(crate) use compaction::verify_with_session;

#[derive(Debug, Serialize)]
pub(crate) struct Receipt {
    pub tool_key: String,
    pub assistant_turn: usize,
    pub origin_response_id: String,
    pub origin_request_index: u32,
    pub following_request_index: u32,
    pub assistant_message_index: usize,
    pub tool_message_index: usize,
    pub result_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<compaction::Receipt>,
}

#[derive(Default, Debug, Serialize)]
pub(crate) struct Evidence {
    pub verified: Vec<Receipt>,
    pub unproven: BTreeMap<String, String>,
}
impl Evidence {
    pub fn complete(&self, events: &Events) -> bool {
        !events.tools.is_empty()
            && self.unproven.is_empty()
            && self.verified.len() == events.tools.len()
    }
}

pub(crate) fn verify(events: &Events, requests: &[&RequestRecord]) -> Evidence {
    let mut evidence = Evidence::default();
    for (key, tool) in &events.tools {
        match witness(key, tool, requests) {
            Ok(receipt) => evidence.verified.push(receipt),
            Err(error) => {
                evidence.unproven.insert(key.clone(), format!("{error:#}"));
            }
        }
    }
    evidence
}

fn witness(key: &str, tool: &Tool, requests: &[&RequestRecord]) -> Result<Receipt> {
    anyhow::ensure!(
        tool.requested
            && tool.started_ns.is_some()
            && tool.ended_ns.is_some()
            && tool.returned_to_session,
        "invocation did not execute and return"
    );
    let result = tool
        .result_text
        .as_ref()
        .context("missing exact returned tool text")?;
    let id = tool
        .origin_response_id
        .as_ref()
        .context("missing originating assistant response id")?;
    let origins: Vec<_> = requests
        .iter()
        .filter(|r| r.server_request_id.as_ref() == Some(id))
        .collect();
    anyhow::ensure!(
        origins.len() == 1,
        "origin response id has no unique request"
    );
    let origin = origins[0];
    let prefix = normalize(&origin.messages)?;
    for following in requests
        .iter()
        .filter(|r| r.request_index > origin.request_index && successful(r))
    {
        let messages = normalize(&following.messages)?;
        if !messages.starts_with(&prefix) {
            continue;
        }
        let assistant_index = prefix.len();
        let Some(assistant) = messages
            .get(assistant_index)
            .filter(|m| m["role"] == "assistant")
        else {
            continue;
        };
        if !assistant["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|call| {
                call["id"] == tool.call_id
                    && call["function"]["name"] == tool.name
                    && call["function"]["arguments"] == tool.arguments
            })
        {
            continue;
        }
        // Only this appended assistant turn's consecutive tool results count.
        // Earlier identical calls/results cannot witness a later invocation.
        for (index, message) in messages
            .iter()
            .enumerate()
            .skip(assistant_index + 1)
            .take_while(|(_, m)| m["role"] == "tool")
        {
            if message["tool_call_id"] == tool.call_id
                && message["content"].as_str() == Some(result)
            {
                return Ok(Receipt {
                    tool_key: key.into(),
                    assistant_turn: tool.assistant_turn,
                    origin_response_id: id.clone(),
                    origin_request_index: origin.request_index,
                    following_request_index: following.request_index,
                    assistant_message_index: assistant_index,
                    tool_message_index: index,
                    result_sha256: format!("{:x}", Sha256::digest(result.as_bytes())),
                    compaction: None,
                });
            }
        }
    }
    anyhow::bail!("no successful subsequent request appends this origin turn and its exact result; missing evidence or rewritten/compacted history is unproven")
}

fn successful(r: &RequestRecord) -> bool {
    r.error.is_none()
        && r.http_status == Some(200)
        && r.saw_done
        && r.finish_reasons
            .iter()
            .any(|s| s == "stop" || s == "tool_calls")
}

fn normalize(messages: &[Value]) -> Result<Vec<Value>> {
    messages
        .iter()
        .map(|message| {
            let mut message = message.clone();
            if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    let arguments = call["function"]["arguments"]
                        .as_str()
                        .context("wire tool arguments are not JSON text")?;
                    call["function"]["arguments"] =
                        serde_json::from_str(arguments).context("invalid wire tool arguments")?;
                }
            }
            Ok(message)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn assistant(path: &str) -> Value {
        json!({"role":"assistant","content":null,"tool_calls":[
        {"id":"call_0","type":"function","function":{"name":"read","arguments":json!({"path":path}).to_string()}}]})
    }
    fn result(text: &str) -> Value {
        json!({"role":"tool","tool_call_id":"call_0","content":text})
    }
    fn request(index: u32, id: &str, messages: Vec<Value>) -> RequestRecord {
        RequestRecord {
            request_index: index,
            server_request_id: Some(id.into()),
            messages,
            http_status: Some(200),
            saw_done: true,
            finish_reasons: vec!["tool_calls".into()],
            ..Default::default()
        }
    }
    fn tool(origin: &str, path: &str, text: &str) -> Tool {
        Tool {
            call_id: "call_0".into(),
            assistant_turn: 2,
            origin_response_id: Some(origin.into()),
            name: "read".into(),
            arguments: json!({"path":path}),
            requested: true,
            started_ns: Some(100),
            ended_ns: Some(110),
            returned_to_session: true,
            returned_ns: Some(120),
            result_text: Some(text.into()),
            ..Default::default()
        }
    }
    #[test]
    fn repeated_identical_id_arguments_and_result_cannot_borrow_old_turn() {
        let input = vec![json!({"role":"user","content":"fix"})];
        let first = request(0, "r0", input.clone());
        let mut history = input;
        history.extend([assistant("same"), result("same output\n")]);
        let second = request(1, "r1", history.clone());
        let stale = request(2, "r2", history.clone());
        let current = tool("r1", "same", "same output\n");
        assert!(witness("2:call_0", &current, &[&first, &second, &stale]).is_err());
        history.extend([assistant("same"), result("same output\n")]);
        let fresh = request(3, "r3", history);
        let receipt = witness("2:call_0", &current, &[&first, &second, &stale, &fresh]).unwrap();
        assert_eq!(receipt.assistant_message_index, 3);
        assert_eq!(receipt.following_request_index, 3);
        // Cross-channel observation can lag; HTTP receipt timestamps are not
        // assumed to order pi's asynchronous stdout event collection.
        assert!(fresh.submitted_ns < current.ended_ns.unwrap());
    }
    #[test]
    fn failed_requests_wrong_arguments_and_changed_whitespace_do_not_witness() {
        let origin = request(4, "r4", vec![json!({"role":"user","content":"fix"})]);
        let current = tool("r4", "new", " x\n");
        let mut next = request(
            5,
            "r5",
            vec![origin.messages[0].clone(), assistant("old"), result(" x\n")],
        );
        assert!(witness("2:call_0", &current, &[&origin, &next]).is_err());
        next.messages[1] = assistant("new");
        next.messages[2] = result("x");
        assert!(witness("2:call_0", &current, &[&origin, &next]).is_err());
        next.messages[2] = result(" x\n");
        next.error = Some("HTTP failed".into());
        assert!(witness("2:call_0", &current, &[&origin, &next]).is_err());
        next.error = None;
        next.request_index = 3;
        assert!(witness("2:call_0", &current, &[&origin, &next]).is_err());
        next.request_index = 5;
        assert!(witness("2:call_0", &current, &[&origin, &next]).is_ok());
    }
    #[test]
    fn every_parallel_call_needs_its_own_return() {
        let origin = request(0, "r0", vec![json!({"role":"user","content":"fix"})]);
        let mut group = assistant("a");
        group["tool_calls"].as_array_mut().unwrap().push(json!({"id":"call_1","type":"function", "function":{"name":"read","arguments":"{\"path\":\"b\"}"}}));
        let next = request(
            1,
            "r1",
            vec![origin.messages[0].clone(), group, result("a text")],
        );
        let mut events = Events::default();
        events
            .tools
            .insert("1:call_0".into(), tool("r0", "a", "a text"));
        let mut second = tool("r0", "b", "b text");
        second.call_id = "call_1".into();
        events.tools.insert("1:call_1".into(), second);
        let evidence = verify(&events, &[&origin, &next]);
        assert_eq!(evidence.verified.len(), 1);
        assert!(!evidence.complete(&events));
    }
}
