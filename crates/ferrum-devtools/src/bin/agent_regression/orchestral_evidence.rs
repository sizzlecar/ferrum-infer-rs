//! Bounded reader for public Orchestral Run/Session journal evidence.
//!
//! This checks the single-run headless lifecycle and retains original canonical
//! messages. It neither interprets Generic private checkpoints nor authenticates
//! an HTTP replay or an independently correct task result. Journal digests are
//! retained in the source files, not reimplemented as a second protocol reducer.
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Default, Serialize)]
pub(crate) struct Evidence {
    pub session_id: Option<String>,
    pub run_id: Option<String>,
    pub terminal: Option<String>,
    pub delivered: bool,
    pub run_path: Option<PathBuf>,
    pub session_path: Option<PathBuf>,
    pub input: Option<String>,
    pub output: Option<String>,
    pub tool_exchanges: Vec<ToolExchange>,
    pub usage: Usage,
    pub compaction_events: usize,
    pub errors: Vec<String>,
    pub integrity_scope: &'static str,
    #[serde(skip)]
    lifecycle_complete: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolExchange {
    pub session_seq: u64,
    pub request_id: String,
    /// Unmodified public ModelMessage values, including provider-owned metadata.
    pub assistant: Value,
    pub tool: Value,
    pub calls: Vec<ToolCall>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolCall {
    pub call_id: String,
    pub native_call_id: Option<String>,
    pub name: String,
    pub arguments: Value,
    pub result: Value,
    pub is_error: bool,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct Usage {
    pub observed_requests: usize,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub unknown_input_requests: usize,
    pub unknown_output_requests: usize,
}

impl Evidence {
    /// Public lifecycle evidence only. Callers must separately bind transport,
    /// require normal process completion and run the independent validator.
    pub fn complete(&self) -> bool {
        self.errors.is_empty() && self.delivered && self.lifecycle_complete
    }
}

pub(crate) fn read(journal: &Path, expected_session: &str) -> Evidence {
    let mut evidence = Evidence {
        integrity_scope: "public schema, identities, sequence and lifecycle; digest cryptography and HTTP replay not certified by this reader",
        ..Evidence::default()
    };
    if let Err(error) = inspect(journal, expected_session, &mut evidence) {
        evidence.errors.push(error);
    }
    evidence
}

fn inspect(journal: &Path, expected_session: &str, evidence: &mut Evidence) -> Result<(), String> {
    require(!expected_session.is_empty(), "empty expected session")?;
    let mut runs = Vec::new();
    let mut sessions = Vec::new();
    for entry in fs::read_dir(journal).map_err(|error| format!("read journal: {error}"))? {
        let entry = entry.map_err(|error| format!("read journal entry: {error}"))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let destination = if name.starts_with("run-") && name.ends_with(".json") {
            &mut runs
        } else if name.starts_with("session-") && name.ends_with(".json") {
            &mut sessions
        } else {
            continue;
        };
        require(
            entry
                .file_type()
                .map_err(|error| error.to_string())?
                .is_file(),
            "public journal entry is not a regular file",
        )?;
        destination.push(entry.path());
    }
    require(
        runs.len() == 1 && sessions.len() == 1,
        "expected exactly one public Run and Session journal",
    )?;
    evidence.run_path = Some(runs[0].clone());
    evidence.session_path = Some(sessions[0].clone());
    let run = read_json(&runs[0])?;
    require(
        run["schema_version"] == 1,
        "unsupported public Run journal schema",
    )?;
    let spec = &run["run"]["registration"]["request"]["run"]["spec"];
    require(
        spec["protocol_version"]["major"] == 1,
        "unsupported public Agent Protocol major version",
    )?;
    let execution = &run["run"]["registration"]["execution"];
    let session = string(spec, "session_id")?;
    let run_id = string(spec, "run_id")?;
    evidence.session_id = Some(session.to_owned());
    evidence.run_id = Some(run_id.to_owned());
    require(
        session == expected_session,
        "Run belongs to a different session",
    )?;
    require(
        execution["session_id"] == session && execution["run_id"] == run_id,
        "Run registration/execution identity mismatch",
    )?;
    let spec_digest = string(&run["run"]["registration"]["request"]["run"], "spec_digest")?;
    require(
        execution["spec_digest"] == spec_digest,
        "Run spec digest identity mismatch",
    )?;
    let input = inline_texts(array(spec, "input")?)?;
    evidence.input = Some(input.clone());
    let events = array(&run["run"], "records")?;
    let mut event_ids = BTreeSet::new();
    let mut accepted = false;
    let mut started = false;
    let mut output_committed = None;
    for (index, record) in events.iter().enumerate() {
        let event = &record["event"];
        require(
            event["run_id"] == run_id
                && event["run_seq"].as_u64() == Some(index as u64 + 1)
                && event_ids.insert(string(event, "event_id")?.to_owned()),
            "Run event identity/sequence is not unique and continuous",
        )?;
        require(
            evidence.terminal.is_none(),
            "Run has events after its terminal",
        )?;
        let payload = &event["payload"];
        let kind = string(payload, "type")?;
        match kind {
            "run_accepted" => {
                require(
                    !accepted && index == 0,
                    "RunAccepted is not the unique first event",
                )?;
                require(
                    payload["session_id"] == session && payload["spec_digest"] == spec_digest,
                    "RunAccepted identity mismatch",
                )?;
                accepted = true;
            }
            "run_started" => {
                require(
                    accepted && !started,
                    "RunStarted is missing acceptance or repeated",
                )?;
                started = true;
            }
            "output_committed" => {
                require(started, "output precedes RunStarted")?;
                output_committed = Some(inline_texts(array(payload, "content")?)?);
            }
            "delivery_committed" => {
                require(started, "delivery precedes RunStarted")?;
                let delivery = &payload["delivery"];
                require(
                    delivery["run_id"] == run_id && delivery["spec_digest"] == spec_digest,
                    "Delivery identity mismatch",
                )?;
                let text = inline_text(&delivery["final_response"])?;
                require(
                    output_committed.as_deref() == Some(text),
                    "Delivery does not match committed output",
                )?;
                evidence.output = Some(text.to_owned());
                evidence.delivered = true;
                evidence.terminal = Some(kind.to_owned());
            }
            "run_failed" | "run_incomplete" | "run_cancelled" => {
                require(accepted, "terminal precedes RunAccepted")?;
                evidence.terminal = Some(kind.to_owned());
            }
            "resource_binding_skipped"
            | "command_received"
            | "command_disposition_recorded"
            | "input_committed"
            | "request_opened"
            | "request_resolved"
            | "request_closed"
            | "stop_requested"
            | "continuity_lost"
            | "continuity_restored" => require(accepted, "event precedes RunAccepted")?,
            _ => return Err("unsupported public Run event".to_owned()),
        }
    }
    require(
        accepted && evidence.terminal.is_some(),
        "Run has no accepted terminal lifecycle",
    )?;

    let session_document = read_json(&sessions[0])?;
    let records = session_document
        .as_array()
        .ok_or("public Session journal is not an array")?;
    let mut session_ids = BTreeSet::new();
    let mut call_ids = BTreeSet::new();
    let mut input_seen = false;
    let mut output_seen = false;
    let mut usages = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        let seq = index as u64 + 1;
        require(
            record["session_id"] == session
                && record["run_id"] == run_id
                && record["session_seq"].as_u64() == Some(seq)
                && session_ids.insert(string(record, "event_id")?.to_owned()),
            "Session event identity/sequence is not unique and continuous",
        )?;
        require(!output_seen, "Session events follow its final output")?;
        let payload = &record["payload"];
        match string(payload, "type")? {
            "run_input_committed" => {
                require(
                    !input_seen && index == 0,
                    "Session initial input is missing or repeated",
                )?;
                require(
                    model_text(&payload["message"], "user")? == input,
                    "Session input differs from Run input",
                )?;
                input_seen = true;
            }
            "tool_exchange_committed" => {
                require(input_seen, "Tool exchange precedes input")?;
                let exchange = tool_exchange(seq, payload, &mut call_ids)?;
                observe_usage(&mut usages, &exchange.request_id, &payload["usage"])?;
                evidence.tool_exchanges.push(exchange);
            }
            "run_output_committed" => {
                require(input_seen, "Session output precedes input")?;
                let request_id = string(payload, "request_id")?;
                let output = model_text(&payload["message"], "assistant")?;
                require(
                    evidence.output.as_deref() == Some(output.as_str()),
                    "Session output differs from Delivery",
                )?;
                observe_usage(&mut usages, request_id, &payload["usage"])?;
                output_seen = true;
            }
            "compaction_committed" | "active_run_compaction_committed" => {
                let first = payload["source"]["first_session_seq"]
                    .as_u64()
                    .ok_or("missing compaction source start")?;
                let last = payload["source"]["last_session_seq"]
                    .as_u64()
                    .ok_or("missing compaction source end")?;
                require(
                    input_seen && first > 0 && first <= last && last < seq,
                    "invalid compaction source range",
                )?;
                evidence.compaction_events += 1;
            }
            "skill_loaded" | "effect_uncertainty_committed" => {
                require(input_seen, "Session fact precedes input")?;
            }
            _ => return Err("unsupported public Session event".to_owned()),
        }
    }
    evidence.usage = summarize_usage(&usages)?;
    require(input_seen, "Session has no committed input")?;
    if evidence.delivered {
        require(output_seen, "Delivered Run has no Session output")?;
    }
    evidence.lifecycle_complete = accepted && started && input_seen && output_seen;
    Ok(())
}

fn tool_exchange(
    seq: u64,
    payload: &Value,
    seen: &mut BTreeSet<String>,
) -> Result<ToolExchange, String> {
    let request_id = string(payload, "request_id")?.to_owned();
    let assistant = &payload["assistant"];
    let tool = &payload["tool"];
    require(
        assistant["role"] == "assistant" && tool["role"] == "tool",
        "Tool exchange roles mismatch",
    )?;
    let mut results = BTreeMap::new();
    for content in array(tool, "content")? {
        require(
            content["type"] == "tool_result",
            "non-result in Tool message",
        )?;
        let id = string(content, "call_id")?;
        require(
            results.insert(id, content).is_none(),
            "duplicate Tool result",
        )?;
    }
    let mut calls = Vec::new();
    for call in array(assistant, "content")? {
        if call["type"] != "tool_call" {
            require(
                matches!(call["type"].as_str(), Some("text" | "json" | "data")),
                "unsupported Assistant content",
            )?;
            continue;
        }
        let id = string(call, "call_id")?;
        require(
            seen.insert(id.to_owned()),
            "duplicate canonical Tool call identity",
        )?;
        let result = results
            .remove(id)
            .ok_or("Tool call has no exact paired result")?;
        calls.push(ToolCall {
            call_id: id.to_owned(),
            native_call_id: call["extensions"]["openai/tool_call_id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .map(str::to_owned),
            name: string(call, "name")?.to_owned(),
            arguments: call
                .get("arguments")
                .ok_or("Tool call omitted arguments")?
                .clone(),
            result: result
                .get("result")
                .ok_or("Tool result omitted result")?
                .clone(),
            is_error: result["is_error"]
                .as_bool()
                .ok_or("Tool result omitted error status")?,
        });
    }
    require(
        !calls.is_empty() && results.is_empty(),
        "Tool calls/results are not one nonempty pair set",
    )?;
    Ok(ToolExchange {
        session_seq: seq,
        request_id,
        assistant: assistant.clone(),
        tool: tool.clone(),
        calls,
    })
}

fn observe_usage(
    usages: &mut BTreeMap<String, Value>,
    request: &str,
    usage: &Value,
) -> Result<(), String> {
    if let Some(previous) = usages.get(request) {
        require(previous == usage, "conflicting usage for one model request")?;
    } else {
        usages.insert(request.to_owned(), usage.clone());
    }
    Ok(())
}

fn summarize_usage(usages: &BTreeMap<String, Value>) -> Result<Usage, String> {
    let mut result = Usage {
        observed_requests: usages.len(),
        ..Usage::default()
    };
    for (field, total, unknown) in [
        (
            "input_tokens",
            &mut result.input_tokens,
            &mut result.unknown_input_requests,
        ),
        (
            "output_tokens",
            &mut result.output_tokens,
            &mut result.unknown_output_requests,
        ),
    ] {
        let mut sum = 0_u64;
        for usage in usages.values() {
            match usage.get(field) {
                None | Some(Value::Null) => *unknown += 1,
                Some(value) => {
                    let value = value.as_u64().ok_or("invalid model token usage")?;
                    sum = sum.checked_add(value).ok_or("model token usage overflow")?;
                }
            }
        }
        if !usages.is_empty() && *unknown == 0 {
            *total = Some(sum);
        }
    }
    Ok(result)
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("missing or empty {key}"))
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], String> {
    value[key]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing array {key}"))
}

fn inline_text(content: &Value) -> Result<&str, String> {
    require(
        content["body"]["kind"] == "inline",
        "headless evidence requires inline text content",
    )?;
    content["body"]["value"]
        .as_str()
        .ok_or_else(|| "headless content is not text".to_owned())
}

fn inline_texts(content: &[Value]) -> Result<String, String> {
    require(!content.is_empty(), "empty headless input/output content")?;
    content
        .iter()
        .map(inline_text)
        .collect::<Result<Vec<_>, _>>()
        .map(|texts| texts.join("\n"))
}

fn model_text(message: &Value, role: &str) -> Result<String, String> {
    require(message["role"] == role, "ModelMessage role mismatch")?;
    let content = array(message, "content")?;
    require(!content.is_empty(), "empty ModelMessage")?;
    content
        .iter()
        .map(|item| {
            require(
                item["type"] == "text",
                "headless final/input message is not plain text",
            )?;
            item["text"]
                .as_str()
                .ok_or_else(|| "missing ModelMessage text".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|texts| texts.join("\n"))
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn content(text: &str) -> Value {
        json!({"body":{"kind":"inline","value":text}})
    }

    fn message(role: &str, text: &str) -> Value {
        json!({"role":role,"content":[{"type":"text","text":text}]})
    }

    fn fixtures() -> (Value, Value) {
        let events = vec![
            json!({"type":"run_accepted","session_id":"session","spec_digest":"spec"}),
            json!({"type":"run_started"}),
            json!({"type":"output_committed","content":[content("Finished")]}),
            json!({"type":"delivery_committed","delivery":{"run_id":"run","spec_digest":"spec","final_response":content("Finished")}}),
        ];
        let run = json!({"schema_version":1,"run":{
            "registration":{
                "request":{"run":{"spec":{"protocol_version":{"major":1,"minor":0},"session_id":"session","run_id":"run","input":[content("Repair code")]},"spec_digest":"spec"}},
                "execution":{"session_id":"session","run_id":"run","spec_digest":"spec"}
            },
            "records":events.into_iter().enumerate().map(|(index,payload)| json!({"event":{"event_id":format!("event-{index}"),"run_id":"run","run_seq":index+1,"payload":payload}})).collect::<Vec<_>>()
        }});
        let session = vec![
            json!({"type":"run_input_committed","message":message("user","Repair code")}),
            json!({"type":"tool_exchange_committed","request_id":"model-1",
                "assistant":{"role":"assistant","content":[{"type":"tool_call","call_id":"canonical-1","name":"file_read","arguments":{"path":"src/lib.rs"},"extensions":{"openai/tool_call_id":"native-1"}}]},
                "tool":{"role":"tool","content":[{"type":"tool_result","call_id":"canonical-1","result":{"text":"fn answer() {}\n"},"is_error":false}]},
                "usage":{"input_tokens":30,"output_tokens":8}}),
            json!({"type":"run_output_committed","request_id":"model-2","message":message("assistant","Finished"),"usage":{"input_tokens":45,"output_tokens":4}}),
        ];
        (run,json!(session.into_iter().enumerate().map(|(index,payload)| json!({"session_seq":index+1,"event_id":format!("session-{index}"),"session_id":"session","run_id":"run","payload":payload})).collect::<Vec<_>>()))
    }

    fn inspect_fixture(run: &Value, session: &Value) -> Evidence {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("run-test.json"),
            serde_json::to_vec(run).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.path().join("session-test.json"),
            serde_json::to_vec(session).unwrap(),
        )
        .unwrap();
        read(dir.path(), "session")
    }

    #[test]
    fn public_delivery_and_exact_tool_pair_are_retained_without_pi_events() {
        let (run, session) = fixtures();
        let evidence = inspect_fixture(&run, &session);
        assert!(evidence.complete(), "{:?}", evidence.errors);
        assert_eq!(evidence.tool_exchanges.len(), 1);
        let call = &evidence.tool_exchanges[0].calls[0];
        assert_eq!(call.native_call_id.as_deref(), Some("native-1"));
        assert_eq!(call.arguments, json!({"path":"src/lib.rs"}));
        assert_eq!(call.result, json!({"text":"fn answer() {}\n"}));
        assert_eq!(evidence.usage.input_tokens, Some(75));
        assert_eq!(evidence.usage.output_tokens, Some(12));
    }

    #[test]
    fn failed_run_retains_real_partial_usage_without_claiming_delivery() {
        let (mut run, mut session) = fixtures();
        run["run"]["records"].as_array_mut().unwrap().truncate(3);
        run["run"]["records"][2]["event"]["payload"] =
            json!({"type":"run_failed","failure":{"code":"context_overflow"}});
        session.as_array_mut().unwrap().truncate(2);
        let evidence = inspect_fixture(&run, &session);
        assert!(evidence.errors.is_empty(), "{:?}", evidence.errors);
        assert!(!evidence.complete());
        assert_eq!(evidence.terminal.as_deref(), Some("run_failed"));
        assert_eq!(evidence.usage.input_tokens, Some(30));
    }

    #[test]
    fn foreign_identity_discontinuous_sequence_and_post_terminal_events_fail_closed() {
        let (run, session) = fixtures();
        for pointer in [
            "/run/registration/execution/session_id",
            "/run/records/1/event/run_id",
        ] {
            let mut bad = run.clone();
            *bad.pointer_mut(pointer).unwrap() = json!("other");
            assert!(!inspect_fixture(&bad, &session).complete());
        }
        let mut bad = run.clone();
        bad["run"]["records"][1]["event"]["run_seq"] = json!(4);
        assert!(!inspect_fixture(&bad, &session).complete());
        let mut unknown = run.clone();
        unknown["run"]["records"][2]["event"]["payload"]["type"] = json!("future_event");
        assert!(!inspect_fixture(&unknown, &session).complete());
        let mut bad = run.clone();
        let mut extra = bad["run"]["records"][3].clone();
        extra["event"]["run_seq"] = json!(5);
        extra["event"]["event_id"] = json!("second-terminal");
        bad["run"]["records"].as_array_mut().unwrap().push(extra);
        assert!(!inspect_fixture(&bad, &session).complete());
        let mut bad_session = session.clone();
        bad_session[1]["session_seq"] = json!(9);
        assert!(!inspect_fixture(&run, &bad_session).complete());
    }

    #[test]
    fn missing_or_changed_committed_input_and_delivery_do_not_pass() {
        let (run, session) = fixtures();
        let mut bad = session.clone();
        bad[0]["payload"]["message"] = message("user", "Different task");
        assert!(!inspect_fixture(&run, &bad).complete());
        bad = session.clone();
        bad[2]["payload"]["message"] = message("assistant", "Fabricated success");
        assert!(!inspect_fixture(&run, &bad).complete());
        bad.as_array_mut().unwrap().pop();
        assert!(!inspect_fixture(&run, &bad).complete());
    }

    #[test]
    fn missing_tool_result_and_duplicate_canonical_calls_are_rejected() {
        let (run, mut session) = fixtures();
        session[1]["payload"]["tool"]["content"][0]["call_id"] = json!("different-call");
        assert!(!inspect_fixture(&run, &session).complete());
        let (_, mut session) = fixtures();
        let mut duplicate = session[1].clone();
        duplicate["session_seq"] = json!(3);
        duplicate["event_id"] = json!("another-event");
        session.as_array_mut().unwrap().insert(2, duplicate);
        session[3]["session_seq"] = json!(4);
        assert!(!inspect_fixture(&run, &session).complete());
    }

    #[test]
    fn unknown_usage_remains_unknown_and_shared_request_usage_is_not_added_twice() {
        let (run, mut session) = fixtures();
        session[2]["payload"]["usage"] = Value::Null;
        let evidence = inspect_fixture(&run, &session);
        assert!(evidence.complete());
        assert_eq!(evidence.usage.input_tokens, None);
        assert_eq!(evidence.usage.unknown_input_requests, 1);
        let mut usage = BTreeMap::new();
        let value = json!({"input_tokens":30,"output_tokens":8});
        observe_usage(&mut usage, "request", &value).unwrap();
        observe_usage(&mut usage, "request", &value).unwrap();
        assert_eq!(summarize_usage(&usage).unwrap().input_tokens, Some(30));
        assert!(observe_usage(&mut usage, "request", &Value::Null).is_err());
    }

    #[test]
    fn malformed_or_missing_public_journals_are_reported_not_panicked() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!read(dir.path(), "session").errors.is_empty());
        fs::write(dir.path().join("run-test.json"), "{").unwrap();
        fs::write(dir.path().join("session-test.json"), "[]").unwrap();
        let evidence = read(dir.path(), "session");
        assert!(!evidence.complete());
        assert!(evidence.errors[0].contains("parse"));
    }
}
