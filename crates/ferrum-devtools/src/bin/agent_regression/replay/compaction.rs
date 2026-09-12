//! A retained session branch can witness replay after Pi compacts old history.
//! Summaries alone, substring matches and repeated call ids never prove replay.
use super::{normalize, successful, Events, Evidence, RequestRecord, Tool};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

const SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
const SUMMARY_SUFFIX: &str = "\n</summary>";

#[derive(Debug, Serialize)]
pub(crate) struct Receipt {
    pub session_id: String,
    pub session_sha256: String,
    pub compaction_entry_id: String,
    pub first_kept_entry_id: String,
    pub origin_assistant_entry_id: String,
    pub result_entry_id: String,
    pub following_assistant_entry_id: String,
}

/// The caller supplies the actual same-session file after Pi has stopped.
/// Failure to load/bind it augments unproven reasons; it never drops the task
/// report, revokes already-proven ordinary replay, or converts a failure to pass.
pub(crate) fn verify_with_session(
    events: &Events,
    requests: &[&RequestRecord],
    session_file: &Path,
) -> Evidence {
    let mut evidence = super::verify(events, requests);
    if evidence.unproven.is_empty() {
        return evidence;
    }
    let session = Session::read(session_file, events.session_id.as_deref());
    let keys: Vec<_> = evidence.unproven.keys().cloned().collect();
    for key in keys {
        let result = match &session {
            Ok(session) => session.witness(&key, &events.tools[&key], requests),
            Err(error) => Err(anyhow::anyhow!("session evidence unavailable: {error:#}")),
        };
        match result {
            Ok(receipt) => {
                evidence.unproven.remove(&key);
                evidence.verified.push(receipt);
            }
            Err(error) => evidence
                .unproven
                .get_mut(&key)
                .unwrap()
                .push_str(&format!("; compaction: {error:#}")),
        }
    }
    evidence
}

struct Session {
    id: String,
    sha256: String,
    entries: BTreeMap<String, Value>,
}

impl Session {
    fn read(path: &Path, expected_id: Option<&str>) -> Result<Self> {
        ensure!(
            std::fs::symlink_metadata(path)?.file_type().is_file(),
            "session must be a regular file"
        );
        let expected_id = expected_id
            .filter(|id| !id.is_empty())
            .context("missing observed session id")?;
        let mut reader = BufReader::new(File::open(path)?);
        let mut bytes = Vec::new();
        let mut digest = Sha256::new();
        let mut entries = BTreeMap::new();
        let mut header = None;
        while reader.read_until(b'\n', &mut bytes)? != 0 {
            digest.update(&bytes);
            if !bytes.iter().all(u8::is_ascii_whitespace) {
                let entry: Value =
                    serde_json::from_slice(&bytes).context("invalid session JSONL")?;
                if header.is_none() {
                    ensure!(
                        entry["type"] == "session"
                            && entry["version"] == 3
                            && entry["id"] == expected_id,
                        "session header/version differs from the observed Pi session"
                    );
                    header = Some(expected_id.to_owned());
                } else {
                    ensure!(entry["type"] != "session", "multiple session headers");
                    let id = entry["id"]
                        .as_str()
                        .filter(|id| !id.is_empty())
                        .context("session entry has no id")?
                        .to_owned();
                    ensure!(!entries.contains_key(&id), "duplicate session entry id");
                    match entry.get("parentId") {
                        Some(Value::Null) => {}
                        Some(Value::String(parent)) => ensure!(
                            entries.contains_key(parent),
                            "session parent is absent or not earlier"
                        ),
                        _ => anyhow::bail!("session entry has no valid parent"),
                    }
                    entries.insert(id, entry);
                }
            }
            bytes.clear();
        }
        Ok(Self {
            id: header.context("empty session")?,
            sha256: format!("{:x}", digest.finalize()),
            entries,
        })
    }

    fn path_to<'a>(&'a self, leaf: &'a Value) -> Result<Vec<&'a Value>> {
        // read() accepts parents only after their definition: cycles are impossible.
        let mut path = vec![leaf];
        let mut current = leaf;
        while let Some(parent) = current["parentId"].as_str() {
            current = self
                .entries
                .get(parent)
                .context("missing session ancestor")?;
            path.push(current);
        }
        path.reverse();
        Ok(path)
    }

    fn witness(
        &self,
        key: &str,
        tool: &Tool,
        requests: &[&RequestRecord],
    ) -> Result<super::Receipt> {
        ensure!(
            tool.requested
                && tool.started_ns.is_some()
                && tool.ended_ns.is_some()
                && tool.returned_to_session,
            "invocation did not execute and return"
        );
        let response_id = tool
            .origin_response_id
            .as_ref()
            .context("missing origin response id")?;
        let origins: Vec<_> = requests
            .iter()
            .filter(|r| r.server_request_id.as_ref() == Some(response_id))
            .collect();
        ensure!(
            origins.len() == 1,
            "origin response id has no unique request"
        );
        let assistants: Vec<_> = self
            .entries
            .values()
            .filter(|entry| {
                entry["type"] == "message"
                    && entry["message"]["role"] == "assistant"
                    && entry["message"]["responseId"] == *response_id
            })
            .collect();
        ensure!(
            assistants.len() == 1,
            "origin response id has no unique session assistant"
        );
        let origin = origins[0];
        let assistant = assistants[0];
        let mut last_error = None;
        for compaction in self
            .entries
            .values()
            .filter(|entry| entry["type"] == "compaction")
        {
            match self.compacted_witness(key, tool, origin, assistant, compaction, requests) {
                Ok(receipt) => return Ok(receipt),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no recorded compaction")))
    }

    fn compacted_witness(
        &self,
        key: &str,
        tool: &Tool,
        origin: &RequestRecord,
        assistant: &Value,
        compaction: &Value,
        requests: &[&RequestRecord],
    ) -> Result<super::Receipt> {
        let path = self.path_to(compaction)?;
        let first = compaction["firstKeptEntryId"]
            .as_str()
            .context("compaction has no first kept entry")?;
        let kept = path
            .iter()
            .position(|entry| entry["id"] == first)
            .context("first kept entry is not an ancestor")?;
        let assistant_at = path
            .iter()
            .position(|entry| entry["id"] == assistant["id"])
            .context("origin assistant is not a compaction ancestor")?;
        ensure!(
            kept <= assistant_at && assistant_at < path.len() - 1,
            "compaction did not retain the origin assistant"
        );
        let result = tool
            .result_text
            .as_ref()
            .context("missing exact returned tool text")?;
        let assistant_wire = project(assistant)?.context("missing assistant projection")?;
        ensure!(
            assistant_wire["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|call| call["id"] == tool.call_id
                    && call["function"]["name"] == tool.name
                    && call["function"]["arguments"] == tool.arguments),
            "session origin tool call differs from observed invocation"
        );
        let mut result_at = None;
        for (index, entry) in path
            .iter()
            .enumerate()
            .take(path.len() - 1)
            .skip(assistant_at + 1)
        {
            let Some(message) = project(entry)? else {
                continue;
            };
            if message["role"] != "tool" {
                break;
            }
            if message["tool_call_id"] == tool.call_id {
                ensure!(
                    result_at.is_none(),
                    "duplicate tool result in the same origin turn"
                );
                ensure!(
                    entry["message"]["toolName"] == tool.name
                        && message["content"].as_str() == Some(result),
                    "session tool result differs from actual returned text"
                );
                result_at = Some(index);
            }
        }
        let result_at = result_at.context("origin has no exact consecutive session result")?;
        let before: Vec<_> = path[kept..assistant_at]
            .iter()
            .filter_map(|entry| project(entry).transpose())
            .collect::<Result<_>>()?;
        let origin_messages = normalize(&origin.messages)?;
        ensure!(
            origin_messages.ends_with(&before),
            "retained pre-origin entries differ from the actual origin request tail"
        );
        let system_len = origin_messages
            .iter()
            .take_while(|message| message["role"] == "system" || message["role"] == "developer")
            .count();
        ensure!(system_len > 0, "origin has no explicit system context");
        let mut expected = origin_messages[..system_len].to_vec();
        let summary = compaction["summary"]
            .as_str()
            .context("compaction has no summary")?;
        expected.push(json!({"role":"user","content":[{"type":"text", "text":format!("{SUMMARY_PREFIX}{summary}{SUMMARY_SUFFIX}")}]}));
        let offset = expected.len();
        let mut assistant_index = None;
        let mut result_index = None;
        for (index, entry) in path.iter().enumerate().take(path.len() - 1).skip(kept) {
            if let Some(message) = project(entry)? {
                if index == assistant_at {
                    assistant_index = Some(expected.len());
                }
                if index == result_at {
                    result_index = Some(expected.len());
                }
                expected.push(message);
            }
        }
        ensure!(expected.len() > offset, "compaction retained no messages");
        for following in requests
            .iter()
            .filter(|r| r.request_index > origin.request_index && successful(r))
        {
            let Some(response_id) = following.server_request_id.as_ref() else {
                continue;
            };
            let next_assistants: Vec<_> = self
                .entries
                .values()
                .filter(|entry| {
                    entry["type"] == "message"
                        && entry["message"]["role"] == "assistant"
                        && entry["message"]["responseId"] == *response_id
                })
                .collect();
            if next_assistants.len() != 1 {
                continue;
            }
            let next_assistant = next_assistants[0];
            let next_path = self.path_to(next_assistant)?;
            let Some(compaction_at) = next_path
                .iter()
                .position(|entry| entry["id"] == compaction["id"])
            else {
                continue;
            };
            // Bind the following request to its own response/session branch too.
            // A later request with a coincidentally identical old call/result
            // cannot borrow this compaction from an unrelated session branch.
            let mut next_expected = expected.clone();
            for entry in &next_path[compaction_at + 1..next_path.len() - 1] {
                if let Some(message) = project(entry)? {
                    next_expected.push(message);
                }
            }
            if normalize(&following.messages)? == next_expected {
                return Ok(super::Receipt {
                    tool_key: key.into(),
                    assistant_turn: tool.assistant_turn,
                    origin_response_id: tool.origin_response_id.clone().unwrap(),
                    origin_request_index: origin.request_index,
                    following_request_index: following.request_index,
                    assistant_message_index: assistant_index
                        .context("missing retained assistant")?,
                    tool_message_index: result_index.context("missing retained result")?,
                    result_sha256: format!("{:x}", Sha256::digest(result.as_bytes())),
                    compaction: Some(Receipt {
                        session_id: self.id.clone(),
                        session_sha256: self.sha256.clone(),
                        compaction_entry_id: compaction["id"].as_str().unwrap().into(),
                        first_kept_entry_id: first.into(),
                        origin_assistant_entry_id: assistant["id"].as_str().unwrap().into(),
                        result_entry_id: path[result_at]["id"].as_str().unwrap().into(),
                        following_assistant_entry_id: next_assistant["id"].as_str().unwrap().into(),
                    }),
                });
            }
        }
        anyhow::bail!("no successful request contains the complete bound compaction prefix")
    }
}

/// Exact text-only OpenAI projection used by this local Pi harness. Do not guess
/// image, thinking, branch-summary or extension transformations; leave unproven.
fn project(entry: &Value) -> Result<Option<Value>> {
    match entry["type"].as_str() {
        Some("model_change" | "thinking_level_change" | "custom" | "label" | "session_info") => {
            return Ok(None)
        }
        Some("message") => {}
        _ => anyhow::bail!("unsupported retained context entry: {}", entry["type"]),
    }
    let message = &entry["message"];
    let content = message["content"]
        .as_array()
        .context("session message content is not an array")?;
    let text = || -> Result<Vec<&str>> {
        content
            .iter()
            .map(|block| {
                ensure!(block["type"] == "text", "unsupported non-text content");
                block["text"].as_str().context("text block has no text")
            })
            .collect()
    };
    Ok(Some(match message["role"].as_str() {
        Some("user") => {
            text()?;
            json!({"role":"user", "content":content})
        }
        Some("toolResult") => {
            let text = text()?.join("\n");
            json!({"role":"tool", "tool_call_id":message["toolCallId"],
                "content":if text.is_empty() { "(no tool output)" } else { &text }})
        }
        Some("assistant") => {
            let mut text = String::new();
            let mut calls = Vec::new();
            for block in content {
                match block["type"].as_str() {
                    Some("text") => {
                        text.push_str(block["text"].as_str().context("assistant text absent")?)
                    }
                    Some("toolCall") => {
                        ensure!(
                            block["id"].is_string()
                                && block["name"].is_string()
                                && block["arguments"].is_object(),
                            "invalid session tool call"
                        );
                        calls.push(json!({"id":block["id"],"type":"function", "function":{"name":block["name"],"arguments":block["arguments"]}}));
                    }
                    _ => anyhow::bail!("unsupported assistant content projection"),
                }
            }
            let mut wire = json!({"role":"assistant", "content":if text.is_empty() { Value::Null } else { Value::String(text) }});
            if !calls.is_empty() {
                wire["tool_calls"] = calls.into();
            }
            wire
        }
        _ => anyhow::bail!("unsupported session message role"),
    }))
}

#[cfg(test)]
#[path = "compaction/tests.rs"]
mod tests;
