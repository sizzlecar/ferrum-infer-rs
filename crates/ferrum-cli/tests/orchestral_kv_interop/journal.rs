use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(super) struct Snapshot {
    pub(super) session: Vec<Value>,
    pub(super) runs: BTreeMap<String, Vec<Value>>,
}

impl Snapshot {
    pub(super) fn read(directory: &Path) -> Result<Self> {
        let mut snapshot = Self::default();
        if !directory.exists() {
            return Ok(snapshot);
        }
        let mut sessions = 0;
        // Deliberately non-recursive: this is the explicit, newly created test
        // journal directory. Never discover or open a user's existing journals.
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            ensure!(
                !entry.file_type()?.is_symlink(),
                "unexpected journal symlink"
            );
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.ends_with(".json") {
                continue;
            }
            if name.starts_with("session-") {
                sessions += 1;
                snapshot.session = serde_json::from_slice(&fs::read(entry.path())?)?;
            } else if name.starts_with("run-") {
                let value: Value = serde_json::from_slice(&fs::read(entry.path())?)?;
                let records = value["run"]["records"]
                    .as_array()
                    .or_else(|| value["records"].as_array())
                    .context("run journal missing records")?
                    .clone();
                let run_id = records
                    .first()
                    .and_then(|r| r["event"]["run_id"].as_str())
                    .context("run journal missing event.run_id")?
                    .to_owned();
                ensure!(
                    snapshot.runs.insert(run_id, records).is_none(),
                    "duplicate run journal identity"
                );
            }
        }
        ensure!(sessions <= 1, "expected a single isolated session");
        for (index, record) in snapshot.session.iter().enumerate() {
            ensure!(
                record["session_seq"].as_u64() == Some(index as u64 + 1),
                "session sequence gap"
            );
            ensure!(
                record["session_id"] == SESSION,
                "different session identity"
            );
            ensure!(
                record["payload"]["type"] != "effect_uncertainty_committed",
                "uncertain tool effect"
            );
        }
        Ok(snapshot)
    }

    pub(super) fn run_ids(&self) -> BTreeSet<String> {
        self.runs.keys().cloned().collect()
    }

    pub(super) fn tool_exchange_count(&self) -> usize {
        self.session
            .iter()
            .filter(|record| record["payload"]["type"] == "tool_exchange_committed")
            .count()
    }

    pub(super) fn validate_file_read(&self, filename: &str, facts: &Value) -> Result<()> {
        let mut matched = false;
        for exchange in self
            .session
            .iter()
            .filter(|record| record["payload"]["type"] == "tool_exchange_committed")
        {
            let payload = &exchange["payload"];
            let calls = payload["assistant"]["content"]
                .as_array()
                .context("missing assistant tool content")?;
            let results = payload["tool"]["content"]
                .as_array()
                .context("missing tool result content")?;
            for call in calls.iter().filter(|item| item["type"] == "tool_call") {
                ensure!(
                    call["name"] == "file_read",
                    "unexpected tool {} in read-only scenario",
                    call["name"]
                );
                let path = call["arguments"]["path"]
                    .as_str()
                    .context("file_read missing path")?;
                ensure!(
                    Path::new(path).file_name().and_then(|v| v.to_str()) == Some(filename),
                    "wrong file_read path {path}"
                );
                let id = call["call_id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .context("empty tool call identity")?;
                let matches: Vec<_> = results
                    .iter()
                    .filter(|item| item["type"] == "tool_result" && item["call_id"] == id)
                    .collect();
                ensure!(
                    matches.len() == 1,
                    "tool call needs exactly one matching result"
                );
                let result = matches[0];
                ensure!(
                    result["is_error"] == false,
                    "file_read returned an error: {result}"
                );
                let content = result["result"]["content"]
                    .as_str()
                    .context("file_read result has no content")?;
                let actual: Value = serde_json::from_str(content)
                    .context("file_read did not return the actual JSON file")?;
                ensure!(
                    &actual == facts,
                    "file_read result differs from generated file"
                );
                matched = true;
            }
        }
        ensure!(
            matched,
            "no canonical file_read call/result exchange; textual claims do not count"
        );
        Ok(())
    }

    pub(super) fn new_delivery(&self, before: &BTreeSet<String>) -> Result<String> {
        let new: Vec<_> = self
            .runs
            .iter()
            .filter(|(id, _)| !before.contains(*id))
            .collect();
        ensure!(
            new.len() == 1,
            "one CLI turn must produce exactly one new run"
        );
        let (run_id, records) = new[0];
        let mut delivery = None;
        for record in records {
            let event = &record["event"];
            ensure!(
                &event["run_id"] == run_id,
                "run identity changed inside journal"
            );
            let payload = &event["payload"];
            match payload["type"].as_str().context("missing run event type")? {
                "run_failed" | "run_incomplete" | "run_cancelled" | "continuity_lost" => {
                    anyhow::bail!("unsuccessful agent event: {payload}")
                }
                "delivery_committed" => {
                    ensure!(delivery.is_none(), "duplicate terminal delivery");
                    let content = &payload["delivery"]["final_response"];
                    ensure!(
                        content["body"]["kind"] == "inline",
                        "delivery is not inline text"
                    );
                    delivery = Some(
                        content["body"]["value"]
                            .as_str()
                            .context("delivery body is not text")?
                            .to_owned(),
                    );
                }
                _ => {}
            }
        }
        delivery.context("no committed final delivery")
    }

    pub(super) fn validate_turn_count(&self, turns: usize) -> Result<()> {
        ensure!(self.runs.len() == turns, "unexpected run count");
        for kind in ["run_input_committed", "run_output_committed"] {
            ensure!(
                self.session
                    .iter()
                    .filter(|record| record["payload"]["type"] == kind)
                    .count()
                    == turns,
                "expected {turns} {kind} records"
            );
        }
        Ok(())
    }
}

pub(super) fn validate_answer(text: &str, expected: &Value) -> Result<()> {
    let fields = expected
        .as_object()
        .context("expected semantic fields must be an object")?;
    for (key, value) in fields {
        let needle = match value {
            Value::String(value) => value.clone(),
            Value::Number(value) => value.to_string(),
            _ => anyhow::bail!("unsupported semantic field {key}"),
        };
        let token_character = |c: char| c.is_alphanumeric() || c == '_' || c == '-';
        let matched = text.match_indices(&needle).any(|(start, _)| {
            !text[..start]
                .chars()
                .next_back()
                .is_some_and(token_character)
                && !text[start + needle.len()..]
                    .chars()
                    .next()
                    .is_some_and(token_character)
        });
        ensure!(
            matched,
            "assistant omitted or changed exact {key} value {needle:?}: {text}"
        );
    }
    Ok(())
}
