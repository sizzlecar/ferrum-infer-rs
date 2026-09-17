use super::Args;
use anyhow::{ensure, Context, Result};
use ferrum_bench_core::release_regression::model_sources::pinned_hf_source;
use ferrum_types::KvStorageFormat;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;

pub(super) struct Generation {
    pub tokens: Vec<u32>,
    pub usage: Value,
}

pub(super) fn token_digest(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn backend_matches(actual: &str, requested: &str) -> bool {
    actual.eq_ignore_ascii_case(requested)
        || (requested == "cuda"
            && actual
                .strip_prefix("CUDA(")
                .and_then(|s| s.strip_suffix(')'))
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())))
}

pub(super) fn parse_generation(
    text: &str,
    args: &Args,
    require_tokens: bool,
) -> Result<Generation> {
    let records = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("parse complete JSONL stdout")?;
    let one = |event: &str| -> Result<&Value> {
        let values = records
            .iter()
            .filter(|r| r["event"] == event)
            .collect::<Vec<_>>();
        ensure!(values.len() == 1, "expected exactly one {event} record");
        Ok(values[0])
    };
    let ready = one("ready")?;
    let user = one("user")?;
    let assistant = one("assistant")?;
    let exit = one("exit")?;
    ensure!(
        ready["requested_model"] == args.model
            && backend_matches(ready["backend"].as_str().unwrap_or(""), &args.backend),
        "ready evidence differs from requested model/backend"
    );
    let session = ready["session_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .context("missing session ID")?;
    let request = user["request_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .context("missing request ID")?;
    ensure!(
        assistant["request_id"] == request && assistant["turn"] == 0 && user["turn"] == 0,
        "generation must be one request/turn"
    );
    ensure!(
        exit["reason"] == "one_shot_complete",
        "missing normal one-shot exit"
    );
    ensure!(
        assistant["finish_reason"] == "length",
        "teacher seed/capture must complete the entire requested length"
    );
    ensure!(
        records.first() == Some(ready) && records.last() == Some(exit),
        "invalid JSONL event order"
    );
    let mut terminal = false;
    let mut tokens = Vec::new();
    for record in &records {
        ensure!(
            record["schema_version"] == 2
                && record["session_id"] == session
                && record["history_epoch"] == 0,
            "JSONL schema/session/history changed"
        );
        match record["event"].as_str() {
            Some("assistant_delta") => {
                ensure!(
                    !terminal && record["request_id"] == request && record["turn"] == 0,
                    "delta outside the single active request"
                );
                if require_tokens {
                    ensure!(
                        record["index"].as_u64() == Some(tokens.len() as u64),
                        "missing or reordered token delta index"
                    );
                    let token = record["token_id"].as_u64().and_then(|n| u32::try_from(n).ok())
                        .context("delta lacks a u32 token_id; text reconstruction is not valid teacher evidence")?;
                    tokens.push(token);
                }
            }
            Some("assistant") => terminal = true,
            Some("ready" | "user" | "exit") => {}
            _ => anyhow::bail!("unexpected JSONL event: {}", record["event"]),
        }
    }
    let usage = &assistant["usage"];
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .filter(|n| *n > 0)
        .context("missing prompt usage")?;
    let completion = usage["completion_tokens"]
        .as_u64()
        .context("missing completion usage")?;
    ensure!(
        completion == args.max_tokens as u64 && assistant["n_tokens"] == completion,
        "generation was truncated or has incomplete token usage"
    );
    ensure!(
        prompt.checked_add(completion) == usage["total_tokens"].as_u64()
            && prompt + completion <= args.context_tokens as u64,
        "invalid or overflowing context usage"
    );
    if require_tokens {
        ensure!(
            tokens.len() as u64 == completion && assistant["chunk_count"] == completion,
            "JSONL token IDs do not cover all generated tokens; refusing a partial teacher history"
        );
    }
    Ok(Generation {
        tokens,
        usage: usage.clone(),
    })
}

fn nonempty(value: &Value) -> bool {
    value.as_str().is_some_and(|s| !s.is_empty())
}
fn digest(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

pub(super) fn validate_config(
    args: &Args,
    config: &Value,
    expected: KvStorageFormat,
) -> Result<()> {
    ensure!(
        config["backend"] == args.backend && config["attention_execution_policy"] == "portable",
        "executing backend/attention policy differs from explicit comparison settings"
    );
    for field in ["selected_max_model_len", "selected_kv_capacity"] {
        ensure!(
            config[field] == args.context_tokens,
            "actual {field} differs from requested context"
        );
    }
    ensure!(
        config["selected_max_sequences"] == 1,
        "actual sequence concurrency is not one"
    );
    let storage = &config["kv_storage"];
    ensure!(
        storage["source"] == "resolved_model_plan",
        "KV selection is not actual plan evidence"
    );
    for field in ["requested", "selected"] {
        let actual: KvStorageFormat =
            serde_json::from_value(storage[field].clone()).context("missing typed KV format")?;
        ensure!(
            actual == expected,
            "KV {field} differs from requested storage"
        );
        let actual: KvStorageFormat = serde_json::from_value(
            config["numerical_execution"][format!("{field}_kv_storage")].clone(),
        )
        .context("missing numerical resolution storage")?;
        ensure!(
            actual == expected,
            "numerical resolution KV {field} differs"
        );
    }
    ensure!(
        nonempty(&storage["numerical_profile"])
            && storage["numerical_profile"] == config["numerical_execution"]["selected_profile"],
        "KV and numerical profile evidence disagree"
    );
    let source = &config["resolution_evidence"];
    ensure!(
        source["schema_version"] == 1 && source["requested_model"] == args.model,
        "missing actual model-source identity"
    );
    for role in ["weights", "semantic", "tokenizer"] {
        let resolved = &source["resolved_sources"][role];
        ensure!(
            nonempty(&resolved["canonical_location"]) && nonempty(&resolved["resolved_revision"]),
            "missing {role} source identity"
        );
        let files = resolved["files"]
            .as_array()
            .filter(|files| !files.is_empty())
            .context("missing source file inventory")?;
        let mut names = BTreeSet::new();
        for file in files {
            ensure!(
                nonempty(&file["relative_path"])
                    && names.insert(file["relative_path"].as_str().unwrap())
                    && file["size_bytes"].as_u64().is_some_and(|n| n > 0)
                    && digest(&file["sha256"]),
                "invalid {role} file fingerprint"
            );
        }
    }
    for binding in ["semantic_config", "tokenizer", "template"] {
        ensure!(
            nonempty(&source[binding]["source_file"])
                && digest(&source[binding]["container_sha256"]),
            "missing selected {binding} artifact identity"
        );
    }
    if let Some((repository, revision)) =
        pinned_hf_source(&args.model).map_err(anyhow::Error::msg)?
    {
        let original = &source["original_sources"]["weights"];
        let resolved = &source["resolved_sources"]["weights"];
        ensure!(
            original["kind"] == "repository"
                && original["location"] == repository
                && original["requested_revision"]
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case(revision))
                && resolved["canonical_location"] == repository
                && resolved["resolved_revision"]
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case(revision)),
            "weights do not resolve to the requested immutable HF revision"
        );
    } else {
        let original = &source["original_sources"]["weights"];
        ensure!(
            matches!(
                original["kind"].as_str(),
                Some("local_directory" | "local_file")
            ) && original["location"] == args.model,
            "weights differ from the explicit local source"
        );
    }
    if let Some(filename) = &args.gguf_file {
        let files = source["resolved_sources"]["weights"]["files"]
            .as_array()
            .unwrap();
        ensure!(
            files.len() == 1 && files[0]["relative_path"] == *filename,
            "selected GGUF artifact differs"
        );
    }
    for (role, requested) in [
        ("semantic", &args.semantic_source),
        ("tokenizer", &args.tokenizer_source),
    ] {
        if let Some(path) = requested {
            ensure!(
                source["original_sources"][role]["location"].as_str() == path.to_str(),
                "{role} metadata differs from the explicit path"
            );
        }
    }
    Ok(())
}

pub(super) fn read_config(args: &Args, name: &str, expected: KvStorageFormat) -> Result<Value> {
    let path = args
        .report_dir
        .join(format!("{name}.effective-config.json"));
    let value: Value = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )?;
    validate_config(args, &value, expected)?;
    Ok(value)
}
