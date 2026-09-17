//! Compare the actual initial model inputs from product request-dump bundles.
use super::*;
use sha2::{Digest, Sha256};

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("read request evidence {}", path.display()))
}

fn first_request_bundle(directory: &Path) -> Result<(PathBuf, Value)> {
    let mut candidates = Vec::new();
    // Only inspect this case's newly created product dump directory. UUID
    // filenames and filesystem iteration order do not establish request order.
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        ensure!(entry.file_type()?.is_dir(), "unexpected request-dump entry");
        let bundle = entry.path();
        let request = read_json(&bundle.join("request.json"))?;
        ensure!(
            request["schema_version"] == ferrum_types::OBSERVABILITY_PROFILE_SCHEMA_VERSION
                && request["entrypoint"] == "serve"
                && request["method"] == "POST"
                && request["endpoint"] == "/v1/chat/completions"
                && request["request_id"].as_str()
                    == bundle.file_name().and_then(|name| name.to_str()),
            "unrecognized request-dump schema or identity"
        );
        let body = read_json(&bundle.join("replay_body.json"))?;
        ensure!(request["http"]["body"] == body, "inconsistent replay body");
        let messages = body["messages"]
            .as_array()
            .context("replay body has no messages")?;
        // With compaction/retries disabled, only the first request has one
        // user message and no generated assistant/tool history. Reject zero
        // or multiple matches instead of guessing by timestamps or names.
        if messages.iter().filter(|m| m["role"] == "user").count() == 1
            && messages
                .iter()
                .all(|m| matches!(m["role"].as_str(), Some("system" | "user")))
        {
            candidates.push((bundle, body));
        }
    }
    ensure!(
        candidates.len() == 1,
        "expected exactly one initial model request, found {}",
        candidates.len()
    );
    Ok(candidates.remove(0))
}

fn input_evidence(directory: &Path) -> Result<(Value, Value)> {
    let (bundle, body) = first_request_bundle(directory)?;
    let request_id = bundle.file_name().and_then(|name| name.to_str()).unwrap();
    let tokens = read_json(&bundle.join("prompt_token_ids.json"))?;
    let sampling = read_json(&bundle.join("sampling_params.json"))?;
    for evidence in [&tokens, &sampling] {
        ensure!(
            evidence["schema_version"] == ferrum_types::OBSERVABILITY_PROFILE_SCHEMA_VERSION
                && evidence["request_id"] == request_id
                && evidence.get("unavailable_reason") == Some(&Value::Null),
            "incomplete or mismatched initial request evidence"
        );
    }
    let token_ids: Vec<u32> = serde_json::from_value(tokens["token_ids"].clone())
        .context("initial prompt token IDs must be complete u32 values")?;
    ensure!(
        !token_ids.is_empty() && tokens["token_count"].as_u64() == Some(token_ids.len() as u64),
        "incomplete initial prompt token inventory"
    );
    ensure!(
        sampling["sampling_params"].is_object(),
        "missing effective sampling parameters"
    );
    let mut digest = Sha256::new();
    for token in &token_ids {
        digest.update(token.to_le_bytes());
    }
    let evidence = json!({
        "bundle": bundle,
        "prompt_token_count": token_ids.len(),
        "prompt_token_ids_sha256_u32_le": format!("{:x}", digest.finalize()),
        "replay_body_sha256": format!("{:x}", Sha256::digest(fs::read(bundle.join("replay_body.json"))?)),
        "sampling_params_sha256": format!("{:x}", Sha256::digest(serde_json::to_vec(&sampling["sampling_params"])?))
    });
    // The body intentionally redacts content. Complete actual prompt tokens
    // also compare system/workspace text, the rendered template and tool schema.
    Ok((
        json!({"body": body, "tokens": token_ids, "sampling": sampling["sampling_params"]}),
        evidence,
    ))
}

pub(super) fn compare_first_requests(directory: &Path) -> Result<Value> {
    let (reference, reference_evidence) = input_evidence(&directory.join("fp16/model-requests"))?;
    let (candidate, candidate_evidence) = input_evidence(&directory.join("int8/model-requests"))?;
    Ok(json!({
        "equal": reference == candidate,
        "fp16": reference_evidence,
        "int8": candidate_evidence,
        "scope": "initial actual prompt token IDs, sanitized request and effective sampling; later generated histories may diverge"
    }))
}
