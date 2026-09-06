//! Replay the stop oracle against recorded observations, including raw JSONL
//! bytes when the product actually exposes them. HTTP fields are not raw text.
use super::model_stop::{verify_stop_observations, StopBoundary, StopChannel, StopMode};
use super::model_tasks::{backend_name, run_backend, ExpectedModelRun};
use ferrum_types::ModelOutputProtocol;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn verify_stop_raw_observations(
    baseline: &Value,
    output: &Value,
    boundary: &StopBoundary,
) -> Result<(), String> {
    let availability = baseline["raw_availability"]
        .as_str()
        .ok_or("missing raw availability")?;
    if output["raw_availability"].as_str() != Some(availability) {
        return Err("stop replay lost or changed its raw evidence availability".into());
    }
    match availability {
        "captured" => {
            let full = baseline["raw_text"]
                .as_str()
                .ok_or("missing captured baseline raw text")?;
            let partial = output["raw_text"]
                .as_str()
                .ok_or("missing captured stop raw text")?;
            for (value, text) in [(baseline, full), (output, partial)] {
                let hash = format!("{:x}", Sha256::digest(text.trim().as_bytes()));
                if value["raw_text_sha256"].as_str() != Some(hash.as_str()) {
                    return Err("stop raw text disagrees with its assistant hash".into());
                }
            }
            let position = full
                .find(&boundary.stop)
                .ok_or("selected stop is missing from baseline raw text")?;
            if position == 0 || partial != &full[..position] || partial.contains(&boundary.stop) {
                return Err(
                    "stop replay is not the exact raw prefix before its first sentinel".into(),
                );
            }
        }
        "unavailable" | "not_exposed" => {
            if !baseline["raw_text"].is_null() || !output["raw_text"].is_null() {
                return Err("unavailable raw text must not be reconstructed from channels".into());
            }
            if availability == "unavailable" {
                for value in [baseline, output] {
                    if !value["raw_text_sha256"]
                        .as_str()
                        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                    {
                        return Err(
                            "buffered run observation is missing its actual raw hash".into()
                        );
                    }
                }
            }
        }
        _ => return Err("unknown stop raw evidence availability".into()),
    }
    Ok(())
}

pub(super) fn verify_case(
    expected: &ExpectedModelRun,
    evidence: &Value,
    is_run: bool,
) -> Result<(), String> {
    let probes = evidence["probes"]
        .as_array()
        .ok_or("stop case is missing channel observations")?;
    if probes.is_empty() || probes.len() > 2 {
        return Err(
            "stop case must contain its default probe and only a necessary final probe".into(),
        );
    }
    let mut channels = Vec::new();
    for (index, probe) in probes.iter().enumerate() {
        let mode: StopMode =
            serde_json::from_value(probe["mode"].clone()).map_err(|e| e.to_string())?;
        if mode
            != if index == 0 {
                StopMode::TaskDefault
            } else {
                StopMode::DisabledThinking
            }
            || (index == 1 && expected.disable_thinking)
        {
            return Err("stop probe mode does not match its required execution order".into());
        }
        let thinking = if expected.disable_thinking || mode == StopMode::DisabledThinking {
            Some(false)
        } else {
            None
        };
        if probe["inputs"]
            != json!({"prompt": expected.stop_prompt, "temperature": 0,
            "seed": 7, "max_tokens": expected.max_tokens, "runtime_capacity": expected.runtime_capacity, "enable_thinking": thinking})
        {
            return Err("stop probe inputs differ from the fixed task and recorded mode".into());
        }
        let boundary: StopBoundary =
            serde_json::from_value(probe["boundary"].clone()).map_err(|e| e.to_string())?;
        let baseline = &probe["baseline"];
        let outputs = probe["outputs"]
            .as_object()
            .ok_or("stop probe is missing outputs")?;
        let names: &[&str] = if is_run {
            &["run"]
        } else {
            &["sync", "stream"]
        };
        if outputs.len() != names.len() || names.iter().any(|name| !outputs.contains_key(*name)) {
            return Err("stop probe is missing a required product entrypoint observation".into());
        }
        for value in std::iter::once(baseline).chain(outputs.values()) {
            if is_run {
                let ready = &value["ready"];
                if ready["event"] != "ready"
                    || ready["requested_model"].as_str() != Some(expected.profile.model.as_str())
                    || run_backend(&ready["backend"]) != Some(expected.profile.target.backend)
                {
                    return Err(format!(
                        "stop observation is not the selected model/{} backend",
                        backend_name(expected.profile.target.backend)
                    ));
                }
                let availability = value["raw_availability"].as_str();
                if availability != Some("captured")
                    && !(availability == Some("unavailable")
                        && expected.profile.target.protocol == ModelOutputProtocol::HarmonyGptOss)
                {
                    return Err(
                        "run stop is missing its protocol's actual raw delta evidence".into(),
                    );
                }
            } else if value["raw_availability"] != "not_exposed" {
                return Err("HTTP stop observations must not claim model raw output".into());
            }
        }
        for output in outputs.values() {
            verify_stop_observations(baseline, output, &boundary, expected.max_tokens)?;
            verify_stop_raw_observations(baseline, output, &boundary)?;
        }
        channels.push(boundary.channel);
    }
    if channels.last() != Some(&StopChannel::Final)
        || (channels.len() == 2 && channels[0] != StopChannel::Reasoning)
    {
        return Err(
            "final-channel stop uncovered; a reasoning stop cannot substitute for it".into(),
        );
    }
    Ok(())
}
