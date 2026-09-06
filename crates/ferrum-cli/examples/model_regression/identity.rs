//! Validate observed runtime identity before accepting model evidence.

use super::Args;
use anyhow::{ensure, Context, Result};
use serde_json::Value;

#[derive(Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
    Metal,
    Cuda,
}

fn backend(value: &str) -> Result<Backend> {
    match value {
        "cpu" | "CPU" => Ok(Backend::Cpu),
        "metal" | "Metal" => Ok(Backend::Metal),
        "cuda" => Ok(Backend::Cuda),
        _ => {
            // `run` records Device's Debug representation, including its index.
            let index = value
                .strip_prefix("CUDA(")
                .and_then(|value| value.strip_suffix(')'))
                .filter(|index| {
                    !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit())
                })
                .context("missing or unknown runtime backend")?;
            index
                .parse::<usize>()
                .context("invalid CUDA device index")?;
            Ok(Backend::Cuda)
        }
    }
}

fn validate_backend(args: &Args, actual: &Value) -> Result<()> {
    let actual_text = actual
        .as_str()
        .context("runtime backend must be a string")?;
    ensure!(
        backend(actual_text)? == backend(&args.backend)?,
        "runtime backend {actual_text:?} does not match expected {:?}",
        args.backend
    );
    Ok(())
}

pub(super) fn validate_run(args: &Args, ready: &Value) -> Result<()> {
    ensure!(ready["event"] == "ready", "missing run ready event");
    ensure!(
        ready["requested_model"].as_str() == Some(args.model.as_str()),
        "run requested_model does not match requested model {:?}",
        args.model
    );
    validate_backend(args, &ready["backend"]).context("run runtime identity")
}

pub(super) fn validate_serve(args: &Args, health: &Value) -> Result<()> {
    ensure!(
        health["status"] == "healthy",
        "serve health must report healthy status"
    );
    validate_backend(
        args,
        &health["auto_config"]["hardware_capabilities"]["backend"],
    )
    .context("serve runtime identity")
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
