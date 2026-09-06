//! Observable reasoning and truncation probes. HTTP shares the existing server;
//! run needs a separate process for each output-budget configuration.
use super::{chat, request, run_deltas};
use crate::{identity, process, protocol, Args};
use anyhow::{ensure, Context, Result};
use ferrum_bench_core::release_regression::model_tasks::{
    length_probe_budget, verify_length_observations, verify_reasoning_observation,
};
use serde_json::{json, Value};
use std::time::Duration;

const REASONING_PROMPT: &str = "Think through the addition of 17 and 25 privately. In your final answer reply with only the number.";
const LENGTH_PROMPT: &str = "Repeat exactly this sentence, without quotes or explanation: The quiet river flows past the old bridge while the bright morning sun rises above the green hills and the sleeping village.";

fn observed(chat: protocol::Chat) -> Value {
    json!({"message": chat.message, "finish_reason": chat.finish, "usage": chat.usage})
}

fn within_budget(output: &Value, max_tokens: u32) -> Result<()> {
    ensure!(
        output["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0 && n <= u64::from(max_tokens)),
        "observed completion usage exceeds the requested budget or is missing"
    );
    Ok(())
}

fn run_argv(args: &Args, prompt: &str, thinking: bool, max_tokens: u32) -> Vec<String> {
    let mut argv = args.common_args("run");
    // These dedicated probes have explicit modes, independent of the Quick
    // Start basic cases. Do not mutate Args or the environment of other probes.
    argv.retain(|argument| argument != "--disable-thinking");
    argv.extend([
        if thinking {
            "--enable-thinking"
        } else {
            "--disable-thinking"
        }
        .into(),
        "--output-format".into(),
        "jsonl".into(),
        "--temperature".into(),
        "0".into(),
        "--seed".into(),
        "7".into(),
        "--max-tokens".into(),
        max_tokens.to_string(),
        "--prompt".into(),
        prompt.into(),
    ]);
    argv
}

async fn run_once(
    args: &Args,
    name: &str,
    prompt: &str,
    thinking: bool,
    budget: u32,
) -> Result<(Value, Value)> {
    let stdout = process::run(
        args,
        name,
        run_argv(args, prompt, thinking, budget),
        None,
        Duration::from_secs(args.run_timeout_secs),
    )
    .await?;
    let records = protocol::run_records(&stdout)?;
    let ready = records
        .iter()
        .find(|record| record["event"] == "ready")
        .context("missing ready")?
        .clone();
    identity::validate_run(args, &ready)?;
    let assistants: Vec<_> = records
        .iter()
        .filter(|record| record["event"] == "assistant")
        .collect();
    ensure!(
        assistants.len() == 1,
        "boundary probe requires one actual assistant response"
    );
    let assistant = assistants[0];
    // JSONL raw deltas intentionally retain protocol/reasoning evidence. Check
    // their hash, while visible-channel checks consume parsed assistant content.
    run_deltas(&records, assistant, false)?;
    let mut message = json!({"role": "assistant", "content": assistant["content"], "reasoning": assistant["reasoning"]});
    if let Some(alias) = assistant.get("reasoning_content") {
        message["reasoning_content"] = alias.clone();
    }
    if let Some(calls) = assistant.get("tool_calls") {
        message["tool_calls"] = calls.clone();
    }
    let output = json!({"message": message, "finish_reason": assistant["finish_reason"], "usage": assistant["usage"]});
    within_budget(&output, budget)?;
    Ok((ready, output))
}

pub(crate) async fn run_reasoning(args: &Args) -> Result<Value> {
    let (ready, output) = run_once(
        args,
        "run-reasoning",
        REASONING_PROMPT,
        true,
        args.max_tokens,
    )
    .await?;
    let capability: ferrum_types::ModelReasoningProtocol =
        serde_json::from_value(ready["reasoning_protocol"].clone())
            .context("missing actual reasoning capability")?;
    ensure!(
        capability.supports_reasoning(),
        "loaded template does not support the reasoning positive probe"
    );
    verify_reasoning_observation(&output).map_err(anyhow::Error::msg)?;
    Ok(
        json!({"ready": ready, "enable_thinking": true, "max_tokens": args.max_tokens, "output": output}),
    )
}

pub(crate) async fn serve_reasoning(server: &process::Server<'_>) -> Result<Value> {
    let capability: ferrum_types::ModelReasoningProtocol =
        serde_json::from_value(server.health["reasoning_protocol"].clone())
            .context("missing actual reasoning capability")?;
    ensure!(
        capability.supports_reasoning(),
        "loaded template does not support the reasoning positive probe"
    );
    let mut body = request(
        server,
        vec![json!({"role": "user", "content": REASONING_PROMPT})],
    );
    body["chat_template_kwargs"] = json!({"enable_thinking": true});
    let mut evidence = json!({"enable_thinking": true, "max_tokens": server.args.max_tokens});
    for (mode, stream) in [("sync", false), ("stream", true)] {
        let output =
            observed(chat(server, &format!("reasoning-{mode}"), body.clone(), stream).await?);
        within_budget(&output, server.args.max_tokens)?;
        verify_reasoning_observation(&output).map_err(anyhow::Error::msg)?;
        evidence[mode] = output;
    }
    Ok(evidence)
}

pub(crate) async fn run_length(args: &Args) -> Result<Value> {
    let (baseline_ready, baseline) = run_once(
        args,
        "run-length-baseline",
        LENGTH_PROMPT,
        false,
        args.max_tokens,
    )
    .await?;
    let budget = length_probe_budget(
        baseline["usage"]["completion_tokens"]
            .as_u64()
            .context("baseline missing completion usage")?,
    )
    .map_err(anyhow::Error::msg)?;
    ensure!(
        matches!(baseline["finish_reason"].as_str(), Some("stop" | "eos")),
        "length baseline exhausted its budget"
    );
    let (ready, output) = run_once(args, "run-length-replay", LENGTH_PROMPT, false, budget).await?;
    verify_length_observations(&baseline, &output, budget).map_err(anyhow::Error::msg)?;
    Ok(
        json!({"ready": ready, "baseline_ready": baseline_ready, "enable_thinking": false,
        "baseline_max_tokens": args.max_tokens, "budget": budget, "baseline": baseline, "output": output}),
    )
}

pub(crate) async fn serve_length(server: &process::Server<'_>) -> Result<Value> {
    let mut body = request(
        server,
        vec![json!({"role": "user", "content": LENGTH_PROMPT})],
    );
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
    let baseline = observed(chat(server, "length-baseline", body.clone(), false).await?);
    within_budget(&baseline, server.args.max_tokens)?;
    ensure!(
        baseline["finish_reason"] == "stop",
        "length baseline exhausted its budget"
    );
    let budget = length_probe_budget(
        baseline["usage"]["completion_tokens"]
            .as_u64()
            .context("baseline missing completion usage")?,
    )
    .map_err(anyhow::Error::msg)?;
    body["max_tokens"] = json!(budget);
    let mut evidence = json!({"enable_thinking": false, "baseline_max_tokens": server.args.max_tokens,
        "budget": budget, "baseline": baseline});
    for (mode, stream) in [("sync", false), ("stream", true)] {
        let output = observed(chat(server, &format!("length-{mode}"), body.clone(), stream).await?);
        verify_length_observations(&evidence["baseline"], &output, budget)
            .map_err(anyhow::Error::msg)?;
        evidence[mode] = output;
    }
    Ok(evidence)
}

#[cfg(test)]
#[path = "boundary_tests.rs"]
mod tests;
