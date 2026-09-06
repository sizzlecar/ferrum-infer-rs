//! User stops are checked at their first observed channel occurrence. A thought
//! stop cannot substitute for the separately required visible-answer stop.
use super::{capture_run, chat, observation, request, run_deltas, Input, Run, RunCaptureMode};
use crate::{process::Server, write_json, Args};
use anyhow::{ensure, Context, Result};
use ferrum_bench_core::release_regression::model_stop::{
    select_stop_boundary, verify_stop_observations, StopChannel, StopMode,
};
use ferrum_bench_core::release_regression::model_tasks::verify_stop_raw_observations;
use serde_json::{json, Value};

fn inputs(args: &Args, mode: StopMode) -> Value {
    json!({"prompt": args.stop_prompt, "temperature": 0, "seed": 7,
        "max_tokens": args.max_tokens, "runtime_capacity": args.runtime_capacity(),
        "enable_thinking": if args.disable_thinking || mode == StopMode::DisabledThinking {
            Some(false)
        } else { None }})
}

fn run_observation(run: &Run) -> Result<Value> {
    ensure!(
        run.assistants.len() == 1,
        "stop probe requires one assistant response"
    );
    let assistant = &run.assistants[0];
    let raw = run_deltas(&run.records, assistant, false)?;
    let mut output = json!({"ready": run.ready,
        "message": {"role": "assistant", "content": assistant["content"],
            "reasoning": assistant["reasoning"], "tool_calls": assistant["tool_calls"]},
        "finish_reason": assistant["finish_reason"], "usage": assistant["usage"],
        "raw_availability": if raw.is_some() { "captured" } else { "unavailable" },
        "raw_text": raw, "raw_text_sha256": assistant["raw_text_sha256"]});
    if let Some(alias) = assistant.get("reasoning_content") {
        output["message"]["reasoning_content"] = alias.clone();
    }
    Ok(output)
}

fn http_observation(response: &super::Chat) -> Value {
    let mut output = observation(response);
    output["raw_availability"] = json!("not_exposed");
    output
}

fn suffix(mode: StopMode) -> &'static str {
    match mode {
        StopMode::TaskDefault => "default",
        StopMode::DisabledThinking => "disabled-thinking",
    }
}

pub(crate) async fn run_stop(args: &Args) -> Result<Value> {
    let mut probes = Vec::new();
    for mode in [StopMode::TaskDefault, StopMode::DisabledThinking] {
        ensure!(
            mode != StopMode::DisabledThinking || !args.disable_thinking,
            "final-channel stop uncovered: task already disables thinking but no final boundary was available"
        );
        let capture_mode = RunCaptureMode::Stop {
            disable_thinking: mode == StopMode::DisabledThinking,
        };
        let name = format!("run-stop-{}", suffix(mode));
        let baseline = run_observation(
            &capture_run(
                args,
                &format!("{name}-baseline"),
                Input::Prompt(&args.stop_prompt),
                None,
                capture_mode,
            )
            .await?,
        )?;
        let boundary = select_stop_boundary(&baseline).map_err(anyhow::Error::msg)?;
        ensure!(
            mode != StopMode::DisabledThinking || boundary.channel == StopChannel::Final,
            "final-channel stop uncovered: disabling thinking did not produce a distinct final boundary"
        );
        let output = run_observation(
            &capture_run(
                args,
                &format!("{name}-replay"),
                Input::Prompt(&args.stop_prompt),
                Some(&boundary.stop),
                capture_mode,
            )
            .await?,
        )?;
        verify_stop_observations(&baseline, &output, &boundary, args.max_tokens)
            .map_err(anyhow::Error::msg)?;
        verify_stop_raw_observations(&baseline, &output, &boundary).map_err(anyhow::Error::msg)?;
        let ready = output["ready"].clone();
        probes.push(
            json!({"mode": mode, "inputs": inputs(args, mode), "boundary": boundary,
            "baseline": baseline, "outputs": {"run": output}}),
        );
        write_json(args.report_dir.join("run-stop-probes.json"), &probes)?;
        if boundary.channel == StopChannel::Final {
            return Ok(json!({"ready": ready, "probes": probes}));
        }
    }
    // Both modes must have observed their own completed stop; this is never a
    // fallback after a failed replay or permission to omit the final channel.
    anyhow::bail!("final-channel stop uncovered")
}

pub(crate) async fn serve_stop(server: &Server<'_>) -> Result<Value> {
    let args = server.args;
    let mut probes = Vec::new();
    for mode in [StopMode::TaskDefault, StopMode::DisabledThinking] {
        ensure!(
            mode != StopMode::DisabledThinking || !args.disable_thinking,
            "final-channel stop uncovered: task already disables thinking but no final boundary was available"
        );
        let name = format!("serve-stop-{}", suffix(mode));
        let mut body = request(
            server,
            vec![json!({"role": "user", "content": args.stop_prompt})],
        );
        if mode == StopMode::DisabledThinking {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
        let baseline = http_observation(
            &chat(server, &format!("{name}-baseline"), body.clone(), false).await?,
        );
        let boundary = select_stop_boundary(&baseline).map_err(anyhow::Error::msg)?;
        ensure!(
            mode != StopMode::DisabledThinking || boundary.channel == StopChannel::Final,
            "final-channel stop uncovered: disabling thinking did not produce a distinct final boundary"
        );
        body["stop"] = json!([boundary.stop]);
        let mut outputs = json!({});
        for (entrypoint, stream) in [("sync", false), ("stream", true)] {
            let output = http_observation(
                &chat(
                    server,
                    &format!("{name}-{entrypoint}"),
                    body.clone(),
                    stream,
                )
                .await?,
            );
            verify_stop_observations(&baseline, &output, &boundary, args.max_tokens)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("{name}-{entrypoint}"))?;
            verify_stop_raw_observations(&baseline, &output, &boundary)
                .map_err(anyhow::Error::msg)?;
            outputs[entrypoint] = output;
        }
        probes.push(
            json!({"mode": mode, "inputs": inputs(args, mode), "boundary": boundary,
            "baseline": baseline, "outputs": outputs}),
        );
        write_json(args.report_dir.join("serve-stop-probes.json"), &probes)?;
        if boundary.channel == StopChannel::Final {
            return Ok(json!({"probes": probes}));
        }
    }
    anyhow::bail!("final-channel stop uncovered")
}

#[cfg(test)]
#[path = "stop_tests.rs"]
mod tests;
