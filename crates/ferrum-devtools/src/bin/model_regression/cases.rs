#[path = "boundaries.rs"]
mod boundaries;
#[path = "stop.rs"]
mod stop;
pub(super) use boundaries::{run_length, run_reasoning, serve_length, serve_reasoning};
pub(super) use stop::{run_stop, serve_stop};

use super::process::{self, Server};
use super::protocol::{self, answer, Chat};
use super::{identity, Args};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Duration;

use ferrum_bench_core::release_regression::model_basic::{
    verify_basic_run, verify_basic_serve, ARITHMETIC_PROMPT, MEMORY_PROMPT, RECALL_PROMPT,
};
use ferrum_bench_core::release_regression::model_tool::{verify_tool_case, ValidatedCalcCall};

fn observation(chat: &Chat) -> Value {
    json!({"message": chat.message, "finish_reason": chat.finish, "usage": chat.usage})
}

enum Input<'a> {
    Repl(&'a str),
    Prompt(&'a str),
}

struct Run {
    ready: Value,
    records: Vec<Value>,
    assistants: Vec<Value>,
}

fn run_deltas(
    records: &[Value],
    assistant: &Value,
    require_deltas: bool,
) -> Result<Option<String>> {
    let request_id = assistant["request_id"]
        .as_str()
        .filter(|id| !id.is_empty())
        .context("run assistant missing request id")?;
    let mut text = String::new();
    let mut saw_delta = false;
    for record in records
        .iter()
        .filter(|record| record["event"] == "assistant_delta" && record["request_id"] == request_id)
    {
        saw_delta = true;
        text.push_str(
            record["raw_text_delta"]
                .as_str()
                .context("run delta missing raw text")?,
        );
    }
    if !saw_delta {
        // Buffered protocols such as Harmony legitimately emit no raw deltas.
        ensure!(
            !require_deltas,
            "run stop replay lost all deltas emitted by its baseline"
        );
        return Ok(None);
    }
    // `run` hashes display_response_text(raw_text), which trims outer whitespace.
    let digest = format!("{:x}", Sha256::digest(text.trim().as_bytes()));
    ensure!(
        assistant["raw_text_sha256"].as_str() == Some(digest.as_str()),
        "run delta text disagrees with final raw text for request {request_id}"
    );
    Ok(Some(text))
}

#[derive(Clone, Copy)]
enum RunCaptureMode {
    Natural,
    Stop { disable_thinking: bool },
}

impl RunCaptureMode {
    fn accepts(self, finish: Option<&str>) -> bool {
        matches!(finish, Some("stop" | "eos"))
            || matches!((self, finish), (Self::Stop { .. }, Some("length")))
    }
}

async fn run_chat(args: &Args, name: &str, input: Input<'_>, stop: Option<&str>) -> Result<Run> {
    capture_run(args, name, input, stop, RunCaptureMode::Natural).await
}

async fn capture_run(
    args: &Args,
    name: &str,
    input: Input<'_>,
    stop: Option<&str>,
    mode: RunCaptureMode,
) -> Result<Run> {
    let mut argv = args.common_args("run");
    if let RunCaptureMode::Stop {
        disable_thinking: true,
    } = mode
    {
        if !argv.iter().any(|arg| arg == "--disable-thinking") {
            argv.push("--disable-thinking".into());
        }
    }
    argv.extend([
        "--output-format".into(),
        "jsonl".into(),
        "--temperature".into(),
        "0".into(),
        "--seed".into(),
        "7".into(),
        "--max-tokens".into(),
        args.max_tokens.to_string(),
    ]);
    let stdin = match input {
        Input::Repl(text) => Some(text),
        Input::Prompt(text) => {
            argv.extend(["--prompt".into(), text.into()]);
            None
        }
    };
    if let Some(stop) = stop {
        argv.extend(["--stop".into(), stop.into()]);
    }
    let stdout = process::run(
        args,
        name,
        argv,
        stdin,
        Duration::from_secs(args.run_timeout_secs),
    )
    .await?;
    let records = protocol::run_records(&stdout)?;
    let ready = records
        .iter()
        .find(|record| record["event"] == "ready")
        .context("missing run ready event")?
        .clone();
    identity::validate_run(args, &ready)?;
    let assistants: Vec<_> = records
        .iter()
        .filter(|record| record["event"] == "assistant")
        .cloned()
        .collect();
    for assistant in &assistants {
        ensure!(
            mode.accepts(assistant["finish_reason"].as_str()),
            "run finish reason is invalid for this probe: {assistant}"
        );
        ensure!(
            assistant["usage"]["completion_tokens"]
                .as_u64()
                .is_some_and(|tokens| tokens > 0),
            "run missing completion usage"
        );
        run_deltas(&records, assistant, false)?;
    }
    Ok(Run {
        ready,
        records,
        assistants,
    })
}

pub(super) async fn run_basic(args: &Args) -> Result<Value> {
    let prompts = [MEMORY_PROMPT, ARITHMETIC_PROMPT, RECALL_PROMPT];
    let stdin = format!("{}\n/bye\n", prompts.join("\n"));
    let run = run_chat(args, "run-basic", Input::Repl(&stdin), None).await?;
    let protocol = serde_json::from_value(run.ready["reasoning_protocol"].clone())
        .context("missing run reasoning capability")?;
    let evidence = json!({"ready": run.ready, "prompts": prompts, "answers": run.assistants,
        "source_identity": identity::source_evidence(args, "run-basic")?});
    verify_basic_run(&evidence, protocol, args.max_tokens).map_err(anyhow::Error::msg)?;
    Ok(evidence)
}

fn request(server: &Server<'_>, messages: Vec<Value>) -> Value {
    json!({"model": "regression-model", "messages": messages, "temperature": 0.0, "seed": 7, "max_tokens": server.args.max_tokens})
}

async fn chat(server: &Server<'_>, name: &str, mut body: Value, stream: bool) -> Result<Chat> {
    body["stream"] = json!(stream);
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    let text = server.request(name, &body).await?;
    if stream {
        protocol::stream(&text)
    } else {
        protocol::sync(&text)
    }
}

fn finished_answer(chat: &Chat, expected: &str) -> Result<()> {
    ensure!(
        chat.finish == "stop",
        "answer did not terminate normally: {}",
        chat.finish
    );
    answer(chat.content(), expected)
}

pub(super) async fn serve_basic(server: &Server<'_>) -> Result<Value> {
    let models = server.models().await?;
    let first_user = json!({"role": "user", "content": MEMORY_PROMPT});
    let memory_body = request(server, vec![first_user.clone()]);
    let memory_write = chat(server, "serve-basic-memory", memory_body.clone(), false).await?;
    finished_answer(&memory_write, "OK").context("memory acknowledgement")?;
    let arithmetic_body = request(
        server,
        vec![
            first_user,
            memory_write.message.clone(),
            json!({"role": "user", "content": ARITHMETIC_PROMPT}),
        ],
    );
    let first = chat(server, "serve-basic-sync", arithmetic_body.clone(), false).await?;
    finished_answer(&first, "42").context("sync arithmetic")?;
    let streamed = chat(server, "serve-basic-stream", arithmetic_body.clone(), true).await?;
    finished_answer(&streamed, "42").context("streamed arithmetic")?;
    let recall_body = |arithmetic: &Chat| -> Result<Value> {
        let mut messages = arithmetic_body["messages"]
            .as_array()
            .context("arithmetic request messages")?
            .clone();
        // Each wire mode replays its own actual answer and reasoning.
        messages.push(arithmetic.message.clone());
        messages.push(json!({"role": "user", "content": RECALL_PROMPT}));
        Ok(request(server, messages))
    };
    let sync_history = recall_body(&first)?;
    let stream_history = recall_body(&streamed)?;
    let recall = chat(server, "serve-basic-recall", sync_history.clone(), false).await?;
    finished_answer(&recall, "cobalt-731").context("actual HTTP history recall")?;
    let streamed_recall = chat(
        server,
        "serve-basic-recall-stream",
        stream_history.clone(),
        true,
    )
    .await?;
    finished_answer(&streamed_recall, "cobalt-731").context("actual SSE history recall")?;
    let evidence = json!({"observations": {
        "memory_write": observation(&memory_write),
        "sync": observation(&first), "stream": observation(&streamed),
        "recall": observation(&recall), "stream_recall": observation(&streamed_recall)},
        "requests": {"memory_write": memory_body["messages"],
        "sync": arithmetic_body["messages"], "stream": arithmetic_body["messages"],
        "recall": sync_history["messages"], "stream_recall": stream_history["messages"]},
        "models": models});
    let protocol = serde_json::from_value(server.health["reasoning_protocol"].clone())
        .context("missing serve reasoning capability")?;
    verify_basic_serve(&evidence, protocol, server.args.max_tokens).map_err(anyhow::Error::msg)?;
    Ok(evidence)
}

pub(super) async fn structured(server: &Server<'_>) -> Result<Value> {
    let mut body = request(
        server,
        vec![
            json!({"role": "user", "content": "What is 17 + 25? Return a JSON object with the integer result in the answer field."}),
        ],
    );
    body["response_format"] = json!({"type": "json_schema", "json_schema": {
        "name": "ArithmeticAnswer", "strict": true,
        "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": false}
    }});
    let mut outputs = Vec::new();
    for (name, stream) in [("structured-sync", false), ("structured-stream", true)] {
        let result = chat(server, name, body.clone(), stream).await?;
        ensure!(
            result.finish == "stop",
            "strict JSON exhausted output budget"
        );
        let value: Value =
            serde_json::from_str(result.content()).context("invalid strict JSON content")?;
        ensure!(
            value.as_object().is_some_and(|object| object.len() == 1)
                && value["answer"].as_i64() == Some(42),
            "strict JSON failed schema or arithmetic semantics: {value}"
        );
        outputs.push(value);
    }
    Ok(json!({"answers": outputs}))
}

pub(super) async fn tools(server: &Server<'_>) -> Result<Value> {
    let user = json!({"role": "user", "content": "Use the calc tool to evaluate 123+456. After receiving its result, reply with only the resulting number."});
    let declarations = json!([
        {"type": "function", "function": {
            "name": "calc", "description": "Evaluate an arithmetic expression.",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": false}
        }},
        {"type": "function", "function": {
            "name": "lookup_weather", "description": "Look up current weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": false}
        }}
    ]);
    let mut body = request(server, vec![user.clone()]);
    body["tools"] = declarations.clone();
    body["tool_choice"] = json!({"type": "function", "function": {"name": "calc"}});
    // Reuse the loaded server; each HTTP mode must actually select and hand off
    // the named tool from the same declarations, including a distractor.
    let sync_called = chat(server, "tools-call-sync", body.clone(), false).await?;
    let sync_call = protocol::calc_call(&sync_called, server.args.max_tokens)
        .context("sync named tool handoff")?;
    let stream_called = chat(server, "tools-call-stream", body, true).await?;
    let stream_call = protocol::calc_call(&stream_called, server.args.max_tokens)
        .context("SSE named tool handoff")?;
    // Only execute this typed local fixture. Model text is never shell code.
    let continuation = |called: &Chat, call: &ValidatedCalcCall| {
        let result_message = json!({"role": "tool", "tool_call_id": call.call["id"], "content": json!({"result": call.result}).to_string()});
        let mut body = request(
            server,
            vec![user.clone(), called.message.clone(), result_message],
        );
        body["tools"] = declarations.clone();
        body["tool_choice"] = json!("none");
        body
    };
    let sync_body = continuation(&sync_called, &sync_call);
    let canonical = chat(server, "tools-final-sync", sync_body.clone(), false).await?;
    finished_answer(&canonical, &sync_call.result.to_string())
        .context("canonical tool-result replay")?;
    let mut final_body = continuation(&stream_called, &stream_call);
    if server.args.reasoning_alias_replay {
        let assistant = final_body["messages"][1]
            .as_object_mut()
            .context("assistant history object")?;
        let reasoning = assistant
            .remove("reasoning")
            .context("alias replay requires actual reasoning in the model's tool call")?;
        ensure!(reasoning.as_str().is_some_and(|text| !text.trim().is_empty()), "alias replay requires nonempty actual tool-call reasoning; keep model thinking enabled");
        assistant.insert("reasoning_content".into(), reasoning);
    }
    let streamed = chat(server, "tools-final-stream", final_body.clone(), true).await?;
    finished_answer(&streamed, &stream_call.result.to_string())
        .context("streamed tool-result replay")?;
    let evidence = json!({
        "sync_call": {"message": sync_called.message, "finish_reason": sync_called.finish, "usage": sync_called.usage},
        "stream_call": {"message": stream_called.message, "finish_reason": stream_called.finish, "usage": stream_called.usage},
        "sync_continuation": {"message": canonical.message, "finish_reason": canonical.finish, "usage": canonical.usage,
            "tool_call_id": sync_call.call["id"], "replayed_assistant": sync_body["messages"][1], "tool_result_message": sync_body["messages"][2]},
        "stream_continuation": {"message": streamed.message, "finish_reason": streamed.finish, "usage": streamed.usage,
            "tool_call_id": stream_call.call["id"], "replayed_assistant": final_body["messages"][1], "tool_result_message": final_body["messages"][2]},
        "tool_result": sync_call.result,
        "reasoning_alias_replayed": server.args.reasoning_alias_replay
    });
    verify_tool_case(
        &evidence,
        server.args.max_tokens,
        server.args.reasoning_alias_replay,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(evidence)
}

pub(super) async fn auto_tools_json(server: &Server<'_>) -> Result<Value> {
    use ferrum_bench_core::release_regression::model_tool::{
        auto_tools_json_controls, verify_auto_tools_json_case, AUTO_TOOLS_JSON_PROMPT,
    };
    let mut evidence = json!({});
    for (mode, stream) in [("sync", false), ("stream", true)] {
        let mut first = request(
            server,
            vec![json!({"role": "user", "content": AUTO_TOOLS_JSON_PROMPT})],
        );
        first
            .as_object_mut()
            .unwrap()
            .extend(auto_tools_json_controls().as_object().unwrap().clone());
        first["stream"] = json!(stream);
        if stream {
            first["stream_options"] = json!({"include_usage": true});
        }
        let called = chat(
            server,
            &format!("auto-tools-json-call-{mode}"),
            first.clone(),
            stream,
        )
        .await?;
        let calculation = protocol::calc_call(&called, server.args.max_tokens)
            .context("automatic calculator handoff")?;
        // The sole executed operation is the validated local integer fixture.
        // Replay the actual model message, including its reasoning and call ID.
        let mut continuation = first.clone();
        let messages = continuation["messages"].as_array_mut().unwrap();
        messages.push(called.message.clone());
        messages.push(json!({
            "role": "tool", "tool_call_id": calculation.call["id"],
            "content": json!({"result": calculation.result}).to_string(),
        }));
        let completed = chat(
            server,
            &format!("auto-tools-json-final-{mode}"),
            continuation.clone(),
            stream,
        )
        .await?;
        evidence[mode] = json!({
            "call_request": first, "call": observation(&called),
            "continuation_request": continuation, "continuation": observation(&completed),
        });
    }
    verify_auto_tools_json_case(&evidence, server.args.max_tokens).map_err(anyhow::Error::msg)?;
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(request_id: &str, raw: &str) -> Value {
        json!({
            "event": "assistant", "request_id": request_id,
            "raw_text_sha256": format!("{:x}", Sha256::digest(raw.trim().as_bytes()))
        })
    }

    #[test]
    fn run_delta_hash_covers_tokenless_tail_and_separates_requests() {
        let records = vec![
            json!({"event": "assistant_delta", "request_id": "previous", "raw_text_delta": "unrelated"}),
            json!({"event": "assistant_delta", "request_id": "current", "raw_text_delta": " alpha "}),
            json!({"event": "assistant_delta", "request_id": "current", "raw_text_delta": "尾 ", "token_id": null}),
        ];
        let final_record = assistant("current", "alpha 尾");
        assert_eq!(
            run_deltas(&records, &final_record, false)
                .unwrap()
                .as_deref(),
            Some(" alpha 尾 ")
        );
        assert!(run_deltas(&records[..2], &final_record, false).is_err());
    }

    #[test]
    fn run_delta_absence_is_allowed_only_when_baseline_did_not_stream() {
        let final_record = assistant("buffered", "analysis and final text");
        assert!(run_deltas(&[], &final_record, false).unwrap().is_none());
        assert!(run_deltas(&[], &final_record, true).is_err());
    }
}
