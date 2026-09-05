use super::process::{self, Server};
use super::protocol::{self, answer, Chat};
use super::Args;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Duration;

const FIRST_TURN: &str =
    "Remember the code cobalt-731. What is 17 + 25? Reply with only the number.";
const SECOND_TURN: &str = "What code did I ask you to remember? Reply with only that code.";
const STOP_PROMPT: &str =
    "Write exactly this text, without quotes or explanation: alpha beta gamma delta epsilon.";

enum Input<'a> {
    Repl(&'a str),
    Prompt(&'a str),
}

struct Run {
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

async fn run_chat(args: &Args, name: &str, input: Input<'_>, stop: Option<&str>) -> Result<Run> {
    let mut argv = args.common_args("run");
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
    let records: Vec<Value> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()
        .context("parse run JSONL")?;
    ensure!(
        records
            .iter()
            .filter(|record| record["event"] == "ready")
            .count()
            == 1,
        "run must emit one ready event"
    );
    ensure!(
        records
            .iter()
            .filter(|record| record["event"] == "exit")
            .count()
            == 1,
        "run must exit cleanly"
    );
    ensure!(
        !records.iter().any(|record| record["event"] == "error"),
        "run emitted an error event"
    );
    let assistants: Vec<_> = records
        .iter()
        .filter(|record| record["event"] == "assistant")
        .cloned()
        .collect();
    for assistant in &assistants {
        ensure!(
            matches!(assistant["finish_reason"].as_str(), Some("stop" | "eos")),
            "run did not finish naturally: {assistant}"
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
        records,
        assistants,
    })
}

pub(super) async fn run_basic(args: &Args) -> Result<Value> {
    let stdin = format!("{FIRST_TURN}\n{SECOND_TURN}\n/bye\n");
    let run = run_chat(args, "run-basic", Input::Repl(&stdin), None).await?;
    ensure!(
        run.assistants.len() == 2,
        "expected an assistant answer for each actual REPL turn"
    );
    answer(
        run.assistants[0]["content"]
            .as_str()
            .context("missing first answer")?,
        "42",
    )?;
    answer(
        run.assistants[1]["content"]
            .as_str()
            .context("missing recall answer")?,
        "cobalt-731",
    )?;
    Ok(
        json!({"ready": run.records.iter().find(|record| record["event"] == "ready"), "answers": run.assistants}),
    )
}

pub(super) async fn run_stop(args: &Args) -> Result<Value> {
    let baseline = run_chat(args, "run-stop-baseline", Input::Prompt(STOP_PROMPT), None).await?;
    ensure!(
        baseline.assistants.len() == 1,
        "baseline must return one answer"
    );
    let content = baseline.assistants[0]["content"]
        .as_str()
        .context("baseline content")?;
    let reasoning = baseline.assistants[0]["reasoning"].as_str().unwrap_or("");
    let baseline_streamed =
        run_deltas(&baseline.records, &baseline.assistants[0], false)?.is_some();
    let (expected, stop) = protocol::stop_from_baseline(content, reasoning)?;
    let stopped = run_chat(
        args,
        "run-stop-replay",
        Input::Prompt(STOP_PROMPT),
        Some(&stop),
    )
    .await?;
    ensure!(
        stopped.assistants.len() == 1,
        "stop replay must return one answer"
    );
    let actual = stopped.assistants[0]["content"]
        .as_str()
        .context("stopped content")?;
    ensure!(
        actual.trim_end() == expected.trim_end(),
        "run stop prefix mismatch: expected {expected:?}, received {actual:?}"
    );
    ensure!(
        stopped.assistants[0]["finish_reason"] == "stop",
        "run did not hit the selected stop"
    );
    let deltas = run_deltas(&stopped.records, &stopped.assistants[0], baseline_streamed)?
        .unwrap_or_default();
    ensure!(
        !deltas.contains(&stop),
        "run streamed the stop sentinel before hiding it in the final answer"
    );
    Ok(json!({"baseline": content, "stop": stop, "expected_prefix": expected, "actual": actual}))
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
    let first_user = json!({"role": "user", "content": FIRST_TURN});
    let first = chat(
        server,
        "serve-basic-sync",
        request(server, vec![first_user.clone()]),
        false,
    )
    .await?;
    finished_answer(&first, "42").context("sync arithmetic")?;
    // Feed back the actual returned assistant, including its reasoning.
    let recall = chat(
        server,
        "serve-basic-recall",
        request(
            server,
            vec![
                first_user,
                first.message.clone(),
                json!({"role": "user", "content": SECOND_TURN}),
            ],
        ),
        false,
    )
    .await?;
    finished_answer(&recall, "cobalt-731").context("actual HTTP history recall")?;
    let streamed = chat(
        server,
        "serve-basic-stream",
        request(
            server,
            vec![
                json!({"role": "user", "content": "What is 19 + 23? Reply with only the number."}),
            ],
        ),
        true,
    )
    .await?;
    finished_answer(&streamed, "42").context("streamed arithmetic")?;
    Ok(
        json!({"models": models, "sync_answer": first.content(), "recall": recall.content(), "stream_answer": streamed.content(), "stream_usage": streamed.usage}),
    )
}

pub(super) async fn serve_stop(server: &Server<'_>) -> Result<Value> {
    let mut body = request(
        server,
        vec![json!({"role": "user", "content": STOP_PROMPT})],
    );
    let baseline = chat(server, "serve-stop-baseline", body.clone(), false).await?;
    ensure!(
        baseline.finish == "stop",
        "baseline exhausted its output budget"
    );
    let (expected, stop) = protocol::stop_from_baseline(
        baseline.content(),
        baseline.message["reasoning"].as_str().unwrap_or(""),
    )?;
    body["stop"] = json!([stop]);
    let mut outputs = Vec::new();
    for (name, stream) in [("serve-stop-sync", false), ("serve-stop-stream", true)] {
        let actual = chat(server, name, body.clone(), stream).await?;
        ensure!(
            actual.finish == "stop",
            "stop replay did not hit the selected boundary"
        );
        ensure!(
            actual.content().trim_end() == expected.trim_end(),
            "{name} stop prefix mismatch: expected {expected:?}, received {:?}",
            actual.content()
        );
        ensure!(
            !actual.content().contains(&stop)
                && !actual.message["reasoning"]
                    .as_str()
                    .unwrap_or("")
                    .contains(&stop),
            "{name} leaked stop sentinel"
        );
        outputs.push(actual.message);
    }
    Ok(
        json!({"baseline": baseline.message, "stop": stop, "expected_prefix": expected, "outputs": outputs}),
    )
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
    let declarations = json!([{"type": "function", "function": {
        "name": "calc", "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": false}
    }}]);
    let mut body = request(server, vec![user.clone()]);
    body["tools"] = declarations.clone();
    body["tool_choice"] = json!({"type": "function", "function": {"name": "calc"}});
    let called = chat(server, "tools-call-stream", body, true).await?;
    let calls = called.message["tool_calls"]
        .as_array()
        .context("model did not call a tool")?;
    ensure!(
        calls.len() == 1 && calls[0]["function"]["name"] == "calc",
        "expected one calc invocation"
    );
    let arguments: Value = serde_json::from_str(
        calls[0]["function"]["arguments"]
            .as_str()
            .context("tool arguments")?,
    )?;
    let expression = arguments["expression"]
        .as_str()
        .context("expression argument")?;
    ensure!(
        arguments
            .as_object()
            .is_some_and(|object| object.len() == 1),
        "tool arguments violated additionalProperties: false: {arguments}"
    );
    ensure!(
        expression
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            == "123+456",
        "wrong tool expression: {arguments}"
    );
    // Only execute this typed local fixture. Model text is never shell code.
    let tool_result = 123u64 + 456u64;
    let result_message = json!({"role": "tool", "tool_call_id": calls[0]["id"], "content": json!({"result": tool_result}).to_string()});
    let mut final_body = request(server, vec![user, called.message.clone(), result_message]);
    final_body["tools"] = declarations;
    final_body["tool_choice"] = json!("none");
    let canonical = chat(server, "tools-final-sync", final_body.clone(), false).await?;
    finished_answer(&canonical, "579").context("canonical tool-result replay")?;
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
    let streamed = chat(server, "tools-final-stream", final_body, true).await?;
    finished_answer(&streamed, "579").context("streamed tool-result replay")?;
    Ok(
        json!({"tool_call": called.message, "tool_result": tool_result, "canonical_answer": canonical.content(), "stream_answer": streamed.content(), "reasoning_alias_replayed": server.args.reasoning_alias_replay}),
    )
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
