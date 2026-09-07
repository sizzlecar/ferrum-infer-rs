//! Independent Chat/Responses tool-result round trips against the loaded model.
use super::{observation, request};
use crate::process::{ApiEndpoint, Server};
use crate::{protocol, write_json};
use anyhow::{Context, Result};
use ferrum_bench_core::release_regression::model_tool::{
    auto_tools_json_controls, auto_tools_json_responses_controls, verify_auto_tools_json_case,
    AUTO_TOOLS_JSON_PROMPT,
};
use serde_json::{json, Value};

fn checkpoint(server: &Server<'_>, name: &str, evidence: &Value) -> Result<()> {
    write_json(
        server.args.report_dir.join(format!("{name}.evidence.json")),
        evidence,
    )
}

async fn exchange(
    server: &Server<'_>,
    name: &str,
    body: &Value,
    endpoint: ApiEndpoint,
    stream: bool,
) -> Result<(protocol::Chat, Option<Value>)> {
    let text = server.request_to(name, body, endpoint).await?;
    match endpoint {
        ApiEndpoint::ChatCompletions => Ok((
            if stream {
                protocol::stream(&text)?
            } else {
                protocol::sync(&text)?
            },
            None,
        )),
        ApiEndpoint::Responses => {
            let parsed: protocol::Responses = if stream {
                protocol::responses_stream(&text)?
            } else {
                protocol::responses_sync(&text)?
            };
            Ok((parsed.chat, Some(parsed.response)))
        }
    }
}

async fn round_trip(
    server: &Server<'_>,
    endpoint: ApiEndpoint,
    mode: &str,
    stream: bool,
    evidence: &mut Value,
) -> Result<()> {
    let responses = matches!(endpoint, ApiEndpoint::Responses);
    let prefix = if responses {
        "auto-tools-json-responses"
    } else {
        "auto-tools-json"
    };
    let checkpoint_name = format!("{prefix}-{mode}");
    let user = json!({"role": "user", "content": AUTO_TOOLS_JSON_PROMPT});
    let mut first = if responses {
        let mut body = auto_tools_json_responses_controls();
        body["model"] = json!("regression-model");
        body["input"] = json!([user]);
        body["temperature"] = json!(0.0);
        body["max_output_tokens"] = json!(server.args.max_tokens);
        body
    } else {
        let mut body = request(server, vec![user]);
        body.as_object_mut()
            .unwrap()
            .extend(auto_tools_json_controls().as_object().unwrap().clone());
        if stream {
            body["stream_options"] = json!({"include_usage": true});
        }
        body
    };
    first["stream"] = json!(stream);
    evidence["call_request"] = first.clone();
    evidence["phase"] = json!("call_response");
    checkpoint(server, &checkpoint_name, evidence)?;
    let (called, native) = exchange(
        server,
        &format!("{prefix}-call-{mode}"),
        &first,
        endpoint,
        stream,
    )
    .await
    .context("call response transport or protocol")?;
    evidence["call"] = observation(&called);
    if let Some(native) = &native {
        evidence["call_response"] = native.clone();
    }
    evidence["phase"] = json!("call_validation");
    checkpoint(server, &checkpoint_name, evidence)?;
    // Only this parsed integer fixture is executed; model text is never code.
    let calculation = protocol::calc_call(&called, server.args.max_tokens)
        .context("automatic calculator handoff")?;
    let result = json!({"result": calculation.result}).to_string();
    let mut continuation = first;
    if responses {
        let native = native.as_ref().context("missing actual Responses call")?;
        let output = native["output"]
            .as_array()
            .context("missing native output items")?;
        let input = continuation["input"].as_array_mut().unwrap();
        // Preserve the complete native output, including reasoning, item IDs
        // and order. The tool result references call_id, never the output ID.
        input.extend(output.iter().cloned());
        input.push(json!({"type": "function_call_output", "call_id": calculation.call["id"], "output": result}));
    } else {
        let messages = continuation["messages"].as_array_mut().unwrap();
        messages.push(called.message);
        messages.push(
            json!({"role": "tool", "tool_call_id": calculation.call["id"], "content": result}),
        );
    }
    evidence["continuation_request"] = continuation.clone();
    evidence["phase"] = json!("final_response");
    checkpoint(server, &checkpoint_name, evidence)?;
    let (completed, native) = exchange(
        server,
        &format!("{prefix}-final-{mode}"),
        &continuation,
        endpoint,
        stream,
    )
    .await
    .context("final response transport or protocol")?;
    evidence["continuation"] = observation(&completed);
    if let Some(native) = native {
        evidence["continuation_response"] = native;
    }
    // A parsed second tool call remains an observation, never a passed final.
    evidence["phase"] = json!("final_validation");
    checkpoint(server, &checkpoint_name, evidence)
}

pub(crate) async fn auto_tools_json(server: &Server<'_>) -> Result<Value> {
    let mut evidence = json!({"responses": {}});
    for endpoint in [ApiEndpoint::ChatCompletions, ApiEndpoint::Responses] {
        for (mode, stream) in [("sync", false), ("stream", true)] {
            let mut observed = json!({"phase": "not_started"});
            if let Err(error) = round_trip(server, endpoint, mode, stream, &mut observed).await {
                observed["error"] = json!(format!("{error:#}"));
            }
            if matches!(endpoint, ApiEndpoint::Responses) {
                evidence["responses"][mode] = observed;
            } else {
                evidence[mode] = observed;
            }
            // Crash diagnostics supplement the structured failure returned to
            // the main runner; they are not a second evidence input contract.
            if let Err(error) = checkpoint(server, "auto-tools-json", &evidence) {
                return Err(crate::case_failure(evidence, error));
            }
        }
    }
    if let Err(error) = verify_auto_tools_json_case(&evidence, server.args.max_tokens) {
        return Err(crate::case_failure(evidence, anyhow::Error::msg(error)));
    }
    Ok(evidence)
}
