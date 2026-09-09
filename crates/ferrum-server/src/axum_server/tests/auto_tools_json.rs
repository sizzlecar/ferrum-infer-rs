//! Automatic function calls and strict final answers are separate valid branches.
use super::*;
use ferrum_engine::ContinuousBatchEngine;
use ferrum_interfaces::{sampler::GreedySampler, Tokenizer};
use ferrum_scheduler::ContinuousBatchScheduler;
use ferrum_testkit::MockTensorFactory;
use ferrum_tokenizer::HuggingFaceTokenizer;
use ferrum_types::FerrumError;
use futures::FutureExt;
use std::{
    panic::{resume_unwind, AssertUnwindSafe},
    sync::atomic::Ordering,
    time::Duration,
};
use tokenizers::{
    decoders::fuse::Fuse,
    models::bpe::{Vocab, BPE},
    AddedToken,
};

// Simulate logits and cache storage only; production code owns tokenization,
// constrained sampling, completion, and both HTTP response adapters.
#[path = "engine_stop_contract/executor.rs"]
mod executor;
use executor::{LogitStep, ScriptedExecutor};

#[derive(Clone, Copy, Debug)]
enum Endpoint {
    Chat,
    Responses,
}

impl Endpoint {
    fn path(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Responses => "/v1/responses",
        }
    }

    fn request(self, schema: Value, stream: bool) -> Value {
        let function = json!({
            "name": "weather",
            "parameters": city_schema(),
        });
        let format = json!({"name": "answer", "strict": true, "schema": schema});
        match self {
            Self::Chat => json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [{"type": "function", "function": function}],
                "tool_choice": "auto",
                "response_format": {"type": "json_schema", "json_schema": format},
                "stream": stream,
            }),
            Self::Responses => json!({
                "model": "stub-model",
                "input": "What is the weather in Paris?",
                "tools": [{"type": "function", "name": function["name"], "parameters": function["parameters"]}],
                "tool_choice": "auto",
                "text": {"format": {"type": "json_schema", "name": format["name"], "strict": true, "schema": format["schema"]}},
                "stream": stream,
            }),
        }
    }

    fn replay(self, request: &mut Value) {
        self.replay_call(
            request,
            &json!({
                "id": "call_1", "name": "weather", "arguments": "{\"city\":\"Paris\"}"
            }),
        );
    }

    fn replay_call(self, request: &mut Value, call: &Value) {
        let call_id = call["id"].as_str().expect("tool call must have an ID");
        assert!(!call_id.is_empty());
        match self {
            Self::Chat => {
                request["messages"] = json!([
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {"role": "assistant", "content": null, "tool_calls": [{
                        "id": call_id, "type": "function", "function": {
                            "name": call["name"], "arguments": call["arguments"]
                        }
                    }]},
                    {"role": "tool", "tool_call_id": call_id, "content": "{\"temperature\":21}"}
                ])
            }
            Self::Responses => {
                request["input"] = json!([
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {"type": "function_call", "call_id": call_id, "name": call["name"], "arguments": call["arguments"]},
                    {"type": "function_call_output", "call_id": call_id, "output": "{\"temperature\":21}"}
                ])
            }
        }
    }
}

fn city_schema() -> Value {
    json!({
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"], "additionalProperties": false
    })
}

fn weather_schema() -> Value {
    json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "temperature": {"type": "integer"}},
        "required": ["city", "temperature"], "additionalProperties": false
    })
}

struct Answer {
    content: String,
    calls: Vec<Value>,
}

async fn answer(response: Response, endpoint: Endpoint, stream: bool) -> Answer {
    let status = response.status();
    let text = response_text(response).await;
    assert_eq!(
        status,
        AxumStatusCode::OK,
        "{endpoint:?}, stream={stream}: {text}"
    );
    let body = if stream {
        let events = responses_sse_json_events(&text);
        assert!(!events.is_empty(), "{text}");
        assert!(
            events.iter().all(|event| event["error"].is_null()),
            "{text}"
        );
        match endpoint {
            Endpoint::Chat => {
                assert!(text.contains("data: [DONE]"), "{text}");
                let mut content = String::new();
                let mut calls: Vec<Value> = Vec::new();
                let mut finish = None;
                for event in &events {
                    if let Some(choices) = event["choices"].as_array() {
                        for choice in choices {
                            let delta = &choice["delta"];
                            if let Some(piece) = delta["content"].as_str() {
                                content.push_str(piece);
                            }
                            if let Some(deltas) = delta["tool_calls"].as_array() {
                                for call in deltas {
                                    let index = call["index"].as_u64().unwrap() as usize;
                                    while calls.len() <= index {
                                        calls.push(json!({"id": "", "name": "", "arguments": ""}));
                                    }
                                    if let Some(id) = call["id"].as_str() {
                                        let current = calls[index]["id"].as_str().unwrap();
                                        calls[index]["id"] = json!(format!("{current}{id}"));
                                    }
                                    for field in ["name", "arguments"] {
                                        if let Some(piece) = call["function"][field].as_str() {
                                            let current = calls[index][field].as_str().unwrap();
                                            calls[index][field] =
                                                json!(format!("{current}{piece}"));
                                        }
                                    }
                                }
                            }
                            if !choice["finish_reason"].is_null() {
                                assert!(
                                    finish.replace(choice["finish_reason"].clone()).is_none(),
                                    "{text}"
                                );
                            }
                        }
                    }
                }
                assert_eq!(
                    finish,
                    Some(json!(if calls.is_empty() {
                        "stop"
                    } else {
                        "tool_calls"
                    })),
                    "{text}"
                );
                return Answer { content, calls };
            }
            Endpoint::Responses => {
                let body = events
                    .iter()
                    .find(|event| event["type"] == "response.completed")
                    .unwrap_or_else(|| panic!("no completed response: {text}"))["response"]
                    .clone();
                if body["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|item| item["type"] != "function_call")
                {
                    assert!(
                        events.iter().all(|event| {
                            event["item"]["type"] != "function_call"
                                && !event["type"].as_str().is_some_and(|kind| {
                                    kind.starts_with("response.function_call_arguments.")
                                })
                        }),
                        "final answers must not publish transient tool-call events: {text}"
                    );
                }
                body
            }
        }
    } else {
        serde_json::from_str(&text).unwrap()
    };
    assert!(body["error"].is_null(), "{body}");
    match endpoint {
        Endpoint::Chat => {
            let message = &body["choices"][0]["message"];
            let calls = message["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|call| {
                    json!({
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(
                body["choices"][0]["finish_reason"],
                if calls.is_empty() {
                    "stop"
                } else {
                    "tool_calls"
                }
            );
            Answer {
                content: message["content"].as_str().unwrap_or_default().to_owned(),
                calls,
            }
        }
        Endpoint::Responses => {
            assert_eq!(body["status"], "completed", "{body}");
            let mut content = String::new();
            let mut calls = Vec::new();
            for item in body["output"].as_array().unwrap() {
                match item["type"].as_str() {
                    Some("function_call") => {
                        calls.push(json!({"id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}))
                    }
                    Some("message") => {
                        for part in item["content"].as_array().unwrap() {
                            if part["type"] == "output_text" {
                                content.push_str(part["text"].as_str().unwrap());
                            }
                        }
                    }
                    _ => {}
                }
            }
            Answer { content, calls }
        }
    }
}

#[tokio::test]
async fn explicit_tool_call_uses_argument_schema_instead_of_final_schema() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            let response = post_json(
                router_with_stub_api_response("", weather_tool_api_response()),
                endpoint.path(),
                endpoint.request(weather_schema(), stream),
            )
            .await;
            let result = answer(response, endpoint, stream).await;
            assert!(result.content.is_empty());
            assert_eq!(result.calls.len(), 1);
            assert_eq!(result.calls[0]["name"], "weather");
            assert_eq!(
                serde_json::from_str::<Value>(result.calls[0]["arguments"].as_str().unwrap())
                    .unwrap(),
                json!({"city": "Paris"})
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum InvalidOutput {
    ToolArguments,
    UnknownTool,
    FinalSchema,
    TruncatedStructured,
}

impl InvalidOutput {
    fn assert_diagnostic(self, message: &str) {
        let identifies_contract = match self {
            Self::ToolArguments => message.contains("arguments") && message.contains("schema"),
            Self::UnknownTool => message.contains("undeclared"),
            Self::FinalSchema => message.contains("response_format"),
            Self::TruncatedStructured => {
                message.contains("stop sequence truncated the structured output")
            }
        };
        assert!(
            identifies_contract,
            "wrong failed contract for {self:?}: {message}"
        );
    }
}

fn assert_no_answer_payload(message: &Value) {
    assert!(
        message["content"].is_null() || message["content"].as_str() == Some(""),
        "{message}"
    );
    assert!(
        message["tool_calls"].is_null()
            || message["tool_calls"].as_array().is_some_and(Vec::is_empty),
        "{message}"
    );
    assert!(message["function_call"].is_null(), "{message}");
}

async fn assert_invalid_output(
    response: Response,
    endpoint: Endpoint,
    stream: bool,
    invalid: InvalidOutput,
) {
    let status = response.status();
    let body = response_text(response).await;
    if !stream {
        assert_eq!(
            status,
            AxumStatusCode::INTERNAL_SERVER_ERROR,
            "{endpoint:?}: {body}"
        );
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["type"], "internal_server_error", "{body}");
        invalid.assert_diagnostic(error["error"]["message"].as_str().unwrap());
        assert!(error["choices"].is_null(), "{body}");
        assert!(error["output"].is_null(), "{body}");
        return;
    }
    assert_eq!(status, AxumStatusCode::OK, "{endpoint:?}: {body}");
    let events = responses_sse_json_events(&body);
    assert!(body.contains("data: [DONE]"), "{body}");
    match endpoint {
        Endpoint::Chat => {
            let errors: Vec<_> = events
                .iter()
                .filter(|event| !event["error"].is_null())
                .collect();
            assert_eq!(errors.len(), 1, "{body}");
            assert_eq!(
                errors[0]["error"]["type"], "internal_server_error",
                "{body}"
            );
            invalid.assert_diagnostic(errors[0]["error"]["message"].as_str().unwrap());
            for event in &events {
                for choice in event["choices"].as_array().into_iter().flatten() {
                    assert_no_answer_payload(&choice["delta"]);
                    assert_no_answer_payload(&choice["message"]);
                    assert!(
                        choice["finish_reason"].is_null(),
                        "invalid output must not finish successfully: {body}"
                    );
                }
            }
        }
        Endpoint::Responses => {
            let failures: Vec<_> = events
                .iter()
                .filter(|event| event["type"] == "response.failed")
                .collect();
            assert_eq!(failures.len(), 1, "{body}");
            let failed = &failures[0]["response"];
            assert_eq!(failed["status"], "failed", "{body}");
            assert_eq!(failed["error"]["code"], "internal_server_error", "{body}");
            invalid.assert_diagnostic(failed["error"]["message"].as_str().unwrap());
            assert_eq!(failed["output"], json!([]), "{body}");
            for event in &events {
                assert_ne!(event["type"], "response.completed", "{body}");
                assert_ne!(
                    event["type"], "response.function_call_arguments.delta",
                    "{body}"
                );
                assert_ne!(
                    event["type"], "response.function_call_arguments.done",
                    "{body}"
                );
                assert_ne!(event["item"]["type"], "function_call", "{body}");
                if event["type"] == "response.output_text.delta" {
                    assert_eq!(event["delta"], "", "invalid final text leaked: {body}");
                }
                if event["type"] == "response.output_text.done" {
                    assert_eq!(event["text"], "", "invalid final text leaked: {body}");
                }
                for part in event["item"]["content"].as_array().into_iter().flatten() {
                    if part["type"] == "output_text" {
                        assert_eq!(part["text"], "", "invalid final item leaked: {body}");
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn strict_auto_rejects_tool_arguments_outside_the_declared_schema() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            let mut response = weather_tool_api_response();
            let ferrum_types::ApiResponse::Chat(chat) = &mut response else {
                panic!("chat fixture")
            };
            // This remains valid JSON but violates the tool's city:string schema.
            chat.message.tool_calls[0].function.arguments = json!({"city": 7}).to_string();
            let response = post_json(
                router_with_stub_api_response("", response),
                endpoint.path(),
                endpoint.request(weather_schema(), stream),
            )
            .await;
            assert_invalid_output(response, endpoint, stream, InvalidOutput::ToolArguments).await;
        }
    }
}

#[tokio::test]
async fn strict_auto_rejects_undeclared_tool_before_emitting_a_call() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            let mut response = weather_tool_api_response();
            let ferrum_types::ApiResponse::Chat(chat) = &mut response else {
                panic!("chat fixture")
            };
            chat.message.tool_calls[0].function.name = "undeclared_weather".to_string();
            let response = post_json(
                router_with_stub_api_response("", response),
                endpoint.path(),
                endpoint.request(weather_schema(), stream),
            )
            .await;
            assert_invalid_output(response, endpoint, stream, InvalidOutput::UnknownTool).await;
        }
    }
}

#[tokio::test]
async fn strict_auto_rejects_final_json_outside_the_final_schema_without_leaking_text() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            let mut wire = endpoint.request(weather_schema(), stream);
            endpoint.replay(&mut wire);
            // A complete object with the wrong temperature type must not be
            // streamed as a successful answer after a completed tool round trip.
            let response = post_json(
                router_with_stub_stream_chunks(&[
                    "{\"city\":\"Paris\",",
                    "\"temperature\":\"warm\"}",
                ]),
                endpoint.path(),
                wire,
            )
            .await;
            assert_invalid_output(response, endpoint, stream, InvalidOutput::FinalSchema).await;
        }
    }
}

#[tokio::test]
async fn replayed_tool_result_allows_strict_final_answer_with_same_controls() {
    let expected = json!({"city": "Paris", "temperature": 21});
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            let mut wire = endpoint.request(weather_schema(), stream);
            endpoint.replay(&mut wire);
            let response = post_json(
                router_with_stub(&expected.to_string()),
                endpoint.path(),
                wire,
            )
            .await;
            let result = answer(response, endpoint, stream).await;
            assert!(result.calls.is_empty());
            assert_eq!(
                serde_json::from_str::<Value>(&result.content).unwrap(),
                expected
            );
        }
    }
}

#[tokio::test]
async fn final_json_matching_sole_tool_arguments_is_content() {
    assert_direct_final(city_schema(), json!({"city": "Paris"})).await;
}

#[tokio::test]
async fn final_json_with_name_and_arguments_fields_is_content() {
    let schema = json!({
        "type": "object",
        "properties": {"name": {"type": "string"}, "arguments": city_schema()},
        "required": ["name", "arguments"], "additionalProperties": false
    });
    assert_direct_final(
        schema,
        json!({"name": "weather", "arguments": {"city": "Paris"}}),
    )
    .await;
}

async fn assert_direct_final(schema: Value, expected: Value) {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            let response = post_json(
                router_with_stub(&expected.to_string()),
                endpoint.path(),
                endpoint.request(schema.clone(), stream),
            )
            .await;
            let result = answer(response, endpoint, stream).await;
            assert!(
                result.calls.is_empty(),
                "final JSON must not be inferred as a call: {:?}",
                result.calls
            );
            assert_eq!(
                serde_json::from_str::<Value>(&result.content).unwrap(),
                expected
            );
        }
    }
}

const EOS: &str = "<|endoftext|>";
const FINAL: &str = r#"{"city":"Paris","temperature":21}"#;
const INVALID: &str = "the grammar must reject this unrestricted prose";

#[derive(Clone, Copy, Debug)]
enum ToolProtocol {
    Json,
    FunctionParameterXml,
}

impl ToolProtocol {
    fn template(self) -> ModelChatTemplate {
        let mut template = ModelChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}",
            "auto-tools-contract",
        );
        template.tool_call_protocol = match self {
            Self::Json => ferrum_types::ApiToolCallProtocol::Json,
            Self::FunctionParameterXml => ferrum_types::ApiToolCallProtocol::FunctionParameterXml,
        };
        template
    }

    fn envelope(self) -> &'static str {
        match self {
            Self::Json => r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#,
            Self::FunctionParameterXml => "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>",
        }
    }
}

async fn infer_competing_branches(
    protocol: ToolProtocol,
    endpoint: Endpoint,
    stream: bool,
    prefer_tool: bool,
) -> Answer {
    infer_branches(protocol, endpoint, stream, Some(prefer_tool)).await
}

async fn infer_branches(
    protocol: ToolProtocol,
    endpoint: Endpoint,
    stream: bool,
    prefer_tool: Option<bool>,
) -> Answer {
    infer_branches_with_stop(protocol, endpoint, stream, prefer_tool, None)
        .await
        .expect("uninterrupted generation must return an answer")
}

async fn branch_tokenizer(pieces: &[&str]) -> Arc<HuggingFaceTokenizer> {
    let mut vocab = Vocab::new();
    for piece in (32u8..=126)
        .map(|byte| (byte as char).to_string())
        .chain(["\n".to_owned(), EOS.to_owned()])
        .chain(pieces.iter().map(|piece| (*piece).to_string()))
    {
        let id = vocab.len() as u32;
        vocab.entry(piece).or_insert(id);
    }
    let mut inner = tokenizers::Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    inner.with_decoder(Some(Fuse::new()));
    inner.add_special_tokens(&[AddedToken::from(EOS, true)]);
    let generation = json!({"eos_token_id": inner.token_to_id(EOS).unwrap()});
    Arc::new(
        HuggingFaceTokenizer::from_source_bytes(
            inner.to_string(false).unwrap().as_bytes(),
            None,
            Some(generation.to_string().as_bytes()),
        )
        .await
        .unwrap(),
    )
}

async fn infer_branches_with_stop(
    protocol: ToolProtocol,
    endpoint: Endpoint,
    stream: bool,
    prefer_tool: Option<bool>,
    stop: Option<&str>,
) -> Option<Answer> {
    let envelope = protocol.envelope();
    let tokenizer = branch_tokenizer(&[FINAL, INVALID, envelope]).await;
    let selected = if prefer_tool.unwrap_or(true) {
        envelope
    } else {
        FINAL
    };
    let other = if prefer_tool.unwrap_or(true) {
        FINAL
    } else {
        envelope
    };
    let executor = Arc::new(if prefer_tool.is_some() {
        ScriptedExecutor::from_steps(
            tokenizer.vocab_size(),
            vec![
                LogitStep::candidates(vec![
                    (tokenizer.token_id(INVALID).unwrap(), 100.0),
                    (tokenizer.token_id(selected).unwrap(), 50.0),
                    (tokenizer.token_id(other).unwrap(), 10.0),
                ]),
                LogitStep::only(tokenizer.token_id(EOS).unwrap()),
            ],
        )
    } else {
        // No merges exist for the envelope: every framing and argument byte
        // crosses the actual matcher/sampler one token at a time.
        let mut script = tokenizer.encode(selected, false).unwrap();
        assert!(script.len() > 1);
        script.push(tokenizer.token_id(EOS).unwrap());
        ScriptedExecutor::new(tokenizer.vocab_size(), script)
    });
    let mut wire = endpoint.request(weather_schema(), stream);
    if let Some(stop) = stop {
        wire["stop"] = json!([stop]);
    }
    run_branch_request(
        tokenizer,
        executor,
        protocol.template(),
        endpoint,
        stream,
        wire,
        selected,
        stop.is_some(),
    )
    .await
    .0
}

async fn run_branch_request(
    tokenizer: Arc<HuggingFaceTokenizer>,
    executor: Arc<ScriptedExecutor>,
    template: ModelChatTemplate,
    endpoint: Endpoint,
    stream: bool,
    mut wire: Value,
    selected: &str,
    expect_stop_error: bool,
) -> (Option<Answer>, String) {
    let mut config = EngineConfig::default();
    config.model.model_id = ModelId::new("auto-tools-contract");
    config.scheduler.max_running_requests = 1;
    config.batching.max_num_batched_tokens = 256;
    let engine = Arc::new(
        ContinuousBatchEngine::new_plan_runtime(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            tokenizer.clone(),
            Arc::new(GreedySampler),
            executor.clone(),
            Arc::new(MockTensorFactory),
        )
        .unwrap(),
    );
    let router = AxumServer::from_llm(engine.clone())
        .with_prompt_template(Some(template))
        .build_router();
    wire["model"] = json!("auto-tools-contract");
    wire["temperature"] = json!(0);
    match endpoint {
        Endpoint::Chat => wire["max_tokens"] = json!(selected.len() + 4),
        Endpoint::Responses => wire["max_output_tokens"] = json!(selected.len() + 4),
    }
    let outcome = AssertUnwindSafe(tokio::time::timeout(Duration::from_secs(5), async {
        let response = post_json(router, endpoint.path(), wire).await;
        if expect_stop_error {
            assert_invalid_output(
                response,
                endpoint,
                stream,
                InvalidOutput::TruncatedStructured,
            )
            .await;
            None
        } else {
            Some(answer(response, endpoint, stream).await)
        }
    }))
    .catch_unwind()
    .await;
    let shutdown = tokio::time::timeout(Duration::from_secs(5), engine.shutdown()).await;
    executor.assert_released();
    shutdown.expect("engine shutdown must terminate").unwrap();
    let result = match outcome {
        Ok(response) => response.expect("engine HTTP request must terminate"),
        Err(panic) => resume_unwind(panic),
    };
    if !expect_stop_error {
        executor.assert_completed();
        assert_eq!(
            tokenizer.decode(&executor.decoded_inputs(), false).unwrap(),
            selected,
            "actual constrained sampling selected the wrong branch: {endpoint:?}, stream={stream}"
        );
    }
    let prefill = tokenizer.decode(&executor.prefill_tokens(), false).unwrap();
    (result, prefill)
}

fn llama_json_template() -> ModelChatTemplate {
    let mut template = ModelChatTemplate::new(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/chat_template/unsloth__Meta-Llama-3.1-8B-Instruct/template.jinja"
        )),
        "native-bare-json-contract",
    );
    template.bos_token = Some("<|begin_of_text|>".into());
    template.eos_token = Some("<|eot_id|>".into());
    assert_eq!(
        template.tool_call_protocol,
        ferrum_types::ApiToolCallProtocol::Json
    );
    template
}

fn named_call_schema(arguments_field: &str) -> Value {
    json!({
        "type": "object",
        "properties": {"name": {"type": "string"}, (arguments_field): city_schema()},
        "required": ["name", arguments_field], "additionalProperties": false
    })
}

fn bare_call(arguments_field: &str, reversed: bool, spaced: bool) -> String {
    let (name, arguments) = if spaced {
        (
            "\"name\" : \"weather\"".to_owned(),
            format!("\"{arguments_field}\" : {{\n  \"city\" : \"Paris\"\n}}"),
        )
    } else {
        (
            "\"name\":\"weather\"".to_owned(),
            format!("\"{arguments_field}\":{{\"city\":\"Paris\"}}"),
        )
    };
    let (first, second) = if reversed {
        (arguments, name)
    } else {
        (name, arguments)
    };
    if spaced {
        format!("{{ \n {first},\n {second} \n}}")
    } else {
        format!("{{{first},{second}}}")
    }
}

async fn infer_json_script(
    template: ModelChatTemplate,
    endpoint: Endpoint,
    stream: bool,
    wire: Value,
    emitted: &[&str],
    alternative: Option<&str>,
) -> (Answer, String) {
    let mut pieces = emitted.to_vec();
    pieces.push(INVALID);
    pieces.extend(alternative);
    let tokenizer = branch_tokenizer(&pieces).await;
    let mut steps = emitted
        .iter()
        .map(|piece| LogitStep::only(tokenizer.token_id(piece).unwrap()))
        .collect::<Vec<_>>();
    if let Some(alternative) = alternative {
        // Compete at the first token, including prefixes whose remainder is
        // emitted by later steps. Choosing the wrong branch must fail rather
        // than silently turn a valid final prefix into a tool call.
        steps[0] = LogitStep::candidates(vec![
            (tokenizer.token_id(INVALID).unwrap(), 100.0),
            (tokenizer.token_id(emitted[0]).unwrap(), 50.0),
            (tokenizer.token_id(alternative).unwrap(), 10.0),
        ]);
    }
    steps.push(LogitStep::only(tokenizer.token_id(EOS).unwrap()));
    let executor = Arc::new(ScriptedExecutor::from_steps(tokenizer.vocab_size(), steps));
    let selected = emitted.concat();
    let (answer, prefill) = run_branch_request(
        tokenizer, executor, template, endpoint, stream, wire, &selected, false,
    )
    .await;
    (
        answer.expect("complete JSON must return an answer"),
        prefill,
    )
}

#[tokio::test]
async fn actual_sampler_replays_bare_named_json_calls_with_same_strict_controls() {
    for template in [llama_json_template(), ToolProtocol::Json.template()] {
        for arguments_field in ["arguments", "parameters"] {
            let call = bare_call(arguments_field, false, false);
            for endpoint in [Endpoint::Chat, Endpoint::Responses] {
                for stream in [false, true] {
                    let mut wire = endpoint.request(weather_schema(), stream);
                    let (first, _) = infer_json_script(
                        template.clone(),
                        endpoint,
                        stream,
                        wire.clone(),
                        &[&call],
                        Some(FINAL),
                    )
                    .await;
                    assert!(first.content.is_empty());
                    assert_eq!(first.calls.len(), 1);
                    assert_eq!(first.calls[0]["name"], "weather");
                    assert_eq!(
                        serde_json::from_str::<Value>(
                            first.calls[0]["arguments"].as_str().unwrap()
                        )
                        .unwrap(),
                        json!({"city": "Paris"})
                    );
                    // Preserve the same tools, auto choice, and final schema;
                    // only append the call returned over HTTP and its result.
                    endpoint.replay_call(&mut wire, &first.calls[0]);
                    let (second, prefill) = infer_json_script(
                        template.clone(),
                        endpoint,
                        stream,
                        wire,
                        &[FINAL],
                        Some(&call),
                    )
                    .await;
                    assert!(second.calls.is_empty());
                    assert_eq!(second.content, FINAL);
                    if template.source == "native-bare-json-contract" {
                        let history = prefill
                            .split("<|start_header_id|>assistant<|end_header_id|>\n\n")
                            .nth(1)
                            .expect("native template must render the previous call")
                            .split("<|eot_id|>")
                            .next()
                            .unwrap();
                        assert_eq!(
                            serde_json::from_str::<Value>(history).unwrap(),
                            json!({"name": "weather", "parameters": {"city": "Paris"}})
                        );
                        let result = prefill
                            .split("<|start_header_id|>ipython<|end_header_id|>\n\n")
                            .nth(1)
                            .expect("native template must render the tool result")
                            .split("<|eot_id|>")
                            .next()
                            .unwrap();
                        let result = serde_json::from_str::<Value>(result).unwrap();
                        let result = if let Some(encoded) = result.as_str() {
                            serde_json::from_str::<Value>(encoded).unwrap()
                        } else {
                            result
                        };
                        assert_eq!(result, json!({"temperature": 21}));
                    } else {
                        assert!(prefill.contains("temperature"), "{prefill}");
                        assert!(prefill.contains("21"), "{prefill}");
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn actual_sampler_prefers_final_schema_for_identical_named_call_bytes() {
    for arguments_field in ["arguments", "parameters"] {
        for reversed in [false, true] {
            for spaced in [false, true] {
                let payload = bare_call(arguments_field, reversed, spaced);
                let expected = serde_json::from_str::<Value>(&payload).unwrap();
                for endpoint in [Endpoint::Chat, Endpoint::Responses] {
                    for stream in [false, true] {
                        let (result, _) = infer_json_script(
                            llama_json_template(),
                            endpoint,
                            stream,
                            endpoint.request(named_call_schema(arguments_field), stream),
                            &[&payload],
                            None,
                        )
                        .await;
                        assert!(
                            result.calls.is_empty(),
                            "final schema accepts the entire call-shaped payload: {payload}; {:?}",
                            result.calls
                        );
                        assert_eq!(
                            serde_json::from_str::<Value>(&result.content).unwrap(),
                            expected
                        );
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn actual_sampler_prefers_final_after_reasoning_closes_in_the_same_token() {
    let template = ModelChatTemplate::new(
        concat!(
            "{% if tools %}Tools: {{ tools | tojson }}{% endif %}",
            "{% for message in messages %}{{ message.content }}{% endfor %}",
            "{% if add_generation_prompt %}<assistant><think>{% endif %}",
        ),
        "prompt-opened-json-contract",
    );
    assert_eq!(
        template.reasoning_protocol,
        ModelReasoningProtocol::PromptOpened
    );
    for arguments_field in ["arguments", "parameters"] {
        // Reordered keys and whitespace remain valid final-schema JSON, even
        // if a grammar's canonical property order differs from these bytes.
        let payload = bare_call(arguments_field, true, true);
        let closing_and_payload = format!("</think>{payload}");
        for endpoint in [Endpoint::Chat, Endpoint::Responses] {
            for stream in [false, true] {
                let (result, _) = infer_json_script(
                    template.clone(),
                    endpoint,
                    stream,
                    endpoint.request(named_call_schema(arguments_field), stream),
                    &["I can answer directly.", &closing_and_payload],
                    None,
                )
                .await;
                assert!(result.calls.is_empty());
                assert_eq!(
                    serde_json::from_str::<Value>(&result.content).unwrap(),
                    serde_json::from_str::<Value>(&payload).unwrap()
                );
            }
        }
    }
}

#[tokio::test]
async fn actual_sampler_keeps_think_tags_inside_final_json_strings() {
    for city in ["<think>Paris</think>", "<think>Paris", "Paris</think>"] {
        let mut schema = weather_schema();
        schema["properties"]["city"] = json!({"type": "string", "const": city});
        let payload = json!({"city": city, "temperature": 21}).to_string();
        for endpoint in [Endpoint::Chat, Endpoint::Responses] {
            for stream in [false, true] {
                let (result, _) = infer_json_script(
                    ToolProtocol::Json.template(),
                    endpoint,
                    stream,
                    endpoint.request(schema.clone(), stream),
                    &[&payload],
                    Some(ToolProtocol::Json.envelope()),
                )
                .await;
                // The grammar classified this complete payload as Final.
                // Tags within its string cannot change either the content
                // validated at the HTTP boundary or the published branch.
                assert!(result.calls.is_empty());
                assert_eq!(result.content, payload);
            }
        }
    }
}

#[tokio::test]
async fn actual_sampler_keeps_pretty_json_tokens_available_alongside_tools() {
    let packets = [
        " {\n",
        "  \"city\": \"Paris\",\n",
        "  \"temperature\": 21\n",
        "}",
    ];
    for protocol in [ToolProtocol::Json, ToolProtocol::FunctionParameterXml] {
        for endpoint in [Endpoint::Chat, Endpoint::Responses] {
            for stream in [false, true] {
                let (result, _) = infer_json_script(
                    protocol.template(),
                    endpoint,
                    stream,
                    endpoint.request(weather_schema(), stream),
                    &packets,
                    Some(protocol.envelope()),
                )
                .await;
                assert!(result.calls.is_empty());
                assert_eq!(
                    serde_json::from_str::<Value>(&result.content).unwrap(),
                    json!({"city": "Paris", "temperature": 21})
                );
            }
        }
    }
}

#[tokio::test]
async fn text_stop_inside_a_merged_valid_result_cannot_publish_a_truncated_success() {
    // Responses has no stop-string field; this boundary is a Chat contract.
    for protocol in [ToolProtocol::Json, ToolProtocol::FunctionParameterXml] {
        for stream in [false, true] {
            for prefer_tool in [false, true] {
                infer_branches_with_stop(
                    protocol,
                    Endpoint::Chat,
                    stream,
                    Some(prefer_tool),
                    Some("ris"),
                )
                .await;
            }
        }
    }
}

#[tokio::test]
async fn actual_sampler_allows_explicit_tool_branch_while_rejecting_invalid_prose() {
    for protocol in [ToolProtocol::Json, ToolProtocol::FunctionParameterXml] {
        for endpoint in [Endpoint::Chat, Endpoint::Responses] {
            for stream in [false, true] {
                let result = infer_competing_branches(protocol, endpoint, stream, true).await;
                assert!(result.content.is_empty());
                assert_eq!(result.calls.len(), 1);
                assert_eq!(result.calls[0]["name"], "weather");
                assert_eq!(
                    serde_json::from_str::<Value>(result.calls[0]["arguments"].as_str().unwrap())
                        .unwrap(),
                    json!({"city": "Paris"})
                );
            }
        }
    }
}

#[tokio::test]
async fn actual_sampler_keeps_final_schema_when_tools_are_available() {
    for protocol in [ToolProtocol::Json, ToolProtocol::FunctionParameterXml] {
        for endpoint in [Endpoint::Chat, Endpoint::Responses] {
            for stream in [false, true] {
                let result = infer_competing_branches(protocol, endpoint, stream, false).await;
                assert!(result.calls.is_empty());
                assert_eq!(
                    serde_json::from_str::<Value>(&result.content).unwrap(),
                    serde_json::from_str::<Value>(FINAL).unwrap()
                );
            }
        }
    }
}

#[tokio::test]
async fn forced_native_tools_mask_bare_arguments_through_both_http_adapters() {
    let protocol = ToolProtocol::FunctionParameterXml;
    let envelope = protocol.envelope();
    let bare = r#"{"city":"Paris"}"#;
    let unselected = envelope.replace("weather", "delete_file");
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            for named in [false, true] {
                for split in [false, true] {
                    let tokenizer = branch_tokenizer(&[envelope, bare, &unselected]).await;
                    let mut script = if split {
                        tokenizer.encode(envelope, false).unwrap()
                    } else {
                        vec![tokenizer.token_id(envelope).unwrap()]
                    };
                    let first = script.remove(0);
                    let mut steps = vec![LogitStep::candidates(vec![
                        (tokenizer.token_id(bare).unwrap(), 400.0),
                        (tokenizer.token_id(&unselected).unwrap(), 300.0),
                        (tokenizer.token_id(EOS).unwrap(), 200.0),
                        (first, 100.0),
                    ])];
                    steps.extend(script.into_iter().map(LogitStep::only));
                    steps.push(LogitStep::only(tokenizer.token_id(EOS).unwrap()));
                    let executor =
                        Arc::new(ScriptedExecutor::from_steps(tokenizer.vocab_size(), steps));
                    let mut wire = endpoint.request(weather_schema(), stream);
                    wire.as_object_mut().unwrap().remove("response_format");
                    wire.as_object_mut().unwrap().remove("text");
                    wire["tool_choice"] = if named {
                        match endpoint {
                            Endpoint::Chat => {
                                json!({"type":"function", "function":{"name":"weather"}})
                            }
                            Endpoint::Responses => json!({"type":"function", "name":"weather"}),
                        }
                    } else {
                        json!("required")
                    };
                    let (result, _) = run_branch_request(
                        tokenizer,
                        executor,
                        protocol.template(),
                        endpoint,
                        stream,
                        wire,
                        envelope,
                        false,
                    )
                    .await;
                    let result = result.unwrap();
                    assert!(result.content.is_empty());
                    assert_eq!(result.calls.len(), 1);
                    assert_eq!(result.calls[0]["name"], "weather");
                    assert_eq!(
                        serde_json::from_str::<Value>(
                            result.calls[0]["arguments"].as_str().unwrap()
                        )
                        .unwrap(),
                        json!({"city":"Paris"}),
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn actual_sampler_tracks_tool_framing_across_token_boundaries() {
    for protocol in [ToolProtocol::Json, ToolProtocol::FunctionParameterXml] {
        for endpoint in [Endpoint::Chat, Endpoint::Responses] {
            for stream in [false, true] {
                let result = infer_branches(protocol, endpoint, stream, None).await;
                assert!(result.content.is_empty());
                assert_eq!(result.calls.len(), 1);
                assert_eq!(result.calls[0]["name"], "weather");
                assert_eq!(
                    serde_json::from_str::<Value>(result.calls[0]["arguments"].as_str().unwrap())
                        .unwrap(),
                    json!({"city": "Paris"})
                );
            }
        }
    }
}
