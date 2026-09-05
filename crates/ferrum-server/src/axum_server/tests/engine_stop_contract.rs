//! Wire stop parameters must reach the production engine, and the engine's
//! actual finish cause must survive the response adapter. No stub responses.
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

mod executor;
mod structured;
use executor::ScriptedExecutor;

const STOP: &str = "STOP";
const ORDINARY_EOS: &str = "<|endoftext|>";
const RETURN: &str = "<|return|>";
const FINAL_HEADER: [&str; 3] = ["<|channel|>", "final", "<|message|>"];

struct Observation {
    status: AxumStatusCode,
    body: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    decoded_inputs: Vec<String>,
    executor: Arc<ScriptedExecutor>,
}

fn template(protocol: ModelOutputProtocol) -> ModelChatTemplate {
    let mut template = match protocol {
        ModelOutputProtocol::Text => {
            ModelChatTemplate::new("{{ messages[0].content }}", "text-contract")
        }
        ModelOutputProtocol::HarmonyGptOss => harmony_json_template(),
        _ => panic!("unsupported protocol fixture"),
    };
    template.set_output_protocol(protocol);
    template
}

async fn tokenizer(pieces: &[&str]) -> HuggingFaceTokenizer {
    // Compact vocabulary, but real BPE decoding and real special-token flags.
    // Composite output tokens are deliberately not BPE merges: stop text may
    // encode to different tokens than those containing it in generated output.
    let mut vocab = Vocab::new();
    for piece in (32u8..=126)
        .map(|byte| (byte as char).to_string())
        .chain(std::iter::once("\n".to_owned()))
        .chain(pieces.iter().map(|piece| (*piece).to_owned()))
        .chain(
            ModelOutputProtocol::HarmonyGptOss
                .preserved_special_token_texts()
                .iter()
                .map(|piece| (*piece).to_owned()),
        )
        .chain(std::iter::once(ORDINARY_EOS.to_owned()))
    {
        let next = vocab.len() as u32;
        vocab.entry(piece).or_insert(next);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(bpe);
    inner.with_decoder(Some(Fuse::new()));
    for marker in ModelOutputProtocol::HarmonyGptOss
        .preserved_special_token_texts()
        .iter()
        .copied()
        .chain(std::iter::once(ORDINARY_EOS))
    {
        inner.add_special_tokens(&[AddedToken::from(marker, true)]);
    }
    let generation = json!({"eos_token_id": [
        inner.token_to_id(ORDINARY_EOS).unwrap(),
        inner.token_to_id(RETURN).unwrap(),
    ]});
    HuggingFaceTokenizer::from_source_bytes(
        inner.to_string(false).unwrap().as_bytes(),
        None,
        Some(generation.to_string().as_bytes()),
    )
    .await
    .unwrap()
}

async fn request(
    protocol: ModelOutputProtocol,
    body_pieces: &[&str],
    terminal: &str,
    stop: Option<&str>,
    stream: bool,
) -> Observation {
    let mut pieces = Vec::new();
    if protocol == ModelOutputProtocol::HarmonyGptOss {
        pieces.extend(FINAL_HEADER);
    }
    pieces.extend_from_slice(body_pieces);
    pieces.push(terminal);
    let tokenizer = Arc::new(tokenizer(&pieces).await);
    let script: Vec<_> = pieces
        .iter()
        .map(|piece| tokenizer.token_id(piece).unwrap())
        .collect();
    let executor = Arc::new(ScriptedExecutor::new(
        tokenizer.vocab_size(),
        script.clone(),
    ));
    let response = request_with_executor(
        protocol,
        tokenizer,
        executor,
        pieces.len() + 4,
        stop,
        stream,
        None,
    )
    .await;
    response.executor.assert_completed();
    let actual = response.executor.decoded_inputs();
    assert!(actual.len() <= script.len());
    assert_eq!(actual, script[..actual.len()]);
    response
}

async fn request_with_executor(
    protocol: ModelOutputProtocol,
    tokenizer: Arc<HuggingFaceTokenizer>,
    executor: Arc<ScriptedExecutor>,
    max_tokens: usize,
    stop: Option<&str>,
    stream: bool,
    response_format: Option<Value>,
) -> Observation {
    let mut config = EngineConfig::default();
    config.model.model_id = ModelId::new("protocol-contract");
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
        .with_prompt_template(Some(template(protocol)))
        .build_router();
    let mut wire = json!({
        "model": "protocol-contract",
        "messages": [{"role": "user", "content": "Continue."}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    });
    if let Some(format) = response_format {
        wire["response_format"] = format;
    }
    if let Some(stop) = stop {
        wire["stop"] = json!([stop]);
    }
    if stream {
        wire["stream_options"] = json!({"include_usage": true});
    }
    // Preserve timeout/panic outcomes until the production engine is shut down.
    let outcome = AssertUnwindSafe(tokio::time::timeout(Duration::from_secs(5), async {
        let response = post_json(router, "/v1/chat/completions", wire).await;
        (response.status(), response_text(response).await)
    }))
    .catch_unwind()
    .await;
    let shutdown = tokio::time::timeout(Duration::from_secs(5), engine.shutdown()).await;
    executor.assert_released();
    shutdown.expect("engine shutdown must terminate").unwrap();
    let (status, body) = match outcome {
        Ok(response) => response.expect("production engine HTTP response must terminate"),
        Err(panic) => resume_unwind(panic),
    };
    // Return failures as HTTP observations too, so negative controls fail on
    // the protocol result rather than a preselected-token assertion.
    let decoded_inputs = executor
        .decoded_inputs()
        .iter()
        .map(|token| tokenizer.decode(&[*token], false).unwrap())
        .collect();
    Observation {
        status,
        body,
        prompt_tokens: executor.prompt_tokens.load(Ordering::Relaxed),
        generated_tokens: executor.generated_tokens.load(Ordering::Relaxed),
        decoded_inputs,
        executor,
    }
}

fn sse_events(body: &str) -> Vec<Value> {
    // SSE permits CRLF/CR line endings, multiple data lines, and non-data
    // fields. Validate event boundaries without requiring one serialization.
    let normalized = body.replace("\r\n", "\n").replace('\r', "\n");
    let mut data = Vec::new();
    let mut events = Vec::new();
    let mut done = false;
    for line in normalized.split_terminator('\n') {
        if line.is_empty() {
            if !data.is_empty() {
                assert!(!done, "data event after DONE: {body}");
                let payload = data.join("\n");
                data.clear();
                if payload == "[DONE]" {
                    done = true;
                } else {
                    events.push(serde_json::from_str(&payload).expect("SSE JSON payload"));
                }
            }
        } else if !line.starts_with(':') {
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            if field == "data" {
                data.push(value.strip_prefix(' ').unwrap_or(value));
            }
        }
    }
    assert!(data.is_empty(), "incomplete SSE data frame: {body}");
    assert!(done, "missing DONE event: {body}");
    events
}

#[test]
fn sse_oracle_accepts_standard_framing() {
    assert_eq!(
        sse_events(": keepalive\r\nid: 7\r\nevent: message\r\nretry: 100\r\ndata: {\r\ndata: \"answer\":42}\r\n\r\ndata:[DONE]\r\n\r\n: trailing comment\r\n\r\n"),
        vec![json!({"answer": 42})],
    );
}

#[test]
fn sse_oracle_rejects_incomplete_or_post_done_data() {
    for body in [
        "data: [DONE]\n",
        "data: [DONE]\n\ndata: [DONE]\n\n",
        "data: [DONE]\n\ndata: {}\n\n",
    ] {
        assert!(
            std::panic::catch_unwind(|| sse_events(body)).is_err(),
            "invalid SSE completion accepted: {body:?}"
        );
    }
}

fn assert_success(response: &Observation, stream: bool, expected: &str, expected_generated: usize) {
    assert_eq!(response.status, AxumStatusCode::OK, "{}", response.body);
    response.executor.assert_completed();
    // The HTTP sync adapter also strips stop text. Consumption/usage, not
    // merely final text, proves that the stop reached and halted the engine.
    assert_eq!(
        response.generated_tokens, expected_generated,
        "engine must stop at the matched boundary, before later model tokens"
    );
    let usage = if stream {
        let events = sse_events(&response.body);
        assert!(
            events.iter().all(|event| event.get("error").is_none()),
            "{}",
            response.body
        );
        let mut content = String::new();
        for event in &events {
            if let Some(delta) = event["choices"][0]["delta"]["content"].as_str() {
                content.push_str(delta);
                assert!(
                    expected.starts_with(&content),
                    "stream leaked text: {content:?}"
                );
            }
            for field in ["reasoning", "reasoning_content"] {
                let value = &event["choices"][0]["delta"][field];
                assert!(
                    value.is_null() || value.as_str() == Some(""),
                    "unexpected {field}: {event}"
                );
            }
            assert!(event["choices"][0]["delta"].get("tool_calls").is_none());
        }
        assert_eq!(content, expected);
        let terminals = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event["choices"][0]["finish_reason"].is_string())
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 1, "{}", response.body);
        assert_eq!(terminals[0].1["choices"][0]["finish_reason"], "stop");
        let usages = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event["usage"].is_object())
            .collect::<Vec<_>>();
        assert_eq!(usages.len(), 1, "{}", response.body);
        assert!(
            terminals[0].0 < usages[0].0,
            "usage must follow terminal choice"
        );
        assert_eq!(
            usages[0].0,
            events.len() - 1,
            "usage must be the final JSON frame"
        );
        assert_eq!(terminals[0].1["choices"][0]["delta"]["content"], "");
        for event in &events[terminals[0].0..] {
            for field in ["content", "reasoning", "reasoning_content"] {
                assert!(
                    event["choices"][0]["delta"][field]
                        .as_str()
                        .unwrap_or("")
                        .is_empty(),
                    "payload after terminal choice: {event}"
                );
            }
        }
        assert_eq!(usages[0].1["choices"], json!([]));
        usages[0].1["usage"].clone()
    } else {
        let body: Value = serde_json::from_str(&response.body).unwrap();
        assert!(body.get("error").is_none(), "{body}");
        assert_eq!(body["choices"][0]["message"]["content"], expected);
        for field in ["reasoning", "reasoning_content"] {
            let value = &body["choices"][0]["message"][field];
            assert!(
                value.is_null() || value.as_str() == Some(""),
                "unexpected {field}: {body}"
            );
        }
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert!(body["choices"][0]["message"].get("tool_calls").is_none());
        body["usage"].clone()
    };
    assert_eq!(usage["prompt_tokens"], response.prompt_tokens);
    assert_eq!(usage["completion_tokens"], expected_generated);
    assert_eq!(
        usage["total_tokens"],
        response.prompt_tokens + expected_generated
    );
}

#[tokio::test]
async fn wire_stop_halts_the_engine_before_the_natural_terminal() {
    for protocol in [
        ModelOutputProtocol::Text,
        ModelOutputProtocol::HarmonyGptOss,
    ] {
        let header_len = if protocol == ModelOutputProtocol::HarmonyGptOss {
            FINAL_HEADER.len()
        } else {
            0
        };
        let terminal = if protocol == ModelOutputProtocol::HarmonyGptOss {
            RETURN
        } else {
            ORDINARY_EOS
        };
        for body in [&["answer ", "ST", "OPtail"][..], &["answer STOPtail"][..]] {
            for stream in [false, true] {
                let uncut = request(protocol, body, terminal, Some("UNMATCHED"), stream).await;
                assert_success(
                    &uncut,
                    stream,
                    "answer STOPtail",
                    header_len + body.len() + 1,
                );
                let cut = request(protocol, body, terminal, Some(STOP), stream).await;
                assert_success(&cut, stream, "answer ", header_len + body.len());
            }
        }
    }
}

#[tokio::test]
async fn natural_eos_cannot_impersonate_a_matched_wire_stop() {
    for stop in [None, Some("UNMATCHED")] {
        for stream in [false, true] {
            let response = request(
                ModelOutputProtocol::HarmonyGptOss,
                &["answer "],
                ORDINARY_EOS,
                stop,
                stream,
            )
            .await;
            assert_eq!(response.generated_tokens, FINAL_HEADER.len() + 2);
            if stream {
                assert_eq!(response.status, AxumStatusCode::OK, "{}", response.body);
                let events = sse_events(&response.body);
                let error = events
                    .iter()
                    .find(|event| event["error"].is_object())
                    .expect("malformed natural EOS must produce an SSE error");
                assert_eq!(error["error"]["type"], "internal_server_error");
                assert_eq!(error["error"]["param"], "model_output");
                for field in ["content", "reasoning", "reasoning_content"] {
                    assert!(
                        events
                            .iter()
                            .all(|event| event["choices"][0]["delta"][field]
                                .as_str()
                                .unwrap_or("")
                                .is_empty()),
                        "malformed Harmony must not leak buffered {field}"
                    );
                }
                assert!(
                    events
                        .iter()
                        .all(|event| event["choices"][0]["finish_reason"].is_null()
                            && event["choices"][0]["delta"].get("tool_calls").is_none()
                            && event["usage"].is_null()),
                    "malformed natural EOS must not publish successful completion"
                );
            } else {
                assert_eq!(
                    response.status,
                    AxumStatusCode::INTERNAL_SERVER_ERROR,
                    "{}",
                    response.body
                );
                let body: Value = serde_json::from_str(&response.body).unwrap();
                assert_eq!(body["error"]["type"], "internal_server_error");
                assert!(body.get("choices").is_none());
                assert!(body.get("usage").is_none());
            }
        }
    }
}
