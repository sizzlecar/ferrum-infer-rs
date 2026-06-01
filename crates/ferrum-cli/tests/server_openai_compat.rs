//! OpenAI client contract tests — drive `cli serve` with `async-openai`,
//! the community-maintained Rust client for OpenAI's API.
//!
//! Where `server_smoke.rs` tests *behaviour* (does the server respond
//! correctly for a given request), this file tests *contract*: can a
//! real OpenAI client successfully parse our responses? If we deviate
//! from the spec (renamed field, wrong type, missing required field,
//! malformed SSE), `async-openai` raises a parse error and the test
//! fails. Catches schema drift the hand-written `reqwest` tests would
//! miss.
//!
//! Loads a real model; `#[ignore]` by default. Opt in:
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --features metal --test server_openai_compat \
//!       -- --ignored --test-threads=1
//!
//! The Python SDK smoke additionally requires:
//!
//!     python3 -m pip install openai
//!     # optionally: FERRUM_PYTHON=python3.12

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionStreamOptions, ChatCompletionToolArgs, ChatCompletionToolChoiceOption,
        CreateChatCompletionRequestArgs, FunctionObjectArgs, ResponseFormat,
        ResponseFormatJsonSchema,
    },
    Client,
};
use futures::StreamExt;
use reqwest::Client as ReqwestClient;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const SMOKE_MODEL: &str = "qwen3:0.6b";
const SMOKE_MODEL_HF_ID: &str = "Qwen/Qwen3-0.6B";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

fn ferrum_bin() -> PathBuf {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ferrum") {
        return PathBuf::from(bin);
    }
    let current = std::env::current_exe().expect("test exe path");
    let dir = current
        .parent()
        .and_then(|p| p.parent())
        .expect("target dir");
    let mut bin = dir.join("ferrum");
    if cfg!(windows) {
        bin.set_extension("exe");
    }
    assert!(bin.exists(), "ferrum binary not found at {}", bin.display());
    bin
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local_addr").port()
}

fn python_bin() -> String {
    std::env::var("FERRUM_PYTHON")
        .or_else(|_| std::env::var("PYTHON"))
        .unwrap_or_else(|_| "python3".to_string())
}

fn smoke_model_path() -> PathBuf {
    let config = ferrum_cli::CliConfig::default();
    let cache_dir = ferrum_cli::commands::run::get_hf_cache_dir(&config);
    ferrum_cli::source_resolver::find_cached_model(&cache_dir, SMOKE_MODEL_HF_ID)
        .unwrap_or_else(|| panic!("{SMOKE_MODEL_HF_ID} not found in {}", cache_dir.display()))
        .local_path
}

fn qwen3_prompt_tokens_for_user_message(content: &str) -> usize {
    let model_path = smoke_model_path();
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|err| panic!("load {}: {err}", tokenizer_path.display()));
    let rendered_prompt = format!(
        "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    tokenizer
        .encode(rendered_prompt, true)
        .expect("encode rendered prompt")
        .len()
}

/// Same fixture as `server_smoke.rs`; duplicated rather than shared via
/// a `tests/common` module because cargo's test layout makes the common
/// pattern noisy. Move to a shared helper if a third HTTP test file
/// appears.
struct ServerFixture {
    base_url: String,
    child: Child,
}

impl ServerFixture {
    async fn spawn(model: &str) -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let child = Command::new(ferrum_bin())
            .args(["serve", model, "--port", &port.to_string()])
            .env("NO_COLOR", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn ferrum serve");

        let probe = ReqwestClient::new();
        let healthz = format!("{base_url}/health");
        let start = Instant::now();
        loop {
            if start.elapsed() > STARTUP_TIMEOUT {
                panic!("server did not become healthy within {STARTUP_TIMEOUT:?}");
            }
            let ok = probe
                .get(&healthz)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            if ok {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        Self { base_url, child }
    }

    fn client(&self) -> Client<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_base(format!("{}/v1", self.base_url))
            .with_api_key("dummy-key-not-checked");
        Client::with_config(config)
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PR 9 — async-openai client contract tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model — run with `cargo test -- --ignored`"]
async fn test_openai_client_chat_basic() {
    // async-openai will reject our response if it doesn't match the
    // ChatCompletionResponse schema (missing required fields, wrong
    // types). Successful return here is the contract proof.
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = fx.client();

    let request = CreateChatCompletionRequestArgs::default()
        .model(SMOKE_MODEL)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("Say hi in one short sentence.")
            .build()
            .expect("build user msg")
            .into()])
        .max_tokens(50u32)
        .temperature(0.0)
        .build()
        .expect("build request");

    let response = client.chat().create(request).await.expect("chat request");
    assert!(!response.choices.is_empty(), "no choices in response");
    let content = response.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.trim().is_empty(), "content empty");
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model"]
async fn test_openai_client_chat_usage_fields() {
    // Real-model SDK usage smoke: async-openai should parse Ferrum's usage
    // object, and prompt accounting should match the real Qwen3 tokenizer
    // rather than a whitespace estimate.
    const PROMPT: &str = "Count to two, then stop.";
    let expected_prompt_tokens = qwen3_prompt_tokens_for_user_message(PROMPT);
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = fx.client();

    let request = CreateChatCompletionRequestArgs::default()
        .model(SMOKE_MODEL)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content(PROMPT)
            .build()
            .expect("build user msg")
            .into()])
        .max_tokens(20u32)
        .temperature(0.0)
        .build()
        .expect("build usage request");

    let response = client.chat().create(request).await.expect("chat request");
    let usage = response
        .usage
        .as_ref()
        .expect("non-streaming chat should include SDK usage");
    assert_eq!(
        usage.prompt_tokens as usize, expected_prompt_tokens,
        "prompt_tokens should match tokenizer-encoded rendered prompt"
    );
    assert!(
        usage.completion_tokens > 0,
        "completion_tokens should be positive"
    );
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens,
        "total_tokens should equal prompt_tokens + completion_tokens"
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model"]
async fn test_openai_client_chat_streaming() {
    // async-openai parses SSE events into typed `CreateChatCompletionStreamResponse`.
    // Bad event format (missing `data:` prefix, wrong JSON, `[DONE]` malformed)
    // causes the stream to error mid-way; we assert clean iteration to
    // end-of-stream + non-empty concatenated delta. The include_usage final
    // chunk also verifies streaming usage uses the same tokenizer accounting
    // as the non-streaming path.
    const PROMPT: &str = "Say hi in one short sentence.";
    let expected_prompt_tokens = qwen3_prompt_tokens_for_user_message(PROMPT);
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = fx.client();

    let request = CreateChatCompletionRequestArgs::default()
        .model(SMOKE_MODEL)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content(PROMPT)
            .build()
            .expect("build user msg")
            .into()])
        .max_tokens(50u32)
        .temperature(0.0)
        .stream(true)
        .stream_options(ChatCompletionStreamOptions {
            include_usage: true,
        })
        .build()
        .expect("build streaming request");

    let mut stream = client
        .chat()
        .create_stream(request)
        .await
        .expect("open stream");

    let mut content = String::new();
    let mut chunk_count = 0usize;
    let mut usage_prompt_tokens = None;
    while let Some(result) = stream.next().await {
        let chunk = result.expect("parse stream chunk");
        chunk_count += 1;
        if let Some(usage) = &chunk.usage {
            usage_prompt_tokens = Some(usage.prompt_tokens);
            assert_eq!(
                usage.total_tokens,
                usage.prompt_tokens + usage.completion_tokens,
                "stream usage total_tokens should equal prompt_tokens + completion_tokens"
            );
        }
        if let Some(choice) = chunk.choices.first() {
            if let Some(delta) = &choice.delta.content {
                content.push_str(delta);
            }
        }
    }
    assert!(chunk_count > 0, "no stream chunks parsed");
    assert!(
        !content.trim().is_empty(),
        "concatenated stream content empty"
    );
    assert_eq!(
        usage_prompt_tokens.map(|value| value as usize),
        Some(expected_prompt_tokens),
        "stream usage prompt_tokens should match tokenizer-encoded rendered prompt"
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model"]
async fn test_openai_client_tools_stream_options_include_usage() {
    // Exercises async-openai's typed request fields for tools, tool_choice,
    // and stream_options.include_usage. Ferrum does not implement tool-call
    // generation yet, but it must accept the SDK request shape, stream valid
    // chat chunks, and expose the final usage chunk.
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = fx.client();
    let weather_tool = ChatCompletionToolArgs::default()
        .function(
            FunctionObjectArgs::default()
                .name("get_weather")
                .description("Return a short weather summary for a city.")
                .parameters(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }))
                .build()
                .expect("build function object"),
        )
        .build()
        .expect("build tool");

    let request = CreateChatCompletionRequestArgs::default()
        .model(SMOKE_MODEL)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("Say hi in one short sentence. Do not call a tool.")
            .build()
            .expect("build user msg")
            .into()])
        .max_tokens(50u32)
        .temperature(0.0)
        .stream(true)
        .stream_options(ChatCompletionStreamOptions {
            include_usage: true,
        })
        .tools([weather_tool])
        .tool_choice(ChatCompletionToolChoiceOption::Auto)
        .build()
        .expect("build tools streaming request");

    let mut stream = client
        .chat()
        .create_stream(request)
        .await
        .expect("open tools stream");

    let mut content = String::new();
    let mut chunk_count = 0usize;
    let mut usage_seen = false;
    while let Some(result) = stream.next().await {
        let chunk = result.expect("parse tools stream chunk");
        chunk_count += 1;
        if let Some(usage) = &chunk.usage {
            usage_seen = usage.total_tokens > 0;
        }
        if let Some(choice) = chunk.choices.first() {
            if let Some(delta) = &choice.delta.content {
                content.push_str(delta);
            }
        }
    }

    assert!(chunk_count > 0, "no stream chunks parsed");
    assert!(
        !content.trim().is_empty(),
        "concatenated stream content empty"
    );
    assert!(
        usage_seen,
        "stream_options.include_usage did not produce final SDK usage"
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model"]
async fn test_openai_client_response_format_json_object() {
    // `response_format = json_object` activates ferrum's `JsonModeProcessor`
    // (continuous_engine.rs:148) which constrains the sampler to emit
    // valid JSON. Verifies that (a) the OpenAI response_format field is
    // wired through to the engine and (b) the resulting content is
    // actually parseable JSON.
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = fx.client();

    let request = CreateChatCompletionRequestArgs::default()
        .model(SMOKE_MODEL)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content(
                "Return a JSON object with two fields: \"name\" (any name) \
                 and \"age\" (any number). Reply with the JSON only.",
            )
            .build()
            .expect("build user msg")
            .into()])
        .max_tokens(80u32)
        .temperature(0.0)
        .response_format(ResponseFormat::JsonObject)
        .build()
        .expect("build request");

    let response = client.chat().create(request).await.expect("chat request");
    let content = response.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.trim().is_empty(), "json_object response empty");
    // Strict: the whole content must be valid JSON. The server strips
    // markdown fences in `strip_markdown_json_fence` before returning;
    // models that don't emit a fence at all pass through unchanged.
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(content.trim());
    assert!(
        parsed.is_ok(),
        "response_format=json_object should produce parseable JSON \
         (server strips markdown fences); got: {content:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model"]
async fn test_openai_client_strict_json_schema_20_runs() {
    // Milestone G real-model smoke: a simple strict object schema should
    // succeed repeatedly at temperature 0. The server validates before
    // returning, so any hard-mask/validation failure surfaces as an SDK
    // request error or non-JSON content.
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = fx.client();
    let response_format = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: Some("A short answer object.".to_string()),
            name: "answer_object".to_string(),
            schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": {"type": "string"}
                },
                "required": ["answer"]
            })),
            strict: Some(true),
        },
    };

    for run in 0..20 {
        let request = CreateChatCompletionRequestArgs::default()
            .model(SMOKE_MODEL)
            .messages([ChatCompletionRequestUserMessageArgs::default()
                .content("Return an object with one string field named answer.")
                .build()
                .expect("build user msg")
                .into()])
            .max_tokens(64u32)
            .temperature(0.0)
            .response_format(response_format.clone())
            .build()
            .expect("build strict schema request");

        let response = client
            .chat()
            .create(request)
            .await
            .unwrap_or_else(|e| panic!("strict schema run {run} request failed: {e}"));
        let content = response.choices[0].message.content.as_deref().unwrap_or("");
        let parsed: serde_json::Value = serde_json::from_str(content).unwrap_or_else(|e| {
            panic!("strict schema run {run} returned invalid JSON: {e}; content={content:?}")
        });
        assert!(
            parsed.get("answer").and_then(|v| v.as_str()).is_some(),
            "strict schema run {run} missing string answer: {parsed}"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model"]
async fn test_openai_client_multi_turn() {
    // Verify that messages constructed via async-openai's typed builders
    // (User / Assistant / Function variants) tokenize correctly server-side.
    // A wrong `role` enum on the wire would surface here.
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = fx.client();

    let user1 = ChatCompletionRequestUserMessageArgs::default()
        .content("Remember this fact: my name is XiaoMing.")
        .build()
        .expect("build user msg 1")
        .into();
    let asst1 = ChatCompletionRequestAssistantMessageArgs::default()
        .content("Got it. Your name is XiaoMing.")
        .build()
        .expect("build asst msg")
        .into();
    let user2 = ChatCompletionRequestUserMessageArgs::default()
        .content("What is my name? Reply with just the name.")
        .build()
        .expect("build user msg 2")
        .into();

    let request = CreateChatCompletionRequestArgs::default()
        .model(SMOKE_MODEL)
        .messages([user1, asst1, user2])
        .max_tokens(50u32)
        .temperature(0.0)
        .build()
        .expect("build request");

    let response = client.chat().create(request).await.expect("chat request");
    let content = response.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        content.to_lowercase().contains("xiaoming"),
        "expected recall of 'XiaoMing' via async-openai message builders; got: {content:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model and requires the Python `openai` package"]
async fn test_python_openai_sdk_chat_and_stream_smoke() {
    // Exercises the official Python OpenAI SDK against Ferrum's local
    // OpenAI-compatible server. This catches client-side schema/SSE
    // incompatibilities outside Rust's async-openai type model.
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let script = r#"
import os
import sys

try:
    from openai import OpenAI
except Exception as exc:
    raise SystemExit(
        "Python package `openai` is required for this ignored smoke: "
        "python3 -m pip install openai\n"
        f"import error: {exc}"
    )

base_url = os.environ["FERRUM_OPENAI_BASE_URL"]
model = os.environ["FERRUM_OPENAI_MODEL"]
client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy-key-not-checked")

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
    max_tokens=50,
    temperature=0,
)
content = response.choices[0].message.content or ""
if not content.strip():
    raise SystemExit("empty non-streaming Python SDK chat content")

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
    max_tokens=50,
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},
)
chunks = 0
stream_content = []
usage_seen = False
for chunk in stream:
    chunks += 1
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            stream_content.append(delta)
    if getattr(chunk, "usage", None) is not None:
        usage_seen = True

if chunks == 0:
    raise SystemExit("Python SDK stream yielded no chunks")
if not "".join(stream_content).strip():
    raise SystemExit("empty streaming Python SDK chat content")
if not usage_seen:
    raise SystemExit("Python SDK stream_options.include_usage did not expose usage")
"#;

    let output = Command::new(python_bin())
        .arg("-c")
        .arg(script)
        .env("FERRUM_OPENAI_BASE_URL", &fx.base_url)
        .env("FERRUM_OPENAI_MODEL", SMOKE_MODEL)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn Python OpenAI SDK smoke");

    assert!(
        output.status.success(),
        "Python OpenAI SDK smoke failed\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
