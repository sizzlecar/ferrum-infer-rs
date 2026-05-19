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

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, ResponseFormat,
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
async fn test_openai_client_chat_streaming() {
    // async-openai parses SSE events into typed `CreateChatCompletionStreamResponse`.
    // Bad event format (missing `data:` prefix, wrong JSON, `[DONE]` malformed)
    // causes the stream to error mid-way; we assert clean iteration to
    // end-of-stream + non-empty concatenated delta.
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
        .stream(true)
        .build()
        .expect("build streaming request");

    let mut stream = client
        .chat()
        .create_stream(request)
        .await
        .expect("open stream");

    let mut content = String::new();
    let mut chunk_count = 0usize;
    while let Some(result) = stream.next().await {
        let chunk = result.expect("parse stream chunk");
        chunk_count += 1;
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
