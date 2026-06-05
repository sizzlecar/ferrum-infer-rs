//! G1 vLLM migration compatibility smoke.
//!
//! Run with a cached small model:
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --test vllm_migration_compat -- --ignored --test-threads=1

use async_openai::{
    config::OpenAIConfig,
    types::{ChatCompletionStreamOptions, CreateChatCompletionRequestArgs},
    Client as OpenAiClient,
};
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_SMOKE_MODEL: &str = "qwen3:0.6b";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

fn smoke_model() -> String {
    std::env::var("FERRUM_G1_SMOKE_MODEL").unwrap_or_else(|_| DEFAULT_SMOKE_MODEL.to_string())
}

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

fn unique_path(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ferrum-g1-{name}-{}-{now}.json",
        std::process::id()
    ))
}

fn python_bin() -> String {
    std::env::var("FERRUM_PYTHON")
        .or_else(|_| std::env::var("PYTHON"))
        .unwrap_or_else(|_| "python3".to_string())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

struct ServerFixture {
    base_url: String,
    effective_config_json: PathBuf,
    stdout_log: PathBuf,
    stderr_log: PathBuf,
    child: Child,
}

fn log_tail(path: &PathBuf) -> String {
    const MAX_CHARS: usize = 16_000;
    let Ok(text) = fs::read_to_string(path) else {
        return format!("unable to read {}", path.display());
    };
    if text.chars().count() <= MAX_CHARS {
        return text;
    }
    let tail = text
        .chars()
        .rev()
        .take(MAX_CHARS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("... truncated ...\n{tail}")
}

fn spawn_diag(bin: &PathBuf) -> String {
    let env_value = |key: &str| std::env::var(key).unwrap_or_else(|_| "<unset>".to_string());
    format!(
        "ferrum_bin={}\nHOME={}\nHF_HOME={}\nXDG_CACHE_HOME={}\nCARGO_BIN_EXE_ferrum={}",
        bin.display(),
        env_value("HOME"),
        env_value("HF_HOME"),
        env_value("XDG_CACHE_HOME"),
        env_value("CARGO_BIN_EXE_ferrum")
    )
}

impl ServerFixture {
    async fn spawn() -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let effective_config_json = unique_path("effective-config");
        let stdout_log = unique_path("serve-stdout");
        let stderr_log = unique_path("serve-stderr");
        let model = smoke_model();
        let port = port.to_string();
        let bin = ferrum_bin();
        let diag = spawn_diag(&bin);
        let mut child = Command::new(&bin)
            .args([
                "serve",
                model.as_str(),
                "--host",
                "127.0.0.1",
                "--port",
                port.as_str(),
                "--max-model-len",
                "2048",
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                "2048",
                "--no-enable-prefix-caching",
                "--effective-config-json",
                effective_config_json.to_str().expect("utf8 temp path"),
            ])
            .current_dir(workspace_root())
            .env("NO_COLOR", "1")
            .stdout(Stdio::from(
                File::create(&stdout_log).expect("create server stdout log"),
            ))
            .stderr(Stdio::from(
                File::create(&stderr_log).expect("create server stderr log"),
            ))
            .spawn()
            .expect("spawn ferrum serve");

        let client = Client::new();
        let healthz = format!("{base_url}/health");
        let start = Instant::now();
        loop {
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "server did not become healthy within {STARTUP_TIMEOUT:?}\n{diag}\nstdout:\n{}\nstderr:\n{}",
                    log_tail(&stdout_log),
                    log_tail(&stderr_log)
                );
            }
            if let Some(status) = child.try_wait().expect("poll ferrum serve child") {
                panic!(
                    "server exited before healthy: {status}\n{diag}\nstdout:\n{}\nstderr:\n{}",
                    log_tail(&stdout_log),
                    log_tail(&stderr_log)
                );
            }
            let ok = client
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

        Self {
            base_url,
            effective_config_json,
            stdout_log,
            stderr_log,
            child,
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url)
    }

    fn openai_client(&self) -> OpenAiClient<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_base(format!("{}/v1", self.base_url))
            .with_api_key("dummy-key-not-checked");
        OpenAiClient::with_config(config)
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_file(&self.effective_config_json);
        let _ = fs::remove_file(&self.stdout_log);
        let _ = fs::remove_file(&self.stderr_log);
    }
}

fn sse_json_chunks(body: &str) -> (Vec<Value>, usize) {
    let mut chunks = Vec::new();
    let mut done_count = 0usize;
    for block in body.split("\n\n") {
        for line in block.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                done_count += 1;
            } else if !data.is_empty() {
                chunks.push(serde_json::from_str(data).expect("valid SSE JSON chunk"));
            }
        }
    }
    (chunks, done_count)
}

fn assert_effective_config(path: &PathBuf) {
    let body = fs::read_to_string(path).expect("effective config json exists");
    let data: Value = serde_json::from_str(&body).expect("effective config JSON");
    let entries = data["entries"].as_array().expect("entries array");
    let entry = |key: &str| -> &Value {
        entries
            .iter()
            .find(|entry| entry["key"] == key)
            .unwrap_or_else(|| panic!("missing runtime entry {key}: {data}"))
    };
    assert_eq!(entry("FERRUM_MAX_MODEL_LEN")["effective_value"], "2048");
    assert_eq!(entry("FERRUM_PAGED_MAX_SEQS")["effective_value"], "4");
    assert_eq!(
        entry("FERRUM_MAX_BATCHED_TOKENS")["effective_value"],
        "2048"
    );
    assert_eq!(entry("FERRUM_PREFIX_CACHE")["effective_value"], "0");
    assert_eq!(entry("FERRUM_PREFIX_CACHE")["source"], "cli");
}

async fn assert_python_openai_sdk_if_available(base_url: &str, model: &str) {
    let python = python_bin();
    let import_status = Command::new(&python)
        .arg("-c")
        .arg("import openai")
        .status();
    if !matches!(import_status, Ok(status) if status.success()) {
        eprintln!("skip Python OpenAI SDK smoke: `import openai` failed");
        return;
    }

    let script = r#"
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["FERRUM_OPENAI_BASE_URL"] + "/v1", api_key="dummy")
model = os.environ["FERRUM_OPENAI_MODEL"]
response = client.chat.completions.create(
    model=model,
    messages=[{"role":"user","content":"Say hi in one short sentence."}],
    max_tokens=128,
    temperature=0,
)
if not (response.choices[0].message.content or "").strip():
    raise SystemExit("empty Python SDK content")
stream = client.chat.completions.create(
    model=model,
    messages=[{"role":"user","content":"Say hi in one short sentence."}],
    max_tokens=128,
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},
)
chunks = 0
usage = 0
content = []
for chunk in stream:
    chunks += 1
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            content.append(delta)
    if getattr(chunk, "usage", None) is not None:
        usage += 1
if chunks == 0 or not "".join(content).strip() or usage != 1:
    raise SystemExit(f"bad Python SDK stream chunks={chunks} usage={usage}")
"#;
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .env("FERRUM_OPENAI_BASE_URL", base_url)
        .env("FERRUM_OPENAI_MODEL", model)
        .output()
        .expect("run Python OpenAI SDK smoke");
    assert!(
        output.status.success(),
        "Python OpenAI SDK smoke failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn serve_help_lists_vllm_compat_flags() {
    let output = Command::new(ferrum_bin())
        .args(["serve", "--help"])
        .output()
        .expect("run ferrum serve --help");
    assert!(output.status.success(), "serve --help failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
    ] {
        assert!(
            stdout.contains(flag),
            "missing {flag} in serve --help:\n{stdout}"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model; run with qwen3:0.6b cached"]
async fn g1_vllm_migration_smoke() {
    let fx = ServerFixture::spawn().await;
    let client = Client::new();
    let model = smoke_model();

    assert_effective_config(&fx.effective_config_json);

    let models: Value = client
        .get(fx.models_url())
        .send()
        .await
        .expect("GET /v1/models")
        .json()
        .await
        .expect("models JSON");
    assert_eq!(models["object"], "list");
    assert!(
        models["data"]
            .as_array()
            .is_some_and(|items| !items.is_empty()),
        "models list is empty: {models}"
    );

    let chat: Value = client
        .post(fx.chat_url())
        .json(&json!({
            "model": model.clone(),
            "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
            "max_tokens": 128,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("POST chat")
        .json()
        .await
        .expect("chat JSON");
    assert_eq!(chat["object"], "chat.completion");
    assert!(
        chat["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty()),
        "empty chat content: {chat}"
    );
    assert!(chat["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0);

    let stream_body = client
        .post(fx.chat_url())
        .json(&json!({
            "model": model.clone(),
            "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
            "max_tokens": 128,
            "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await
        .expect("POST streaming chat")
        .text()
        .await
        .expect("stream body");
    let (chunks, done_count) = sse_json_chunks(&stream_body);
    assert_eq!(done_count, 1, "expected exactly one [DONE]: {stream_body}");
    assert!(!chunks.is_empty(), "no SSE chunks: {stream_body}");
    let content = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert!(
        !content.trim().is_empty(),
        "empty stream content: {stream_body}"
    );

    let usage_stream_body = client
        .post(fx.chat_url())
        .json(&json!({
            "model": model.clone(),
            "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
            "max_tokens": 128,
            "temperature": 0.0,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .send()
        .await
        .expect("POST usage streaming chat")
        .text()
        .await
        .expect("usage stream body");
    let (usage_chunks, usage_done_count) = sse_json_chunks(&usage_stream_body);
    assert_eq!(usage_done_count, 1, "expected one usage [DONE]");
    let usage_chunk_count = usage_chunks
        .iter()
        .filter(|chunk| chunk["choices"].as_array().is_some_and(|c| c.is_empty()))
        .filter(|chunk| chunk.get("usage").is_some_and(|usage| !usage.is_null()))
        .count();
    assert_eq!(usage_chunk_count, 1, "expected one usage chunk");

    let openai = fx.openai_client();
    let request = CreateChatCompletionRequestArgs::default()
        .model(model.clone())
        .messages([
            async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                .content("Say hi in one short sentence.")
                .build()
                .expect("build user msg")
                .into(),
        ])
        .max_tokens(128u32)
        .temperature(0.0)
        .build()
        .expect("build async-openai request");
    let response = openai
        .chat()
        .create(request)
        .await
        .expect("async-openai chat");
    assert!(
        response.choices[0]
            .message
            .content
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty()),
        "empty async-openai content"
    );

    let stream_request = CreateChatCompletionRequestArgs::default()
        .model(model.clone())
        .messages([
            async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                .content("Say hi in one short sentence.")
                .build()
                .expect("build user msg")
                .into(),
        ])
        .max_tokens(128u32)
        .temperature(0.0)
        .stream(true)
        .stream_options(ChatCompletionStreamOptions {
            include_usage: true,
        })
        .build()
        .expect("build async-openai stream request");
    let mut stream = openai
        .chat()
        .create_stream(stream_request)
        .await
        .expect("open async-openai stream");
    let mut async_content = String::new();
    let mut async_usage_chunks = 0usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("parse async-openai stream chunk");
        if chunk.usage.is_some() {
            async_usage_chunks += 1;
        }
        if let Some(choice) = chunk.choices.first() {
            if let Some(delta) = &choice.delta.content {
                async_content.push_str(delta);
            }
        }
    }
    assert!(
        !async_content.trim().is_empty(),
        "empty async stream content"
    );
    assert_eq!(async_usage_chunks, 1, "expected one async usage chunk");

    assert_python_openai_sdk_if_available(&fx.base_url, &model).await;
}
