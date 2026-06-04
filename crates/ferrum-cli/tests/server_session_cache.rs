//! G3 session-cache product smoke for `ferrum serve`.
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --test server_session_cache -- --ignored --test-threads=1

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
    std::env::var("FERRUM_G3_SMOKE_MODEL").unwrap_or_else(|_| DEFAULT_SMOKE_MODEL.to_string())
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

fn unique_log_path(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("ferrum-g3-{name}-{}-{now}.log", std::process::id()))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
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

struct ServerFixture {
    base_url: String,
    child: Child,
    log_path: PathBuf,
}

impl ServerFixture {
    async fn spawn() -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let log_path = unique_log_path("session-cache-server");
        let log = File::create(&log_path).expect("create server log");
        let bin = ferrum_bin();
        let diag = spawn_diag(&bin);
        let mut child = Command::new(&bin)
            .args([
                "serve",
                smoke_model().as_str(),
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--disable-prefix-cache",
                "--session-cache",
                "memory",
                "--session-cache-max-entries",
                "8",
                "--session-cache-max-tokens",
                "1024",
            ])
            .current_dir(workspace_root())
            .env("NO_COLOR", "1")
            .stdout(Stdio::from(log.try_clone().expect("clone server log")))
            .stderr(Stdio::from(log))
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
                    "server did not become healthy within {STARTUP_TIMEOUT:?}\n{diag}\nlog:\n{}",
                    log_tail(&log_path)
                );
            }
            if let Some(status) = child.try_wait().expect("poll ferrum serve child") {
                panic!(
                    "server exited before healthy: {status}\n{diag}\nlog:\n{}",
                    log_tail(&log_path)
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
            child,
            log_path,
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    fn metrics_url(&self) -> String {
        format!("{}/metrics", self.base_url)
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Ok(text) = fs::read_to_string(&self.log_path) {
            for bad in [
                "panicked",
                "KV cache overflow",
                "failed to render model chat template",
                "<unk>",
                "[PAD]",
            ] {
                assert!(!text.contains(bad), "server log contains {bad}: {text}");
            }
        }
        let _ = fs::remove_file(&self.log_path);
    }
}

async fn chat_with_session(
    client: &Client,
    fx: &ServerFixture,
    session: &str,
    content: &str,
) -> String {
    let response = client
        .post(fx.chat_url())
        .header("X-Ferrum-Session", session)
        .json(&json!({
            "model": smoke_model(),
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 256
        }))
        .send()
        .await
        .expect("chat post");
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("chat json");
    body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("missing content: {body}"))
        .to_string()
}

fn metric_value(metrics: &str, name: &str) -> f64 {
    metrics
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            (parts.next()? == name)
                .then(|| parts.next()?.parse::<f64>().ok())
                .flatten()
        })
        .unwrap_or_else(|| panic!("missing metric {name}:\n{metrics}"))
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads qwen3:0.6b real model"]
async fn g3_session_cache_real_model_smoke() {
    let fx = ServerFixture::spawn().await;
    let client = Client::new();

    let _ = chat_with_session(
        &client,
        &fx,
        "session-a",
        "The session secret is ferrum-red. Remember it for the next question.",
    )
    .await;
    let recalled = chat_with_session(
        &client,
        &fx,
        "session-a",
        "Based on the previous messages in this same session, what is the exact session secret?",
    )
    .await;
    assert!(
        recalled.contains("ferrum-red"),
        "same-session recall failed: {recalled}"
    );

    let isolated = chat_with_session(&client, &fx, "session-b", "Based on previous messages in this session, what is the exact session secret? If none, reply NONE.").await;
    assert!(
        !isolated.contains("ferrum-red"),
        "cross-session leak: {isolated}"
    );

    let metadata_response = client
        .post(fx.chat_url())
        .json(&json!({
            "model": smoke_model(),
            "metadata": {"ferrum_session_id": "session-c"},
            "messages": [{"role": "user", "content": "Remember this exact secret: ferrum-green. Reply only OK."}],
            "temperature": 0.0,
            "max_tokens": 96
        }))
        .send()
        .await
        .expect("metadata session post");
    assert_eq!(metadata_response.status(), 200);

    let metrics = client
        .get(fx.metrics_url())
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("metrics text");
    assert!(metric_value(&metrics, "ferrum_session_cache_hits_total") > 0.0);
    assert!(metric_value(&metrics, "ferrum_session_cache_entries") >= 2.0);
}
