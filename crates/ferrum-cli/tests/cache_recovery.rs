//! Interrupted product downloads must recover through the same real CLI
//! command and produce valid inference, rather than stopping at model loading.

#[path = "download_jsonl/hub.rs"]
mod hub;
#[path = "cache_recovery/model.rs"]
mod model;

use hub::{Hub, REVISION};
use model::{EXPECTED_TOKEN, FIRST_SHARD, INDEX, LAST_SHARD, REPO};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::{Output, Stdio},
    time::Duration,
};
use tokio::{net::TcpListener, process::Command, time::Instant};

struct Flow {
    arguments: Vec<String>,
    server_url: Option<String>,
}

impl Flow {
    fn run(model: &str) -> Self {
        let arguments = [
            "run",
            model,
            "--backend",
            "cpu",
            "--output-format",
            "jsonl",
            "--disable-thinking",
            "--prompt",
            "fixture",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--kv-capacity",
            "64",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        Self {
            arguments,
            server_url: None,
        }
    }

    async fn serve(model: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let port = address.port().to_string();
        let arguments = [
            "serve",
            "--model",
            model,
            "--backend",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            &port,
            "--served-model-name",
            "recovery",
            "--disable-thinking",
            "--max-model-len",
            "64",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "64",
            "--kv-capacity",
            "64",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        Self {
            arguments,
            server_url: Some(format!("http://{address}")),
        }
    }

    fn command(&self, hub: &Hub, cache: &Path) -> Command {
        let mut command = command(hub, cache);
        command.args(&self.arguments);
        command
    }

    async fn successful(&self, hub: &Hub, cache: &Path) {
        let Some(url) = &self.server_url else {
            assert_run_success(output(self.command(hub, cache)).await);
            return;
        };
        let stdout = cache.join("server.stdout");
        let stderr = cache.join("server.stderr");
        let mut child = self
            .command(hub, cache)
            .stdout(fs::File::create(&stdout).unwrap())
            .stderr(fs::File::create(&stderr).unwrap())
            .spawn()
            .expect("start recovered serve command");
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(45);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                panic!(
                    "serve exited {status}: {}",
                    fs::read_to_string(&stderr).unwrap()
                );
            }
            if let Ok(response) = client.get(format!("{url}/health")).send().await {
                if response.status().is_success() {
                    let health: Value = response.json().await.unwrap();
                    assert_eq!(health["status"], "healthy", "{health}");
                    break;
                }
            }
            assert!(
                Instant::now() < deadline,
                "serve did not become ready: {}",
                fs::read_to_string(&stderr).unwrap()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let response = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&json!({
                "model": "recovery", "messages": [{"role":"user","content":"fixture"}],
                "temperature": 0, "max_tokens": 1, "stream": false
            }))
            .send()
            .await
            .expect("recovered server inference request");
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert!(status.is_success(), "{status}: {body}");
        assert_eq!(
            body["choices"][0]["message"]["content"], EXPECTED_TOKEN,
            "{body}"
        );
        assert_eq!(body["usage"]["completion_tokens"], 1, "{body}");
        child.start_kill().unwrap();
        tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .unwrap()
            .unwrap();
    }
}

fn command(hub: &Hub, cache: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ferrum"));
    command
        .current_dir(cache)
        .env("HOME", cache)
        .env("HF_HOME", cache)
        .env("HF_ENDPOINT", &hub.endpoint)
        .env("NO_COLOR", "1")
        .env("NO_PROXY", "*")
        .env("no_proxy", "*")
        .stdin(Stdio::null())
        .kill_on_drop(true);
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("FERRUM_")
            || matches!(
                name.as_ref(),
                "HTTP_PROXY"
                    | "http_proxy"
                    | "HTTPS_PROXY"
                    | "https_proxy"
                    | "ALL_PROXY"
                    | "all_proxy"
                    | "HF_TOKEN"
                    | "HUGGING_FACE_HUB_TOKEN"
            )
        {
            command.env_remove(key);
        }
    }
    command
}

async fn output(mut command: Command) -> Output {
    tokio::time::timeout(Duration::from_secs(45), command.output())
        .await
        .expect("CLI timed out; child is killed on drop")
        .expect("launch actual ferrum binary")
}

fn jsonl(output: &Output) -> Vec<Value> {
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("non-JSONL stdout {line:?}: {error}"))
        })
        .collect()
}

fn assert_run_success(output: Output) {
    assert!(
        output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records = jsonl(&output);
    let assistants: Vec<_> = records
        .iter()
        .filter(|record| record["event"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 1, "{records:?}");
    assert_eq!(assistants[0]["content"], EXPECTED_TOKEN, "{records:?}");
    assert_eq!(assistants[0]["n_tokens"], 1, "{records:?}");
}

fn snapshot(cache: &Path) -> PathBuf {
    cache
        .join("hub")
        .join(format!("models--{}", REPO.replace('/', "--")))
        .join("snapshots")
        .join(REVISION)
}

async fn assert_list(hub: &Hub, cache: &Path, expected_status: &str) {
    let mut list = command(hub, cache);
    list.arg("list");
    let output = output(list).await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let row = stdout
        .lines()
        .find(|line| line.starts_with(REPO))
        .unwrap_or_else(|| panic!("cache omitted from list: {stdout}"));
    assert!(
        row.split_whitespace().any(|field| field == expected_status),
        "expected {expected_status} cache, got {row}"
    );
}

async fn recover(flow: Flow, missing: &str, pinned: bool) {
    recover_files(flow, missing, pinned, model::files()).await;
}

async fn recover_files(flow: Flow, missing: &str, pinned: bool, files: BTreeMap<String, Vec<u8>>) {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start([(REPO.to_owned(), files.clone())].into()).await;
    hub.fail_get(REPO, missing);
    let failed = output(flow.command(&hub, cache.path())).await;
    assert!(!failed.status.success(), "interrupted transfer must fail");
    assert!(
        hub.requested("GET", REPO, missing),
        "did not reach injected transport failure"
    );
    assert!(
        String::from_utf8_lossy(&failed.stderr).contains("503"),
        "failure was not the fixture HTTP failure: {}",
        String::from_utf8_lossy(&failed.stderr)
    );
    if flow.server_url.is_none() {
        assert!(!jsonl(&failed)
            .iter()
            .any(|record| record["event"] == "assistant"));
    }
    let snapshot = snapshot(cache.path());
    for name in [INDEX, FIRST_SHARD, LAST_SHARD]
        .into_iter()
        .filter(|name| *name != missing)
    {
        assert_eq!(
            fs::read(snapshot.join(name)).unwrap(),
            files[name],
            "completed download must be retained: {name}"
        );
    }
    assert!(
        !snapshot.join(missing).is_file(),
        "failed file unexpectedly ready"
    );
    let completed: BTreeSet<_> = files
        .keys()
        .filter(|name| snapshot.join(name).is_file())
        .cloned()
        .collect();
    assert_list(&hub, cache.path(), "incomplete").await;

    hub.allow_get(REPO, missing);
    hub.clear_requests();
    // Identical CLI arguments and cache: the resolver must reach the downloader
    // again, reuse completed blobs and then run the actual model successfully.
    flow.successful(&hub, cache.path()).await;
    for (name, expected) in &files {
        assert_eq!(fs::read(snapshot.join(name)).unwrap(), *expected, "{name}");
        assert_eq!(
            hub.requested("GET", REPO, name),
            !completed.contains(name),
            "recovery should transfer only missing files: {name}"
        );
    }
    assert_list(&hub, cache.path(), "ready").await;
    if pinned {
        assert!(
            hub.request_paths()
                .iter()
                .all(|path| !path.contains("/main")),
            "an exact commit must not resolve main: {:?}",
            hub.request_paths()
        );
    }
    hub.clear_requests();
    flow.successful(&hub, cache.path()).await;
    assert_list(&hub, cache.path(), "ready").await;
    assert!(
        hub.request_paths().is_empty(),
        "warm cache touched the network: {:?}",
        hub.request_paths()
    );
}

#[tokio::test]
async fn complete_tiny_llama_fixture_runs_real_inference() {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start([(REPO.to_owned(), model::files())].into()).await;
    Flow::run(REPO).successful(&hub, cache.path()).await;
}

#[tokio::test]
async fn run_recovers_interrupted_indexed_checkpoint_with_the_same_command() {
    recover(Flow::run(REPO), LAST_SHARD, false).await;
}

#[tokio::test]
async fn serve_recovers_exact_commit_and_completes_http_inference() {
    let model = format!("{REPO}@{REVISION}");
    recover(Flow::serve(&model).await, LAST_SHARD, true).await;
}

#[tokio::test]
async fn run_recovers_missing_tokenizer_after_weights_finished() {
    recover(Flow::run(REPO), "tokenizer.json", false).await;
}

#[tokio::test]
async fn run_recovers_selected_chat_template_and_applies_it() {
    recover_files(
        Flow::run(REPO),
        "chat_template.jinja",
        false,
        model::standalone_template_files(),
    )
    .await;
}
