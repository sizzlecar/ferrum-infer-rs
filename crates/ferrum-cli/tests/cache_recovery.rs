//! Interrupted product downloads must recover through the same real CLI
//! command and produce valid inference, rather than stopping at model loading.

#[path = "cache_recovery/execution_feedback.rs"]
mod execution_feedback;
#[path = "download_jsonl/hub.rs"]
mod hub;
#[path = "cache_recovery/journal_lifecycle.rs"]
mod journal_lifecycle;
#[path = "cache_recovery/model.rs"]
mod model;

use hub::{Hub, REVISION};
use model::{EXPECTED_TOKEN, FIRST_SHARD, INDEX, LAST_SHARD, REPO};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Output, Stdio},
    time::Duration,
};
use tokio::{process::Command, time::Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Entrypoint {
    Run,
    Serve,
}

struct Flow {
    arguments: Vec<String>,
    entrypoint: Entrypoint,
    expect_execution_feedback: bool,
}

#[derive(Clone, Copy)]
enum Shutdown {
    Kill,
    #[cfg(unix)]
    Terminate,
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
            entrypoint: Entrypoint::Run,
            expect_execution_feedback: false,
        }
    }

    async fn serve(model: &str) -> Self {
        let arguments = [
            "serve",
            "--model",
            model,
            "--backend",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
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
            entrypoint: Entrypoint::Serve,
            expect_execution_feedback: false,
        }
    }

    fn command(&self, hub: &Hub, cache: &Path) -> Command {
        let mut command = command(hub, cache);
        command.args(&self.arguments);
        command
    }

    async fn successful(&self, hub: &Hub, cache: &Path) {
        self.successful_with_shutdown(hub, cache, Shutdown::Kill)
            .await;
    }

    async fn successful_with_shutdown(&self, hub: &Hub, cache: &Path, shutdown: Shutdown) {
        if self.entrypoint == Entrypoint::Run {
            assert_run_success(output(self.command(hub, cache)).await);
            return;
        }
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
        let url = loop {
            if let Some(status) = child.try_wait().unwrap() {
                panic!(
                    "serve exited {status}\nstdout: {}\nstderr: {}",
                    fs::read_to_string(&stdout).unwrap(),
                    fs::read_to_string(&stderr).unwrap()
                );
            }
            // The child owns the OS-selected socket continuously. Discover it
            // only from the actual ready announcement, never from a dropped
            // parent listener or a guessed port. Files retain all diagnostics.
            if let Some(address) = announced_server_address(&fs::read(&stdout).unwrap())
                .expect("valid completed serve readiness announcement")
            {
                let url = format!("http://{address}");
                if let Ok(response) = client.get(format!("{url}/health")).send().await {
                    if response.status().is_success() {
                        let health: Value = response.json().await.unwrap();
                        assert_eq!(health["status"], "healthy", "{health}");
                        break url;
                    }
                }
            }
            assert!(
                Instant::now() < deadline,
                "serve did not become ready\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&fs::read(&stdout).unwrap()),
                fs::read_to_string(&stderr).unwrap()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
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
        if self.expect_execution_feedback {
            execution_feedback::assert_worker_observed_execution(&client, &url).await;
        }
        match shutdown {
            Shutdown::Kill => child.start_kill().unwrap(),
            #[cfg(unix)]
            Shutdown::Terminate => {
                let pid = child.id().expect("live inference server");
                // The child remains owned and unreaped here; SIGTERM exercises
                // the actual product shutdown handler instead of killing it.
                assert_eq!(unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) }, 0);
            }
        }
        let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
            .await
            .unwrap()
            .unwrap();
        #[cfg(unix)]
        if matches!(shutdown, Shutdown::Terminate) {
            assert!(
                status.success(),
                "graceful serve shutdown failed: {}",
                fs::read_to_string(&stderr).unwrap()
            );
        }
        #[cfg(not(unix))]
        let _ = status;
    }
}

/// Read only complete stdout lines: a concurrent file read can split either a
/// UTF-8 character or the port digits. The public ready banner uses the actual
/// bound SocketAddr; a zero port or non-loopback URL is not usable by this fixture.
fn announced_server_address(stdout: &[u8]) -> Result<Option<SocketAddr>, String> {
    let mut address = None;
    for line in stdout.split_inclusive(|byte| *byte == b'\n') {
        let Some(line) = line.strip_suffix(b"\n") else {
            continue;
        };
        let line = String::from_utf8_lossy(line);
        let Some(value) = line
            .trim_end_matches('\r')
            .strip_prefix("🚀 Server running at http://")
        else {
            continue;
        };
        let parsed: SocketAddr = value
            .parse()
            .map_err(|error| format!("invalid ready address {value:?}: {error}"))?;
        if !parsed.ip().is_loopback() || parsed.port() == 0 {
            return Err(format!(
                "ready address must be an assigned loopback socket: {parsed}"
            ));
        }
        if address.replace(parsed).is_some() {
            return Err("serve announced readiness more than once".into());
        }
    }
    Ok(address)
}

#[test]
fn readiness_requires_a_complete_bound_address_not_a_partial_stdout_write() {
    let prefix = "Loading model...\n🚀 Server running at http://127.0.0.1:43210";
    for end in 0..prefix.len() {
        assert_eq!(
            announced_server_address(&prefix.as_bytes()[..end]).unwrap(),
            None
        );
    }
    assert_eq!(announced_server_address(prefix.as_bytes()).unwrap(), None);
    assert_eq!(
        announced_server_address(format!("{prefix}\nEndpoints:\n").as_bytes()).unwrap(),
        Some("127.0.0.1:43210".parse().unwrap())
    );
    assert_eq!(
        announced_server_address("🚀 Server running at http://[::1]:43210\r\n".as_bytes()).unwrap(),
        Some("[::1]:43210".parse().unwrap())
    );
    for address in ["127.0.0.1:0", "0.0.0.0:43210", "127.0.0.1:invalid"] {
        assert!(announced_server_address(
            format!("🚀 Server running at http://{address}\n").as_bytes()
        )
        .is_err());
    }
    assert!(announced_server_address(format!("{prefix}\n{prefix}\n").as_bytes()).is_err());
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
    if flow.entrypoint == Entrypoint::Run {
        assert!(!jsonl(&failed)
            .iter()
            .any(|record| record["event"] == "assistant"));
    } else {
        assert!(
            announced_server_address(&failed.stdout).unwrap().is_none(),
            "a bound socket must not announce readiness after model download fails"
        );
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
