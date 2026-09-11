//! Download diagnostics must not corrupt `run --output-format jsonl`.
//! Tiny, deliberately incomplete models reach the CPU weight loader and fail
//! there. This covers stdout ownership, not inference correctness.

#[path = "download_jsonl/hub.rs"]
mod hub;

use hub::{Hub, ModelFiles, REVISION};
use std::fs;
use std::path::Path;
use std::process::Output;
use std::time::Duration;
use tokio::process::Command;

async fn run(hub: &Hub, model: &str, cache: &Path) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ferrum"));
    command
        .args([
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
        ])
        .current_dir(cache)
        .env("HOME", cache)
        .env("HF_HOME", cache)
        .env("HF_ENDPOINT", &hub.endpoint)
        .env("NO_COLOR", "1")
        .env("NO_PROXY", "*")
        .env("no_proxy", "*")
        .kill_on_drop(true);
    // Only change the child environment; no network/auth/config from the host
    // should redirect a localhost fixture or bypass the intended loader path.
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
    tokio::time::timeout(Duration::from_secs(45), command.output())
        .await
        .expect("local download/CPU load timed out; child is killed on drop")
        .expect("launch actual ferrum binary")
}

fn assert_downloaded(hub: &Hub, cache: &Path, files: &ModelFiles, cold: bool) {
    for (repo, entries) in files {
        let model = cache
            .join("hub")
            .join(format!("models--{}", repo.replace('/', "--")));
        assert_eq!(
            fs::read_to_string(model.join("refs/main")).unwrap(),
            REVISION
        );
        for (name, expected) in entries {
            let snapshot = model.join("snapshots").join(REVISION).join(name);
            assert_eq!(
                fs::read(snapshot).unwrap(),
                *expected,
                "wrong cached payload: {repo}/{name}"
            );
            assert!(
                hub.requested("HEAD", repo, name),
                "download was bypassed: {repo}/{name}"
            );
            assert_eq!(
                hub.requested("GET", repo, name),
                cold,
                "expected cold transfer or completed-blob reuse: {repo}/{name}"
            );
        }
    }
}

fn assert_loader_failure_and_clean_stdout(output: Output) {
    assert!(
        !output.status.success(),
        "fixture intentionally omits model weights"
    );
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 diagnostics");
    // A transport, config or tokenizer failure cannot stand in for reaching
    // the production model loader and asking for the omitted embedding.
    assert!(
        stderr.contains("model.embed_tokens") || stderr.contains("token_embd"),
        "did not reach the intended missing-embedding failure: {stderr}"
    );
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let _: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|error| panic!("download corrupted JSONL stdout: {line:?}: {error}"));
    }
    // Startup failed before a ready event; neither a fake ready nor diagnostic
    // text belongs on stdout. The successful transfer assertions prevent an
    // empty stream caused by an early network failure from passing this test.
    assert!(stdout.is_empty(), "unexpected pre-ready stdout: {stdout:?}");
}

async fn exercise(model: &str, files: ModelFiles) {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start(files.clone()).await;
    let cold = run(&hub, model, cache.path()).await;
    assert_downloaded(&hub, cache.path(), &files, true);
    // Check the actual stream only after establishing transfer + loader reach.
    assert_loader_failure_and_clean_stdout(cold);

    // A normal warm run skips HfDownloader. Remove only these temporary
    // snapshot links, retaining complete blobs to exercise its cached branch.
    for repo in files.keys() {
        fs::remove_dir_all(
            cache
                .path()
                .join("hub")
                .join(format!("models--{}", repo.replace('/', "--")))
                .join("snapshots"),
        )
        .unwrap();
    }
    hub.clear_requests();
    let cached = run(&hub, model, cache.path()).await;
    assert_downloaded(&hub, cache.path(), &files, false);
    assert_loader_failure_and_clean_stdout(cached);
}

#[tokio::test]
async fn cold_and_blob_cached_safetensors_download_preserves_jsonl_stdout() {
    let repo = "fixture/download-jsonl";
    exercise(repo, [(repo.to_owned(), hub::safetensors_files())].into()).await;
}

#[tokio::test]
async fn cold_and_blob_cached_gguf_alias_download_preserves_jsonl_stdout() {
    // Exercise the public alias path that triggered the release finding. The
    // named files below contain only local tiny fixtures, never real weights.
    let alias = "llama3.1:8b-q4_k_m";
    let (repo, filename) = ferrum_cli::source_resolver::resolve_gguf_alias(alias).unwrap();
    let sidecars = ferrum_cli::source_resolver::tokenizer_sibling_repo(&repo).unwrap();
    let files = [
        (repo, [(filename, hub::gguf_without_weights())].into()),
        (sidecars, hub::sidecar_files()),
    ]
    .into();
    exercise(alias, files).await;
}

#[tokio::test]
async fn repository_and_quantization_names_resolve_metadata_before_reaching_the_loader() {
    let repo = "fixture/checkpoint-GGUF";
    for model in [repo.to_owned(), format!("{repo}:Q4_K_M")] {
        let files = [
            (
                repo.to_owned(),
                [("model-Q4_K_M.gguf".to_owned(), hub::gguf_without_weights())].into(),
            ),
            ("fixture/checkpoint".to_owned(), hub::sidecar_files()),
        ]
        .into();
        exercise(&model, files).await;
    }
}
