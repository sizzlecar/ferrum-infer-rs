//! Download diagnostics must not corrupt `run --output-format jsonl`.
//! Tiny, deliberately incomplete models reach the CPU weight loader and fail
//! there. This covers source resolution and stdout ownership, not inference
//! correctness.

#[path = "download_jsonl/hub.rs"]
mod hub;

use hub::{Hub, ModelFiles, REVISION};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Output;
use std::time::Duration;
use tokio::process::Command;

async fn run(hub: &Hub, model: &str, cache: &Path) -> Output {
    invoke(hub, model, cache, false).await
}

async fn invoke(hub: &Hub, model: &str, cache: &Path, serve: bool) -> Output {
    invoke_with_sources(hub, model, cache, serve, &[]).await
}

async fn invoke_with_sources(
    hub: &Hub,
    model: &str,
    cache: &Path,
    serve: bool,
    source_args: &[String],
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ferrum"));
    if serve {
        command.args([
            "serve",
            "--model",
            model,
            "--backend",
            "cpu",
            "--disable-thinking",
            "--port",
            "0",
        ]);
    } else {
        command.args([
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
        ]);
    }
    command
        .args(source_args)
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

#[tokio::test]
async fn bonsai_alias_run_and_serve_download_pinned_weights_and_only_required_metadata() {
    let weights_repo = "prism-ml/Ternary-Bonsai-2-27B-gguf";
    let weights_revision = "6ed5e12bf84b7a63069882c91dd9e9218647d17b";
    let metadata_repo = "Qwen/Qwen3.8-27B";
    let metadata_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
    let filename = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
    // The official repository names exercise the public alias contract. Their
    // payloads remain tiny Llama fixtures: this tests fetching and loading the
    // resolved sources, not Bonsai numerical correctness or backend support.
    let weights = hub::gguf_without_weights();
    let mut metadata = hub::sidecar_files();
    metadata.insert(
        "generation_config.json".into(),
        br#"{"eos_token_id":2}"#.to_vec(),
    );
    metadata.insert(
        "chat_template.jinja".into(),
        b"{% for message in messages %}{{ message.content }}{% endfor %}".to_vec(),
    );
    for model in ["bonsai2:27b", "bonsai2:27b-pq2_0"] {
        for serve in [false, true] {
            let cache = tempfile::tempdir().unwrap();
            let mut metadata_files = metadata.clone();
            metadata_files.insert(
                "model.safetensors".into(),
                b"metadata source weights must not be downloaded".to_vec(),
            );
            let hub = Hub::start_with_revisions(
                [
                    (
                        weights_repo.to_owned(),
                        [(filename.to_owned(), weights.clone())].into(),
                    ),
                    (metadata_repo.to_owned(), metadata_files),
                ]
                .into(),
                [
                    (weights_repo.to_owned(), weights_revision.to_owned()),
                    (metadata_repo.to_owned(), metadata_revision.to_owned()),
                ]
                .into(),
            )
            .await;
            // No source or GGUF filename flags: both product entrypoints must
            // obtain the complete pinned source bundle from the alias alone.
            let diagnostic_args = [
                "--effective-config-json".into(),
                cache.path().join("effective.json").display().to_string(),
                "--decision-trace-jsonl".into(),
                cache.path().join("decisions.jsonl").display().to_string(),
            ];
            let output =
                invoke_with_sources(&hub, model, cache.path(), serve, &diagnostic_args).await;
            assert!(
                !output.status.success(),
                "fixture deliberately omits GGUF tensors"
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("token_embd") || stderr.contains("model.embed_tokens"),
                "{model} (serve={serve}) must resolve both pins and reach the weight loader: {stderr}"
            );
            let weights_root = cache
                .path()
                .join("hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots")
                .join(weights_revision);
            assert_eq!(fs::read(weights_root.join(filename)).unwrap(), weights);
            assert!(hub.requested("GET", weights_repo, filename));
            let metadata_root = cache
                .path()
                .join("hub/models--Qwen--Qwen3.8-27B/snapshots")
                .join(metadata_revision);
            for (name, bytes) in &metadata {
                assert_eq!(fs::read(metadata_root.join(name)).unwrap(), *bytes);
                assert!(hub.requested("GET", metadata_repo, name), "{name}");
            }
            assert!(!metadata_root.join("model.safetensors").exists());
            assert!(!hub.requested("GET", metadata_repo, "model.safetensors"));
            assert!(hub
                .request_paths()
                .iter()
                .all(|path| !path.contains("/main")));
            let config = assert_failed_startup_diagnostics(cache.path());
            let identity = &config["resolution_evidence"];
            assert_eq!(identity["requested_model"], model);
            assert_eq!(identity["resolved_model"], weights_repo);
            for (role, repo, revision) in [
                ("weights", weights_repo, weights_revision),
                ("semantic", metadata_repo, metadata_revision),
                ("tokenizer", metadata_repo, metadata_revision),
            ] {
                let original = &identity["original_sources"][role];
                assert_eq!(original["kind"], "repository");
                assert_eq!(original["location"], repo);
                assert_eq!(original["requested_revision"], revision);
                let resolved = &identity["resolved_sources"][role];
                assert_eq!(resolved["canonical_location"], repo);
                assert_eq!(resolved["resolved_revision"], revision);
            }
            if !serve {
                assert_loader_failure_and_clean_stdout(output);
            }
        }
    }
}

#[tokio::test]
async fn run_and_serve_download_explicit_metadata_pin_without_following_cached_main() {
    let weights_repo = "fixture/pinned-weights";
    let metadata_repo = "fixture/pinned-metadata";
    let filename = "model-Q4_K_M.gguf";
    let old_revision = "a".repeat(40);
    for serve in [false, true] {
        let cache = tempfile::tempdir().unwrap();
        let mut metadata_files = hub::sidecar_files();
        let generation = br#"{"eos_token_id":2}"#;
        metadata_files.insert("generation_config.json".into(), generation.to_vec());
        // A source repository may contain weights. Metadata selection must not
        // accidentally download them while fixing a cold semantic/tokenizer pin.
        metadata_files.insert(
            "model.safetensors".into(),
            b"not selected metadata".to_vec(),
        );
        let hub = Hub::start(
            [
                (
                    weights_repo.to_owned(),
                    [(filename.to_owned(), hub::gguf_without_weights())].into(),
                ),
                (metadata_repo.to_owned(), metadata_files.clone()),
            ]
            .into(),
        )
        .await;
        let repo_root = cache.path().join("hub/models--fixture--pinned-metadata");
        let old_root = repo_root.join("snapshots").join(&old_revision);
        fs::create_dir_all(&old_root).unwrap();
        fs::create_dir(repo_root.join("refs")).unwrap();
        fs::write(repo_root.join("refs/main"), &old_revision).unwrap();
        for (name, bytes) in hub::sidecar_files() {
            fs::write(old_root.join(name), bytes).unwrap();
        }
        // If the wrong warm snapshot is consumed, tokenizer parsing fails
        // before the intentional missing-weights failure asserted below.
        let stale_tokenizer = b"stale snapshot tokenizer must not be selected";
        fs::write(old_root.join("tokenizer.json"), stale_tokenizer).unwrap();
        let source_args = [
            "--gguf-file".into(),
            filename.into(),
            "--semantic-source".into(),
            format!("{metadata_repo}@{REVISION}"),
            "--effective-config-json".into(),
            cache.path().join("effective.json").display().to_string(),
            "--decision-trace-jsonl".into(),
            cache.path().join("decisions.jsonl").display().to_string(),
        ];
        let model = format!("{weights_repo}@{REVISION}");
        let output = invoke_with_sources(&hub, &model, cache.path(), serve, &source_args).await;
        assert!(
            !output.status.success(),
            "fixture deliberately omits GGUF tensors"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("token_embd") || stderr.contains("model.embed_tokens"),
            "metadata must resolve and reach the missing-embedding loader error: {stderr}"
        );
        let wanted_root = repo_root.join("snapshots").join(REVISION);
        for (name, bytes) in hub::sidecar_files() {
            assert_eq!(fs::read(wanted_root.join(&name)).unwrap(), bytes);
            assert!(hub.requested("GET", metadata_repo, &name));
        }
        assert_eq!(
            fs::read(wanted_root.join("generation_config.json")).unwrap(),
            generation
        );
        assert!(!wanted_root.join("model.safetensors").exists());
        assert!(!hub.requested("GET", metadata_repo, "model.safetensors"));
        assert_eq!(
            fs::read_to_string(repo_root.join("refs/main")).unwrap(),
            old_revision
        );
        assert_eq!(
            fs::read(old_root.join("tokenizer.json")).unwrap(),
            stale_tokenizer
        );
        assert!(hub
            .request_paths()
            .iter()
            .all(|path| !path.contains("/main") && !path.contains(&old_revision)));
        assert_legacy_pinned_identity(
            cache.path(),
            &model,
            weights_repo,
            metadata_repo,
            metadata_repo,
        );
        if !serve {
            assert_loader_failure_and_clean_stdout(output);
        }
    }
}

#[tokio::test]
async fn run_and_serve_download_independent_tokenizer_pin_with_generation_metadata() {
    let weights_repo = "fixture/separate-weights";
    let semantic_repo = "fixture/separate-semantic";
    let tokenizer_repo = "fixture/separate-tokenizer";
    let filename = "model-Q4_K_M.gguf";
    for serve in [false, true] {
        let cache = tempfile::tempdir().unwrap();
        let mut tokenizer_files = hub::sidecar_files();
        let semantic_files = [(
            "config.json".into(),
            tokenizer_files.remove("config.json").unwrap(),
        )]
        .into();
        let generation = br#"{"eos_token_id":2}"#;
        tokenizer_files.insert("generation_config.json".into(), generation.to_vec());
        tokenizer_files.insert(
            "model.safetensors".into(),
            b"must not be downloaded".to_vec(),
        );
        let hub = Hub::start(
            [
                (
                    weights_repo.to_owned(),
                    [(filename.to_owned(), hub::gguf_without_weights())].into(),
                ),
                (semantic_repo.to_owned(), semantic_files),
                (tokenizer_repo.to_owned(), tokenizer_files),
            ]
            .into(),
        )
        .await;
        let model = format!("{weights_repo}@{REVISION}");
        let source_args = [
            "--gguf-file".into(),
            filename.into(),
            "--semantic-source".into(),
            format!("{semantic_repo}@{REVISION}"),
            "--tokenizer-source".into(),
            format!("{tokenizer_repo}@{REVISION}"),
            "--effective-config-json".into(),
            cache.path().join("effective.json").display().to_string(),
            "--decision-trace-jsonl".into(),
            cache.path().join("decisions.jsonl").display().to_string(),
        ];
        let output = invoke_with_sources(&hub, &model, cache.path(), serve, &source_args).await;
        assert!(
            !output.status.success(),
            "fixture deliberately omits GGUF tensors"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("token_embd") || stderr.contains("model.embed_tokens"),
            "both metadata roles must resolve before the missing-embedding error: {stderr}"
        );
        let tokenizer_root = cache
            .path()
            .join("hub/models--fixture--separate-tokenizer/snapshots")
            .join(REVISION);
        assert_eq!(
            fs::read(tokenizer_root.join("generation_config.json")).unwrap(),
            generation
        );
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ] {
            assert!(hub.requested("GET", tokenizer_repo, name), "{name}");
        }
        assert!(hub.requested("GET", semantic_repo, "config.json"));
        assert!(!hub.requested("GET", semantic_repo, "tokenizer.json"));
        assert!(!hub.requested("GET", tokenizer_repo, "model.safetensors"));
        assert!(hub
            .request_paths()
            .iter()
            .all(|path| !path.contains("/main")));
        assert_legacy_pinned_identity(
            cache.path(),
            &model,
            weights_repo,
            semantic_repo,
            tokenizer_repo,
        );
        if !serve {
            assert_loader_failure_and_clean_stdout(output);
        }
    }
}

fn assert_failed_startup_diagnostics(cache: &Path) -> serde_json::Value {
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(cache.join("effective.json"))
            .expect("startup must retain actual source evidence before the dummy weight error"),
    )
    .unwrap();
    let failure = &config["startup"];
    assert_eq!(failure["status"], "failed");
    assert_eq!(failure["phase"], "engine_initialization");
    assert_eq!(failure["configuration"], "preliminary");
    let error = failure["error"].as_str().expect("initialization error");
    assert!(error.contains("token_embd") || error.contains("model.embed_tokens"));
    assert!(config["startup_memory_plan"].is_null());
    let trace = fs::read_to_string(cache.join("decisions.jsonl")).unwrap();
    assert!(!trace.trim().is_empty());
    for line in trace.lines() {
        let decision: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(&decision["startup"], failure);
        // Failure metadata is additive: consumers can still decode decisions.
        serde_json::from_value::<ferrum_types::AutoConfigDecision>(decision).unwrap();
    }
    config
}

fn assert_legacy_pinned_identity(
    cache: &Path,
    requested_model: &str,
    weights_repo: &str,
    semantic_repo: &str,
    tokenizer_repo: &str,
) {
    let config = assert_failed_startup_diagnostics(cache);
    assert_eq!(config["execution_resource_authority"], "legacy_engine");
    let identity = &config["resolution_evidence"];
    assert_eq!(identity["requested_model"], requested_model);
    assert_eq!(identity["resolved_model"], weights_repo);
    for (role, repo) in [
        ("weights", weights_repo),
        ("semantic", semantic_repo),
        ("tokenizer", tokenizer_repo),
    ] {
        assert_eq!(identity["original_sources"][role]["kind"], "repository");
        assert_eq!(identity["original_sources"][role]["location"], repo);
        assert_eq!(
            identity["original_sources"][role]["requested_revision"],
            REVISION
        );
        let resolved = &identity["resolved_sources"][role];
        assert_eq!(resolved["canonical_location"], repo);
        assert_eq!(resolved["resolved_revision"], REVISION);
        let files = resolved["files"]
            .as_array()
            .expect("observed file fingerprints");
        assert!(!files.is_empty());
        let root = cache
            .join("hub")
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(REVISION);
        for file in files {
            let bytes = fs::read(root.join(file["relative_path"].as_str().unwrap())).unwrap();
            assert_eq!(file["size_bytes"], bytes.len());
            assert_eq!(file["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        }
    }
    let root = cache
        .join("hub")
        .join(format!("models--{}", tokenizer_repo.replace('/', "--")))
        .join("snapshots")
        .join(REVISION);
    let bytes = fs::read(root.join("tokenizer_config.json")).unwrap();
    let tokenizer: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(identity["template"]["role"], "tokenizer");
    assert_eq!(identity["template"]["source_file"], "tokenizer_config.json");
    assert_eq!(
        identity["template"]["container_sha256"],
        format!("{:x}", Sha256::digest(&bytes))
    );
    assert_eq!(
        identity["template"]["content_sha256"],
        format!(
            "{:x}",
            Sha256::digest(tokenizer["chat_template"].as_str().unwrap().as_bytes())
        )
    );
}

#[tokio::test]
async fn run_and_serve_recover_transient_weight_download_before_reaching_the_loader() {
    let repo = "fixture/recover-weight";
    for serve in [false, true] {
        let cache = tempfile::tempdir().unwrap();
        let files: ModelFiles = [(repo.to_owned(), hub::safetensors_files())].into();
        let hub = Hub::start(files.clone()).await;
        hub.fail_get_once(repo, "model.safetensors");
        let output = invoke(&hub, repo, cache.path(), serve).await;
        assert_downloaded(&hub, cache.path(), &files, true);
        assert_eq!(hub.get_count(repo, "model.safetensors"), 2);
        if serve {
            assert!(
                !output.status.success(),
                "fixture omits actual model weights"
            );
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(
                stderr.contains("model.embed_tokens") || stderr.contains("token_embd"),
                "serve must recover transfer and reach the loader: {stderr}"
            );
        } else {
            assert_loader_failure_and_clean_stdout(output);
        }
    }
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
