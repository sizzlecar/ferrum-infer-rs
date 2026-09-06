use super::*;
use ferrum_bench_core::release_regression::{
    performance::{performance_check_descriptors, Limits, Workload},
    Backend, Behavior, Entrypoint, EvidenceLayer, ExecutionTarget, Impact, ModelProfile,
    Obligation, ObligationScope, PlanCost, SelectedProfile,
};
use ferrum_types::{ModelOutputProtocol, ModelReasoningProtocol};

fn profile() -> ModelProfile {
    ModelProfile {
        id: "legacy-metal".into(),
        model: "fixture:gguf".into(),
        reasoning_protocol: ModelReasoningProtocol::None,
        available: true,
        estimate: None,
        target: ExecutionTarget {
            architecture: "llama_dense".into(),
            protocol: ModelOutputProtocol::Text,
            backend: Backend::Metal,
            precision: "gguf-q4_k_m".into(),
            execution_path: "legacy-model-executor".into(),
        },
    }
}
fn plan() -> Plan {
    let profile = profile();
    Plan {
        stage: Stage::Release,
        impact: Impact {
            areas: vec![],
            paths: vec![],
            unknown_paths: vec![],
            product_contract_changed: false,
        },
        obligations: vec![Obligation {
            behavior: Behavior::Performance,
            layer: EvidenceLayer::Performance,
            entrypoints: vec![Entrypoint::ServeStream],
            scope: ObligationScope::Profile {
                profile_id: profile.id.clone(),
                target: profile.target.clone(),
            },
            reason: "changed legacy submission".into(),
            checkers: performance_check_descriptors(&[profile.target.clone()])
                .into_iter()
                .map(|c| c.id)
                .collect(),
        }],
        selected: vec![SelectedProfile {
            profile,
            obligations: vec![0],
            reasons: vec![],
        }],
        omitted: vec![],
        gaps: vec![],
        cost: PlanCost::default(),
    }
}
fn document(plan: Plan) -> Value {
    json!({"schema_version":2,"stage":"release", "provenance":{"base":"a".repeat(40),"candidate":"b".repeat(40),"release_base_tag":"v1.2.3"},
        "plan":plan, "performance_tasks":{"runs":[],"unsupported_obligations":[]}})
}
fn policy() -> PerformancePolicy {
    PerformancePolicy {
        workload: Workload {
            input_tokens: 32,
            output_tokens: 16,
            measured_requests: 2,
            warmup_requests: 1,
            repeats: 3,
            seed: 7,
            max_model_len: 128,
        },
        limits: Limits {
            ttft_max_relative_increase: 0.1,
            tpot_max_relative_increase: 0.1,
        },
        runtime_memory_budget_bytes: 1024,
        startup_timeout_secs: 60,
        request_timeout_secs: 30,
        task_timeout_secs: 600,
    }
}
fn source() -> cache::SourceSelector {
    cache::SourceSelector {
        model: profile().model,
        gguf: cache::WeightSelector {
            repository: "owner/weights".into(),
            filename: "model.gguf".into(),
            revision: None,
        },
        tokenizer: cache::RepositorySelector {
            repository: "owner/tokenizer".into(),
            revision: None,
        },
    }
}
fn snapshot(root: &Path, repository: &str, revision: char, files: &[(&str, &[u8])]) -> PathBuf {
    let path = root
        .join("hub")
        .join(format!("models--{}", repository.replace('/', "--")))
        .join("snapshots")
        .join(revision.to_string().repeat(40));
    fs::create_dir_all(&path).unwrap();
    for (name, bytes) in files {
        fs::write(path.join(name), bytes).unwrap();
    }
    path
}
fn cache(root: &Path) {
    snapshot(
        root,
        "owner/weights",
        'c',
        &[("model.gguf", b"GGUF tiny test data")],
    );
    snapshot(
        root,
        "owner/tokenizer",
        'd',
        &[("tokenizer.json", b"{}"), ("tokenizer_config.json", b"{}")],
    );
}
fn write_metadata(root: &Path, role: &str, version: &str, commit: char, binary: char) -> PathBuf {
    let directory = root.join(role);
    fs::create_dir(&directory).unwrap();
    let asset = "ferrum-macos-aarch64.tar.gz";
    let common = json!({"schema_version":1,"asset_name":asset,"asset_sha256":"f".repeat(64),"binary_name":"ferrum","binary_sha256":binary.to_string().repeat(64),
        "release_candidate_sha":commit.to_string().repeat(40),"release_candidate_tag":format!("v{version}-rc.1"),"staging_label":"fixture","workflow_run_id":"1","workflow_run_attempt":"1"});
    let mut abi = common.clone();
    abi["backend"] = json!("metal");
    abi["target_triple"] = json!("aarch64-apple-darwin");
    let mut version_metadata = common;
    version_metadata["version"] = json!(version);
    let path = directory.join(format!("{asset}.abi.json"));
    write_json(&path, &abi).unwrap();
    write_json(
        &directory.join(format!("{asset}.version.json")),
        &version_metadata,
    )
    .unwrap();
    path
}
fn fixture(root: &Path) -> PerformancePrepareArgs {
    cache(root);
    let args = PerformancePrepareArgs {
        plan: root.join("plan.json"),
        profile_id: profile().id,
        candidate_abi: write_metadata(root, "new", "1.2.4", 'b', '2'),
        baseline_abi: write_metadata(root, "old", "1.2.3", 'a', '1'),
        policy: root.join("policy.json"),
        hf_cache: root.join("hub"),
        source_selector: root.join("source.json"),
        output_dir: root.join("tasks"),
    };
    write_json(&args.plan, &document(plan())).unwrap();
    write_json(&args.policy, &policy()).unwrap();
    write_json(&args.source_selector, &source()).unwrap();
    args
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

#[tokio::test]
async fn prepare_freezes_recomputed_task_and_preserves_ci_origin_without_model_execution() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path());
    // Published metadata hashes cover exact bytes, including indentation and final newline.
    let baseline_version = args
        .baseline_abi
        .with_file_name("ferrum-macos-aarch64.tar.gz.version.json");
    let mut originals = Vec::new();
    for path in [&args.baseline_abi, &baseline_version] {
        let value: Value = read_json(path).unwrap();
        let mut serializer = serde_json::Serializer::with_formatter(
            Vec::new(),
            serde_json::ser::PrettyFormatter::with_indent(b"\t"),
        );
        serde::Serialize::serialize(&value, &mut serializer).unwrap();
        let mut bytes = serializer.into_inner();
        bytes.extend_from_slice(b"\n\n");
        fs::write(path, &bytes).unwrap();
        originals.push((path.file_name().unwrap().to_owned(), bytes));
    }
    fs::create_dir(&args.output_dir).unwrap();
    fs::write(args.output_dir.join("ci-origin.json"), b"preserved").unwrap();
    prepare(args).await.unwrap();
    let directory = root.path().join("tasks/legacy-metal");
    let expected: ExpectedPerformanceRun =
        read_json(&directory.join("expected-task.json")).unwrap();
    expected.validate().unwrap();
    assert_eq!(expected.obligations, vec![0]);
    assert_eq!(expected.candidate_sha256, "2".repeat(64));
    assert_eq!(expected.client_sha256, expected.candidate_sha256);
    assert_eq!(expected.baseline_version, "1.2.3");
    assert_eq!(expected.policy, policy());
    assert_eq!(
        fs::read(root.path().join("tasks/ci-origin.json")).unwrap(),
        b"preserved"
    );
    let evidence: Value = read_json(&directory.join("prepare-evidence.json")).unwrap();
    assert_eq!(
        evidence["resolved_source"]["gguf"]["revision"],
        "c".repeat(40)
    );
    assert_eq!(evidence["models_executed"], false);
    assert_eq!(evidence["archive_bytes_verified"], false);
    assert!(!directory.join("model.gguf").exists());
    assert!(root
        .path()
        .join("tasks/baseline/ferrum-macos-aarch64.tar.gz.abi.json")
        .is_file());
    for (name, original) in originals {
        let frozen = root.path().join("tasks/baseline").join(name);
        assert_eq!(fs::read(&frozen).unwrap(), original);
        preserve_metadata(&frozen, &original).unwrap();
        let value: Value = serde_json::from_slice(&original).unwrap();
        let reformatted = serde_json::to_vec_pretty(&value).unwrap();
        assert!(preserve_metadata(&frozen, &reformatted).is_err());
        assert_eq!(fs::read(&frozen).unwrap(), original);
    }
    let original = fs::read(directory.join("expected-task.json")).unwrap();
    let mut again = fixture_other_input(root.path());
    again.output_dir = root.path().join("tasks");
    assert!(prepare(again).await.unwrap_err().contains("must be new"));
    assert_eq!(
        fs::read(directory.join("expected-task.json")).unwrap(),
        original
    );
}
// Reuse the existing fixture inputs without creating or replacing them.
fn fixture_other_input(root: &Path) -> PerformancePrepareArgs {
    PerformancePrepareArgs {
        plan: root.join("plan.json"),
        profile_id: profile().id,
        candidate_abi: root.join("new/ferrum-macos-aarch64.tar.gz.abi.json"),
        baseline_abi: root.join("old/ferrum-macos-aarch64.tar.gz.abi.json"),
        policy: root.join("policy.json"),
        hf_cache: root.join("hub"),
        source_selector: root.join("source.json"),
        output_dir: root.join("other-tasks"),
    }
}
#[test]
fn prepare_ignores_saved_schedule_and_rejects_missing_or_unsupported_real_assignment() {
    let mut doc: PlanDocument = serde_json::from_value(document(plan())).unwrap();
    assert_eq!(
        requirements(&doc, &profile().id).unwrap().obligations,
        vec![0]
    );
    doc.plan.selected.clear();
    assert!(requirements(&doc, &profile().id)
        .unwrap_err()
        .contains("unsupported"));
    doc.plan = plan();
    doc.plan.obligations[0].entrypoints.push(Entrypoint::Run);
    assert!(requirements(&doc, &profile().id)
        .unwrap_err()
        .contains("unsupported"));
    doc.plan = plan();
    assert!(requirements(&doc, "missing").is_err());
    doc.stage = Stage::PullRequest;
    assert!(requirements(&doc, &profile().id).is_err());
}
#[tokio::test]
async fn prepare_rejects_wrong_source_commit_version_abi_and_model_before_output() {
    for case in ["commit", "tag", "abi", "model", "policy"] {
        let root = tempfile::tempdir().unwrap();
        let args = fixture(root.path());
        match case {
            "commit" => {
                let mut v = document(plan());
                v["provenance"]["candidate"] = json!("e".repeat(40));
                write_json(&args.plan, &v).unwrap();
            }
            "tag" => {
                let mut v = document(plan());
                v["provenance"]["release_base_tag"] = json!("v1.2.2");
                write_json(&args.plan, &v).unwrap();
            }
            "abi" => {
                let mut v: Value = read_json(&args.candidate_abi).unwrap();
                v["target_triple"] = json!("x86_64-apple-darwin");
                write_json(&args.candidate_abi, &v).unwrap();
            }
            "model" => {
                let mut v = source();
                v.model = "unrelated:model".into();
                write_json(&args.source_selector, &v).unwrap();
            }
            "policy" => {
                let mut p = policy();
                p.workload.warmup_requests = 0;
                write_json(&args.policy, &p).unwrap();
            }
            _ => unreachable!(),
        }
        let output = args.output_dir.clone();
        assert!(prepare(args).await.is_err(), "{case}");
        assert!(!output.exists(), "{case}");
    }
}
#[test]
fn cache_requires_unambiguous_complete_revision_and_never_falls_back_from_main() {
    let root = tempfile::tempdir().unwrap();
    cache(root.path());
    let selector = source();
    cache::resolve(root.path(), &selector, &profile(), deadline()).unwrap();
    snapshot(
        root.path(),
        "owner/weights",
        'e',
        &[("model.gguf", b"other bytes")],
    );
    assert!(
        cache::resolve(root.path(), &selector, &profile(), deadline())
            .unwrap_err()
            .contains("2 complete snapshots")
    );
    let refs = root.path().join("hub/models--owner--weights/refs");
    fs::create_dir(&refs).unwrap();
    fs::write(refs.join("main"), "e".repeat(40)).unwrap();
    let (_, evidence) = cache::resolve(root.path(), &selector, &profile(), deadline()).unwrap();
    assert_eq!(evidence.gguf.revision, "e".repeat(40));
    fs::write(refs.join("main"), "f".repeat(40)).unwrap();
    assert!(
        cache::resolve(root.path(), &selector, &profile(), deadline())
            .unwrap_err()
            .contains("incomplete")
    );
    let mut explicit = source();
    explicit.gguf.revision = Some("c".repeat(40));
    let (_, evidence) = cache::resolve(root.path(), &explicit, &profile(), deadline()).unwrap();
    assert_eq!(evidence.gguf.revision, "c".repeat(40));
    explicit.gguf.revision = Some("main".into());
    assert!(cache::resolve(root.path(), &explicit, &profile(), deadline()).is_err());
}
#[test]
#[cfg(unix)]
fn cache_follows_weight_blob_symlink_without_copying_and_pins_sidecar_changes() {
    let root = tempfile::tempdir().unwrap();
    cache(root.path());
    let snapshot = root
        .path()
        .join("hub/models--owner--weights/snapshots")
        .join("c".repeat(40));
    let blob = root.path().join("blob");
    fs::rename(snapshot.join("model.gguf"), &blob).unwrap();
    std::os::unix::fs::symlink(&blob, snapshot.join("model.gguf")).unwrap();
    let (manifest, _) = cache::resolve(root.path(), &source(), &profile(), deadline()).unwrap();
    manifest.validate(&profile(), deadline()).unwrap();
    assert!(fs::symlink_metadata(&manifest.gguf.path)
        .unwrap()
        .file_type()
        .is_symlink());
    let identity = manifest.identity().unwrap();
    fs::write(manifest.tokenizer_dir.join("tokenizer_config.json"), b"[]").unwrap();
    assert!(manifest.validate(&profile(), deadline()).is_err());
    let (changed, _) = cache::resolve(root.path(), &source(), &profile(), deadline()).unwrap();
    assert_ne!(changed.identity().unwrap(), identity);
}
#[test]
fn cache_rejects_source_paths_and_incomplete_tokenizer_metadata() {
    let root = tempfile::tempdir().unwrap();
    cache(root.path());
    for bad in [
        "../model.gguf",
        "/tmp/model.gguf",
        "model.gguf?download=true",
    ] {
        let mut selector = source();
        selector.gguf.filename = bad.into();
        assert!(cache::resolve(root.path(), &selector, &profile(), deadline()).is_err());
    }
    let mut selector = source();
    selector.tokenizer.repository = "owner/../tokenizer".into();
    assert!(cache::resolve(root.path(), &selector, &profile(), deadline()).is_err());
    fs::remove_file(
        root.path()
            .join("hub/models--owner--tokenizer/snapshots")
            .join("d".repeat(40))
            .join("tokenizer_config.json"),
    )
    .unwrap();
    assert!(cache::resolve(root.path(), &source(), &profile(), deadline()).is_err());
}
