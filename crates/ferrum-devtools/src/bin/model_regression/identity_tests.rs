use super::*;
use clap::Parser;
use serde_json::json;

fn args(expected_backend: &str) -> Args {
    Args::try_parse_from([
        "model-regression",
        "--ferrum-bin",
        "fixture-ferrum",
        "--model",
        "fixture:model-alias",
        "--backend",
        expected_backend,
        "--report-dir",
        "fixture-report",
    ])
    .unwrap()
}

fn ready(actual_backend: &str) -> Value {
    json!({
        "event": "ready", "requested_model": "fixture:model-alias",
        "resolved_model": "/cache/model", "backend": actual_backend
    })
}

fn health(actual_backend: &str) -> Value {
    json!({"status": "healthy", "auto_config": {"hardware_capabilities": {"backend": actual_backend}}})
}

#[test]
fn pinned_source_capture_requires_the_same_process_configuration() {
    let directory = tempfile::tempdir().unwrap();
    let mut args = args("cuda");
    args.report_dir = directory.path().to_owned();
    let revision = "a".repeat(40);
    args.model = format!("fixture/model@{revision}");
    let mut evidence = json!({"schema_version": 1, "resolved_model": "fixture/model", "requested_model": args.model,
        "original_sources": {}, "resolved_sources": {}});
    for role in ["weights", "semantic", "tokenizer"] {
        evidence["original_sources"][role] = json!({
            "kind": "repository", "location": "fixture/model", "requested_revision": revision});
        evidence["resolved_sources"][role] = json!({
            "canonical_location": "fixture/model", "resolved_revision": revision,
            "files": [{"relative_path": format!("{role}.bin"), "size_bytes": 16, "sha256": "c".repeat(64)}]});
    }
    for name in ["run-basic", "serve"] {
        assert!(source_evidence(&args, name).is_err());
        let path = args
            .report_dir
            .join(format!("{name}.effective-config.json"));
        crate::write_json(&path, &json!({"resolution_evidence": evidence})).unwrap();
        assert_eq!(source_evidence(&args, name).unwrap(), evidence);
        let mut wrong = evidence.clone();
        wrong["resolved_sources"]["weights"]["resolved_revision"] = json!("b".repeat(40));
        crate::write_json(&path, &json!({"resolution_evidence": wrong})).unwrap();
        assert!(source_evidence(&args, name).is_err());
    }
}

#[test]
fn selected_gguf_is_forwarded_to_both_entrypoints() {
    let mut selected = args("cuda");
    selected.model = format!("quantizer/model@{}", "a".repeat(40));
    selected.gguf_file = Some("weights/model.gguf".into());
    for entrypoint in ["run", "serve"] {
        assert!(selected
            .common_args(entrypoint)
            .windows(2)
            .any(|args| args == ["--gguf-file", "weights/model.gguf"]));
        assert!(!args("cuda")
            .common_args(entrypoint)
            .iter()
            .any(|arg| arg == "--gguf-file"));
    }
}

#[test]
fn every_gguf_child_config_must_match_retained_task_metadata_expectations() {
    let mut selected = args("cuda");
    selected.model = format!("quantizer/model@{}", "a".repeat(40));
    selected.gguf_file = Some("model.gguf".into());
    selected.source_expectation = Some(ferrum_bench_core::release_regression::GgufSourceProfile {
        filename: "model.gguf".into(),
        semantic_source: format!("author/model@{}", "b".repeat(40)),
        tokenizer_source: None,
    });
    let mut identity = json!({"schema_version":1,"requested_model":selected.model,"resolved_model":"quantizer/model",
        "original_sources":{},"resolved_sources":{}});
    for (role, repo, revision, file) in [
        ("weights", "quantizer/model", "a".repeat(40), "model.gguf"),
        ("semantic", "author/model", "b".repeat(40), "config.json"),
        (
            "tokenizer",
            "author/model",
            "b".repeat(40),
            "tokenizer.json",
        ),
    ] {
        identity["original_sources"][role] = json!({"kind":"repository","location":repo,
            "requested_revision":if role=="weights"{Some(&revision)}else{None}});
        identity["resolved_sources"][role] = json!({"canonical_location":repo,"resolved_revision":revision,
            "files":[{"relative_path":file,"size_bytes":16,"sha256":"c".repeat(64)}]});
    }
    validate_source_config(&selected, &json!({"resolution_evidence":identity})).unwrap();
    for role in ["semantic", "tokenizer"] {
        let mut changed = identity.clone();
        changed["resolved_sources"][role]["resolved_revision"] = json!("d".repeat(40));
        assert!(
            validate_source_config(&selected, &json!({"resolution_evidence":changed})).is_err()
        );
    }
}

#[test]
fn actual_run_and_serve_backend_representations_match_requested_backend() {
    for (expected, run_backend, serve_backend) in [
        ("cpu", "CPU", "cpu"),
        ("metal", "Metal", "metal"),
        ("cuda", "CUDA(0)", "cuda"),
        ("cuda", "CUDA(12)", "cuda"),
    ] {
        validate_run(&args(expected), &ready(run_backend)).unwrap();
        validate_serve(&args(expected), &health(serve_backend)).unwrap();
    }
}

#[test]
fn fallback_or_missing_runtime_backend_is_rejected() {
    let expected = args("cuda");
    for actual in [
        "CPU",
        "metal",
        "unknown",
        "",
        "CUDA()",
        "CUDA(-1)",
        "CUDA(0)cpu",
        "cuda:0",
    ] {
        assert!(
            validate_run(&expected, &ready(actual)).is_err(),
            "accepted {actual:?}"
        );
        assert!(
            validate_serve(&expected, &health(actual)).is_err(),
            "accepted {actual:?}"
        );
    }
    for actual in [Value::Null, json!(0), json!({"backend": "cuda"})] {
        let mut record = ready("CUDA(0)");
        record["backend"] = actual.clone();
        assert!(validate_run(&expected, &record).is_err());
        let mut record = health("cuda");
        record["auto_config"]["hardware_capabilities"]["backend"] = actual;
        assert!(validate_serve(&expected, &record).is_err());
    }
    // A top-level declaration must not substitute for the actual health field.
    assert!(validate_serve(&expected, &json!({"status": "healthy", "backend": "cuda"})).is_err());
    assert!(validate_serve(
        &expected,
        &json!({
            "status": "healthy", "backend": "cuda", "auto_config": {"hardware_capabilities": {"backend": "cpu"}}
        })
    )
    .is_err());
}

#[test]
fn run_must_report_the_requested_alias_even_when_resolved_source_matches() {
    let expected = args("cpu");
    for wrong in [Value::Null, json!("another:alias"), json!("/cache/model")] {
        let mut record = ready("CPU");
        record["requested_model"] = wrong;
        assert!(validate_run(&expected, &record).is_err());
    }
    let mut record = ready("CPU");
    record["event"] = json!("assistant");
    assert!(validate_run(&expected, &record).is_err());
}

#[test]
fn successful_http_status_cannot_replace_healthy_runtime_status() {
    let expected = args("cuda");
    for status in [
        Value::Null,
        json!("unhealthy"),
        json!("starting"),
        json!(true),
    ] {
        let mut record = health("cuda");
        record["status"] = status;
        assert!(validate_serve(&expected, &record).is_err());
    }
}

#[test]
fn functional_capacity_uses_public_flags_in_both_entrypoints_and_actual_health() {
    let mut functional = args("metal");
    functional.context_tokens = Some(2048);
    functional.max_num_seqs = Some(1);
    functional
        .runtime_capacity()
        .unwrap()
        .validate(functional.max_tokens)
        .unwrap();
    for entrypoint in ["run", "serve"] {
        let command = functional.common_args(entrypoint);
        assert_eq!(
            command.iter().any(|arg| arg == "--no-context-shift"),
            entrypoint == "run"
        );
        for pair in [
            ["--kv-capacity", "2048"],
            ["--max-model-len", "2048"],
            ["--max-num-seqs", "1"],
        ] {
            assert!(command.windows(2).any(|args| args == pair));
        }
        let defaults = args("metal").common_args(entrypoint);
        assert!(!defaults.iter().any(|arg| matches!(
            arg.as_str(),
            "--kv-capacity" | "--max-model-len" | "--max-num-seqs" | "--no-context-shift"
        )));
    }
    let mut observed = health("metal");
    observed["auto_config"]["selected_max_model_len"] = json!(2048);
    observed["auto_config"]["selected_kv_capacity"] = json!(2048);
    observed["auto_config"]["selected_max_sequences"] = json!(1);
    validate_serve(&functional, &observed).unwrap();
    for (field, value) in [
        ("selected_max_model_len", json!(4096)),
        ("selected_kv_capacity", json!(512)),
        ("selected_max_sequences", json!(2)),
    ] {
        let mut mismatched = observed.clone();
        mismatched["auto_config"][field] = value;
        assert!(validate_serve(&functional, &mismatched).is_err());
    }
    functional.context_tokens = Some(functional.max_tokens);
    assert!(functional
        .runtime_capacity()
        .unwrap()
        .validate(functional.max_tokens)
        .is_err());
}

#[test]
fn capacity_cli_requires_both_positive_limits() {
    let base = [
        "model-regression",
        "--ferrum-bin",
        "fixture",
        "--model",
        "fixture:model",
        "--backend",
        "metal",
        "--report-dir",
        "fixture",
    ];
    for extra in [
        vec!["--context-tokens", "2048"],
        vec!["--max-num-seqs", "1"],
        vec!["--context-tokens", "0", "--max-num-seqs", "1"],
    ] {
        assert!(Args::try_parse_from(base.into_iter().chain(extra)).is_err());
    }
    assert!(Args::try_parse_from(base.into_iter().chain([
        "--context-tokens",
        "2048",
        "--max-num-seqs",
        "1"
    ]))
    .is_ok());
}
