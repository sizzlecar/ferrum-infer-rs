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
