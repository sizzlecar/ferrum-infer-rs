use super::*;
use ferrum_bench_core::release_regression::{
    model_tasks::DEFAULT_CUDA_FUNCTIONAL_CAPACITY, Backend, ExecutionTarget, ModelProfile,
};

fn parse(extra: &[&str]) -> Result<Args, clap::Error> {
    Args::try_parse_from(
        [
            "model-regression",
            "--ferrum-bin",
            "fixture",
            "--model",
            "fixture/model",
            "--backend",
            "cuda",
            "--profile-id",
            "cuda-small",
            "--report-dir",
            "fixture-report",
        ]
        .into_iter()
        .chain(extra.iter().copied()),
    )
}

#[test]
fn memory_budget_cli_requires_positive_bytes_and_explicit_capacity() {
    for extra in [
        vec!["--runtime-memory-budget-bytes", "4294967296"],
        vec![
            "--context-tokens",
            "2048",
            "--runtime-memory-budget-bytes",
            "4294967296",
        ],
        vec![
            "--context-tokens",
            "2048",
            "--max-num-seqs",
            "1",
            "--runtime-memory-budget-bytes",
            "0",
        ],
        vec![
            "--context-tokens",
            "2048",
            "--max-num-seqs",
            "1",
            "--runtime-memory-budget-bytes",
            "-1",
        ],
    ] {
        assert!(parse(&extra).is_err());
    }
    assert!(parse(&[]).unwrap().runtime_capacity().is_none());
}

#[test]
fn memory_budget_is_a_public_argument_for_run_and_serve_without_changing_device_selection() {
    let mut args = parse(&[
        "--context-tokens",
        "2048",
        "--max-num-seqs",
        "1",
        "--runtime-memory-budget-bytes",
        "4294967296",
    ])
    .unwrap();
    assert_eq!(
        args.runtime_capacity(),
        Some(DEFAULT_CUDA_FUNCTIONAL_CAPACITY)
    );
    for automatic in [false, true] {
        args.use_default_backend = automatic;
        for entrypoint in ["run", "serve"] {
            let words = args.common_args(entrypoint);
            assert_eq!(words.first().map(String::as_str), Some(entrypoint));
            assert!(words
                .windows(2)
                .any(|pair| pair == ["--runtime-memory-budget-bytes", "4294967296"]));
            assert_eq!(
                words.windows(2).any(|pair| pair == ["--backend", "cuda"]),
                !automatic
            );
            assert_eq!(
                words.iter().any(|word| word == "--no-context-shift"),
                entrypoint == "run"
            );
        }
    }
    for args in [
        parse(&[]).unwrap(),
        parse(&["--context-tokens", "2048", "--max-num-seqs", "1"]).unwrap(),
    ] {
        for entrypoint in ["run", "serve"] {
            assert!(!args
                .common_args(entrypoint)
                .iter()
                .any(|word| word == "--runtime-memory-budget-bytes"));
        }
        assert!(serde_json::to_value(&args).unwrap()["runtime_memory_budget_bytes"].is_null());
    }
}

#[test]
fn serialized_runner_options_bind_the_prepared_memory_budget_and_reject_drift() {
    let args = parse(&[
        "--context-tokens",
        "2048",
        "--max-num-seqs",
        "1",
        "--runtime-memory-budget-bytes",
        "4294967296",
    ])
    .unwrap();
    let expected = ExpectedModelRun {
        profile: ModelProfile {
            id: args.profile_id.clone().unwrap(),
            model: args.model.clone(),
            target: ExecutionTarget {
                architecture: "dense".into(),
                protocol: ferrum_types::ModelOutputProtocol::Text,
                precision: "bf16".into(),
                backend: Backend::Cuda,
                execution_path: "safetensors".into(),
            },
            gguf: None,
            reasoning_protocol: ferrum_types::ModelReasoningProtocol::PromptOpened,
            available: true,
            estimate: None,
        },
        binary_sha256: "a".repeat(64),
        version: "0.10.0".into(),
        checks: args.checks.clone(),
        disable_thinking: args.disable_thinking,
        use_default_backend: args.use_default_backend,
        max_tokens: args.max_tokens,
        runtime_capacity: args.runtime_capacity(),
        reasoning_alias_replay: args.reasoning_alias_replay,
        stop_prompt: args.stop_prompt.clone(),
    };
    let options = serde_json::to_value(&args).unwrap();
    verify_model_options(&expected, &options).unwrap();
    assert_eq!(
        options["runtime_memory_budget_bytes"],
        json!(4294967296_u64)
    );
    for value in [
        Value::Null,
        json!(0),
        json!(8589934592_u64),
        json!("4294967296"),
    ] {
        let mut changed = options.clone();
        changed["runtime_memory_budget_bytes"] = value;
        assert!(verify_model_options(&expected, &changed)
            .unwrap_err()
            .iter()
            .any(|issue| issue.contains("runtime_memory_budget_bytes")));
    }
    let mut missing = options;
    missing
        .as_object_mut()
        .unwrap()
        .remove("runtime_memory_budget_bytes");
    assert!(verify_model_options(&expected, &missing).is_err());
}

#[test]
fn sequence_fit_policy_is_explicit_for_both_entrypoints_and_excludes_prepared_tasks() {
    for value in ["full-input-must-fit", "immediate-only"] {
        let args = parse(&["--sequence-fit-policy", value]).unwrap();
        for entrypoint in ["run", "serve"] {
            assert!(args
                .common_args(entrypoint)
                .windows(2)
                .any(|pair| pair == ["--sequence-fit-policy", value]));
        }
        assert!(parse(&[
            "--sequence-fit-policy",
            value,
            "--expected-task",
            "task.json"
        ])
        .is_err());
    }
    assert!(parse(&["--sequence-fit-policy", "unknown"]).is_err());
    assert!(parse(&[]).unwrap().sequence_fit_policy.is_none());
}

#[test]
fn scheduled_token_limit_is_positive_explicit_and_forwarded_to_both_entrypoints() {
    let args = parse(&["--max-num-batched-tokens", "128"]).unwrap();
    for entrypoint in ["run", "serve"] {
        assert!(args
            .common_args(entrypoint)
            .windows(2)
            .any(|pair| pair == ["--max-num-batched-tokens", "128"]));
        assert!(!parse(&[])
            .unwrap()
            .common_args(entrypoint)
            .iter()
            .any(|word| word == "--max-num-batched-tokens"));
    }
    for value in ["0", "-1", "many"] {
        assert!(parse(&["--max-num-batched-tokens", value]).is_err());
    }
    assert!(parse(&[
        "--max-num-batched-tokens",
        "128",
        "--expected-task",
        "task.json"
    ])
    .is_err());
    assert_eq!(
        serde_json::to_value(&args).unwrap()["max_num_batched_tokens"],
        128
    );
}

#[test]
fn scheduled_token_evidence_requires_actual_executor_even_without_fit_policy() {
    let args = parse(&["--max-num-batched-tokens", "128"]).unwrap();
    let config =
        json!({"entries": [{"key": "FERRUM_MAX_BATCHED_TOKENS", "effective_value": "128"}]});
    let health = json!({"auto_config": config, "cache": {"prefix_cache": {
        "source": "vnext-native-sequence-checkpoint-cache",
        "runtime_admission_policy": {"maximum_scheduled_tokens": 128}
    }}});
    capacity::validate_runtime_policy(&args, &health).unwrap();
    for value in [Value::Null, json!(256), json!("128")] {
        let mut changed = health.clone();
        changed["cache"]["prefix_cache"]["runtime_admission_policy"]["maximum_scheduled_tokens"] =
            value;
        assert!(capacity::validate_runtime_policy(&args, &changed).is_err());
    }
    assert!(capacity::validate_runtime_policy(&args, &json!({"auto_config": config})).is_err());
    for config in [
        json!({"entries": []}),
        json!({"entries": [{"key": "FERRUM_MAX_BATCHED_TOKENS", "effective_value": "256"}]}),
        json!({"entries": [config["entries"][0], config["entries"][0]]}),
    ] {
        assert!(capacity::validate_effective_policy(&args, &config).is_err());
    }
    // Existing prepared tasks that select neither option retain their original
    // execution authority rather than inheriting this vNext-only probe contract.
    let default_args = parse(&[]).unwrap();
    capacity::validate_runtime_policy(&default_args, &json!({})).unwrap();
    capacity::validate_effective_policy(&default_args, &json!({})).unwrap();
}

#[test]
fn admission_policy_evidence_rejects_echoes_missing_runtime_and_budget_drift() {
    let args = parse(&[
        "--sequence-fit-policy",
        "full-input-must-fit",
        "--context-tokens",
        "2048",
        "--max-num-seqs",
        "1",
        "--runtime-memory-budget-bytes",
        "4000",
        "--max-num-batched-tokens",
        "128",
    ])
    .unwrap();
    let config = json!({"entries": [
        {"key": "FERRUM_SEQUENCE_FIT_POLICY", "effective_value": "full-input-must-fit"},
        {"key": "FERRUM_MAX_BATCHED_TOKENS", "effective_value": "128"}
    ]});
    capacity::validate_effective_policy(&args, &config).unwrap();
    let mut health = json!({"auto_config": config, "cache": {"prefix_cache": {
        "source": "vnext-native-sequence-checkpoint-cache",
        "runtime_admission_policy": {"sequence_fit_policy": "full_input_must_fit", "maximum_scheduled_tokens": 128},
        "runtime_memory_policy": {"capacity_bytes": 5000, "reserve_bytes": 1000},
        "dynamic_pools": {"budget_device_wide_usable_ceiling_bytes": 4000,
            "effective_device_usable_ceiling_bytes": 4000}
    }}});
    capacity::validate_runtime_policy(&args, &health).unwrap();
    for pointer in [
        "/cache/prefix_cache/runtime_memory_policy/reserve_bytes",
        "/cache/prefix_cache/dynamic_pools/budget_device_wide_usable_ceiling_bytes",
        "/cache/prefix_cache/dynamic_pools/effective_device_usable_ceiling_bytes",
    ] {
        let mut changed = health.clone();
        *changed.pointer_mut(pointer).unwrap() = json!(3999);
        assert!(
            capacity::validate_runtime_policy(&args, &changed).is_err(),
            "{pointer}"
        );
    }
    for pointer in [
        "/cache/prefix_cache/source",
        "/cache/prefix_cache/runtime_admission_policy/sequence_fit_policy",
        "/auto_config/entries/0/effective_value",
    ] {
        let mut changed = health.clone();
        *changed.pointer_mut(pointer).unwrap() = Value::Null;
        assert!(
            capacity::validate_runtime_policy(&args, &changed).is_err(),
            "{pointer}"
        );
    }
    health["cache"]["prefix_cache"]["runtime_admission_policy"]["sequence_fit_policy"] =
        json!("immediate_only");
    assert!(capacity::validate_runtime_policy(&args, &health).is_err());
    assert!(capacity::validate_runtime_policy(&args, &json!({"auto_config": config})).is_err());
    assert!(capacity::validate_effective_policy(
        &args,
        &json!({"entries": [config["entries"][0], config["entries"][0]]})
    )
    .is_err());
}
