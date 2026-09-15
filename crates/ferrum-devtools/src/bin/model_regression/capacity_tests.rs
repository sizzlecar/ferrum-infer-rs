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
