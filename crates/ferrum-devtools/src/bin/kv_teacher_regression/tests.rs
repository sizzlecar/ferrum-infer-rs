use super::*;
use ferrum_types::KvStorageFormat;

fn args() -> Args {
    Args::try_parse_from([
        "kv_teacher_regression",
        "--ferrum-bin",
        "ferrum",
        "--checkpoint-diff-bin",
        "checkpoint_diff",
        "--model",
        "/fixture/model",
        "--backend",
        "metal",
        "--prompt-file",
        "prompt.txt",
        "--report-dir",
        "report",
        "--context-tokens",
        "64",
        "--max-tokens",
        "2",
        "--mean-delta-nll-limit",
        "0.03",
        "--max-delta-nll-limit",
        "0.30",
        "--mean-kl-limit",
        "0.02",
        "--max-kl-limit",
        "0.15",
    ])
    .unwrap()
}

#[test]
#[cfg(unix)]
fn local_gguf_symlink_survives_teacher_normalization_and_command_generation() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let blob = root.join("content-address");
    fs::write(&blob, b"fixture").unwrap();
    let selected = root.join("selected.gguf");
    std::os::unix::fs::symlink("content-address", &selected).unwrap();
    let mut args = args();
    args.ferrum_bin = blob.clone();
    args.checkpoint_diff_bin = blob.clone();
    args.prompt_file = blob;
    args.model = selected.to_str().unwrap().to_owned();
    args.report_dir = root.join("report");
    args.normalize().unwrap();
    for dtype in ["fp16", "int8"] {
        let words = args.run_args("fixture", dtype, "prompt", None);
        assert_eq!(words[1], selected.to_str().unwrap());
        assert_eq!(fs::read(&words[1]).unwrap(), b"fixture");
    }
}

fn records() -> Vec<Value> {
    let mut records = vec![
        json!({"event":"ready","requested_model":"/fixture/model","backend":"Metal"}),
        json!({"event":"user","request_id":"request","turn":0,"content":"A prompt"}),
        json!({"event":"assistant_delta","request_id":"request","turn":0,"index":0,"token_id":3}),
        json!({"event":"assistant_delta","request_id":"request","turn":0,"index":1,"token_id":5}),
        json!({"event":"assistant","request_id":"request","turn":0,"finish_reason":"length",
            "n_tokens":2,"chunk_count":2,"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}),
        json!({"event":"exit","reason":"one_shot_complete"}),
    ];
    for record in &mut records {
        record["schema_version"] = json!(2);
        record["session_id"] = json!("session");
        record["history_epoch"] = json!(0);
    }
    records
}

fn jsonl(records: &[Value]) -> String {
    records
        .iter()
        .map(|r| serde_json::to_string(r).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn seed_history_requires_every_contiguous_token_and_complete_length_usage() {
    let args = args();
    let valid = records();
    assert_eq!(
        evidence::parse_generation(&jsonl(&valid), &args, true)
            .unwrap()
            .tokens,
        [3, 5]
    );
    for (record, field, replacement) in [
        (2, "token_id", Value::Null),
        (2, "token_id", json!(u64::MAX)),
        (3, "index", json!(2)),
        (3, "request_id", json!("other")),
        (4, "finish_reason", json!("stop")),
        (4, "n_tokens", json!(1)),
        (
            4,
            "usage",
            json!({"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}),
        ),
        (
            4,
            "usage",
            json!({"prompt_tokens":63,"completion_tokens":2,"total_tokens":65}),
        ),
        (5, "reason", json!("eof")),
        (0, "backend", json!("CUDA(0)")),
    ] {
        let mut invalid = valid.clone();
        invalid[record][field] = replacement;
        assert!(
            evidence::parse_generation(&jsonl(&invalid), &args, true).is_err(),
            "accepted {record}.{field}"
        );
    }
    let mut missing = valid.clone();
    missing.remove(3);
    assert!(evidence::parse_generation(&jsonl(&missing), &args, true).is_err());
    assert!(evidence::parse_generation(&jsonl(&valid[..5]), &args, true).is_err());
    assert!(evidence::parse_generation("not json", &args, true).is_err());
}

fn config(format: KvStorageFormat) -> Value {
    let hash = "a".repeat(64);
    let binding = json!({"source_file":"metadata.json","container_sha256":hash});
    let source = json!({"canonical_location":"/fixture/model","resolved_revision":"files-sha256",
        "files":[{"relative_path":"fixture.bin","size_bytes":8,"sha256":hash}]});
    json!({"backend":"metal","attention_execution_policy":"portable","selected_max_model_len":64,
        "selected_kv_capacity":64,"selected_max_sequences":1,
        "kv_storage":{"source":"resolved_model_plan","requested":format,"selected":format,"numerical_profile":"fixture.profile"},
        "numerical_execution":{"requested_kv_storage":format,"selected_kv_storage":format,"selected_profile":"fixture.profile"},
        "resolution_evidence":{"schema_version":1,"requested_model":"/fixture/model",
            "original_sources":{"weights":{"kind":"local_directory","location":"/fixture/model"}},
            "resolved_sources":{"weights":source,"semantic":source,"tokenizer":source},
            "semantic_config":binding,"tokenizer":binding,"template":binding}})
}

#[test]
fn actual_dtype_backend_and_source_evidence_cannot_be_replaced_by_a_request_echo() {
    let args = args();
    for format in [
        KvStorageFormat::F16,
        KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
    ] {
        let valid = config(format);
        evidence::validate_config(&args, &valid, format).unwrap();
        for pointer in [
            "/kv_storage/source",
            "/kv_storage/selected",
            "/numerical_execution/selected_kv_storage",
            "/numerical_execution/selected_profile",
            "/resolution_evidence/resolved_sources/weights/files/0/sha256",
            "/backend",
            "/attention_execution_policy",
        ] {
            let mut bad = valid.clone();
            *bad.pointer_mut(pointer).unwrap() = Value::Null;
            assert!(
                evidence::validate_config(&args, &bad, format).is_err(),
                "accepted {pointer}"
            );
        }
        let other = if format == KvStorageFormat::F16 {
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        } else {
            KvStorageFormat::F16
        };
        assert!(evidence::validate_config(&args, &valid, other).is_err());
    }
}

fn comparison() -> Value {
    let tokens = [3, 5];
    let waves = tokens.iter().enumerate().map(|(index, token)| {
        let decision = json!({"token_index":index,"token_id":token});
        json!({"report":{"participant_count":1,"reference_schema_version":4,"candidate_schema_version":4,
            "teacher_forced_decision":decision,"comparisons":[{"nmse":0.0001,"max_abs":0.1,
                "distribution":{"vocabulary_size":8,"kl_reference_to_candidate_nats":0.01,
                    "teacher_forced":{"decision":decision,"reference_nll_nats":1.0,"candidate_nll_nats":1.02,"delta_nll_nats":0.02}}}]}})
    }).collect::<Vec<_>>();
    json!({"schema_version":1,"scope":"checkpoint_directory_diagnostic",
        "teacher_forcing":{"mode":"canonical-history","encoding":"u32-le","token_count":2,"token_ids_sha256":evidence::token_digest(&tokens)},
        "wave_count":2,"waves":waves,"aggregate":{"distribution_count":2,"teacher_forced_target_count":2,
            "mean_kl_reference_to_candidate_nats":0.01,"mean_delta_nll_nats":0.02,
            "mean_reference_nll_nats":1.0,"mean_candidate_nll_nats":1.02}})
}

#[test]
fn each_budget_is_enforced_independently_and_nonfinite_or_partial_evidence_fails() {
    let budgets = args().budgets;
    let valid = comparison();
    assert_eq!(
        scoring::evaluate(&valid, &[3, 5], budgets).unwrap()["passed"],
        true
    );
    for lower in [
        scoring::Budgets {
            mean_delta_nll_limit: 0.01,
            ..budgets
        },
        scoring::Budgets {
            max_delta_nll_limit: 0.01,
            ..budgets
        },
        scoring::Budgets {
            mean_kl_limit: 0.001,
            ..budgets
        },
        scoring::Budgets {
            max_kl_limit: 0.001,
            ..budgets
        },
    ] {
        let result = scoring::evaluate(&valid, &[3, 5], lower).unwrap();
        assert_eq!(result["passed"], false);
        assert_eq!(
            result["checks"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["passed"] == false)
                .count(),
            1
        );
    }
    for pointer in [
        "/waves/0/report/comparisons/0/distribution/kl_reference_to_candidate_nats",
        "/waves/0/report/comparisons/0/distribution/teacher_forced/delta_nll_nats",
        "/aggregate/mean_delta_nll_nats",
    ] {
        for invalid in [Value::Null, json!("NaN"), json!("inf")] {
            let mut bad = valid.clone();
            *bad.pointer_mut(pointer).unwrap() = invalid;
            assert!(scoring::evaluate(&bad, &[3, 5], budgets).is_err());
        }
    }
    let mut incomplete = valid;
    incomplete["waves"].as_array_mut().unwrap().pop();
    assert!(scoring::evaluate(&incomplete, &[3, 5], budgets).is_err());
    for limit in [f64::NAN, f64::INFINITY, -0.1] {
        assert!(scoring::Budgets {
            mean_kl_limit: limit,
            ..budgets
        }
        .validate()
        .is_err());
    }
}

#[test]
fn paired_commands_preserve_sampling_capacity_and_only_vary_storage_and_capture_paths() {
    let args = args();
    let teacher = Path::new("teacher.json");
    for dtype in ["fp16", "int8"] {
        let words = args.run_args(dtype, dtype, "plain prompt", Some(teacher));
        for pair in [
            ["--kv-dtype", dtype],
            ["--temperature", "0"],
            ["--top-k", "0"],
            ["--repeat-penalty", "1"],
            ["--max-num-seqs", "1"],
            ["--max-model-len", "64"],
            ["--vnext-checkpoint-teacher-token-file", "teacher.json"],
        ] {
            assert!(words.windows(2).any(|words| words == pair));
        }
        assert!(words
            .iter()
            .any(|word| word == "--vnext-checkpoint-product-output"));
    }
}

#[test]
#[ignore = "subprocess fixture invoked by the timeout ownership test"]
fn sleeping_child() {
    std::thread::sleep(Duration::from_secs(10));
}

#[tokio::test]
async fn child_timeout_is_bounded_and_keeps_command_and_exit_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let result = process::run(
        &std::env::current_exe().unwrap(),
        &[
            "--exact".into(),
            "tests::sleeping_child".into(),
            "--ignored".into(),
        ],
        directory.path(),
        "timeout",
        Duration::from_millis(50),
    )
    .await;
    assert!(result.is_err());
    let evidence: Value =
        serde_json::from_slice(&fs::read(directory.path().join("timeout.exit.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["timed_out"], true);
    assert!(directory.path().join("timeout.command.json").is_file());
    assert!(directory.path().join("timeout.stderr.txt").is_file());
}
