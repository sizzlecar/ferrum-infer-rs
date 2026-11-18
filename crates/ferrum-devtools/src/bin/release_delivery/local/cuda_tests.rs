use super::*;
use ferrum_bench_core::release_regression::{
    model_basic::{ARITHMETIC_PROMPT, MEMORY_PROMPT, RECALL_PROMPT},
    model_tasks::DEFAULT_CUDA_FUNCTIONAL_CAPACITY,
};

fn cuda_task(id: &str) -> ExpectedModelRun {
    let mut expected = task(Backend::Cuda, id);
    expected.checks = vec![ModelCheck::Basic];
    expected.reasoning_alias_replay = false;
    expected.runtime_capacity = Some(DEFAULT_CUDA_FUNCTIONAL_CAPACITY);
    expected
}

fn document(expectations: Vec<ExpectedModelRun>) -> PreparedTasks {
    PreparedTasks {
        schema_version: 1,
        expectations,
        unsupported_obligations: vec![],
        remaining_plan_gaps: vec![],
    }
}

#[test]
fn cuda_selection_preserves_serial_order_and_rejects_unavailable_or_invalid_capacity() {
    let first = cuda_task("first");
    let second = cuda_task("second");
    let (selected, other, _) = select(
        document(vec![
            first.clone(),
            task(Backend::Cpu, "cpu"),
            second.clone(),
            task(Backend::Metal, "metal"),
        ]),
        LocalBackend::Cuda,
    )
    .unwrap();
    assert_eq!(selected, vec![first.clone(), second]);
    assert_eq!(other, vec!["cpu", "metal"]);
    let mut unavailable = first.clone();
    unavailable.profile.available = false;
    assert!(select(document(vec![unavailable]), LocalBackend::Cuda)
        .unwrap_err()
        .contains("unavailable"));
    let mut invalid = first;
    invalid
        .runtime_capacity
        .as_mut()
        .unwrap()
        .runtime_memory_budget_bytes = Some(0);
    assert!(select(document(vec![invalid]), LocalBackend::Cuda)
        .unwrap_err()
        .contains("memory budget"));
}

#[test]
fn cuda_runner_arguments_bind_backend_and_public_capacity_budget() {
    let directory = tempfile::tempdir().unwrap();
    let mut input = args(directory.path());
    input.backend = LocalBackend::Cuda;
    let mut expected = cuda_task("cuda");
    for automatic in [false, true] {
        expected.use_default_backend = automatic;
        let words = runner_arguments(&input, &expected, Path::new("task"), Path::new("report"));
        for (flag, value) in [
            ("--backend", "cuda"),
            ("--context-tokens", "2048"),
            ("--max-num-seqs", "1"),
            ("--runtime-memory-budget-bytes", "4294967296"),
        ] {
            assert!(words
                .windows(2)
                .any(|pair| pair[0] == flag && pair[1] == value));
        }
        assert_eq!(
            words.iter().any(|word| word == "--use-default-backend"),
            automatic
        );
    }
    expected
        .runtime_capacity
        .as_mut()
        .unwrap()
        .runtime_memory_budget_bytes = None;
    let words = runner_arguments(&input, &expected, Path::new("task"), Path::new("report"));
    assert!(!words
        .iter()
        .any(|word| word == "--runtime-memory-budget-bytes"));
}

// Semantic report fixture, not evidence of a GPU execution. These observations
// exercise the same verifier used on untouched reports from the real runner.
fn report(expected: &ExpectedModelRun) -> Value {
    let observation = |answer| {
        json!({
            "message":{"role":"assistant","content":answer,"reasoning":null},
            "finish_reason":"stop","usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}
        })
    };
    let answers: Vec<_> = ["OK", "42", "cobalt-731"].into_iter().map(|answer| {
        let output = observation(answer);
        json!({"content":answer,"reasoning":null,"finish_reason":output["finish_reason"],"usage":output["usage"]})
    }).collect();
    let observations = json!({"memory_write":observation("OK"),"sync":observation("42"),"stream":observation("42"),
        "recall":observation("cobalt-731"),"stream_recall":observation("cobalt-731")});
    let memory = json!({"role":"user","content":MEMORY_PROMPT});
    let history = vec![
        memory.clone(),
        observations["memory_write"]["message"].clone(),
        json!({"role":"user","content":ARITHMETIC_PROMPT}),
    ];
    let recall = |mode: &str| {
        let mut request = history.clone();
        request.extend([
            observations[mode]["message"].clone(),
            json!({"role":"user","content":RECALL_PROMPT}),
        ]);
        request
    };
    let capacity = expected.runtime_capacity.unwrap();
    json!({
        "schema_version":2,"status":"passed","profile_id":expected.profile.id,
        "target":expected.profile.target,"binary_sha256":expected.binary_sha256,
        "options":{"profile_id":expected.profile.id,"model":expected.profile.model,"backend":"cuda",
            "checks":expected.checks,"disable_thinking":expected.disable_thinking,"use_default_backend":expected.use_default_backend,
            "max_tokens":expected.max_tokens,"context_tokens":capacity.context_tokens,"max_num_seqs":capacity.max_num_seqs,
            "runtime_memory_budget_bytes":capacity.runtime_memory_budget_bytes,
            "reasoning_alias_replay":expected.reasoning_alias_replay,"stop_prompt":expected.stop_prompt},
        "sampling":{"temperature":0,"seed":7,"max_tokens":expected.max_tokens},
        "environment_policy":if expected.use_default_backend {"remove_inherited_ferrum_overrides"} else {"inherit"},
        "cases":[
            {"case":"binary-version","status":"passed","evidence":{"version":format!("ferrum {}",expected.version)}},
            {"case":"run-basic","status":"passed","evidence":{"ready":{"event":"ready","requested_model":expected.profile.model,"backend":"CUDA(0)"},"answers":answers,"prompts":[MEMORY_PROMPT,ARITHMETIC_PROMPT,RECALL_PROMPT]}},
            {"case":"serve-startup","status":"passed","evidence":{"version":expected.version,"status":"healthy","auto_config":{"hardware_capabilities":{"backend":"cuda"},
                "selected_max_model_len":capacity.context_tokens,"selected_kv_capacity":capacity.context_tokens,"selected_max_sequences":capacity.max_num_seqs}}},
            {"case":"serve-basic","status":"passed","evidence":{"observations":observations,"requests":{"memory_write":[memory],"sync":history,"stream":history,"recall":recall("sync"),"stream_recall":recall("stream")}}},
            {"case":"binary-unchanged","status":"passed","evidence":{"sha256":expected.binary_sha256}}
        ]
    })
}

#[test]
fn public_sm120_binds_fresh_automatic_installation_and_both_entrypoints_to_public_bytes() {
    use clap::Parser;
    use sha2::{Digest, Sha256};
    let directory = tempfile::tempdir().unwrap();
    let dir = directory.path();
    let mut expected = cuda_task("public");
    expected.use_default_backend = true;
    let hash = &expected.binary_sha256;
    let version = &expected.version;
    let source = "1".repeat(40);
    let archive_hash = "b".repeat(64);
    let archive = "ferrum-linux-x86_64-cuda-sm120.tar.gz";
    let binary = format!(
        "/home/operator/.local/share/ferrum/installer/releases/{version}-cuda-{hash}/ferrum"
    );
    let script = b"#!/bin/sh\n";
    let script_hash = format!("{:x}", Sha256::digest(script));
    fs::write(dir.join("install.sh"), script).unwrap();
    fs::write(dir.join("binary.sha256"), format!("{hash}  ferrum\n")).unwrap();
    let public = json!({"id":15,"target_commitish":source,"tag_name":format!("v{version}"),"draft":false,"prerelease":false,
        "published_at":"2026-01-01T12:00:00Z","assets":[{"id":19,"name":archive,"size":2048,"state":"uploaded","digest":format!("sha256:{archive_hash}")}]});
    let evidence_asset = json!({"created_at":"2026-01-01T12:06:00Z"});
    write_json(&dir.join("release.json"), &public).unwrap();
    write_json(&dir.join("evidence-asset.json"), &evidence_asset).unwrap();
    let mut model = report(&expected);
    model["options"]["ferrum_bin"] = json!(binary);
    model["started_at"] = json!("2026-01-01T12:04:00Z");
    model["finished_at"] = json!("2026-01-01T12:05:00Z");
    model["cases"][1]["evidence"]["ready"]["reasoning_protocol"] = json!("none");
    model["cases"][2]["evidence"]["reasoning_protocol"] = json!("none");
    model["cases"][2]["evidence"]["auto_config"]["hardware_capabilities"]["compute_capability"] =
        json!("12.0");
    let probes: Vec<_> = [vec!["--version"], vec!["--help"], vec!["run","--help"], vec!["serve","--help"]].into_iter()
        .map(|args| json!({"arguments":args,"exit_code":0,"stdout":if args == ["--version"] {format!("ferrum {version}")} else {"usage".into()},"stderr":""})).collect();
    let url = "https://ferrum.pandaailabs.com/install.sh";
    let proof = json!({"schema_version":1,"release_id":15,"source_sha":source,"version":version,"archive_asset_id":19,
        "install_home_before":{"home":"/home/operator","installer_root_exists":false,"launcher_exists":false},
        "archive_transfer":{"requested_url":format!("https://ferrum.pandaailabs.com/downloads/v{version}/{archive}"),"download_bytes":2048,"exit_code":0,"http_status":200},
        "install_started_at":"2026-01-01T12:01:00Z","install_finished_at":"2026-01-01T12:03:00Z","installer_args":[],
        "installer_exit_code":0,"installer_output":"transfer complete","managed_archive_checksum":format!("{archive_hash}  {archive}\n"),
        "entry":{"schema_version":1,"status":"passed","requested_url":url,"effective_url":url,"expected_sha256":script_hash,
            "actual_sha256":script_hash,"http_status":200,"response_bytes":script.len(),"error":null},
        "installed":{"schema_version":1,"status":"passed","channel":"bootstrap","version":version,"requested_binary":"/home/operator/.local/bin/ferrum",
            "binary":binary,"binary_sha256":hash,"binary_sha256_after":hash,"observations":probes,"error":null},"model":model});
    let verify = || {
        let crate::Action::VerifyPublicSm120(args) = crate::Args::parse_from([
            "release_delivery",
            "verify-public-sm120",
            "--directory",
            dir.to_str().unwrap(),
            "--version",
            version,
            "--source-sha",
            &source,
        ])
        .action
        else {
            unreachable!()
        };
        crate::public_sm120::verify(args)
    };
    let check = |proof: &Value| {
        write_json(&dir.join("evidence.json"), proof).unwrap();
        let mut asset: Value =
            serde_json::from_slice(&fs::read(dir.join("evidence-asset.json")).unwrap()).unwrap();
        let bytes = fs::read(dir.join("evidence.json")).unwrap();
        asset["digest"] = json!(format!("sha256:{:x}", Sha256::digest(&bytes)));
        write_json(&dir.join("evidence-asset.json"), &asset).unwrap();
        verify()
    };
    check(&proof).unwrap();
    let mut pinned = proof.clone();
    pinned["installer_args"] = json!([
        "--version",
        version,
        "--backend",
        "auto",
        "--no-modify-path"
    ]);
    check(&pinned).unwrap();
    let mut fallback = proof.clone();
    fallback["archive_transfer"]["requested_url"] = json!(format!(
        "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v{version}/{archive}"
    ));
    check(&fallback).unwrap();
    write_json(&dir.join("evidence.json"), &proof).unwrap();
    assert!(verify().unwrap_err().contains("digest"));
    for (field, wrong) in [
        ("/release_id", json!(99)),
        ("/install_home_before/home", json!("/somewhere-else")),
        ("/install_home_before/installer_root_exists", json!(true)),
        ("/install_home_before/launcher_exists", json!(true)),
        ("/archive_transfer", Value::Null),
        (
            "/archive_transfer/requested_url",
            json!(format!(
                "https://ferrum.pandaailabs.com/downloads/v{version}/{archive}.sha256"
            )),
        ),
        ("/archive_transfer/download_bytes", json!(100)),
        ("/archive_transfer/exit_code", json!(28)),
        ("/archive_transfer/http_status", json!(206)),
        ("/source_sha", json!("2".repeat(40))),
        ("/archive_asset_id", json!(88)),
        (
            "/managed_archive_checksum",
            json!(format!("{}  {archive}", "a".repeat(64))),
        ),
        ("/installer_args", json!(["--backend", "cuda"])),
        (
            "/installer_args",
            json!(["--release-base-url", "https://local.invalid"]),
        ),
        ("/install_started_at", json!("2026-01-01T11:55:00Z")),
        ("/entry/actual_sha256", json!("0".repeat(64))),
        (
            "/installed/requested_binary",
            json!("/tmp/candidate/ferrum"),
        ),
        ("/installed/observations/2/exit_code", json!(1)),
        ("/model/started_at", json!("2026-01-01T11:56:00Z")),
        ("/model/binary_sha256", json!("0".repeat(64))),
        ("/model/options/ferrum_bin", json!("/tmp/candidate/ferrum")),
        ("/model/cases/1/evidence/ready/backend", json!("CPU")),
        (
            "/model/cases/2/evidence/auto_config/hardware_capabilities/compute_capability",
            json!("8.9"),
        ),
        (
            "/model/cases/3/evidence/observations/recall/message/content",
            json!("incorrect"),
        ),
    ] {
        let mut invalid = proof.clone();
        *invalid.pointer_mut(field).unwrap() = wrong;
        assert!(check(&invalid).is_err(), "accepted invalid {field}");
    }
    write_json(
        &dir.join("evidence-asset.json"),
        &json!({"created_at":"2026-01-01T11:59:00Z"}),
    )
    .unwrap();
    assert!(check(&proof).is_err());
}

#[test]
fn cuda_report_requires_observed_cuda_in_both_entrypoints_and_preserves_raw_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("report.json");
    for automatic in [false, true] {
        let mut expected = cuda_task("cuda");
        expected.use_default_backend = automatic;
        let valid = report(&expected);
        write_json(&path, &valid).unwrap();
        assert_eq!(checked_report(&path, &expected).unwrap(), valid);
        for (pointer, replacement) in [
            ("/cases/1/evidence/ready/backend", json!("CPU")),
            ("/cases/1/evidence/ready/backend", Value::Null),
            (
                "/cases/2/evidence/auto_config/hardware_capabilities/backend",
                json!("cpu"),
            ),
            (
                "/cases/2/evidence/auto_config/hardware_capabilities/backend",
                Value::Null,
            ),
        ] {
            let mut fallback = valid.clone();
            *fallback.pointer_mut(pointer).unwrap() = replacement;
            write_json(&path, &fallback).unwrap();
            let original = fs::read(&path).unwrap();
            assert!(checked_report(&path, &expected)
                .unwrap_err()
                .contains("backend"));
            assert_eq!(fs::read(&path).unwrap(), original);
        }
    }
}

#[test]
fn cuda_report_cannot_replace_bound_inputs_or_semantics_with_pass_labels() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("report.json");
    let expected = cuda_task("cuda");
    for (pointer, replacement) in [
        ("/options/runtime_memory_budget_bytes", Value::Null),
        (
            "/options/runtime_memory_budget_bytes",
            json!(8_u64 * 1024 * 1024 * 1024),
        ),
        ("/binary_sha256", json!("c".repeat(64))),
        ("/cases/4/evidence/sha256", json!("c".repeat(64))),
        ("/cases/1/evidence/answers/1/content", json!("43")),
        (
            "/cases/3/evidence/observations/sync/message/content",
            json!("43"),
        ),
        ("/cases/1/status", json!("failed")),
    ] {
        let mut invalid = report(&expected);
        *invalid.pointer_mut(pointer).unwrap() = replacement;
        write_json(&path, &invalid).unwrap();
        let original = fs::read(&path).unwrap();
        assert!(checked_report(&path, &expected).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
    }
}

#[tokio::test]
async fn cuda_changed_runner_or_binary_fails_before_launch_or_report_creation() {
    let directory = tempfile::tempdir().unwrap();
    for runner_changed in [true, false] {
        let mut input = inputs(directory.path(), vec![cuda_task("cuda")]);
        input.backend = LocalBackend::Cuda;
        if runner_changed {
            input.runner_sha256 = "c".repeat(64);
        }
        let error = execute(input).await.unwrap_err();
        assert!(error.contains(if runner_changed {
            "runner SHA-256"
        } else {
            "Ferrum SHA-256"
        }));
        assert!(!directory.path().join("reports").exists());
    }
}
