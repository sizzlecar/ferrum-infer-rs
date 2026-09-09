use super::*;
use ferrum_bench_core::release_regression::{
    performance::{FileDigest, Limits, SourceIdentity, Workload},
    submission::{submission_scope, SUBMISSION_CHECK_ID},
    Backend, Entrypoint, ExecutionTarget, ModelProfile,
};
use serde_json::json;
use std::io::Write;

fn origin() -> Origin {
    Origin {
        schema_version: 1,
        run_id: 17,
        run_attempt: 1,
        head_sha: "a".repeat(40),
        job_id: 21,
    }
}
fn job() -> Job {
    serde_json::from_value(json!({
        "id":21,"run_id":17,"run_attempt":1,"head_sha":"a".repeat(40),
        "name":PERFORMANCE_JOB,"status":"completed","conclusion":"success",
        "started_at":"2026-09-06T00:00:00Z","completed_at":"2026-09-06T00:00:30Z",
        "steps":[
            {"name":PERFORMANCE_PREPARE,"number":1,"status":"completed","conclusion":"success","started_at":"2026-09-06T00:00:01Z","completed_at":"2026-09-06T00:00:03Z"},
            {"name":PERFORMANCE_REGISTER,"number":2,"status":"completed","conclusion":"success","started_at":"2026-09-06T00:00:04Z","completed_at":"2026-09-06T00:00:05Z"},
            {"name":PERFORMANCE_EXECUTE,"number":3,"status":"completed","conclusion":"success","started_at":"2026-09-06T00:00:07Z","completed_at":"2026-09-06T00:00:20Z"},
            {"name":PERFORMANCE_UPLOAD,"number":4,"status":"completed","conclusion":"success","started_at":"2026-09-06T00:00:21Z","completed_at":"2026-09-06T00:00:23Z"}
        ]
    })).unwrap()
}
fn run() -> Value {
    json!({"id":17,"run_attempt":2,"head_sha":"a".repeat(40),
        "repository":{"id":10,"full_name":"owner/repo"},"head_repository":{"id":10},
        "path":".github/workflows/release-delivery.yml@refs/heads/main","event":"workflow_dispatch"})
}
fn artifact_record() -> Value {
    json!({"id":91,"name":"release-performance-tasks","expired":false,"size_in_bytes":123,
        "workflow_run":{"id":17,"head_sha":"a".repeat(40),"repository_id":10,"head_repository_id":10},
        "created_at":"2026-09-06T00:00:04.125Z","updated_at":"2026-09-06T00:00:05.123Z"})
}
fn validate_producer(original: &[Job], all: &[Job]) -> Result<Job, String> {
    producer(
        &origin(),
        original,
        all,
        17,
        &"a".repeat(40),
        2,
        PERFORMANCE_JOB,
    )
}

#[test]
fn failed_jobs_retry_can_carry_forward_success_with_new_id_and_attempt() {
    let original = job();
    let mut carried = original.clone();
    carried.id = 92;
    carried.run_attempt = 2;
    let verified = validate_producer(
        std::slice::from_ref(&original),
        &[original.clone(), carried],
    )
    .unwrap();
    assert_eq!(verified.id, origin().job_id);
    preregistered(&artifact_record(), &verified).unwrap();
}

#[test]
fn new_execution_or_failed_latest_producer_invalidates_old_evidence() {
    for mutation in ["failed", "pending", "new-success", "step-time", "job-time"] {
        let original = job();
        let mut later = original.clone();
        later.id = 92;
        later.run_attempt = 2;
        match mutation {
            "failed" => later.conclusion = Some("failure".into()),
            "pending" => {
                later.status = "in_progress".into();
                later.completed_at = None;
            }
            "new-success" => {
                later.started_at = Some("2026-09-06T01:00:00Z".into());
                later.completed_at = Some("2026-09-06T01:00:30Z".into());
            }
            "step-time" => later.steps[2].started_at = Some("2026-09-06T00:00:08Z".into()),
            "job-time" => later.completed_at = Some("2026-09-06T00:00:31Z".into()),
            _ => unreachable!(),
        }
        assert!(
            validate_producer(std::slice::from_ref(&original), &[original.clone(), later]).is_err(),
            "accepted {mutation}"
        );
    }
}

#[test]
fn origin_must_resolve_the_original_job_and_candidate_without_ambiguity() {
    let original = job();
    assert!(validate_producer(&[], std::slice::from_ref(&original)).is_err());
    assert!(validate_producer(
        &[original.clone(), original.clone()],
        std::slice::from_ref(&original)
    )
    .is_err());
    for mutation in ["id", "attempt", "sha", "failed"] {
        let mut changed = original.clone();
        match mutation {
            "id" => changed.id += 1,
            "attempt" => changed.run_attempt = 2,
            "sha" => changed.head_sha = "b".repeat(40),
            "failed" => changed.conclusion = Some("failure".into()),
            _ => unreachable!(),
        }
        assert!(validate_producer(&[changed], std::slice::from_ref(&original)).is_err());
    }
    assert!(validate_producer(
        std::slice::from_ref(&original),
        &[original.clone(), original.clone()]
    )
    .is_err());
}

#[test]
fn artifact_requires_same_repository_candidate_run_and_nonexpired_bounded_bytes() {
    artifact_provenance(&artifact_record(), &run(), 17, &"a".repeat(40)).unwrap();
    for (pointer, value) in [
        ("/workflow_run/id", json!(18)),
        ("/workflow_run/head_sha", json!("b".repeat(40))),
        ("/workflow_run/repository_id", json!(11)),
        ("/workflow_run/head_repository_id", json!(11)),
        ("/expired", json!(true)),
        ("/size_in_bytes", json!(ZIP_LIMIT + 1)),
    ] {
        let mut changed = artifact_record();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(
            artifact_provenance(&changed, &run(), 17, &"a".repeat(40)).is_err(),
            "accepted {pointer}"
        );
    }
    for (pointer, value) in [
        ("/event", json!("pull_request")),
        ("/path", json!(".github/workflows/ci.yml")),
        ("/head_sha", json!("b".repeat(40))),
        ("/repository/full_name", json!("other/repo")),
    ] {
        let mut changed = run();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(verify_run(&changed, "owner/repo", 17, &"a".repeat(40)).is_err());
    }
}

#[test]
fn registration_must_finish_before_measurement_and_within_the_upload_step() {
    preregistered(&artifact_record(), &job()).unwrap();
    for (created, updated) in [
        ("2026-09-06T00:00:03Z", "2026-09-06T00:00:04Z"),
        ("2026-09-06T00:00:04Z", "2026-09-06T00:00:07Z"),
        ("2026-09-06T00:00:05Z", "2026-09-06T00:00:04Z"),
    ] {
        let mut record = artifact_record();
        record["created_at"] = json!(created);
        record["updated_at"] = json!(updated);
        assert!(preregistered(&record, &job()).is_err());
    }
    let mut same_second = job();
    same_second.steps[2].started_at = Some("2026-09-06T00:00:05Z".into());
    preregistered(&artifact_record(), &same_second).unwrap();
    for mutation in ["missing", "skipped", "failed", "duplicate", "order"] {
        let mut changed = job();
        match mutation {
            "missing" => {
                changed.steps.remove(2);
            }
            "skipped" => changed.steps[2].conclusion = Some("skipped".into()),
            "failed" => changed.steps[2].conclusion = Some("failure".into()),
            "duplicate" => changed.steps.push(changed.steps[2].clone()),
            "order" => changed.steps[2].number = 1,
            _ => unreachable!(),
        }
        assert!(
            preregistered(&artifact_record(), &changed).is_err(),
            "accepted {mutation}"
        );
    }
}

fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes) in entries {
        writer
            .start_file(
                *name,
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}
#[test]
fn zip_digest_and_extraction_reject_changed_bytes_traversal_and_expansion() {
    let bytes = archive(&[
        ("ci-origin.json", b"{}"),
        ("profile/expected-task.json", b"{\"schema_version\":1}"),
    ]);
    let metadata = json!({"digest":format!("sha256:{:x}", Sha256::digest(&bytes))});
    digest_matches(&bytes, &metadata).unwrap();
    let extracted = unpack(&bytes).unwrap();
    assert_eq!(
        fs::read(extracted.path().join("profile/expected-task.json")).unwrap(),
        b"{\"schema_version\":1}"
    );
    assert!(digest_matches(b"different archive", &metadata).is_err());
    assert!(digest_matches(&bytes, &json!({})).is_err());
    for name in [
        "../outside",
        "/absolute",
        "C:/absolute",
        "a/../../outside",
        "a\\..\\outside",
    ] {
        let bytes = archive(&[(name, b"no")]);
        let mut reader = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        // Exercise actual hostile ZIP names, not writer-sanitized fixtures.
        assert_eq!(reader.by_index(0).unwrap().name_raw(), name.as_bytes());
        assert!(unpack(&bytes).is_err(), "accepted {name}");
    }
    let oversized = vec![0; FILE_LIMIT as usize + 1];
    assert!(unpack(&archive(&[("oversized.json", &oversized)])).is_err());
    assert!(unpack(b"invalid zip").is_err());
}

fn expected() -> ExpectedPerformanceRun {
    ExpectedPerformanceRun {
        schema_version: 1,
        profile: ModelProfile {
            gguf: None,
            id: "declared-profile".into(),
            model: "fixture:model".into(),
            available: true,
            estimate: None,
            reasoning_protocol: ferrum_types::ModelReasoningProtocol::None,
            target: ExecutionTarget {
                architecture: "dense".into(),
                protocol: ferrum_types::ModelOutputProtocol::Text,
                precision: "gguf-q4_k_m".into(),
                backend: Backend::Metal,
                execution_path: "legacy-model-executor".into(),
            },
        },
        baseline_version: "1.2.3".into(),
        candidate_version: "1.2.4".into(),
        client_version: "1.2.4".into(),
        baseline_sha256: "a".repeat(64),
        candidate_sha256: "b".repeat(64),
        client_sha256: "b".repeat(64),
        policy: PerformancePolicy {
            workload: Workload {
                input_tokens: 32,
                output_tokens: 16,
                measured_requests: 3,
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
        },
        source: SourceIdentity {
            gguf: FileDigest {
                bytes: 1024,
                sha256: "c".repeat(64),
            },
            sidecars: [
                (
                    "tokenizer.json".into(),
                    FileDigest {
                        bytes: 32,
                        sha256: "d".repeat(64),
                    },
                ),
                (
                    "tokenizer_config.json".into(),
                    FileDigest {
                        bytes: 32,
                        sha256: "e".repeat(64),
                    },
                ),
            ]
            .into(),
        },
        obligations: vec![2],
    }
}
#[test]
fn baseline_abi_must_be_official_release_bytes_and_name_the_expected_binary() {
    let expected = expected();
    let abi = serde_json::to_vec(&json!({"asset_name":BASELINE_ARCHIVE,"backend":"metal",
        "target_triple":"aarch64-apple-darwin","binary_name":"ferrum",
        "binary_sha256":expected.baseline_sha256,"asset_sha256":"f".repeat(64)}))
    .unwrap();
    let release = json!({"tag_name":"v1.2.3","draft":false,"prerelease":false,"assets":[
        {"name":format!("{BASELINE_ARCHIVE}.abi.json"),"state":"uploaded","digest":format!("sha256:{:x}",Sha256::digest(&abi))},
        {"name":BASELINE_ARCHIVE,"state":"uploaded","digest":format!("sha256:{}","f".repeat(64))}
    ]});
    baseline_identity(&release, "v1.2.3", &abi, &expected).unwrap();
    for (pointer, value) in [
        ("/tag_name", json!("v1.2.2")),
        ("/prerelease", json!(true)),
        ("/draft", json!(true)),
        (
            "/assets/0/digest",
            json!(format!("sha256:{}", "0".repeat(64))),
        ),
        (
            "/assets/1/digest",
            json!(format!("sha256:{}", "0".repeat(64))),
        ),
    ] {
        let mut changed = release.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(
            baseline_identity(&changed, "v1.2.3", &abi, &expected).is_err(),
            "accepted {pointer}"
        );
    }
    let mut altered = expected.clone();
    altered.baseline_sha256 = "0".repeat(64);
    assert!(baseline_identity(&release, "v1.2.3", &abi, &altered).is_err());
    let mut changed_abi = abi.clone();
    changed_abi.push(b' ');
    assert!(baseline_identity(&release, "v1.2.3", &changed_abi, &expected).is_err());
}

#[test]
fn task_inventory_is_exact_and_policy_comes_from_candidate_configuration() {
    let expected = expected();
    let bytes =
        serde_json::to_vec(&json!({"schema_version":1,"policy":expected.policy,"sources":[]}))
            .unwrap();
    assert_eq!(candidate_policy(&bytes).unwrap(), expected.policy);
    assert!(candidate_policy(b"{\"schema_version\":2,\"policy\":{}}").is_err());
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("baseline")).unwrap();
    fs::create_dir(directory.path().join("candidate")).unwrap();
    fs::write(directory.path().join("ci-origin.json"), b"{}").unwrap();
    let ids = BTreeSet::from(["declared-profile"]);
    assert!(check_profile_directories(directory.path(), &ids, true).is_err());
    fs::create_dir(directory.path().join("declared-profile")).unwrap();
    check_profile_directories(directory.path(), &ids, true).unwrap();
    fs::create_dir(directory.path().join("unregistered")).unwrap();
    assert!(check_profile_directories(directory.path(), &ids, true).is_err());
}

#[test]
fn absent_submission_or_performance_never_satisfies_known_or_unknown_obligations() {
    let evidence = CiEvidence::default();
    let mut obligation = Obligation {
        behavior: Behavior::SubmissionCompletion,
        layer: EvidenceLayer::BackendNumerics,
        entrypoints: vec![],
        scope: submission_scope(),
        reason: "submission".into(),
        checkers: vec![SUBMISSION_CHECK_ID.into()],
    };
    assert!(!evidence.covers(0, &obligation));
    obligation.behavior = Behavior::KernelNumerics;
    assert!(!evidence.covers(0, &obligation));
    obligation.layer = EvidenceLayer::Performance;
    obligation.behavior = Behavior::Performance;
    obligation.entrypoints = vec![Entrypoint::ServeStream];
    assert!(!evidence.covers(0, &obligation));
    // A bound result cannot be moved to a different index or altered checker.
    let mut bound = CiEvidence::default();
    bound.performance.insert(0, obligation.clone());
    assert!(bound.covers(0, &obligation));
    assert!(!bound.covers(1, &obligation));
    obligation.checkers.push("unexecuted-checker".into());
    assert!(!bound.covers(0, &obligation));
}
