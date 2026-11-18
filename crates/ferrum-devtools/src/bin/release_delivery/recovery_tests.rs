use super::*;

fn step(name: &str, conclusion: &str) -> Value {
    json!({"name":name,"status":"completed","conclusion":conclusion})
}
fn job(conclusion: &str, steps: Vec<Value>) -> Value {
    json!({"status":"completed","conclusion":conclusion,"steps":steps,"run_id":25,"run_attempt":2,"id":53})
}

#[test]
fn recovery_only_accepts_failed_download_and_unexecuted_model_lanes() {
    let download = step("Run actions/download-artifact@v8", "failure");
    let skipped = job("skipped", vec![]);
    let cpu = BTreeMap::from([(2, skipped.clone())]);
    let original = job(
        "failure",
        vec![
            download.clone(),
            step(CUDA_MODELS, "skipped"),
            step(CUDA_NUMERICS, "skipped"),
            step("Run actions/upload-artifact@v4", "failure"),
        ],
    );
    let mut cuda = BTreeMap::from([(1, skipped), (2, original.clone())]);
    let receipt = verify_transport(&cuda, &cpu).unwrap();
    assert_eq!(receipt["original_cuda_job"]["conclusion"], "failure");
    for failed in [CUDA_MODELS, CUDA_NUMERICS, "Verify prepared release tools"] {
        let mut observed = original.clone();
        observed["steps"]
            .as_array_mut()
            .unwrap()
            .push(step(failed, "failure"));
        cuda.insert(2, observed.clone());
        assert!(verify_transport(&cuda, &cpu).is_err());
        cuda.insert(1, observed);
        cuda.insert(2, original.clone());
        assert!(
            verify_transport(&cuda, &cpu).is_err(),
            "older execution failure for {failed}"
        );
        cuda.remove(&1);
    }
    assert!(verify_transport(&cuda, &BTreeMap::from([(2, job("failure", vec![]))])).is_err());
}

#[test]
fn standalone_windows_binds_the_candidate_tag_original_repo_and_workflow() {
    let source = json!({"id":25,"run_number":12,"head_sha":"a".repeat(40),"repository":{"id":9}});
    let run = json!({"id":31,"repository":{"id":9,"full_name":"s/f"},"head_repository":{"id":9},
       "head_sha":"a".repeat(40),"head_branch":"v1.3.0-rc.12","event":"workflow_dispatch","path":".github/workflows/release-windows.yml","status":"completed","conclusion":"success"});
    let verify = |r: &Value| verify_standalone_run(r, &source, 31, "v1.3.0-rc.12", "1.3.0", "s/f");
    verify(&run).unwrap();
    for (key, value) in [
        ("event", json!("push")),
        ("path", json!(".github/workflows/ci.yml")),
        ("head_repository", json!({"id":3})),
        ("head_sha", json!("b".repeat(40))),
        ("head_branch", json!("main")),
        ("head_branch", json!("v1.3.0-rc.11")),
        ("id", json!(25)),
        ("conclusion", json!("failure")),
    ] {
        let mut bad = run.clone();
        bad[key] = value;
        assert!(verify(&bad).is_err(), "{key}");
    }
}

#[test]
fn numerical_recovery_requires_original_checker_and_executed_oracle() {
    let dir = tempfile::tempdir().unwrap();
    let checker = dir.path().join("sm89-fp8-runtime-check");
    let log = dir.path().join("runtime.log");
    fs::write(&checker, b"checker").unwrap();
    let sha = installation::sha256(&checker).unwrap();
    fs::write(
        checker.with_file_name("sm89-fp8-runtime-check.sha256"),
        format!("{sha}  sm89-fp8-runtime-check\n"),
    )
    .unwrap();
    fs::write(&log, format!("test {SM89_TEST} ... ok\n")).unwrap();
    let mut fp8 = Fp8Execution {
        test_name: SM89_TEST.into(),
        checker_sha256: sha,
        log_sha256: installation::sha256(&log).unwrap(),
        exit_code: 0,
    };
    verify_fp8(&fp8, &checker, &log).unwrap();
    fp8.exit_code = 1;
    assert!(verify_fp8(&fp8, &checker, &log).is_err());
    fp8.exit_code = 0;
    fs::write(
        &log,
        "running 0 tests\ntest result: ok. 0 passed; 0 failed\n",
    )
    .unwrap();
    fp8.log_sha256 = installation::sha256(&log).unwrap();
    assert!(verify_fp8(&fp8, &checker, &log).is_err());
    fs::write(&log, format!("test {SM89_TEST} ... ok\n")).unwrap();
    assert!(verify_fp8(&fp8, &checker, &log).is_err());
    fp8.log_sha256 = installation::sha256(&log).unwrap();
    fs::write(&checker, b"another").unwrap();
    assert!(verify_fp8(&fp8, &checker, &log).is_err());
}

#[test]
fn manual_runner_rejects_partial_reports_or_failed_process() {
    let profiles = BTreeSet::from(["a".to_owned(), "b".to_owned()]);
    let execution = json!({"schema_version":1,"binary_sha256":"bin","backend":"cuda","status":"passed","error":null,
        "profiles":["a","b"],"completed_profiles":["a","b"],"runner_sha256":"f".repeat(64)});
    let verify = |e: &Value| verify_execution(e, "cuda", "bin", &profiles);
    verify(&execution).unwrap();
    for (key, value) in [
        ("completed_profiles", json!(["a"])),
        ("binary_sha256", json!("other")),
        ("backend", json!("cpu")),
        ("error", json!("runner exited 1")),
        ("status", json!("failed")),
    ] {
        let mut bad = execution.clone();
        bad[key] = value;
        assert!(verify(&bad).is_err(), "{key}");
    }
}
