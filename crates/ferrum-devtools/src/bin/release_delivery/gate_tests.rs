use super::*;
use ferrum_bench_core::release_regression::model_basic::{
    ARITHMETIC_PROMPT, MEMORY_PROMPT, RECALL_PROMPT,
};
use ferrum_bench_core::release_regression::{
    model_tasks::ModelRunCapacity, Entrypoint, ExecutionTarget, Impact, ModelProfile, PlanCost,
    SelectedProfile,
};
use serde_json::json;

fn task() -> ExpectedModelRun {
    ExpectedModelRun {
        profile: ModelProfile {
            gguf: None,
            reasoning_protocol: ferrum_types::ModelReasoningProtocol::PromptOpened,
            id: "quick-start-cuda".into(),
            model: "fixture/model".into(),
            available: true,
            estimate: None,
            target: ExecutionTarget {
                architecture: "dense".into(),
                protocol: ferrum_types::ModelOutputProtocol::Text,
                precision: "bf16".into(),
                backend: Backend::Cuda,
                execution_path: "production-plan".into(),
            },
        },
        binary_sha256: "b".repeat(64),
        version: "2.3.4".into(),
        checks: vec![ModelCheck::Basic],
        disable_thinking: true,
        use_default_backend: true,
        max_tokens: 128,
        runtime_capacity: None,
        reasoning_alias_replay: false,
        stop_prompt: "Write a sentence.".into(),
    }
}
fn make_plan(task: &ExpectedModelRun) -> Plan {
    Plan {
        stage: Stage::Release,
        impact: Impact {
            areas: vec![],
            paths: vec![],
            unknown_paths: vec![],
            product_contract_changed: false,
        },
        obligations: vec![Obligation {
            behavior: Behavior::QuickStart,
            layer: EvidenceLayer::ModelRuntime,
            entrypoints: vec![
                Entrypoint::Run,
                Entrypoint::ServeSync,
                Entrypoint::ServeStream,
            ],
            scope: ObligationScope::Profile {
                profile_id: task.profile.id.clone(),
                target: task.profile.target.clone(),
            },
            reason: "declared Quick Start".into(),
            checkers: vec!["model-regression.basic.quick-start".into()],
        }],
        selected: vec![SelectedProfile {
            profile: task.profile.clone(),
            obligations: vec![0],
            reasons: vec![],
        }],
        omitted: vec![],
        gaps: vec![],
        cost: PlanCost::default(),
    }
}
fn plan_document(plan: &Plan) -> Value {
    json!({"schema_version":2,"stage":"release","provenance":{"candidate":"a".repeat(40)},"plan":plan,"model_tasks":model_task_schedule(plan)})
}
fn distributions() -> Distributions {
    BTreeMap::from([(
        (Backend::Cuda, "x86_64-unknown-linux-gnu".into()),
        Distribution {
            backend: Backend::Cuda,
            name: "fixture.tar.gz".into(),
            sha256: "c".repeat(64),
            binary_sha256: "b".repeat(64),
            target: "x86_64-unknown-linux-gnu".into(),
            assets: vec![],
        },
    )])
}
// Runner-result fixture: the release gate rechecks shared semantic assertions;
// the model runner separately checks actual protocol framing.
fn model_report(task: &ExpectedModelRun) -> Value {
    let observation = |answer| {
        json!({"message":{"role":"assistant","content":answer,"reasoning":null},
        "finish_reason":"stop","usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}})
    };
    let answers: Vec<_> = ["OK", "42", "cobalt-731"]
        .into_iter()
        .map(|answer| {
            let output = observation(answer);
            json!({"content":output["message"]["content"],"reasoning":null,
            "finish_reason":output["finish_reason"],"usage":output["usage"]})
        })
        .collect();
    let observations = json!({"memory_write":observation("OK"),"sync":observation("42"),"stream":observation("42"),
        "recall":observation("cobalt-731"),"stream_recall":observation("cobalt-731")});
    let memory = json!({"role":"user","content":MEMORY_PROMPT});
    let history = vec![
        memory.clone(),
        observations["memory_write"]["message"].clone(),
        json!({"role":"user","content":ARITHMETIC_PROMPT}),
    ];
    let mut sync_recall = history.clone();
    sync_recall.extend([
        observations["sync"]["message"].clone(),
        json!({"role":"user","content":RECALL_PROMPT}),
    ]);
    let mut stream_recall = history.clone();
    stream_recall.extend([
        observations["stream"]["message"].clone(),
        json!({"role":"user","content":RECALL_PROMPT}),
    ]);
    let requests = json!({"memory_write":[memory],"sync":history,"stream":history,"recall":sync_recall,"stream_recall":stream_recall});
    json!({"schema_version":2,"status":"passed","profile_id":task.profile.id,"target":task.profile.target,"binary_sha256":task.binary_sha256,
    "options":{"profile_id":task.profile.id,"model":task.profile.model,"backend":"cuda","checks":task.checks,"disable_thinking":true,"use_default_backend":true,"max_tokens":task.max_tokens,"context_tokens":task.runtime_capacity.as_ref().map(|capacity|capacity.context_tokens),"max_num_seqs":task.runtime_capacity.as_ref().map(|capacity|capacity.max_num_seqs),"reasoning_alias_replay":false,"stop_prompt":task.stop_prompt},
    "sampling":{"temperature":0,"seed":7,"max_tokens":task.max_tokens},"environment_policy":"remove_inherited_ferrum_overrides",
    "cases":[
        {"case":"binary-version","status":"passed","evidence":{"version":format!("ferrum {}",task.version)}},
        {"case":"run-basic","status":"passed","evidence":{"ready":{"event":"ready","requested_model":task.profile.model,"backend":"CUDA(0)"},"answers":answers,"prompts":[MEMORY_PROMPT,ARITHMETIC_PROMPT,RECALL_PROMPT]}},
        {"case":"serve-startup","status":"passed","evidence":{"version":task.version,"status":"healthy","auto_config":{"hardware_capabilities":{"backend":"cuda"}}}},
        {"case":"serve-basic","status":"passed","evidence":{"observations":observations,"requests":requests}},
        {"case":"binary-unchanged","status":"passed","evidence":{"sha256":task.binary_sha256}}
    ]})
}
fn extract_only() -> installation::InstallationReport {
    installation::InstallationReport {
        schema_version: 1,
        status: "not_run".into(),
        version: "2.3.4".into(),
        candidate_sha: "a".repeat(40),
        asset_name: "fixture.tar.gz".into(),
        asset_sha256: "c".repeat(64),
        binary_sha256: "b".repeat(64),
        backend: "cuda".into(),
        target_triple: "x86_64-unknown-linux-gnu".into(),
        binary: PathBuf::from("/other-host/ferrum"),
        observations: vec![],
        error: None,
    }
}

#[test]
fn release_plan_recomputes_schedule_and_rejects_removed_gap_or_stale_assignments() {
    let task = task();
    let plan = make_plan(&task);
    let good = plan_document(&plan);
    release_plan(&good).unwrap();
    let mut stale = good.clone();
    stale["model_tasks"]["runs"] = json!([]);
    assert!(release_plan(&stale).unwrap_err().contains("schedule"));
    let mut gap = good.clone();
    gap["plan"]["gaps"] = json!([{"kind":"empty_inventory"}]);
    assert!(release_plan(&gap).unwrap_err().contains("gaps"));
    let mut schema = good;
    schema["schema_version"] = json!(1);
    assert!(release_plan(&schema).is_err());
    let schedule = model_task_schedule(&plan);
    let mut prepared = PreparedTasks {
        schema_version: 1,
        expectations: vec![task],
        unsupported_obligations: vec![],
        remaining_plan_gaps: vec![],
    };
    validate_tasks(&plan, &schedule, &prepared, &distributions(), "2.3.4").unwrap();
    prepared.remaining_plan_gaps.push(Gap::EmptyInventory);
    assert!(validate_tasks(&plan, &schedule, &prepared, &distributions(), "2.3.4").is_err());
    prepared.remaining_plan_gaps.clear();
    prepared.expectations[0].binary_sha256 = "d".repeat(64);
    assert!(
        validate_tasks(&plan, &schedule, &prepared, &distributions(), "2.3.4")
            .unwrap_err()
            .contains("staged bytes")
    );
    prepared.expectations[0].binary_sha256 = "b".repeat(64);
    prepared.expectations[0].use_default_backend = false;
    assert!(validate_tasks(&plan, &schedule, &prepared, &distributions(), "2.3.4").is_err());
}

#[test]
fn prepared_capacity_cannot_override_quick_start_or_the_functional_workload() {
    for quick_start in [true, false] {
        let mut expected = task();
        expected.disable_thinking = quick_start;
        expected.use_default_backend = quick_start;
        expected.runtime_capacity = (!quick_start).then_some(DEFAULT_FUNCTIONAL_CAPACITY);
        let mut plan = make_plan(&expected);
        if !quick_start {
            plan.obligations[0].behavior = Behavior::ModelForward;
            plan.obligations[0].checkers = vec!["model-regression.basic.model-forward".into()];
        }
        let schedule = model_task_schedule(&plan);
        assert_eq!(schedule.runs[0].quick_start, quick_start);
        let mut prepared = PreparedTasks {
            schema_version: 1,
            expectations: vec![expected.clone()],
            unsupported_obligations: vec![],
            remaining_plan_gaps: vec![],
        };
        validate_tasks(&plan, &schedule, &prepared, &distributions(), "2.3.4").unwrap();
        for capacity in [
            None,
            Some(DEFAULT_FUNCTIONAL_CAPACITY),
            Some(ModelRunCapacity {
                context_tokens: DEFAULT_FUNCTIONAL_CAPACITY.context_tokens * 2,
                ..DEFAULT_FUNCTIONAL_CAPACITY
            }),
            Some(ModelRunCapacity {
                max_num_seqs: DEFAULT_FUNCTIONAL_CAPACITY.max_num_seqs + 1,
                ..DEFAULT_FUNCTIONAL_CAPACITY
            }),
        ] {
            if capacity == expected.runtime_capacity {
                continue;
            }
            prepared.expectations[0].runtime_capacity = capacity;
            assert!(
                validate_tasks(&plan, &schedule, &prepared, &distributions(), "2.3.4")
                    .unwrap_err()
                    .contains("capacity")
            );
        }
    }
}

#[test]
fn release_gate_rejects_wrong_basic_answers_despite_passed_runner_summaries() {
    let expected = task();
    VerifiedModels::new(std::slice::from_ref(&expected), &[model_report(&expected)]).unwrap();
    for (name, pointer, wrong) in [
        ("run-basic", "/evidence/answers/1/content", json!("43")),
        (
            "serve-basic",
            "/evidence/observations/memory_write",
            Value::Null,
        ),
        (
            "serve-basic",
            "/evidence/observations/stream_recall/message/content",
            json!("other-code"),
        ),
    ] {
        let mut report = model_report(&expected);
        let case = report["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case"] == name)
            .unwrap();
        *case.pointer_mut(pointer).unwrap() = wrong;
        assert!(
            VerifiedModels::new(std::slice::from_ref(&expected), &[report]).is_err(),
            "accepted {name} {pointer}"
        );
    }
}

#[test]
fn cuda_extract_only_needs_validated_same_binary_run_and_serve_reports() {
    let task = task();
    let tasks = vec![task.clone()];
    let reports = vec![model_report(&task)];
    let verified = VerifiedModels::new(&tasks, &reports).unwrap();
    let report = extract_only();
    verify_installations(
        &distributions(),
        std::slice::from_ref(&report),
        &verified,
        "2.3.4",
        &"a".repeat(40),
    )
    .unwrap();
    assert_eq!(report.status, "not_run");
    assert!(
        verify_installations(&distributions(), &[], &verified, "2.3.4", &"a".repeat(40)).is_err()
    );
    let mut stale = report.clone();
    stale.binary_sha256 = "e".repeat(64);
    assert!(verify_installations(
        &distributions(),
        &[stale],
        &verified,
        "2.3.4",
        &"a".repeat(40)
    )
    .is_err());
    let mut failed = report;
    failed.status = "failed".into();
    assert!(verify_installations(
        &distributions(),
        &[failed],
        &verified,
        "2.3.4",
        &"a".repeat(40)
    )
    .is_err());
    for case_name in ["run-basic", "serve-startup"] {
        let mut missing = model_report(&task);
        missing["cases"]
            .as_array_mut()
            .unwrap()
            .retain(|case| case["case"] != case_name);
        assert!(VerifiedModels::new(&tasks, &[missing]).is_err());
        let mut failed = model_report(&task);
        failed["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case"] == case_name)
            .unwrap()["status"] = json!("failed");
        assert!(VerifiedModels::new(&tasks, &[failed]).is_err());
    }
    assert!(VerifiedModels::new(
        &tasks,
        &[json!({"status":"passed","profile_id":task.profile.id})]
    )
    .is_err());
}

#[test]
fn green_registered_checks_cannot_satisfy_unknown_numerics_or_performance_obligations() {
    let mut plan = make_plan(&task());
    verify_obligations(&plan).unwrap();
    for layer in [EvidenceLayer::BackendNumerics, EvidenceLayer::Performance] {
        plan.obligations[0].layer = layer;
        plan.obligations[0].behavior = Behavior::KernelNumerics;
        plan.obligations[0].checkers = vec!["rms-norm-only".into()];
        assert!(verify_obligations(&plan).is_err());
    }
    plan = make_plan(&task());
    plan.obligations[0].checkers = vec!["unknown-checker".into()];
    assert!(verify_obligations(&plan).is_err());
    let group = contract_groups().remove(0);
    let contract = ContractReport {
        schema_version: 1,
        status: ferrum_bench_core::release_regression::contracts::ContractStatus::Passed,
        groups: vec![],
    };
    assert!(verify_contract_report(&contract_groups(), &contract).is_err());
    plan.obligations[0] = Obligation {
        behavior: group.behavior,
        layer: EvidenceLayer::Contract,
        entrypoints: group.entrypoints,
        scope: ObligationScope::Global,
        reason: String::new(),
        checkers: vec![group.id],
    };
    verify_obligations(&plan).unwrap();
    plan.obligations[0].behavior = Behavior::KernelNumerics;
    assert!(verify_obligations(&plan).is_err());
}

fn ci_run() -> Value {
    json!({"id":17,"head_sha":"a".repeat(40),"repository":{"full_name":"owner/repo"},"event":"workflow_dispatch","path":".github/workflows/release-delivery.yml","run_attempt":2,"status":"in_progress"})
}
fn ci_job() -> Value {
    json!({"id":21,"run_id":17,"run_attempt":2,"head_sha":"a".repeat(40),"name":"Quality / CI required","status":"completed","conclusion":"success"})
}
async fn ci_fixture(
    run: Value,
    jobs: Vec<Value>,
    status: u16,
    expect_jobs: bool,
) -> Result<(), String> {
    ci_fixture_with_windows(run, jobs, status, expect_jobs, None).await
}
async fn ci_fixture_with_windows(
    run: Value,
    jobs: Vec<Value>,
    status: u16,
    expect_jobs: bool,
    windows_attempt: Option<u64>,
) -> Result<(), String> {
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        net::TcpListener,
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let mut responses = vec![("/repos/owner/repo/actions/runs/17", run, status)];
    if expect_jobs {
        responses.push((
            "/repos/owner/repo/actions/runs/17/jobs?filter=all&per_page=100&page=1",
            json!({"total_count":jobs.len(),"jobs":jobs}),
            200,
        ));
    }
    // Read complete HTTP header lines asynchronously. Blocking a current-thread
    // Tokio test with a join/read timeout can prevent the client from progressing.
    let worker = tokio::spawn(async move {
        for (path, body, status) in responses {
            let (stream, _) = tokio::time::timeout(Duration::from_secs(3), listener.accept())
                .await
                .expect("missing expected CI request")
                .unwrap();
            let mut stream = BufReader::new(stream);
            let mut request_line = String::new();
            stream.read_line(&mut request_line).await.unwrap();
            assert_eq!(request_line, format!("GET {path} HTTP/1.1\r\n"));
            loop {
                let mut header = String::new();
                let count = stream.read_line(&mut header).await.unwrap();
                assert!(count > 0, "truncated CI request headers");
                if header == "\r\n" {
                    break;
                }
            }
            let body = body.to_string();
            let headers=format!("HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len());
            stream
                .get_mut()
                .write_all(headers.as_bytes())
                .await
                .unwrap();
            stream.get_mut().write_all(body.as_bytes()).await.unwrap();
            stream.get_mut().shutdown().await.unwrap();
        }
    });
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let result = match windows_attempt {
        Some(attempt) => {
            ci_at_with_windows(
                &client,
                &url,
                "owner/repo",
                17,
                &"a".repeat(40),
                None,
                Some(attempt),
            )
            .await
        }
        None => ci_at(&client, &url, "owner/repo", 17, &"a".repeat(40), None).await,
    };
    tokio::time::timeout(Duration::from_secs(3), worker)
        .await
        .expect("CI fixture did not finish expected requests")
        .unwrap();
    result
}

#[tokio::test]
async fn windows_assets_require_the_latest_successful_staging_attempt() {
    let mut windows = ci_job();
    windows["id"] = json!(23);
    windows["name"] = json!("stage-cuda / Stage Windows x86_64 CUDA sm89");
    ci_fixture_with_windows(
        ci_run(),
        vec![ci_job(), windows.clone()],
        200,
        true,
        Some(2),
    )
    .await
    .unwrap();
    // A publication-only retry may reuse this exact, still-latest staging job.
    let mut retry = ci_run();
    retry["run_attempt"] = json!(3);
    ci_fixture_with_windows(retry, vec![ci_job(), windows.clone()], 200, true, Some(2))
        .await
        .unwrap();
    for changed in [json!("failure"), json!("cancelled")] {
        let mut failure = windows.clone();
        failure["conclusion"] = changed;
        assert!(
            ci_fixture_with_windows(ci_run(), vec![ci_job(), failure], 200, true, Some(2))
                .await
                .is_err()
        );
    }
    assert!(
        ci_fixture_with_windows(ci_run(), vec![ci_job(), windows], 200, true, Some(1))
            .await
            .is_err()
    );
    assert!(
        ci_fixture_with_windows(ci_run(), vec![ci_job()], 200, true, Some(2))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn required_ci_job_must_match_workflow_candidate_and_latest_execution() {
    ci_fixture(ci_run(), vec![ci_job()], 200, true)
        .await
        .unwrap();
    let mut push = ci_run();
    push["event"] = json!("push");
    ci_fixture(push, vec![ci_job()], 200, true).await.unwrap();
    for field in ["conclusion", "status", "name", "head_sha"] {
        let mut job = ci_job();
        job[field] = json!(match field {
            "conclusion" => "failure",
            "status" => "in_progress",
            "name" => "Some other green job",
            _ => "another candidate",
        });
        assert!(ci_fixture(ci_run(), vec![job], 200, true).await.is_err());
    }
    for field in ["event", "path", "head_sha"] {
        let mut run = ci_run();
        run[field] = json!(match field {
            "event" => "pull_request",
            "path" => ".github/workflows/ci.yml",
            _ => "another candidate",
        });
        assert!(ci_fixture(run, vec![], 200, false).await.is_err());
    }
    assert!(ci_fixture(ci_run(), vec![], 503, false)
        .await
        .unwrap_err()
        .contains("status unknown"));
}

#[tokio::test]
async fn publication_retry_reuses_unrerun_quality_but_never_hides_a_newer_failure() {
    let mut previous = ci_job();
    previous["id"] = json!(11);
    previous["run_attempt"] = json!(1);
    let publishing = json!({"id":22,"run_id":17,"run_attempt":2,"head_sha":"a".repeat(40),"name":"Publish channels","status":"in_progress","conclusion":null});
    ci_fixture(ci_run(), vec![previous.clone(), publishing], 200, true)
        .await
        .unwrap();
    for (status, conclusion) in [
        ("completed", json!("failure")),
        ("queued", Value::Null),
        ("in_progress", Value::Null),
    ] {
        let mut current = ci_job();
        current["status"] = json!(status);
        current["conclusion"] = conclusion;
        assert!(
            ci_fixture(ci_run(), vec![current, previous.clone()], 200, true)
                .await
                .unwrap_err()
                .contains("latest Quality"),
            "older successful execution cannot replace the most recent Quality job"
        );
    }
    let mut old_failure = previous.clone();
    old_failure["conclusion"] = json!("failure");
    ci_fixture(ci_run(), vec![old_failure, ci_job()], 200, true)
        .await
        .unwrap();
    let mut future = ci_job();
    future["run_attempt"] = json!(3);
    assert!(ci_fixture(ci_run(), vec![previous, future], 200, true)
        .await
        .is_err());
}

#[tokio::test]
async fn distribution_rehashes_embedded_binary_and_reaudits_raw_dependencies() {
    let directory = tempfile::tempdir().unwrap();
    let payload = directory.path().join("payload");
    fs::create_dir(&payload).unwrap();
    fs::write(payload.join("ferrum"), b"tiny binary fixture").unwrap();
    let name = "fixture.tar.gz";
    let archive = directory.path().join(name);
    let output = std::process::Command::new("tar")
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(&payload)
        .arg("ferrum")
        .output()
        .unwrap();
    assert!(output.status.success());
    let audit = "ferrum: ELF 64-bit executable\n\tlibc.so.6 => /lib/libc.so.6 (0x00001234)\n";
    let input = CandidateInput {
        version: "2.3.4".into(),
        release_candidate_sha: "a".repeat(40),
        release_candidate_tag: "v2.3.4-rc.1".into(),
        staging_label: "fixture".into(),
        workflow_run_id: "17".into(),
        workflow_run_attempt: "2".into(),
    };
    let abi_input = AbiInput {
        backend: staging::Backend::Cpu,
        target_triple: "x86_64-unknown-linux-gnu".into(),
        cargo_features: vec![],
        cuda_compute_capability: None,
        cuda_toolkit_image: None,
    };
    let generated = staging::generate_manifests(
        &input,
        &abi_input,
        name,
        &fs::read(&archive).unwrap(),
        b"tiny binary fixture",
        "native.dependencies.txt",
        audit,
    )
    .unwrap();
    let abi_path = directory.path().join(format!("{name}.abi.json"));
    for (suffix, value) in [
        (".abi.json", &generated.abi),
        (".version.json", &generated.version),
        (".dependency.json", &generated.dependency),
    ] {
        fs::write(
            directory.path().join(format!("{name}{suffix}")),
            serde_json::to_vec(value).unwrap(),
        )
        .unwrap();
    }
    fs::write(
        directory.path().join(format!("{name}.sha256")),
        generated.asset_checksum,
    )
    .unwrap();
    fs::write(
        directory.path().join(format!("{name}.binary.sha256")),
        generated.binary_checksum,
    )
    .unwrap();
    fs::write(directory.path().join("native.dependencies.txt"), audit).unwrap();
    let accepted = distribution(&abi_path, "2.3.4", &"a".repeat(40))
        .await
        .unwrap();
    assert!(accepted
        .assets
        .iter()
        .any(|asset| asset.name == "native.dependencies.txt"));
    assert!(!accepted.assets.iter().any(|asset| asset.name == "payload"));
    let mut altered = generated.abi.clone();
    altered["binary_sha256"] = json!("d".repeat(64));
    fs::write(&abi_path, serde_json::to_vec(&altered).unwrap()).unwrap();
    assert!(distribution(&abi_path, "2.3.4", &"a".repeat(40))
        .await
        .unwrap_err()
        .contains("actual archive/binary"));
    fs::write(&abi_path, serde_json::to_vec(&generated.abi).unwrap()).unwrap();
    fs::write(
        directory.path().join("native.dependencies.txt"),
        "ferrum: ELF 64-bit executable\n\tlibpython3.so => /lib/libpython3.so (0x00001234)\n",
    )
    .unwrap();
    assert!(
        distribution(&abi_path, "2.3.4", &"a".repeat(40))
            .await
            .is_err(),
        "a false forbidden-runtime summary must not bypass raw dependency audit"
    );
}

#[test]
fn plan_version_must_advance_its_formal_base_even_when_resuming_the_candidate() {
    let mut document = plan_document(&make_plan(&task()));
    document["provenance"]["release_base_tag"] = json!("v2.3.3");
    verify_plan_version(&document, "2.3.4").unwrap();
    // A published candidate's own tag is not its preceding release base.
    verify_plan_version(&document, "2.3.4").unwrap();
    for base in [
        Value::Null,
        json!("2.3.3"),
        json!("v2.3.4"),
        json!("v2.4.0"),
        json!("v2.3.3-rc.1"),
        json!("native-cuda-v2.3.3"),
        json!("v2.3.3\n"),
    ] {
        document["provenance"]["release_base_tag"] = base;
        assert!(verify_plan_version(&document, "2.3.4").is_err());
    }
}

async fn published_fixture(pages: Vec<(u16, Value)>, target: &str) -> Result<(), String> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let worker = tokio::spawn(async move {
        for (index, (status, value)) in pages.into_iter().enumerate() {
            let (socket, _) = tokio::time::timeout(Duration::from_secs(3), listener.accept())
                .await
                .expect("missing release inventory request")
                .unwrap();
            let mut socket = BufReader::new(socket);
            let mut line = String::new();
            socket.read_line(&mut line).await.unwrap();
            assert_eq!(
                line,
                format!(
                    "GET /repos/owner/repo/releases?per_page=100&page={} HTTP/1.1\r\n",
                    index + 1
                )
            );
            loop {
                let mut header = String::new();
                assert!(socket.read_line(&mut header).await.unwrap() > 0);
                if header == "\r\n" {
                    break;
                }
            }
            let body = value.to_string();
            let response = format!("HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            socket
                .get_mut()
                .write_all(response.as_bytes())
                .await
                .unwrap();
            socket.get_mut().shutdown().await.unwrap();
        }
    });
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let result = public_version_at(&client, &url, "owner/repo", target, None).await;
    tokio::time::timeout(Duration::from_secs(3), worker)
        .await
        .unwrap()
        .unwrap();
    result
}

fn published(tag: &str) -> Value {
    json!({"tag_name": tag, "draft": false, "prerelease": false, "published_at": "2026-09-01T00:00:00Z"})
}

#[tokio::test]
async fn published_inventory_rejects_historical_downgrades_but_allows_identical_version_resume() {
    let mut draft = published("v99.0.0");
    draft["draft"] = json!(true);
    let mut prerelease = published("v100.0.0");
    prerelease["prerelease"] = json!(true);
    let rows = json!([
        published("v2.3.9"),
        published("v2.3.10"),
        published("native-cuda-v101"),
        published("v102.0.0-rc.1"),
        draft,
        prerelease
    ]);
    for target in ["2.3.10", "2.3.11", "3.0.0"] {
        published_fixture(vec![(200, rows.clone())], target)
            .await
            .unwrap();
    }
    assert!(published_fixture(vec![(200, rows)], "2.3.9")
        .await
        .unwrap_err()
        .contains("refusing downgrade"));
    // GitHub pages are ordered by release creation, not semantic version. A
    // higher formal release on a later page must not be missed behind native assets.
    let first_page = json!(vec![published("native-operator-assets"); 100]);
    assert!(published_fixture(
        vec![(200, first_page), (200, json!([published("v3.0.0")]))],
        "2.3.10"
    )
    .await
    .unwrap_err()
    .contains("refusing downgrade"));
    published_fixture(vec![(200, json!([]))], "2.3.10")
        .await
        .unwrap();
}

#[tokio::test]
async fn unknown_published_inventory_is_never_treated_as_no_previous_release() {
    for (status, body) in [
        (503, json!({"error": "unavailable"})),
        (200, json!({"message": "not an inventory"})),
        (200, json!([{"tag_name": "v1.0.0"}])),
        (
            200,
            json!([{"tag_name": "v1.0.0", "draft": false, "prerelease": false}]),
        ),
    ] {
        assert!(published_fixture(vec![(status, body)], "2.3.10")
            .await
            .is_err());
    }
}
