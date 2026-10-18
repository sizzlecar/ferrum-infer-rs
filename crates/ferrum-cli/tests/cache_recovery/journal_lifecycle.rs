//! Real CPU inference and product shutdown keep every role on its shared journal.
use super::*;

fn with_journal(mut flow: Flow, path: &Path) -> Flow {
    flow.arguments.extend([
        "--profile-detail".to_owned(),
        "verify".to_owned(),
        "--profile-jsonl".to_owned(),
        path.display().to_string(),
        "--memory-profile-jsonl".to_owned(),
        path.display().to_string(),
        "--scheduler-trace-jsonl".to_owned(),
        path.display().to_string(),
    ]);
    flow
}

fn journal(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("complete JSONL record after process exit"))
        .collect()
}

fn with_off_profile(mut flow: Flow, path: &Path) -> Flow {
    flow.arguments.extend([
        "--profile-detail".to_owned(),
        "off".to_owned(),
        "--profile-jsonl".to_owned(),
        path.display().to_string(),
    ]);
    flow
}

fn standard_off_records(path: &Path) -> Vec<Value> {
    let records = journal(path);
    assert!(!records.is_empty());
    for record in &records {
        let event: ferrum_types::FerrumProfileEvent =
            serde_json::from_value(record.clone()).unwrap();
        event.validate().unwrap();
        assert!(!matches!(
            event.phase.as_str(),
            "vnext.execution_facts"
                | "vnext.device_execution_span"
                | "vnext.device_physical_submission"
        ));
    }
    records
}

#[tokio::test]
async fn run_off_profile_keeps_standard_actual_output() {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start([(REPO.to_owned(), model::files())].into()).await;
    let path = cache.path().join("off-run.jsonl");
    with_off_profile(Flow::run(REPO), &path)
        .successful(&hub, cache.path())
        .await;
    let records = standard_off_records(&path);
    let generation = &records[unique_phase(&records, "actual_run_generation")];
    assert_eq!(generation["attributes"]["output_tokens"], 1);
    assert_eq!(generation["attributes"]["profile_detail"], "off");
    assert_eq!(generation["status"], "ok");
}

#[cfg(unix)]
#[tokio::test]
async fn serve_off_profile_keeps_standard_metadata_http_and_final_close() {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start([(REPO.to_owned(), model::files())].into()).await;
    let path = cache.path().join("off-serve.jsonl");
    let mut flow = with_off_profile(Flow::serve(REPO).await, &path);
    flow.arguments.extend(
        [
            "--profile-commit-sha",
            "declared-commit",
            "--profile-env-hash",
            "sha256:declared",
            "--profile-model",
            "declared-label",
            "--profile-concurrency",
            "3",
            "--profile-runtime-flags-json",
            "{\"declared\":true}",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    flow.successful_with_shutdown(&hub, cache.path(), Shutdown::Terminate)
        .await;
    let records = standard_off_records(&path);
    let metadata = &records[unique_phase(&records, "product_session_metadata")];
    assert_eq!(
        metadata["attributes"]["declared_metadata"],
        json!({
            "commit_sha":"declared-commit", "env_hash":"sha256:declared", "model":"declared-label",
            "concurrency":3, "runtime_flags":{"declared":true}
        })
    );
    assert_ne!(metadata["model"], "declared-label");
    let startup = unique_phase(&records, "actual_serve_startup");
    let output = unique_phase(&records, "actual_serve_first_request_done");
    let shutdown = unique_phase(&records, "actual_serve_shutdown");
    assert!(startup < output && output < shutdown);
    assert_eq!(shutdown + 1, records.len());
    assert!(records.iter().any(|event| matches!(
        event["phase"].as_str(),
        Some("chat_completions_sync_complete" | "chat_completions_stream_complete")
    )));
}

fn unique_phase(records: &[Value], phase: &str) -> usize {
    let positions: Vec<_> = records
        .iter()
        .enumerate()
        .filter_map(|(index, event)| (event["phase"] == phase).then_some(index))
        .collect();
    assert_eq!(
        positions.len(),
        1,
        "missing or duplicated {phase}: {records:?}"
    );
    positions[0]
}

#[tokio::test]
async fn run_journal_keeps_actual_output_after_engine_shutdown() {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start([(REPO.to_owned(), model::files())].into()).await;
    let path = cache.path().join("run.jsonl");
    with_journal(Flow::run(REPO), &path)
        .successful(&hub, cache.path())
        .await;
    let records = journal(&path);
    let generation = &records[unique_phase(&records, "actual_run_generation")];
    assert_eq!(generation["attributes"]["output_tokens"], 1);
    assert_eq!(generation["status"], "ok");
    assert_eq!(
        generation["attributes"]["engine_request_start_boundary"],
        "engine_api"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn serve_journal_keeps_startup_http_and_graceful_shutdown_records() {
    let cache = tempfile::tempdir().unwrap();
    let hub = Hub::start([(REPO.to_owned(), model::files())].into()).await;
    let path = cache.path().join("serve.jsonl");
    with_journal(Flow::serve(REPO).await, &path)
        .successful_with_shutdown(&hub, cache.path(), Shutdown::Terminate)
        .await;
    let records = journal(&path);
    let startup = unique_phase(&records, "actual_serve_startup");
    let response = unique_phase(&records, "actual_serve_first_request_done");
    let shutdown = unique_phase(&records, "actual_serve_shutdown");
    let generation = records
        .iter()
        .find(|event| {
            matches!(
                event["phase"].as_str(),
                Some("chat_completions_sync_complete" | "chat_completions_stream_complete")
            )
        })
        .expect("HTTP generation profile");
    assert_eq!(
        generation["attributes"]["engine_request_start_boundary"],
        "http_handler"
    );
    assert!(startup < response && response < shutdown);
    assert_eq!(
        shutdown + 1,
        records.len(),
        "no producer writes beyond final shutdown"
    );
    assert_eq!(records[shutdown]["status"], "ok");
}
