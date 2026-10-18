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
    assert!(startup < response && response < shutdown);
    assert_eq!(
        shutdown + 1,
        records.len(),
        "no producer writes beyond final shutdown"
    );
    assert_eq!(records[shutdown]["status"], "ok");
}
