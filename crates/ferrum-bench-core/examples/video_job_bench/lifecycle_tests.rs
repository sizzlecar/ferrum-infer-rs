use super::super::tests::{mock, request};
use super::*;
use clap::Parser;
use std::time::Duration;

fn report(directory: &Path, plan: Value) -> Report {
    let arguments = Arguments::parse_from([
        "video-job-bench",
        "--base-url",
        "http://127.0.0.1:8090",
        "--request-json",
        "request.json",
        "--provenance",
        "provenance.json",
        "--server-config",
        "server.json",
        "--output",
        directory.to_str().unwrap(),
    ]);
    Report {
        schema_version: 1,
        mode: "lifecycle_observation",
        execution_state: "running",
        client_executable_sha256: String::new(),
        arguments,
        input: input::Inputs {
            provenance: Value::Null,
            provenance_sha256: String::new(),
            configurations: vec![],
            requests: vec![request()],
        },
        plan: serde_json::from_value(plan).unwrap(),
        plan_sha256: String::new(),
        evidence_contract: "test",
        actions: vec![],
        requests: vec![],
        stopped_reason: None,
    }
}

#[tokio::test]
async fn removal_after_delete_preserves_native_uncertainty_and_next_submission_identity() {
    let (client, server) = mock(vec![
        (
            "200 OK",
            r#"{"id":"cancel_me","status":"in_progress"}"#,
            Duration::ZERO,
        ),
        (
            "200 OK",
            r#"{"id":"cancel_me","status":"in_progress","cancel_requested":true}"#,
            Duration::ZERO,
        ),
        (
            "200 OK",
            r#"{"id":"replacement","status":"in_progress"}"#,
            Duration::ZERO,
        ),
        ("404 Not Found", r#"{"error":"removed"}"#, Duration::ZERO),
        (
            "200 OK",
            r#"{"id":"replacement","status":"completed"}"#,
            Duration::ZERO,
        ),
        ("200 OK", "fixture-media", Duration::ZERO),
    ])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let mut report = report(
        directory.path(),
        serde_json::json!({"actions":[
            {"action":"submit","request_index":0},
            {"action":"delete","slot":0},
            {"action":"submit","request_index":0}
        ]}),
    );
    report.plan.validate(1).unwrap();
    for index in 0..3 {
        action(&client, &mut report, index, Instant::now())
            .await
            .unwrap();
    }
    drain(&client, &mut report).await.unwrap();
    let removed = &report.requests[0];
    assert!(removed.record_removed_after_delete);
    assert!(removed.observation.remote_terminal_unknown);
    assert!(removed.observation.terminal_observed_seconds.is_none());
    assert!(removed.observation.artifact.is_none());
    let replacement = &report.requests[1].observation;
    assert_eq!(replacement.job_id.as_deref(), Some("replacement"));
    assert!(!replacement.remote_terminal_unknown);
    assert_eq!(replacement.artifact_bytes, Some(13));
    assert_eq!(
        server.await.unwrap(),
        [
            "POST /v1/videos HTTP/1.1",
            "DELETE /v1/videos/cancel_me HTTP/1.1",
            "POST /v1/videos HTTP/1.1",
            "GET /v1/videos/cancel_me HTTP/1.1",
            "GET /v1/videos/replacement HTTP/1.1",
            "GET /v1/videos/replacement/content HTTP/1.1"
        ]
    );
}

#[tokio::test]
async fn cancelled_post_future_keeps_possibly_submitted_slot_and_does_not_retry() {
    let (client, server) = mock(vec![(
        "200 OK",
        r#"{"id":"slow_receipt","status":"in_progress"}"#,
        Duration::from_millis(100),
    )])
    .await;
    let directory = tempfile::tempdir().unwrap();
    let mut report = report(
        directory.path(),
        serde_json::json!({"actions":[
            {"action":"submit","request_index":0}
        ]}),
    );
    let timed = tokio::time::timeout(
        Duration::from_millis(20),
        action(&client, &mut report, 0, Instant::now()),
    )
    .await;
    assert!(timed.is_err());
    assert_eq!(report.requests.len(), 1);
    assert!(report.requests[0].observation.remote_terminal_unknown);
    assert!(report.requests[0].observation.job_id.is_none());
    drain(&client, &mut report).await.unwrap();
    assert!(report.requests[0].observation.remote_terminal_unknown);
    assert_eq!(server.await.unwrap(), ["POST /v1/videos HTTP/1.1"]);
}

#[test]
fn gates_require_real_numeric_evidence_and_delete_an_existing_unique_slot() {
    let plan: Plan = serde_json::from_value(serde_json::json!({"actions":[
        {"action":"submit","request_index":0},
        {"action":"wait_health","conditions":[{"pointer":"/scheduler/steps","comparison":"at_least","value":2}]},
        {"action":"delete","slot":0}
    ]})).unwrap();
    plan.validate(1).unwrap();
    let Action::WaitHealth { conditions } = &plan.actions[1] else {
        panic!("gate")
    };
    assert!(!conditions[0]
        .matches(&serde_json::json!({"scheduler":{"steps":1}}))
        .unwrap());
    assert!(conditions[0]
        .matches(&serde_json::json!({"scheduler":{"steps":2}}))
        .unwrap());
    assert!(conditions[0]
        .matches(&serde_json::json!({"scheduler":{"steps":"2"}}))
        .is_err());
    assert!(conditions[0].matches(&Value::Null).is_err());
    for actions in [
        serde_json::json!([{"action":"delete","slot":0}]),
        serde_json::json!([{"action":"submit","request_index":1}]),
        serde_json::json!([{"action":"submit","request_index":0},{"action":"delete","slot":0},{"action":"delete","slot":0}]),
    ] {
        let plan: Plan = serde_json::from_value(serde_json::json!({"actions":actions})).unwrap();
        assert!(plan.validate(1).is_err());
    }
}
