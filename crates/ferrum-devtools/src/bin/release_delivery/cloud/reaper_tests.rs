use super::*;
use crate::cloud::api::tests::{http_fixture, Reply};

fn args() -> ReapArgs {
    ReapArgs {
        repository_id: 12,
        run_id: Some(34),
        completed_run_repository: Some("owner/repo".into()),
        output: "unused-fixture-output.json".into(),
    }
}
fn owner(repository: u64, run: u64, attempt: u32, expiry: u64) -> Ownership {
    Ownership {
        repository,
        run,
        attempt,
        expiry,
        nonce: "a".repeat(32),
    }
}
fn row(id: u64, label: &str) -> Value {
    json!({"id":id,"label":label,"actual_status":"running"})
}
fn run() -> Value {
    json!({"id":34,"repository":{"id":12,"full_name":"owner/repo"},
        "head_repository":{"id":12},"head_branch":"main",
        "path":".github/workflows/release-delivery.yml","event":"push",
        "run_attempt":2,"status":"completed","conclusion":"cancelled"})
}
fn reply(method: &'static str, path: &'static str, body: Value) -> Reply {
    Reply {
        method,
        path,
        status: 200,
        body,
    }
}
fn listing(rows: Vec<Value>) -> Reply {
    reply(
        "GET",
        "/api/v1/instances/",
        json!({"instances":rows,"next_token":null}),
    )
}
fn github(base: String) -> Github {
    let mut client = Github::new("owner/repo".into(), "fixture-token".into()).unwrap();
    client.base = base;
    client.http = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    client
}

#[test]
fn terminal_state_requires_actual_release_workflow_repository_and_known_conclusion() {
    for conclusion in [
        "success",
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "neutral",
        "skipped",
        "stale",
    ] {
        let mut value = run();
        value["conclusion"] = json!(conclusion);
        assert_eq!(
            completed_attempt(&value, "owner/repo", 12, 34).unwrap(),
            Some(2)
        );
    }
    for (pointer, wrong) in [
        ("/id", json!(35)),
        ("/repository/id", json!(13)),
        ("/repository/full_name", json!("other/repo")),
        ("/head_repository/id", json!(13)),
        ("/path", json!(".github/workflows/ci.yml")),
        ("/event", json!("pull_request")),
        ("/run_attempt", json!(0)),
        ("/run_attempt", json!(u64::MAX)),
        ("/status", json!("unknown")),
        ("/conclusion", Value::Null),
        ("/conclusion", json!("unknown")),
    ] {
        let mut value = run();
        *value.pointer_mut(pointer).unwrap() = wrong;
        assert!(
            completed_attempt(&value, "owner/repo", 12, 34).is_err(),
            "{pointer}: {value}"
        );
    }
    let mut value = run();
    value["event"] = json!("workflow_dispatch");
    value["head_branch"] = json!("v0.8.8-rc.1");
    assert_eq!(
        completed_attempt(&value, "owner/repo", 12, 34).unwrap(),
        Some(2)
    );
    for invalid in [
        "",
        "owner",
        "owner/repo/extra",
        "../repo",
        "owner/repo?query",
        "owner/\nrepo",
    ] {
        assert!(validate_repository(invalid).is_err());
    }
}

#[tokio::test]
async fn completed_run_reclaims_only_matching_owned_lease_before_expiry() {
    let label = owner(12, 34, 1, 999).label();
    let (base, server) = http_fixture(vec![
        listing(vec![
            row(41, &label),
            row(42, &owner(13, 34, 1, 999).label()),
            row(43, &owner(12, 35, 1, 999).label()),
            row(44, "unowned"),
        ]),
        reply(
            "GET",
            "/api/v0/instances/41/",
            json!({"instances":row(41, &label)}),
        ),
        reply("GET", "/repos/owner/repo/actions/runs/34", run()),
        reply("DELETE", "/api/v0/instances/41/", json!({"success":true})),
        listing(vec![]),
    ])
    .await;
    let results = reap_instances(
        &args(),
        &api::Client::for_test(base.clone()),
        Some(&github(base)),
        100,
    )
    .await
    .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["destroyed_and_absent"], true);
    assert_eq!(results[0]["reason"], "owning_workflow_completed");
    assert_eq!(results[0]["completed_run_attempt"], 2);
    let requests = server.await.unwrap();
    let deletes: Vec<_> = requests
        .iter()
        .filter(|request| request["method"] == "DELETE")
        .collect();
    assert_eq!(deletes.len(), 1);
    assert_eq!(deletes[0]["path"], "/api/v0/instances/41/");
}

#[tokio::test]
async fn completion_event_does_not_delete_a_currently_active_rerun() {
    for status in ["queued", "in_progress", "waiting", "requested", "pending"] {
        let label = owner(12, 34, 1, 999).label();
        let mut value = run();
        value["status"] = json!(status);
        value["conclusion"] = Value::Null;
        let (base, server) = http_fixture(vec![
            listing(vec![row(41, &label)]),
            reply(
                "GET",
                "/api/v0/instances/41/",
                json!({"instances":row(41, &label)}),
            ),
            reply("GET", "/repos/owner/repo/actions/runs/34", value),
        ])
        .await;
        let results = reap_instances(
            &args(),
            &api::Client::for_test(base.clone()),
            Some(&github(base)),
            100,
        )
        .await
        .unwrap();
        assert_eq!(results[0]["retained"], "owning_workflow_is_active");
        assert!(server
            .await
            .unwrap()
            .iter()
            .all(|request| request["method"] != "DELETE"));
    }
}

#[tokio::test]
async fn lookup_failure_wrong_repository_unknown_status_and_future_attempt_never_delete() {
    let mut wrong_repository = run();
    wrong_repository["repository"]["id"] = json!(13);
    let mut unknown_status = run();
    unknown_status["status"] = json!("new-state");
    let mut incomplete = run();
    incomplete["conclusion"] = Value::Null;
    for (status, body, attempt) in [
        (503, json!({"secret":"must not appear in errors"}), 1),
        (200, wrong_repository, 1),
        (200, unknown_status, 1),
        (200, incomplete, 1),
        (200, run(), 3),
    ] {
        let label = owner(12, 34, attempt, 999).label();
        let (base, server) = http_fixture(vec![
            listing(vec![row(41, &label)]),
            reply(
                "GET",
                "/api/v0/instances/41/",
                json!({"instances":row(41, &label)}),
            ),
            Reply {
                method: "GET",
                path: "/repos/owner/repo/actions/runs/34",
                status,
                body,
            },
        ])
        .await;
        let error = reap_instances(
            &args(),
            &api::Client::for_test(base.clone()),
            Some(&github(base)),
            100,
        )
        .await
        .unwrap_err();
        assert!(!error.contains("must not appear"));
        assert!(server
            .await
            .unwrap()
            .iter()
            .all(|request| request["method"] != "DELETE"));
    }
}

#[tokio::test]
async fn changed_provider_label_invalidates_discovery_ownership() {
    let label = owner(12, 34, 1, 999).label();
    let changed = owner(12, 34, 2, 999).label();
    let (base, server) = http_fixture(vec![
        listing(vec![row(41, &label)]),
        reply(
            "GET",
            "/api/v0/instances/41/",
            json!({"instances":row(41, &changed)}),
        ),
    ])
    .await;
    assert!(reap_instances(
        &args(),
        &api::Client::for_test(base.clone()),
        Some(&github(base)),
        100
    )
    .await
    .unwrap()
    .is_empty());
    assert!(server
        .await
        .unwrap()
        .iter()
        .all(|request| request["method"] != "DELETE"));
}

#[tokio::test]
async fn scheduled_expiry_cleanup_does_not_need_github_or_touch_live_leases() {
    let expired = owner(12, 34, 1, 100).label();
    let live = row(42, &owner(12, 35, 1, 999).label());
    let (base, server) = http_fixture(vec![
        listing(vec![row(41, &expired), live.clone()]),
        reply(
            "GET",
            "/api/v0/instances/41/",
            json!({"instances":row(41, &expired)}),
        ),
        reply("DELETE", "/api/v0/instances/41/", json!({"success":true})),
        listing(vec![live]),
    ])
    .await;
    let mut input = args();
    input.run_id = None;
    input.completed_run_repository = None;
    let results = reap_instances(&input, &api::Client::for_test(base), None, 100)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["reason"], "lease_expired");
    assert_eq!(results[0]["destroyed_and_absent"], true);
    assert!(server
        .await
        .unwrap()
        .iter()
        .all(|request| !request["path"].as_str().unwrap().contains("/repos/")));
}
