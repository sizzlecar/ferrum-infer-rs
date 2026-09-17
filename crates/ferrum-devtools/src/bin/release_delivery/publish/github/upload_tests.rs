use super::super::super::test_http::{Response, Server};
use super::super::tests::{accepted, api, release, remote};
use super::*;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug)]
enum Failure {
    Http(u16),
    LostResponse,
    UnreadableSuccess,
    TruncatedSuccess,
}

#[derive(Clone, Copy, Debug, Default)]
enum Stored {
    #[default]
    Absent,
    Starter,
    Uploaded,
    WrongDigest,
    WrongSize,
    UnknownState,
    StarterWithDigest,
    WrongStarterSize,
}

#[derive(Clone, Copy, Debug, Default)]
enum Fault {
    #[default]
    None,
    PublicRelease,
    ReleaseId,
    Notes,
    Candidate,
    TagMissing,
    TagMoved,
    ListError(u16),
    DuplicateName,
    DuplicateId,
    ByIdUploaded,
    ByIdChanged,
    ByIdConflictingUpload,
    DeleteStatus(u16),
    DeleteResponseLost,
    DeletedStillPresent,
    ReadAfterDelete,
}

struct Scenario {
    failures: usize,
    failure: Failure,
    stored: Stored,
    fault: Fault,
    preexisting_starter: bool,
    complete_on_attempt: Option<usize>,
}
impl Default for Scenario {
    fn default() -> Self {
        Self {
            failures: 1,
            failure: Failure::Http(503),
            stored: Stored::Starter,
            fault: Fault::None,
            preexisting_starter: false,
            complete_on_attempt: None,
        }
    }
}

#[derive(Default)]
struct State {
    asset: Option<Value>,
    posts: usize,
    deletes: usize,
    deleted: bool,
    mutations: Vec<String>,
}

fn uploaded(id: u64) -> Value {
    let mut asset = remote("one.tar.gz");
    asset["id"] = json!(id);
    asset
}

fn stored_asset(kind: Stored, id: u64) -> Option<Value> {
    if matches!(kind, Stored::Absent) {
        return None;
    }
    let mut asset = uploaded(id);
    match kind {
        Stored::Starter | Stored::StarterWithDigest | Stored::WrongStarterSize => {
            asset["state"] = json!("starter");
            if !matches!(kind, Stored::StarterWithDigest) {
                asset["digest"] = Value::Null;
            }
            if matches!(kind, Stored::WrongStarterSize) {
                asset["size"] = json!(123);
            }
        }
        Stored::WrongDigest => asset["digest"] = json!(format!("sha256:{}", "0".repeat(64))),
        Stored::WrongSize => asset["size"] = json!(123),
        Stored::UnknownState => asset["state"] = json!("unknown"),
        Stored::Uploaded | Stored::Absent => {}
    }
    Some(asset)
}

fn server(scenario: Scenario) -> (Server, Arc<Mutex<State>>) {
    let state = Arc::new(Mutex::new(State {
        asset: scenario
            .preexisting_starter
            .then(|| stored_asset(Stored::Starter, 50).unwrap()),
        ..State::default()
    }));
    let view = state.clone();
    let server = Server::new(move |request| {
        let mut state = view.lock().unwrap();
        let recovering = state.posts > 0;
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/repos/test/repo/releases/7") => {
                let mut value = release(true);
                if recovering {
                    match scenario.fault {
                        Fault::PublicRelease => value = release(false),
                        Fault::ReleaseId => value["id"] = json!(8),
                        Fault::Notes => value["body"] = json!("changed"),
                        Fault::Candidate => value["target_commitish"] = json!("b".repeat(40)),
                        _ => {}
                    }
                }
                Response::json(200, value)
            }
            ("GET", "/repos/test/repo/git/ref/tags/v2.3.4") => {
                if recovering && matches!(scenario.fault, Fault::TagMissing) {
                    return Response::text(404, "missing");
                }
                let sha = if recovering && matches!(scenario.fault, Fault::TagMoved) {
                    "b"
                } else {
                    "a"
                };
                Response::json(
                    200,
                    json!({"object":{"type":"commit","sha":sha.repeat(40)}}),
                )
            }
            ("GET", "/repos/test/repo/releases/7/assets?per_page=100&page=1") => {
                if recovering {
                    if let Fault::ListError(status) = scenario.fault {
                        return Response::text(status, "unavailable");
                    }
                }
                if state.deleted && matches!(scenario.fault, Fault::ReadAfterDelete) {
                    return Response::text(503, "unavailable");
                }
                let mut other = remote("two.tar.gz");
                if recovering && matches!(scenario.fault, Fault::DuplicateId) {
                    other["id"] = state.asset.as_ref().unwrap()["id"].clone();
                }
                let mut rows = vec![other];
                if let Some(asset) = &state.asset {
                    rows.push(asset.clone());
                    if recovering && matches!(scenario.fault, Fault::DuplicateName) {
                        rows.push(asset.clone());
                    }
                }
                Response::json(200, json!(rows))
            }
            ("POST", "/repos/test/repo/releases/7/assets?name=one.tar.gz") => {
                assert!(state.asset.is_none(), "cannot replace an existing asset");
                assert_eq!(request.body, b"one.tar.gz");
                state.posts += 1;
                state.mutations.push("POST".into());
                state.deleted = false;
                let id = 100 + state.posts as u64;
                if state.posts > scenario.failures {
                    state.asset = Some(uploaded(id));
                    return Response::json(201, state.asset.clone().unwrap());
                }
                state.asset = stored_asset(
                    if scenario.complete_on_attempt == Some(state.posts) {
                        Stored::Uploaded
                    } else {
                        scenario.stored
                    },
                    id,
                );
                match scenario.failure {
                    Failure::Http(status) => {
                        Response::text(status, "server response is not logged")
                    }
                    Failure::LostResponse => Response::disconnect(),
                    Failure::UnreadableSuccess => Response::text(201, "{truncated"),
                    Failure::TruncatedSuccess => Response::truncated(201, "{truncated"),
                }
            }
            ("GET", path) if path.starts_with("/repos/test/repo/releases/assets/") => {
                let id: u64 = path.rsplit('/').next().unwrap().parse().unwrap();
                assert_eq!(state.asset.as_ref().unwrap()["id"], id);
                match scenario.fault {
                    Fault::ByIdUploaded => state.asset = Some(uploaded(id)),
                    Fault::ByIdChanged => state.asset.as_mut().unwrap()["id"] = json!(999),
                    Fault::ByIdConflictingUpload => {
                        state.asset = stored_asset(Stored::WrongDigest, id)
                    }
                    _ => {}
                }
                Response::json(200, state.asset.clone().unwrap())
            }
            ("DELETE", path) if path.starts_with("/repos/test/repo/releases/assets/") => {
                let id: u64 = path.rsplit('/').next().unwrap().parse().unwrap();
                let asset = state.asset.as_ref().unwrap();
                assert_eq!(asset["id"], id);
                assert_eq!(asset["state"], "starter");
                assert_eq!(asset["digest"], Value::Null);
                state.deletes += 1;
                state.mutations.push(format!("DELETE {id}"));
                if let Fault::DeleteStatus(status) = scenario.fault {
                    return Response::text(status, "delete was not confirmed");
                }
                state.deleted = true;
                if !matches!(scenario.fault, Fault::DeletedStillPresent) {
                    state.asset = None;
                }
                if matches!(scenario.fault, Fault::DeleteResponseLost) {
                    Response::disconnect()
                } else {
                    Response::text(204, "")
                }
            }
            _ => panic!("unexpected request {} {}", request.method, request.path),
        }
    });
    (server, state)
}

async fn run(scenario: Scenario) -> (Result<(), String>, Arc<Mutex<State>>) {
    let (server, state) = server(scenario);
    let directory = tempfile::tempdir().unwrap();
    let accepted = accepted(directory.path());
    let result = upload_missing_asset(
        &api(&server),
        "test/repo",
        &accepted,
        7,
        &accepted.assets[0],
        Duration::ZERO,
    )
    .await;
    (result, state)
}

#[tokio::test]
async fn lost_and_unreadable_success_responses_are_reconciled_without_reupload() {
    for failure in [
        Failure::LostResponse,
        Failure::UnreadableSuccess,
        Failure::TruncatedSuccess,
        Failure::Http(500),
    ] {
        let (result, state) = run(Scenario {
            failure,
            stored: Stored::Uploaded,
            ..Scenario::default()
        })
        .await;
        result.unwrap();
        let state = state.lock().unwrap();
        assert_eq!(state.posts, 1);
        assert_eq!(state.deletes, 0);
    }
}

#[tokio::test]
async fn failed_upload_cleans_only_its_new_starter_then_retries() {
    for failure in [
        Failure::Http(500),
        Failure::Http(502),
        Failure::Http(503),
        Failure::Http(504),
        Failure::LostResponse,
    ] {
        let (result, state) = run(Scenario {
            failure,
            ..Scenario::default()
        })
        .await;
        result.unwrap();
        let state = state.lock().unwrap();
        assert_eq!(state.mutations, ["POST", "DELETE 101", "POST"]);
        assert_eq!(state.asset.as_ref().unwrap()["state"], "uploaded");
    }
}

#[tokio::test]
async fn preexisting_starter_never_grants_cleanup_permission() {
    let (result, state) = run(Scenario {
        preexisting_starter: true,
        ..Scenario::default()
    })
    .await;
    assert!(result.unwrap_err().contains("incomplete"));
    let state = state.lock().unwrap();
    assert!(state.mutations.is_empty());
    assert_eq!(state.asset.as_ref().unwrap()["id"], 50);
}

#[tokio::test]
async fn retry_budget_leaves_the_final_failed_state_for_review() {
    for stored in [Stored::Absent, Stored::Starter] {
        let (result, state) = run(Scenario {
            failures: usize::MAX,
            stored,
            ..Scenario::default()
        })
        .await;
        assert!(result.unwrap_err().contains("exhausted 3"));
        let state = state.lock().unwrap();
        assert_eq!(state.posts, 3);
        match stored {
            Stored::Starter => {
                assert_eq!(state.deletes, 2);
                assert_eq!(state.asset.as_ref().unwrap()["id"], 103);
            }
            _ => {
                assert_eq!(state.deletes, 0);
                assert!(state.asset.is_none());
            }
        }
    }
}

#[tokio::test]
async fn final_attempt_that_committed_before_losing_response_is_success() {
    let (result, state) = run(Scenario {
        failures: 3,
        failure: Failure::LostResponse,
        stored: Stored::Absent,
        complete_on_attempt: Some(3),
        ..Scenario::default()
    })
    .await;
    result.unwrap();
    let state = state.lock().unwrap();
    assert_eq!(state.posts, 3);
    assert_eq!(state.deletes, 0);
    assert_eq!(state.asset.as_ref().unwrap()["state"], "uploaded");
}

#[tokio::test]
async fn non_retryable_http_statuses_do_not_clean_up_or_post_again() {
    for status in [400, 401, 403, 404, 422, 501] {
        let (result, state) = run(Scenario {
            failure: Failure::Http(status),
            ..Scenario::default()
        })
        .await;
        assert!(result.unwrap_err().contains(&status.to_string()));
        let state = state.lock().unwrap();
        assert_eq!(state.posts, 1);
        assert_eq!(state.deletes, 0);
    }
}

#[tokio::test]
async fn conflicting_or_unknown_remote_bytes_are_never_removed() {
    for stored in [
        Stored::WrongDigest,
        Stored::WrongSize,
        Stored::UnknownState,
        Stored::StarterWithDigest,
        Stored::WrongStarterSize,
    ] {
        let (result, state) = run(Scenario {
            stored,
            ..Scenario::default()
        })
        .await;
        assert!(result.is_err(), "{stored:?}");
        let state = state.lock().unwrap();
        assert_eq!(state.posts, 1);
        assert_eq!(state.deletes, 0);
    }
}

#[tokio::test]
async fn changed_release_tag_inventory_or_read_failure_stops_before_cleanup() {
    for fault in [
        Fault::PublicRelease,
        Fault::ReleaseId,
        Fault::Notes,
        Fault::Candidate,
        Fault::TagMissing,
        Fault::TagMoved,
        Fault::ListError(404),
        Fault::ListError(503),
        Fault::DuplicateName,
        Fault::DuplicateId,
    ] {
        let (result, state) = run(Scenario {
            fault,
            ..Scenario::default()
        })
        .await;
        assert!(result.is_err(), "{fault:?}");
        let state = state.lock().unwrap();
        assert_eq!(state.posts, 1);
        assert_eq!(state.deletes, 0);
    }
}

#[tokio::test]
async fn fresh_asset_by_id_check_accepts_completion_and_rejects_changed_identity() {
    for fault in [
        Fault::ByIdUploaded,
        Fault::ByIdChanged,
        Fault::ByIdConflictingUpload,
    ] {
        let (result, state) = run(Scenario {
            fault,
            ..Scenario::default()
        })
        .await;
        assert_eq!(
            result.is_ok(),
            matches!(fault, Fault::ByIdUploaded),
            "{fault:?}: {result:?}"
        );
        let state = state.lock().unwrap();
        assert_eq!(state.posts, 1);
        assert_eq!(state.deletes, 0);
    }
}

#[tokio::test]
async fn uncertain_delete_or_unconfirmed_absence_never_authorizes_another_post() {
    for fault in [
        Fault::DeleteStatus(404),
        Fault::DeleteStatus(500),
        Fault::DeleteResponseLost,
        Fault::DeletedStillPresent,
        Fault::ReadAfterDelete,
    ] {
        let (result, state) = run(Scenario {
            fault,
            ..Scenario::default()
        })
        .await;
        assert!(result.is_err(), "{fault:?}");
        let state = state.lock().unwrap();
        assert_eq!(state.posts, 1);
        assert_eq!(state.deletes, 1);
    }
}
