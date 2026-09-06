use super::super::test_http::{Response, Server};
use super::*;
use std::sync::{Arc, Mutex};

fn accepted(directory: &std::path::Path) -> AcceptedRelease {
    let assets = ["one.tar.gz", "two.tar.gz"]
        .into_iter()
        .map(|name| {
            let path = directory.join(name);
            std::fs::write(&path, name.as_bytes()).unwrap();
            AcceptedAsset {
                path,
                name: name.into(),
                sha256: sha256(name.as_bytes()),
            }
        })
        .collect();
    AcceptedRelease {
        version: "2.3.4".into(),
        candidate_sha: "a".repeat(40),
        workspace: directory.into(),
        notes: "Verified candidate release".into(),
        assets,
    }
}
fn api(server: &Server) -> GitHub {
    GitHub {
        client: Client::builder().no_proxy().build().unwrap(),
        base: server.url.clone(),
        uploads: server.url.clone(),
        token: "loopback-fixture-token".into(),
    }
}
fn remote(name: &str) -> Value {
    json!({"id": if name == "one.tar.gz" { 10 } else { 11 }, "name":name,"size":name.len(),"digest":format!("sha256:{}",sha256(name.as_bytes())),"state":"uploaded"})
}
fn release(draft: bool) -> Value {
    json!({"id": 7,"tag_name":"v2.3.4","body":"Verified candidate release","draft":draft,"prerelease":false,"published_at": if draft { Value::Null } else { json!("2026-01-01T00:00:00Z") }})
}

#[tokio::test]
async fn partial_draft_resumes_only_missing_assets_and_formalizes_after_complete_inventory() {
    #[derive(Default)]
    struct State {
        uploaded: bool,
        formal: bool,
        writes: Vec<String>,
    }
    let state = Arc::new(Mutex::new(State::default()));
    let view = state.clone();
    let server = Server::new(move |request| {
        let mut state = view.lock().unwrap();
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/repos/test/repo/git/ref/tags/v2.3.4") => Response::json(
                200,
                json!({"object":{"type":"commit","sha":"a".repeat(40)}}),
            ),
            ("GET", "/repos/test/repo/releases/tags/v2.3.4") => {
                Response::json(200, release(!state.formal))
            }
            ("GET", "/repos/test/repo/releases/7/assets?per_page=100&page=1") => Response::json(
                200,
                if state.uploaded {
                    json!([remote("one.tar.gz"), remote("two.tar.gz")])
                } else {
                    json!([remote("one.tar.gz")])
                },
            ),
            ("POST", "/repos/test/repo/releases/7/assets?name=two.tar.gz") => {
                assert!(!state.uploaded);
                assert!(!state.formal);
                assert_eq!(request.body, b"two.tar.gz");
                state.uploaded = true;
                state.writes.push("upload two".into());
                Response::json(201, remote("two.tar.gz"))
            }
            ("PATCH", "/repos/test/repo/releases/7") => {
                assert!(state.uploaded);
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                assert_eq!(body["draft"], false);
                state.formal = true;
                state.writes.push("formal".into());
                Response::json(200, release(false))
            }
            _ => panic!(
                "unexpected loopback request {} {}",
                request.method, request.path
            ),
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let accepted = accepted(directory.path());
    reconcile_release(&api(&server), "test/repo", &accepted)
        .await
        .unwrap();
    assert_eq!(state.lock().unwrap().writes, ["upload two", "formal"]);
    reconcile_release(&api(&server), "test/repo", &accepted)
        .await
        .unwrap();
    assert_eq!(
        state.lock().unwrap().writes,
        ["upload two", "formal"],
        "identical published state must perform no writes"
    );
}

#[tokio::test]
async fn existing_asset_conflict_and_query_error_do_not_upload() {
    for conflict in [false, true] {
        let writes = Arc::new(Mutex::new(0));
        let view = writes.clone();
        let server = Server::new(move |request| {
            if request.method != "GET" {
                *view.lock().unwrap() += 1;
                return Response::text(500, "unexpected mutation");
            }
            match request.path.as_str() {
                "/repos/test/repo/git/ref/tags/v2.3.4" => Response::json(
                    200,
                    json!({"object":{"type":"commit","sha":"a".repeat(40)}}),
                ),
                "/repos/test/repo/releases/tags/v2.3.4" => {
                    if conflict {
                        Response::json(200, release(true))
                    } else {
                        Response::text(503, "unavailable")
                    }
                }
                "/repos/test/repo/releases/7/assets?per_page=100&page=1" => {
                    let mut asset = remote("one.tar.gz");
                    asset["digest"] = json!(format!("sha256:{}", sha256(b"conflicting")));
                    Response::json(200, json!([asset]))
                }
                _ => panic!("unexpected read {}", request.path),
            }
        });
        let directory = tempfile::tempdir().unwrap();
        let error = reconcile_release(&api(&server), "test/repo", &accepted(directory.path()))
            .await
            .unwrap_err();
        assert!(error.contains(if conflict {
            "checksum conflict"
        } else {
            "not treating this as missing"
        }));
        assert_eq!(*writes.lock().unwrap(), 0);
    }
}

#[tokio::test]
async fn missing_tag_is_created_only_for_an_existing_accepted_commit_and_then_verified() {
    let created = Arc::new(Mutex::new(false));
    let view = created.clone();
    let server = Server::new(move |request| {
        let mut created = view.lock().unwrap();
        if request.method == "GET" && request.path.ends_with("/git/ref/tags/v2.3.4") {
            return if *created {
                Response::json(
                    200,
                    json!({"object":{"type":"commit","sha":"a".repeat(40)}}),
                )
            } else {
                Response::text(404, "missing")
            };
        }
        if request.method == "GET" && request.path.contains("/git/commits/") {
            return Response::json(200, json!({"sha":"a".repeat(40)}));
        }
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/repos/test/repo/git/refs");
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["sha"], "a".repeat(40));
        assert_eq!(body["ref"], "refs/tags/v2.3.4");
        *created = true;
        Response::json(201, json!({"ok":true}))
    });
    let directory = tempfile::tempdir().unwrap();
    ensure_tag(&api(&server), "test/repo", &accepted(directory.path()))
        .await
        .unwrap();
    assert!(*created.lock().unwrap());
}

#[test]
fn formula_edits_only_release_coordinates_and_rejects_same_version_conflicts() {
    let directory = tempfile::tempdir().unwrap();
    let accepted = accepted(directory.path());
    let source = format!("class Ferrum < Formula\n  desc \"unchanged\"\n  url \"https://github.com/test/repo/releases/download/v2.3.3/one.tar.gz\"\n  sha256 \"{}\" # retained comment\n  def install\n    bin.install \"ferrum\"\n  end\nend\n", "b".repeat(64));
    let updated = update_formula(&source, "test/repo", &accepted).unwrap();
    assert_eq!(
        updated,
        source
            .replace("v2.3.3/", "v2.3.4/")
            .replace(&"b".repeat(64), &accepted.assets[0].sha256)
    );
    assert_eq!(
        update_formula(&updated, "test/repo", &accepted).unwrap(),
        updated
    );
    let conflict = updated.replace(&accepted.assets[0].sha256, &"c".repeat(64));
    assert!(update_formula(&conflict, "test/repo", &accepted)
        .unwrap_err()
        .contains("different archive bytes"));
    assert!(update_formula(
        &source.replace("v2.3.3/", "v9.0.0/"),
        "test/repo",
        &accepted
    )
    .unwrap_err()
    .contains("newer release"));
    std::fs::write(&accepted.assets[0].path, b"changed locally").unwrap();
    assert!(update_formula(&source, "test/repo", &accepted)
        .unwrap_err()
        .contains("asset bytes changed"));
}
