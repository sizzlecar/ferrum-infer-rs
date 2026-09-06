use super::super::test_http::{Response, Server};
use super::*;
use std::sync::{Arc, Mutex};

fn artifact(name: &str) -> CrateArtifact {
    CrateArtifact {
        name: name.into(),
        version: "2.3.4".into(),
        sha256: sha256(name.as_bytes()),
    }
}
fn row(artifact: &CrateArtifact) -> Value {
    json!({"name": artifact.name, "vers": artifact.version, "cksum": artifact.sha256, "yanked": false})
}
fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test]
async fn registry_distinguishes_missing_existing_and_conflict_from_real_http() {
    for (status, body, expected) in [
        (404, "not found".into(), Ok(RegistryStatus::Missing)),
        (
            200,
            row(&artifact("example")).to_string(),
            Ok(RegistryStatus::Identical),
        ),
        (
            200,
            json!({"name":"example","vers":"2.3.3","cksum":sha256(b"older"),"yanked":false})
                .to_string(),
            Ok(RegistryStatus::Missing),
        ),
        (
            200,
            json!({"name":"example","vers":"2.3.4","cksum":sha256(b"different"),"yanked":false})
                .to_string(),
            Err("checksum conflict"),
        ),
        (
            200,
            json!({"name":"example","vers":"2.3.4","cksum":sha256(b"example"),"yanked":true})
                .to_string(),
            Err("yanked"),
        ),
        (
            500,
            "temporary failure".into(),
            Err("not treating it as missing"),
        ),
        (403, "forbidden".into(), Err("not treating it as missing")),
        (200, String::new(), Err("empty successful")),
        (200, "not json".into(), Err("malformed")),
    ] {
        let server = Server::new(move |request| {
            assert_eq!(request.method, "GET");
            assert_eq!(request.path, "/ex/am/example");
            Response::text(status, body.clone())
        });
        let result = registry_status(&client(), &server.url, &artifact("example")).await;
        match expected {
            Ok(value) => assert_eq!(result.unwrap(), value),
            Err(reason) => assert!(result.unwrap_err().contains(reason)),
        }
    }
}

#[tokio::test]
async fn partial_registry_progress_selects_only_missing_then_confirms_every_package() {
    let published = Arc::new(Mutex::new(BTreeSet::from(["base".to_string()])));
    let view = published.clone();
    let server = Server::new(move |request| {
        let name = request.path.rsplit('/').next().unwrap();
        if view.lock().unwrap().contains(name) {
            Response::text(200, row(&artifact(name)).to_string())
        } else {
            Response::text(404, "not found")
        }
    });
    let artifacts = [artifact("base"), artifact("consumer")];
    assert_eq!(
        reconcile(&client(), &server.url, &artifacts).await.unwrap(),
        ["consumer"]
    );
    published.lock().unwrap().insert("consumer".into());
    assert!(reconcile(&client(), &server.url, &artifacts)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn one_unknown_registry_result_prevents_returning_a_partial_missing_plan() {
    let server = Server::new(|request| {
        if request.path.ends_with("/base") {
            Response::text(404, "not found")
        } else {
            Response::text(429, "try later")
        }
    });
    assert!(reconcile(
        &client(),
        &server.url,
        &[artifact("base"), artifact("consumer")]
    )
    .await
    .unwrap_err()
    .contains("not treating it as missing"));
}

#[test]
fn inventory_obeys_manifest_publish_policy_and_rejects_version_drift() {
    let mut metadata = json!({"workspace_members":["a","b","c"],"packages":[
        {"id":"a","name":"public","version":"2.3.4","publish":null},
        {"id":"b","name":"private","version":"9.9.9","publish":[]},
        {"id":"c","name":"another-registry","version":"9.9.9","publish":["internal"]}
    ]});
    assert_eq!(
        publishable_packages(&metadata, "2.3.4").unwrap(),
        BTreeSet::from(["public".into()])
    );
    metadata["packages"][0]["version"] = json!("2.3.3");
    assert!(publishable_packages(&metadata, "2.3.4")
        .unwrap_err()
        .contains("accepted version"));
    metadata["packages"][0]["publish"] = json!([]);
    assert!(publishable_packages(&metadata, "2.3.4")
        .unwrap_err()
        .contains("no publishable"));
}

#[tokio::test]
async fn package_source_check_rejects_unaccepted_inputs_and_ignores_unrelated_files() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("package.rs"), "accepted source").unwrap();
    async fn git(directory: &Path, args: &[&str]) -> Vec<u8> {
        let output = Command::new("git")
            .arg("-c")
            .arg("user.name=Fixture")
            .arg("-c")
            .arg("user.email=fixture@example.invalid")
            .args(args)
            .current_dir(directory)
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        output.stdout
    }
    git(directory.path(), &["init", "-q"]).await;
    git(directory.path(), &["add", "package.rs"]).await;
    git(directory.path(), &["commit", "-qm", "fixture"]).await;
    let sha = String::from_utf8(git(directory.path(), &["rev-parse", "HEAD"]).await)
        .unwrap()
        .trim()
        .to_string();
    let accepted = AcceptedRelease {
        version: "2.3.4".into(),
        candidate_sha: sha,
        workspace: directory.path().into(),
        notes: String::new(),
        assets: vec![],
    };
    let inputs = BTreeMap::from([("package.rs".into(), directory.path().join("package.rs"))]);
    let expected = source_fingerprint(&accepted, &inputs).await.unwrap();
    std::fs::write(directory.path().join("local.log"), "unrelated output").unwrap();
    assert_eq!(
        source_fingerprint(&accepted, &inputs).await.unwrap(),
        expected
    );
    std::fs::write(directory.path().join("package.rs"), "unaccepted source").unwrap();
    assert!(source_fingerprint(&accepted, &inputs)
        .await
        .unwrap_err()
        .contains("differs from the accepted candidate"));
}
