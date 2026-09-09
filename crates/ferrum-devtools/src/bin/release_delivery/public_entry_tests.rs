use super::*;
use std::collections::BTreeMap;

#[path = "../../../tests/support/http_fixture.rs"]
mod http_fixture;

#[tokio::test]
async fn public_entry_requires_actual_candidate_bytes_and_retains_http_failures() {
    let source = b"#!/bin/sh\nprintf 'candidate installer\\n'\n";
    let server = http_fixture::Server::new(BTreeMap::from([
        ("/install.sh".into(), source.to_vec()),
        (
            "/stale.sh".into(),
            b"#!/bin/sh\necho old release\n".to_vec(),
        ),
    ]));
    let dir = tempfile::tempdir().unwrap();
    let expected = dir.path().join("candidate.sh");
    fs::write(&expected, source).unwrap();
    for (path, passed, http_status) in [
        ("install.sh", true, 200),
        ("stale.sh", false, 200),
        ("missing.sh", false, 404),
    ] {
        let output = dir.path().join(format!("{path}.json"));
        let result = verify(EntryArgs {
            url: format!("{}/{path}", server.url),
            expected_script: expected.clone(),
            output: output.clone(),
        })
        .await;
        assert_eq!(result.is_ok(), passed);
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(report["http_status"], http_status);
        assert_eq!(report["status"], if passed { "passed" } else { "failed" });
        assert_eq!(report["error"].is_null(), passed);
        if passed {
            assert_eq!(report["actual_sha256"], report["expected_sha256"]);
        }
    }
}
