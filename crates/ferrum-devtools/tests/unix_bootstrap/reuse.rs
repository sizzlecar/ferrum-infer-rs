use super::*;
use std::os::unix::fs::MetadataExt;

#[test]
fn repeat_verifies_small_checksums_without_redownloading_or_replacing_running_programs() {
    for (os, platform, backend, caps) in [
        ("Darwin", "macos-aarch64", "metal", None),
        ("Darwin", "macos-aarch64-cpu", "cpu", None),
        ("Linux", "linux-x86_64", "cpu", None),
        ("Linux", "linux-x86_64", "cuda", Some("8.9")),
    ] {
        let mut fixture = Fixture::new("1.2.3");
        fixture.asset("1.2.3", platform, backend == "cuda", false);
        let server = Server::new(fixture.assets.clone());
        let host = fixture.host();
        success(&fixture.run(host.path(), &server, "1.2.3", os, backend, caps, false));
        let program = host.path().join(".local/bin/ferrum");
        let pointer = fs::symlink_metadata(&program).unwrap();
        let target = fs::read_link(&program).unwrap();
        let mut session = VersionSession::start(&program);
        let pid = session.child.id();
        assert_eq!(session.version(), "ferrum 1.2.3");
        for _ in 0..2 {
            let before = server.requests().len();
            success(&fixture.run(host.path(), &server, "1.2.3", os, backend, caps, false));
            let requests = server.requests();
            assert_eq!(requests.len() - before, 2);
            assert!(requests[before..]
                .iter()
                .all(|path| path.ends_with(".sha256")));
            let after = fs::symlink_metadata(&program).unwrap();
            assert_eq!(after.ino(), pointer.ino());
            assert_eq!(after.modified().unwrap(), pointer.modified().unwrap());
            assert_eq!(fs::read_link(&program).unwrap(), target);
            assert_eq!(session.child.id(), pid);
            assert_eq!(session.version(), "ferrum 1.2.3");
        }
        session.finish();
    }
}

#[test]
fn linux_upgrade_preserves_old_process_and_activates_the_new_version_for_new_launches() {
    let mut first = Fixture::new("1.2.3");
    first.asset("1.2.3", "linux-x86_64", false, false);
    let mut next = Fixture::new("1.2.4");
    next.asset("1.2.4", "linux-x86_64", false, false);
    let mut assets = first.assets.clone();
    assets.extend(next.assets.clone());
    let server = Server::new(assets);
    let host = first.host();
    success(&first.run(host.path(), &server, "1.2.3", "Linux", "cpu", None, false));
    let program = host.path().join(".local/bin/ferrum");
    let previous = fs::read_link(&program).unwrap();
    let bytes = fs::read(&previous).unwrap();
    let mut session = VersionSession::start(&program);
    assert_eq!(session.version(), "ferrum 1.2.3");
    success(&first.run(host.path(), &server, "1.2.4", "Linux", "cpu", None, false));
    assert_eq!(session.version(), "ferrum 1.2.3");
    assert_eq!(fs::read(&previous).unwrap(), bytes);
    assert_ne!(fs::read_link(&program).unwrap(), previous);
    let next_launch = Command::new(&program).arg("--version").output().unwrap();
    success(&next_launch);
    assert_eq!(
        String::from_utf8_lossy(&next_launch.stdout).trim(),
        "ferrum 1.2.4"
    );
    session.finish();
}

#[test]
fn changing_backend_checks_the_selected_asset_and_can_reuse_a_retained_version() {
    let mut fixture = Fixture::new("1.2.3");
    fixture.asset("1.2.3", "macos-aarch64", false, false);
    fixture.asset("1.2.3", "macos-aarch64-cpu", false, false);
    let server = Server::new(fixture.assets.clone());
    let host = fixture.host();
    let program = host.path().join(".local/bin/ferrum");
    success(&fixture.run(
        host.path(),
        &server,
        "1.2.3",
        "Darwin",
        "metal",
        None,
        false,
    ));
    let metal = fs::read_link(&program).unwrap();
    let mut session = VersionSession::start(&program);
    assert_eq!(session.version(), "ferrum 1.2.3");
    let before = server.requests().len();
    success(&fixture.run(host.path(), &server, "1.2.3", "Darwin", "cpu", None, false));
    assert!(server.requests()[before..]
        .iter()
        .any(|path| path == "/v1.2.3/ferrum-macos-aarch64-cpu.tar.gz"));
    assert_ne!(fs::read_link(&program).unwrap(), metal);
    let before = server.requests().len();
    success(&fixture.run(
        host.path(),
        &server,
        "1.2.3",
        "Darwin",
        "metal",
        None,
        false,
    ));
    assert!(server.requests()[before..]
        .iter()
        .all(|path| path.ends_with(".sha256")));
    assert_eq!(fs::read_link(&program).unwrap(), metal);
    assert_eq!(session.version(), "ferrum 1.2.3");
    session.finish();
}

#[test]
fn failed_reuse_preserves_the_selected_version_and_running_process() {
    for changed in [
        "published-checksum",
        "missing-file",
        "symlink-file",
        "loader",
    ] {
        let mut fixture = Fixture::new("1.2.3");
        fixture.asset("1.2.3", "linux-x86_64", true, false);
        let server = Server::new(fixture.assets.clone());
        let host = fixture.host();
        success(&fixture.run(
            host.path(),
            &server,
            "1.2.3",
            "Linux",
            "cuda",
            Some("8.9"),
            false,
        ));
        let program = host.path().join(".local/bin/ferrum");
        let previous = fs::read_link(&program).unwrap();
        let mut session = VersionSession::start(&program);
        assert_eq!(session.version(), "ferrum 1.2.3");
        match changed {
            "published-checksum" => {
                fixture.assets.insert(
                    "/v1.2.3/ferrum-linux-x86_64-cuda-sm89.tar.gz.sha256".into(),
                    format!("{}  ferrum-linux-x86_64-cuda-sm89.tar.gz\n", "a".repeat(64))
                        .into_bytes(),
                );
            }
            "missing-file" => {
                fs::remove_file(previous.parent().unwrap().join("README.md")).unwrap()
            }
            "symlink-file" => {
                let path = previous.parent().unwrap().join("LICENSE");
                fs::remove_file(&path).unwrap();
                symlink(&previous, path).unwrap();
            }
            "loader" => {}
            _ => unreachable!(),
        }
        let server = Server::new(fixture.assets.clone());
        let result = fixture.run(
            host.path(),
            &server,
            "1.2.3",
            "Linux",
            "cuda",
            Some("8.9"),
            changed == "loader",
        );
        assert!(!result.status.success(), "{changed}: {result:?}");
        assert!(server
            .requests()
            .iter()
            .all(|path| path.ends_with(".sha256")));
        assert_eq!(fs::read_link(&program).unwrap(), previous);
        assert_eq!(session.version(), "ferrum 1.2.3");
        session.finish();
    }
}
