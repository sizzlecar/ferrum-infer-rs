//! Manual native installation check using an actual staged CPU archive. The
//! ordinary tests retain tiny fixtures for fault injection and device selection.
use super::*;

#[test]
#[ignore = "requires FERRUM_BOOTSTRAP_ASSET_DIR and FERRUM_BOOTSTRAP_EVIDENCE_DIR"]
fn staged_cpu_archive_installs_through_http_and_starts_on_its_native_host() {
    let assets = PathBuf::from(std::env::var_os("FERRUM_BOOTSTRAP_ASSET_DIR").unwrap());
    let evidence = PathBuf::from(std::env::var_os("FERRUM_BOOTSTRAP_EVIDENCE_DIR").unwrap());
    fs::create_dir(&evidence).expect("reserve a fresh installation evidence directory");
    let (target, name) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => ("aarch64-apple-darwin", "ferrum-macos-aarch64-cpu.tar.gz"),
        ("linux", "x86_64") => ("x86_64-unknown-linux-gnu", "ferrum-linux-x86_64.tar.gz"),
        other => panic!("no official CPU installer target for {other:?}"),
    };
    let abi: serde_json::Value =
        serde_json::from_slice(&fs::read(assets.join(format!("{name}.abi.json"))).unwrap())
            .unwrap();
    let version: serde_json::Value =
        serde_json::from_slice(&fs::read(assets.join(format!("{name}.version.json"))).unwrap())
            .unwrap();
    assert_eq!(abi["backend"], "cpu");
    assert_eq!(abi["target_triple"], target);
    assert_eq!(abi["asset_name"], name);
    for field in ["binary_sha256", "asset_sha256", "release_candidate_sha"] {
        assert_eq!(abi[field], version[field], "staged {field} identity");
    }
    let version = version["version"].as_str().unwrap();
    let archive = fs::read(assets.join(name)).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&archive)),
        abi["asset_sha256"].as_str().unwrap()
    );
    let script =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/install.sh")).unwrap();
    let script_sha256 = format!("{:x}", Sha256::digest(&script));
    let mut responses = BTreeMap::from([
        ("/install.sh".into(), script),
        (format!("/v{version}/{name}"), archive),
    ]);
    for suffix in [".sha256", ".binary.sha256"] {
        responses.insert(
            format!("/v{version}/{name}{suffix}"),
            fs::read(assets.join(format!("{name}{suffix}"))).unwrap(),
        );
    }
    let server = Server::new(responses);
    let host = evidence.join("user");
    fs::create_dir(&host).unwrap();
    let curl = Command::new("curl")
        .args(["-fsSL", &format!("{}/install.sh", server.url)])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut curl = curl;
    let installed = Command::new("sh")
        .args([
            "-s",
            "--",
            "--backend",
            "cpu",
            "--version",
            version,
            "--release-base-url",
            &server.url,
        ])
        .env("HOME", &host)
        .env("SHELL", "/bin/sh")
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .stdin(curl.stdout.take().unwrap())
        .output()
        .unwrap();
    let downloaded = curl.wait_with_output().unwrap();
    fs::write(evidence.join("download.stderr.log"), &downloaded.stderr).unwrap();
    fs::write(evidence.join("installer.stdout.log"), &installed.stdout).unwrap();
    fs::write(evidence.join("installer.stderr.log"), &installed.stderr).unwrap();
    assert!(
        downloaded.status.success(),
        "installer HTTP download failed"
    );
    assert!(installed.status.success(), "inspect installer.stderr.log");
    let binary = host.join(".local/bin/ferrum");
    let binary_sha256 = format!("{:x}", Sha256::digest(fs::read(&binary).unwrap()));
    assert_eq!(binary_sha256, abi["binary_sha256"].as_str().unwrap());
    let checked = Command::new(env!("CARGO_BIN_EXE_release_delivery"))
        .args([
            "installed",
            "--channel",
            "bootstrap",
            "--version",
            version,
            "--binary",
        ])
        .arg(&binary)
        .arg("--output")
        .arg(evidence.join("startup.json"))
        .output()
        .unwrap();
    fs::write(evidence.join("startup.stderr.log"), &checked.stderr).unwrap();
    assert!(checked.status.success(), "installed program failed startup");
    fs::write(
        evidence.join("source.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "scope": "native_cpu_http_installation_and_startup",
            "source_abi": abi,
            "script_sha256": script_sha256,
            "binary_sha256": binary_sha256,
            "model_execution_checked": false,
        }))
        .unwrap(),
    )
    .unwrap();
}
