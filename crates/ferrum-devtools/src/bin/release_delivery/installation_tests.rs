use super::*;
use ferrum_bench_core::release_candidate::staging::{
    generate_manifests, AbiInput, Backend, CandidateInput,
};
use serde_json::json;

struct Fixture {
    directory: tempfile::TempDir,
    archive: PathBuf,
    abi: PathBuf,
    version_file: PathBuf,
}
impl Fixture {
    fn new(bytes: &[u8], entry: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let payload = directory.path().join("payload");
        fs::create_dir(&payload).unwrap();
        fs::write(payload.join(entry), bytes).unwrap();
        let archive = directory.path().join("fixture.tar.gz");
        let output = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(&payload)
            .arg(entry)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "tar fixture: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let input = CandidateInput {
            version: "2.3.4".into(),
            release_candidate_sha: "a".repeat(40),
            release_candidate_tag: "v2.3.4-rc.1".into(),
            staging_label: "fixture".into(),
            workflow_run_id: "fixture-run".into(),
            workflow_run_attempt: "1".into(),
        };
        let manifest = generate_manifests(
            &input,
            &AbiInput {
                backend: Backend::Cpu,
                target_triple: "x86_64-unknown-linux-gnu".into(),
                cargo_features: vec![],
                cuda_compute_capability: None,
                cuda_toolkit_image: None,
            },
            "fixture.tar.gz",
            &fs::read(&archive).unwrap(),
            bytes,
            "dependencies.txt",
            "ferrum: ELF 64-bit LSB executable\n\tlibc.so.6 => /lib/libc.so.6 (0x00001234)\n",
        )
        .unwrap();
        let abi = directory.path().join("fixture.tar.gz.abi.json");
        let version_file = directory.path().join("fixture.tar.gz.version.json");
        fs::write(&abi, serde_json::to_vec(&manifest.abi).unwrap()).unwrap();
        fs::write(
            &version_file,
            serde_json::to_vec(&manifest.version).unwrap(),
        )
        .unwrap();
        Self {
            directory,
            archive,
            abi,
            version_file,
        }
    }
    fn args(&self, extract_only: bool) -> InspectArgs {
        InspectArgs {
            abi: self.abi.clone(),
            version: "2.3.4".into(),
            candidate_sha: "a".repeat(40),
            extract_dir: self.directory.path().join("extracted"),
            output: self.directory.path().join("report.json"),
            extract_only,
        }
    }
    fn report(&self) -> InstallationReport {
        serde_json::from_slice(&fs::read(self.directory.path().join("report.json")).unwrap())
            .unwrap()
    }
    fn update_metadata(&self, key: &str, value: Value) {
        for path in [&self.abi, &self.version_file] {
            let mut metadata = read_json(path).unwrap();
            metadata[key] = value.clone();
            fs::write(path, serde_json::to_vec(&metadata).unwrap()).unwrap();
        }
    }
}

#[tokio::test]
async fn archive_tamper_is_rejected_before_extraction_or_execution() {
    let fixture = Fixture::new(b"this is never executed", "ferrum");
    fs::OpenOptions::new()
        .append(true)
        .open(&fixture.archive)
        .unwrap()
        .write_all(b"tamper")
        .unwrap();
    let args = fixture.args(false);
    let extracted = args.extract_dir.clone();
    let error = inspect(args).await.unwrap_err();
    assert!(error.contains("archive SHA-256"));
    assert!(
        !extracted.exists(),
        "unaccepted archive must not reach extraction"
    );
    assert!(!fixture.directory.path().join("report.json").exists());
}

#[tokio::test]
async fn embedded_binary_mismatch_is_rejected_before_writing_an_executable() {
    let fixture = Fixture::new(b"unaccepted embedded executable", "ferrum");
    fixture.update_metadata(
        "binary_sha256",
        json!(format!("{:x}", Sha256::digest(b"another binary"))),
    );
    let error = inspect(fixture.args(false)).await.unwrap_err();
    assert!(error.contains("different binary"));
    let report = fixture.report();
    assert_eq!(report.status, "failed");
    assert!(report.observations.is_empty());
    assert!(
        !report.binary.exists(),
        "a mismatched binary must not become executable"
    );
    assert!(verify_runtime(&report).is_err());
}

#[tokio::test]
async fn missing_ferrum_archive_entry_never_reaches_startup() {
    let fixture = Fixture::new(b"not at the executable entry", "renamed");
    assert!(inspect(fixture.args(false))
        .await
        .unwrap_err()
        .contains("could not read ferrum"));
    let report = fixture.report();
    assert!(report.observations.is_empty());
    assert!(!report.binary.exists());
    assert!(verify_runtime(&report).is_err());
}

#[tokio::test]
async fn extraction_only_records_not_run_even_when_all_bytes_match() {
    // Not a native executable: success proves extract-only did not try to spawn it.
    let fixture = Fixture::new(b"verified bytes, deliberately not an executable", "ferrum");
    inspect(fixture.args(true)).await.unwrap();
    let report = fixture.report();
    assert_eq!(report.status, "not_run");
    assert!(report.error.is_none());
    assert!(report.observations.is_empty());
    assert_eq!(sha256(&report.binary).unwrap(), report.binary_sha256);
    assert!(
        verify_runtime(&report).is_err(),
        "byte validation cannot count as startup execution"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn failed_native_executable_records_failure_and_stops_remaining_probes() {
    // A real native program with a nonzero exit; no generated shell executable.
    let path = if cfg!(target_os = "macos") {
        "/usr/bin/false"
    } else {
        "/bin/false"
    };
    let fixture = Fixture::new(&fs::read(path).unwrap(), "ferrum");
    assert!(inspect(fixture.args(false))
        .await
        .unwrap_err()
        .contains("startup probe"));
    let report = fixture.report();
    assert_eq!(report.status, "failed");
    assert_eq!(
        report.observations.len(),
        1,
        "failed version startup must stop later commands"
    );
    assert_eq!(report.observations[0].arguments, ["--version"]);
    assert_ne!(report.observations[0].exit_code, Some(0));
    assert!(verify_runtime(&report).is_err());
}

#[tokio::test]
async fn successful_probe_records_the_actual_test_executable_exit_and_output() {
    let executable = std::env::current_exe().unwrap();
    let result = probe(&executable, &["--help"]).await.unwrap();
    assert_eq!(result.exit_code, Some(0));
    assert!(!result.stdout.trim().is_empty());
    assert_eq!(result.arguments, ["--help"]);
}

fn complete_report() -> InstallationReport {
    InstallationReport {
        schema_version: 1,
        status: "passed".into(),
        version: "2.3.4".into(),
        candidate_sha: "a".repeat(40),
        asset_name: "fixture.tar.gz".into(),
        asset_sha256: "b".repeat(64),
        binary_sha256: "c".repeat(64),
        backend: "cpu".into(),
        target_triple: "x86_64-unknown-linux-gnu".into(),
        binary: PathBuf::from("/fixture/ferrum"),
        error: None,
        observations: [
            vec!["--version"],
            vec!["--help"],
            vec!["run", "--help"],
            vec!["serve", "--help"],
        ]
        .into_iter()
        .map(|arguments| {
            let stdout = if arguments == ["--version"] {
                "ferrum 2.3.4"
            } else {
                "Usage: ferrum [OPTIONS]"
            };
            Observation {
                arguments: arguments.into_iter().map(str::to_string).collect(),
                exit_code: Some(0),
                stdout: stdout.into(),
                stderr: String::new(),
            }
        })
        .collect(),
    }
}

#[test]
fn runtime_verifier_requires_each_actual_probe_and_cannot_accept_status_alone() {
    let good = complete_report();
    verify_runtime(&good).unwrap();
    for status in ["failed", "not_run", "", "pending"] {
        let mut report = good.clone();
        report.status = status.into();
        assert!(verify_runtime(&report).is_err());
    }
    for index in 0..good.observations.len() {
        let mut missing = good.clone();
        missing.observations.remove(index);
        assert!(verify_runtime(&missing).is_err());
        let mut failed = good.clone();
        failed.observations[index].exit_code = Some(1);
        assert!(verify_runtime(&failed).is_err());
        failed.observations[index].exit_code = None;
        assert!(verify_runtime(&failed).is_err());
        let mut absent = good.clone();
        absent.observations[index].stdout.clear();
        assert!(verify_runtime(&absent).is_err());
        let mut wrong_command = good.clone();
        wrong_command.observations[index].arguments = vec!["--version".into()];
        if index != 0 {
            assert!(verify_runtime(&wrong_command).is_err());
        }
    }
    let mut wrong_version = good.clone();
    wrong_version.observations[0].stdout = "ferrum 2.3.3".into();
    assert!(verify_runtime(&wrong_version).is_err());
    let mut with_error = good.clone();
    with_error.error = Some("startup failed".into());
    assert!(verify_runtime(&with_error).is_err());
    let mut no_identity = good.clone();
    no_identity.binary_sha256.clear();
    assert!(verify_runtime(&no_identity).is_err());
    let mut wrong_platform = good;
    wrong_platform.backend = "metal".into();
    assert!(verify_runtime(&wrong_platform).is_err());
}

#[test]
fn metadata_consumes_current_staging_schema_and_rejects_wrong_entry_or_platform() {
    let fixture = Fixture::new(b"fixture", "ferrum");
    let original_abi = read_json(&fixture.abi).unwrap();
    let original_version = read_json(&fixture.version_file).unwrap();
    validate_metadata(&original_abi, &original_version, &fixture.args(false)).unwrap();
    for field in [
        "binary_name",
        "binary_sha256",
        "asset_sha256",
        "release_candidate_sha",
    ] {
        let mut abi = original_abi.clone();
        abi.as_object_mut().unwrap().remove(field);
        assert!(validate_metadata(&abi, &original_version, &fixture.args(false)).is_err());
    }
    let mut abi = original_abi.clone();
    let mut version = original_version.clone();
    abi["binary_name"] = json!("other");
    version["binary_name"] = json!("other");
    assert!(validate_metadata(&abi, &version, &fixture.args(false))
        .unwrap_err()
        .contains("ferrum archive entry"));
    abi = original_abi.clone();
    abi["target_triple"] = json!("aarch64-apple-darwin");
    validate_metadata(&abi, &original_version, &fixture.args(false)).unwrap();
    abi["target_triple"] = json!("x86_64-pc-windows-msvc");
    assert!(
        validate_metadata(&abi, &original_version, &fixture.args(false))
            .unwrap_err()
            .contains("platform")
    );
    abi = original_abi;
    abi["target_triple"] = json!("");
    assert!(validate_metadata(&abi, &original_version, &fixture.args(false)).is_err());
}
