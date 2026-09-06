use super::*;

fn good() -> Vec<Observation> {
    COMMANDS
        .iter()
        .enumerate()
        .map(|(index, arguments)| Observation {
            arguments: arguments
                .iter()
                .map(|argument| (*argument).into())
                .collect(),
            exit_code: Some(0),
            stdout: if index == 0 {
                "ferrum 0.8.8\n".into()
            } else {
                "Usage: documented startup interface\n".into()
            },
            stderr: String::new(),
        })
        .collect()
}
#[test]
fn public_install_startup_requires_all_commands_and_formal_matching_version() {
    verify_startup("0.8.8", &good()).unwrap();
    assert!(verify_startup("0.8.9", &good()).is_err());
    assert!(verify_startup("0.8.8-rc.1", &good()).is_err());
    for index in 0..COMMANDS.len() {
        let mut observations = good();
        observations.remove(index);
        assert!(verify_startup("0.8.8", &observations).is_err());
        let mut observations = good();
        observations[index].exit_code = Some(1);
        assert!(verify_startup("0.8.8", &observations).is_err());
        let mut observations = good();
        observations[index].stdout = " \n".into();
        assert!(verify_startup("0.8.8", &observations).is_err());
        let mut observations = good();
        observations[index].arguments = vec!["other".into()];
        assert!(verify_startup("0.8.8", &observations).is_err());
    }
}
#[tokio::test]
async fn public_install_reuses_actual_probe_and_rejects_non_ferrum_executable() {
    let binary = std::env::current_exe().unwrap();
    let observation = probe(&binary, &["--help"]).await.unwrap();
    assert_eq!(observation.exit_code, Some(0));
    assert!(!observation.stdout.trim().is_empty());
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("cargo.json");
    assert!(verify(InstalledArgs {
        binary: binary.clone(),
        version: "0.8.8".into(),
        channel: Channel::Cargo,
        output: output.clone()
    })
    .await
    .is_err());
    let report: InstalledReport = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report.channel, Channel::Cargo);
    assert_eq!(report.status, Status::Failed);
    assert!(report.error.is_some());
    assert_eq!(report.binary_sha256, Some(sha256(&binary).unwrap()));
    assert_eq!(report.binary_sha256_after, report.binary_sha256);
    assert_eq!(
        report
            .observations
            .iter()
            .map(|item| item.arguments.clone())
            .collect::<Vec<_>>(),
        COMMANDS
            .iter()
            .map(|items| items
                .iter()
                .map(|item| (*item).to_owned())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
}
#[cfg(unix)]
#[tokio::test]
async fn public_install_preserves_real_nonzero_exit_observations() {
    let directory = tempfile::tempdir().unwrap();
    let binary = PathBuf::from("/usr/bin/false");
    let output = directory.path().join("brew.json");
    assert!(verify(InstalledArgs {
        binary: binary.clone(),
        version: "0.8.8".into(),
        channel: Channel::Homebrew,
        output: output.clone()
    })
    .await
    .is_err());
    let report: InstalledReport = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    assert_eq!(report.channel, Channel::Homebrew);
    assert_eq!(report.status, Status::Failed);
    assert_eq!(report.observations.len(), COMMANDS.len());
    // GNU false implements successful --help/--version responses; BSD false
    // does not. Product subcommands exercise the portable nonzero behavior.
    for arguments in [["run", "--help"], ["serve", "--help"]] {
        let observation = report
            .observations
            .iter()
            .find(|item| item.arguments == arguments)
            .expect("each requested product subcommand must retain its observation");
        assert_eq!(observation.exit_code, Some(1));
        assert!(observation.stdout.is_empty());
    }
    assert_eq!(report.binary_sha256, Some(sha256(&binary).unwrap()));
    assert_eq!(report.binary_sha256_after, report.binary_sha256);
}
#[tokio::test]
async fn public_install_missing_binary_keeps_failure_and_existing_evidence_is_not_overwritten() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("missing.json");
    let arguments = || InstalledArgs {
        binary: directory.path().join("absent"),
        version: "0.8.8".into(),
        channel: Channel::Cargo,
        output: output.clone(),
    };
    assert!(verify(arguments())
        .await
        .unwrap_err()
        .contains("resolve installed binary"));
    let bytes = fs::read(&output).unwrap();
    let report: InstalledReport = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(report.status, Status::Failed);
    assert!(report.binary_sha256.is_none());
    assert!(report.observations.is_empty());
    assert!(report.error.is_some());
    assert!(verify(arguments()).await.unwrap_err().contains("fresh"));
    assert_eq!(fs::read(&output).unwrap(), bytes);
}
