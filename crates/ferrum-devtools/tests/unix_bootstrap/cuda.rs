//! Architecture selection and failure boundaries through the real installer.
use super::*;

fn cuda_fixture() -> Fixture {
    let mut fixture = Fixture::new("1.2.3");
    fixture.asset("1.2.3", "linux-x86_64", false, false);
    for arch in [CudaArch::Sm89, CudaArch::Sm120] {
        fixture.asset_with_arch("1.2.3", "linux-x86_64", Some(arch), false);
    }
    fixture
}

#[test]
fn cuda_packages_follow_all_visible_compute_capabilities() {
    let fixture = cuda_fixture();
    let server = Server::new(fixture.assets.clone());
    for (caps, arch) in [
        ("8.9", CudaArch::Sm89),
        (" 8.9\r\n8.9 ", CudaArch::Sm89),
        ("12.0", CudaArch::Sm120),
        (" 12.0\r\n12.0 ", CudaArch::Sm120),
    ] {
        for backend in ["auto", "cuda"] {
            let host = fixture.host();
            let before = server.requests().len();
            success(&fixture.run(
                host.path(),
                &server,
                "1.2.3",
                "Linux",
                backend,
                Some(caps),
                false,
            ));
            let asset = format!("/v1.2.3/ferrum-linux-x86_64{}.tar.gz", arch.suffix());
            assert_eq!(
                &server.requests()[before..],
                [
                    format!("{asset}.sha256"),
                    format!("{asset}.binary.sha256"),
                    asset
                ]
            );
            let binary = host.path().join(".local/bin/ferrum");
            assert!(fs::read_link(&binary)
                .unwrap()
                .to_string_lossy()
                .contains("/1.2.3-cuda-"));
            let installed = Command::new(binary).arg("--version").output().unwrap();
            success(&installed);
            assert_eq!(
                String::from_utf8(installed.stdout).unwrap().trim(),
                "ferrum 1.2.3"
            );
        }
    }
}

#[test]
fn cuda_unknown_empty_or_mixed_capabilities_only_auto_select_cpu() {
    let fixture = cuda_fixture();
    let server = Server::new(fixture.assets.clone());
    for caps in [
        None,
        Some(""),
        Some("8.6"),
        Some("8.9\n12.0"),
        Some("12.0\n8.9"),
        Some("12.0\nN/A"),
        Some("12.0\n\n12.0"),
        Some("1 2.0"),
    ] {
        for backend in ["auto", "cuda"] {
            let host = fixture.host();
            let before = server.requests().len();
            let result = fixture.run(host.path(), &server, "1.2.3", "Linux", backend, caps, false);
            if backend == "auto" {
                success(&result);
                assert!(fs::read_link(host.path().join(".local/bin/ferrum"))
                    .unwrap()
                    .to_string_lossy()
                    .contains("/1.2.3-cpu-"));
                assert!(server.requests()[before..]
                    .iter()
                    .all(|path| !path.contains("-cuda-")));
            } else {
                assert!(!result.status.success());
                assert!(String::from_utf8_lossy(&result.stderr)
                    .contains("same supported compute capability"));
                assert_eq!(server.requests().len(), before);
                assert!(!host.path().join(".local/bin/ferrum").exists());
            }
        }
    }
}

#[test]
fn cuda_wrong_sha_preserves_installation_without_architecture_or_cpu_fallback() {
    let fixture = cuda_fixture();
    let good = Server::new(fixture.assets.clone());
    for (caps, arch) in [("8.9", CudaArch::Sm89), ("12.0", CudaArch::Sm120)] {
        for corrupt_binary_checksum in [false, true] {
            let host = fixture.host();
            success(&fixture.run(host.path(), &good, "1.2.3", "Linux", "cpu", None, false));
            let binary = host.path().join(".local/bin/ferrum");
            let previous = fs::read_link(&binary).unwrap();
            let previous_bytes = fs::read(&binary).unwrap();
            let asset = format!("/v1.2.3/ferrum-linux-x86_64{}.tar.gz", arch.suffix());
            let mut assets = fixture.assets.clone();
            if corrupt_binary_checksum {
                assets.insert(
                    format!("{asset}.binary.sha256"),
                    format!("{}  ferrum\n", "0".repeat(64)).into_bytes(),
                );
            } else {
                assets.get_mut(&asset).unwrap().push(1);
            }
            let bad = Server::new(assets);
            let result = fixture.run(
                host.path(),
                &bad,
                "1.2.3",
                "Linux",
                "auto",
                Some(caps),
                false,
            );
            assert!(!result.status.success());
            assert!(
                String::from_utf8_lossy(&result.stderr).contains(if corrupt_binary_checksum {
                    "Binary SHA-256 mismatch"
                } else {
                    "Archive SHA-256 mismatch"
                })
            );
            assert_eq!(fs::read_link(&binary).unwrap(), previous);
            assert_eq!(fs::read(&binary).unwrap(), previous_bytes);
            assert!(bad.requests().iter().all(|path| path.starts_with(&asset)));
        }
    }
}

#[test]
fn cuda_release_missing_matching_asset_does_not_use_another_architecture() {
    let fixture = cuda_fixture();
    let good = Server::new(fixture.assets.clone());
    // Includes an older release containing only sm89 and the inverse boundary:
    // an sm89 host must never substitute an available sm120 package either.
    for (caps, missing) in [("12.0", CudaArch::Sm120), ("8.9", CudaArch::Sm89)] {
        let asset = format!("/v1.2.3/ferrum-linux-x86_64{}.tar.gz", missing.suffix());
        let assets = fixture
            .assets
            .iter()
            .filter(|(path, _)| !path.starts_with(&asset))
            .map(|(path, bytes)| (path.clone(), bytes.clone()))
            .collect();
        let old_release = Server::new(assets);
        for backend in ["auto", "cuda"] {
            let host = fixture.host();
            success(&fixture.run(host.path(), &good, "1.2.3", "Linux", "cpu", None, false));
            let binary = host.path().join(".local/bin/ferrum");
            let previous = fs::read_link(&binary).unwrap();
            let before = old_release.requests().len();
            let result = fixture.run(
                host.path(),
                &old_release,
                "1.2.3",
                "Linux",
                backend,
                Some(caps),
                false,
            );
            assert!(!result.status.success());
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert!(
                stderr.contains("matching") && stderr.contains(asset.rsplit('/').next().unwrap()),
                "{stderr}"
            );
            assert!(stderr.contains("--backend cpu"), "{stderr}");
            assert_eq!(fs::read_link(&binary).unwrap(), previous);
            assert_eq!(
                &old_release.requests()[before..],
                [format!("{asset}.sha256")]
            );
        }
    }
}

#[test]
fn cuda_probe_uses_executable_wsl_fallback_when_path_has_no_smi() {
    let fixture = Fixture::new("1.2.3");
    let host = fixture.host();
    let commands = host.path().join("commands");
    fs::remove_file(commands.join("nvidia-smi")).unwrap();
    symlink("/usr/bin/awk", commands.join("awk")).unwrap();
    let wsl = host.path().join("wsl");
    fs::create_dir(&wsl).unwrap();
    let fallback = wsl.join("nvidia-smi");
    fs::copy(&fixture.executable, &fallback).unwrap();
    fs::set_permissions(&fallback, fs::Permissions::from_mode(0o755)).unwrap();
    let source =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/install.sh"))
            .unwrap();
    let functions = source.strip_suffix("main \"$@\"\n").unwrap();
    let query = |caps: Option<&str>| {
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                &format!("{functions}\ndetect_cuda_arch \"$1\""),
                "probe-test",
            ])
            .arg(&fallback)
            .env("PATH", &commands)
            .env_remove("FIXTURE_GPU_CAPABILITIES");
        if let Some(caps) = caps {
            command.env("FIXTURE_GPU_CAPABILITIES", caps);
        }
        command.output().unwrap()
    };
    let output = query(Some("12.0\n12.0"));
    success(&output);
    assert_eq!(output.stdout, b"sm120\n");
    assert!(
        !query(None).status.success(),
        "a failed device query cannot grant eligibility"
    );
    fs::set_permissions(&fallback, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(
        !query(Some("12.0")).status.success(),
        "fallback must be executable"
    );
    symlink(&fixture.executable, commands.join("nvidia-smi")).unwrap();
    let output = query(Some("8.9"));
    success(&output);
    assert_eq!(
        output.stdout, b"sm89\n",
        "PATH utility takes priority over fallback"
    );
}
