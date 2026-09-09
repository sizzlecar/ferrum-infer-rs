#![cfg(unix)]
//! Execute the product shell installer against local HTTP assets and tiny native Rust binaries.
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    os::unix::fs::{symlink, PermissionsExt},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
use tempfile::TempDir;

#[path = "support/http_fixture.rs"]
mod http_fixture;
use http_fixture::Server;

struct Fixture {
    root: TempDir,
    executable: PathBuf,
    assets: BTreeMap<String, Vec<u8>>,
}
impl Fixture {
    fn new(version: &str) -> Self {
        let root = tempfile::Builder::new()
            .prefix("ferrum bootstrap 中文 ")
            .tempdir()
            .unwrap();
        let executable = root.path().join("program");
        let result = Command::new("rustc")
            .args(["--edition=2021", "-C", "debuginfo=0"])
            .arg(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/unix_bootstrap_program.rs"),
            )
            .arg("-o")
            .arg(&executable)
            .env("BOOTSTRAP_FIXTURE_VERSION", version)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        Self {
            root,
            executable,
            assets: BTreeMap::new(),
        }
    }
    fn asset(&mut self, version: &str, platform: &str, cuda: bool, extra: bool) {
        let name = format!(
            "ferrum-{platform}{}.tar.gz",
            if cuda { "-cuda-sm89" } else { "" }
        );
        let directory = tempfile::tempdir_in(self.root.path()).unwrap();
        fs::copy(&self.executable, directory.path().join("ferrum")).unwrap();
        fs::write(directory.path().join("LICENSE"), b"fixture license").unwrap();
        fs::write(directory.path().join("README.md"), b"fixture readme").unwrap();
        let mut members = vec!["ferrum", "LICENSE", "README.md"];
        if cuda {
            fs::write(
                directory.path().join("CUDA-BUILD.txt"),
                b"fixture CUDA runtime",
            )
            .unwrap();
            members.push("CUDA-BUILD.txt");
        }
        if extra {
            fs::write(directory.path().join("unexpected"), b"must reject").unwrap();
            members.push("unexpected");
        }
        let archive = directory.path().join("asset.tar.gz");
        assert!(Command::new("tar")
            .env("LC_ALL", "C")
            .env("LANG", "C")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(directory.path())
            .args(members)
            .status()
            .unwrap()
            .success());
        let bytes = fs::read(&archive).unwrap();
        let prefix = format!("/v{version}/{name}");
        self.assets.insert(
            format!("{prefix}.sha256"),
            format!("{:x}  {name}\n", Sha256::digest(&bytes)).into_bytes(),
        );
        self.assets.insert(
            format!("{prefix}.binary.sha256"),
            format!(
                "{:x}  ferrum\n",
                Sha256::digest(fs::read(&self.executable).unwrap())
            )
            .into_bytes(),
        );
        self.assets.insert(prefix, bytes);
    }
    fn host(&self) -> TempDir {
        let root = tempfile::Builder::new()
            .prefix("ferrum user 中文 ")
            .tempdir()
            .unwrap();
        fs::create_dir(root.path().join("commands")).unwrap();
        for command in ["uname", "nvidia-smi"] {
            symlink(&self.executable, root.path().join("commands").join(command)).unwrap();
        }
        root
    }
    fn run(
        &self,
        host: &Path,
        server: &Server,
        version: &str,
        os: &str,
        backend: &str,
        caps: Option<&str>,
        missing: bool,
    ) -> Output {
        let path = format!(
            "{}:{}:/usr/bin:/bin:/usr/sbin:/sbin",
            host.join(".local/bin").display(),
            host.join("commands").display()
        );
        let mut command = Command::new("sh");
        command
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/install.sh"))
            .args([
                "--version",
                version,
                "--backend",
                backend,
                "--release-base-url",
                &server.url,
            ])
            .env("HOME", host)
            .env("SHELL", "/bin/bash")
            .env("PATH", path)
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("no_proxy", "127.0.0.1,localhost")
            .env("FIXTURE_OS", os)
            .env(
                "FIXTURE_ARCH",
                if os == "Darwin" { "arm64" } else { "x86_64" },
            )
            .env_remove("FIXTURE_GPU_CAPABILITIES")
            .env_remove("FIXTURE_MISSING_CUDA");
        if let Some(caps) = caps {
            command.env("FIXTURE_GPU_CAPABILITIES", caps);
        }
        if missing {
            command.env("FIXTURE_MISSING_CUDA", "1");
        }
        command.output().unwrap()
    }
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

struct VersionSession {
    child: Child,
    replies: mpsc::Receiver<String>,
}
impl VersionSession {
    fn start(binary: &Path) -> Self {
        let mut child = Command::new(binary)
            .arg("--version-session")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (send, replies) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if send.send(line.unwrap()).is_err() {
                    break;
                }
            }
        });
        Self { child, replies }
    }
    fn version(&mut self) -> String {
        assert!(self.child.try_wait().unwrap().is_none());
        self.child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(b"version\n")
            .unwrap();
        self.replies.recv_timeout(Duration::from_secs(5)).unwrap()
    }
    fn finish(mut self) {
        self.child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(b"quit\n")
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "fixture did not exit normally: {status}");
                return;
            }
            assert!(Instant::now() < deadline, "fixture did not finish normally");
            thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for VersionSession {
    fn drop(&mut self) {
        // A failed assertion still owns and cleans up only this test's child.
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

#[test]
fn actual_download_install_repeat_and_upgrade_preserve_profiles_and_models() {
    let mut first = Fixture::new("1.2.3");
    first.asset("1.2.3", "macos-aarch64", false, false);
    let mut next = Fixture::new("1.2.4");
    next.asset("1.2.4", "macos-aarch64", false, false);
    let mut assets = first.assets.clone();
    assets.extend(next.assets.clone());
    let server = Server::new(assets);
    let host = first.host();
    fs::write(host.path().join(".bashrc"), b"# existing user settings\n").unwrap();
    fs::write(
        host.path().join(".bash_login"),
        b"# existing login settings\n",
    )
    .unwrap();
    fs::create_dir(host.path().join("models")).unwrap();
    fs::write(host.path().join("models/weights"), b"keep model bytes").unwrap();
    let output = first.run(host.path(), &server, "1.2.3", "Darwin", "auto", None, false);
    success(&output);
    // Feedback must remain visible when the installer is captured by a pipe or
    // CI job, and identify each payload/checksum rather than only the release.
    let log = String::from_utf8_lossy(&output.stderr);
    for suffix in ["", ".sha256", ".binary.sha256"] {
        let name = format!("ferrum-macos-aarch64.tar.gz{suffix}");
        assert!(log.contains(&format!("Downloading {name}...")), "{log}");
        assert!(log.contains(&format!("Downloaded {name}.")), "{log}");
    }
    let first_link = fs::read_link(host.path().join(".local/bin/ferrum")).unwrap();
    let mut session = VersionSession::start(&host.path().join(".local/bin/ferrum"));
    let original_pid = session.child.id();
    assert_eq!(session.version(), "ferrum 1.2.3");
    success(&first.run(host.path(), &server, "1.2.3", "Darwin", "auto", None, false));
    let profile = fs::read_to_string(host.path().join(".bashrc")).unwrap();
    assert!(profile.starts_with("# existing user settings\n"));
    let login = fs::read_to_string(host.path().join(".bash_login")).unwrap();
    assert!(login.starts_with("# existing login settings\n"));
    assert_eq!(login.matches("# >>> Ferrum installer PATH >>>").count(), 1);
    assert!(!host.path().join(".profile").exists());
    assert_eq!(
        profile.matches("# >>> Ferrum installer PATH >>>").count(),
        1
    );
    success(&first.run(
        host.path(),
        &server,
        "1.2.4",
        "Darwin",
        "metal",
        None,
        false,
    ));
    assert_ne!(
        fs::read_link(host.path().join(".local/bin/ferrum")).unwrap(),
        first_link
    );
    assert!(first_link.is_file());
    assert_eq!(session.child.id(), original_pid);
    assert_eq!(session.version(), "ferrum 1.2.3");
    let new_entry = Command::new(host.path().join(".local/bin/ferrum"))
        .arg("--version")
        .output()
        .unwrap();
    success(&new_entry);
    assert_eq!(
        String::from_utf8(new_entry.stdout).unwrap().trim(),
        "ferrum 1.2.4"
    );
    session.finish();
    assert_eq!(
        fs::read(host.path().join("models/weights")).unwrap(),
        b"keep model bytes"
    );
}

#[test]
fn readonly_profile_failure_keeps_the_previous_version_selected() {
    let mut first = Fixture::new("1.2.3");
    first.asset("1.2.3", "macos-aarch64", false, false);
    let mut next = Fixture::new("1.2.4");
    next.asset("1.2.4", "macos-aarch64", false, false);
    let mut assets = first.assets.clone();
    assets.extend(next.assets.clone());
    let server = Server::new(assets);
    let host = first.host();
    success(&first.run(host.path(), &server, "1.2.3", "Darwin", "auto", None, false));
    let binary = host.path().join(".local/bin/ferrum");
    let old_link = fs::read_link(&binary).unwrap();
    let old_bytes = fs::read(&binary).unwrap();
    // A user can create a login profile after the first installation. It is a
    // regular readable file, but the product must handle actual append failure.
    let profile = host.path().join(".bash_login");
    let contents = b"# existing protected login settings\n";
    fs::write(&profile, contents).unwrap();
    fs::set_permissions(&profile, fs::Permissions::from_mode(0o400)).unwrap();
    assert!(
        fs::OpenOptions::new().append(true).open(&profile).is_err(),
        "this permissions regression requires an unprivileged process; append to mode 0400 unexpectedly succeeded"
    );
    let failed = first.run(host.path(), &server, "1.2.4", "Darwin", "auto", None, false);
    assert!(!failed.status.success());
    assert_eq!(fs::read_link(&binary).unwrap(), old_link);
    assert_eq!(fs::read(&binary).unwrap(), old_bytes);
    assert_eq!(fs::read(&profile).unwrap(), contents);
    let selected = Command::new(&binary).arg("--version").output().unwrap();
    success(&selected);
    assert_eq!(
        String::from_utf8(selected.stdout).unwrap().trim(),
        "ferrum 1.2.3"
    );
}

#[test]
fn cuda_auto_uses_actual_capability_and_loader_failure_falls_back_explicit_cuda_fails() {
    let mut fixture = Fixture::new("1.2.3");
    fixture.asset("1.2.3", "linux-x86_64", false, false);
    fixture.asset("1.2.3", "linux-x86_64", true, false);
    let server = Server::new(fixture.assets.clone());
    for (caps, missing, expected) in [
        (None, false, "-cpu-"),
        (Some("8.9"), false, "-cuda-"),
        (Some("8.6"), false, "-cpu-"),
        (Some("8.9"), true, "-cpu-"),
    ] {
        let host = fixture.host();
        let result = fixture.run(
            host.path(),
            &server,
            "1.2.3",
            "Linux",
            "auto",
            caps,
            missing,
        );
        success(&result);
        assert!(fs::read_link(host.path().join(".local/bin/ferrum"))
            .unwrap()
            .to_string_lossy()
            .contains(expected));
        if missing {
            assert!(String::from_utf8_lossy(&result.stderr).contains("selecting CPU"));
        }
    }
    let host = fixture.host();
    let result = fixture.run(
        host.path(),
        &server,
        "1.2.3",
        "Linux",
        "cuda",
        Some("8.9"),
        true,
    );
    assert!(!result.status.success());
    assert!(!host.path().join(".local/bin/ferrum").exists());
}

#[test]
fn corrupt_hash_and_unexpected_archive_member_cannot_install() {
    for extra in [false, true] {
        let mut fixture = Fixture::new("1.2.3");
        fixture.asset("1.2.3", "macos-aarch64", false, extra);
        if !extra {
            fixture
                .assets
                .get_mut("/v1.2.3/ferrum-macos-aarch64.tar.gz")
                .unwrap()
                .push(1);
        }
        let server = Server::new(fixture.assets.clone());
        let host = fixture.host();
        let result = fixture.run(host.path(), &server, "1.2.3", "Darwin", "auto", None, false);
        assert!(!result.status.success());
        assert!(!host.path().join(".local/bin/ferrum").exists());
        assert!(String::from_utf8_lossy(&result.stderr).contains(if extra {
            "archive member"
        } else {
            "SHA-256 mismatch"
        }));
    }
}

#[test]
fn unmanaged_binary_and_edited_managed_binary_are_preserved() {
    let mut fixture = Fixture::new("1.2.3");
    fixture.asset("1.2.3", "macos-aarch64", false, false);
    let server = Server::new(fixture.assets.clone());
    let host = fixture.host();
    fs::create_dir_all(host.path().join(".local/bin")).unwrap();
    fs::write(host.path().join(".local/bin/ferrum"), b"other installer").unwrap();
    assert!(!fixture
        .run(host.path(), &server, "1.2.3", "Darwin", "auto", None, false)
        .status
        .success());
    assert_eq!(
        fs::read(host.path().join(".local/bin/ferrum")).unwrap(),
        b"other installer"
    );
    fs::remove_file(host.path().join(".local/bin/ferrum")).unwrap();
    success(&fixture.run(host.path(), &server, "1.2.3", "Darwin", "auto", None, false));
    fs::write(host.path().join(".local/bin/ferrum"), b"user edited").unwrap();
    assert!(!fixture
        .run(host.path(), &server, "1.2.3", "Darwin", "auto", None, false)
        .status
        .success());
    assert_eq!(
        fs::read(host.path().join(".local/bin/ferrum")).unwrap(),
        b"user edited"
    );
}
