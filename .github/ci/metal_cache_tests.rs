use super::*;
use std::{
    io::Read,
    os::{
        fd::FromRawFd,
        unix::{
            fs::symlink,
            net::{UnixListener, UnixStream},
        },
    },
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);

#[test]
fn protected_residual_suppression_ends_at_growth_or_changed_limit() {
    let previous = Some((48 * GIB, 60 * GIB));
    assert!(!size_cleanup_needed(61 * GIB, 48 * GIB, previous));
    assert!(!size_cleanup_needed(72 * GIB - 1, 48 * GIB, previous));
    assert!(size_cleanup_needed(72 * GIB, 48 * GIB, previous));
    assert!(size_cleanup_needed(61 * GIB, 47 * GIB, previous));
    assert!(!size_cleanup_needed(48 * GIB, 48 * GIB, previous));
    assert!(size_cleanup_needed(49 * GIB, 48 * GIB, None));
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = env::temp_dir().join(format!(
            "fcm-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn inherited_lease_child() {
    match env::var("FERRUM_CACHE_TEST_ROLE").as_deref() {
        Ok("supervisor") => {
            let file = lease(Path::new(&env::var_os("FERRUM_CACHE_TEST_TARGET").unwrap())).unwrap();
            let status = Command::new(env::current_exe().unwrap())
                .args(["--exact", "tests::inherited_lease_child", "--nocapture"])
                .env("FERRUM_CACHE_TEST_ROLE", "worker")
                .env("FERRUM_CACHE_TEST_FD", file.as_raw_fd().to_string())
                .status()
                .unwrap();
            assert!(status.success());
        }
        Ok("worker") => {
            // Take ownership of the actual inherited lease, rather than opening
            // a second file and accidentally testing independent lock behavior.
            let fd = env::var("FERRUM_CACHE_TEST_FD").unwrap().parse().unwrap();
            let file = unsafe { File::from_raw_fd(fd) };
            let mut socket =
                UnixStream::connect(env::var_os("FERRUM_CACHE_TEST_SOCKET").unwrap()).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(20)))
                .unwrap();
            socket.write_all(b"ready").unwrap();
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            drop(file);
            socket.write_all(b"released").unwrap();
        }
        _ => {}
    }
}

#[test]
fn killing_supervisor_keeps_lease_until_running_worker_finishes() {
    let fixture = Fixture::new();
    let target = fixture.path("target");
    fs::create_dir(&target).unwrap();
    let socket_path = fixture.path("events");
    let listener = UnixListener::bind(&socket_path).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut supervisor = Command::new(env::current_exe().unwrap())
        .args(["--exact", "tests::inherited_lease_child", "--nocapture"])
        .env("FERRUM_CACHE_TEST_ROLE", "supervisor")
        .env("FERRUM_CACHE_TEST_TARGET", &target)
        .env("FERRUM_CACHE_TEST_SOCKET", &socket_path)
        .spawn()
        .unwrap();
    let started = Instant::now();
    let (mut socket, _) = loop {
        match listener.accept() {
            Ok(connection) => break connection,
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    && started.elapsed() < Duration::from_secs(20) =>
            {
                if let Some(status) = supervisor.try_wait().unwrap() {
                    panic!("supervisor exited before worker readiness: {status}");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => {
                let _ = supervisor.kill();
                let _ = supervisor.wait();
                panic!("worker did not signal readiness: {error}");
            }
        }
    };
    socket.set_nonblocking(false).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(20)))
        .unwrap();
    let mut ready = [0; 5];
    socket.read_exact(&mut ready).unwrap();
    assert_eq!(&ready, b"ready");
    let contender = OpenOptions::new()
        .read(true)
        .write(true)
        .open(fixture.path(".target.lease"))
        .unwrap();
    assert!(matches!(
        contender.try_lock(),
        Err(std::fs::TryLockError::WouldBlock)
    ));
    supervisor.kill().unwrap();
    assert!(!supervisor.wait().unwrap().success());
    assert!(
        matches!(contender.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
        "live worker must retain the lease after cancellation"
    );
    socket.write_all(b"x").unwrap();
    let mut released = [0; 8];
    socket.read_exact(&mut released).unwrap();
    assert_eq!(&released, b"released");
    contender.try_lock().unwrap();
}

#[test]
fn rejects_overlapping_roots_and_symlinks_without_removing_anything() {
    let fixture = Fixture::new();
    let workspace = fixture.path("workspace");
    fs::create_dir(&workspace).unwrap();
    fs::write(workspace.join("keep"), b"user data").unwrap();
    assert!(roots(&workspace, &workspace).is_err());
    assert!(roots(&workspace, &workspace.join("target")).is_err());
    let target = fixture.path("target");
    symlink(&workspace, &target).unwrap();
    assert!(roots(&workspace, &target).is_err());
    assert_eq!(fs::read(workspace.join("keep")).unwrap(), b"user data");
    let real = fixture.path("real-target");
    fs::create_dir(&real).unwrap();
    symlink(workspace.join("keep"), fixture.path(".real-target.lease")).unwrap();
    assert!(lease(&real).is_err());
    assert_eq!(fs::read(workspace.join("keep")).unwrap(), b"user data");
}

#[test]
fn command_failure_is_preserved_and_recorded() {
    let fixture = Fixture::new();
    let workspace = fixture.path("workspace");
    fs::create_dir(&workspace).unwrap();
    let report = fixture.path("report.log");
    assert_eq!(
        run(Options {
            workspace,
            target: fixture.path("target"),
            report: report.clone(),
            max_bytes: u64::MAX,
            min_free: 0,
            command: vec!["/usr/bin/false".into()],
        })
        .unwrap(),
        1
    );
    let log = fs::read_to_string(report).unwrap();
    assert!(log.contains("cleanup=not_needed"));
    assert!(log.contains("phase=completed command_exit=1"));
}

fn crate_fixture(root: &Path, name: &str) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        format!(
            "[package]\nname = {name:?}\nversion = \"0.0.0\"\nedition = \"2021\"\n[workspace]\n"
        ),
    )
    .unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
}

fn build_fixture(workspace: &Path, target: &Path, release: bool) {
    let mut command = Command::new("cargo");
    command
        .args(["build", "--offline", "--manifest-path"])
        .arg(workspace.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target)
        .env("CARGO_INCREMENTAL", "0")
        .env("CARGO_PROFILE_DEV_DEBUG", "0");
    if release {
        command.arg("--release");
    }
    assert!(command.status().unwrap().success());
}

#[test]
fn exact_dev_cleanup_preserves_release_other_project_and_nested_targets() {
    let fixture = Fixture::new();
    let workspace = fixture.path("ferrum");
    let other = fixture.path("orch");
    let target = fixture.path("target");
    crate_fixture(&workspace, "ferrum-cache-fixture");
    crate_fixture(&other, "orchestral-cache-fixture");
    build_fixture(&workspace, &target, false);
    build_fixture(&workspace, &target, true);
    build_fixture(&other, &target, false);
    let keep = [
        "wasm32-unknown-unknown/debug/ferrum-cache-fixture",
        "wasm32-unknown-unknown/debug/deps/libferrum_cache_fixture-test.rlib",
        "tests/trybuild/debug/ferrum-cache-fixture",
        "tests/trybuild/debug/deps/libferrum_cache_fixture-test.rlib",
        "dx/orchestral-web/keep",
    ];
    for path in keep {
        let path = target.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"retained").unwrap();
    }
    assert_eq!(package_names(&workspace).unwrap(), ["ferrum-cache-fixture"]);
    let _lease = lease(&target).unwrap();
    let mut report = File::create(fixture.path("clean.log")).unwrap();
    assert!(clean(&workspace, &target, &[], &mut report).is_err());
    assert!(target.join("debug/ferrum-cache-fixture").exists());
    let options = Options {
        workspace: workspace.clone(),
        target: target.clone(),
        report: fixture.path("unused"),
        max_bytes: 0,
        min_free: 0,
        command: vec![],
    };
    maintain(&options, &workspace, &target, &mut report).unwrap();
    assert!(!target.join("debug/ferrum-cache-fixture").exists());
    assert!(target.join("release/ferrum-cache-fixture").exists());
    assert!(target.join("debug/orchestral-cache-fixture").exists());
    for path in keep {
        assert_eq!(fs::read(target.join(path)).unwrap(), b"retained");
    }
    // A residual above the size goal must not make every job rebuild Ferrum.
    build_fixture(&workspace, &target, false);
    maintain(&options, &workspace, &target, &mut report).unwrap();
    assert!(target.join("debug/ferrum-cache-fixture").exists());
    // Insufficient disk never broadens selection to protected artifacts.
    let options = Options {
        min_free: u64::MAX,
        ..options
    };
    assert!(maintain(&options, &workspace, &target, &mut report)
        .unwrap_err()
        .contains("below reserve"));
    assert!(target.join("release/ferrum-cache-fixture").exists());
    assert!(target.join("debug/orchestral-cache-fixture").exists());
    assert!(!target.join("debug/ferrum-cache-fixture").exists());
    let options = Options {
        max_bytes: u64::MAX,
        min_free: 0,
        ..options
    };
    maintain(&options, &workspace, &target, &mut report).unwrap();
    assert!(!fixture.path(".target.capacity-limit").exists());
}
