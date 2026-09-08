//! Actual launcher/child process contracts; the child is this Rust test binary.
#![cfg(windows)]

use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    os::windows::{
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU32, Ordering},
    thread,
    time::{Duration, Instant},
};
use windows_sys::Win32::{
    Foundation::WAIT_OBJECT_0,
    System::{
        Console::{
            GenerateConsoleCtrlEvent, SetConsoleCtrlHandler, CTRL_BREAK_EVENT, CTRL_C_EVENT,
        },
        Threading::{OpenProcess, WaitForSingleObject, CREATE_NEW_CONSOLE, PROCESS_SYNCHRONIZE},
    },
};

const OLD: &str = "1.2.3-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const NEW: &str = "1.2.4-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const MODE: &str = "FERRUM_LAUNCHER_TEST_MODE";
const RECEIPT: &str = "FERRUM_LAUNCHER_TEST_RECEIPT";
const RELEASE: &str = "FERRUM_LAUNCHER_TEST_RELEASE";

struct Running(Child);
impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait(child: &mut Child) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "child {} did not exit",
            child.id()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn read_receipt(path: &Path) -> Value {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Ok(bytes) = fs::read(path) {
            if let Ok(value) = serde_json::from_slice(&bytes) {
                return value;
            }
        }
        assert!(
            Instant::now() < deadline,
            "missing child receipt {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

struct Layout {
    root: tempfile::TempDir,
    launcher: PathBuf,
}
impl Layout {
    fn new() -> Self {
        let root = tempfile::Builder::new()
            .prefix("Ferrum 中文 launcher ")
            .tempdir()
            .unwrap();
        let launcher = root.path().join("ferrum.exe");
        fs::copy(env!("CARGO_BIN_EXE_ferrum-launcher"), &launcher).unwrap();
        for (id, text) in [(OLD, "old"), (NEW, "new")] {
            let directory = root.path().join("versions").join(id);
            fs::create_dir_all(&directory).unwrap();
            fs::copy(
                std::env::current_exe().unwrap(),
                directory.join("ferrum.exe"),
            )
            .unwrap();
            fs::write(directory.join("version.txt"), text).unwrap();
        }
        let layout = Self { root, launcher };
        layout.select(OLD);
        layout
    }
    fn select(&self, id: &str) {
        let temporary = self.root.path().join("next.json");
        fs::write(
            &temporary,
            json!({"schema_version":1,"version_dir":id}).to_string(),
        )
        .unwrap();
        fs::rename(temporary, self.root.path().join("current.json")).unwrap();
    }
    fn command(&self, mode: &str, receipt: &Path) -> Command {
        let mut command = fixture_command(&self.launcher, mode, receipt);
        command
            .current_dir(self.root.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command
    }
}

fn fixture_command(program: &Path, mode: &str, receipt: &Path) -> Command {
    let mut command = Command::new(program);
    command
        .args(["--exact", "launcher_fixture", "--ignored", "--nocapture"])
        .env(MODE, mode)
        .env(RECEIPT, receipt);
    command
}

#[test]
fn launcher_preserves_argv_cwd_stdio_environment_and_exit_code() {
    let layout = Layout::new();
    let receipt = layout.root.path().join("echo.json");
    let input = layout.root.path().join("input.bin");
    let bytes = b"prompt\0\xff\nsecond line";
    fs::write(&input, bytes).unwrap();
    let arguments = [
        "",
        "中文 with spaces",
        "{\"name\":\"a \\\" b\"}",
        "trailing\\",
        "$(literal); & more",
        "--looks-like-an-option",
    ];
    for exit in [37i32, 0xc000013au32 as i32] {
        let result = layout
            .command("echo", &receipt)
            .arg("--")
            .args(arguments)
            .env("FERRUM_LAUNCHER_TEST_EXIT", exit.to_string())
            .env("FERRUM_LAUNCHER_TEST_UNCHANGED", "keep 中文 value")
            .stdin(fs::File::open(&input).unwrap())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(exit));
        assert!(result.stdout.ends_with(bytes));
        assert_eq!(result.stderr, b"fixture stderr\n");
        let observed = read_receipt(&receipt);
        let actual: Vec<String> = serde_json::from_value(observed["arguments"].clone()).unwrap();
        assert_eq!(&actual[actual.len() - arguments.len()..], arguments);
        assert_eq!(
            observed["cwd"],
            layout.root.path().canonicalize().unwrap().to_str().unwrap()
        );
        assert_eq!(observed["environment"], "keep 中文 value");
        assert_eq!(observed["version"], "old");
    }
}

#[test]
fn version_switch_keeps_old_session_and_new_launch_selects_new_payload() {
    let layout = Layout::new();
    let old_receipt = layout.root.path().join("old.json");
    let release = layout.root.path().join("release-old");
    let mut old = Running(
        layout
            .command("hold", &old_receipt)
            .env(RELEASE, &release)
            .spawn()
            .unwrap(),
    );
    let before = read_receipt(&old_receipt);
    assert_eq!(before["version"], "old");
    layout.select(NEW);
    let new_receipt = layout.root.path().join("new.json");
    let mut new = Running(layout.command("version", &new_receipt).spawn().unwrap());
    assert!(wait(&mut new.0).success());
    assert_eq!(read_receipt(&new_receipt)["version"], "new");
    assert!(old.0.try_wait().unwrap().is_none());
    fs::write(release, b"finish own old session").unwrap();
    assert!(wait(&mut old.0).success());
    let after = read_receipt(&old_receipt);
    assert_eq!(after["version"], "old");
    assert_eq!(after["pid"], before["pid"]);
    assert_eq!(after["phase"], "finished");
}

fn process_handle(pid: u32) -> OwnedHandle {
    let handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) };
    assert!(
        !handle.is_null(),
        "OpenProcess: {}",
        std::io::Error::last_os_error()
    );
    unsafe { OwnedHandle::from_raw_handle(handle) }
}

#[test]
fn terminating_launcher_closes_job_and_reaps_child_and_grandchild() {
    let layout = Layout::new();
    let receipt = layout.root.path().join("tree.json");
    let mut launcher = Running(layout.command("tree", &receipt).spawn().unwrap());
    let core = read_receipt(&receipt);
    let grandchild = read_receipt(&receipt.with_extension("grandchild.json"));
    let handles = [
        process_handle(core["pid"].as_u64().unwrap() as u32),
        process_handle(grandchild["pid"].as_u64().unwrap() as u32),
    ];
    launcher.0.kill().unwrap();
    let _ = wait(&mut launcher.0);
    for handle in handles {
        assert_eq!(
            unsafe { WaitForSingleObject(handle.as_raw_handle(), 5000) },
            WAIT_OBJECT_0,
            "launcher left a descendant alive"
        );
    }
}

static CONTROL: AtomicU32 = AtomicU32::new(0);
unsafe extern "system" fn capture_control(event: u32) -> i32 {
    if event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT {
        CONTROL.store(event + 1, Ordering::SeqCst);
        1
    } else {
        0
    }
}

#[test]
fn interactive_ctrl_c_reaches_child_and_launcher_returns_its_exit() {
    let layout = Layout::new();
    let receipt = layout.root.path().join("console.json");
    let mut controller = Running(
        fixture_command(
            &std::env::current_exe().unwrap(),
            "console-controller",
            &receipt,
        )
        .env("FERRUM_LAUNCHER_TEST_PROGRAM", &layout.launcher)
        .creation_flags(CREATE_NEW_CONSOLE)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap(),
    );
    assert!(wait(&mut controller.0).success());
    assert_eq!(read_receipt(&receipt)["child_exit"], 41);
}

#[test]
#[ignore = "Rust payload/controller fixture invoked by the launcher integration tests"]
fn launcher_fixture() {
    let mode = std::env::var(MODE).expect("parent-owned fixture mode");
    let receipt = PathBuf::from(std::env::var_os(RECEIPT).expect("parent-owned receipt"));
    if mode == "console-controller" {
        assert_ne!(unsafe { SetConsoleCtrlHandler(None, 0) }, 0);
        assert_ne!(
            unsafe { SetConsoleCtrlHandler(Some(capture_control), 1) },
            0
        );
        let child_receipt = receipt.with_extension("child.json");
        let program = PathBuf::from(std::env::var_os("FERRUM_LAUNCHER_TEST_PROGRAM").unwrap());
        let mut child = Running(
            fixture_command(&program, "ctrl-c", &child_receipt)
                .spawn()
                .unwrap(),
        );
        read_receipt(&child_receipt);
        assert_ne!(unsafe { GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0) }, 0);
        let status = wait(&mut child.0);
        assert_eq!(status.code(), Some(41));
        fs::write(receipt, json!({"child_exit":status.code()}).to_string()).unwrap();
        std::process::exit(0);
    }
    let program = std::env::current_exe().unwrap();
    let version = fs::read_to_string(program.parent().unwrap().join("version.txt")).unwrap();
    if mode == "ctrl-c" {
        assert_ne!(
            unsafe { SetConsoleCtrlHandler(Some(capture_control), 1) },
            0
        );
    }
    let mut record = json!({"pid":std::process::id(), "version":version, "phase":"started",
        "arguments":std::env::args().skip(1).collect::<Vec<_>>(),
        "cwd":std::env::current_dir().unwrap().canonicalize().unwrap().to_str().unwrap(),
        "environment":std::env::var("FERRUM_LAUNCHER_TEST_UNCHANGED").ok()});
    fs::write(&receipt, record.to_string()).unwrap();
    if mode == "echo" {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).unwrap();
        std::io::stdout().write_all(&bytes).unwrap();
        std::io::stdout().flush().unwrap();
        std::io::stderr().write_all(b"fixture stderr\n").unwrap();
        std::io::stderr().flush().unwrap();
        std::process::exit(
            std::env::var("FERRUM_LAUNCHER_TEST_EXIT")
                .unwrap()
                .parse()
                .unwrap(),
        );
    }
    if mode == "tree" {
        let _grandchild = fixture_command(
            &program,
            "grandchild",
            &receipt.with_extension("grandchild.json"),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    }
    if mode == "version" {
        std::process::exit(0);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if mode == "ctrl-c" && CONTROL.load(Ordering::SeqCst) == CTRL_C_EVENT + 1 {
            std::process::exit(41);
        }
        if mode == "hold" && Path::new(&std::env::var_os(RELEASE).unwrap()).exists() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "fixture was not released or terminated"
        );
        thread::sleep(Duration::from_millis(20));
    }
    record["phase"] = json!("finished");
    fs::write(receipt, record.to_string()).unwrap();
    std::process::exit(0);
}
