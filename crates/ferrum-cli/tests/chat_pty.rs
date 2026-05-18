//! PTY-driven smoke tests for `ferrum run` — covers exit paths and
//! signal handling that stdin-pipe tests can't exercise:
//!
//! - `/bye` slash command exits cleanly
//! - Ctrl+D (EOF on TTY) exits cleanly
//! - Ctrl+C (SIGINT) terminates the process
//!
//! Loads a real model, `#[ignore]` by default. Opt in:
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --features metal --test chat_pty \
//!       -- --ignored --test-threads=1

use rexpect::session::spawn_command;
use std::path::PathBuf;
use std::process::Command;

const SMOKE_MODEL: &str = "qwen3:0.6b";
const SPAWN_TIMEOUT_MS: u64 = 90_000;

fn ferrum_bin() -> PathBuf {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ferrum") {
        return PathBuf::from(bin);
    }
    let current = std::env::current_exe().expect("test exe path");
    let dir = current
        .parent()
        .and_then(|p| p.parent())
        .expect("target dir");
    let mut bin = dir.join("ferrum");
    if cfg!(windows) {
        bin.set_extension("exe");
    }
    assert!(bin.exists(), "ferrum binary not found at {}", bin.display());
    bin
}

fn spawn_ferrum_run() -> rexpect::session::PtySession {
    let mut cmd = Command::new(ferrum_bin());
    cmd.args(["run", SMOKE_MODEL]);
    cmd.env("NO_COLOR", "1");
    spawn_command(cmd, Some(SPAWN_TIMEOUT_MS)).expect("spawn ferrum on pty")
}

#[test]
#[ignore = "loads real model — run with `cargo test -- --ignored`"]
fn test_pty_bye_command_exits() {
    // /bye should drop us out of the REPL with the "Goodbye!" banner and
    // a clean exit. Catches regressions where the slash-command match
    // gets dropped or the cleanup path hangs.
    let mut p = spawn_ferrum_run();
    p.exp_string("Ready.").expect("ready banner");
    p.send_line("/bye").expect("send /bye");
    p.exp_string("Goodbye").expect("goodbye banner");
    p.exp_eof().expect("clean EOF after /bye");
}

#[test]
#[ignore = "loads real model"]
fn test_pty_ctrl_d_clean_exit() {
    // Ctrl+D (EOF on TTY) hits the `read_line -> Ok(0)` branch, which
    // must `break` out of the loop. If a regression turns it into
    // `continue`, the process hangs on the next read forever.
    let mut p = spawn_ferrum_run();
    p.exp_string("Ready.").expect("ready banner");
    p.send_control('d').expect("send Ctrl+D");
    p.exp_string("Goodbye").expect("goodbye banner after EOF");
    p.exp_eof().expect("clean EOF after Ctrl+D");
}

#[test]
#[ignore = "loads real model"]
fn test_pty_sigint_terminates() {
    // Ctrl+C (SIGINT) must terminate the process; default tokio behavior
    // is fine here, but the test guards against a future regression that
    // installs a signal handler swallowing SIGINT into an infinite loop.
    let mut p = spawn_ferrum_run();
    p.exp_string("Ready.").expect("ready banner");
    p.send_control('c').expect("send Ctrl+C");
    // Process should terminate (any exit status acceptable for SIGINT).
    p.exp_eof()
        .expect("process should die on SIGINT within timeout");
}
