//! G4 LoRA startup serving CLI contract.
//!
//! Full routing behavior is covered in ferrum-server unit tests; this file
//! keeps the required ferrum-cli integration entry point focused on serve CLI
//! flags and pre-load validation.

use std::path::PathBuf;
use std::process::Command;

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

#[test]
fn serve_help_lists_lora_startup_flags() {
    let output = Command::new(ferrum_bin())
        .args(["serve", "--help"])
        .env("NO_COLOR", "1")
        .output()
        .expect("run ferrum serve --help");
    assert!(output.status.success(), "status={:?}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--lora <NAME=PATH>"), "stdout: {stdout}");
    assert!(
        stdout.contains("--lora-model-id-template <TEMPLATE>"),
        "stdout: {stdout}"
    );
}

#[test]
fn serve_rejects_invalid_lora_arg_before_model_load() {
    let output = Command::new(ferrum_bin())
        .args(["serve", "qwen3:0.6b", "--lora", "missing_equals"])
        .env("NO_COLOR", "1")
        .output()
        .expect("run ferrum serve invalid lora");
    assert!(!output.status.success(), "invalid --lora should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid --lora value") || stderr.contains("expected NAME=PATH"),
        "stderr: {stderr}"
    );
}
