//! Compile a protected external contract, then execute its Rust assertions.
use super::process;
use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde_json::Value;
use std::{fs, io::Write, path::PathBuf};

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long)]
    contract_path: PathBuf,
    #[arg(long)]
    workdir: PathBuf,
    #[arg(long)]
    target_dir: PathBuf,
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,
}

pub(crate) async fn run(args: &Args) -> Result<i32> {
    ensure!(args.timeout_secs > 0, "validation timeout must be positive");
    let contract = args.contract_path.canonicalize()?;
    let candidate = args.workdir.canonicalize()?;
    ensure!(
        super::config::disjoint(&candidate, &contract),
        "contract must be outside candidate"
    );
    fs::create_dir_all(&args.target_dir)?;
    let target = args.target_dir.canonicalize()?;
    let scratch = tempfile::Builder::new()
        .prefix("ferrum-agent-contract-")
        .tempdir()?;
    let root = scratch.path();
    fs::create_dir(root.join("tests"))?;
    fs::create_dir(root.join("agent"))?;
    fs::copy(&contract, root.join("tests/acceptance.rs"))?;
    let candidate = serde_json::to_string(candidate.to_str().context("non-UTF8 candidate path")?)?;
    fs::write(root.join("Cargo.toml"), format!(
        "[package]\nname = \"ferrum-agent-contract\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[workspace]\n\n[dependencies]\ncandidate = {{ package = \"ferrum-task-candidate\", path = {candidate} }}\nserde_json = \"1\"\nregex-lite = \"0.1\"\n"
    ))?;
    let build_args = vec![
        "test".into(),
        "--no-run".into(),
        "--offline".into(),
        "--message-format=json".into(),
        "--test".into(),
        "acceptance".into(),
        "--jobs".into(),
        "1".into(),
        "--target-dir".into(),
        target.to_string_lossy().into_owned(),
    ];
    let build = process::logged(
        std::path::Path::new("cargo"),
        &build_args,
        root,
        &root.join("agent"),
        &root.join("build"),
        args.timeout_secs,
    )
    .await?;
    let stdout = fs::read_to_string(root.join("build/stdout.txt"))?;
    std::io::stderr().write_all(&fs::read(root.join("build/stderr.txt"))?)?;
    if build.timed_out || build.exit_code != Some(0) {
        print!("{stdout}");
        eprintln!("contract compilation did not succeed: {build:?}");
        return Ok(2);
    }
    let executables = test_executables(&stdout)?;
    ensure!(
        executables.len() == 1,
        "expected the requested acceptance test executable"
    );
    let executable = &executables[0];
    let listing = process::logged(
        executable,
        &["--list".into(), "--format".into(), "terse".into()],
        root,
        &root.join("agent"),
        &root.join("list"),
        args.timeout_secs,
    )
    .await?;
    ensure!(
        listing.exit_code == Some(0) && !listing.timed_out,
        "could not list contract tests"
    );
    ensure!(
        fs::read_to_string(root.join("list/stdout.txt"))?
            .lines()
            .any(|line| line.ends_with(": test")),
        "contract has no executable assertions"
    );
    let result = process::logged(
        executable,
        &["--test-threads=1".into()],
        root,
        &root.join("agent"),
        &root.join("execute"),
        args.timeout_secs,
    )
    .await?;
    std::io::stdout().write_all(&fs::read(root.join("execute/stdout.txt"))?)?;
    std::io::stderr().write_all(&fs::read(root.join("execute/stderr.txt"))?)?;
    if result.timed_out {
        eprintln!("contract execution timed out");
        return Ok(2);
    }
    Ok(match result.exit_code {
        Some(0) => 0,
        Some(101) => 1,
        _ => 2,
    })
}

fn test_executables(stdout: &str) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let message: Value = serde_json::from_str(line).context("invalid cargo JSON output")?;
        if message["reason"] == "compiler-artifact" && message["profile"]["test"] == true {
            if let Some(path) = message["executable"].as_str() {
                paths.push(PathBuf::from(path));
            }
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    #[test]
    fn uses_only_cargo_reported_test_artifacts() {
        let data = concat!(
            "{\"reason\":\"compiler-artifact\",\"profile\":{\"test\":false},\"executable\":\"/wrong\"}\n",
            "{\"reason\":\"compiler-artifact\",\"profile\":{\"test\":true},\"executable\":\"/right\"}\n",
            "{\"reason\":\"build-finished\",\"success\":true}\n");
        assert_eq!(
            super::test_executables(data).unwrap(),
            vec![std::path::PathBuf::from("/right")]
        );
    }
}
