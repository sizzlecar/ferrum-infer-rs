//! Compile a protected external contract, then execute its Rust assertions.
use super::process;
use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
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
    /// Optional machine-readable result for validation repair consumers.
    #[arg(long)]
    result_json: Option<PathBuf>,
}

pub(crate) async fn run(args: &Args) -> Result<i32> {
    let mut cancellation = args
        .result_json
        .as_ref()
        .map(|_| process::Cancellation::controller_termination())
        .transpose()?;
    let result = run_inner(args, cancellation.as_mut()).await;
    if let Some(path) = &args.result_json {
        let evidence = match &result {
            Ok(evidence) => evidence.clone(),
            Err(error) => Evidence::new(Class::InfrastructureFailure, vec![format!("{error:#}")]),
        };
        super::write_json(path, &evidence)?;
    }
    result.map(|evidence| evidence.exit_code())
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Class {
    Passed,
    SemanticFailure,
    CandidateCompilationFailure,
    InfrastructureFailure,
    TimedOut,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Evidence {
    pub schema_version: u32,
    pub class: Class,
    pub diagnostics: Vec<String>,
}

impl Evidence {
    fn new(class: Class, diagnostics: Vec<String>) -> Self {
        Self {
            schema_version: 1,
            class,
            diagnostics,
        }
    }

    pub(crate) fn exit_code(&self) -> i32 {
        match self.class {
            Class::Passed => 0,
            Class::SemanticFailure => 1,
            _ => 2,
        }
    }

    pub(crate) fn repairable(&self) -> bool {
        matches!(
            self.class,
            Class::SemanticFailure | Class::CandidateCompilationFailure
        )
    }
}

async fn run_inner(
    args: &Args,
    mut cancellation: Option<&mut process::Cancellation>,
) -> Result<Evidence> {
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
    let candidate_path = candidate.clone();
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
    let build = logged_stage(
        std::path::Path::new("cargo"),
        &build_args,
        root,
        &root.join("agent"),
        &root.join("build"),
        args.timeout_secs,
        cancellation.as_deref_mut(),
    )
    .await?;
    let stdout = fs::read_to_string(root.join("build/stdout.txt"))?;
    std::io::stderr().write_all(&fs::read(root.join("build/stderr.txt"))?)?;
    if build.timed_out || build.exit_code != Some(0) {
        print!("{stdout}");
        eprintln!("contract compilation did not succeed: {build:?}");
        let diagnostics = compiler_diagnostics(&stdout, &candidate_path);
        return Ok(Evidence::new(
            if build.timed_out {
                Class::TimedOut
            } else if build.exit_code == Some(101) && !diagnostics.is_empty() {
                Class::CandidateCompilationFailure
            } else {
                Class::InfrastructureFailure
            },
            diagnostics,
        ));
    }
    let executables = test_executables(&stdout)?;
    ensure!(
        executables.len() == 1,
        "expected the requested acceptance test executable"
    );
    let executable = &executables[0];
    let listing = logged_stage(
        executable,
        &["--list".into(), "--format".into(), "terse".into()],
        root,
        &root.join("agent"),
        &root.join("list"),
        args.timeout_secs,
        cancellation.as_deref_mut(),
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
    let result = logged_stage(
        executable,
        &["--test-threads=1".into()],
        root,
        &root.join("agent"),
        &root.join("execute"),
        args.timeout_secs,
        cancellation.as_deref_mut(),
    )
    .await?;
    std::io::stdout().write_all(&fs::read(root.join("execute/stdout.txt"))?)?;
    std::io::stderr().write_all(&fs::read(root.join("execute/stderr.txt"))?)?;
    if result.timed_out {
        eprintln!("contract execution timed out");
        return Ok(Evidence::new(
            Class::TimedOut,
            vec!["contract execution timed out".into()],
        ));
    }
    Ok(Evidence::new(
        match result.exit_code {
            Some(0) => Class::Passed,
            Some(101) => Class::SemanticFailure,
            _ => Class::InfrastructureFailure,
        },
        Vec::new(),
    ))
}

async fn logged_stage(
    program: &std::path::Path,
    args: &[String],
    cwd: &std::path::Path,
    agent_dir: &std::path::Path,
    output: &std::path::Path,
    timeout_secs: u64,
    cancellation: Option<&mut process::Cancellation>,
) -> Result<process::Outcome> {
    match cancellation {
        Some(cancellation) => {
            process::logged_cancellable(
                program,
                args,
                cwd,
                agent_dir,
                output,
                timeout_secs,
                cancellation,
            )
            .await
        }
        None => process::logged(program, args, cwd, agent_dir, output, timeout_secs).await,
    }
}

/// Compiler JSON is supplied by the protected validator's Cargo invocation.
/// Only attributed candidate Rust errors are repairable. Acceptance-contract
/// errors (even API mismatches), missing crates, linker/toolchain failures and
/// arbitrary stderr remain infrastructure/unknown rather than guessed fixes.
fn compiler_diagnostics(stdout: &str, candidate: &std::path::Path) -> Vec<String> {
    let mut diagnostics = Vec::new();
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            return Vec::new();
        };
        if message["reason"] != "compiler-message" || message["message"]["level"] != "error" {
            continue;
        }
        let Some(source) = message["target"]["src_path"].as_str() else {
            return Vec::new();
        };
        let Some(rendered) = message["message"]["rendered"].as_str() else {
            return Vec::new();
        };
        if !std::path::Path::new(source).starts_with(candidate)
            || !message["message"]["spans"]
                .as_array()
                .is_some_and(|spans| !spans.is_empty())
        {
            return Vec::new();
        }
        diagnostics.push(rendered.to_owned());
    }
    diagnostics
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
    fn compilation_failure_requires_attributed_rust_diagnostics() {
        let rust = serde_json::json!({"reason":"compiler-message", "target":{"src_path":"/candidate/src/lib.rs"},
            "message":{"level":"error","spans":[{"file_name":"src/lib.rs"}],"rendered":"type mismatch"}});
        let candidate = std::path::Path::new("/candidate");
        assert_eq!(
            super::compiler_diagnostics(&rust.to_string(), candidate),
            ["type mismatch"]
        );
        assert!(super::compiler_diagnostics("error: linker failed", candidate).is_empty());
        let mut unrelated = rust.clone();
        unrelated["target"]["src_path"] = "/registry/dependency/lib.rs".into();
        assert!(super::compiler_diagnostics(&unrelated.to_string(), candidate).is_empty());
        unrelated["target"]["src_path"] = "/protected/acceptance.rs".into();
        assert!(super::compiler_diagnostics(&format!("{rust}\n{unrelated}"), candidate).is_empty());
        assert!(!super::Evidence::new(super::Class::TimedOut, vec![]).repairable());
        assert!(!super::Evidence::new(super::Class::InfrastructureFailure, vec![]).repairable());
    }
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
