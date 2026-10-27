//! Execute release delivery through explicit resource and publication gates.
use clap::{Parser, Subcommand};
use std::{path::PathBuf, process::ExitCode};

#[cfg(unix)]
#[path = "../process_group.rs"]
mod process_group;

#[path = "release_delivery/cloud.rs"]
mod cloud;
#[path = "release_delivery/fetch_verified.rs"]
mod fetch_verified;
#[path = "release_delivery/gate.rs"]
mod gate;
#[path = "release_delivery/installation.rs"]
mod installation;
#[path = "release_delivery/local.rs"]
mod local;
#[path = "release_delivery/performance.rs"]
mod performance;
#[path = "release_delivery/portable.rs"]
mod portable;
#[path = "release_delivery/progress.rs"]
mod progress;
#[path = "release_delivery/public_entry.rs"]
mod public_entry;
#[path = "release_delivery/public_install.rs"]
mod public_install;
#[path = "release_delivery/publish.rs"]
mod publish;
#[path = "release_delivery/submission.rs"]
mod submission;

#[derive(Debug, Clone, serde::Serialize)]
struct AcceptedAsset {
    path: PathBuf,
    name: String,
    sha256: String,
}
#[derive(Debug, Clone, serde::Serialize)]
struct AcceptedRelease {
    version: String,
    candidate_sha: String,
    workspace: PathBuf,
    notes: String,
    assets: Vec<AcceptedAsset>,
    /// Model execution and extracted-archive startup, not public installer execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    sm120_runtime: Option<Sm120RuntimeCoverage>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Sm120RuntimeCoverage {
    asset_name: String,
    asset_sha256: String,
    binary_sha256: String,
    #[serde(flatten)]
    outcome: Sm120RuntimeOutcome,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Sm120RuntimeOutcome {
    Passed {
        source: gate::Sm120RuntimeSource,
        ci_job: Sm120CiObservation,
    },
    NotRun {
        reason: String,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
struct Sm120CiObservation {
    workflow_run_id: u64,
    job_id: Option<u64>,
    run_attempt: Option<u64>,
    status: String,
    conclusion: Option<String>,
    failed_steps: Vec<String>,
}
#[derive(Parser)]
#[command(about = "Run bounded model regression and deliver verified release assets")]
struct Args {
    #[command(subcommand)]
    action: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Fetch immutable bytes through a fully SHA-256-verified persistent cache.
    FetchVerified(fetch_verified::FetchArgs),
    /// Package or inspect a Windows CPU/CUDA portable ZIP; no publication gate.
    Portable(portable::PortableArgs),
    Inspect(installation::InspectArgs),
    Installed(public_install::InstalledArgs),
    Entry(public_entry::EntryArgs),
    Gate {
        #[command(flatten)]
        inputs: gate::GateArgs,
        #[arg(long)]
        output: PathBuf,
    },
    Local(local::LocalArgs),
    Performance(performance::PerformanceArgs),
    PerformancePrepare(performance::PerformancePrepareArgs),
    Cloud(cloud::ExecuteArgs),
    Reap(cloud::ReapArgs),
    Publish {
        #[command(flatten)]
        options: publish::PublishArgs,
        /// Completed release evidence, checked again before any publication.
        #[command(flatten)]
        inputs: gate::GateArgs,
    },
}
async fn run(action: Action) -> Result<(), String> {
    match action {
        Action::FetchVerified(args) => fetch_verified::execute(args).await,
        Action::Portable(args) => portable::execute(args).await,
        Action::Installed(args) => public_install::verify(args).await,
        Action::Entry(args) => public_entry::verify(args).await,
        Action::Gate { inputs, output } => {
            let accepted = gate::verify(inputs).await?;
            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(output)
                .map_err(|e| e.to_string())?;
            serde_json::to_writer_pretty(&mut file, &accepted).map_err(|e| e.to_string())
        }
        Action::Inspect(args) => installation::inspect(args).await,
        Action::Local(args) => local::execute(args).await,
        Action::Performance(args) => performance::execute(args).await,
        Action::PerformancePrepare(args) => performance::prepare(args).await,
        Action::Cloud(args) => cloud::execute(args).await,
        Action::Reap(args) => cloud::reap(args).await,
        Action::Publish {
            mut options,
            inputs,
        } => {
            options.repo = inputs.repo.clone();
            let accepted = gate::verify(inputs).await?;
            publish::publish(options, accepted).await
        }
    }
}
#[tokio::main]
async fn main() -> ExitCode {
    match run(Args::parse().action).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("release delivery: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn composed_delivery_commands_have_unambiguous_options() {
        Args::command().debug_assert();
    }

    #[test]
    fn sm120_manual_runtime_source_requires_explicit_typed_cli_selection() {
        let common = [
            "release_delivery",
            "gate",
            "--plan",
            "plan.json",
            "--tasks",
            "tasks.json",
            "--contracts",
            "contracts.json",
            "--installation",
            "installation.json",
            "--abi",
            "asset.abi.json",
            "--windows-staged",
            "windows",
            "--version",
            "0.13.0",
            "--workspace",
            "candidate",
            "--notes",
            "notes.md",
            "--ci-run-id",
            "17",
            "--output",
            "accepted.json",
        ];
        let Action::Gate { inputs, .. } = Args::try_parse_from(common).unwrap().action else {
            panic!("expected gate")
        };
        assert_eq!(inputs.sm120_runtime_source, gate::Sm120RuntimeSource::Ci);
        let mut manual = common.to_vec();
        manual.extend(["--sm120-runtime-source", "manual"]);
        let Action::Gate { inputs, .. } = Args::try_parse_from(manual).unwrap().action else {
            panic!("expected gate")
        };
        assert_eq!(
            inputs.sm120_runtime_source,
            gate::Sm120RuntimeSource::Manual
        );
        let mut invalid = common.to_vec();
        invalid.extend(["--sm120-runtime-source", "skip"]);
        assert!(Args::try_parse_from(invalid).is_err());
    }
}
