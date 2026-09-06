//! Execute release delivery through explicit resource and publication gates.
use clap::{Parser, Subcommand};
use std::{path::PathBuf, process::ExitCode};

#[path = "release_delivery/cloud.rs"]
mod cloud;
#[path = "release_delivery/gate.rs"]
mod gate;
#[path = "release_delivery/installation.rs"]
mod installation;
#[path = "release_delivery/local.rs"]
mod local;
#[path = "release_delivery/performance.rs"]
mod performance;
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
}
#[derive(Parser)]
#[command(about = "Run bounded model regression and deliver verified release assets")]
struct Args {
    #[command(subcommand)]
    action: Action,
}
#[derive(Subcommand)]
enum Action {
    Inspect(installation::InspectArgs),
    Installed(public_install::InstalledArgs),
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
        Action::Installed(args) => public_install::verify(args).await,
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
}
