//! Run real pi processes against one local Ferrum server, retaining task and
//! transport evidence separately from independently observed engine concurrency.
#[path = "agent_regression/config.rs"]
mod config;
#[path = "agent_regression/events.rs"]
mod events;
#[path = "agent_regression/http_replay.rs"]
mod http_replay;
#[path = "agent_regression/orchestral.rs"]
mod orchestral;
#[path = "agent_regression/orchestral_evidence.rs"]
mod orchestral_evidence;
#[path = "agent_regression/orchestral_wire.rs"]
mod orchestral_wire;
#[path = "agent_regression/process.rs"]
mod process;
#[path = "agent_regression/proxy.rs"]
mod proxy;
#[path = "agent_regression/repair.rs"]
mod repair;
#[path = "agent_regression/replay.rs"]
mod replay;
#[path = "agent_regression/rpc.rs"]
mod rpc;
#[path = "agent_regression/runner.rs"]
mod runner;
#[path = "agent_regression/validator.rs"]
mod validator;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(about = "Real local coding-agent task regression and independent Rust validation")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Replay exact streaming OpenAI bodies concurrently, without an agent or tools.
    ReplayHttp(http_replay::Args),
    Run {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        report_dir: PathBuf,
        #[arg(long, value_enum)]
        mode: Mode,
        /// Same server's latency/detail profile; without it execution overlap is unproven.
        #[arg(long)]
        server_profile_jsonl: Option<PathBuf>,
        /// Independent validation feedback rounds in the same Pi RPC session.
        /// Zero preserves the original print-mode run and single final validation.
        #[arg(long, default_value_t = 0)]
        validation_repairs: u32,
    },
    Validate(validator::Args),
    /// Recheck retained Orchestral journals and exact HTTP replay without rerunning a model.
    AuditOrchestral {
        #[arg(long)]
        report_dir: PathBuf,
        #[arg(long)]
        task_id: String,
        /// Explicit codec revision for saved text-parts evidence. Omit to retain
        /// the manifest's interpretation; this never changes the saved manifest.
        #[arg(long, value_enum)]
        tool_result_format: Option<config::OrchestralToolResultFormat>,
        /// New evidence file; existing reports are never overwritten.
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Mode {
    Concurrent,
    Sequential,
}

pub(crate) fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    fs::write(path.as_ref(), serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.as_ref().display()))
}

#[tokio::main]
async fn main() {
    let result = match Args::parse().command {
        Command::ReplayHttp(args) => http_replay::run(&args).await,
        Command::Run {
            manifest,
            report_dir,
            mode,
            server_profile_jsonl,
            validation_repairs,
        } => {
            runner::run(
                &manifest,
                &report_dir,
                mode,
                server_profile_jsonl.as_deref(),
                validation_repairs,
            )
            .await
        }
        Command::Validate(args) => validator::run(&args).await,
        Command::AuditOrchestral {
            report_dir,
            task_id,
            tool_result_format,
            output,
        } => orchestral_wire::audit_saved(&report_dir, &task_id, &output, tool_result_format),
    };
    let code = match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error:#}");
            2
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_audit_codec_override_is_typed_and_not_implicit() {
        let args = [
            "agent_regression",
            "audit-orchestral",
            "--report-dir",
            "report",
            "--task-id",
            "coding",
            "--output",
            "new-audit.json",
        ];
        assert!(matches!(
            Args::try_parse_from(args).unwrap().command,
            Command::AuditOrchestral {
                tool_result_format: None,
                ..
            }
        ));
        for (name, expected) in [
            (
                "text_parts_v2",
                config::OrchestralToolResultFormat::TextPartsV2,
            ),
            (
                "text_parts_v3",
                config::OrchestralToolResultFormat::TextPartsV3,
            ),
        ] {
            assert!(matches!(
                Args::try_parse_from(args.into_iter().chain(["--tool-result-format", name]))
                    .unwrap()
                    .command,
                Command::AuditOrchestral {
                    tool_result_format: Some(selected),
                    ..
                } if selected == expected
            ));
        }
        assert!(
            Args::try_parse_from(args.into_iter().chain(["--tool-result-format", "auto"])).is_err()
        );
    }

    #[test]
    fn repairs_are_explicit_and_default_to_the_original_print_run() {
        let args = [
            "agent_regression",
            "run",
            "--manifest",
            "manifest.json",
            "--report-dir",
            "report",
            "--mode",
            "sequential",
        ];
        assert!(matches!(
            Args::try_parse_from(args).unwrap().command,
            Command::Run {
                validation_repairs: 0,
                ..
            }
        ));
        let enabled = args.into_iter().chain(["--validation-repairs", "2"]);
        assert!(matches!(
            Args::try_parse_from(enabled).unwrap().command,
            Command::Run {
                validation_repairs: 2,
                ..
            }
        ));
    }
}
