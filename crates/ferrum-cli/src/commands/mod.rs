//! CLI Commands - Ollama-style interface

use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PrefillDecodeExecutionArg {
    Split,
    Mixed,
}

impl PrefillDecodeExecutionArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SequenceFitPolicyArg {
    FullInputMustFit,
    ImmediateOnly,
}

impl SequenceFitPolicyArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::FullInputMustFit => "full-input-must-fit",
            Self::ImmediateOnly => "immediate-only",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VNextDiagnosticFaultArg {
    PrefillResourceAfterSubmitOnce,
}

impl VNextDiagnosticFaultArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::PrefillResourceAfterSubmitOnce => "prefill-resource-after-submit-once",
        }
    }
}

pub mod bench;
pub mod bench_serve;
pub mod doctor;
pub mod embed;
pub mod list;
pub mod pull;
pub mod replay_bundle;
pub mod run;
pub mod serve;
pub mod stop;
pub mod transcribe;
pub mod tts;
pub mod vnext_checkpoint;
pub mod vnext_determinism;

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Parser, Subcommand};

    #[test]
    fn decode_lookahead_is_explicit_and_disableable_for_run_and_serve() {
        for command in ["run", "serve"] {
            for (flag, expected) in [
                (None, None),
                (Some("--decode-lookahead"), Some(true)),
                (Some("--decode-lookahead=false"), Some(false)),
            ] {
                let mut args = vec!["ferrum", command, "test-model"];
                args.extend(flag);
                let value = match TestCli::try_parse_from(args).unwrap().command {
                    TestCommand::Run(cmd) => cmd.decode_lookahead,
                    TestCommand::Serve(cmd) => cmd.decode_lookahead,
                };
                assert_eq!(value, expected);
            }
        }
    }

    #[derive(Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommand,
    }

    #[derive(Subcommand)]
    enum TestCommand {
        Run(run::RunCommand),
        Serve(serve::ServeCommand),
    }

    impl TestCli {
        fn execution(self) -> Option<PrefillDecodeExecutionArg> {
            match self.command {
                TestCommand::Run(command) => command.prefill_decode_execution,
                TestCommand::Serve(command) => command.prefill_decode_execution,
            }
        }
    }

    #[test]
    fn basic_metrics_only_cli_configuration_matches_run_and_serve() {
        for command in ["run", "serve"] {
            for detail in ["off", "basic", "latency", "full"] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--profile-detail",
                    detail,
                ])
                .unwrap();
                let config = match parsed.command {
                    TestCommand::Run(cmd) => {
                        crate::observability_product::ProductObservabilityConfig::new(
                            ferrum_types::ProfileEntrypoint::Run,
                            "test-model",
                            cmd.profile_jsonl.as_ref(),
                            cmd.profile_detail,
                            cmd.memory_profile_jsonl.as_ref(),
                            cmd.scheduler_trace_jsonl.as_ref(),
                            cmd.request_dump_dir.as_ref(),
                            cmd.profile_sample_rate,
                        )
                    }
                    TestCommand::Serve(cmd) => {
                        crate::observability_product::ProductObservabilityConfig::new(
                            ferrum_types::ProfileEntrypoint::Serve,
                            "test-model",
                            cmd.profile_jsonl.as_ref(),
                            cmd.profile_detail,
                            cmd.memory_profile_jsonl.as_ref(),
                            cmd.scheduler_trace_jsonl.as_ref(),
                            cmd.request_dump_dir.as_ref(),
                            cmd.profile_sample_rate,
                        )
                    }
                };
                assert_eq!(config.metrics_only(), detail == "basic");
                assert_eq!(
                    config.core.validate().is_ok(),
                    matches!(detail, "off" | "basic")
                );
            }
        }
    }

    #[test]
    fn prefill_decode_execution_cli_values_and_default_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .execution(),
                None
            );
            for (value, expected) in [
                ("split", PrefillDecodeExecutionArg::Split),
                ("mixed", PrefillDecodeExecutionArg::Mixed),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--prefill-decode-execution",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.execution(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--prefill-decode-execution",
                "automatic"
            ])
            .is_err());
        }
    }
}
