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
pub enum InvocationPreparationStrategyArg {
    Full,
    IdentityProjection,
    DecodeSegment,
}

impl InvocationPreparationStrategyArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::IdentityProjection => "identity-projection",
            Self::DecodeSegment => "decode-segment",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ProgramBindingUploadStrategyArg {
    Sparse,
    UniformLivePrefix,
    CompactScatter,
}

impl ProgramBindingUploadStrategyArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::UniformLivePrefix => "uniform-live-prefix",
            Self::CompactScatter => "compact-scatter",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SegmentBindingOwnerViewModeArg {
    Legacy,
    Indexed,
}

impl SegmentBindingOwnerViewModeArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Legacy => ferrum_types::SegmentBindingOwnerViewMode::Legacy.as_runtime_value(),
            Self::Indexed => ferrum_types::SegmentBindingOwnerViewMode::Indexed.as_runtime_value(),
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
        fn segment_binding_owner_view_mode(self) -> Option<SegmentBindingOwnerViewModeArg> {
            match self.command {
                TestCommand::Run(command) => command.segment_binding_owner_view_mode,
                TestCommand::Serve(command) => command.segment_binding_owner_view_mode,
            }
        }

        fn binding_upload_strategy(self) -> Option<ProgramBindingUploadStrategyArg> {
            match self.command {
                TestCommand::Run(command) => command.program_binding_upload_strategy,
                TestCommand::Serve(command) => command.program_binding_upload_strategy,
            }
        }

        fn preparation_strategy(self) -> Option<InvocationPreparationStrategyArg> {
            match self.command {
                TestCommand::Run(command) => command.invocation_preparation_strategy,
                TestCommand::Serve(command) => command.invocation_preparation_strategy,
            }
        }

        fn execution(self) -> Option<PrefillDecodeExecutionArg> {
            match self.command {
                TestCommand::Run(command) => command.prefill_decode_execution,
                TestCommand::Serve(command) => command.prefill_decode_execution,
            }
        }
    }

    #[test]
    fn segment_binding_owner_view_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .segment_binding_owner_view_mode(),
                None
            );
            for (value, expected) in [
                ("legacy", SegmentBindingOwnerViewModeArg::Legacy),
                ("indexed", SegmentBindingOwnerViewModeArg::Indexed),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--segment-binding-owner-view-mode",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.segment_binding_owner_view_mode(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--segment-binding-owner-view-mode",
                "automatic",
            ])
            .is_err());
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
    #[test]
    fn invocation_preparation_cli_values_and_default_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .preparation_strategy(),
                None
            );
            for (value, expected) in [
                ("full", InvocationPreparationStrategyArg::Full),
                (
                    "identity-projection",
                    InvocationPreparationStrategyArg::IdentityProjection,
                ),
                (
                    "decode-segment",
                    InvocationPreparationStrategyArg::DecodeSegment,
                ),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--invocation-preparation-strategy",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.preparation_strategy(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--invocation-preparation-strategy",
                "automatic"
            ])
            .is_err());
        }
    }
    #[test]
    fn program_binding_upload_cli_values_and_default_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .binding_upload_strategy(),
                None
            );
            for (value, expected) in [
                ("sparse", ProgramBindingUploadStrategyArg::Sparse),
                (
                    "uniform-live-prefix",
                    ProgramBindingUploadStrategyArg::UniformLivePrefix,
                ),
                (
                    "compact-scatter",
                    ProgramBindingUploadStrategyArg::CompactScatter,
                ),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--program-binding-upload-strategy",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.binding_upload_strategy(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--program-binding-upload-strategy",
                "automatic"
            ])
            .is_err());
        }
    }
}
