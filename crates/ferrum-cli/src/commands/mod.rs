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
    DecodeSegmentProgram,
    ResidentDecodeProgram,
}

impl InvocationPreparationStrategyArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::IdentityProjection => "identity-projection",
            Self::DecodeSegment => "decode-segment",
            Self::DecodeSegmentProgram => "decode-segment-program",
            Self::ResidentDecodeProgram => "resident-decode-program",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SamplingReadbackModeArg {
    Uniform,
    Participant,
}

impl SamplingReadbackModeArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Uniform => ferrum_types::SamplingReadbackMode::Uniform.as_runtime_value(),
            Self::Participant => ferrum_types::SamplingReadbackMode::Participant.as_runtime_value(),
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
pub enum CompletionReadbackPolicyArg {
    Synchronous,
    BatchedU32,
}

impl CompletionReadbackPolicyArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Synchronous => {
                ferrum_types::CompletionReadbackPolicy::Synchronous.as_runtime_value()
            }
            Self::BatchedU32 => {
                ferrum_types::CompletionReadbackPolicy::BatchedU32.as_runtime_value()
            }
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
pub enum CausalDecodePreparationModeArg {
    PerParticipant,
    Packed,
}

impl CausalDecodePreparationModeArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::PerParticipant => {
                ferrum_types::CausalDecodePreparationMode::PerParticipant.as_runtime_value()
            }
            Self::Packed => ferrum_types::CausalDecodePreparationMode::Packed.as_runtime_value(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CausalDecodeAttentionGroupingArg {
    Contiguous,
    Indexed,
    FixedCapacity,
    FixedExecutionCapacity,
}

impl CausalDecodeAttentionGroupingArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Contiguous => {
                ferrum_types::CausalDecodeAttentionGrouping::Contiguous.as_runtime_value()
            }
            Self::Indexed => {
                ferrum_types::CausalDecodeAttentionGrouping::Indexed.as_runtime_value()
            }
            Self::FixedCapacity => {
                ferrum_types::CausalDecodeAttentionGrouping::FixedCapacity.as_runtime_value()
            }
            Self::FixedExecutionCapacity => {
                ferrum_types::CausalDecodeAttentionGrouping::FixedExecutionCapacity
                    .as_runtime_value()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CausalDecodeKvReuseArg {
    None,
    PairedQueryHeads,
    PairedQueryHeadsUnifiedV2,
}

impl CausalDecodeKvReuseArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::None => ferrum_types::CausalDecodeKvReuse::None.as_runtime_value(),
            Self::PairedQueryHeads => {
                ferrum_types::CausalDecodeKvReuse::PairedQueryHeads.as_runtime_value()
            }
            Self::PairedQueryHeadsUnifiedV2 => {
                ferrum_types::CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2.as_runtime_value()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SequenceFitPolicyArg {
    FullInputMustFit,
    ImmediateOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SequenceBackingGrowthArg {
    Exact,
    #[value(name = "paged-group4")]
    PagedGroup4,
    #[value(name = "paged-group4-grow-only")]
    PagedGroup4GrowOnly,
}

impl SequenceBackingGrowthArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Exact => ferrum_types::SequenceBackingGrowth::Exact.as_runtime_value(),
            Self::PagedGroup4 => {
                ferrum_types::SequenceBackingGrowth::PagedGroup4.as_runtime_value()
            }
            Self::PagedGroup4GrowOnly => {
                ferrum_types::SequenceBackingGrowth::PagedGroup4GrowOnly.as_runtime_value()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Q4MmvqWeightDecodeArg {
    Auto,
    Baseline,
    SharedM8,
    #[value(name = "shared-m4-m8")]
    SharedM4M8,
}

impl Q4MmvqWeightDecodeArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Auto => ferrum_types::Q4MmvqWeightDecode::Auto.as_runtime_value(),
            Self::Baseline => ferrum_types::Q4MmvqWeightDecode::Baseline.as_runtime_value(),
            Self::SharedM8 => ferrum_types::Q4MmvqWeightDecode::SharedM8.as_runtime_value(),
            Self::SharedM4M8 => ferrum_types::Q4MmvqWeightDecode::SharedM4M8.as_runtime_value(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Q5MmvqWeightDecodeArg {
    Auto,
    Baseline,
    SharedM8,
    #[value(name = "shared-m4-m8")]
    SharedM4M8,
}

impl Q5MmvqWeightDecodeArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Auto => ferrum_types::Q5MmvqWeightDecode::Auto.as_runtime_value(),
            Self::Baseline => ferrum_types::Q5MmvqWeightDecode::Baseline.as_runtime_value(),
            Self::SharedM8 => ferrum_types::Q5MmvqWeightDecode::SharedM8.as_runtime_value(),
            Self::SharedM4M8 => ferrum_types::Q5MmvqWeightDecode::SharedM4M8.as_runtime_value(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Q8ProjectionWeightPreparationArg {
    Compressed,
    PredecodedF32,
}

impl Q8ProjectionWeightPreparationArg {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Compressed => {
                ferrum_types::Q8ProjectionWeightPreparation::Compressed.as_runtime_value()
            }
            Self::PredecodedF32 => {
                ferrum_types::Q8ProjectionWeightPreparation::PredecodedF32.as_runtime_value()
            }
        }
    }
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

        fn causal_decode_preparation_mode(self) -> Option<CausalDecodePreparationModeArg> {
            match self.command {
                TestCommand::Run(command) => command.causal_decode_preparation_mode,
                TestCommand::Serve(command) => command.causal_decode_preparation_mode,
            }
        }

        fn causal_decode_attention_grouping(self) -> Option<CausalDecodeAttentionGroupingArg> {
            match self.command {
                TestCommand::Run(command) => command.causal_decode_attention_grouping,
                TestCommand::Serve(command) => command.causal_decode_attention_grouping,
            }
        }

        fn q4_mmvq_weight_decode(self) -> Option<Q4MmvqWeightDecodeArg> {
            match self.command {
                TestCommand::Run(command) => command.q4_mmvq_weight_decode,
                TestCommand::Serve(command) => command.q4_mmvq_weight_decode,
            }
        }
        fn q5_mmvq_weight_decode(self) -> Option<Q5MmvqWeightDecodeArg> {
            match self.command {
                TestCommand::Run(command) => command.q5_mmvq_weight_decode,
                TestCommand::Serve(command) => command.q5_mmvq_weight_decode,
            }
        }

        fn q8_projection_weight_preparation(self) -> Option<Q8ProjectionWeightPreparationArg> {
            match self.command {
                TestCommand::Run(command) => command.q8_projection_weight_preparation,
                TestCommand::Serve(command) => command.q8_projection_weight_preparation,
            }
        }

        fn causal_decode_kv_reuse(self) -> Option<CausalDecodeKvReuseArg> {
            match self.command {
                TestCommand::Run(command) => command.causal_decode_kv_reuse,
                TestCommand::Serve(command) => command.causal_decode_kv_reuse,
            }
        }

        fn sequence_backing_growth(self) -> Option<SequenceBackingGrowthArg> {
            match self.command {
                TestCommand::Run(command) => command.sequence_backing_growth,
                TestCommand::Serve(command) => command.sequence_backing_growth,
            }
        }

        fn completion_readback_policy(self) -> Option<CompletionReadbackPolicyArg> {
            match self.command {
                TestCommand::Run(command) => command.completion_readback_policy,
                TestCommand::Serve(command) => command.completion_readback_policy,
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

        fn sampling_readback_mode(self) -> Option<SamplingReadbackModeArg> {
            match self.command {
                TestCommand::Run(command) => command.sampling_readback_mode,
                TestCommand::Serve(command) => command.sampling_readback_mode,
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
    fn causal_decode_preparation_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .causal_decode_preparation_mode(),
                None
            );
            for (value, expected) in [
                (
                    "per-participant",
                    CausalDecodePreparationModeArg::PerParticipant,
                ),
                ("packed", CausalDecodePreparationModeArg::Packed),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--causal-decode-preparation-mode",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.causal_decode_preparation_mode(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--causal-decode-preparation-mode",
                "automatic",
            ])
            .is_err());
        }
    }

    #[test]
    fn causal_decode_attention_grouping_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .causal_decode_attention_grouping(),
                None
            );
            for (value, expected) in [
                ("contiguous", CausalDecodeAttentionGroupingArg::Contiguous),
                ("indexed", CausalDecodeAttentionGroupingArg::Indexed),
                (
                    "fixed-capacity",
                    CausalDecodeAttentionGroupingArg::FixedCapacity,
                ),
                (
                    "fixed-execution-capacity",
                    CausalDecodeAttentionGroupingArg::FixedExecutionCapacity,
                ),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--causal-decode-attention-grouping",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.causal_decode_attention_grouping(), Some(expected));
                assert_eq!(expected.as_runtime_value(), value);
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--causal-decode-attention-grouping",
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
    fn sampling_readback_cli_accepts_only_explicit_values_for_both_entrypoints() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .sampling_readback_mode(),
                None
            );
            for (value, expected) in [
                ("uniform", SamplingReadbackModeArg::Uniform),
                ("participant", SamplingReadbackModeArg::Participant),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--sampling-readback-mode",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.sampling_readback_mode(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--sampling-readback-mode",
                "automatic",
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
                (
                    "decode-segment-program",
                    InvocationPreparationStrategyArg::DecodeSegmentProgram,
                ),
                (
                    "resident-decode-program",
                    InvocationPreparationStrategyArg::ResidentDecodeProgram,
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
    fn completion_readback_policy_cli_values_and_default_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .completion_readback_policy(),
                None
            );
            for (value, expected) in [
                ("synchronous", CompletionReadbackPolicyArg::Synchronous),
                ("batched-u32", CompletionReadbackPolicyArg::BatchedU32),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--completion-readback-policy",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.completion_readback_policy(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--completion-readback-policy",
                "automatic",
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
    #[test]
    fn q4_mmvq_weight_decode_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .q4_mmvq_weight_decode(),
                None
            );
            for (value, expected) in [
                ("auto", Q4MmvqWeightDecodeArg::Auto),
                ("baseline", Q4MmvqWeightDecodeArg::Baseline),
                ("shared-m8", Q4MmvqWeightDecodeArg::SharedM8),
                ("shared-m4-m8", Q4MmvqWeightDecodeArg::SharedM4M8),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--q4-mmvq-weight-decode",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.q4_mmvq_weight_decode(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--q4-mmvq-weight-decode",
                "automatic",
            ])
            .is_err());
        }
    }
    #[test]
    fn q5_mmvq_weight_decode_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .q5_mmvq_weight_decode(),
                None
            );
            for (value, expected) in [
                ("auto", Q5MmvqWeightDecodeArg::Auto),
                ("baseline", Q5MmvqWeightDecodeArg::Baseline),
                ("shared-m8", Q5MmvqWeightDecodeArg::SharedM8),
                ("shared-m4-m8", Q5MmvqWeightDecodeArg::SharedM4M8),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--q5-mmvq-weight-decode",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.q5_mmvq_weight_decode(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--q5-mmvq-weight-decode",
                "automatic",
            ])
            .is_err());
        }
    }
    #[test]
    fn q8_projection_weight_preparation_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .q8_projection_weight_preparation(),
                None
            );
            for (value, expected) in [
                ("compressed", Q8ProjectionWeightPreparationArg::Compressed),
                (
                    "predecoded-f32",
                    Q8ProjectionWeightPreparationArg::PredecodedF32,
                ),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--q8-projection-weight-preparation",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.q8_projection_weight_preparation(), Some(expected));
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--q8-projection-weight-preparation",
                "automatic",
            ])
            .is_err());
        }
    }

    #[test]
    fn causal_decode_kv_reuse_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .causal_decode_kv_reuse(),
                None
            );
            for (value, expected) in [
                ("none", CausalDecodeKvReuseArg::None),
                (
                    "paired-query-heads",
                    CausalDecodeKvReuseArg::PairedQueryHeads,
                ),
                (
                    "paired-query-heads-unified-v2",
                    CausalDecodeKvReuseArg::PairedQueryHeadsUnifiedV2,
                ),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--causal-decode-kv-reuse",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.causal_decode_kv_reuse(), Some(expected));
                assert_eq!(expected.as_runtime_value(), value);
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--causal-decode-kv-reuse",
                "automatic",
            ])
            .is_err());
        }
    }

    #[test]
    fn sequence_backing_growth_cli_values_match_for_run_and_serve() {
        for command in ["run", "serve"] {
            assert_eq!(
                TestCli::try_parse_from(["ferrum", command, "test-model"])
                    .unwrap()
                    .sequence_backing_growth(),
                None
            );
            for (value, expected) in [
                ("exact", SequenceBackingGrowthArg::Exact),
                ("paged-group4", SequenceBackingGrowthArg::PagedGroup4),
                (
                    "paged-group4-grow-only",
                    SequenceBackingGrowthArg::PagedGroup4GrowOnly,
                ),
            ] {
                let parsed = TestCli::try_parse_from([
                    "ferrum",
                    command,
                    "test-model",
                    "--sequence-backing-growth",
                    value,
                ])
                .unwrap();
                assert_eq!(parsed.sequence_backing_growth(), Some(expected));
                assert_eq!(expected.as_runtime_value(), value);
            }
            assert!(TestCli::try_parse_from([
                "ferrum",
                command,
                "test-model",
                "--sequence-backing-growth",
                "automatic"
            ])
            .is_err());
        }
    }
}
