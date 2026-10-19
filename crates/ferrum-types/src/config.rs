//! Configuration types for Ferrum components

use crate::{
    parse_bool_env_value, parse_path_env_value, parse_usize_env_value, AttentionExecutionPolicy,
    DataType, Device, ModelId, ModelInfo, ObservabilityProfileDetail, ProfileEntrypoint,
    RuntimeConfigSnapshot, SamplingParams, SamplingPresets, TokenId,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, path::PathBuf, time::Duration};

/// Product policy for proving that a sequence fits before prefill admission.
///
/// This is a non-reserving fit gate: it does not claim future KV blocks. The
/// execution runtime still acquires exact live-frontier resources transactionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SequenceFitPolicy {
    FullInputMustFit,
    ImmediateOnly,
}

impl SequenceFitPolicy {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::FullInputMustFit => "full-input-must-fit",
            Self::ImmediateOnly => "immediate-only",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "full-input-must-fit" => Ok(Self::FullInputMustFit),
            "immediate-only" => Ok(Self::ImmediateOnly),
            _ => Err(format!(
                "expected full-input-must-fit or immediate-only; got {raw:?}"
            )),
        }
    }
}

impl Default for SequenceFitPolicy {
    fn default() -> Self {
        Self::ImmediateOnly
    }
}

/// Physical execution policy for a PlanRuntime batch containing prefill and decode.
/// Legacy executors retain their existing execution policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillDecodeExecution {
    #[default]
    Split,
    Mixed,
}

impl PrefillDecodeExecution {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Mixed => "mixed",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "split" => Ok(Self::Split),
            "mixed" => Ok(Self::Mixed),
            _ => Err(format!("expected split or mixed; got {raw:?}")),
        }
    }
}

/// Host invocation preparation for native plan runtimes. Identity projection
/// borrows compiled identity metadata while preserving per-invocation validation.
/// DecodeSegment enables supported CUDA decode segments; unsupported or cold
/// segments retain the full preparation path. Full remains the default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InvocationPreparationStrategy {
    #[default]
    Full,
    IdentityProjection,
    DecodeSegment,
    /// Compile the entire binding write program and bind it from one fresh frame.
    DecodeSegmentProgram,
    /// Retain a complete decode program and acquire only fresh execution authority
    /// on a stable cohort. Cold or unsupported uses retain compiled preparation.
    ResidentDecodeProgram,
}

impl InvocationPreparationStrategy {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::IdentityProjection => "identity-projection",
            Self::DecodeSegment => "decode-segment",
            Self::DecodeSegmentProgram => "decode-segment-program",
            Self::ResidentDecodeProgram => "resident-decode-program",
        }
    }

    pub const fn uses_identity_projection(self) -> bool {
        matches!(
            self,
            Self::IdentityProjection
                | Self::DecodeSegment
                | Self::DecodeSegmentProgram
                | Self::ResidentDecodeProgram
        )
    }

    pub const fn uses_decode_segment(self) -> bool {
        matches!(
            self,
            Self::DecodeSegment | Self::DecodeSegmentProgram | Self::ResidentDecodeProgram
        )
    }

    pub const fn uses_decode_segment_program(self) -> bool {
        matches!(
            self,
            Self::DecodeSegmentProgram | Self::ResidentDecodeProgram
        )
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "full" => Ok(Self::Full),
            "identity-projection" => Ok(Self::IdentityProjection),
            "decode-segment" => Ok(Self::DecodeSegment),
            "decode-segment-program" => Ok(Self::DecodeSegmentProgram),
            "resident-decode-program" => Ok(Self::ResidentDecodeProgram),
            _ => Err(format!(
                "expected full, identity-projection, decode-segment, decode-segment-program or resident-decode-program; got {raw:?}"
            )),
        }
    }
}

/// Product sampling readback granularity. Participant selection applies only
/// to eligible pure decode rows; other waves retain the uniform policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SamplingReadbackMode {
    #[default]
    Uniform,
    Participant,
}

impl SamplingReadbackMode {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Participant => "participant",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "uniform" => Ok(Self::Uniform),
            "participant" => Ok(Self::Participant),
            _ => Err(format!("expected uniform or participant; got {raw:?}")),
        }
    }
}

/// Upload policy for CUDA program bindings. Uniform live prefixes pad rows
/// only to the largest live payload in the same node and current batch.
/// Compact scatter uploads one packed packet and scatters its live bytes on
/// the GPU, falling back to sparse uploads when scratch capacity is unavailable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProgramBindingUploadStrategy {
    #[default]
    Sparse,
    UniformLivePrefix,
    CompactScatter,
}

/// Completion transfer policy. Unsupported batched scalar transfers retain
/// synchronous readback; sampling and output layouts are unchanged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompletionReadbackPolicy {
    #[default]
    Synchronous,
    BatchedU32,
}

/// Snapshot identity for an explicit CLI/config option, not an environment switch.
pub const COMPLETION_READBACK_POLICY_CONFIG_KEY: &str = "FERRUM_COMPLETION_READBACK_POLICY";

impl CompletionReadbackPolicy {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Synchronous => "synchronous",
            Self::BatchedU32 => "batched-u32",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "synchronous" => Ok(Self::Synchronous),
            "batched-u32" => Ok(Self::BatchedU32),
            _ => Err(format!("expected synchronous or batched-u32; got {raw:?}")),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err("select completion readback through CLI or config, not environment".into());
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

impl ProgramBindingUploadStrategy {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::UniformLivePrefix => "uniform-live-prefix",
            Self::CompactScatter => "compact-scatter",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sparse" => Ok(Self::Sparse),
            "uniform-live-prefix" => Ok(Self::UniformLivePrefix),
            "compact-scatter" => Ok(Self::CompactScatter),
            _ => Err(format!(
                "expected sparse, uniform-live-prefix or compact-scatter; got {raw:?}"
            )),
        }
    }
}

/// Internal snapshot identity for an explicit CLI/config option, not an
/// environment-variable interface. Environment-sourced entries are rejected.
pub const SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY: &str =
    "FERRUM_SEGMENT_BINDING_OWNER_VIEW_MODE";

/// Snapshot identity for an explicit CLI/config option, never an environment switch.
pub const CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY: &str = "FERRUM_CAUSAL_DECODE_PREPARATION_MODE";

/// Snapshot identity for explicit attention grouping, never an environment switch.
pub const CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY: &str =
    "FERRUM_CAUSAL_DECODE_ATTENTION_GROUPING";

/// Snapshot identity for explicit Q4 MMVQ decode, never an environment switch.
pub const Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY: &str = "FERRUM_Q4_MMVQ_WEIGHT_DECODE";

/// Weight decode implementation for already selected Q4_K MarkerV2 MMVQ rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q4MmvqWeightDecode {
    /// Resolve against the selected numerical route and prepared native geometry.
    Auto,
    #[default]
    Baseline,
    SharedM8,
    SharedM4M8,
}

impl Q4MmvqWeightDecode {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Baseline => "baseline",
            Self::SharedM8 => "shared-m8",
            Self::SharedM4M8 => "shared-m4-m8",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "baseline" => Ok(Self::Baseline),
            "shared-m8" => Ok(Self::SharedM8),
            "shared-m4-m8" => Ok(Self::SharedM4M8),
            _ => Err(format!(
                "expected auto, baseline, shared-m8 or shared-m4-m8; got {raw:?}"
            )),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select Q4 MMVQ weight decode through CLI or config, not environment".into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Snapshot identity for explicit Q5 MMVQ decode, never an environment switch.
pub const Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY: &str = "FERRUM_Q5_MMVQ_WEIGHT_DECODE";

/// Weight decode implementation for already selected Q5_K MarkerV2 MMVQ rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q5MmvqWeightDecode {
    /// Resolve against the selected numerical route and prepared native geometry.
    Auto,
    #[default]
    Baseline,
    SharedM8,
    SharedM4M8,
}

impl Q5MmvqWeightDecode {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Baseline => "baseline",
            Self::SharedM8 => "shared-m8",
            Self::SharedM4M8 => "shared-m4-m8",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "baseline" => Ok(Self::Baseline),
            "shared-m8" => Ok(Self::SharedM8),
            "shared-m4-m8" => Ok(Self::SharedM4M8),
            _ => Err(format!(
                "expected auto, baseline, shared-m8 or shared-m4-m8; got {raw:?}"
            )),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select Q5 MMVQ weight decode through CLI or config, not environment".into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Explicit preparation policy snapshot key; never an environment switch.
pub const Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY: &str =
    "FERRUM_Q8_PROJECTION_WEIGHT_PREPARATION";

/// Retained representation of eligible strict Q8_0 projection leaves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q8ProjectionWeightPreparation {
    /// Try eligible F32 destinations only after the compressed plan's capacity
    /// is fixed; retain compressed storage when the optional layout cannot fit.
    Auto,
    #[default]
    Compressed,
    PredecodedF32,
}

impl Q8ProjectionWeightPreparation {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Compressed => "compressed",
            Self::PredecodedF32 => "predecoded-f32",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "compressed" => Ok(Self::Compressed),
            "predecoded-f32" => Ok(Self::PredecodedF32),
            _ => Err(format!(
                "expected auto, compressed or predecoded-f32; got {raw:?}"
            )),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select Q8 projection weight preparation through CLI or config, not environment"
                    .into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Configuration provenance key; this policy is selected through CLI or config.
pub const CAUSAL_DECODE_KV_REUSE_CONFIG_KEY: &str = "FERRUM_CAUSAL_DECODE_KV_REUSE";

/// Provenance key for the explicit CLI/config Sequence reservation policy.
pub const SEQUENCE_BACKING_GROWTH_CONFIG_KEY: &str = "FERRUM_SEQUENCE_BACKING_GROWTH";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SequenceBackingGrowth {
    #[default]
    Exact,
    PagedGroup4,
    PagedGroup4GrowOnly,
}

impl SequenceBackingGrowth {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::PagedGroup4 => "paged-group4",
            Self::PagedGroup4GrowOnly => "paged-group4-grow-only",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "exact" => Ok(Self::Exact),
            "paged-group4" => Ok(Self::PagedGroup4),
            "paged-group4-grow-only" => Ok(Self::PagedGroup4GrowOnly),
            _ => Err(format!(
                "expected exact, paged-group4, or paged-group4-grow-only; got {raw:?}"
            )),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select Sequence backing growth through CLI or config, not environment".into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Reuse KV loads across query heads without changing each head's arithmetic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CausalDecodeKvReuse {
    #[default]
    None,
    PairedQueryHeads,
    /// Use paired-query V2 for every positive row in an eligible fixed batch.
    /// Unsupported geometries retain their existing attention path.
    PairedQueryHeadsUnifiedV2,
}

impl CausalDecodeKvReuse {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PairedQueryHeads => "paired-query-heads",
            Self::PairedQueryHeadsUnifiedV2 => "paired-query-heads-unified-v2",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "paired-query-heads" => Ok(Self::PairedQueryHeads),
            "paired-query-heads-unified-v2" => Ok(Self::PairedQueryHeadsUnifiedV2),
            _ => Err(format!(
                "expected none, paired-query-heads or paired-query-heads-unified-v2; got {raw:?}"
            )),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select causal decode KV reuse through CLI or config, not environment".into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Grouping of already selected native V1/V2 causal decode rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CausalDecodeAttentionGrouping {
    #[default]
    Contiguous,
    Indexed,
    /// Preserve decode command capacities while current device lengths select V1/V2 rows.
    FixedCapacity,
    /// Bound fixed native scratch by the declared execution context limit.
    FixedExecutionCapacity,
}

impl CausalDecodeAttentionGrouping {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::Indexed => "indexed",
            Self::FixedCapacity => "fixed-capacity",
            Self::FixedExecutionCapacity => "fixed-execution-capacity",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "contiguous" => Ok(Self::Contiguous),
            "indexed" => Ok(Self::Indexed),
            "fixed-capacity" => Ok(Self::FixedCapacity),
            "fixed-execution-capacity" => Ok(Self::FixedExecutionCapacity),
            _ => Err(format!(
                "expected contiguous, indexed, fixed-capacity or fixed-execution-capacity; got {raw:?}"
            )),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select causal decode attention grouping through CLI or config, not environment"
                    .into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Preparation of rows in an eligible native batched causal decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CausalDecodePreparationMode {
    #[default]
    PerParticipant,
    Packed,
}

impl CausalDecodePreparationMode {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::PerParticipant => "per-participant",
            Self::Packed => "packed",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "per-participant" => Ok(Self::PerParticipant),
            "packed" => Ok(Self::Packed),
            _ => Err(format!("expected per-participant or packed; got {raw:?}")),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select causal decode preparation through CLI or config, not environment".into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Representation of freshly checked decode-segment binding owners. Both
/// modes retain the same current-wave authorization and exact physical ranges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentBindingOwnerViewMode {
    #[default]
    Legacy,
    Indexed,
}

impl SegmentBindingOwnerViewMode {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Indexed => "indexed",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "legacy" => Ok(Self::Legacy),
            "indexed" => Ok(Self::Indexed),
            _ => Err(format!("expected legacy or indexed; got {raw:?}")),
        }
    }

    pub(crate) fn parse_config_entry(
        entry: &crate::RuntimeConfigEntry,
    ) -> std::result::Result<Self, String> {
        if entry.source == crate::RuntimeConfigSource::Env {
            return Err(
                "select segment binding owner views through CLI or config, not environment".into(),
            );
        }
        Self::parse_runtime_value(&entry.effective_value)
    }
}

/// Explicit one-shot faults used to prove product-path failure attribution.
/// These are never inferred and remain disabled in normal execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VNextDiagnosticFault {
    PrefillResourceAfterSubmitOnce,
}

impl VNextDiagnosticFault {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::PrefillResourceAfterSubmitOnce => "prefill-resource-after-submit-once",
        }
    }

    pub fn parse_runtime_value(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "prefill-resource-after-submit-once" => Ok(Self::PrefillResourceAfterSubmitOnce),
            _ => Err(format!(
                "expected prefill-resource-after-submit-once; got {raw:?}"
            )),
        }
    }
}

/// Explicit diagnostic capture of semantic vNext activations. Product paths
/// leave this unset; release tooling must supply a dedicated empty directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VNextCheckpointCaptureConfig {
    pub output_dir: PathBuf,
    pub value_ids: Vec<String>,
    pub maximum_prefill_waves: usize,
    #[serde(default)]
    pub maximum_decode_waves: usize,
    /// Persist the product output that the executor already reads back. Unlike
    /// `value_ids`, this does not retain an activation or alter memory planning.
    #[serde(default)]
    pub capture_product_output: bool,
    /// Optional canonical output history for a same-history numerical
    /// diagnostic. The vNext executor persists each unmodified full-logits
    /// result before forcing the corresponding token into the engine-facing
    /// copy. Ordinary product inference leaves this unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_forcing: Option<VNextTeacherForcingConfig>,
}

pub const MAX_VNEXT_TEACHER_FORCED_TOKENS: usize = 512;

/// Bounded token history used only by explicit vNext checkpoint diagnostics.
///
/// Construction and executor binding both validate this contract. The latter
/// is required because deserialization can bypass `new` and only model binding
/// knows the live vocabulary size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VNextTeacherForcingConfig {
    token_ids: Vec<TokenId>,
}

impl VNextTeacherForcingConfig {
    pub fn new(token_ids: Vec<TokenId>) -> std::result::Result<Self, String> {
        let value = Self { token_ids };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.token_ids.is_empty() || self.token_ids.len() > MAX_VNEXT_TEACHER_FORCED_TOKENS {
            return Err(format!(
                "vNext checkpoint teacher forcing requires 1..={MAX_VNEXT_TEACHER_FORCED_TOKENS} tokens"
            ));
        }
        Ok(())
    }

    pub fn token_ids(&self) -> &[TokenId] {
        &self.token_ids
    }

    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    /// Canonical identity used by release validators: concatenated little-
    /// endian u32 token IDs, with no JSON or path-dependent representation.
    pub fn token_ids_sha256(&self) -> String {
        let mut digest = Sha256::new();
        for token in &self.token_ids {
            digest.update(token.get().to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }
}

/// Engine runtime knobs the CLI/autosizer resolves and injects via the
/// runtime-config snapshot. The continuous engine reads these from the typed
/// config instead of `std::env::vars()`, so the env bridge stays at the
/// composition root and tests can vary the knobs per `EngineConfig`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeKnobs {
    #[serde(default)]
    pub backend_context: crate::BackendContextConfig,
    /// Retained Llama-family construction controls, never installed globally.
    #[serde(default)]
    pub llama: crate::LlamaExecutionOptions,
    /// Native GPTQ loader diagnostics, scoped to this model's loader.
    #[serde(default)]
    pub gptq_gidx_trace: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_memory_request: Option<crate::StartupMemoryRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_memory_plan: Option<crate::StartupMemoryPlan>,
    pub kv_capacity: Option<usize>,
    pub max_model_len: Option<usize>,
    pub chunked_prefill_size: Option<usize>,
    pub batch_decode_prof: bool,
    #[serde(default)]
    pub profile_jsonl: Option<PathBuf>,
    pub scheduler_trace_jsonl: Option<PathBuf>,
    pub legacy_scheduler_trace_jsonl: Option<PathBuf>,
    pub profile_entrypoint: Option<ProfileEntrypoint>,
    pub profile_detail: ObservabilityProfileDetail,
    #[serde(default)]
    pub profile_journal: crate::ObservabilityJournalConfig,
    /// Retain independent vNext sequence-state checkpoints across requests.
    #[serde(default)]
    pub prefix_state_cache_enabled: bool,
    pub attention_execution_policy: AttentionExecutionPolicy,
    /// Host invocation preparation policy; full preparation is the default.
    #[serde(default)]
    pub invocation_preparation_strategy: InvocationPreparationStrategy,
    /// Sampling readback granularity; uniform preserves the existing wave policy.
    #[serde(default)]
    pub sampling_readback_mode: SamplingReadbackMode,
    /// CUDA upload policy; uniform prefixes pad to the current batch maximum per node.
    #[serde(default)]
    pub program_binding_upload_strategy: ProgramBindingUploadStrategy,
    /// Optional batching of terminal scalar token transfers; synchronous by default.
    #[serde(default)]
    pub completion_readback_policy: CompletionReadbackPolicy,
    /// Fresh owner representation for CUDA decode segments; Legacy is default.
    #[serde(default)]
    pub segment_binding_owner_view_mode: SegmentBindingOwnerViewMode,
    /// Packed preparation applies only to eligible native batched causal decode.
    #[serde(default)]
    pub causal_decode_preparation_mode: CausalDecodePreparationMode,
    /// Indexed grouping requires packed preparation with native CUDA attention.
    #[serde(default)]
    pub causal_decode_attention_grouping: CausalDecodeAttentionGrouping,
    /// Eligible fixed-capacity CUDA V2 attention may share KV loads across two heads.
    #[serde(default)]
    pub causal_decode_kv_reuse: CausalDecodeKvReuse,
    /// Opportunistic paged Sequence capacity reservation; Exact is default.
    #[serde(default)]
    pub sequence_backing_growth: SequenceBackingGrowth,
    /// Shared weight decoding affects only eligible Q4_K MarkerV2 MMVQ M8 leaves.
    #[serde(default)]
    pub q4_mmvq_weight_decode: Q4MmvqWeightDecode,
    /// Shared weight decoding affects only eligible Q5_K MarkerV2 MMVQ M8 leaves.
    #[serde(default)]
    pub q5_mmvq_weight_decode: Q5MmvqWeightDecode,
    /// Cold preparation of eligible strict Q8_0 projection leaves on CUDA.
    #[serde(default)]
    pub q8_projection_weight_preparation: Q8ProjectionWeightPreparation,

    // Engine-build composition knobs. Previously read directly from the
    // environment by `builder.rs` (FERRUM_MODEL_PATH) and `registry.rs`
    // (FERRUM_DTYPE / FERRUM_METAL_DTYPE /
    // FERRUM_TP). The CLI composition root now resolves them into this typed
    // field so the engine builder and component registry read the snapshot,
    // not `std::env`.
    pub model_path: Option<String>,
    pub dtype: Option<String>,
    pub metal_dtype: Option<String>,
    pub tp: Option<usize>,
    #[serde(default)]
    pub vnext_checkpoint_capture: Option<VNextCheckpointCaptureConfig>,
    #[serde(default)]
    pub vnext_diagnostic_fault: Option<VNextDiagnosticFault>,
}

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    /// Numerical policy is resolved against family contracts and the actual
    /// runtime catalog before any model weight allocation.
    #[serde(default)]
    pub numerical_execution: crate::NumericalExecutionPolicy,
    pub model: EngineModelConfig,
    pub scheduler: SchedulerConfig,
    pub sampling: SamplingConfig,
    pub backend: BackendConfig,
    pub kv_cache: KvCacheConfig,
    pub memory: MemoryConfig,
    pub batching: BatchConfig,
    pub monitoring: MonitoringConfig,
    #[serde(default)]
    pub output: crate::OutputConfig,
    #[serde(default)]
    pub runtime: RuntimeKnobs,
}

impl EngineConfig {
    pub fn apply_runtime_config_snapshot(
        &mut self,
        snapshot: &RuntimeConfigSnapshot,
    ) -> std::result::Result<(), String> {
        snapshot
            .validate_retired_options()
            .map_err(|(key, reason)| format!("{key}: {reason}"))?;
        self.scheduler.apply_runtime_config_snapshot(snapshot)?;
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PREFILL_DECODE_EXECUTION") {
            self.batching.prefill_decode_execution =
                PrefillDecodeExecution::parse_runtime_value(value)
                    .map_err(|reason| format!("FERRUM_PREFILL_DECODE_EXECUTION: {reason}"))?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_KV_MAX_BLOCKS") {
            self.kv_cache.max_blocks =
                parse_required_positive_usize("FERRUM_KV_MAX_BLOCKS", value)?;
            self.runtime.llama.kv_max_blocks = Some(self.kv_cache.max_blocks);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_MAX_BATCHED_TOKENS") {
            self.batching.max_num_batched_tokens =
                parse_required_positive_usize("FERRUM_MAX_BATCHED_TOKENS", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PAGED_MAX_SEQS") {
            self.scheduler.max_running_requests =
                parse_required_positive_usize("FERRUM_PAGED_MAX_SEQS", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES") {
            self.memory.usable_capacity_bytes = Some(parse_required_positive_usize(
                "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
                value,
            )?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_BATCHED_GRAPH") {
            self.backend.enable_cuda_graphs = parse_presence_bool(value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_REUSABLE_EXECUTION") {
            self.backend.enable_reusable_execution = parse_presence_bool(value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_REUSABLE_EXECUTION_PREPARATION")
        {
            self.backend.reusable_execution_capture.preparation = value.parse()?;
        }
        if let Some(value) =
            runtime_config_value(snapshot, "FERRUM_REUSABLE_EXECUTION_EXACT_DECODE_WIDTHS")
        {
            let widths =
                parse_positive_usize_list("FERRUM_REUSABLE_EXECUTION_EXACT_DECODE_WIDTHS", value)?;
            if widths
                .iter()
                .any(|width| *width > MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH)
            {
                return Err(format!(
                    "FERRUM_REUSABLE_EXECUTION_EXACT_DECODE_WIDTHS: startup capture widths must be within 1..={MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH}"
                ));
            }
            self.backend.reusable_execution_capture.exact_decode_widths = Some(widths);
        }
        if let Some(value) = runtime_config_value(
            snapshot,
            "FERRUM_REUSABLE_EXECUTION_MAX_AUTOMATIC_EXACT_DECODE_WIDTH",
        ) {
            let maximum = parse_required_positive_usize(
                "FERRUM_REUSABLE_EXECUTION_MAX_AUTOMATIC_EXACT_DECODE_WIDTH",
                value,
            )?;
            if maximum > MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH {
                return Err(format!(
                    "FERRUM_REUSABLE_EXECUTION_MAX_AUTOMATIC_EXACT_DECODE_WIDTH: must be within 1..={MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH}"
                ));
            }
            self.backend
                .reusable_execution_capture
                .maximum_automatic_exact_decode_width = maximum;
        }
        // Engine runtime knobs (previously read by the engine from env). The
        // CLI/autosizer resolves these into the snapshot; the engine reads the
        // typed `runtime` field instead of `std::env::vars()`.
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_KV_CAPACITY") {
            self.runtime.kv_capacity =
                Some(parse_required_positive_usize("FERRUM_KV_CAPACITY", value)?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_MAX_MODEL_LEN") {
            self.runtime.max_model_len = Some(parse_required_positive_usize(
                "FERRUM_MAX_MODEL_LEN",
                value,
            )?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_ATTENTION_POLICY") {
            self.runtime.attention_execution_policy =
                AttentionExecutionPolicy::parse_runtime_value(value)
                    .map_err(|reason| format!("FERRUM_ATTENTION_POLICY: {reason}"))?;
        }
        if let Some(value) =
            runtime_config_value(snapshot, "FERRUM_INVOCATION_PREPARATION_STRATEGY")
        {
            self.runtime.invocation_preparation_strategy =
                InvocationPreparationStrategy::parse_runtime_value(value).map_err(|reason| {
                    format!("FERRUM_INVOCATION_PREPARATION_STRATEGY: {reason}")
                })?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SAMPLING_READBACK_MODE") {
            self.runtime.sampling_readback_mode = SamplingReadbackMode::parse_runtime_value(value)
                .map_err(|reason| format!("FERRUM_SAMPLING_READBACK_MODE: {reason}"))?;
        }
        if let Some(value) =
            runtime_config_value(snapshot, "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY")
        {
            self.runtime.program_binding_upload_strategy =
                ProgramBindingUploadStrategy::parse_runtime_value(value).map_err(|reason| {
                    format!("FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY: {reason}")
                })?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == COMPLETION_READBACK_POLICY_CONFIG_KEY)
        {
            self.runtime.completion_readback_policy =
                CompletionReadbackPolicy::parse_config_entry(entry).map_err(|reason| {
                    format!("{COMPLETION_READBACK_POLICY_CONFIG_KEY}: {reason}")
                })?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY)
        {
            self.runtime.segment_binding_owner_view_mode =
                SegmentBindingOwnerViewMode::parse_config_entry(entry).map_err(|reason| {
                    format!("{SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY}: {reason}")
                })?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY)
        {
            self.runtime.causal_decode_preparation_mode =
                CausalDecodePreparationMode::parse_config_entry(entry).map_err(|reason| {
                    format!("{CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY}: {reason}")
                })?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY)
        {
            self.runtime.causal_decode_attention_grouping =
                CausalDecodeAttentionGrouping::parse_config_entry(entry).map_err(|reason| {
                    format!("{CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY}: {reason}")
                })?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == CAUSAL_DECODE_KV_REUSE_CONFIG_KEY)
        {
            self.runtime.causal_decode_kv_reuse = CausalDecodeKvReuse::parse_config_entry(entry)
                .map_err(|reason| format!("{CAUSAL_DECODE_KV_REUSE_CONFIG_KEY}: {reason}"))?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == SEQUENCE_BACKING_GROWTH_CONFIG_KEY)
        {
            self.runtime.sequence_backing_growth = SequenceBackingGrowth::parse_config_entry(entry)
                .map_err(|reason| format!("{SEQUENCE_BACKING_GROWTH_CONFIG_KEY}: {reason}"))?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY)
        {
            self.runtime.q4_mmvq_weight_decode = Q4MmvqWeightDecode::parse_config_entry(entry)
                .map_err(|reason| format!("{Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY}: {reason}"))?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY)
        {
            self.runtime.q5_mmvq_weight_decode = Q5MmvqWeightDecode::parse_config_entry(entry)
                .map_err(|reason| format!("{Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY}: {reason}"))?;
        }
        if let Some(entry) = snapshot
            .entries
            .iter()
            .find(|entry| entry.key == Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY)
        {
            self.runtime.q8_projection_weight_preparation =
                Q8ProjectionWeightPreparation::parse_config_entry(entry).map_err(|reason| {
                    format!("{Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY}: {reason}")
                })?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_CHUNKED_PREFILL") {
            self.runtime.chunked_prefill_size =
                parse_usize_env_value(value).ok().filter(|&v| v > 0);
        }
        self.runtime.batch_decode_prof |=
            runtime_config_value(snapshot, "FERRUM_BATCH_DECODE_PROF").is_some();
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PROFILE_JSONL") {
            self.runtime.profile_jsonl = Some(parse_path_env_value(value)?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SCHEDULER_TRACE_JSONL") {
            self.runtime.scheduler_trace_jsonl = Some(parse_path_env_value(value)?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_LEGACY_SCHEDULER_TRACE_JSONL") {
            self.runtime.legacy_scheduler_trace_jsonl = Some(parse_path_env_value(value)?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PROFILE_ENTRYPOINT") {
            self.runtime.profile_entrypoint = Some(parse_profile_entrypoint(
                "FERRUM_PROFILE_ENTRYPOINT",
                value,
            )?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PROFILE_DETAIL") {
            self.runtime.profile_detail =
                ObservabilityProfileDetail::parse(value).ok_or_else(|| {
                    format!(
                    "FERRUM_PROFILE_DETAIL: expected one of off, basic, resource, latency, kernel, debug, replay, verify, full; got {value:?}"
                    )
                })?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_VNEXT_DIAGNOSTIC_FAULT") {
            self.runtime.vnext_diagnostic_fault = Some(
                VNextDiagnosticFault::parse_runtime_value(value)
                    .map_err(|reason| format!("FERRUM_VNEXT_DIAGNOSTIC_FAULT: {reason}"))?,
            );
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PREFIX_CACHE") {
            self.runtime.prefix_state_cache_enabled = parse_bool_env_value(value)
                .map_err(|reason| format!("FERRUM_PREFIX_CACHE: {reason}"))?;
        }

        // Engine-build composition knobs (previously read by builder.rs /
        // registry.rs from env). Only overwrite when the key is present so a
        // later snapshot apply without the key keeps an earlier value.
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_MODEL_PATH") {
            self.runtime.model_path = Some(value.to_string());
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_DTYPE") {
            self.runtime.dtype = Some(value.to_string());
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_METAL_DTYPE") {
            self.runtime.metal_dtype = Some(value.to_string());
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_TP") {
            self.runtime.tp = value.parse::<usize>().ok();
        }

        self.runtime.llama.apply_runtime_config_snapshot(snapshot);
        // Validate the same effective policy that model construction will use.
        // A rank-only snapshot must see an existing explicit TP setting, and
        // all other legacy overrides keep the single projection precedence.
        let mut backend_context = self.llama_execution_config().backend_context;
        backend_context
            .apply_runtime_config_snapshot(snapshot)
            .map_err(|error| error.to_string())?;
        self.runtime.backend_context = backend_context;
        if runtime_config_value(snapshot, "FERRUM_CUDA_DEVICE").is_some()
            && matches!(self.backend.device, Device::CUDA(_))
        {
            self.backend.device = Device::CUDA(self.runtime.backend_context.device_ordinal);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_GPTQ_GIDX_TRACE") {
            self.runtime.gptq_gidx_trace = value == "1";
        }
        Ok(())
    }

    /// Freeze model construction options before opening or loading weights.
    pub fn llama_execution_config(&self) -> crate::LlamaExecutionConfig {
        let mut backend_context = self.runtime.backend_context.clone();
        // Existing explicit product fields remain authoritative during cold
        // assembly. The constructed backend reads only this frozen value.
        // CPU/Metal have no ordinal and must not inherit a CUDA environment
        // selection captured for another backend.
        backend_context.device_ordinal = self.backend.device.index().unwrap_or(0);
        if let Some(dtype) = &self.runtime.metal_dtype {
            backend_context.metal.prefer_f16_weights = dtype.eq_ignore_ascii_case("f16");
        }
        if let Some(world_size) = self.runtime.tp {
            backend_context.collective.world_size = world_size;
        }
        if let Some(capacity) = self.runtime.kv_capacity {
            backend_context.cuda.kv_capacity = capacity;
        }
        if backend_context.cuda.split_k_attn.is_none() {
            backend_context.cuda.split_k_attn = self.runtime.llama.split_k_attn;
        }
        // Preserve explicit conflicts for LlamaExecutionConfig::validate;
        // the model alias must not silently replace the backend policy.
        crate::LlamaExecutionConfig {
            backend_context,
            kv_capacity: self.runtime.kv_capacity,
            paged_max_seqs: self.scheduler.max_running_requests,
            prefix_cache: self.runtime.prefix_state_cache_enabled,
            batched_graph: self.backend.enable_cuda_graphs,
            options: self.runtime.llama.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineModelConfig {
    pub model_id: ModelId,
    pub model_info: Option<ModelInfo>,
    pub tokenizer: TokenizerConfig,
    /// Typed identity of the source requested by the product entrypoint.
    /// The resolved local path remains a runtime/backend concern, while this
    /// value preserves repository/revision provenance for product composition.
    #[serde(default)]
    pub source: Option<crate::ModelSource>,
}

impl Default for EngineModelConfig {
    fn default() -> Self {
        Self {
            model_id: ModelId::new("default"),
            model_info: None,
            tokenizer: TokenizerConfig::default(),
            source: None,
        }
    }
}

/// Latency targets used by the optional adaptive scheduler, in milliseconds.
/// These are scheduling hints; client-observed SLO compliance is measured
/// separately, including transport and queueing time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerSloConfig {
    #[serde(deserialize_with = "deserialize_slo_milliseconds")]
    pub ttft_ms: f64,
    #[serde(deserialize_with = "deserialize_slo_milliseconds")]
    pub tpot_ms: f64,
    #[serde(deserialize_with = "deserialize_slo_milliseconds")]
    pub itl_ms: f64,
    #[serde(default)]
    pub reference: crate::ReferenceEstimatorConfig,
    #[serde(default)]
    pub unknown_cost: crate::UnknownCostPolicy,
}

impl SchedulerSloConfig {
    pub fn validate(&self) -> std::result::Result<(), String> {
        self.reference.validate()?;
        self.unknown_cost.validate()?;
        for (name, value) in [
            ("ttft", self.ttft_ms),
            ("tpot", self.tpot_ms),
            ("itl", self.itl_ms),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "scheduler SLO {name} must be positive finite milliseconds"
                ));
            }
        }
        Ok(())
    }
}

impl std::str::FromStr for SchedulerSloConfig {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        let mut values = [None; 3];
        for item in raw.split(',') {
            let (name, value) = item.split_once(':').ok_or_else(|| {
                "expected scheduler SLO ttft:200,tpot:15,itl:50 (milliseconds)".to_owned()
            })?;
            let index = match name.trim() {
                "ttft" => 0,
                "tpot" => 1,
                "itl" => 2,
                other => return Err(format!("unknown scheduler SLO metric {other:?}")),
            };
            if values[index].is_some() {
                return Err(format!("duplicate scheduler SLO metric {:?}", name.trim()));
            }
            values[index] = Some(
                value
                    .trim()
                    .parse::<f64>()
                    .map_err(|error| format!("invalid scheduler SLO {}: {error}", name.trim()))?,
            );
        }
        let [Some(ttft_ms), Some(tpot_ms), Some(itl_ms)] = values else {
            return Err("scheduler SLO requires ttft, tpot, and itl thresholds".to_owned());
        };
        let slo = Self {
            ttft_ms,
            tpot_ms,
            itl_ms,
            reference: crate::ReferenceEstimatorConfig::default(),
            unknown_cost: crate::UnknownCostPolicy::default(),
        };
        slo.validate()?;
        Ok(slo)
    }
}

fn deserialize_slo_milliseconds<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = f64::deserialize(deserializer)?;
    if !value.is_finite() || value <= 0.0 {
        return Err(serde::de::Error::custom(
            "scheduler SLO thresholds must be positive finite milliseconds",
        ));
    }
    Ok(value)
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Optional feedback-driven latency targets. None retains static scheduling.
    #[serde(default)]
    pub slo: Option<SchedulerSloConfig>,
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Maximum waiting queue size
    pub max_waiting_requests: usize,
    /// Maximum running requests
    pub max_running_requests: usize,
    /// Enable request preemption
    pub enable_preemption: bool,
    /// Use prompt-token metadata for initial continuous-batch admission estimates.
    #[serde(default = "default_prompt_token_estimate")]
    pub prompt_token_estimate: bool,
    /// Prefer new prefills over early decodes until this many requests are active.
    #[serde(default)]
    pub prefill_first_until_active: Option<usize>,
    /// Opt in to one bounded wait for another request's exact in-flight prefix.
    /// Expiry releases the optional dependency and resumes cold admission.
    #[serde(default)]
    pub prefix_rendezvous_max_wait_ms: Option<std::num::NonZeroU64>,
    /// Optional hard cap for per-request prefill chunks. `None` spends the
    /// live per-step token budget and lets capacity feedback narrow or regrow
    /// each request independently.
    #[serde(default)]
    pub prefill_step_chunk: Option<usize>,
    /// Cap prefill admission chunks only while decode requests are already active.
    #[serde(default)]
    pub active_decode_prefill_chunk: Option<usize>,
    /// Limit total prefill tokens across one iteration that schedules runnable
    /// decode work. This is independent of the per-request chunk cap and does
    /// not restrict pure-prefill iterations. `None` or zero disables the limit.
    #[serde(default)]
    pub active_decode_prefill_token_budget: Option<usize>,
    /// Non-reserving sequence fit gate used before prefill admission.
    #[serde(default)]
    pub sequence_fit_policy: SequenceFitPolicy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            slo: None,
            policy: SchedulingPolicy::ContinuousBatch,
            max_waiting_requests: 1000,
            max_running_requests: 32,
            enable_preemption: true,
            prompt_token_estimate: default_prompt_token_estimate(),
            prefill_first_until_active: None,
            prefix_rendezvous_max_wait_ms: None,
            prefill_step_chunk: None,
            active_decode_prefill_chunk: None,
            active_decode_prefill_token_budget: None,
            sequence_fit_policy: SequenceFitPolicy::default(),
        }
    }
}

fn default_prompt_token_estimate() -> bool {
    true
}

impl SchedulerConfig {
    pub fn apply_runtime_config_snapshot(
        &mut self,
        snapshot: &RuntimeConfigSnapshot,
    ) -> std::result::Result<(), String> {
        snapshot
            .validate_retired_options()
            .map_err(|(key, reason)| format!("{key}: {reason}"))?;
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PREFIX_RENDEZVOUS_MAX_WAIT_MS")
        {
            self.prefix_rendezvous_max_wait_ms = Some(
                value
                    .parse::<std::num::NonZeroU64>()
                    .map_err(|error| format!("FERRUM_PREFIX_RENDEZVOUS_MAX_WAIT_MS: {error}"))?,
            );
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE") {
            self.prompt_token_estimate = parse_bool_env_value(value)
                .map_err(|reason| format!("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE: {reason}"))?;
        }
        if let Some(value) =
            runtime_config_value(snapshot, "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
        {
            self.prefill_first_until_active =
                parse_optional_positive_usize("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SCHED_PREFILL_STEP_CHUNK") {
            self.prefill_step_chunk =
                parse_optional_positive_usize("FERRUM_SCHED_PREFILL_STEP_CHUNK", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK") {
            self.active_decode_prefill_chunk =
                parse_optional_positive_usize("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK", value)?;
        }
        if let Some(value) =
            runtime_config_value(snapshot, "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET")
        {
            self.active_decode_prefill_token_budget =
                parse_optional_positive_usize("FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SEQUENCE_FIT_POLICY") {
            self.sequence_fit_policy = SequenceFitPolicy::parse_runtime_value(value)
                .map_err(|reason| format!("FERRUM_SEQUENCE_FIT_POLICY: {reason}"))?;
        }
        Ok(())
    }
}

fn runtime_config_value<'a>(snapshot: &'a RuntimeConfigSnapshot, key: &str) -> Option<&'a str> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| entry.effective_value.as_str())
}

fn parse_optional_positive_usize(
    key: &str,
    value: &str,
) -> std::result::Result<Option<usize>, String> {
    let parsed = parse_usize_env_value(value).map_err(|reason| format!("{key}: {reason}"))?;
    Ok((parsed > 0).then_some(parsed))
}

fn parse_required_positive_usize(key: &str, value: &str) -> std::result::Result<usize, String> {
    let parsed = parse_usize_env_value(value).map_err(|reason| format!("{key}: {reason}"))?;
    if parsed == 0 {
        Err(format!("{key}: must be greater than zero"))
    } else {
        Ok(parsed)
    }
}

fn parse_positive_usize_list(key: &str, value: &str) -> std::result::Result<Vec<usize>, String> {
    let values = value
        .split(',')
        .map(str::trim)
        .map(|value| parse_required_positive_usize(key, value))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.is_empty() {
        Err(format!("{key}: must contain at least one width"))
    } else {
        Ok(values)
    }
}

fn parse_profile_entrypoint(
    key: &str,
    value: &str,
) -> std::result::Result<ProfileEntrypoint, String> {
    ProfileEntrypoint::parse(value).ok_or_else(|| {
        format!("{key}: expected one of run, serve, bench_serve, synthetic; got {value:?}")
    })
}

fn parse_presence_bool(value: &str) -> std::result::Result<bool, String> {
    if value.trim().is_empty() {
        Ok(true)
    } else {
        parse_bool_env_value(value)
    }
}

/// Scheduling policies
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// Iteration-level continuous batching with preemption
    #[default]
    ContinuousBatch,
}

/// KV Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvCacheConfig {
    /// Cache implementation type
    pub cache_type: KvCacheType,
    /// Element dtype (Dim 5 polymorphism point). FP16 is the
    /// validated production path; INT8 / FP8 require a backend impl
    /// of `BackendKvDtype<KvInt8>` / `BackendKvDtype<KvFp8>` and a
    /// model wired through `KvCacheQuant<B, K>`.
    #[serde(default)]
    pub dtype: KvCacheDtype,
    /// Block size for paged attention
    pub block_size: usize,
    /// Maximum number of blocks
    pub max_blocks: usize,
    /// Enable cache compression
    pub enable_compression: bool,
    /// Compression ratio target
    pub compression_ratio: f32,
    /// Enable multi-level caching (GPU + CPU)
    pub enable_multi_level: bool,
    /// Swap threshold (when to move to CPU)
    pub swap_threshold: f32,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        // 2048 blocks covers c=32 ShareGPT prompts (~32×500/16 = 1000
        // blocks). The previous 1024 floor crashed at c≥16 on real
        // workloads with "Block pool exhausted". Runtime overrides are
        // applied through EngineConfig::apply_runtime_config_snapshot.
        Self {
            cache_type: KvCacheType::Contiguous,
            dtype: KvCacheDtype::default(),
            block_size: 16,
            max_blocks: 2048,
            enable_compression: false,
            compression_ratio: 0.5,
            enable_multi_level: true,
            swap_threshold: 0.8,
        }
    }
}

/// KV Cache implementation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KvCacheType {
    /// Simple contiguous memory allocation
    Contiguous,
    /// Paged attention with block-based allocation
    Paged,
    /// Tree-based cache for prefix sharing
    Tree,
}

/// KV Cache element dtype (Dim 5 polymorphism point).
///
/// Mirrors `ferrum_interfaces::kv_dtype::KvDtypeKind` markers but
/// lives here because `KvCacheConfig` is part of the user-facing
/// `EngineConfig` and needs `Serialize` / `Deserialize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheDtype {
    /// FP16 K/V — the validated production path on every backend.
    #[default]
    Fp16,
    /// BF16 K/V — same memory cost as FP16, slightly different precision.
    /// Marker only; no backend impl ships yet.
    Bf16,
    /// INT8 K/V. The selected execution plan declares the scale and layout ABI;
    /// vNext uses per-token/head F32 scales, while legacy CUDA uses F16 scales.
    /// This is an explicit lossy storage request, subject to model/backend
    /// support. Memory savings include scale, alignment and workspace overhead.
    Int8,
    /// FP8 (E4M3) K/V. Marker only; CUDA kernels pending.
    Fp8,
}

impl KvCacheDtype {
    /// Parse from a CLI / env-var string. Accepts fp16/f16/bf16/int8/fp8/f8e4m3.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp16" | "f16" | "float16" => Some(Self::Fp16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "int8" | "i8" => Some(Self::Int8),
            "fp8" | "f8" | "f8e4m3" | "e4m3" => Some(Self::Fp8),
            _ => None,
        }
    }

    /// Short label for display + telemetry.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
            Self::Fp8 => "fp8",
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory pool size in bytes
    pub pool_size: Option<usize>,
    /// Exact device-wide usable runtime budget in bytes. When present this
    /// overrides the pressure-threshold calculation without changing the raw
    /// device or pool capacity.
    #[serde(default)]
    pub usable_capacity_bytes: Option<usize>,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Memory alignment in bytes
    pub alignment: usize,
    /// Enable memory defragmentation
    pub enable_defragmentation: bool,
    /// Defragmentation threshold
    pub defragmentation_threshold: f32,
    /// Enable memory statistics tracking
    pub enable_memory_stats: bool,
    /// Memory pressure warning threshold
    pub pressure_warning_threshold: f32,
    /// Memory pressure critical threshold
    pub pressure_critical_threshold: f32,
}

/// Resolved device-wide memory budget consumed by runtime planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryCapacityBudget {
    pub capacity_bytes: u64,
    pub usable_capacity_bytes: u64,
    pub reserve_bytes: u64,
}

impl MemoryConfig {
    pub fn resolve_capacity_budget(
        &self,
        device_capacity_bytes: u64,
    ) -> std::result::Result<MemoryCapacityBudget, String> {
        if device_capacity_bytes == 0 {
            return Err("runtime device memory capacity must be greater than zero".to_string());
        }
        let capacity_bytes = self
            .pool_size
            .map(|bytes| bytes as u64)
            .unwrap_or(device_capacity_bytes)
            .min(device_capacity_bytes);
        if capacity_bytes == 0 {
            return Err("runtime memory capacity must be greater than zero".to_string());
        }

        let usable_capacity_bytes = if let Some(bytes) = self.usable_capacity_bytes {
            let bytes = bytes as u64;
            if bytes == 0 || bytes > capacity_bytes {
                return Err(format!(
                    "memory.usable_capacity_bytes must be in 1..={capacity_bytes}, got {bytes}"
                ));
            }
            bytes
        } else {
            let critical = self.pressure_critical_threshold;
            if !critical.is_finite() || critical <= 0.0 || critical > 1.0 {
                return Err(format!(
                    "memory.pressure_critical_threshold must be in (0, 1], got {critical}"
                ));
            }
            let threshold_bytes = ((capacity_bytes as f64) * f64::from(critical)).floor() as u64;
            capacity_bytes.saturating_sub(
                capacity_bytes
                    .saturating_sub(threshold_bytes)
                    .min(capacity_bytes - 1),
            )
        };
        Ok(MemoryCapacityBudget {
            capacity_bytes,
            usable_capacity_bytes,
            reserve_bytes: capacity_bytes - usable_capacity_bytes,
        })
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: None,
            usable_capacity_bytes: None,
            enable_pooling: true,
            alignment: 256,
            enable_defragmentation: false,
            defragmentation_threshold: 0.7,
            enable_memory_stats: true,
            pressure_warning_threshold: 0.8,
            pressure_critical_threshold: 0.95,
        }
    }
}

/// Non-configurable upper bound for startup decode capture width.
///
/// This bound is independent from scheduler admission. It protects startup
/// worker/resource creation even when a user supplies an extreme concurrency
/// value or an explicit capture list. Wider runtime batches remain valid and
/// use the documented eager fallback.
pub const MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH: usize = 32;

/// Default upper bound for automatically generated exact decode capture widths.
///
/// This is an independent startup-work safety bound, not a concurrency limit.
/// A resolver may automatically request every exact width through the minimum
/// of admission, this ceiling, and the hard startup bound. Admission above the
/// bound remains valid; wider runtime waves use eager fallback unless a future
/// independently bounded policy supports them.
pub const DEFAULT_MAXIMUM_AUTOMATIC_EXACT_DECODE_WIDTH: usize =
    MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH;

/// When a supported backend prepares reusable device programs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReusableExecutionPreparationMode {
    /// Select on-demand preparation when declared by the runtime, otherwise startup.
    #[default]
    Auto,
    /// Prepare and validate the configured matrix before accepting requests.
    Startup,
    /// Warm up with ordinary requests, then capture recurring shapes within the budget.
    OnDemand,
}

impl ReusableExecutionPreparationMode {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Startup => "startup",
            Self::OnDemand => "on_demand",
        }
    }

    pub fn resolve(self, on_demand_supported: bool) -> Result<Self, String> {
        match (self, on_demand_supported) {
            (Self::Auto, true) => Ok(Self::OnDemand),
            (Self::Auto, false) => Ok(Self::Startup),
            (Self::OnDemand, false) => Err(
                "runtime.reusable_execution_preparation=on_demand requires a runtime declaring on-demand reusable execution support".to_owned(),
            ),
            (mode, _) => Ok(mode),
        }
    }
}

impl std::str::FromStr for ReusableExecutionPreparationMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "auto" => Ok(Self::Auto),
            "startup" => Ok(Self::Startup),
            "on_demand" => Ok(Self::OnDemand),
            _ => {
                Err("reusable execution preparation must be auto, startup, or on_demand".to_owned())
            }
        }
    }
}

/// Product policy for bounded reusable-execution capture.
///
/// `None` requests an automatically resolved exact-width matrix. Explicit
/// widths let operators bound startup work deliberately; widths omitted from
/// that matrix remain eligible for the runtime's documented eager fallback.
/// Neither form may exceed
/// [`MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH`].
/// Validation against scheduler admission and backend capabilities belongs to
/// the runtime-policy resolver, where those resolved limits are available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReusableExecutionCaptureConfig {
    /// Preparation lifecycle, resolved from declared device capabilities.
    pub preparation: ReusableExecutionPreparationMode,
    /// Exact concurrent decode widths to prepare. `None` selects automatic
    /// resolution from the admitted runtime capacity.
    pub exact_decode_widths: Option<Vec<usize>>,
    /// Safety ceiling for automatic exact-width expansion. This does not cap
    /// user concurrency and may only lower the independent startup hard bound.
    pub maximum_automatic_exact_decode_width: usize,
}

impl Default for ReusableExecutionCaptureConfig {
    fn default() -> Self {
        Self {
            preparation: ReusableExecutionPreparationMode::default(),
            exact_decode_widths: None,
            maximum_automatic_exact_decode_width: DEFAULT_MAXIMUM_AUTOMATIC_EXACT_DECODE_WIDTH,
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend type
    pub backend_type: BackendType,
    /// Target device
    pub device: Device,
    /// Data type for computation
    pub dtype: DataType,
    /// Enable optimizations
    pub enable_optimizations: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable CUDA graphs
    pub enable_cuda_graphs: bool,
    /// Prepare backend-owned reusable device programs when supported.
    #[serde(default = "default_enable_reusable_execution")]
    pub enable_reusable_execution: bool,
    /// Typed reusable-program capture policy shared by every product
    /// entrypoint. This is deliberately not a backend option or hidden env
    /// bridge: the resolved execution policy must fingerprint its outcome.
    #[serde(default)]
    pub reusable_execution_capture: ReusableExecutionCaptureConfig,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Custom backend-specific options
    pub backend_options: HashMap<String, serde_json::Value>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Candle,
            device: Device::CPU,
            dtype: DataType::FP16,
            enable_optimizations: true,
            optimization_level: 2,
            enable_cuda_graphs: false,
            enable_reusable_execution: default_enable_reusable_execution(),
            reusable_execution_capture: ReusableExecutionCaptureConfig::default(),
            enable_kernel_fusion: true,
            backend_options: HashMap::new(),
        }
    }
}

const fn default_enable_reusable_execution() -> bool {
    true
}

/// Supported backend types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BackendType {
    /// Candle framework
    Candle,
    /// ONNX Runtime
    OnnxRuntime,
    /// TensorRT
    TensorRT,
    /// Custom backend
    Custom,
}

/// Shared tokenizer-instance cache policy, independent of request output credit.
/// Either zero disables caching; it never means an unlimited cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TokenizerDecodeCacheConfig {
    pub max_entries: usize,
    /// Retained token IDs (four bytes each) plus decoded UTF-8 text bytes.
    pub max_bytes: usize,
}

impl Default for TokenizerDecodeCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}

impl TokenizerDecodeCacheConfig {
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_bytes > isize::MAX as usize || self.max_entries > isize::MAX as usize {
            return Err(crate::FerrumError::config(
                "tokenizer decode cache limits exceed addressable storage",
            ));
        }
        Ok(())
    }
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub decode_cache: TokenizerDecodeCacheConfig,
    /// Tokenizer type
    pub tokenizer_type: TokenizerType,
    /// Path to tokenizer files
    pub tokenizer_path: Option<String>,
    /// Enable fast tokenization
    pub enable_fast: bool,
    /// Add special tokens
    pub add_special_tokens: bool,
    /// Truncation strategy
    pub truncation: Option<TruncationConfig>,
    /// Padding strategy
    pub padding: Option<PaddingConfig>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            decode_cache: TokenizerDecodeCacheConfig::default(),
            tokenizer_type: TokenizerType::BPE,
            tokenizer_path: None,
            enable_fast: true,
            add_special_tokens: true,
            truncation: None,
            padding: None,
        }
    }
}

/// Tokenizer algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerType {
    /// Byte Pair Encoding
    BPE,
    /// WordPiece tokenizer (BERT-style)
    WordPiece,
    /// SentencePiece tokenizer
    SentencePiece,
    /// Tiktoken tokenizer family
    Tiktoken,
    /// Any custom tokenizer implementation
    Custom,
}

/// Truncation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationConfig {
    /// Maximum length
    pub max_length: usize,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
}

/// Truncation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Remove from the beginning
    TruncateStart,
    /// Remove from the end
    TruncateEnd,
    /// Remove from both sides
    TruncateBoth,
}

/// Padding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingConfig {
    /// Padding strategy
    pub strategy: PaddingStrategy,
    /// Padding token ID
    pub token_id: u32,
    /// Target length
    pub target_length: Option<usize>,
}

/// Padding strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to maximum length in batch
    MaxLength,
    /// Pad to specific length
    FixedLength,
}

/// Sampling configuration presets

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable API authentication
    pub enable_auth: bool,
    /// API keys for authentication
    pub api_keys: Vec<String>,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Rate limit per client (requests per minute)
    pub rate_limit_rpm: u32,
    /// Enable content filtering
    pub enable_content_filter: bool,
    /// Maximum prompt length
    pub max_prompt_length: usize,
    /// Enable prompt validation
    pub enable_prompt_validation: bool,
    /// Allowed file extensions for uploads
    pub allowed_extensions: Vec<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_auth: false,
            api_keys: vec![],
            enable_rate_limiting: true,
            rate_limit_rpm: 60,
            enable_content_filter: false,
            max_prompt_length: 32768,
            enable_prompt_validation: true,
            allowed_extensions: vec!["txt".to_string(), "json".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SamplingConfig {
    pub default_params: SamplingParams,
    pub presets: SamplingPresets,
    pub enable_custom_processors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub export_interval: Duration,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            export_interval: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Physical execution policy for PlanRuntime prefill/decode scheduler batches.
    #[serde(default)]
    pub prefill_decode_execution: PrefillDecodeExecution,
    pub max_batch_size: usize,
    pub max_wait_ms: u64,
    pub enable_dynamic: bool,
    pub enable_continuous: bool,
    /// vLLM-style per-iteration token budget. The scheduler emits a
    /// mixed prefill+decode batch summing to at most this many Q
    /// tokens (decode = 1 each, prefill chunk = its chunk size).
    /// Default 2048. Runtime snapshots can override this with
    /// `FERRUM_MAX_BATCHED_TOKENS`, usually from the GPU autosizer or a
    /// named workload preset rather than a user hand-written env bundle.
    #[serde(default = "BatchConfig::default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,
}

impl BatchConfig {
    fn default_max_num_batched_tokens() -> usize {
        2048
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            prefill_decode_execution: PrefillDecodeExecution::default(),
            max_batch_size: 32,
            max_wait_ms: 8,
            enable_dynamic: true,
            enable_continuous: false,
            max_num_batched_tokens: Self::default_max_num_batched_tokens(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuntimeConfigEntry, RuntimeConfigSource};

    #[test]
    fn retired_native_overrides_reject_before_mutating_engine_config() {
        for key in [
            "FERRUM_MARLIN_TILE",
            "FERRUM_VLLM_MOE_THREAD_K",
            "FERRUM_VLLM_MOE_THREAD_N",
            "FERRUM_VLLM_MOE_BLOCKS_PER_SM",
        ] {
            for value in ["", "0", "-1", "128"] {
                for source in [RuntimeConfigSource::Env, RuntimeConfigSource::ConfigFile] {
                    let mut config = EngineConfig::default();
                    let before = serde_json::to_value(&config).unwrap();
                    let snapshot = RuntimeConfigSnapshot::from_entries([
                        RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "1024", source),
                        RuntimeConfigEntry::new(key, value, source),
                    ]);
                    let error = config.apply_runtime_config_snapshot(&snapshot).unwrap_err();
                    assert!(error.contains(key) && error.contains("retired"), "{error}");
                    assert_eq!(serde_json::to_value(&config).unwrap(), before);
                }
            }
        }
    }

    #[test]
    fn retired_legacy_moe_controls_reject_before_mutating_engine_config() {
        for key in [
            "FERRUM_MOE_GRAPH",
            "FERRUM_MOE_DEVICE_ROUTE",
            "FERRUM_MOE_HOST_TOPK",
            "FERRUM_MOE_FUSED_GATE_UP_SILU",
            "FERRUM_MOE_DIRECT_DISPATCH",
            "FERRUM_MOE_BATCHED",
            "FERRUM_MOE_BATCHED_DECODE",
            "FERRUM_MOE_BATCH_THRESHOLD",
            "FERRUM_MOE_BUCKETED",
            "FERRUM_MOE_PREFILL_THRESHOLD",
            "FERRUM_QWEN_UNIFIED_PREFILL",
            "FERRUM_QWEN_UNIFIED_TRACE",
            "FERRUM_UNIFIED_GREEDY_ARGMAX",
            "FERRUM_UNIFIED_LAYER_PROF",
            "FERRUM_UNIFIED_LAYER_PROF_EVERY",
            "FERRUM_UNIFIED_LAYER_PROF_MAX_M",
            "FERRUM_UNIFIED_LAYER_PROF_MIN_SEQS",
            "FERRUM_VLLM_DECODE_VARLEN",
            "FERRUM_VLLM_VARLEN_TILED_Q4",
        ] {
            for source in [RuntimeConfigSource::Env, RuntimeConfigSource::ConfigFile] {
                let mut config = EngineConfig::default();
                let before = serde_json::to_value(&config).unwrap();
                let snapshot = RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new("FERRUM_KV_CAPACITY", "1024", source),
                    RuntimeConfigEntry::new(key, "0", source),
                ]);
                let error = config.apply_runtime_config_snapshot(&snapshot).unwrap_err();
                assert!(error.contains(key) && error.contains("retired"), "{error}");
                assert_eq!(serde_json::to_value(&config).unwrap(), before);
            }
        }
        EngineConfig::default()
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([
                ("FERRUM_VLLM_MOE", "1"),
                ("FERRUM_VLLM_MOE_PAIR_IDS", "1"),
                ("FERRUM_MOE_HOST_ROUTE", "0"),
            ]))
            .unwrap();
    }

    #[test]
    fn sampling_readback_defaults_to_uniform_and_validates_typed_overrides() {
        let mut engine = EngineConfig::default();
        let mut legacy = serde_json::to_value(&engine.runtime).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("sampling_readback_mode");
        let restored: RuntimeKnobs = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            restored.sampling_readback_mode,
            SamplingReadbackMode::Uniform
        );
        for mode in [
            SamplingReadbackMode::Participant,
            SamplingReadbackMode::Uniform,
        ] {
            engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    "FERRUM_SAMPLING_READBACK_MODE",
                    mode.as_runtime_value(),
                )]))
                .unwrap();
            assert_eq!(engine.runtime.sampling_readback_mode, mode);
            engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                .unwrap();
            assert_eq!(engine.runtime.sampling_readback_mode, mode);
            assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
        }
        assert!(engine
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_SAMPLING_READBACK_MODE",
                "automatic"
            ),]))
            .unwrap_err()
            .contains("FERRUM_SAMPLING_READBACK_MODE"));
    }

    #[test]
    fn invocation_preparation_defaults_to_full_and_rejects_unknown_overrides() {
        let key = "FERRUM_INVOCATION_PREPARATION_STRATEGY";
        let mut config = EngineConfig::default();
        assert_eq!(
            config.runtime.invocation_preparation_strategy,
            InvocationPreparationStrategy::Full
        );
        let mut legacy = serde_json::to_value(&config.runtime).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("invocation_preparation_strategy");
        let restored: RuntimeKnobs = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            restored.invocation_preparation_strategy,
            InvocationPreparationStrategy::Full
        );
        for strategy in [
            InvocationPreparationStrategy::IdentityProjection,
            InvocationPreparationStrategy::DecodeSegment,
            InvocationPreparationStrategy::DecodeSegmentProgram,
            InvocationPreparationStrategy::ResidentDecodeProgram,
            InvocationPreparationStrategy::Full,
        ] {
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    key,
                    strategy.as_runtime_value(),
                )]))
                .unwrap();
            assert_eq!(config.runtime.invocation_preparation_strategy, strategy);
            assert_eq!(
                serde_json::to_value(strategy).unwrap(),
                strategy.as_runtime_value()
            );
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                .unwrap();
            assert_eq!(config.runtime.invocation_preparation_strategy, strategy);
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                key,
                "automatic"
            )]))
            .unwrap_err()
            .contains(key));
    }

    #[test]
    fn program_binding_upload_defaults_to_sparse_and_rejects_unknown_overrides() {
        let key = "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY";
        let mut config = EngineConfig::default();
        assert_eq!(
            config.runtime.program_binding_upload_strategy,
            ProgramBindingUploadStrategy::Sparse
        );
        let mut legacy = serde_json::to_value(&config.runtime).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("program_binding_upload_strategy");
        let restored: RuntimeKnobs = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            restored.program_binding_upload_strategy,
            ProgramBindingUploadStrategy::Sparse
        );
        for strategy in [
            ProgramBindingUploadStrategy::UniformLivePrefix,
            ProgramBindingUploadStrategy::CompactScatter,
            ProgramBindingUploadStrategy::Sparse,
        ] {
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    key,
                    strategy.as_runtime_value(),
                )]))
                .unwrap();
            assert_eq!(config.runtime.program_binding_upload_strategy, strategy);
            assert_eq!(
                serde_json::to_value(strategy).unwrap(),
                strategy.as_runtime_value()
            );
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                .unwrap();
            assert_eq!(config.runtime.program_binding_upload_strategy, strategy);
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                key,
                "automatic"
            )]))
            .unwrap_err()
            .contains(key));
    }

    #[test]
    fn completion_readback_policy_defaults_and_explicit_overrides() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let key = COMPLETION_READBACK_POLICY_CONFIG_KEY;
        let mut config = EngineConfig::default();
        assert_eq!(
            config.runtime.completion_readback_policy,
            CompletionReadbackPolicy::Synchronous
        );
        let mut old = serde_json::to_value(&config.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("completion_readback_policy");
        let restored: RuntimeKnobs = serde_json::from_value(old).unwrap();
        assert_eq!(
            restored.completion_readback_policy,
            CompletionReadbackPolicy::Synchronous
        );
        for policy in [
            CompletionReadbackPolicy::BatchedU32,
            CompletionReadbackPolicy::Synchronous,
        ] {
            for source in [RuntimeConfigSource::ConfigFile, RuntimeConfigSource::Cli] {
                let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    key,
                    policy.as_runtime_value(),
                    source,
                )]);
                config.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(config.runtime.completion_readback_policy, policy);
                assert_eq!(
                    serde_json::to_value(policy).unwrap(),
                    policy.as_runtime_value()
                );
            }
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                key,
                "batched-u32"
            ),]))
            .unwrap_err()
            .contains("not environment"));
        assert!(CompletionReadbackPolicy::parse_runtime_value("automatic").is_err());
    }

    #[test]
    fn segment_owner_mode_defaults_and_explicit_config_preserve_typed_values() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let key = SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY;
        let mut config = EngineConfig::default();
        assert_eq!(
            config.runtime.segment_binding_owner_view_mode,
            SegmentBindingOwnerViewMode::Legacy
        );
        let mut old = serde_json::to_value(&config.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("segment_binding_owner_view_mode");
        let restored: RuntimeKnobs = serde_json::from_value(old).unwrap();
        assert_eq!(
            restored.segment_binding_owner_view_mode,
            SegmentBindingOwnerViewMode::Legacy
        );
        for source in [RuntimeConfigSource::ConfigFile, RuntimeConfigSource::Cli] {
            for mode in [
                SegmentBindingOwnerViewMode::Indexed,
                SegmentBindingOwnerViewMode::Legacy,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    key,
                    mode.as_runtime_value(),
                    source,
                )]);
                config.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(config.runtime.segment_binding_owner_view_mode, mode);
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                config
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                    .unwrap();
                assert_eq!(config.runtime.segment_binding_owner_view_mode, mode);
            }
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "automatic", RuntimeConfigSource::Cli)
            ]))
            .is_err());
    }

    #[test]
    fn causal_decode_preparation_mode_defaults_and_explicit_config_preserve_typed_values() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let key = CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY;
        let mut config = EngineConfig::default();
        assert_eq!(
            config.runtime.causal_decode_preparation_mode,
            CausalDecodePreparationMode::PerParticipant
        );
        let mut old = serde_json::to_value(&config.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("causal_decode_preparation_mode");
        let restored: RuntimeKnobs = serde_json::from_value(old).unwrap();
        assert_eq!(
            restored.causal_decode_preparation_mode,
            CausalDecodePreparationMode::PerParticipant
        );
        for source in [RuntimeConfigSource::ConfigFile, RuntimeConfigSource::Cli] {
            for mode in [
                CausalDecodePreparationMode::Packed,
                CausalDecodePreparationMode::PerParticipant,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    key,
                    mode.as_runtime_value(),
                    source,
                )]);
                config.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(config.runtime.causal_decode_preparation_mode, mode);
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                config
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                    .unwrap();
                assert_eq!(config.runtime.causal_decode_preparation_mode, mode);
            }
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "automatic", RuntimeConfigSource::Cli)
            ]))
            .is_err());
    }

    #[test]
    fn causal_decode_attention_grouping_defaults_and_explicit_config_preserve_typed_values() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let key = CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY;
        let mut config = EngineConfig::default();
        assert_eq!(
            config.runtime.causal_decode_attention_grouping,
            CausalDecodeAttentionGrouping::Contiguous
        );
        let mut old = serde_json::to_value(&config.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("causal_decode_attention_grouping");
        let restored: RuntimeKnobs = serde_json::from_value(old).unwrap();
        assert_eq!(
            restored.causal_decode_attention_grouping,
            CausalDecodeAttentionGrouping::Contiguous
        );
        for source in [RuntimeConfigSource::ConfigFile, RuntimeConfigSource::Cli] {
            for mode in [
                CausalDecodeAttentionGrouping::Indexed,
                CausalDecodeAttentionGrouping::FixedCapacity,
                CausalDecodeAttentionGrouping::FixedExecutionCapacity,
                CausalDecodeAttentionGrouping::Contiguous,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    key,
                    mode.as_runtime_value(),
                    source,
                )]);
                config.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(config.runtime.causal_decode_attention_grouping, mode);
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                assert_eq!(
                    serde_json::from_value::<CausalDecodeAttentionGrouping>(
                        serde_json::to_value(mode).unwrap()
                    )
                    .unwrap(),
                    mode
                );
                config
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                    .unwrap();
                assert_eq!(config.runtime.causal_decode_attention_grouping, mode);
            }
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "automatic", RuntimeConfigSource::Cli)
            ]))
            .is_err());
    }

    #[test]
    fn segment_owner_mode_environment_cannot_activate_or_override_typed_config() {
        let key = SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY;
        let mut config = EngineConfig::default();
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                key, "indexed"
            )]))
            .unwrap_err()
            .contains("not environment"));
        assert_eq!(
            config.runtime.segment_binding_owner_view_mode,
            SegmentBindingOwnerViewMode::Legacy
        );
        config.runtime.segment_binding_owner_view_mode = SegmentBindingOwnerViewMode::Indexed;
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(key, "legacy")]))
            .is_err());
        assert_eq!(
            config.runtime.segment_binding_owner_view_mode,
            SegmentBindingOwnerViewMode::Indexed
        );
    }

    #[test]
    fn causal_decode_preparation_mode_environment_cannot_activate_or_override_typed_config() {
        let key = CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY;
        let mut config = EngineConfig::default();
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(key, "packed")]))
            .unwrap_err()
            .contains("not environment"));
        assert_eq!(
            config.runtime.causal_decode_preparation_mode,
            CausalDecodePreparationMode::PerParticipant
        );
        config.runtime.causal_decode_preparation_mode = CausalDecodePreparationMode::Packed;
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                key,
                "per-participant"
            )]))
            .is_err());
        assert_eq!(
            config.runtime.causal_decode_preparation_mode,
            CausalDecodePreparationMode::Packed
        );
    }

    #[test]
    fn causal_decode_attention_grouping_environment_cannot_activate_or_override_typed_config() {
        let key = CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY;
        for grouping in [
            CausalDecodeAttentionGrouping::Indexed,
            CausalDecodeAttentionGrouping::FixedCapacity,
            CausalDecodeAttentionGrouping::FixedExecutionCapacity,
        ] {
            let mut config = EngineConfig::default();
            assert!(config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    key,
                    grouping.as_runtime_value()
                )]))
                .unwrap_err()
                .contains("not environment"));
            assert_eq!(
                config.runtime.causal_decode_attention_grouping,
                CausalDecodeAttentionGrouping::Contiguous
            );
            config.runtime.causal_decode_attention_grouping = grouping;
            assert!(config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    key,
                    "contiguous"
                )]))
                .is_err());
            assert_eq!(config.runtime.causal_decode_attention_grouping, grouping);
        }
    }

    #[test]
    fn prefill_decode_execution_defaults_to_split_and_validates_runtime_overrides() {
        let mut config = EngineConfig::default();
        assert_eq!(
            config.batching.prefill_decode_execution,
            PrefillDecodeExecution::Split
        );
        let mut serialized = serde_json::to_value(&config.batching).unwrap();
        serialized
            .as_object_mut()
            .unwrap()
            .remove("prefill_decode_execution");
        let restored: BatchConfig = serde_json::from_value(serialized).unwrap();
        assert_eq!(
            restored.prefill_decode_execution,
            PrefillDecodeExecution::Split
        );
        for policy in [PrefillDecodeExecution::Mixed, PrefillDecodeExecution::Split] {
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    "FERRUM_PREFILL_DECODE_EXECUTION",
                    policy.as_runtime_value(),
                )]))
                .unwrap();
            assert_eq!(config.batching.prefill_decode_execution, policy);
            assert_eq!(
                serde_json::to_value(policy).unwrap(),
                policy.as_runtime_value()
            );
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                .unwrap();
            assert_eq!(config.batching.prefill_decode_execution, policy);
        }
        let error = config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_PREFILL_DECODE_EXECUTION",
                "automatic",
            )]))
            .unwrap_err();
        assert!(error.contains("FERRUM_PREFILL_DECODE_EXECUTION"));
    }

    #[test]
    fn prefix_state_cache_canonical_boolean_overrides_and_missing_key_preserves() {
        let mut config = EngineConfig::default();
        for (value, expected) in [("1", true), ("0", false), ("true", true), ("false", false)] {
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    "FERRUM_PREFIX_CACHE",
                    value,
                )]))
                .unwrap();
            assert_eq!(config.runtime.prefix_state_cache_enabled, expected);
            config
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                .unwrap();
            assert_eq!(config.runtime.prefix_state_cache_enabled, expected);
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_PREFIX_CACHE",
                "invalid"
            ),]))
            .is_err());
    }

    #[test]
    fn prefix_state_cache_defaults_on_supported_older_wire() {
        let mut config = EngineConfig::default();
        assert!(!config.runtime.prefix_state_cache_enabled);
        config.runtime.prefix_state_cache_enabled = true;
        config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_PREFIX_CACHE",
                "false",
            )]))
            .unwrap();
        assert!(!config.runtime.prefix_state_cache_enabled);
        let mut old = serde_json::to_value(&config.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("prefix_state_cache_enabled");
        let restored: RuntimeKnobs = serde_json::from_value(old).unwrap();
        assert!(!restored.prefix_state_cache_enabled);
    }

    #[test]
    fn scheduler_keeps_immediate_fit_default_until_full_input_policy_is_gated() {
        assert_eq!(
            SchedulerConfig::default().sequence_fit_policy,
            SequenceFitPolicy::ImmediateOnly
        );
    }

    #[test]
    fn scheduler_slo_requires_three_positive_finite_targets() {
        let slo: SchedulerSloConfig = "itl:50, ttft:200,tpot:15".parse().unwrap();
        assert_eq!(slo.ttft_ms, 200.0);
        assert_eq!(slo.tpot_ms, 15.0);
        assert_eq!(slo.itl_ms, 50.0);
        for invalid in [
            "ttft:200,tpot:15",
            "ttft:200,tpot:15,itl:50,itl:60",
            "ttft:200,tpot:15,unknown:50",
            "ttft:0,tpot:15,itl:50",
            "ttft:200,tpot:-1,itl:50",
            "ttft:200,tpot:15,itl:NaN",
            "ttft:inf,tpot:15,itl:50",
        ] {
            assert!(invalid.parse::<SchedulerSloConfig>().is_err(), "{invalid}");
        }
        let mut legacy = serde_json::to_value(SchedulerConfig::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("slo");
        assert!(serde_json::from_value::<SchedulerConfig>(legacy)
            .unwrap()
            .slo
            .is_none());
        assert!(
            serde_json::from_value::<SchedulerSloConfig>(serde_json::json!({
                "ttft_ms": 200, "tpot_ms": 15, "itl_ms": 0,
            }))
            .is_err()
        );
    }

    #[test]
    fn active_decode_prefill_token_budget_is_optional_in_existing_configs() {
        let default = SchedulerConfig::default();
        assert_eq!(default.active_decode_prefill_token_budget, None);
        let mut serialized = serde_json::to_value(default).unwrap();
        serialized
            .as_object_mut()
            .unwrap()
            .remove("active_decode_prefill_token_budget");
        let restored: SchedulerConfig = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored.active_decode_prefill_token_budget, None);
    }

    #[test]
    fn active_decode_prefill_token_budget_runtime_value_is_independent_of_chunk_cap() {
        let mut config = SchedulerConfig {
            active_decode_prefill_chunk: Some(11),
            ..SchedulerConfig::default()
        };
        config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET",
                "29",
            )]))
            .unwrap();
        assert_eq!(config.active_decode_prefill_token_budget, Some(29));
        assert_eq!(config.active_decode_prefill_chunk, Some(11));
        config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
            .unwrap();
        assert_eq!(config.active_decode_prefill_token_budget, Some(29));
        config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET",
                "0",
            )]))
            .unwrap();
        assert_eq!(config.active_decode_prefill_token_budget, None);
        assert_eq!(config.active_decode_prefill_chunk, Some(11));
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_ACTIVE_DECODE_PREFILL_TOKEN_BUDGET",
                "invalid",
            )]))
            .is_err());
    }

    #[test]
    fn sequence_fit_policy_uses_canonical_product_values() {
        assert_eq!(
            serde_json::to_string(&SequenceFitPolicy::FullInputMustFit).unwrap(),
            "\"full-input-must-fit\""
        );
        assert_eq!(
            serde_json::from_str::<SequenceFitPolicy>("\"immediate-only\"").unwrap(),
            SequenceFitPolicy::ImmediateOnly
        );
    }

    #[test]
    fn diagnostic_fault_uses_one_canonical_product_value() {
        assert_eq!(
            VNextDiagnosticFault::parse_runtime_value("prefill_resource_after_submit_once")
                .unwrap(),
            VNextDiagnosticFault::PrefillResourceAfterSubmitOnce
        );
        assert_eq!(
            VNextDiagnosticFault::PrefillResourceAfterSubmitOnce.as_runtime_value(),
            "prefill-resource-after-submit-once"
        );
        assert!(VNextDiagnosticFault::parse_runtime_value("resource-failure").is_err());
    }

    #[test]
    fn engine_config_applies_typed_diagnostic_fault() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([(
            "FERRUM_VNEXT_DIAGNOSTIC_FAULT",
            "prefill-resource-after-submit-once",
        )]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("runtime config should apply");

        assert_eq!(
            config.runtime.vnext_diagnostic_fault,
            Some(VNextDiagnosticFault::PrefillResourceAfterSubmitOnce)
        );
    }

    #[test]
    fn engine_config_rejects_unknown_diagnostic_fault() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([(
            "FERRUM_VNEXT_DIAGNOSTIC_FAULT",
            "resource-failure",
        )]);

        let error = config
            .apply_runtime_config_snapshot(&snapshot)
            .expect_err("unknown diagnostic fault must fail closed");

        assert!(error.contains("FERRUM_VNEXT_DIAGNOSTIC_FAULT"));
    }

    #[test]
    fn scheduler_deserialization_without_fit_policy_keeps_legacy_default() {
        let mut serialized = serde_json::to_value(SchedulerConfig::default()).unwrap();
        serialized
            .as_object_mut()
            .unwrap()
            .remove("sequence_fit_policy");

        let scheduler: SchedulerConfig = serde_json::from_value(serialized).unwrap();

        assert_eq!(
            scheduler.sequence_fit_policy,
            SequenceFitPolicy::ImmediateOnly
        );
    }

    #[test]
    fn checkpoint_capture_deserialization_keeps_decode_capture_disabled_by_default() {
        let capture: VNextCheckpointCaptureConfig = serde_json::from_value(serde_json::json!({
            "output_dir": "capture",
            "value_ids": ["value.output.logits"],
            "maximum_prefill_waves": 1
        }))
        .unwrap();

        assert_eq!(capture.maximum_decode_waves, 0);
        assert!(!capture.capture_product_output);
    }

    #[test]
    fn engine_config_applies_typed_sequence_fit_policy() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([(
            "FERRUM_SEQUENCE_FIT_POLICY",
            "full-input-must-fit",
        )]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("runtime config should apply");

        assert_eq!(
            config.scheduler.sequence_fit_policy,
            SequenceFitPolicy::FullInputMustFit
        );
    }

    #[test]
    fn engine_config_rejects_unknown_sequence_fit_policy() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([(
            "FERRUM_SEQUENCE_FIT_POLICY",
            "reserve-everything",
        )]);

        let error = config
            .apply_runtime_config_snapshot(&snapshot)
            .expect_err("unknown fit policy must fail closed");

        assert!(error.contains("FERRUM_SEQUENCE_FIT_POLICY"));
    }

    #[test]
    fn engine_config_rejects_retired_recurrent_state_slots_before_mutation() {
        for value in ["", "0", "16"] {
            let mut config = EngineConfig::default();
            let original = serde_json::to_value(&config).unwrap();
            let snapshot = RuntimeConfigSnapshot::from_env_vars([
                ("FERRUM_RECURRENT_STATE_MAX_SLOTS", value),
                ("FERRUM_KV_MAX_BLOCKS", "123"),
                ("FERRUM_PAGED_MAX_SEQS", "8"),
            ]);
            let error = config.apply_runtime_config_snapshot(&snapshot).unwrap_err();
            assert!(
                error.contains("FERRUM_RECURRENT_STATE_MAX_SLOTS"),
                "{error}"
            );
            assert!(error.contains("retired"), "{error}");
            assert_eq!(serde_json::to_value(&config).unwrap(), original);
        }
    }

    #[test]
    fn engine_config_does_not_apply_removed_qwen35_slot_alias() {
        let mut config = EngineConfig::default();
        let original = serde_json::to_value(&config).unwrap();
        let snapshot =
            RuntimeConfigSnapshot::from_env_vars([("FERRUM_QWEN35_LINEAR_STATE_MAX_SLOTS", "16")]);
        config.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(serde_json::to_value(&config).unwrap(), original);
    }

    #[test]
    fn engine_config_rejects_retired_recurrent_state_slots_in_serialized_runtime() {
        for value in [0, 16] {
            let mut document = serde_json::to_value(EngineConfig::default()).unwrap();
            document["runtime"]["recurrent_state_max_slots"] = serde_json::json!(value);
            let error = serde_json::from_value::<EngineConfig>(document).unwrap_err();
            assert!(error.to_string().contains("unknown field"), "{error}");
            assert!(
                error.to_string().contains("recurrent_state_max_slots"),
                "{error}"
            );
        }
    }

    #[test]
    fn engine_config_applies_profile_entrypoint_runtime_key() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_ENTRYPOINT", "run")]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("runtime config should apply");

        assert_eq!(
            config.runtime.profile_entrypoint,
            Some(ProfileEntrypoint::Run)
        );
    }

    #[test]
    fn engine_config_applies_typed_profile_detail_runtime_key() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_DETAIL", "full")]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("runtime config should apply");

        assert_eq!(
            config.runtime.profile_detail,
            ObservabilityProfileDetail::Full
        );
    }

    #[test]
    fn engine_config_applies_typed_latency_profile_detail_runtime_key() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_DETAIL", "latency")]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("latency profile detail should apply");

        assert_eq!(
            config.runtime.profile_detail,
            ObservabilityProfileDetail::Latency
        );
    }

    #[test]
    fn engine_config_applies_typed_replay_profile_detail_runtime_key() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_DETAIL", "replay")]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("runtime config should apply");

        assert_eq!(
            config.runtime.profile_detail,
            ObservabilityProfileDetail::Replay
        );
    }

    #[test]
    fn engine_config_applies_typed_verification_profile_detail_runtime_key() {
        let mut config = EngineConfig::default();
        let snapshot = RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_DETAIL", "verify")]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("runtime config should apply");

        assert_eq!(
            config.runtime.profile_detail,
            ObservabilityProfileDetail::Verify
        );
    }

    #[test]
    fn engine_config_applies_typed_profile_jsonl_runtime_key() {
        let mut config = EngineConfig::default();
        let snapshot =
            RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_JSONL", "/tmp/profile.jsonl")]);

        config
            .apply_runtime_config_snapshot(&snapshot)
            .expect("runtime config should apply");

        assert_eq!(
            config.runtime.profile_jsonl.as_deref(),
            Some(std::path::Path::new("/tmp/profile.jsonl"))
        );
    }

    #[test]
    fn engine_config_rejects_unknown_profile_detail() {
        let mut config = EngineConfig::default();
        let snapshot =
            RuntimeConfigSnapshot::from_env_vars([("FERRUM_PROFILE_DETAIL", "everything")]);

        let error = config
            .apply_runtime_config_snapshot(&snapshot)
            .expect_err("unknown profile detail must fail closed");

        assert!(error.contains("FERRUM_PROFILE_DETAIL"));
    }
    #[test]
    fn q4_mmvq_weight_decode_defaults_and_explicit_config_preserve_typed_values() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let key = Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let mut config = EngineConfig::default();
        assert_eq!(
            config.runtime.q4_mmvq_weight_decode,
            Q4MmvqWeightDecode::Baseline
        );
        let mut old = serde_json::to_value(&config.runtime).unwrap();
        old.as_object_mut().unwrap().remove("q4_mmvq_weight_decode");
        let restored: RuntimeKnobs = serde_json::from_value(old).unwrap();
        assert_eq!(restored.q4_mmvq_weight_decode, Q4MmvqWeightDecode::Baseline);
        for source in [RuntimeConfigSource::ConfigFile, RuntimeConfigSource::Cli] {
            for mode in [
                Q4MmvqWeightDecode::Auto,
                Q4MmvqWeightDecode::SharedM8,
                Q4MmvqWeightDecode::SharedM4M8,
                Q4MmvqWeightDecode::Baseline,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    key,
                    mode.as_runtime_value(),
                    source,
                )]);
                config.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(config.runtime.q4_mmvq_weight_decode, mode);
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                config
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                    .unwrap();
                assert_eq!(config.runtime.q4_mmvq_weight_decode, mode);
            }
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "automatic", RuntimeConfigSource::Cli)
            ]))
            .is_err());
    }
    #[test]
    fn q5_mmvq_weight_decode_defaults_and_explicit_config_preserve_typed_values() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let key = Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY;
        let mut config = EngineConfig::default();
        config.runtime.q4_mmvq_weight_decode = Q4MmvqWeightDecode::SharedM8;
        assert_eq!(
            config.runtime.q5_mmvq_weight_decode,
            Q5MmvqWeightDecode::Baseline
        );
        let mut old = serde_json::to_value(&config.runtime).unwrap();
        old.as_object_mut().unwrap().remove("q5_mmvq_weight_decode");
        let restored: RuntimeKnobs = serde_json::from_value(old).unwrap();
        assert_eq!(restored.q5_mmvq_weight_decode, Q5MmvqWeightDecode::Baseline);
        for source in [RuntimeConfigSource::ConfigFile, RuntimeConfigSource::Cli] {
            for mode in [
                Q5MmvqWeightDecode::Auto,
                Q5MmvqWeightDecode::SharedM8,
                Q5MmvqWeightDecode::SharedM4M8,
                Q5MmvqWeightDecode::Baseline,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    key,
                    mode.as_runtime_value(),
                    source,
                )]);
                config.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(config.runtime.q5_mmvq_weight_decode, mode);
                assert_eq!(
                    config.runtime.q4_mmvq_weight_decode,
                    Q4MmvqWeightDecode::SharedM8
                );
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                config
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::default())
                    .unwrap();
                assert_eq!(config.runtime.q5_mmvq_weight_decode, mode);
                assert_eq!(
                    config.runtime.q4_mmvq_weight_decode,
                    Q4MmvqWeightDecode::SharedM8
                );
            }
        }
        assert!(config
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new(key, "automatic", RuntimeConfigSource::Cli)
            ]))
            .is_err());
    }

    #[test]
    fn causal_decode_kv_reuse_defaults_and_explicit_sources() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let mut engine = EngineConfig::default();
        let key = CAUSAL_DECODE_KV_REUSE_CONFIG_KEY;
        assert_eq!(
            engine.runtime.causal_decode_kv_reuse,
            CausalDecodeKvReuse::None
        );
        let mut old = serde_json::to_value(&engine.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("causal_decode_kv_reuse");
        assert_eq!(
            serde_json::from_value::<RuntimeKnobs>(old)
                .unwrap()
                .causal_decode_kv_reuse,
            CausalDecodeKvReuse::None
        );
        for source in [RuntimeConfigSource::Cli, RuntimeConfigSource::ConfigFile] {
            for mode in [
                CausalDecodeKvReuse::PairedQueryHeads,
                CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2,
                CausalDecodeKvReuse::None,
            ] {
                engine
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                        RuntimeConfigEntry::new(key, mode.as_runtime_value(), source),
                    ]))
                    .unwrap();
                assert_eq!(engine.runtime.causal_decode_kv_reuse, mode);
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                assert_eq!(
                    serde_json::from_value::<CausalDecodeKvReuse>(
                        serde_json::to_value(mode).unwrap()
                    )
                    .unwrap(),
                    mode
                );
            }
        }
        for (value, source) in [
            ("paired-query-heads", RuntimeConfigSource::Env),
            ("paired-query-heads-unified-v2", RuntimeConfigSource::Env),
            ("none", RuntimeConfigSource::Env),
            ("auto", RuntimeConfigSource::Cli),
        ] {
            assert!(engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new(key, value, source),
                ]))
                .is_err());
        }
    }

    #[test]
    fn q8_projection_weight_preparation_defaults_and_explicit_sources() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let mut engine = EngineConfig::default();
        let key = Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY;
        assert_eq!(
            engine.runtime.q8_projection_weight_preparation,
            Q8ProjectionWeightPreparation::Compressed
        );
        let mut old = serde_json::to_value(&engine.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("q8_projection_weight_preparation");
        assert_eq!(
            serde_json::from_value::<RuntimeKnobs>(old)
                .unwrap()
                .q8_projection_weight_preparation,
            Q8ProjectionWeightPreparation::Compressed
        );
        engine.runtime.q5_mmvq_weight_decode = Q5MmvqWeightDecode::SharedM8;
        for source in [RuntimeConfigSource::Cli, RuntimeConfigSource::ConfigFile] {
            for mode in [
                Q8ProjectionWeightPreparation::Auto,
                Q8ProjectionWeightPreparation::PredecodedF32,
                Q8ProjectionWeightPreparation::Compressed,
            ] {
                engine
                    .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                        RuntimeConfigEntry::new(key, mode.as_runtime_value(), source),
                    ]))
                    .unwrap();
                assert_eq!(engine.runtime.q8_projection_weight_preparation, mode);
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                assert_eq!(
                    engine.runtime.q5_mmvq_weight_decode,
                    Q5MmvqWeightDecode::SharedM8
                );
            }
        }
        for (value, source) in [
            ("predecoded-f32", RuntimeConfigSource::Env),
            ("compressed", RuntimeConfigSource::Env),
            ("auto", RuntimeConfigSource::Env),
            ("automatic", RuntimeConfigSource::Cli),
        ] {
            assert!(engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new(key, value, source),
                ]))
                .is_err());
        }
    }

    #[test]
    fn sequence_backing_growth_preserves_defaults_and_explicit_sources() {
        use crate::{RuntimeConfigEntry, RuntimeConfigSource};
        let key = SEQUENCE_BACKING_GROWTH_CONFIG_KEY;
        let mut engine = EngineConfig::default();
        assert_eq!(
            engine.runtime.sequence_backing_growth,
            SequenceBackingGrowth::Exact
        );
        let mut old = serde_json::to_value(&engine.runtime).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("sequence_backing_growth");
        assert_eq!(
            serde_json::from_value::<RuntimeKnobs>(old)
                .unwrap()
                .sequence_backing_growth,
            SequenceBackingGrowth::Exact
        );
        for source in [RuntimeConfigSource::ConfigFile, RuntimeConfigSource::Cli] {
            for mode in [
                SequenceBackingGrowth::Exact,
                SequenceBackingGrowth::PagedGroup4,
                SequenceBackingGrowth::PagedGroup4GrowOnly,
            ] {
                let snapshot = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    key,
                    mode.as_runtime_value(),
                    source,
                )]);
                engine.apply_runtime_config_snapshot(&snapshot).unwrap();
                assert_eq!(engine.runtime.sequence_backing_growth, mode);
                assert_eq!(serde_json::to_value(mode).unwrap(), mode.as_runtime_value());
                assert_eq!(
                    serde_json::from_value::<SequenceBackingGrowth>(
                        serde_json::to_value(mode).unwrap()
                    )
                    .unwrap(),
                    mode
                );
            }
        }
        for (value, source) in [
            ("automatic", RuntimeConfigSource::Cli),
            ("paged-group4", RuntimeConfigSource::Env),
            ("paged-group4-grow-only", RuntimeConfigSource::Env),
        ] {
            assert!(engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_entries([
                    RuntimeConfigEntry::new(key, value, source)
                ]))
                .is_err());
        }
    }
}
