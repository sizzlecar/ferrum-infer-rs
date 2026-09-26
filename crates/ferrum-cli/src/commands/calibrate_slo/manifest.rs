use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

const MAX_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub schema_version: u32,
    /// Identity of the pinned source selection and prompt rendering procedure.
    pub input_preprocessing_sha256: [u8; 32],
    pub protocol: Protocol,
    /// Older v1 manifests remain diagnostic. Artifact validation is opt-in.
    #[serde(default)]
    pub validation_model: ValidationSource,
    /// Optional discovery/freeze/fresh singleton trials before the training cut.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<reference::ReferenceConfig>,
    /// Schema 2 only: recover an immutable selection and use product Chat conversion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharegpt: Option<super::sharegpt::FrozenShareGpt>,
    #[serde(default)]
    pub prompts: Vec<Prompt>,
    pub training: Vec<Cohort>,
    pub validation: Vec<Cohort>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ValidationSource {
    #[default]
    LiveFrozen,
    ExportedProfile {
        profile: PathBuf,
        source: PathBuf,
    },
    /// Training fits the model; these fresh cohorts calibrate its residual.
    /// Validation only queries the subsequently imported immutable artifact.
    SelectedWholeWaveV1 {
        export: ferrum_types::SloCostProfileExportConfig,
        residual: Vec<Cohort>,
    },
    /// New explicitly versioned profile7 protocol; no legacy evidence promotion.
    SelectedIndependentAttentionV2 {
        export: ferrum_types::SloCostProfileExportConfig,
        residual: Vec<Cohort>,
    },
    /// Profile8/source3: explicit work-only support with unchanged V2 selected families.
    SelectedWorkSupportV1 {
        export: ferrum_types::SloCostProfileExportConfig,
        residual: Vec<Cohort>,
    },
    /// Live, independently frozen fit/residual/qualification; exports schema 9.
    StructuredWholeWaveV1 {
        capture: structured::CaptureConfig,
        residual: Vec<Cohort>,
    },
    /// Independent observation only; no source members or model are created.
    StructuredDiscoveryV2 {
        #[serde(default)]
        warmup: Vec<Cohort>,
    },
    /// Read-only finite path requirements against genuine imported seed/reference.
    RequiredFutureAuditV2 {
        #[serde(default)]
        warmup: Vec<Cohort>,
        audit: required_audit::AuditConfigV2,
    },
    StructuredWholeWaveV2 {
        capture: structured_v2::CaptureConfigV2,
        residual: Vec<Cohort>,
    },
    /// One complete run, predeclared independent child sources, profile10 catalog.
    StructuredWholeWaveGroupV2 {
        capture: structured_v2::GroupCaptureConfigV2,
        residual: Vec<Cohort>,
    },
}

impl ValidationSource {
    pub(super) fn structured_group_v2(&self) -> Option<&structured_v2::GroupCaptureConfigV2> {
        match self {
            Self::StructuredWholeWaveGroupV2 { capture, .. } => Some(capture),
            _ => None,
        }
    }
    pub(super) fn required_audit(&self) -> Option<&required_audit::AuditConfigV2> {
        match self {
            Self::RequiredFutureAuditV2 { audit, .. } => Some(audit),
            _ => None,
        }
    }
    pub(super) fn is_discovery_v2(&self) -> bool {
        matches!(self, Self::StructuredDiscoveryV2 { .. })
    }
    pub(super) fn warmup_v2(&self) -> &[Cohort] {
        match self {
            Self::StructuredDiscoveryV2 { warmup } | Self::RequiredFutureAuditV2 { warmup, .. } => {
                warmup
            }
            Self::StructuredWholeWaveV2 { capture, .. } => &capture.warmup,
            Self::StructuredWholeWaveGroupV2 { capture, .. } => &capture.warmup,
            _ => &[],
        }
    }
    pub(super) fn structured_v2(&self) -> Option<&structured_v2::CaptureConfigV2> {
        match self {
            Self::StructuredWholeWaveV2 { capture, .. } => Some(capture),
            _ => None,
        }
    }
    pub(super) fn is_structured(&self) -> bool {
        self.structured().is_some()
            || self.structured_v2().is_some()
            || self.structured_group_v2().is_some()
    }
    pub(super) fn structured(&self) -> Option<&structured::CaptureConfig> {
        match self {
            Self::StructuredWholeWaveV1 { capture, .. } => Some(capture),
            _ => None,
        }
    }
    pub(super) fn selected(
        &self,
    ) -> Option<(
        ferrum_types::SloCostPredictor,
        &ferrum_types::SloCostProfileExportConfig,
        &[Cohort],
    )> {
        match self {
            Self::SelectedWholeWaveV1 { export, residual } => Some((
                ferrum_types::SloCostPredictor::SelectedWholeWaveV1,
                export,
                residual,
            )),
            Self::SelectedIndependentAttentionV2 { export, residual } => Some((
                ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2,
                export,
                residual,
            )),
            Self::SelectedWorkSupportV1 { export, residual } => Some((
                ferrum_types::SloCostPredictor::SelectedWorkSupportV1,
                export,
                residual,
            )),
            _ => None,
        }
    }
    pub(super) fn selected_kind(&self) -> Option<&'static str> {
        match self {
            Self::SelectedWholeWaveV1 { .. } => Some("selected_whole_wave_v1"),
            Self::SelectedIndependentAttentionV2 { .. } => {
                Some("selected_independent_attention_v2")
            }
            Self::SelectedWorkSupportV1 { .. } => Some("selected_work_support_v1"),
            _ => None,
        }
    }
    pub(super) fn residual(&self) -> &[Cohort] {
        match self {
            Self::SelectedWholeWaveV1 { residual, .. }
            | Self::SelectedIndependentAttentionV2 { residual, .. }
            | Self::SelectedWorkSupportV1 { residual, .. }
            | Self::StructuredWholeWaveV1 { residual, .. }
            | Self::StructuredWholeWaveGroupV2 { residual, .. }
            | Self::StructuredWholeWaveV2 { residual, .. } => residual,
            _ => &[],
        }
    }
    pub(super) fn destinations(&self) -> Option<(&std::path::Path, &std::path::Path)> {
        match self {
            Self::ExportedProfile { profile, source } => Some((profile, source)),
            Self::SelectedWholeWaveV1 { export, .. }
            | Self::SelectedIndependentAttentionV2 { export, .. }
            | Self::SelectedWorkSupportV1 { export, .. } => {
                Some((&export.path, &export.observations_path))
            }
            Self::StructuredWholeWaveV1 { capture, .. } => {
                Some((&capture.profile, &capture.source))
            }
            Self::StructuredWholeWaveV2 { capture, .. } => {
                Some((&capture.profile, &capture.source))
            }
            Self::LiveFrozen
            | Self::StructuredWholeWaveGroupV2 { .. }
            | Self::StructuredDiscoveryV2 { .. }
            | Self::RequiredFutureAuditV2 { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Protocol {
    /// Manual Structured V2 membership read only; omission preserves the legacy allowance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_prepared_projection_budget_us:
        Option<ferrum_engine::continuous_engine::StructuredPreparedProjectionBudgetV2>,
    pub total_timeout_ms: NonZeroU64,
    pub shutdown_timeout_ms: NonZeroU64,
    pub maximum_wave_attempts: NonZeroU64,
    pub maximum_raw_bytes: NonZeroU64,
    pub maximum_requests: NonZeroUsize,
    pub output: Codec,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Codec {
    CliText,
    CompletionsSse { include_usage: bool },
    ChatSse { include_usage: bool },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Prompt {
    pub source_id: String,
    pub rendered_prompt: String,
    pub rendered_prompt_sha256: [u8; 32],
    pub sampling: ferrum_types::SamplingParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Cohort {
    /// Ordered indices into prompts; duplicates represent distinct real owners.
    pub prompts: Vec<usize>,
    /// Refill only after a real credited consumer has observed terminal wire
    /// and successful completion. None preserves the original cohort barrier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolling_window: Option<RollingWindow>,
    /// Explicit calibration-only cycles, advanced by corresponding successfully
    /// reconciled waves. Omission keeps the original static case choices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wave_plan: Option<WavePlan>,
    pub repetitions: NonZeroUsize,
    pub prefill_chunk_tokens: NonZeroU32,
    pub execution: Execution,
    /// Diagnostic route coverage; FullLogits executes the real CPU sampler
    /// without changing token policy, history or the declared output budget.
    #[serde(default)]
    pub decode_route: ferrum_engine::continuous_engine::CalibrationDecodeRoute,
    /// Calibration-only upload-residency operation before each repetition.
    /// It is not a device/driver cache reset or a full cold-start claim.
    #[serde(default)]
    pub token_policy_residency: TokenPolicyResidencyPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WavePlan {
    /// Repeats by successful prefill-wave ordinal, never request or attempt count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_chunks: Option<Vec<NonZeroU32>>,
    /// Repeats independently by successful decode-wave ordinal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_routes: Option<Vec<ferrum_engine::continuous_engine::CalibrationDecodeRoute>>,
}

impl WavePlan {
    pub(super) fn validate(&self) -> Result<()> {
        let bounded = |len| (1..=16).contains(&len);
        if (self.prefill_chunks.is_none() && self.decode_routes.is_none())
            || self.prefill_chunks.as_ref().is_some_and(|values| {
                !bounded(values.len()) || values.iter().any(|value| value.get() > 1_048_576)
            })
            || self
                .decode_routes
                .as_ref()
                .is_some_and(|values| !bounded(values.len()))
        {
            return Err(FerrumError::config(
                "wave plan needs one or two nonempty cycles of at most 16 bounded choices",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RollingWindow {
    /// Bounds requests whose terminal output and completion have not both been
    /// consumed. Engine admission, allocator and output credits remain separate.
    pub maximum_in_flight: NonZeroUsize,
}

impl Cohort {
    pub(super) fn maximum_in_flight(&self) -> usize {
        self.rolling_window
            .map_or(self.prompts.len(), |window| window.maximum_in_flight.get())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TokenPolicyResidencyPolicy {
    #[default]
    Preserve,
    InvalidateBeforeCohort,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Execution {
    Split,
    Mixed,
}

pub(super) fn load(path: &std::path::Path) -> Result<Manifest> {
    let file = std::fs::File::open(path)
        .map_err(|error| FerrumError::config(format!("open calibration manifest: {error}")))?;
    let mut bytes = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| FerrumError::config(format!("read calibration manifest: {error}")))?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(FerrumError::config("calibration manifest exceeds 8 MiB"));
    }
    let mut manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|error| FerrumError::config(format!("parse calibration manifest: {error}")))?;
    manifest.validate()?;
    let absolute = path
        .canonicalize()
        .map_err(|error| FerrumError::config(format!("resolve calibration manifest: {error}")))?;
    let parent = absolute
        .parent()
        .ok_or_else(|| FerrumError::config("calibration manifest has no parent"))?;
    if let Some(source) = &mut manifest.sharegpt {
        source.resolve_paths(parent);
    }
    if let Some(reference) = &mut manifest.reference {
        reference.resolve_paths(parent);
    }
    if let ValidationSource::StructuredWholeWaveGroupV2 { capture, .. } =
        &mut manifest.validation_model
    {
        capture.resolve_paths(parent);
    }
    let destinations = match &mut manifest.validation_model {
        ValidationSource::ExportedProfile { profile, source } => Some((profile, source)),
        ValidationSource::SelectedWholeWaveV1 { export, .. }
        | ValidationSource::SelectedIndependentAttentionV2 { export, .. }
        | ValidationSource::SelectedWorkSupportV1 { export, .. } => {
            Some((&mut export.path, &mut export.observations_path))
        }
        ValidationSource::StructuredWholeWaveV1 { capture, .. } => {
            Some((&mut capture.profile, &mut capture.source))
        }
        ValidationSource::StructuredWholeWaveV2 { capture, .. } => {
            Some((&mut capture.profile, &mut capture.source))
        }
        ValidationSource::LiveFrozen
        | ValidationSource::StructuredWholeWaveGroupV2 { .. }
        | ValidationSource::StructuredDiscoveryV2 { .. }
        | ValidationSource::RequiredFutureAuditV2 { .. } => None,
    };
    if let Some((profile, source)) = destinations {
        for target in [profile, source] {
            if target.is_relative() {
                *target = parent.join(&*target);
            }
        }
    }
    Ok(manifest)
}

impl Manifest {
    pub(super) fn cohorts(&self) -> impl Iterator<Item = &Cohort> {
        self.training
            .iter()
            .chain(self.validation_model.residual())
            .chain(&self.validation)
            .chain(self.validation_model.warmup_v2())
    }

    pub(super) fn validate(&self) -> Result<()> {
        let invalid = |message| Err(FerrumError::config(message));
        let p = &self.protocol;
        if p.structured_prepared_projection_budget_us.is_some()
            && self.validation_model.structured_v2().is_none()
            && self.validation_model.structured_group_v2().is_none()
        {
            return invalid(
                "Prepared diagnostic budget is only valid for single/group Structured V2 capture",
            );
        }
        if let Some((profile, source)) = self.validation_model.destinations() {
            if profile.as_os_str().is_empty() || source.as_os_str().is_empty() || profile == source
            {
                return invalid(
                    "exported validation profile and source need distinct nonempty paths",
                );
            }
        }
        if let Some((_, export, residual)) = self.validation_model.selected() {
            export.validate().map_err(FerrumError::config)?;
            if residual.is_empty() || residual.len() > 256 {
                return invalid(
                    "selected whole-wave calibration requires bounded independent residual cohorts",
                );
            }
            if self.reference.is_some() {
                return invalid("collect the reference separately before selected whole-wave calibration; its source must not be relabeled as the later fit/residual capture");
            }
        }
        if let Some(capture) = self.validation_model.structured() {
            capture.validate(self)?;
        }
        if let Some(capture) = self.validation_model.structured_v2() {
            capture.validate(self)?;
        }
        if let Some(capture) = self.validation_model.structured_group_v2() {
            capture.validate(self)?;
        }
        let audit = self.validation_model.required_audit();
        if let Some(audit) = audit {
            audit.validate(self)?;
        }
        let discovery = self.validation_model.is_discovery_v2() || audit.is_some();
        if discovery && (self.reference.is_some() || !self.validation.is_empty()) {
            return invalid("structured discovery needs independent discovery cohorts without reference or validation populations");
        }
        if self.validation_model.warmup_v2().len() > 256 {
            return invalid("structured warmup exceeds 256 cohorts");
        }
        if self.input_preprocessing_sha256 == [0; 32] {
            return invalid("calibration requires a nonzero preprocessing digest");
        }
        let prompt_count = match (self.schema_version, &self.sharegpt) {
            (1, None) if !matches!(p.output, Codec::ChatSse { .. }) => self.prompts.len(),
            (2, Some(source)) if self.prompts.is_empty() && matches!(p.output, Codec::ChatSse { include_usage: true }) => {
                source.validate()?;
                // Actual selection length is checked before engine construction.
                4096
            }
            _ => return invalid("schema 1 needs rendered inputs; schema 2 needs only frozen ShareGPT and Chat SSE with usage"),
        };
        if p.total_timeout_ms.get() > 86_400_000
            || p.shutdown_timeout_ms.get() > 600_000
            || p.maximum_wave_attempts.get() > 1_000_000
            || p.maximum_raw_bytes.get() > 256 * 1024 * 1024
            || p.maximum_requests.get() > 256
        {
            return invalid("calibration protocol exceeds its hard resource bounds");
        }
        if prompt_count == 0
            || prompt_count > 4096
            || self.training.is_empty()
            || (!discovery && self.validation.is_empty())
            || self.training.len() > 256
            || self.validation.len() > 256
        {
            return invalid(
                "calibration needs bounded nonempty training and held-out validation cases",
            );
        }
        for prompt in &self.prompts {
            if prompt.source_id.is_empty()
                || prompt.source_id.len() > 1024
                || prompt.rendered_prompt.is_empty()
                || prompt.rendered_prompt.len() > 1024 * 1024
                || <[u8; 32]>::from(Sha256::digest(prompt.rendered_prompt.as_bytes()))
                    != prompt.rendered_prompt_sha256
                || prompt.sampling.max_tokens == 0
                || prompt.sampling.max_tokens > 1_048_576
            {
                return invalid("invalid pinned prompt, digest or full output-token limit");
            }
            prompt.sampling.validate()?;
        }
        let mut owners = 0usize;
        for case in self.cohorts() {
            if let Some(plan) = &case.wave_plan {
                plan.validate()?;
                if self.reference.is_some() {
                    return invalid("wave plans require a separate static reference manifest");
                }
            }
            if case.prompts.is_empty()
                || case.prompts.len() > 65_536
                || case.maximum_in_flight() > p.maximum_requests.get()
                || case.repetitions.get() > 64
                || case.prefill_chunk_tokens.get() > 1_048_576
                || case.prompts.iter().any(|index| *index >= prompt_count)
            {
                return invalid(
                    "invalid cohort arrival width, prompt index, repetition or chunk bound",
                );
            }
            owners = owners
                .checked_add(
                    case.prompts
                        .len()
                        .checked_mul(case.repetitions.get())
                        .ok_or_else(|| FerrumError::config("calibration owner count overflow"))?,
                )
                .ok_or_else(|| FerrumError::config("calibration owner count overflow"))?;
        }
        if owners > 65_536 {
            return invalid("calibration exceeds 65536 fresh owners");
        }
        if let Some(reference) = &self.reference {
            if reference.warmup.iter().any(|case| case.wave_plan.is_some()) {
                return invalid("reference warmup does not support a wave plan");
            }
            reference.validate_shape(self, prompt_count)?;
        }
        Ok(())
    }
}
