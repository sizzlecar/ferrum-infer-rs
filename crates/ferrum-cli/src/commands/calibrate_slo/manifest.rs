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
}

impl ValidationSource {
    pub(super) fn residual(&self) -> &[Cohort] {
        match self {
            Self::SelectedWholeWaveV1 { residual, .. } => residual,
            _ => &[],
        }
    }
    pub(super) fn destinations(&self) -> Option<(&std::path::Path, &std::path::Path)> {
        match self {
            Self::ExportedProfile { profile, source } => Some((profile, source)),
            Self::SelectedWholeWaveV1 { export, .. } => {
                Some((&export.path, &export.observations_path))
            }
            Self::LiveFrozen => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Protocol {
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
    let destinations = match &mut manifest.validation_model {
        ValidationSource::ExportedProfile { profile, source } => Some((profile, source)),
        ValidationSource::SelectedWholeWaveV1 { export, .. } => {
            Some((&mut export.path, &mut export.observations_path))
        }
        ValidationSource::LiveFrozen => None,
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
    }

    pub(super) fn validate(&self) -> Result<()> {
        let invalid = |message| Err(FerrumError::config(message));
        let p = &self.protocol;
        if let Some((profile, source)) = self.validation_model.destinations() {
            if profile.as_os_str().is_empty() || source.as_os_str().is_empty() || profile == source
            {
                return invalid(
                    "exported validation profile and source need distinct nonempty paths",
                );
            }
        }
        if let ValidationSource::SelectedWholeWaveV1 { export, residual } = &self.validation_model {
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
            || self.validation.is_empty()
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
            if case.prompts.is_empty()
                || case.prompts.len() > p.maximum_requests.get()
                || case.repetitions.get() > 64
                || case.prefill_chunk_tokens.get() > 1_048_576
                || case.prompts.iter().any(|index| *index >= prompt_count)
            {
                return invalid(
                    "invalid exact cohort width, prompt index, repetition or chunk bound",
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
            reference.validate_shape(self, prompt_count)?;
        }
        Ok(())
    }
}
