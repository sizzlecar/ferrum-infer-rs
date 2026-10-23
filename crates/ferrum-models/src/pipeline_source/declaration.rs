use ferrum_types::{FerrumError, Result};
use serde::{Deserialize, Serialize};

/// Pipeline identity comes from the package declaration, never its repo name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineArchitecture {
    QwenImage21,
}

impl PipelineArchitecture {
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::QwenImage21 => "QwenImage21Pipeline",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRole {
    Processor,
    ConditionEncoder,
    Denoiser,
    LatentCodec,
    Scheduler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentArchitecture {
    Qwen3VlProcessor,
    Qwen3VlConditionEncoder,
    QwenImage21Denoiser,
    QwenImage21LatentCodec,
    FlowMatchEulerDiscreteScheduler,
}

impl ComponentArchitecture {
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Qwen3VlProcessor => "Qwen3VLProcessor",
            Self::Qwen3VlConditionEncoder => "Qwen3VLForConditionalGeneration",
            Self::QwenImage21Denoiser => "QwenImage21Transformer2DModel",
            Self::QwenImage21LatentCodec => "AutoencoderKLQwenImage21",
            Self::FlowMatchEulerDiscreteScheduler => "FlowMatchEulerDiscreteScheduler",
        }
    }

    pub const fn library(self) -> &'static str {
        match self {
            Self::Qwen3VlProcessor | Self::Qwen3VlConditionEncoder => "transformers",
            _ => "diffusers",
        }
    }
}

/// Checked role/architecture pair. Paths are derived from the registered
/// package layout, not supplied by an untrusted model_index string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ComponentDeclaration {
    role: ComponentRole,
    architecture: ComponentArchitecture,
    directory: &'static str,
    config_path: &'static str,
}

impl ComponentDeclaration {
    /// Stable component instance identity within the registered pipeline.
    /// Current package layouts use their component directory as the ID;
    /// several distinct instances may share one role or architecture.
    pub const fn id(&self) -> &'static str {
        self.directory
    }

    pub(super) const fn registered(
        role: ComponentRole,
        architecture: ComponentArchitecture,
        directory: &'static str,
        config_path: &'static str,
    ) -> Self {
        Self {
            role,
            architecture,
            directory,
            config_path,
        }
    }

    pub const fn role(&self) -> ComponentRole {
        self.role
    }

    pub const fn architecture(&self) -> ComponentArchitecture {
        self.architecture
    }

    pub const fn directory(&self) -> &'static str {
        self.directory
    }

    pub const fn config_path(&self) -> &'static str {
        self.config_path
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PipelineDeclaration {
    architecture: PipelineArchitecture,
    components: Vec<ComponentDeclaration>,
    saved_sigmas: Option<Vec<f64>>,
}

// Exact metadata fields currently understood by this source adapter. Reject
// unknown components rather than silently omitting part of a model pipeline.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelIndex {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    _diffusers_version: Option<String>,
    #[serde(rename = "_name_or_path")]
    _name_or_path: Option<String>,
    processor: Option<[String; 2]>,
    text_encoder: Option<[String; 2]>,
    transformer: Option<[String; 2]>,
    vae: Option<[String; 2]>,
    scheduler: Option<[String; 2]>,
    sample_sigmas: Option<Vec<f64>>,
}

impl PipelineDeclaration {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let index: ModelIndex = serde_json::from_slice(bytes)
            .map_err(|error| FerrumError::model(format!("invalid model_index.json: {error}")))?;
        let architecture = match index.class_name.as_str() {
            "QwenImage21Pipeline" => PipelineArchitecture::QwenImage21,
            other => {
                return Err(FerrumError::unsupported(format!(
                    "unregistered pipeline architecture {other:?}"
                )))
            }
        };
        let entries = match architecture {
            PipelineArchitecture::QwenImage21 => [
                (
                    ComponentRole::ConditionEncoder,
                    ComponentArchitecture::Qwen3VlConditionEncoder,
                    "text_encoder",
                    "text_encoder/config.json",
                    index.text_encoder,
                ),
                (
                    ComponentRole::Denoiser,
                    ComponentArchitecture::QwenImage21Denoiser,
                    "transformer",
                    "transformer/config.json",
                    index.transformer,
                ),
                (
                    ComponentRole::LatentCodec,
                    ComponentArchitecture::QwenImage21LatentCodec,
                    "vae",
                    "vae/config.json",
                    index.vae,
                ),
                (
                    ComponentRole::Scheduler,
                    ComponentArchitecture::FlowMatchEulerDiscreteScheduler,
                    "scheduler",
                    "scheduler/scheduler_config.json",
                    index.scheduler,
                ),
                (
                    ComponentRole::Processor,
                    ComponentArchitecture::Qwen3VlProcessor,
                    "processor",
                    "processor/processor_config.json",
                    index.processor,
                ),
            ],
        };
        let mut components = Vec::with_capacity(entries.len());
        for (role, architecture, directory, config_path, declared) in entries {
            let [library, class] = declared.ok_or_else(|| {
                FerrumError::model(format!(
                    "model_index.json is missing required component {directory:?}"
                ))
            })?;
            if library != architecture.library() || class != architecture.class_name() {
                return Err(FerrumError::unsupported(format!(
                    "unsupported {directory} declaration [{library:?}, {class:?}]; expected [{:?}, {:?}]",
                    architecture.library(), architecture.class_name()
                )));
            }
            components.push(ComponentDeclaration::registered(
                role,
                architecture,
                directory,
                config_path,
            ));
        }
        if let Some(sigmas) = &index.sample_sigmas {
            if sigmas.is_empty()
                || sigmas
                    .iter()
                    .any(|sigma| !sigma.is_finite() || *sigma <= 0.0 || *sigma > 1.0)
                || sigmas.windows(2).any(|pair| pair[0] <= pair[1])
            {
                return Err(FerrumError::model(
                    "model_index.json sample_sigmas must be non-empty, strictly decreasing and in (0, 1]",
                ));
            }
        }
        Ok(Self {
            architecture,
            components,
            saved_sigmas: index.sample_sigmas,
        })
    }

    pub const fn architecture(&self) -> PipelineArchitecture {
        self.architecture
    }

    pub fn components(&self) -> &[ComponentDeclaration] {
        &self.components
    }

    /// Checkpoint schedule only. The domain Solver owns terminal sigma,
    /// timestep conversion and any explicit request override policy.
    pub fn saved_sigmas(&self) -> Option<&[f64]> {
        self.saved_sigmas.as_deref()
    }
}
