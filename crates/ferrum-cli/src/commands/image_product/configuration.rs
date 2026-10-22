use anyhow::{ensure, Result};
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits, StaticInitializationPolicy};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProductConfig {
    /// Public identity only; it does not select a model implementation.
    pub model: String,
    /// Explicit selection of a cold functional execution mode, not G3 qualification.
    pub execution_profile: ImageExecutionProfile,
    pub metadata: MetadataPaths,
    pub weights: WeightPaths,
    pub backend: ImageBackendConfig,
    /// Required by serve; run uses the same assets/backend without opening HTTP.
    pub serving: Option<ServingConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageExecutionProfile {
    ExperimentalSerialCold,
    ExperimentalBatchedCold,
}
impl ImageExecutionProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExperimentalSerialCold => "experimental_serial_cold",
            Self::ExperimentalBatchedCold => "experimental_batched_cold",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingConfig {
    pub maximum_pending: usize,
    pub maximum_prompt_bytes: usize,
    pub max_num_seqs: usize,
    pub request_batch_max_wait_ms: u64,
    pub default_width: u32,
    pub default_height: u32,
    pub default_seed: u64,
    pub rgba_scratch_bytes: usize,
    pub output: ferrum_types::OutputConfig,
}
impl ServingConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&self.max_num_seqs),
            "max_num_seqs exceeds supported image batch capacity"
        );
        ensure!(
            self.maximum_pending > 0 && (1..=1024 * 1024).contains(&self.maximum_prompt_bytes),
            "image queue and prompt bounds must be finite and positive"
        );
        self.maximum_pending
            .checked_add(self.max_num_seqs)
            .and_then(|n| n.checked_mul(self.maximum_prompt_bytes))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| anyhow::anyhow!("image queued prompt bound overflows"))?;
        std::time::Instant::now()
            .checked_add(std::time::Duration::from_millis(
                self.request_batch_max_wait_ms,
            ))
            .ok_or_else(|| anyhow::anyhow!("image batching wait exceeds clock range"))?;
        ensure!(
            (1..=64 * 1024).contains(&self.rgba_scratch_bytes),
            "invalid RGBA scratch bound"
        );
        let dimensions = ferrum_output::png::PngDimensions::new(
            self.default_width.into(),
            self.default_height.into(),
        )?;
        ferrum_output::png::PngLimits::default().validate(dimensions)?;
        self.output.validate()?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataPaths {
    pub model_index: PathBuf,
    pub text_encoder: PathBuf,
    pub transformer: PathBuf,
    pub vae: PathBuf,
    pub scheduler: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: PathBuf,
    pub chat_template: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightPaths {
    pub text_encoder: WeightSource,
    pub transformer: WeightSource,
    pub vae: WeightSource,
}
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WeightSource {
    Single { directory: PathBuf, file: String },
    Index { directory: PathBuf, file: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageBackendConfig {
    Cuda(CudaConfig),
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaConfig {
    pub ordinal: usize,
    pub usable_memory_bytes: u64,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
    pub cudnn: CudnnConfig,
    pub host: HostConfig,
    pub pipeline: PipelineConfig,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudnnConfig {
    pub library_path: PathBuf,
    pub opaque_memory_bytes: u64,
    pub workspace_limit_bytes: u64,
    pub opaque_memory_profile: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostConfig {
    pub total_bytes: usize,
    pub input_preprocess_bytes: usize,
    pub transfer_staging_bytes: usize,
    pub output_bytes: usize,
    pub weight_backing_bytes: usize,
}
impl HostConfig {
    pub fn limits(&self) -> Result<HostMemoryLimits> {
        Ok(HostMemoryLimits::new(
            self.total_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: self.input_preprocess_bytes,
                transfer_staging_bytes: self.transfer_staging_bytes,
                output_bytes: self.output_bytes,
                weight_backing_bytes: self.weight_backing_bytes,
            },
        )?)
    }
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineConfig {
    pub maximum_user_tokens: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    #[serde(default = "default_maximum_batch_size")]
    pub maximum_batch_size: usize,
    #[serde(default)]
    pub decoder_batch_policy: ferrum_engine::image_generation::ImageDecoderBatchPolicy,
    pub maximum_staging_bytes: u64,
    pub maximum_initialization_commands: usize,
    pub maximum_backing_maintenance_passes: usize,
}
fn default_maximum_batch_size() -> usize {
    ferrum_engine::image_generation::ImagePipelineConfig::DEFAULT_MAXIMUM_BATCH_SIZE
}
impl PipelineConfig {
    fn resolved(
        &self,
        device_capacity_bytes: u64,
    ) -> Result<ferrum_engine::image_generation::ImagePipelineConfig> {
        ensure!(
            device_capacity_bytes > 0 && self.maximum_width > 0 && self.maximum_height > 0,
            "backend usable_memory_bytes and image pixel bounds must be positive"
        );
        ensure!(
            (1..=8192).contains(&self.maximum_user_tokens),
            "image user-token limit must be 1..=8192"
        );
        ensure!(
            (1..=ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH)
                .contains(&self.maximum_batch_size),
            "pipeline maximum_batch_size exceeds supported attention layout capacity"
        );
        Ok(ferrum_engine::image_generation::ImagePipelineConfig {
            device_capacity_bytes,
            maximum_user_tokens: self.maximum_user_tokens,
            maximum_width: self.maximum_width,
            maximum_height: self.maximum_height,
            maximum_batch_size: self.maximum_batch_size,
            decoder_batch_policy: self.decoder_batch_policy,
            static_initialization: StaticInitializationPolicy::new(
                self.maximum_staging_bytes,
                self.maximum_initialization_commands,
            )?,
            executor: ferrum_execution::TensorExecutorConfig {
                maximum_backing_maintenance_passes: self.maximum_backing_maintenance_passes,
            },
        })
    }
}

impl CudaConfig {
    /// The product has one device-account ceiling. Internal plan admission
    /// consumes this same value; opaque claims are not subtracted a second time.
    /// This is pure configuration resolution and does not initialize a device.
    pub fn resolved_pipeline(
        &self,
    ) -> Result<ferrum_engine::image_generation::ImagePipelineConfig> {
        self.pipeline.resolved(self.usable_memory_bytes)
    }
}

impl ImageProductConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let path = std::fs::canonicalize(path)?;
        let bytes = super::source::read_bounded(&path, 1024 * 1024)?;
        ensure!(
            bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{'),
            "image configuration must be a JSON object"
        );
        let mut config: Self = serde_json::from_slice(&bytes)?;
        ensure!(
            !config.model.is_empty()
                && config.model.len() <= 160
                && !config.model.chars().any(char::is_control),
            "public image model name must be nonempty and bounded"
        );
        let base = path
            .parent()
            .expect("canonical configuration path has a parent");
        config.resolve_paths(base);
        let ImageBackendConfig::Cuda(cuda) = &config.backend;
        cuda.host.limits()?;
        cuda.resolved_pipeline()?;
        if let Some(serving) = &config.serving {
            serving.validate()?;
        }
        config.validate_execution_profile()?;
        Ok(config)
    }
    fn validate_execution_profile(&self) -> Result<()> {
        let ImageBackendConfig::Cuda(cuda) = &self.backend;
        if self.execution_profile == ImageExecutionProfile::ExperimentalSerialCold {
            ensure!(
                cuda.pipeline.maximum_batch_size == 1,
                "experimental_serial_cold requires pipeline maximum_batch_size=1"
            );
        }
        if let Some(serving) = &self.serving {
            ensure!(
                serving.max_num_seqs <= cuda.pipeline.maximum_batch_size,
                "serving max_num_seqs exceeds configured pipeline maximum_batch_size"
            );
            ensure!(
                serving
                    .max_num_seqs
                    .checked_mul(serving.maximum_prompt_bytes)
                    .is_some_and(|bytes| bytes <= 1024 * 1024),
                "combined image batch prompt bound exceeds the T2I tokenizer's 1 MiB limit"
            );
        }
        Ok(())
    }
    fn resolve_paths(&mut self, base: &Path) {
        for path in [
            &mut self.metadata.model_index,
            &mut self.metadata.text_encoder,
            &mut self.metadata.transformer,
            &mut self.metadata.vae,
            &mut self.metadata.scheduler,
            &mut self.metadata.tokenizer,
            &mut self.metadata.tokenizer_config,
            &mut self.metadata.chat_template,
        ] {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
        for source in [
            &mut self.weights.text_encoder,
            &mut self.weights.transformer,
            &mut self.weights.vae,
        ] {
            let (WeightSource::Single { directory, .. } | WeightSource::Index { directory, .. }) =
                source;
            if directory.is_relative() {
                *directory = base.join(&*directory);
            }
        }
        let ImageBackendConfig::Cuda(cuda) = &mut self.backend;
        if cuda.cudnn.library_path.is_relative() {
            cuda.cudnn.library_path = base.join(&cuda.cudnn.library_path);
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaConfig {
    pub fn resolved(&self) -> Result<ferrum_engine::image_generation::cuda::QwenImage21CudaConfig> {
        Ok(
            ferrum_engine::image_generation::cuda::QwenImage21CudaConfig {
                ordinal: self.ordinal,
                memory: ferrum_engine::image_generation::cuda::ImageCudaMemoryPolicy::ExplicitTotal(
                    std::num::NonZeroU64::new(self.usable_memory_bytes)
                        .ok_or_else(|| anyhow::anyhow!("usable_memory_bytes must be positive"))?,
                ),
                bootstrap_memory_bytes: self.bootstrap_memory_bytes,
                stream_memory_bytes: self.stream_memory_bytes,
                maximum_streams: self.maximum_streams,
                maximum_in_flight_per_stream: self.maximum_in_flight_per_stream,
                maximum_commands_per_submission: self.maximum_commands_per_submission,
                opaque_memory_profile: self.opaque_memory_profile.clone(),
                cudnn: ferrum_kernels::backend::cuda::CudaCudnnConfig {
                    library_path: self.cudnn.library_path.clone(),
                    opaque_memory_bytes: self.cudnn.opaque_memory_bytes,
                    workspace_limit_bytes: self.cudnn.workspace_limit_bytes,
                    opaque_memory_profile: self.cudnn.opaque_memory_profile.clone(),
                },
                host: self.host.limits()?,
                pipeline: self.pipeline.resolved_limits()?,
            },
        )
    }
}

#[cfg(feature = "cuda")]
impl PipelineConfig {
    pub(super) fn resolved_limits(
        &self,
    ) -> Result<ferrum_engine::image_generation::cuda::ImageCudaPipelineLimits> {
        let limits = ferrum_engine::image_generation::cuda::ImageCudaPipelineLimits {
            maximum_user_tokens: self.maximum_user_tokens,
            maximum_width: self.maximum_width,
            maximum_height: self.maximum_height,
            maximum_batch_size: self.maximum_batch_size,
            decoder_batch_policy: self.decoder_batch_policy,
            static_initialization: StaticInitializationPolicy::new(
                self.maximum_staging_bytes,
                self.maximum_initialization_commands,
            )?,
            executor: ferrum_execution::TensorExecutorConfig::new(
                self.maximum_backing_maintenance_passes,
            )?,
        };
        limits.validate()?;
        Ok(limits)
    }
}
