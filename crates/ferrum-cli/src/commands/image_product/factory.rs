//! Backend selection at the product boundary; execution and service remain shared.
use super::{
    configuration::ImageBackendConfig,
    directory::{DirectoryOptions, ImageBackendKind},
};
use anyhow::Result;
use ferrum_engine::image_generation::{
    service::{ImageGenerator, ImageService, ImageServiceConfig},
    ImageEditRequest, ImageGenerationOutput, ImageGenerationRequest, ImagePipelineLimits,
    ImageStageReceipt, QwenImage21Assets, QwenImage21Pipeline,
};

pub(super) enum PreparedAssets {
    QwenImage21(QwenImage21Assets),
    #[cfg(feature = "cuda")]
    PH3Image(ferrum_engine::image_generation::p_h3::PH3ImageAssets),
}
impl PreparedAssets {
    pub(super) fn spatial_scale(&self) -> Result<u32> {
        match self {
            Self::QwenImage21(assets) => assets.spatial_scale(),
            #[cfg(feature = "cuda")]
            Self::PH3Image(assets) => Ok(u32::try_from(assets.spatial_scale())?),
        }
    }
}

pub(super) enum PreparedBackend {
    #[cfg(feature = "cuda")]
    Cuda(
        ferrum_engine::image_generation::cuda::QwenImage21CudaConfig,
        Option<super::host_cache::AutoHostCacheReceipt>,
    ),
    #[cfg(feature = "cuda")]
    PH3Cuda(ferrum_engine::image_generation::p_h3::cuda::PH3ImageCudaConfig),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(ferrum_engine::image_generation::metal::QwenImage21MetalConfig),
}
pub(super) enum PreparedPipeline {
    #[cfg(feature = "cuda")]
    Cuda(Box<QwenImage21Pipeline<ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime>>),
    #[cfg(feature = "cuda")]
    PH3Cuda(
        Box<
            ferrum_engine::image_generation::p_h3::PH3ImagePipeline<
                ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime,
            >,
        >,
    ),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(
        Box<QwenImage21Pipeline<ferrum_kernels::backend::metal::vnext_runtime::MetalDeviceRuntime>>,
    ),
}
impl PreparedBackend {
    pub(super) fn from_p_h3_config(config: &super::p_h3::PH3ImageProductConfig) -> Result<Self> {
        Self::ensure_supported(ImageBackendKind::Cuda)?;
        #[cfg(feature = "cuda")]
        {
            let super::p_h3::PH3ImageBackendConfig::Cuda(backend) = &config.backend;
            let mut resolved =
                backend.resolved(config.vae_spatial_policy, config.vae_decoder_tile_batch)?;
            resolved.options.initial_noise = config.initial_noise;
            resolved.validate()?;
            Ok(Self::PH3Cuda(resolved))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = config;
            anyhow::bail!("H3 Comfy preset requires a binary built with the cuda feature")
        }
    }
    /// Reject an unavailable backend before reading model assets or resolving
    /// automatic resource policy. This check performs no device/library work.
    pub(super) fn ensure_supported(backend: ImageBackendKind) -> Result<()> {
        match backend {
            ImageBackendKind::Cuda => anyhow::ensure!(
                cfg!(feature = "cuda"),
                "image CUDA backend requires a binary built with the cuda feature"
            ),
            ImageBackendKind::Metal => anyhow::ensure!(
                cfg!(all(feature = "metal", target_os = "macos")),
                "image Metal backend requires macOS and a binary built with the metal feature"
            ),
        }
        Ok(())
    }

    pub(super) fn from_config(config: &ImageBackendConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "cuda")]
            ImageBackendConfig::Cuda(cuda) => {
                let mut config = cuda.resolved()?;
                config.cudnn.library_path =
                    super::library::resolve(Some(&config.cudnn.library_path))?;
                config.validate()?;
                Ok(Self::Cuda(config, None))
            }
            #[cfg(not(feature = "cuda"))]
            ImageBackendConfig::Cuda(_) => {
                anyhow::bail!("image CUDA backend requires a binary built with the cuda feature")
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            ImageBackendConfig::Metal(metal) => Ok(Self::Metal(metal.resolved()?)),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            ImageBackendConfig::Metal(_) => anyhow::bail!(
                "image Metal backend requires macOS and a binary built with the metal feature"
            ),
        }
    }
    pub(super) fn from_directory(
        options: &DirectoryOptions,
        _assets: &QwenImage21Assets,
    ) -> Result<Self> {
        match options.backend {
            #[cfg(feature = "cuda")]
            ImageBackendKind::Cuda => {
                let mut config = options.cuda()?;
                let selected = super::host_cache::for_directory(options, _assets)?;
                config.host = selected.host.limits()?;
                config.fp8_host_cache = selected.cache;
                config.validate()?;
                tracing::info!(
                    decision = ?selected.receipt.decision,
                    observed_available_bytes = ?selected.receipt.observed_available_bytes,
                    required_cache_bytes = ?selected.receipt.required_cache_bytes,
                    required_host_total_bytes = ?selected.receipt.required_host_total_bytes,
                    effective_host_total_bytes = selected.receipt.effective_host_total_bytes,
                    output_selection = ?selected.receipt.output_selection,
                    input_selection = ?selected.receipt.input_selection,
                    observation_error = ?selected.receipt.observation_error,
                    "automatic image Host cache and output allowances selected"
                );
                Ok(Self::Cuda(config, Some(selected.receipt)))
            }
            #[cfg(not(feature = "cuda"))]
            ImageBackendKind::Cuda => {
                anyhow::bail!("image CUDA backend requires a binary built with the cuda feature")
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            ImageBackendKind::Metal => Ok(Self::Metal(options.metal()?)),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            ImageBackendKind::Metal => anyhow::bail!(
                "image Metal backend requires macOS and a binary built with the metal feature"
            ),
        }
    }
    pub(super) fn limits(&self) -> &ImagePipelineLimits {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(config, _) => &config.pipeline,
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(config) => &config.pipeline,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(config) => &config.pipeline,
        }
    }
    pub(super) fn host_output_bytes(&self) -> usize {
        self.host_limits()
            .class_bytes(ferrum_runtime::HostMemoryClass::Output)
    }
    pub(super) fn host_input_bytes(&self) -> usize {
        self.host_limits()
            .class_bytes(ferrum_runtime::HostMemoryClass::InputPreprocess)
    }
    fn host_limits(&self) -> ferrum_runtime::HostMemoryLimits {
        let host = match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(config, _) => config.host,
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(config) => config.host,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(config) => config.host,
        };
        host
    }
    pub(super) fn create(
        self,
        assets: PreparedAssets,
    ) -> Result<(PreparedPipeline, serde_json::Value)> {
        match (self, assets) {
            #[cfg(feature = "cuda")]
            (Self::Cuda(config, selection), PreparedAssets::QwenImage21(assets)) => {
                let (pipeline, receipt) = config.create_pipeline_with_receipt(assets)?;
                let mut receipt = serde_json::to_value(receipt)?;
                if let Some(selection) = selection {
                    receipt["fp8_host_cache_selection"] = serde_json::to_value(selection)?;
                }
                Ok((PreparedPipeline::Cuda(Box::new(pipeline)), receipt))
            }
            #[cfg(feature = "cuda")]
            (Self::PH3Cuda(config), PreparedAssets::PH3Image(assets)) => {
                let (pipeline, receipt) = config.create_pipeline_with_receipt(assets)?;
                Ok((
                    PreparedPipeline::PH3Cuda(Box::new(pipeline)),
                    serde_json::to_value(receipt)?,
                ))
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(config), PreparedAssets::QwenImage21(assets)) => {
                let (pipeline, receipt) = config.create_pipeline_with_receipt(assets)?;
                Ok((
                    PreparedPipeline::Metal(Box::new(pipeline)),
                    serde_json::to_value(receipt)?,
                ))
            }
            #[cfg(feature = "cuda")]
            _ => {
                anyhow::bail!("image backend and validated assets declare different architectures")
            }
        }
    }
}
type GenerationFailure = Box<dyn std::error::Error + Send + 'static>;
impl PreparedPipeline {
    /// Bind the original bounded collector before enabling observations. The
    /// caller's existing failure branch retires this same pipeline on error.
    pub(super) fn enable_continuous_trace_artifact(
        &mut self,
        artifact: &super::trace_artifact::PreparedTraceArtifact,
        context: &serde_json::Value,
    ) -> Result<ferrum_engine::image_generation::continuous_trace::ImageContinuousTraceHandle> {
        let handle = self.enable_continuous_trace(artifact.limits(), context)?;
        match self {
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => {
                pipeline
                    .set_prompt_expansion_diagnostics(artifact.prompt_expansion_diagnostics())?;
            }
            #[allow(unreachable_patterns)]
            _ => {
                anyhow::bail!("this product pipeline does not declare prompt expansion diagnostics")
            }
        }
        Ok(handle)
    }

    pub(super) fn enable_continuous_trace(
        &mut self,
        limits: ferrum_engine::image_generation::continuous_trace::ImageContinuousTraceLimits,
        context: &serde_json::Value,
    ) -> Result<ferrum_engine::image_generation::continuous_trace::ImageContinuousTraceHandle> {
        match self {
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => pipeline.enable_continuous_trace(limits, context),
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!("this product pipeline does not declare continuous trace support"),
        }
    }

    pub(super) fn into_service(self, config: ImageServiceConfig) -> Result<ImageService> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => Ok(ImageService::new(*pipeline, config)?),
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => Ok(ImageService::new(*pipeline, config)?),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => Ok(ImageService::new(*pipeline, config)?),
        }
    }
    pub(super) fn into_generator(self) -> Box<dyn ImageGenerator> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => pipeline,
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => pipeline,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => pipeline,
        }
    }
    pub(super) fn host_account(&self) -> &ferrum_runtime::HostMemoryAccount {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => pipeline.host_account(),
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => pipeline.host_account(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => pipeline.host_account(),
        }
    }
    pub(super) fn residency_policy(&self) -> ferrum_engine::image_generation::ImageResidencyPolicy {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => pipeline.residency_policy(),
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => pipeline.residency_policy(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => pipeline.residency_policy(),
        }
    }
    pub(super) fn supports_edits(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => pipeline.supports_edits(),
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => ImageGenerator::supports_edits(&**pipeline),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => pipeline.supports_edits(),
        }
    }
    pub(super) fn generate(
        &mut self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationOutput, GenerationFailure> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => pipeline
                .generate(request)
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => pipeline
                .generate(request)
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => pipeline
                .generate(request)
                .map_err(|e| Box::new(e) as GenerationFailure),
        }
    }
    pub(super) fn edit(
        &mut self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationOutput, GenerationFailure> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => pipeline
                .edit(request)
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => ImageGenerator::edit(&mut **pipeline, request)
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => pipeline
                .edit(request)
                .map_err(|e| Box::new(e) as GenerationFailure),
        }
    }
    pub(super) fn shutdown(&mut self) -> Result<Vec<ImageStageReceipt>, GenerationFailure> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => pipeline
                .shutdown()
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(feature = "cuda")]
            Self::PH3Cuda(pipeline) => pipeline
                .shutdown()
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => pipeline
                .shutdown()
                .map_err(|e| Box::new(e) as GenerationFailure),
        }
    }
    #[cfg(feature = "cuda")]
    pub(super) fn with_device_timing_mode(
        self,
        mode: ferrum_runtime::DeviceTimingMode,
    ) -> Result<Self> {
        match self {
            Self::Cuda(pipeline) => Ok(Self::Cuda(Box::new(
                (*pipeline).with_device_timing_mode(mode)?,
            ))),
            Self::PH3Cuda(pipeline) => Ok(Self::PH3Cuda(Box::new(
                (*pipeline).with_device_timing_mode(mode)?,
            ))),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(_) => anyhow::bail!("this diagnostic requires CUDA"),
        }
    }
    #[cfg(feature = "cuda")]
    pub(super) fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, GenerationFailure> {
        match self {
            Self::Cuda(pipeline) => pipeline
                .generate_batch(requests)
                .map_err(|e| Box::new(e) as GenerationFailure),
            Self::PH3Cuda(pipeline) => ImageGenerator::generate_batch(&mut **pipeline, requests)
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(_) => Err(Box::new(std::io::Error::other(
                "this diagnostic requires CUDA",
            ))),
        }
    }
    #[cfg(feature = "cuda")]
    pub(super) fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, GenerationFailure> {
        match self {
            Self::Cuda(pipeline) => pipeline
                .edit_batch(requests)
                .map_err(|e| Box::new(e) as GenerationFailure),
            Self::PH3Cuda(pipeline) => ImageGenerator::edit_batch(&mut **pipeline, requests)
                .map_err(|e| Box::new(e) as GenerationFailure),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(_) => Err(Box::new(std::io::Error::other(
                "this diagnostic requires CUDA",
            ))),
        }
    }
}
