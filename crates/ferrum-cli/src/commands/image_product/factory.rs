//! Backend selection at the product boundary; execution and service remain shared.
use super::{
    configuration::ImageBackendConfig,
    directory::{DirectoryOptions, ImageBackendKind},
};
use anyhow::Result;
use ferrum_engine::image_generation::{
    service::{ImageService, ImageServiceConfig},
    ImagePipelineLimits, QwenImage21Assets, QwenImage21Pipeline,
};

pub(super) enum PreparedBackend {
    #[cfg(feature = "cuda")]
    Cuda(
        ferrum_engine::image_generation::cuda::QwenImage21CudaConfig,
        Option<super::host_cache::AutoHostCacheReceipt>,
    ),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(ferrum_engine::image_generation::metal::QwenImage21MetalConfig),
}
pub(super) enum PreparedPipeline {
    #[cfg(feature = "cuda")]
    Cuda(Box<QwenImage21Pipeline<ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime>>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(
        Box<QwenImage21Pipeline<ferrum_kernels::backend::metal::vnext_runtime::MetalDeviceRuntime>>,
    ),
}
impl PreparedBackend {
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
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(config) => &config.pipeline,
        }
    }
    pub(super) fn host_output_bytes(&self) -> usize {
        let host = match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(config, _) => config.host,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(config) => config.host,
        };
        host.class_bytes(ferrum_runtime::HostMemoryClass::Output)
    }
    pub(super) fn create(
        self,
        assets: QwenImage21Assets,
    ) -> Result<(PreparedPipeline, serde_json::Value)> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(config, selection) => {
                let (pipeline, receipt) = config.create_pipeline_with_receipt(assets)?;
                let mut receipt = serde_json::to_value(receipt)?;
                if let Some(selection) = selection {
                    receipt["fp8_host_cache_selection"] = serde_json::to_value(selection)?;
                }
                Ok((PreparedPipeline::Cuda(Box::new(pipeline)), receipt))
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(config) => {
                let (pipeline, receipt) = config.create_pipeline_with_receipt(assets)?;
                Ok((
                    PreparedPipeline::Metal(Box::new(pipeline)),
                    serde_json::to_value(receipt)?,
                ))
            }
        }
    }
}
impl PreparedPipeline {
    pub(super) fn into_service(self, config: ImageServiceConfig) -> Result<ImageService> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(pipeline) => Ok(ImageService::new(*pipeline, config)?),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(pipeline) => Ok(ImageService::new(*pipeline, config)?),
        }
    }
}
