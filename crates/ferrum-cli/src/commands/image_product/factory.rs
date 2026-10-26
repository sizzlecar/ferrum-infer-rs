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
    Cuda(ferrum_engine::image_generation::cuda::QwenImage21CudaConfig),
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
    pub(super) fn from_config(config: &ImageBackendConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "cuda")]
            ImageBackendConfig::Cuda(cuda) => {
                let mut config = cuda.resolved()?;
                config.cudnn.library_path =
                    super::library::resolve(Some(&config.cudnn.library_path))?;
                config.validate()?;
                Ok(Self::Cuda(config))
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
    pub(super) fn from_directory(options: &DirectoryOptions) -> Result<Self> {
        match options.backend {
            #[cfg(feature = "cuda")]
            ImageBackendKind::Cuda => {
                let config = options.cuda()?;
                config.validate()?;
                Ok(Self::Cuda(config))
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
            Self::Cuda(config) => &config.pipeline,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(config) => &config.pipeline,
        }
    }
    pub(super) fn create(
        self,
        assets: QwenImage21Assets,
    ) -> Result<(PreparedPipeline, serde_json::Value)> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(config) => {
                let (pipeline, receipt) = config.create_pipeline_with_receipt(assets)?;
                Ok((
                    PreparedPipeline::Cuda(Box::new(pipeline)),
                    serde_json::to_value(receipt)?,
                ))
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
