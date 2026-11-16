//! Explicit provider composition over one tensor module, cuDNN owner and
//! optional bounded SDPA cache. This creates no second execution context.
use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaCudnnConfig, CudaCudnnOwner, CudaCudnnSdpaCache, CudnnSdpaPlanPolicy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaCudnnOperation {
    Conv2dBf16,
    Conv2dF32,
    Conv3dF32,
    ConvTranspose2dF32,
    StructuredAttentionBf16,
    SegmentedAttentionBf16,
    CrossAttentionBf16,
    CausalGqaBf16,
}
impl CudaCudnnOperation {
    fn needs_sdpa(self) -> bool {
        matches!(
            self,
            Self::StructuredAttentionBf16
                | Self::SegmentedAttentionBf16
                | Self::CrossAttentionBf16
                | Self::CausalGqaBf16
        )
    }
    fn capability(self) -> &'static str {
        use ferrum_runtime::*;
        match self {
            Self::Conv2dBf16 => CONV2D_BF16_CAPABILITY_ID,
            Self::Conv2dF32 => CONV2D_F32_CAPABILITY_ID,
            Self::Conv3dF32 => CONV3D_F32_CAPABILITY_ID,
            Self::ConvTranspose2dF32 => CONV_TRANSPOSE2D_F32_CAPABILITY_ID,
            Self::StructuredAttentionBf16 => STRUCTURED_ATTENTION_FUSED_BF16_CAPABILITY_ID,
            Self::SegmentedAttentionBf16 => SEGMENTED_ATTENTION_BF16_CAPABILITY_ID,
            Self::CrossAttentionBf16 => CROSS_ATTENTION_FUSED_BF16_CAPABILITY_ID,
            Self::CausalGqaBf16 => STRUCTURED_GQA_ATTENTION_FUSED_BF16_CAPABILITY_ID,
        }
    }
}

/// Metadata checks precede every module, library or cache acquisition.
fn validate(
    device: &DeviceDescriptor,
    policy: Option<CudnnSdpaPlanPolicy>,
    vendor: &[CudaCudnnOperation],
    selected: &[CudaTensorOperation],
) -> Result<(), CudaDeviceRuntimeError> {
    if vendor.is_empty() {
        return Err(CudaDeviceRuntimeError::contract(
            "explicit cuDNN composition requires a selected vendor operation",
        ));
    }
    if !selected.is_empty() {
        selection::validate_selection(selected)?;
    }
    if selected.contains(&CudaTensorOperation::StructuredGqaAttentionFusedF32) {
        return Err(CudaDeviceRuntimeError::contract(
            "external fused F32 attention requires its independently admitted factory",
        ));
    }
    for (index, op) in vendor.iter().enumerate() {
        if vendor[..index].contains(op) {
            return Err(CudaDeviceRuntimeError::contract(
                "duplicate cuDNN operation selection",
            ));
        }
        let capability = CapabilityId::new(op.capability()).map_err(contract_error)?;
        if !device.capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "CUDA runtime lacks selected cuDNN capability {}",
                op.capability()
            )));
        }
    }
    if vendor.contains(&CudaCudnnOperation::SegmentedAttentionBf16)
        && selected.contains(&CudaTensorOperation::SegmentedAttention)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "segmented attention must select exactly one native or cuDNN implementation",
        ));
    }
    if vendor.iter().any(|op| op.needs_sdpa()) != policy.is_some() {
        return Err(CudaDeviceRuntimeError::contract("selected cuDNN attention and an explicit bounded SDPA cache policy are required together"));
    }
    if let Some(policy) = policy {
        policy
            .validate()
            .map_err(CudaDeviceRuntimeError::contract)?;
    }
    Ok(())
}

pub(crate) fn providers(
    runtime: &CudaDeviceRuntime,
    config: &CudaCudnnConfig,
    policy: Option<CudnnSdpaPlanPolicy>,
    vendor: &[CudaCudnnOperation],
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    validate(runtime.descriptor(), policy, vendor, selected)?;
    let descriptors = if selected.is_empty() {
        Vec::new()
    } else {
        selection::runtime_descriptors(runtime, selected)?
    };
    let owner = CudaCudnnOwner::new(runtime, config)?;
    let cache = policy
        .map(|policy| CudaCudnnSdpaCache::new(runtime, owner.clone(), policy).map(Arc::new))
        .transpose()?;
    let kernels = RetainedFunctions::load(runtime)?;
    let mut providers = selection::providers_from_descriptors(descriptors, &kernels);
    for op in vendor {
        let cache = || {
            cache
                .as_ref()
                .expect("validated SDPA selection has one cache")
                .clone()
        };
        providers.push(match op {
            CudaCudnnOperation::Conv2dBf16 => {
                conv2d::provider(runtime, kernels.clone(), owner.clone())?
            }
            CudaCudnnOperation::Conv2dF32 => {
                conv2d_f32::provider(runtime, kernels.clone(), owner.clone())?
            }
            CudaCudnnOperation::Conv3dF32 => conv2d_f32::provider_for(
                runtime,
                kernels.clone(),
                owner.clone(),
                ferrum_runtime::F32CnnOp::Conv3d,
            )?,
            CudaCudnnOperation::ConvTranspose2dF32 => {
                conv_transpose2d_f32::provider(runtime, kernels.clone(), owner.clone())?
            }
            CudaCudnnOperation::StructuredAttentionBf16 => {
                cudnn_attention::provider(runtime.descriptor(), &kernels, cache())?
            }
            CudaCudnnOperation::SegmentedAttentionBf16 => {
                cudnn_segmented::provider(runtime.descriptor(), &kernels, cache())?
            }
            CudaCudnnOperation::CrossAttentionBf16 => {
                cudnn_cross_attention::provider(runtime.descriptor(), &kernels, cache())?
            }
            CudaCudnnOperation::CausalGqaBf16 => {
                cudnn_gqa::provider(runtime.descriptor(), cache())?
            }
        });
    }
    Ok(providers)
}

#[cfg(test)]
mod tests;
