//! F32 activation transforms; weights remain in their original packed format.

use super::*;
use cudarc::driver::CudaModule;
use ferrum_interfaces::vnext::{
    ElementType, HadamardApplication, HadamardSigns, HadamardTransformSpec,
};

// 32 KiB fits the default per-block shared-memory limit of supported CUDA
// devices. Larger declarations fail during planning, before any allocation.
const MAX_BLOCK_SIZE: u32 = 8192;

pub(in crate::backend::cuda::vnext_ops) fn validate(
    spec: &HadamardTransformSpec,
    width: u64,
) -> Result<(), String> {
    spec.validate(width).map_err(|error| error.to_string())?;
    if width > u64::from(u32::MAX)
        || spec.block_size.get() > MAX_BLOCK_SIZE
        || width / u64::from(spec.block_size.get()) > i32::MAX as u64
    {
        return Err("CUDA Hadamard extent exceeds the installed kernel capacity".into());
    }
    Ok(())
}

pub(in crate::backend::cuda::vnext_ops) fn workspace_bytes_per_token(
    bindings: &[ferrum_interfaces::vnext::ResolvedValueBinding],
) -> Result<u64, String> {
    bindings
        .iter()
        .filter_map(|binding| binding.weight().map(|weight| (binding, weight)))
        .try_fold(0_u64, |maximum, (binding, weight)| {
            fn width(
                layout: &ferrum_interfaces::vnext::PhysicalWeightLayout,
                dimensions: &[u64],
            ) -> Result<u64, String> {
                use ferrum_interfaces::vnext::PhysicalWeightLayout;
                match layout {
                    PhysicalWeightLayout::Hadamard { transform, .. } => {
                        let width = *dimensions
                            .last()
                            .ok_or("CUDA Hadamard has no feature axis")?;
                        validate(transform, width)?;
                        Ok(width)
                    }
                    PhysicalWeightLayout::Composite { parts } => {
                        parts.iter().try_fold(0, |max, part| {
                            Ok(max.max(width(&part.layout, &part.extents)?))
                        })
                    }
                    _ => Ok(0),
                }
            }
            let bytes = width(weight.physical_layout(), binding.tensor().dimensions())?
                .checked_mul(4)
                .and_then(|bytes| bytes.checked_add(15))
                .map(|bytes| bytes & !15)
                .ok_or("CUDA Hadamard workspace overflows")?;
            Ok(maximum.max(bytes))
        })
}

pub(in crate::backend::cuda::vnext_ops) fn token_workspace(
    values: &[ferrum_interfaces::vnext::ResolvedValueBinding],
) -> Result<
    Option<ferrum_interfaces::vnext::ProviderWorkspaceRequirement>,
    ferrum_interfaces::vnext::VNextError,
> {
    use ferrum_interfaces::vnext::*;
    let bytes = workspace_bytes_per_token(values)
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    (bytes > 0)
        .then(|| {
            ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::tokens(bytes)?,
                super::super::VALUE_ALIGNMENT_BYTES,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )
        })
        .transpose()
}

pub(in crate::backend::cuda::vnext_ops) fn retain_workspace(
    invocation: &ferrum_interfaces::vnext::BatchedOperationInvocation<
        '_,
        super::super::CudaDeviceBuffer,
    >,
    regions: &mut Vec<super::super::CudaBufferRegion>,
) -> Result<Option<usize>, String> {
    let bytes = workspace_bytes_per_token(invocation.participants()[0].bindings())?
        .checked_mul(invocation.work_shape().immediate_tokens())
        .ok_or("CUDA Hadamard workspace size overflows")?;
    if bytes == 0 {
        return Ok(None);
    }
    let index = regions.len();
    regions.push(super::super::transformer::shared_scratch_region(
        invocation, bytes,
    )?);
    Ok(Some(index))
}

#[derive(Clone)]
pub(super) struct CudaHadamardKernels {
    f16_f32: CudaFunction,
    f32_f32: CudaFunction,
    f32_f16: CudaFunction,
}

impl CudaHadamardKernels {
    pub(super) fn load(module: &Arc<CudaModule>) -> Result<Self, CudaDeviceRuntimeError> {
        let load = |name| {
            module.load_function(name).map_err(|error| {
                CudaDeviceRuntimeError::driver("native Hadamard function load", error)
            })
        };
        Ok(Self {
            f16_f32: load("vnext_gguf_hadamard_f16_f32")?,
            f32_f32: load("vnext_gguf_hadamard_f32_f32")?,
            f32_f16: load("vnext_gguf_hadamard_f32_f16")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(
        &self,
        stream: &CudaStream,
        input: u64,
        output: u64,
        signs: u64,
        rows: u32,
        width: u32,
        input_type: ElementType,
        output_type: ElementType,
        spec: &HadamardTransformSpec,
    ) -> Result<(), CudaDeviceRuntimeError> {
        validate(spec, u64::from(width)).map_err(CudaDeviceRuntimeError::contract)?;
        if rows == 0
            || rows > u16::MAX as u32
            || matches!(spec.signs, HadamardSigns::Explicit(_)) != (signs != 0)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA Hadamard rows or signs differ from the declaration",
            ));
        }
        let (inverse, permutation) = match spec.application {
            HadamardApplication::BeforeMatmul { input_permutation } => (0_u32, input_permutation),
            HadamardApplication::AfterEmbeddingLookup => (1, None),
        };
        if input == output && (input_type != output_type || permutation.is_some()) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA Hadamard cannot overwrite permuted or differently sized input",
            ));
        }
        let function = match (input_type, output_type) {
            (ElementType::F16, ElementType::F32) => &self.f16_f32,
            (ElementType::F32, ElementType::F32) => &self.f32_f32,
            (ElementType::F32, ElementType::F16) => &self.f32_f16,
            _ => {
                return Err(CudaDeviceRuntimeError::contract(
                    "unsupported CUDA Hadamard precision",
                ))
            }
        };
        let (inner, first, second) = permutation.map_or((0, 0, 0), |permutation| {
            (
                permutation.inner_extent as u32,
                permutation.first_outer_extent as u32,
                permutation.second_outer_extent as u32,
            )
        });
        let block = spec.block_size.get();
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&input)
            .arg(&output)
            .arg(&signs)
            .arg(&width)
            .arg(&block)
            .arg(&inverse)
            .arg(&inner)
            .arg(&first)
            .arg(&second);
        // SAFETY: The caller retains complete row spans and the optional full
        // width F32 signs. Each block owns its shared memory and output span;
        // every thread participates in every butterfly barrier.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (width / block, rows, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: block * 4,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("native Hadamard launch", error))
    }
}

#[cfg(test)]
mod tests;
