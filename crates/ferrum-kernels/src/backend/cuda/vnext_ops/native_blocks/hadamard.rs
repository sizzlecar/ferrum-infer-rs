//! F32 activation transforms; weights remain in their original packed format.

use super::*;
use cudarc::driver::CudaModule;
use ferrum_runtime::{ElementType, HadamardApplication, HadamardSigns, HadamardTransformSpec};

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
    bindings: &[ferrum_runtime::ResolvedValueBinding],
) -> Result<u64, String> {
    bindings
        .iter()
        .filter_map(|binding| binding.weight().map(|weight| (binding, weight)))
        .try_fold(0_u64, |maximum, (binding, weight)| {
            fn width(
                layout: &ferrum_runtime::PhysicalWeightLayout,
                dimensions: &[u64],
            ) -> Result<u64, String> {
                use ferrum_runtime::PhysicalWeightLayout;
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
    values: &[ferrum_runtime::ResolvedValueBinding],
) -> Result<Option<ferrum_runtime::ProviderWorkspaceRequirement>, ferrum_runtime::VNextError> {
    use ferrum_runtime::*;
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
    invocation: &ferrum_runtime::BatchedOperationInvocation<'_, super::super::CudaDeviceBuffer>,
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

#[derive(Clone, Copy)]
enum HadamardKernel {
    F16F32,
    F32F32,
    F32F16,
}

#[derive(Clone, Copy)]
pub(in crate::backend::cuda::vnext_ops) struct HadamardSelection {
    kernel: HadamardKernel,
    parameters: [u32; 6],
    rows: u32,
    explicit_signs: bool,
    in_place_allowed: bool,
}
impl HadamardSelection {
    pub(in crate::backend::cuda::vnext_ops) fn select(
        rows: u32,
        width: u32,
        input_type: ElementType,
        output_type: ElementType,
        spec: &HadamardTransformSpec,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        validate(spec, u64::from(width)).map_err(CudaDeviceRuntimeError::contract)?;
        if rows == 0 || rows > u16::MAX as u32 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA Hadamard rows differ from the declaration",
            ));
        }
        let (inverse, permutation) = match spec.application {
            HadamardApplication::BeforeMatmul { input_permutation } => (0_u32, input_permutation),
            HadamardApplication::AfterEmbeddingLookup => (1, None),
        };
        let kernel = match (input_type, output_type) {
            (ElementType::F16, ElementType::F32) => HadamardKernel::F16F32,
            (ElementType::F32, ElementType::F32) => HadamardKernel::F32F32,
            (ElementType::F32, ElementType::F16) => HadamardKernel::F32F16,
            _ => {
                return Err(CudaDeviceRuntimeError::contract(
                    "unsupported CUDA Hadamard precision",
                ))
            }
        };
        let (inner, first, second) = permutation.map_or((0, 0, 0), |p| {
            (
                p.inner_extent as u32,
                p.first_outer_extent as u32,
                p.second_outer_extent as u32,
            )
        });
        Ok(Self {
            kernel,
            parameters: [width, spec.block_size.get(), inverse, inner, first, second],
            rows,
            explicit_signs: matches!(spec.signs, HadamardSigns::Explicit(_)),
            in_place_allowed: input_type == output_type && permutation.is_none(),
        })
    }
    pub(in crate::backend::cuda::vnext_ops) fn implementation(self) -> &'static str {
        match self.kernel {
            HadamardKernel::F16F32 => "vnext_gguf_hadamard_f16_f32",
            HadamardKernel::F32F32 => "vnext_gguf_hadamard_f32_f32",
            HadamardKernel::F32F16 => "vnext_gguf_hadamard_f32_f16",
        }
    }
    pub(in crate::backend::cuda::vnext_ops) fn launch(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.parameters[0] / self.parameters[1], self.rows, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: self.parameters[1] * 4,
        }
    }
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
        let selection = HadamardSelection::select(rows, width, input_type, output_type, spec)?;
        self.launch_selected(stream, input, output, signs, selection)
    }

    pub(in crate::backend::cuda::vnext_ops) fn launch_selected(
        &self,
        stream: &CudaStream,
        input: u64,
        output: u64,
        signs: u64,
        selection: HadamardSelection,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if selection.explicit_signs != (signs != 0) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA Hadamard signs differ from the declaration",
            ));
        }
        if input == output && !selection.in_place_allowed {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA Hadamard cannot overwrite permuted or differently sized input",
            ));
        }
        let function = match selection.kernel {
            HadamardKernel::F16F32 => &self.f16_f32,
            HadamardKernel::F32F32 => &self.f32_f32,
            HadamardKernel::F32F16 => &self.f32_f16,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&input).arg(&output).arg(&signs);
        for parameter in &selection.parameters {
            launch.arg(parameter);
        }
        // SAFETY: Original retained row spans and signs remain alive; every
        // block owns the selected shared-memory span and butterfly barriers.
        unsafe { launch.launch(selection.launch()) }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("native Hadamard launch", error))
    }
}

#[cfg(test)]
mod tests;
