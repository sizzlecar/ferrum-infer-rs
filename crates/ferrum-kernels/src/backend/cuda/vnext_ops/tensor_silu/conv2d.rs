//! cuDNN BF16 NCHW convolution with explicitly admitted padding/workspace.
//! PyTorch v2.13 Convolution.cpp:1502-1509 writes convolution BF16 before
//! separately adding BF16 bias. No fused bias/activation entrypoint is used.

use super::*;
use crate::backend::cuda::vnext_runtime::{CudaCudnnConfig, CudaCudnnOwner, CudnnConv2dDescriptor};
use ferrum_runtime::{
    conv2d_bf16_contract, Conv2dBf16Geometry, Conv2dPaddingMode, DynamicStorageRequirement,
    PhysicalWeightLayout, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, WeightFormatId,
    CONV2D_BF16_CAPABILITY_ID, CONV2D_BF16_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cudnn9.conv2d.bf16_f32.separate_bias.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "NCHW/OIHW-cross-correlation;BF16-input-weight-output;FP32-compute;",
    "explicit-materialized-zero-padding;deterministic-v7-heuristic-within-admitted-workspace;",
    "math=DEFAULT-or-TENSOR_OP;no-ALLOW_CONVERSION;no-find-or-unbudgeted-workspace;",
    "B(convolution)-then-B(RN32(output+bias));no-Torch-or-Reference-bitwise-claim;",
    "logical-work=nominal-2*output-elements*reduction-extent"
);
const PROVIDER: &str = "provider.cuda.conv2d.bf16.cudnn9";
const ESTIMATOR: &str = "resource-estimator.cuda.conv2d.bf16.cudnn9";
const SCRATCH_ALIGNMENT: u64 = 256;

/// One shared native module plus one separately charged cuDNN owner. Each
/// requested native operation must be advertised by the runtime descriptor.
pub(crate) fn providers(
    runtime: &CudaDeviceRuntime,
    config: &CudaCudnnConfig,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(CONV2D_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks BF16 convolution capability",
        ));
    }
    let descriptors = if selected.is_empty() {
        Vec::new()
    } else {
        selection::selected_descriptors(runtime.descriptor(), selected)?
    };
    let owner = CudaCudnnOwner::new(runtime, config)?;
    let contract = conv2d_bf16_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        owner.fingerprint().as_bytes(),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(PROVIDER).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        runtime.descriptor().id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(3),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    let kernels = RetainedFunctions::load(runtime)?;
    let mut providers = selection::providers_from_descriptors(descriptors, &kernels);
    providers.push(Box::new(ConvProvider {
        descriptor,
        kernels,
        owner,
    }));
    Ok(providers)
}

struct ConvProvider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    owner: Arc<CudaCudnnOwner>,
}
impl OperationResourceEstimator for ConvProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "CUDA convolution estimator received another operation",
            ));
        }
        let s = geometry(
            request.values(),
            request.attributes(),
            self.owner.workspace_limit_bytes(),
        )
        .map_err(invalid_plan)?;
        let scratch = if s.scratch_bytes == 0 {
            None
        } else {
            Some(ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::fixed(s.scratch_bytes)?,
                SCRATCH_ALIGNMENT,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?)
        };
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            SCRATCH_ALIGNMENT,
            scratch,
            None,
        ))
    }
}
impl OperationProvider<CudaDeviceRuntime> for ConvProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        i: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            i.participants()[0].identity().clone(),
            "cuda.conv2d.bf16.encode",
            "CUDA convolution requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(identity, "cuda.conv2d.bf16.encode", e))
    }
}
impl ConvProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = invocation.participants() else {
            return Err("CUDA convolution requires one participant".into());
        };
        if invocation.operation().id.as_str() != CONV2D_BF16_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA convolution requires its fixed tensor contract".into());
        }
        let s = geometry(
            p.bindings(),
            p.attributes(),
            self.owner.workspace_limit_bytes(),
        )?;
        let mut regions = Vec::with_capacity(5);
        for (ordinal, bytes) in [s.input_bytes, s.weight_bytes, s.bias_bytes]
            .into_iter()
            .enumerate()
        {
            regions.push(retained_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, ordinal as u32)?,
                bytes,
            )?);
        }
        let output = retained_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            s.output_bytes,
        )?;
        for input in &regions {
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        if s.scratch_bytes != 0 {
            let view = p
                .scratch_view()
                .ok_or("CUDA convolution has no admitted scratch")?;
            let translated = view
                .translate(0, s.scratch_bytes)
                .map_err(|e| e.to_string())?;
            let mut pieces = translated.iter();
            let piece = pieces.next().ok_or("CUDA convolution scratch is empty")?;
            if pieces.next().is_some() {
                return Err("CUDA convolution scratch must be contiguous".into());
            }
            let (buffer, range, retention) = piece.buffer_and_physical_range();
            let scratch = buffer
                .retained_region(range, retention)
                .map_err(|e| e.to_string())?;
            if scratch.element_type() != ElementType::U8
                || scratch.length_bytes() != s.scratch_bytes
                || scratch.device_ptr() % SCRATCH_ALIGNMENT != 0
            {
                return Err("CUDA convolution scratch dtype/extent/alignment mismatch".into());
            }
            for r in &regions {
                nonoverlapping_ranges(
                    r.device_ptr(),
                    r.length_bytes(),
                    scratch.device_ptr(),
                    scratch.length_bytes(),
                )?;
            }
            regions.push(scratch);
        } else if p.scratch_view().is_some() {
            return Err("CUDA convolution received unexpected scratch".into());
        }
        let owner = Arc::clone(&self.owner);
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            "vnext_conv2d_bf16_cudnn9",
            regions,
            move |stream, regions| {
                let scratch = if s.scratch_bytes == 0 {
                    0
                } else {
                    regions[4].device_ptr()
                };
                let mut input = regions[0].device_ptr();
                if s.padding_bytes != 0 {
                    let elements = s.padding_bytes / 2;
                    let mut launch = stream.launch_builder(kernels.spatial_function(0));
                    launch
                        .arg(&input)
                        .arg(&scratch)
                        .arg(&elements)
                        .arg(&s.height)
                        .arg(&s.width)
                        .arg(&s.padded_height)
                        .arg(&s.padded_width)
                        .arg(&s.top)
                        .arg(&s.left);
                    unsafe { launch.launch(flat_launch(elements)) }.map_err(|e| {
                        CudaDeviceRuntimeError::driver("BF16 convolution external padding", e)
                    })?;
                    input = scratch;
                }
                let workspace = if s.workspace_bytes == 0 {
                    0
                } else {
                    scratch.checked_add(s.workspace_offset).ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("cuDNN scratch address overflows")
                    })?
                };
                owner.convolution(
                    stream,
                    s.native,
                    input,
                    regions[1].device_ptr(),
                    regions[3].device_ptr(),
                    workspace,
                    s.workspace_bytes,
                )?;
                let output = regions[3].device_ptr();
                let bias = regions[2].device_ptr();
                let mut launch = stream.launch_builder(kernels.spatial_function(1));
                launch
                    .arg(&output)
                    .arg(&bias)
                    .arg(&s.elements)
                    .arg(&s.output_channels)
                    .arg(&s.output_spatial);
                unsafe { launch.launch(flat_launch(s.elements)) }
                    .map(|_| ())
                    .map_err(|e| CudaDeviceRuntimeError::driver("BF16 convolution bias", e))
            },
        )
        .and_then(|c| {
            c.with_tensor_work(
                s.elements,
                s.operations,
                2 + u64::from(s.padding_bytes != 0),
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}

fn flat_launch(elements: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            elements.div_ceil(THREADS as u64).min(MAX_BLOCKS) as u32,
            1,
            1,
        ),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    native: CudnnConv2dDescriptor,
    input_bytes: u64,
    weight_bytes: u64,
    bias_bytes: u64,
    output_bytes: u64,
    padding_bytes: u64,
    workspace_offset: u64,
    workspace_bytes: u64,
    scratch_bytes: u64,
    height: u64,
    width: u64,
    padded_height: u64,
    padded_width: u64,
    top: u64,
    left: u64,
    output_channels: u64,
    output_spatial: u64,
    elements: u64,
    operations: u64,
}
impl Shape {
    fn new(g: Conv2dBf16Geometry, workspace_bytes: u64) -> Result<Self, String> {
        if g.padding_mode() != Conv2dPaddingMode::ExplicitZero {
            return Err("CUDA convolution initially supports explicit_zero padding only".into());
        }
        let dim = |x: u64| {
            i32::try_from(x)
                .ok()
                .filter(|&x| x > 0)
                .ok_or_else(|| "CUDA convolution dimensions must be positive i32".to_owned())
        };
        let bytes = |a: &[u64]| {
            a.iter()
                .try_fold(2_u64, |n, x| n.checked_mul(*x))
                .filter(|&n| n <= isize::MAX as u64)
                .ok_or_else(|| "CUDA convolution byte extent exceeds address space".to_owned())
        };
        let pad = g.padding();
        let ph = g
            .input_height()
            .checked_add(pad[0])
            .and_then(|x| x.checked_add(pad[1]))
            .ok_or("padded height overflows")?;
        let pw = g
            .input_width()
            .checked_add(pad[2])
            .and_then(|x| x.checked_add(pad[3]))
            .ok_or("padded width overflows")?;
        for dimensions in [
            [g.batch(), g.input_channels(), ph, pw],
            [
                g.batch(),
                g.output_channels(),
                g.output_height(),
                g.output_width(),
            ],
            [
                g.output_channels(),
                g.input_channels() / g.groups(),
                g.kernel_height(),
                g.kernel_width(),
            ],
        ] {
            let elements = dimensions
                .into_iter()
                .try_fold(1_u64, |n, d| n.checked_mul(d))
                .ok_or("cuDNN tensor element count overflows")?;
            if elements > i32::MAX as u64 {
                return Err(
                    "cuDNN NCHW slice supports at most i32::MAX elements per native descriptor"
                        .into(),
                );
            }
        }
        let padding_bytes = if pad == [0; 4] {
            0
        } else {
            bytes(&[g.batch(), g.input_channels(), ph, pw])?
        };
        let (workspace_offset, scratch_bytes) = scratch_layout(padding_bytes, workspace_bytes)?;
        Ok(Self {
            native: CudnnConv2dDescriptor {
                input: [
                    dim(g.batch())?,
                    dim(g.input_channels())?,
                    dim(ph)?,
                    dim(pw)?,
                ],
                filter: [
                    dim(g.output_channels())?,
                    dim(g.input_channels() / g.groups())?,
                    dim(g.kernel_height())?,
                    dim(g.kernel_width())?,
                ],
                output: [
                    dim(g.batch())?,
                    dim(g.output_channels())?,
                    dim(g.output_height())?,
                    dim(g.output_width())?,
                ],
                padding: [0, 0],
                stride: [dim(g.stride()[0])?, dim(g.stride()[1])?],
                dilation: [dim(g.dilation()[0])?, dim(g.dilation()[1])?],
                groups: dim(g.groups())?,
            },
            input_bytes: bytes(&[
                g.batch(),
                g.input_channels(),
                g.input_height(),
                g.input_width(),
            ])?,
            weight_bytes: bytes(&[
                g.output_channels(),
                g.input_channels() / g.groups(),
                g.kernel_height(),
                g.kernel_width(),
            ])?,
            bias_bytes: bytes(&[g.output_channels()])?,
            output_bytes: bytes(&[g.element_count()])?,
            padding_bytes,
            workspace_offset,
            workspace_bytes,
            scratch_bytes,
            height: g.input_height(),
            width: g.input_width(),
            padded_height: ph,
            padded_width: pw,
            top: pad[0],
            left: pad[2],
            output_channels: g.output_channels(),
            output_spatial: g.output_height() * g.output_width(),
            elements: g.element_count(),
            operations: g.logical_arithmetic_operations(),
        })
    }
}
fn scratch_layout(pad: u64, workspace: u64) -> Result<(u64, u64), String> {
    let offset = pad
        .checked_add(SCRATCH_ALIGNMENT - 1)
        .map(|v| v / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT)
        .ok_or("CUDA convolution scratch alignment overflows")?;
    let bytes = offset
        .checked_add(workspace)
        .filter(|&v| v <= isize::MAX as u64)
        .ok_or("CUDA convolution scratch exceeds address space")?;
    Ok((offset, bytes))
}
fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    workspace: u64,
) -> Result<Shape, String> {
    if values.len() != 4 {
        return Err("CUDA convolution requires three inputs and one output".into());
    }
    let x = binding(values, ResolvedValueRole::Input, 0)?;
    let w = binding(values, ResolvedValueRole::Input, 1)?;
    let b = binding(values, ResolvedValueRole::Input, 2)?;
    let y = binding(values, ResolvedValueRole::Output, 0)?;
    for v in [x, w, b, y] {
        if v.weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("CUDA convolution requires dense weight storage".into());
        }
    }
    Shape::new(
        Conv2dBf16Geometry::new(x.tensor(), w.tensor(), b.tensor(), y.tensor(), attrs)
            .map_err(|e| e.to_string())?,
        workspace,
    )
}

#[cfg(test)]
mod tests;
