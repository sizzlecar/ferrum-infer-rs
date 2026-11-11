//! Explicit cuDNN F32/FMA convolution composition, retaining the native owner
//! and actual prepared workspace. No half or TF32 input conversion is allowed.
use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaCudnnConfig, CudaCudnnOwner, CudaCudnnSdpaCache, CudnnConv2dDescriptor, CudnnConvScalar,
    CudnnSdpaBindings, CudnnSdpaPlanPolicy, CudnnSdpaSignature, PreparedConv,
};
use ferrum_runtime::{
    conv2d_f32_contract, Conv2dF32Geometry, Conv2dF32PaddingMode, DynamicStorageRequirement,
    F32CnnGeometry, F32CnnOp, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, WeightFormatId, CONV2D_F32_CAPABILITY_ID,
    CONV2D_F32_OPERATION_ID,
};
pub const NUMERICAL_PROFILE_ID: &str =
    "cuda.cudnn9.conv2d.f32.fma_only.reflect_or_zero.separate_bias.v1";
const ALIGNMENT: u64 = 256;
pub(crate) fn providers(
    runtime: &CudaDeviceRuntime,
    config: &CudaCudnnConfig,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers_impl(runtime, config, selected, None, None, None)
}

pub(crate) fn providers_with_segmented_sdpa(
    runtime: &CudaDeviceRuntime,
    config: &CudaCudnnConfig,
    policy: CudnnSdpaPlanPolicy,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers_impl(runtime, config, selected, Some(policy), None, None)
}

pub(crate) fn providers_with_fused_f32(
    runtime: &CudaDeviceRuntime,
    config: &CudaCudnnConfig,
    policy: Option<CudnnSdpaPlanPolicy>,
    fused: &CudaCutlassFusedF32Config,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers_impl(runtime, config, selected, policy, Some(fused), None)
}

pub(crate) fn providers_with_native_options(
    runtime: &CudaDeviceRuntime,
    config: &CudaCudnnConfig,
    policy: Option<CudnnSdpaPlanPolicy>,
    fused: Option<&CudaCutlassFusedF32Config>,
    convrot: Option<&CudaConvRotFusedOutputConfig>,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    providers_impl(runtime, config, selected, policy, fused, convrot)
}

fn providers_impl(
    runtime: &CudaDeviceRuntime,
    config: &CudaCudnnConfig,
    selected: &[CudaTensorOperation],
    policy: Option<CudnnSdpaPlanPolicy>,
    fused: Option<&CudaCutlassFusedF32Config>,
    convrot: Option<&CudaConvRotFusedOutputConfig>,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    let cap = require_convolution_capability(runtime.descriptor())?;
    selection::validate_selection(selected)?;
    let wants_fused = selected.contains(&CudaTensorOperation::StructuredGqaAttentionFusedF32);
    if wants_fused != fused.is_some() {
        return Err(CudaDeviceRuntimeError::contract(
            "fused F32 selection and native allowance must be supplied together",
        ));
    }
    let fused_descriptor = fused
        .map(|config| {
            config.validate()?;
            gqa_fused_f32::descriptor(runtime.descriptor())
        })
        .transpose()?;
    let ordinary: Vec<_> = selected
        .iter()
        .copied()
        .filter(|op| *op != CudaTensorOperation::StructuredGqaAttentionFusedF32)
        .filter(|op| convrot.is_none() || !comfy_convrot::supports_fused_operation(*op))
        .collect();
    let mut descriptors = if ordinary.is_empty() {
        Vec::new()
    } else {
        selection::runtime_descriptors(runtime, &ordinary)?
    };
    if let Some(policy) = policy {
        select_segmented_sdpa(&mut descriptors, policy)?;
    }
    // One owner, one module and one optional cache. The convolution provider
    // and selected segmented provider retain these same original authorities.
    let owner = CudaCudnnOwner::new(runtime, config)?;
    let sdpa = policy
        .map(|policy| CudaCudnnSdpaCache::new(runtime, Arc::clone(&owner), policy).map(Arc::new))
        .transpose()?;
    let kernels = RetainedFunctions::load(runtime)?;
    let c = conv2d_f32_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        numerical_fingerprint().as_bytes(),
        owner.fingerprint().as_bytes(),
        include_bytes!("conv2d_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_cnn_f32.cuh"
        )),
    ]);
    let estimator = "resource-estimator.cuda.conv2d.f32.cudnn9_fma";
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.conv2d.f32.cudnn9_fma").map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(3),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    let mut providers = selection::providers_from_descriptors(descriptors, &kernels);
    if let (Some(descriptor), Some(config)) = (fused_descriptor, fused) {
        providers.push(gqa_fused_f32::provider(
            descriptor, &kernels, runtime, config,
        )?);
    }
    if let Some(config) = convrot {
        providers.extend(comfy_convrot::fused_providers(
            runtime, &kernels, config, selected,
        )?);
    }
    if let Some(cache) = sdpa {
        // No generic fused, cross-attention, spatial or F32 GQA replacement.
        // A missing/unsupported native plan is an explicit error, not fallback.
        providers.push(cudnn_segmented::provider(
            runtime.descriptor(),
            &kernels,
            cache,
        )?);
    }
    providers.push(Box::new(Provider {
        descriptor,
        kernels,
        owner,
    }));
    Ok(providers)
}

fn require_convolution_capability(
    device: &DeviceDescriptor,
) -> Result<CapabilityId, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(CONV2D_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks F32 convolution capability",
        ));
    }
    Ok(cap)
}

fn select_segmented_sdpa(
    descriptors: &mut Vec<selection::SelectedDescriptor>,
    policy: CudnnSdpaPlanPolicy,
) -> Result<(), CudaDeviceRuntimeError> {
    // Metadata-only mirror of the cache's closed finite bounds. This rejects
    // invalid opt-ins before native loading; Cache::new remains authoritative
    // and revalidates policy, context, cuDNN version and actual opaque credit.
    if policy.maximum_plans == 0
        || policy.maximum_live_bindings == 0
        || policy.opaque_memory_bytes == 0
        || policy.workspace_limit_bytes > isize::MAX as u64
        || policy
            .maximum_plans
            .checked_mul(std::mem::size_of::<CudnnSdpaSignature>())
            .is_none()
        || policy
            .maximum_live_bindings
            .checked_mul(std::mem::size_of::<CudnnSdpaBindings>())
            .is_none()
    {
        return Err(CudaDeviceRuntimeError::contract("cuDNN SDPA cache requires finite positive plan/binding/opaque limits and bounded workspace"));
    }
    let selected = |d: &selection::SelectedDescriptor| {
        matches!(
            d,
            selection::SelectedDescriptor::Fixed(CudaTensorOperation::SegmentedAttention, _)
        )
    };
    if !descriptors.iter().any(selected) {
        return Err(CudaDeviceRuntimeError::contract(
            "explicit segmented SDPA requires selected SegmentedAttention",
        ));
    }
    descriptors.retain(|d| !selected(d));
    Ok(())
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    owner: Arc<CudaCudnnOwner>,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        r: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if r.operation().id != *self.descriptor.operation_id()
            || r.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "F32 convolution estimator received another operation",
            ));
        }
        let mut shape = geometry(r.values(), r.attributes(), 0).map_err(invalid_plan)?;
        let prepared = self
            .owner
            .prepare_convolution(shape.native)
            .map_err(|e| invalid_plan(e.to_string()))?;
        shape = shape
            .with_workspace(prepared.workspace_bytes())
            .map_err(invalid_plan)?;
        let scratch = if shape.scratch_bytes == 0 {
            None
        } else {
            Some(ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::fixed(shape.scratch_bytes)?,
                ALIGNMENT,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?)
        };
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            ALIGNMENT,
            scratch,
            None,
        )
        .with_prepared_implementation_fingerprint(prepared.fingerprint().to_owned())?
        .with_prepared_implementation_retention(prepared)
    }
}
impl OperationProvider<CudaDeviceRuntime> for Provider {
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
            "cuda.conv2d.f32.encode",
            "requires fixed tensor invocation".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.conv2d.f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("F32 convolution requires one participant".into());
        };
        if i.operation().id.as_str() != CONV2D_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed F32 convolution contract".into());
        }
        let prepared = i
            .prepared_implementation::<PreparedConv>()
            .ok_or("F32 convolution has no retained prepared implementation")?;
        let shape = geometry(p.bindings(), p.attributes(), prepared.workspace_bytes())?;
        prepared.validate_invocation(
            &self.owner,
            shape.native,
            i.prepared_implementation_fingerprint(),
        )?;
        let mut regions = Vec::with_capacity(5);
        for n in 0..3 {
            regions.push(retained_typed_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, n as u32)?,
                shape
                    .geometry
                    .input_elements(n)
                    .ok_or("missing input extent")?
                    * 4,
                ElementType::F32,
            )?);
        }
        let output = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            shape.geometry.elements() * 4,
            ElementType::F32,
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
        if shape.scratch_bytes > 0 {
            let view = p
                .scratch_view()
                .ok_or("F32 convolution missing admitted scratch")?;
            let translated = view
                .translate(0, shape.scratch_bytes)
                .map_err(|e| e.to_string())?;
            let mut pieces = translated.iter();
            let piece = pieces.next().ok_or("empty scratch")?;
            if pieces.next().is_some() {
                return Err("scratch must be contiguous".into());
            }
            let (buffer, range, retention) = piece.buffer_and_physical_range();
            let scratch = buffer
                .retained_region(range, retention)
                .map_err(|e| e.to_string())?;
            if scratch.element_type() != ElementType::U8
                || scratch.length_bytes() != shape.scratch_bytes
                || scratch.device_ptr() % ALIGNMENT != 0
            {
                return Err("F32 convolution scratch dtype/extent/alignment mismatch".into());
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
            return Err("unexpected F32 convolution scratch".into());
        }
        let kernels = Arc::clone(&self.kernels);
        let g = shape.g();
        let elements = g.elements();
        CudaDeviceCommand::operation(
            "vnext_conv2d_f32_cudnn9_fma",
            regions,
            move |stream, regions| {
                let scratch = if shape.scratch_bytes > 0 {
                    regions[4].device_ptr()
                } else {
                    0
                };
                let mut input = regions[0].device_ptr();
                if shape.padding_bytes > 0 {
                    let elements = shape.padding_bytes / 4;
                    let height = g.input_height();
                    let width = g.input_width();
                    let [top, _, left, _] = g.padding();
                    let reflect = u32::from(g.padding_mode() == Conv2dF32PaddingMode::Reflect);
                    let mut launch = stream.launch_builder(kernels.cnn_f32_function(0));
                    launch
                        .arg(&input)
                        .arg(&scratch)
                        .arg(&elements)
                        .arg(&height)
                        .arg(&width)
                        .arg(&shape.padded_height)
                        .arg(&shape.padded_width)
                        .arg(&top)
                        .arg(&left)
                        .arg(&reflect);
                    unsafe { launch.launch(flat(elements)) }.map_err(|e| {
                        CudaDeviceRuntimeError::driver("F32 convolution explicit padding", e)
                    })?;
                    input = scratch;
                }
                let workspace = if shape.workspace_bytes > 0 {
                    scratch.checked_add(shape.workspace_offset).ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "F32 convolution workspace address overflows",
                        )
                    })?
                } else {
                    0
                };
                let output = regions[3].device_ptr();
                prepared.convolution(
                    stream,
                    input,
                    regions[1].device_ptr(),
                    output,
                    workspace,
                    shape.workspace_bytes,
                )?;
                let bias = regions[2].device_ptr();
                let channels = g.output_channels();
                let spatial = g.output_height() * g.output_width();
                let mut launch = stream.launch_builder(kernels.cnn_f32_function(1));
                launch
                    .arg(&output)
                    .arg(&bias)
                    .arg(&elements)
                    .arg(&channels)
                    .arg(&spatial);
                unsafe { launch.launch(flat(elements)) }
                    .map(|_| ())
                    .map_err(|e| CudaDeviceRuntimeError::driver("F32 convolution separate bias", e))
            },
        )
        .and_then(|c| {
            c.with_tensor_work(
                elements,
                g.logical_arithmetic_operations(),
                2 + u64::from(shape.padding_bytes > 0),
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}
fn flat(elements: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (elements.div_ceil(256).min(MAX_BLOCKS) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
#[derive(Clone, Copy, Debug)]
struct Shape {
    native: CudnnConv2dDescriptor,
    geometry: F32CnnGeometry,
    padding_bytes: u64,
    padded_height: u64,
    padded_width: u64,
    workspace_offset: u64,
    workspace_bytes: u64,
    scratch_bytes: u64,
}
impl Shape {
    fn g(self) -> Conv2dF32Geometry {
        let F32CnnGeometry::Conv2d(g) = self.geometry else {
            unreachable!("convolution shape")
        };
        g
    }
    fn with_workspace(mut self, bytes: u64) -> Result<Self, String> {
        self.workspace_bytes = bytes;
        self.workspace_offset = self
            .padding_bytes
            .checked_add(ALIGNMENT - 1)
            .ok_or("scratch alignment overflows")?
            / ALIGNMENT
            * ALIGNMENT;
        self.scratch_bytes = self
            .workspace_offset
            .checked_add(bytes)
            .filter(|n| *n <= isize::MAX as u64)
            .ok_or("scratch exceeds address space")?;
        Ok(self)
    }
    fn new(g: Conv2dF32Geometry, workspace: u64) -> Result<Self, String> {
        let [top, bottom, left, right] = g.padding();
        let ph = g.input_height() + top + bottom;
        let pw = g.input_width() + left + right;
        let dim = |n: u64| {
            i32::try_from(n)
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| "cuDNN F32 dimensions require positive i32".to_owned())
        };
        let tensor = |v: [u64; 4]| -> Result<[i32; 4], String> {
            v.into_iter()
                .try_fold(1u64, |n, x| n.checked_mul(x))
                .filter(|n| *n <= i32::MAX as u64)
                .ok_or("cuDNN F32 tensor numel exceeds i32")?;
            Ok([dim(v[0])?, dim(v[1])?, dim(v[2])?, dim(v[3])?])
        };
        let native = CudnnConv2dDescriptor {
            scalar: CudnnConvScalar::F32Fma,
            input: tensor([g.batch(), g.input_channels(), ph, pw])?,
            filter: tensor([
                g.output_channels(),
                g.input_channels() / g.groups(),
                g.kernel_height(),
                g.kernel_width(),
            ])?,
            output: tensor([
                g.batch(),
                g.output_channels(),
                g.output_height(),
                g.output_width(),
            ])?,
            padding: [0, 0],
            stride: [dim(g.stride()[0])?, dim(g.stride()[1])?],
            dilation: [dim(g.dilation()[0])?, dim(g.dilation()[1])?],
            groups: dim(g.groups())?,
        };
        let padding_bytes = if g.padding() == [0; 4] {
            0
        } else {
            g.batch()
                .checked_mul(g.input_channels())
                .and_then(|n| n.checked_mul(ph))
                .and_then(|n| n.checked_mul(pw))
                .and_then(|n| n.checked_mul(4))
                .filter(|n| *n <= isize::MAX as u64)
                .ok_or("F32 padding bytes overflow")?
        };
        Self {
            native,
            geometry: F32CnnGeometry::Conv2d(g),
            padding_bytes,
            padded_height: ph,
            padded_width: pw,
            workspace_offset: 0,
            workspace_bytes: 0,
            scratch_bytes: 0,
        }
        .with_workspace(workspace)
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    workspace: u64,
) -> Result<Shape, String> {
    let F32CnnGeometry::Conv2d(g) = cnn_f32::geometry(F32CnnOp::Conv2d, b, attrs)? else {
        unreachable!()
    };
    Shape::new(g, workspace)
}

#[cfg(test)]
mod tests;
