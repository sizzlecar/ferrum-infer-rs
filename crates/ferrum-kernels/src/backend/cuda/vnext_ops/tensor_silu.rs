//! Eager BF16 pointwise providers sharing one bounded native module.
//!
//! Each operation keeps its own descriptor, estimator and numerical identity.
//! Source/flags/PTX identity is not a claim that CUDA numerical qualification
//! passed, nor that libdevice and the Reference libm profile agree bitwise.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use ferrum_runtime::{
    gated_residual_bf16_contract, gelu_tanh_bf16_contract, modulation_scale_bf16_contract,
    silu_bf16_contract, silu_mul_bf16_contract, tanh_bf16_contract, AttributeId,
    BatchedOperationInvocation, CapabilityId, ContractVersion, DeviceDescriptor, DeviceRuntime,
    ElementType, EncodedDeviceOperation, InvocationWorkProjection, NodeWorkContract,
    OperationContract, OperationFailure, OperationInvocation, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProviderExecutionSemantics, ProviderId, ResolvedValueBinding,
    ResolvedValueRole, ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue,
    StandardOperationContract, TensorOperationInvocation, VNextError,
    GATED_RESIDUAL_BF16_CAPABILITY_ID, GATED_RESIDUAL_BF16_OPERATION_ID,
    GELU_TANH_BF16_CAPABILITY_ID, GELU_TANH_BF16_OPERATION_ID, MODULATION_SCALE_BF16_CAPABILITY_ID,
    MODULATION_SCALE_BF16_OPERATION_ID, SILU_BF16_CAPABILITY_ID, SILU_BF16_OPERATION_ID,
    SILU_MUL_BF16_CAPABILITY_ID, SILU_MUL_BF16_OPERATION_ID, TANH_BF16_CAPABILITY_ID,
    TANH_BF16_OPERATION_ID,
};

use super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};
use super::{binding, contract_error, implementation_fingerprint, provider_failure};

mod attention;
mod bundle;
mod geometry;
mod norm;
mod rope;
mod selection;
mod text_projection;
mod transport;
pub use attention::{
    NUMERICAL_IDENTITY as ATTENTION_NUMERICAL_IDENTITY,
    NUMERICAL_PROFILE_ID as ATTENTION_NUMERICAL_PROFILE_ID,
};
use bundle::RetainedFunctions;
use geometry::{geometry, Shape};
pub use norm::{
    HEAD_RMS_NORM_PROFILE_ID, LAYER_NORM_PROFILE_ID, NUMERICAL_IDENTITY as NORM_NUMERICAL_IDENTITY,
};
pub use rope::{
    NUMERICAL_IDENTITY as ROPE_NUMERICAL_IDENTITY,
    NUMERICAL_PROFILE_ID as ROPE_NUMERICAL_PROFILE_ID,
};
pub use selection::CudaTensorOperation;
pub use text_projection::{
    GELU_TANH_PROFILE_ID, NUMERICAL_IDENTITY as TEXT_PROJECTION_NUMERICAL_IDENTITY,
    ZERO_CENTERED_RMS_NORM_PROFILE_ID,
};
pub use transport::{
    BATCHED_ROW_CONCAT_PROFILE_ID, NUMERICAL_IDENTITY as TRANSPORT_NUMERICAL_IDENTITY,
    ROW_GATHER_PROFILE_ID,
};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.libdevice.precise.silu.bf16.v1";
pub const SILU_MUL_NUMERICAL_PROFILE_ID: &str = "cuda.libdevice.precise.silu_mul.bf16.v1";
pub const TANH_NUMERICAL_PROFILE_ID: &str = "cuda.libdevice.precise.tanh.bf16.v1";
pub const MODULATION_NUMERICAL_PROFILE_ID: &str = "cuda.ieee.modulation_scale.bf16.v1";
pub const GATED_RESIDUAL_NUMERICAL_PROFILE_ID: &str = "cuda.ieee.gated_residual.bf16.v1";
pub const NUMERICAL_PROFILE_IDENTITY: &str =
    "libdevice-expf;fadd-rn;fdiv-rn;integer-bf16-rne;ftz=false;prec-div=true;prec-sqrt=true;fmad=false;no-fast-math;nan-class-only";
pub const BUNDLE_NUMERICAL_IDENTITY: &str = concat!(
    "fixed-bf16-pointwise-bundle-v1;gradual-underflow;no-fast-math;fmad=false;",
    "B=integer-bf16-rne;RN32-explicit-add-mul-div;NaN-class-only;",
    "silu=B(RN32(x/RN32(1+libdevice-expf(-x))));",
    "silu_mul=B(RN32(decode(silu(g))*u));tanh=B(libdevice-tanhf(x));",
    "modulation=B(RN32(x*decode(B(RN32(1+s)))));",
    "gated_residual=B(RN32(x+decode(B(RN32(g*u)))));",
    "broadcast=explicit-elementwise-or-row;logical-work=4,5,1,2,2;",
    "qualification-requires-target-evidence;no-Reference-or-Torch-bitwise-claim"
);
const THREADS: u32 = 256;
const MAX_BLOCKS: u64 = 65_535;
const ALIGNMENT: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Silu,
    SiluMul,
    Tanh,
    Modulation,
    GatedResidual,
    GeluTanh,
}

impl Kind {
    #[cfg(test)]
    pub(super) const ALL: [Self; 6] = [
        Self::Silu,
        Self::SiluMul,
        Self::Tanh,
        Self::Modulation,
        Self::GatedResidual,
        Self::GeluTanh,
    ];

    fn index(self) -> usize {
        match self {
            Self::Silu => 0,
            Self::SiluMul => 1,
            Self::Tanh => 2,
            Self::Modulation => 3,
            Self::GatedResidual => 4,
            Self::GeluTanh => 5,
        }
    }
    fn function(self) -> &'static str {
        match self {
            Self::Silu => "ferrum_vnext_silu_bf16",
            Self::SiluMul => "ferrum_vnext_silu_mul_bf16",
            Self::Tanh => "ferrum_vnext_tanh_bf16",
            Self::Modulation => "ferrum_vnext_modulation_scale_bf16",
            Self::GatedResidual => "ferrum_vnext_gated_residual_bf16",
            Self::GeluTanh => "ferrum_vnext_gelu_tanh_bf16",
        }
    }
    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::Silu => SILU_BF16_CAPABILITY_ID,
            Self::SiluMul => SILU_MUL_BF16_CAPABILITY_ID,
            Self::Tanh => TANH_BF16_CAPABILITY_ID,
            Self::Modulation => MODULATION_SCALE_BF16_CAPABILITY_ID,
            Self::GatedResidual => GATED_RESIDUAL_BF16_CAPABILITY_ID,
            Self::GeluTanh => GELU_TANH_BF16_CAPABILITY_ID,
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::Silu => SILU_BF16_OPERATION_ID,
            Self::SiluMul => SILU_MUL_BF16_OPERATION_ID,
            Self::Tanh => TANH_BF16_OPERATION_ID,
            Self::Modulation => MODULATION_SCALE_BF16_OPERATION_ID,
            Self::GatedResidual => GATED_RESIDUAL_BF16_OPERATION_ID,
            Self::GeluTanh => GELU_TANH_BF16_OPERATION_ID,
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Silu => silu_bf16_contract(),
            Self::SiluMul => silu_mul_bf16_contract(),
            Self::Tanh => tanh_bf16_contract(),
            Self::Modulation => modulation_scale_bf16_contract(),
            Self::GatedResidual => gated_residual_bf16_contract(),
            Self::GeluTanh => gelu_tanh_bf16_contract(),
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::Silu => NUMERICAL_PROFILE_ID,
            Self::SiluMul => SILU_MUL_NUMERICAL_PROFILE_ID,
            Self::Tanh => TANH_NUMERICAL_PROFILE_ID,
            Self::Modulation => MODULATION_NUMERICAL_PROFILE_ID,
            Self::GatedResidual => GATED_RESIDUAL_NUMERICAL_PROFILE_ID,
            Self::GeluTanh => GELU_TANH_PROFILE_ID,
        }
    }
    fn provider(self) -> &'static str {
        match self {
            Self::Silu => "provider.cuda.silu.bf16.libdevice",
            Self::SiluMul => "provider.cuda.silu_mul.bf16.libdevice",
            Self::Tanh => "provider.cuda.tanh.bf16.libdevice",
            Self::Modulation => "provider.cuda.modulation_scale.bf16.ieee",
            Self::GatedResidual => "provider.cuda.gated_residual.bf16.ieee",
            Self::GeluTanh => "provider.cuda.gelu_tanh.bf16.libdevice",
        }
    }
    fn estimator(self) -> &'static str {
        match self {
            Self::Silu => "resource-estimator.cuda.silu.bf16.libdevice",
            Self::SiluMul => "resource-estimator.cuda.silu_mul.bf16.libdevice",
            Self::Tanh => "resource-estimator.cuda.tanh.bf16.libdevice",
            Self::Modulation => "resource-estimator.cuda.modulation_scale.bf16.ieee",
            Self::GatedResidual => "resource-estimator.cuda.gated_residual.bf16.ieee",
            Self::GeluTanh => "resource-estimator.cuda.gelu_tanh.bf16.libdevice",
        }
    }
    fn inputs(self) -> u32 {
        match self {
            Self::Silu | Self::Tanh | Self::GeluTanh => 1,
            Self::SiluMul | Self::Modulation => 2,
            Self::GatedResidual => 3,
        }
    }
    fn operations_per_element(self) -> u64 {
        match self {
            Self::Silu => 4,
            Self::SiluMul => 5,
            Self::Tanh => 1,
            Self::Modulation | Self::GatedResidual => 2,
            Self::GeluTanh => 9,
        }
    }
    fn broadcasts(self) -> bool {
        matches!(self, Self::Modulation | Self::GatedResidual)
    }
}

pub(super) fn numerical_fingerprint() -> String {
    implementation_fingerprint(&[
        NUMERICAL_PROFILE_IDENTITY.as_bytes(),
        BUNDLE_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("tensor_silu.rs"),
        include_bytes!("tensor_silu/bundle.rs"),
        include_bytes!("tensor_silu/geometry.rs"),
        include_bytes!("tensor_silu/norm.rs"),
        include_bytes!("tensor_silu/rope.rs"),
        include_bytes!("tensor_silu/attention.rs"),
        include_bytes!("tensor_silu/selection.rs"),
        include_bytes!("tensor_silu/text_projection.rs"),
        include_bytes!("tensor_silu/transport.rs"),
        TRANSPORT_NUMERICAL_IDENTITY.as_bytes(),
        TEXT_PROJECTION_NUMERICAL_IDENTITY.as_bytes(),
        norm::NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_bf16_silu.cu"
        )),
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs")),
        crate::ptx::VNEXT_BF16_SILU.as_bytes(),
    ])
}

fn descriptor(
    device: &DeviceDescriptor,
    kind: Kind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks the selected BF16 pointwise capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        kind.profile().as_bytes(),
        kind.provider().as_bytes(),
    ]);
    let estimator =
        implementation_fingerprint(&[kind.estimator().as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(kind.provider()).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::new(),
        BTreeSet::new(),
        super::transformer::contiguous_bindings(kind.inputs()),
        kind.estimator(),
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)
    .map(|descriptor| descriptor.with_fixed_tensor_work(true))
}

pub(super) struct CudaTensorPointwiseProvider {
    descriptor: OperationProviderDescriptor,
    kind: Kind,
    kernels: Arc<RetainedFunctions>,
}

/// Validate the requested subset before loading one shared native module.
pub(super) fn selected_providers(
    runtime: &CudaDeviceRuntime,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    selection::providers(runtime, selected)
}

/// Compatibility factories keep their original ordered operation subsets.
pub(super) fn providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    selected_providers(runtime, &CudaTensorOperation::ALL[..5])
}

pub(super) fn component_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    selected_providers(runtime, &CudaTensorOperation::ALL[..7])
}

pub(super) fn all_component_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    selected_providers(runtime, &CudaTensorOperation::ALL[..8])
}

pub(super) fn attention_providers(
    runtime: &CudaDeviceRuntime,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    selected_providers(runtime, &CudaTensorOperation::ALL[..9])
}

impl OperationResourceEstimator for CudaTensorPointwiseProvider {
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
                "CUDA pointwise estimator received another operation",
            ));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            ALIGNMENT,
            None,
            None,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaTensorPointwiseProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "cuda.pointwise.bf16.encode",
            "CUDA BF16 pointwise supports fixed tensors only".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|message| provider_failure(identity, "cuda.pointwise.bf16.encode", message))
    }
}

impl CudaTensorPointwiseProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA BF16 pointwise requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.kind.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA BF16 pointwise requires its fixed tensor contract".into());
        }
        let shape: Shape = geometry(self.kind, participant.bindings(), participant.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.inputs() as usize + 1);
        for ordinal in 0..self.kind.inputs() {
            let bytes = if self.kind.broadcasts() && ordinal == 1 {
                shape.parameter_bytes
            } else {
                shape.bytes
            };
            regions.push(retained_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                bytes,
            )?);
        }
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.bytes,
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
        let kernels = Arc::clone(&self.kernels);
        let kind = self.kind;
        CudaDeviceCommand::operation(kind.function(), regions, move |stream, regions| {
            // Each command retains the entire shared module and its permit.
            let mut builder = stream.launch_builder(kernels.function(kind));
            let input = regions[0].device_ptr();
            let parameter = regions.get(1).map_or(0, CudaBufferRegion::device_ptr);
            let update = regions.get(2).map_or(0, CudaBufferRegion::device_ptr);
            let output = regions[kind.inputs() as usize].device_ptr();
            let row_broadcast = u32::from(shape.row_broadcast);
            match kind {
                Kind::Silu | Kind::Tanh | Kind::GeluTanh => {
                    builder.arg(&input).arg(&output).arg(&shape.elements);
                }
                Kind::SiluMul => {
                    builder
                        .arg(&input)
                        .arg(&parameter)
                        .arg(&output)
                        .arg(&shape.elements);
                }
                Kind::Modulation => {
                    builder
                        .arg(&input)
                        .arg(&parameter)
                        .arg(&output)
                        .arg(&shape.elements)
                        .arg(&shape.width)
                        .arg(&row_broadcast);
                }
                Kind::GatedResidual => {
                    builder
                        .arg(&input)
                        .arg(&parameter)
                        .arg(&update)
                        .arg(&output)
                        .arg(&shape.elements)
                        .arg(&shape.width)
                        .arg(&row_broadcast);
                }
            }
            let blocks = shape.elements.div_ceil(u64::from(THREADS)).min(MAX_BLOCKS) as u32;
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("BF16 pointwise launch", error))
        })
        .and_then(|command| {
            command.with_tensor_work(shape.elements, shape.arithmetic_operations, 1, 0)
        })
        .map_err(|error| error.to_string())
    }
}

fn retained_region<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
    bytes: u64,
) -> Result<CudaBufferRegion, String> {
    retained_typed_region(participant, binding, bytes, ElementType::Bf16)
}

fn retained_typed_region<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
    bytes: u64,
    dtype: ElementType,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA pointwise requires one physical storage component".into());
    };
    if component.element_type() != dtype || component.length_bytes() != bytes {
        return Err("CUDA component storage differs from its typed tensor".into());
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or("CUDA pointwise has no resource view")?;
    let translated = view
        .translate(component.offset_bytes(), bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or("CUDA pointwise translated to no physical region")?;
    if physical.next().is_some() {
        return Err("CUDA pointwise requires contiguous physical storage".into());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())?;
    if region.element_type() != dtype
        || region.length_bytes() != bytes
        || region.device_ptr() % ALIGNMENT != 0
    {
        return Err("CUDA component physical region differs from its aligned typed tensor".into());
    }
    Ok(region)
}

fn nonoverlapping_ranges(
    input: u64,
    input_bytes: u64,
    output: u64,
    output_bytes: u64,
) -> Result<(), String> {
    let input_end = input
        .checked_add(input_bytes)
        .ok_or("CUDA pointwise input address overflows")?;
    let output_end = output
        .checked_add(output_bytes)
        .ok_or("CUDA pointwise output address overflows")?;
    if input < output_end && output < input_end {
        return Err("CUDA pointwise output overlaps a read-only input".into());
    }
    Ok(())
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
