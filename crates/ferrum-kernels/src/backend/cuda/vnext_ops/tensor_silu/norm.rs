//! Bounded shared-memory BF16 normalization, sharing the pointwise module.
//! The native profile uses an adjacent-pair FP32 tree and separate sqrt/div;
//! it does not claim bitwise equivalence to deployment rsqrt implementations.

use ferrum_runtime::{
    head_rms_norm_bf16_contract, layer_norm_bf16_contract, CanonicalRational,
    HeadRmsNormBf16Geometry, LayerNormBf16Geometry, PhysicalWeightLayout, ResolvedTensorSpec,
    WeightFormatId, HEAD_RMS_NORM_BF16_CAPABILITY_ID, HEAD_RMS_NORM_BF16_OPERATION_ID,
    LAYER_NORM_BF16_CAPABILITY_ID, LAYER_NORM_BF16_OPERATION_ID,
};

use super::*;

pub const LAYER_NORM_PROFILE_ID: &str = "cuda.ieee.layer_norm.bf16.pairwise_sqrt_div.v1";
pub const HEAD_RMS_NORM_PROFILE_ID: &str = "cuda.ieee.head_rms_norm.bf16.pairwise_sqrt_div.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "power-of-two-width-1..4096;adjacent-pair-tree-no-padding-no-initial-zero;",
    "explicit-RN32-add-sub-mul-div-sqrt;separate-sqrt-reciprocal;gradual-underflow;",
    "epsilon=(numerator-as-f64/denominator-as-f64)-as-f32;",
    "LN=two-pass-population-B(center*inverse);",
    "HeadRMS=B(decode(B(x*inverse))*gamma);shared-gamma-per-head;",
    "no-zero-centered-affine;no-rotary;NaN-class-only;",
    "zero-global-scratch;dynamic-CTA-shared=4*width;static-CTA-scalars<=8-bytes;",
    "logical-work-LN=groups*(6*width+3);HeadRMS=groups*(4*width+3);",
    "no-deployment-rsqrt-or-Torch-bitwise-claim;target-validation-required"
);
const MAX_WIDTH: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Layer,
    HeadRms,
}

impl Kind {
    pub(super) const ALL: [Self; 2] = [Self::Layer, Self::HeadRms];

    pub(super) fn index(self) -> usize {
        match self {
            Self::Layer => 0,
            Self::HeadRms => 1,
        }
    }
    pub(super) fn function(self) -> &'static str {
        match self {
            Self::Layer => "ferrum_vnext_layer_norm_bf16",
            Self::HeadRms => "ferrum_vnext_head_rms_norm_bf16",
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::Layer => LAYER_NORM_BF16_OPERATION_ID,
            Self::HeadRms => HEAD_RMS_NORM_BF16_OPERATION_ID,
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::Layer => LAYER_NORM_BF16_CAPABILITY_ID,
            Self::HeadRms => HEAD_RMS_NORM_BF16_CAPABILITY_ID,
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Layer => layer_norm_bf16_contract(),
            Self::HeadRms => head_rms_norm_bf16_contract(),
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::Layer => LAYER_NORM_PROFILE_ID,
            Self::HeadRms => HEAD_RMS_NORM_PROFILE_ID,
        }
    }
    fn provider(self) -> &'static str {
        match self {
            Self::Layer => "provider.cuda.layer_norm.bf16.pairwise_sqrt_div",
            Self::HeadRms => "provider.cuda.head_rms_norm.bf16.pairwise_sqrt_div",
        }
    }
    fn estimator(self) -> &'static str {
        match self {
            Self::Layer => "resource-estimator.cuda.layer_norm.bf16.pairwise_sqrt_div",
            Self::HeadRms => "resource-estimator.cuda.head_rms_norm.bf16.pairwise_sqrt_div",
        }
    }
    fn inputs(self) -> u32 {
        match self {
            Self::Layer => 1,
            Self::HeadRms => 2,
        }
    }
    fn width_attribute(self) -> &'static str {
        match self {
            Self::Layer => "hidden_size",
            Self::HeadRms => "head_dim",
        }
    }
}

pub(super) fn descriptors(
    device: &DeviceDescriptor,
) -> Result<Vec<(Kind, OperationProviderDescriptor)>, CudaDeviceRuntimeError> {
    Kind::ALL
        .into_iter()
        .map(|kind| descriptor(device, kind).map(|d| (kind, d)))
        .collect()
}

fn descriptor(
    device: &DeviceDescriptor,
    kind: Kind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks selected BF16 normalization capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        kind.profile().as_bytes(),
        kind.provider().as_bytes(),
    ]);
    let estimator =
        implementation_fingerprint(&[kind.estimator().as_bytes(), implementation.as_bytes()]);
    let weight_formats = if kind == Kind::HeadRms {
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ])
    } else {
        BTreeSet::new()
    };
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
        weight_formats,
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.inputs()),
        kind.estimator(),
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)
    .map(|d| d.with_fixed_tensor_work(true))
}

pub(super) fn providers(
    descriptors: Vec<(Kind, OperationProviderDescriptor)>,
    kernels: &Arc<RetainedFunctions>,
) -> Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>> {
    descriptors
        .into_iter()
        .map(|(kind, descriptor)| {
            Box::new(NormProvider {
                descriptor,
                kind,
                kernels: Arc::clone(kernels),
            }) as Box<dyn OperationProvider<CudaDeviceRuntime>>
        })
        .collect()
}

struct NormProvider {
    descriptor: OperationProviderDescriptor,
    kind: Kind,
    kernels: Arc<RetainedFunctions>,
}

impl OperationResourceEstimator for NormProvider {
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
                "CUDA normalization estimator received another operation",
            ));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
        // Bounded CTA shared memory is declared by the launch/profile. There
        // is no device allocation or global scratch outside admitted views.
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

impl OperationProvider<CudaDeviceRuntime> for NormProvider {
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
            "cuda.norm.bf16.encode",
            "CUDA BF16 normalization supports fixed tensors only".into(),
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
            .map_err(|message| provider_failure(identity, "cuda.norm.bf16.encode", message))
    }
}

impl NormProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA BF16 normalization requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.kind.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA BF16 normalization requires its fixed tensor contract".into());
        }
        let shape = geometry(self.kind, participant.bindings(), participant.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.inputs() as usize + 1);
        for ordinal in 0..self.kind.inputs() {
            let bytes = if ordinal == 1 {
                shape.gamma_bytes
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
            let input = regions[0].device_ptr();
            let gamma = regions[1].device_ptr();
            let output = regions[kind.inputs() as usize].device_ptr();
            let mut builder = stream.launch_builder(kernels.norm_function(kind));
            builder.arg(&input);
            if kind == Kind::HeadRms {
                builder.arg(&gamma);
            }
            builder
                .arg(&output)
                .arg(&shape.groups)
                .arg(&shape.width)
                .arg(&shape.epsilon);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (shape.groups.min(MAX_BLOCKS) as u32, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: shape.shared_bytes,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("BF16 normalization launch", e))
        })
        .and_then(|c| c.with_tensor_work(shape.elements, shape.arithmetic_operations, 1, 0))
        .map_err(|e| e.to_string())
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    groups: u64,
    width: u32,
    elements: u64,
    bytes: u64,
    gamma_bytes: u64,
    arithmetic_operations: u64,
    epsilon: f32,
    shared_bytes: u32,
}

impl Shape {
    fn new(kind: Kind, rows: u64, columns: u64, width: u64, epsilon: f32) -> Result<Self, String> {
        if rows == 0
            || columns == 0
            || !width.is_power_of_two()
            || width > MAX_WIDTH
            || columns % width != 0
            || (kind == Kind::Layer && columns != width)
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(
                "CUDA BF16 normalization requires positive geometry and power-of-two width <=4096"
                    .into(),
            );
        }
        let elements = rows
            .checked_mul(columns)
            .ok_or("CUDA norm element count overflows")?;
        let bytes = elements
            .checked_mul(2)
            .filter(|&b| b <= isize::MAX as u64)
            .ok_or("CUDA norm byte extent exceeds address space")?;
        let groups = elements / width;
        let operations_per_width = if kind == Kind::Layer { 6_u64 } else { 4_u64 };
        let arithmetic_operations = width
            .checked_mul(operations_per_width)
            .and_then(|v| v.checked_add(3))
            .and_then(|v| v.checked_mul(groups))
            .ok_or("CUDA norm logical arithmetic work overflows")?;
        Ok(Self {
            groups,
            width: width as u32,
            elements,
            bytes,
            gamma_bytes: width * 2,
            arithmetic_operations,
            epsilon,
            shared_bytes: (width * 4) as u32,
        })
    }
}

fn geometry(
    kind: Kind,
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != kind.inputs() as usize + 1 || attributes.len() != 2 {
        return Err("CUDA normalization has an incorrect port or attribute count".into());
    }
    let get = |name| {
        attributes
            .iter()
            .find(|(key, _)| key.as_str() == name)
            .map(|(_, value)| value)
    };
    let Some(SemanticValue::Unsigned(width)) = get(kind.width_attribute()) else {
        return Err("CUDA normalization requires an unsigned width attribute".into());
    };
    let Some(SemanticValue::Rational(epsilon)) = get("epsilon") else {
        return Err("CUDA normalization requires rational epsilon".into());
    };
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    let gamma = if kind == Kind::HeadRms {
        let gamma = binding(bindings, ResolvedValueRole::Input, 1)?;
        if gamma
            .weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("CUDA HeadRmsNorm cannot consume packed or quantized weights".into());
        }
        Some(gamma.tensor())
    } else {
        None
    };
    validate_specs(
        kind,
        input.tensor(),
        gamma,
        output.tensor(),
        *width,
        *epsilon,
    )
}

fn validate_specs(
    kind: Kind,
    input: &ResolvedTensorSpec,
    gamma: Option<&ResolvedTensorSpec>,
    output: &ResolvedTensorSpec,
    width: u64,
    epsilon: CanonicalRational,
) -> Result<Shape, String> {
    match kind {
        Kind::Layer => {
            if gamma.is_some() {
                return Err("CUDA LayerNorm has no affine operand".into());
            }
            let g = LayerNormBf16Geometry::new(input, output, width, epsilon)
                .map_err(|e| e.to_string())?;
            Shape::new(
                kind,
                g.rows(),
                g.hidden_size(),
                g.hidden_size(),
                g.effective_epsilon(),
            )
        }
        Kind::HeadRms => {
            let gamma = gamma.ok_or("CUDA HeadRmsNorm requires gamma")?;
            let g = HeadRmsNormBf16Geometry::new(input, gamma, output, width, epsilon)
                .map_err(|e| e.to_string())?;
            Shape::new(
                kind,
                g.rows(),
                g.columns(),
                g.head_dim(),
                g.effective_epsilon(),
            )
        }
    }
}

#[cfg(test)]
mod tests;
