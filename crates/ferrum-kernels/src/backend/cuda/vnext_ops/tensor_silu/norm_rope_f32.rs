//! Full-F32 gamma/activation/frequency providers with bounded CTA storage.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    half_split_rope_f32_contract, head_rms_norm_f32_contract,
    unit_head_rms_partial_rope_f32_contract, HalfSplitRopeF32Geometry, HeadRmsNormF32Geometry,
    PhysicalWeightLayout, UnitHeadRmsPartialRopeF32Geometry, WeightFormatId,
    HALF_SPLIT_ROPE_F32_CAPABILITY_ID, HALF_SPLIT_ROPE_F32_OPERATION_ID,
    HEAD_RMS_NORM_F32_CAPABILITY_ID, HEAD_RMS_NORM_F32_OPERATION_ID,
    UNIT_HEAD_RMS_PARTIAL_ROPE_F32_CAPABILITY_ID, UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID,
};
pub const HEAD_PROFILE: &str = "cuda.head_rms_norm.f32.cta256_strided_tree.rn_sqrt_div.no_ftz.v1";
pub const ROPE_PROFILE: &str = "cuda.half_split_rope.f32.separate_rn_mul_add.no_ftz.v1";
pub const UNIT_PARTIAL_PROFILE: &str =
    "cuda.unit_head_rms_partial_rope.f32.cta256_tree.separate_rn_mul_add.no_ftz.v1";
pub(super) const FUNCTIONS: [&str; 3] = [
    "ferrum_vnext_head_rms_norm_f32",
    "ferrum_vnext_half_split_rope_f32",
    "ferrum_vnext_unit_head_rms_partial_rope_f32",
];
#[derive(Clone, Copy)]
pub(super) enum Kind {
    HeadRms,
    HalfSplitRope,
    UnitHeadRmsPartialRope,
}
impl Kind {
    fn index(self) -> usize {
        match self {
            Self::HeadRms => 0,
            Self::HalfSplitRope => 1,
            Self::UnitHeadRmsPartialRope => 2,
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::HeadRms => HEAD_RMS_NORM_F32_OPERATION_ID,
            Self::HalfSplitRope => HALF_SPLIT_ROPE_F32_OPERATION_ID,
            Self::UnitHeadRmsPartialRope => UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID,
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::HeadRms => HEAD_RMS_NORM_F32_CAPABILITY_ID,
            Self::HalfSplitRope => HALF_SPLIT_ROPE_F32_CAPABILITY_ID,
            Self::UnitHeadRmsPartialRope => UNIT_HEAD_RMS_PARTIAL_ROPE_F32_CAPABILITY_ID,
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::HeadRms => head_rms_norm_f32_contract(),
            Self::HalfSplitRope => half_split_rope_f32_contract(),
            Self::UnitHeadRmsPartialRope => unit_head_rms_partial_rope_f32_contract(),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::HeadRms => "head_rms_norm",
            Self::HalfSplitRope => "half_split_rope",
            Self::UnitHeadRmsPartialRope => "unit_head_rms_partial_rope",
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::HeadRms => HEAD_PROFILE,
            Self::HalfSplitRope => ROPE_PROFILE,
            Self::UnitHeadRmsPartialRope => UNIT_PARTIAL_PROFILE,
        }
    }
    fn inputs(self) -> u32 {
        match self {
            Self::HeadRms => 2,
            Self::HalfSplitRope | Self::UnitHeadRmsPartialRope => 3,
        }
    }
    fn attributes(self) -> usize {
        match self {
            Self::HeadRms => 2,
            Self::HalfSplitRope => 1,
            Self::UnitHeadRmsPartialRope => 3,
        }
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: Kind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(kind.capability()).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "runtime lacks selected F32 norm/RoPE capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        kind.profile().as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("norm_rope_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_f32_norm_rope.cuh"
        )),
    ]);
    let estimator = format!("resource-estimator.cuda.{}.f32", kind.name());
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}.f32", kind.name())).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.inputs()),
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    kind: Kind,
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kind,
        kernels: Arc::clone(kernels),
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: Kind,
    kernels: Arc<RetainedFunctions>,
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
                "F32 norm/RoPE estimator received another operation",
            ));
        }
        geometry(self.kind, r.values(), r.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            16,
            None,
            None,
        ))
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
            "cuda.norm_rope_f32.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.norm_rope_f32.encode", e))
    }
}
#[derive(Clone, Copy)]
enum Geometry {
    HeadRms(HeadRmsNormF32Geometry),
    HalfSplitRope(HalfSplitRopeF32Geometry),
    UnitHeadRmsPartialRope(UnitHeadRmsPartialRopeF32Geometry),
}
impl Geometry {
    fn elements(self) -> u64 {
        match self {
            Self::HeadRms(g) => g.elements(),
            Self::HalfSplitRope(g) => g.elements(),
            Self::UnitHeadRmsPartialRope(g) => g.elements(),
        }
    }
    fn operations(self) -> u64 {
        match self {
            Self::HeadRms(g) => g.logical_arithmetic_operations(),
            Self::HalfSplitRope(g) => g.logical_arithmetic_operations(),
            Self::UnitHeadRmsPartialRope(g) => g.logical_arithmetic_operations(),
        }
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("F32 norm/RoPE requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed tensor contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.inputs() as usize + 1);
        for (role, ordinal) in (0..self.kind.inputs())
            .map(|n| (ResolvedValueRole::Input, n))
            .chain(std::iter::once((ResolvedValueRole::Output, 0)))
        {
            let b = binding(p.bindings(), role, ordinal)?;
            let [c] = b.storage().components() else {
                return Err("requires one exact F32 component per port".into());
            };
            if c.element_type() != ElementType::F32
                || c.length_bytes()
                    != b.tensor()
                        .minimum_storage_bytes()
                        .map_err(|e| e.to_string())?
            {
                return Err("physical F32 dtype or byte extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        let output = self.kind.inputs() as usize;
        for input in &regions[..output] {
            dense::disjoint(
                input.device_ptr(),
                input.length_bytes(),
                regions[output].device_ptr(),
                regions[output].length_bytes(),
            )?;
        }
        let kind = self.kind;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(FUNCTIONS[kind.index()], regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let parameter = regions[1].device_ptr();
            let output = regions[kind.inputs() as usize].device_ptr();
            let mut launch = stream.launch_builder(kernels.norm_rope_f32_function(kind.index()));
            match g {
                Geometry::HeadRms(shape) => {
                    let groups = shape.elements() / shape.head_dim();
                    let width = shape.head_dim() as u32;
                    let epsilon = shape.effective_epsilon();
                    launch
                        .arg(&x)
                        .arg(&parameter)
                        .arg(&output)
                        .arg(&groups)
                        .arg(&width)
                        .arg(&epsilon);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (groups.min(65535) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
                Geometry::UnitHeadRmsPartialRope(shape) => {
                    let sine = regions[2].device_ptr();
                    let groups = shape.elements() / shape.head_dim();
                    let columns = shape.columns();
                    let width = shape.head_dim() as u32;
                    let rot = shape.rotary_dim() as u32;
                    let epsilon = shape.effective_epsilon();
                    launch
                        .arg(&x)
                        .arg(&parameter)
                        .arg(&sine)
                        .arg(&output)
                        .arg(&groups)
                        .arg(&columns)
                        .arg(&width)
                        .arg(&rot)
                        .arg(&epsilon);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (groups.min(65535) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
                Geometry::HalfSplitRope(shape) => {
                    let sine = regions[2].device_ptr();
                    let elements = shape.elements();
                    let columns = shape.columns();
                    let d = shape.head_dim();
                    launch
                        .arg(&x)
                        .arg(&parameter)
                        .arg(&sine)
                        .arg(&output)
                        .arg(&elements)
                        .arg(&columns)
                        .arg(&d);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 norm/RoPE launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), g.operations(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: Kind,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Geometry, String> {
    if b.len() != kind.inputs() as usize + 1
        || attrs.len() != kind.attributes()
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 norm/RoPE requires exact dense operands/attributes".into());
    }
    let d = super::super::unsigned_attribute(attrs, "head_dim")?;
    let input = |n| binding(b, ResolvedValueRole::Input, n).map(|v| v.tensor());
    let output = binding(b, ResolvedValueRole::Output, 0)?.tensor();
    match kind {
        Kind::HeadRms => {
            let Some(SemanticValue::Rational(eps)) = attrs
                .iter()
                .find(|(k, _)| k.as_str() == "epsilon")
                .map(|(_, v)| v)
            else {
                return Err("requires rational epsilon".into());
            };
            HeadRmsNormF32Geometry::new(input(0)?, input(1)?, output, d, *eps)
                .map(Geometry::HeadRms)
        }
        Kind::UnitHeadRmsPartialRope => {
            let rot = super::super::unsigned_attribute(attrs, "rotary_dim")?;
            let Some(SemanticValue::Rational(eps)) = attrs
                .iter()
                .find(|(k, _)| k.as_str() == "epsilon")
                .map(|(_, v)| v)
            else {
                return Err("requires rational epsilon".into());
            };
            UnitHeadRmsPartialRopeF32Geometry::new(
                input(0)?,
                input(1)?,
                input(2)?,
                output,
                d,
                rot,
                *eps,
            )
            .map(Geometry::UnitHeadRmsPartialRope)
        }
        Kind::HalfSplitRope => {
            HalfSplitRopeF32Geometry::new(input(0)?, input(1)?, input(2)?, output, d)
                .map(Geometry::HalfSplitRope)
        }
    }
    .map_err(|e| e.to_string())
}
