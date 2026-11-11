//! F32 fixed-tensor norm/RoPE with actual retained storage commands.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceF32NormRopeGeometry, ReferenceF32NormRopeLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_HEAD_RMS_NORM_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.head_rms_norm.f32.lane256_tree.rn_sqrt_div.v1";
pub const REFERENCE_HALF_SPLIT_ROPE_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.half_split_rope.f32.separate_rn_mul_add.v1";
pub const REFERENCE_UNIT_HEAD_RMS_PARTIAL_ROPE_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.unit_head_rms_partial_rope.f32.lane256_tree.separate_rn_mul_add.v1";
#[derive(Clone, Copy)]
enum Kind {
    HeadRms,
    HalfSplitRope,
    UnitHeadRmsPartialRope,
}
impl Kind {
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
            Self::HeadRms => REFERENCE_HEAD_RMS_NORM_F32_NUMERICAL_PROFILE_ID,
            Self::HalfSplitRope => REFERENCE_HALF_SPLIT_ROPE_F32_NUMERICAL_PROFILE_ID,
            Self::UnitHeadRmsPartialRope => {
                REFERENCE_UNIT_HEAD_RMS_PARTIAL_ROPE_F32_NUMERICAL_PROFILE_ID
            }
        }
    }
    fn attributes(self) -> usize {
        match self {
            Self::HeadRms => 2,
            Self::HalfSplitRope => 1,
            Self::UnitHeadRmsPartialRope => 3,
        }
    }
    fn inputs(self) -> u32 {
        match self {
            Self::HeadRms => 2,
            Self::HalfSplitRope | Self::UnitHeadRmsPartialRope => 3,
        }
    }
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: Kind,
}
fn provider(
    runtime: &ReferenceDeviceRuntime,
    kind: Kind,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = kind.contract().map_err(contract_error)?;
    let cap = CapabilityId::new(kind.capability()).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference runtime lacks selected F32 norm/RoPE capability",
        ));
    }
    let fingerprint = implementation_fingerprint(&[
        kind.profile().as_bytes(),
        include_bytes!("norm_rope_f32.rs"),
        include_bytes!("runtime/norm_rope_f32.rs"),
    ]);
    let estimator = format!("resource-estimator.reference.{}.f32", kind.name());
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.{}.f32", kind.name()))
            .map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        contiguous_bindings(kind.inputs()),
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, kind }))
}
pub fn reference_vnext_head_rms_norm_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, Kind::HeadRms)
}
pub fn reference_vnext_half_split_rope_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, Kind::HalfSplitRope)
}
pub fn reference_vnext_unit_head_rms_partial_rope_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, Kind::UnitHeadRmsPartialRope)
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
            return Err(invalid(
                "F32 norm/RoPE estimator received another operation",
            ));
        }
        geometry(self.kind, r.values(), r.attributes()).map_err(invalid)?;
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
impl OperationProvider<ReferenceDeviceRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        i: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(failure(
            i.participants()[0].identity().clone(),
            "requires fixed tensor invocation".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| failure(identity, e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<ReferenceDeviceCommand, String> {
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
        let region = |role, ordinal| {
            contiguous_region(p, binding(p.bindings(), role, ordinal)?, ElementType::F32)
        };
        ReferenceDeviceCommand::norm_rope_f32_tensor(ReferenceF32NormRopeLaunch {
            input: region(ResolvedValueRole::Input, 0)?,
            parameter: region(ResolvedValueRole::Input, 1)?,
            sine: if self.kind.inputs() == 3 {
                Some(region(ResolvedValueRole::Input, 2)?)
            } else {
                None
            },
            output: region(ResolvedValueRole::Output, 0)?,
            geometry: g,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: Kind,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<ReferenceF32NormRopeGeometry, String> {
    if b.len() != kind.inputs() as usize + 1
        || attrs.len() != kind.attributes()
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 norm/RoPE requires exact dense operands/attributes".into());
    }
    let get = |name| {
        attrs
            .iter()
            .find(|(k, _)| k.as_str() == name)
            .map(|(_, v)| v)
    };
    let Some(SemanticValue::Unsigned(d)) = get("head_dim") else {
        return Err("requires unsigned head_dim".into());
    };
    let input = |n| binding(b, ResolvedValueRole::Input, n).map(|v| v.tensor());
    let output = binding(b, ResolvedValueRole::Output, 0)?.tensor();
    match kind {
        Kind::HeadRms => {
            let Some(SemanticValue::Rational(eps)) = get("epsilon") else {
                return Err("requires rational epsilon".into());
            };
            HeadRmsNormF32Geometry::new(input(0)?, input(1)?, output, *d, *eps)
                .map(ReferenceF32NormRopeGeometry::HeadRms)
        }
        Kind::UnitHeadRmsPartialRope => {
            let Some(SemanticValue::Rational(eps)) = get("epsilon") else {
                return Err("requires rational epsilon".into());
            };
            let Some(SemanticValue::Unsigned(rot)) = get("rotary_dim") else {
                return Err("requires unsigned rotary_dim".into());
            };
            UnitHeadRmsPartialRopeF32Geometry::new(
                input(0)?,
                input(1)?,
                input(2)?,
                output,
                *d,
                *rot,
                *eps,
            )
            .map(ReferenceF32NormRopeGeometry::UnitHeadRmsPartialRope)
        }
        Kind::HalfSplitRope => {
            HalfSplitRopeF32Geometry::new(input(0)?, input(1)?, input(2)?, output, *d)
                .map(ReferenceF32NormRopeGeometry::HalfSplitRope)
        }
    }
    .map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.norm_rope_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
fn contract_error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
