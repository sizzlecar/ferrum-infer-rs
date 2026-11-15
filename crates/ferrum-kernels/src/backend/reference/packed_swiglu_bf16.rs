//! Packed SwiGLU provider over ordinary admitted tensor owners.
//! Libm exp/exp2 and RNE division are CPU reference approximations, not CUDA bit oracles.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferencePackedSwiGluBf16Launch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_PACKED_SWIGLU_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.packed_swiglu.bf16.libm-0.2.15.exp2-single-round-or-staged-silu.v2";
const ESTIMATOR: &str = "resource-estimator.reference.packed_swiglu.bf16";
struct Provider {
    descriptor: OperationProviderDescriptor,
}
pub fn reference_vnext_packed_swiglu_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = packed_swiglu_bf16_contract().map_err(contract_error)?;
    let cap = CapabilityId::new(PACKED_SWIGLU_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference runtime lacks packed BF16 SwiGLU capability",
        ));
    }
    let implementation = implementation_fingerprint(&[
        REFERENCE_PACKED_SWIGLU_BF16_NUMERICAL_PROFILE_ID.as_bytes(),
        include_bytes!("packed_swiglu_bf16.rs"),
        include_bytes!("runtime/packed_swiglu_bf16.rs"),
        include_bytes!("runtime/transcendental.rs"),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new("provider.reference.packed_swiglu.bf16").map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new(
            super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
        )
        .map_err(contract_error)?]),
        BTreeSet::new(),
        contiguous_bindings(1),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor }))
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
                "packed SwiGLU estimator received another operation",
            ));
        }
        geometry(r.values(), r.attributes()).map_err(invalid)?;
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
            "packed SwiGLU requires fixed tensors".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| failure(identity, e))
    }
}
fn encode(
    i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [p] = i.participants() else {
        return Err("packed SwiGLU requires one fixed tensor participant".into());
    };
    if i.operation().id.as_str() != PACKED_SWIGLU_BF16_OPERATION_ID
        || !matches!(p.work(), NodeWorkContract::Fixed)
        || !i.resource_work_shape().resource_work().is_tensor()
    {
        return Err("packed SwiGLU requires its fixed tensor contract".into());
    }
    let (g, profile) = geometry(p.bindings(), p.attributes())?;
    ReferenceDeviceCommand::packed_swiglu_bf16_tensor(ReferencePackedSwiGluBf16Launch {
        input: contiguous_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::Bf16,
        )?,
        output: contiguous_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::Bf16,
        )?,
        geometry: g,
        profile,
    })
    .map_err(|e| e.to_string())
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<(PackedSwiGluBf16Geometry, PackedSwiGluBf16Profile), String> {
    let profile = PackedSwiGluBf16Profile::from_attributes(attrs).map_err(|e| e.to_string())?;
    if b.len() != 2
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("packed SwiGLU requires two dense ports".into());
    }
    PackedSwiGluBf16Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map(|g| (g, profile))
    .map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.packed_swiglu_bf16.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn contract_error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
