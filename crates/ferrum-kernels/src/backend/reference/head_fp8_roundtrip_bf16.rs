use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceHeadFp8RoundtripBf16Launch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_HEAD_FP8_ROUNDTRIP_BF16_PROFILE_ID: &str =
    "reference.head_fp8_roundtrip_bf16.amax_div448_rn_zero_inverse_e4m3fn_rne_bf16.v1";
const ESTIMATOR: &str = "resource-estimator.reference.head_fp8_roundtrip_bf16";
struct Provider {
    descriptor: OperationProviderDescriptor,
}
pub fn reference_vnext_head_fp8_roundtrip_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider_for(runtime, false)
}
pub(super) fn provider_for(
    runtime: &ReferenceDeviceRuntime,
    indexed: bool,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = if indexed {
        indexed_head_fp8_roundtrip_bf16_contract()
    } else {
        head_fp8_roundtrip_bf16_contract()
    }
    .map_err(error)?;
    let cap = CapabilityId::new(if indexed {
        INDEXED_HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID
    } else {
        HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID
    })
    .map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "missing head FP8 roundtrip capability",
        ));
    }
    let implementation = implementation_fingerprint(&[
        REFERENCE_HEAD_FP8_ROUNDTRIP_BF16_PROFILE_ID.as_bytes(),
        include_bytes!("head_fp8_roundtrip_bf16.rs"),
        include_bytes!("runtime/head_fp8_roundtrip_bf16.rs"),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(if indexed {
            "provider.reference.indexed_head_fp8_roundtrip.bf16"
        } else {
            "provider.reference.head_fp8_roundtrip.bf16"
        })
        .map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new(
            super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
        )
        .map_err(error)?]),
        BTreeSet::new(),
        contiguous_bindings(if indexed { 2 } else { 1 }),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(error)?
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
            return Err(invalid("estimator received another operation"));
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
            "requires fixed tensor work".into(),
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
        return Err("requires one fixed tensor participant".into());
    };
    if ![
        HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID,
        INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID,
    ]
    .contains(&i.operation().id.as_str())
        || !matches!(p.work(), NodeWorkContract::Fixed)
        || !i.resource_work_shape().resource_work().is_tensor()
    {
        return Err("requires head FP8 roundtrip fixed tensor contract".into());
    }
    let geometry = geometry(p.bindings(), p.attributes())?;
    ReferenceDeviceCommand::head_fp8_roundtrip_bf16_tensor(ReferenceHeadFp8RoundtripBf16Launch {
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
        indices: if geometry.gather().is_some() {
            Some(contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, 1)?,
                ElementType::I32,
            )?)
        } else {
            None
        },
        geometry,
    })
    .map_err(|e| e.to_string())
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<HeadFp8RoundtripBf16Geometry, String> {
    if ![2, 3].contains(&b.len())
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires two dense ports".into());
    }
    let x = binding(b, ResolvedValueRole::Input, 0)?.tensor();
    let y = binding(b, ResolvedValueRole::Output, 0)?.tensor();
    if b.len() == 3 {
        HeadFp8RoundtripBf16Geometry::indexed(
            x,
            binding(b, ResolvedValueRole::Input, 1)?.tensor(),
            y,
            attrs,
        )
    } else {
        HeadFp8RoundtripBf16Geometry::new(x, y, attrs)
    }
    .map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.head_fp8_roundtrip_bf16.encode",
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
fn error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
