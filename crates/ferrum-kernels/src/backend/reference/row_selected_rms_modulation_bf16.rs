//! Fused RMS/modulation on existing admitted Reference tensor owners.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceRowSelectedRmsModulationLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_ROW_SELECTED_RMS_MODULATION_PROFILE: &str =
    "reference.row_selected_rms_modulation.bf16.block8192_vec8_warp16_fma.libm_sqrtf_rne_div.v1";
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: RowSelectedRmsModulationKind,
}
pub fn reference_vnext_row_selected_rms_modulation_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
    kind: RowSelectedRmsModulationKind,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = contract(kind).map_err(error)?;
    let cap = CapabilityId::new(kind.capability_id()).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(error("runtime lacks fused RMS modulation capability"));
    }
    let name = kind.operation_id().trim_start_matches("operation.");
    let estimator = format!("resource-estimator.reference.{name}");
    let fingerprint = implementation_fingerprint(&[
        REFERENCE_ROW_SELECTED_RMS_MODULATION_PROFILE.as_bytes(),
        include_bytes!("row_selected_rms_modulation_bf16.rs"),
        include_bytes!("runtime/row_selected_rms_modulation_bf16.rs"),
    ]);
    let mut bindings = contiguous_bindings(kind.input_count() as u32);
    if kind == RowSelectedRmsModulationKind::Residual {
        bindings.push(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            1,
            DynamicStorageRequirement::contiguous(),
        ));
    }
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.{name}")).map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new(
            super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
        )
        .map_err(error)?]),
        BTreeSet::new(),
        bindings,
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, kind }))
}
fn contract(kind: RowSelectedRmsModulationKind) -> Result<StandardOperationContract, VNextError> {
    match kind {
        RowSelectedRmsModulationKind::Normalize => row_selected_rms_modulation_bf16_contract(),
        RowSelectedRmsModulationKind::Residual => {
            row_selected_residual_rms_modulation_bf16_contract()
        }
    }
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
        encode(self.kind, i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| failure(identity, e))
    }
}
fn encode(
    kind: RowSelectedRmsModulationKind,
    i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [p] = i.participants() else {
        return Err("requires one fixed tensor participant".into());
    };
    if i.operation().id.as_str() != kind.operation_id()
        || !matches!(p.work(), NodeWorkContract::Fixed)
        || !i.resource_work_shape().resource_work().is_tensor()
    {
        return Err("requires selected fixed tensor contract".into());
    }
    let g = geometry(kind, p.bindings(), p.attributes())?;
    let region = |role, ordinal| {
        contiguous_region(p, binding(p.bindings(), role, ordinal)?, ElementType::Bf16)
    };
    ReferenceDeviceCommand::row_selected_rms_modulation_tensor(
        ReferenceRowSelectedRmsModulationLaunch {
            input: region(ResolvedValueRole::Input, 0)?,
            parameters: region(ResolvedValueRole::Input, 1)?,
            gamma: region(ResolvedValueRole::Input, 2)?,
            branch: if kind == RowSelectedRmsModulationKind::Residual {
                Some(region(ResolvedValueRole::Input, 3)?)
            } else {
                None
            },
            hidden: if kind == RowSelectedRmsModulationKind::Residual {
                Some(region(ResolvedValueRole::Output, 0)?)
            } else {
                None
            },
            output: region(ResolvedValueRole::Output, kind.output_count() as u32 - 1)?,
            geometry: g,
        },
    )
    .map_err(|e| e.to_string())
}
fn geometry(
    kind: RowSelectedRmsModulationKind,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowSelectedRmsModulationGeometry, String> {
    if b.len() != kind.input_count() + kind.output_count()
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires dense numerical operands".into());
    }
    let inputs = (0..kind.input_count())
        .map(|i| binding(b, ResolvedValueRole::Input, i as u32).map(|b| b.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = (0..kind.output_count())
        .map(|i| binding(b, ResolvedValueRole::Output, i as u32).map(|b| b.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    RowSelectedRmsModulationGeometry::new(kind, &inputs, &outputs, attrs).map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.fused_rms_modulation.encode",
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
fn error(e: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
