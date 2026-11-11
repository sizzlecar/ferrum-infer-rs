//! Checked original-rank patch SGEMM and allocation-free F32 matrix regrouping.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceAffineF32Launch, ReferenceDenseLinearLaunch, ReferenceDenseScalar,
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceReshapeCopyF32Launch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_PATCH_PROJECTION_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.patch_projection.f32.original_rank5.dot_then_bias.v1";
pub const REFERENCE_RESHAPE_COPY_F32_NUMERICAL_PROFILE_ID: &str =
    "reference.reshape_copy.f32.exact_bits.v1";
#[derive(Clone, Copy)]
enum Kind {
    Patch,
    Reshape,
}
impl Kind {
    fn op(self) -> &'static str {
        match self {
            Self::Patch => PATCH_PROJECTION_F32_OPERATION_ID,
            Self::Reshape => RESHAPE_COPY_F32_OPERATION_ID,
        }
    }
    fn cap(self) -> &'static str {
        match self {
            Self::Patch => PATCH_PROJECTION_F32_CAPABILITY_ID,
            Self::Reshape => RESHAPE_COPY_F32_CAPABILITY_ID,
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Patch => patch_projection_f32_contract(),
            Self::Reshape => reshape_copy_f32_contract(),
        }
    }
    fn inputs(self) -> u32 {
        match self {
            Self::Patch => 3,
            Self::Reshape => 1,
        }
    }
}
pub fn reference_vnext_patch_projection_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, Kind::Patch)
}
pub fn reference_vnext_reshape_copy_f32_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    provider(runtime, Kind::Reshape)
}
fn provider(
    runtime: &ReferenceDeviceRuntime,
    kind: Kind,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let c = kind.contract().map_err(contract_error)?;
    let cap = CapabilityId::new(kind.cap()).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "Reference lacks F32 vision layout capability",
        ));
    }
    let profile = match kind {
        Kind::Patch => REFERENCE_PATCH_PROJECTION_F32_NUMERICAL_PROFILE_ID,
        Kind::Reshape => REFERENCE_RESHAPE_COPY_F32_NUMERICAL_PROFILE_ID,
    };
    let implementation = implementation_fingerprint(&[
        profile.as_bytes(),
        include_bytes!("vision_layout_f32.rs"),
        include_bytes!("runtime/vision_layout_f32.rs"),
        include_bytes!("runtime/affine_f32.rs"),
        include_bytes!("runtime.rs"),
    ]);
    let estimator = implementation_fingerprint(&[
        b"resource-estimator.reference.vision_layout_f32",
        implementation.as_bytes(),
    ]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.reference.{}", kind.op())).map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
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
        contiguous_bindings(kind.inputs()),
        "resource-estimator.reference.vision_layout_f32",
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor, kind }))
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kind: Kind,
}
enum Geometry {
    Patch(AffineLinearF32Geometry),
    Reshape(ReshapeCopyF32Geometry),
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
            return Err(invalid("F32 layout estimator received another operation"));
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
            "F32 vision layout requires fixed tensors".into(),
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
            return Err("F32 layout requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.op()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("F32 layout requires its selected fixed tensor contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let region = |role, ordinal| {
            contiguous_region(p, binding(p.bindings(), role, ordinal)?, ElementType::F32)
        };
        match g {
            Geometry::Patch(g) => {
                ReferenceDeviceCommand::patch_projection_f32_tensor(ReferenceAffineF32Launch {
                    dense: ReferenceDenseLinearLaunch {
                        input: region(ResolvedValueRole::Input, 0)?,
                        weight: region(ResolvedValueRole::Input, 1)?,
                        output: region(ResolvedValueRole::Output, 0)?,
                        rows: g.rows() as usize,
                        in_features: g.inputs() as usize,
                        out_features: g.outputs() as usize,
                        scalar: ReferenceDenseScalar::F32,
                    },
                    bias: region(ResolvedValueRole::Input, 2)?,
                })
            }
            Geometry::Reshape(g) => {
                ReferenceDeviceCommand::reshape_copy_f32_tensor(ReferenceReshapeCopyF32Launch {
                    input: region(ResolvedValueRole::Input, 0)?,
                    output: region(ResolvedValueRole::Output, 0)?,
                    geometry: g,
                })
            }
        }
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: Kind,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Geometry, String> {
    if values.len() != kind.inputs() as usize + 1
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 layout requires exact dense physical ports".into());
    }
    kind.contract()
        .map_err(|e| e.to_string())?
        .descriptor()
        .validate_attributes(attrs)
        .map_err(|e| e.to_string())?;
    let input = |n| binding(values, ResolvedValueRole::Input, n).map(|v| v.tensor());
    let out = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    match kind {
        Kind::Patch => PatchProjectionF32Geometry::new(input(0)?, input(1)?, input(2)?, out)
            .map(|g| Geometry::Patch(g.affine()))
            .map_err(|e| e.to_string()),
        Kind::Reshape => {
            let u = |name: &str| {
                attrs
                    .iter()
                    .find(|(k, _)| k.as_str() == name)
                    .and_then(|(_, v)| {
                        if let SemanticValue::Unsigned(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| format!("missing reshape {name}"))
            };
            ReshapeCopyF32Geometry::new(input(0)?, out, u("rows")?, u("columns")?)
                .map(Geometry::Reshape)
                .map_err(|e| e.to_string())
        }
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn contract_error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.vision_layout_f32.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
