//! Two independent single-round operations on the existing managed runtime.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceRowRmsAddcmulGeometry, ReferenceRowRmsAddcmulLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_ROW_RMS_NORM_SINGLE_ROUND_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.row_rms_norm_single_round.bf16.warp32_precise.v1";
pub const REFERENCE_ADDCMUL_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.addcmul.bf16.separate_f32_single_round.v1";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    RowRms,
    Addcmul,
}
impl Kind {
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::RowRms => row_rms_norm_single_round_bf16_contract(),
            Self::Addcmul => addcmul_bf16_contract(),
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::RowRms => ROW_RMS_NORM_SINGLE_ROUND_BF16_CAPABILITY_ID,
            Self::Addcmul => ADDCMUL_BF16_CAPABILITY_ID,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::RowRms => REFERENCE_ROW_RMS_NORM_SINGLE_ROUND_BF16_NUMERICAL_PROFILE_ID,
            Self::Addcmul => REFERENCE_ADDCMUL_BF16_NUMERICAL_PROFILE_ID,
        }
    }
    fn inputs(self) -> u32 {
        match self {
            Self::RowRms => 2,
            Self::Addcmul => 3,
        }
    }
}
pub(super) struct ReferenceRowRmsAddcmulProvider {
    descriptor: OperationProviderDescriptor,
    kind: Kind,
}
impl ReferenceRowRmsAddcmulProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
        kind: Kind,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = kind.contract().map_err(error)?;
        let capability = CapabilityId::new(kind.capability()).map_err(error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(error("runtime lacks requested RowRMS/addcmul capability"));
        }
        let name = if kind == Kind::RowRms {
            "row_rms_norm_single_round"
        } else {
            "addcmul"
        };
        let provider = format!("provider.reference.{name}.bf16");
        let estimator = format!("resource-estimator.reference.{name}.bf16");
        let fingerprint = implementation_fingerprint(&[
            include_bytes!("row_rms_addcmul.rs"),
            include_bytes!("runtime/row_rms_addcmul.rs"),
            kind.profile().as_bytes(),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider).map_err(error)?,
            contract.descriptor().id.clone(),
            contract.descriptor().fingerprint().map_err(error)?,
            fingerprint.clone(),
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([WeightFormatId::new(
                super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
            )
            .map_err(error)?]),
            BTreeSet::new(),
            contiguous_bindings(kind.inputs()),
            &estimator,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
        )
        .map_err(error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor, kind })
    }
}
impl OperationResourceEstimator for ReferenceRowRmsAddcmulProvider {
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
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceRowRmsAddcmulProvider {
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
impl ReferenceRowRmsAddcmulProvider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<ReferenceDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one tensor participant".into());
        };
        if i.operation().id != *self.descriptor.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires requested fixed tensor operation".into());
        }
        let geometry = geometry(self.kind, p.bindings(), p.attributes())?;
        let input = |n| {
            contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, n)?,
                ElementType::Bf16,
            )
        };
        ReferenceDeviceCommand::row_rms_addcmul_tensor(ReferenceRowRmsAddcmulLaunch {
            input: input(0)?,
            parameter: input(1)?,
            update: if self.kind == Kind::Addcmul {
                Some(input(2)?)
            } else {
                None
            },
            output: contiguous_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::Bf16,
            )?,
            geometry,
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: Kind,
    bindings: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<ReferenceRowRmsAddcmulGeometry, String> {
    if bindings.len() != kind.inputs() as usize + 1 || attrs.len() != 1 {
        return Err("requires exact operands and one semantic attribute".into());
    }
    for b in bindings {
        if b.tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?
            > isize::MAX as u64
            || b.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("requires addressable dense operands".into());
        }
    }
    let input = |n| binding(bindings, ResolvedValueRole::Input, n).map(|b| b.tensor());
    let output = binding(bindings, ResolvedValueRole::Output, 0)?.tensor();
    let get = |name| {
        attrs
            .iter()
            .find(|(id, _)| id.as_str() == name)
            .map(|(_, v)| v)
    };
    match kind {
        Kind::RowRms => {
            let Some(SemanticValue::Rational(eps)) = get("epsilon") else {
                return Err("requires rational epsilon".into());
            };
            Ok(ReferenceRowRmsAddcmulGeometry::RowRms(
                RowRmsNormSingleRoundBf16Geometry::new(input(0)?, input(1)?, output, *eps)
                    .map_err(|e| e.to_string())?,
            ))
        }
        Kind::Addcmul => {
            let Some(SemanticValue::Text(mode)) = get("broadcast") else {
                return Err("requires explicit broadcast".into());
            };
            Ok(ReferenceRowRmsAddcmulGeometry::Addcmul(
                AddcmulBf16Geometry::new(
                    input(0)?,
                    input(1)?,
                    input(2)?,
                    output,
                    RowBroadcast::parse(mode).ok_or("invalid broadcast")?,
                )
                .map_err(|e| e.to_string())?,
            ))
        }
    }
}
fn error(e: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn failure(i: ExecutionIdentityEnvelope, s: String) -> OperationFailure {
    OperationFailure::new(
        i,
        ProfilePhase::Forward,
        "reference.row_rms_addcmul.encode",
        s.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
