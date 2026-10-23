//! Fixed-tensor providers sharing a declared scalar numerical implementation.

use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::{
    silu_bf16_contract, silu_mul_bf16_contract, tanh_bf16_contract, AttributeId,
    BatchedOperationInvocation, Bf16SameShapeGeometry, CapabilityId, ContractVersion,
    DeviceRuntime, ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope,
    NodeWorkContract, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProfilePhase, ProviderExecutionSemantics, ProviderId,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, TensorOperationInvocation, VNextError,
    SILU_BF16_CAPABILITY_ID, SILU_BF16_OPERATION_ID, SILU_MUL_BF16_CAPABILITY_ID,
    SILU_MUL_BF16_OPERATION_ID, TANH_BF16_CAPABILITY_ID, TANH_BF16_OPERATION_ID,
};

use super::super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::super::runtime::{
    ReferenceActivationShape, ReferenceDeviceBuffer, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeError, ReferenceSiluMulLaunch,
    ReferenceUnaryKind, ReferenceUnaryLaunch,
};
use super::numerical_implementation_fingerprint;

const VALUE_ALIGNMENT_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend::reference) enum ReferenceTranscendentalKind {
    Silu,
    SiluMul,
    Tanh,
}

impl ReferenceTranscendentalKind {
    fn operation_id(self) -> &'static str {
        match self {
            Self::Silu => SILU_BF16_OPERATION_ID,
            Self::SiluMul => SILU_MUL_BF16_OPERATION_ID,
            Self::Tanh => TANH_BF16_OPERATION_ID,
        }
    }

    fn input_count(self) -> u32 {
        if self == Self::SiluMul {
            2
        } else {
            1
        }
    }

    fn operations_per_element(self) -> u64 {
        match self {
            Self::Silu => 4,
            Self::SiluMul => 5,
            Self::Tanh => 1,
        }
    }
}

pub(in crate::backend::reference) struct ReferenceTranscendentalProvider {
    descriptor: OperationProviderDescriptor,
    kind: ReferenceTranscendentalKind,
}

impl ReferenceTranscendentalProvider {
    pub(in crate::backend::reference) fn new(
        runtime: &ReferenceDeviceRuntime,
        kind: ReferenceTranscendentalKind,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let (contract, capability, provider_id, estimator_id) = match kind {
            ReferenceTranscendentalKind::Silu => (
                silu_bf16_contract(),
                SILU_BF16_CAPABILITY_ID,
                "provider.reference.silu.bf16",
                "resource-estimator.reference.silu.bf16",
            ),
            ReferenceTranscendentalKind::SiluMul => (
                silu_mul_bf16_contract(),
                SILU_MUL_BF16_CAPABILITY_ID,
                "provider.reference.silu_mul.bf16",
                "resource-estimator.reference.silu_mul.bf16",
            ),
            ReferenceTranscendentalKind::Tanh => (
                tanh_bf16_contract(),
                TANH_BF16_CAPABILITY_ID,
                "provider.reference.tanh.bf16",
                "resource-estimator.reference.tanh.bf16",
            ),
        };
        let contract = contract.map_err(contract_error)?;
        let capability = CapabilityId::new(capability).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks selected BF16 activation capability",
            ));
        }
        let numerical = numerical_implementation_fingerprint();
        let implementation = implementation_fingerprint(&[
            numerical.as_bytes(),
            include_str!("provider.rs").as_bytes(),
            provider_id.as_bytes(),
        ]);
        let estimator =
            implementation_fingerprint(&[estimator_id.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider_id).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::new(),
            BTreeSet::new(),
            contiguous_bindings(kind.input_count()),
            estimator_id,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor, kind })
    }
}

impl OperationResourceEstimator for ReferenceTranscendentalProvider {
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
                "reference activation estimator received another operation",
            ));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
        // Scalars stay in registers/stack; no activation-sized temporary is required.
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            None,
            None,
        ))
    }
}

impl OperationProvider<ReferenceDeviceRuntime> for ReferenceTranscendentalProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(provider_failure(invocation.participants()[0].identity().clone(),
            "reference BF16 activations support fixed tensors only; token invocation is unsupported".into()))
    }

    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode(self.kind, invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|message| provider_failure(identity, message))
    }
}

fn encode(
    kind: ReferenceTranscendentalKind,
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("reference BF16 activation requires one tensor participant".into());
    };
    if invocation.operation().id.as_str() != kind.operation_id()
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("reference BF16 activation requires its fixed tensor contract".into());
    }
    let geometry = geometry(kind, participant.bindings(), participant.attributes())?;
    let shape = backend_shape(&geometry)?;
    let region = |role, ordinal| {
        contiguous_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            ElementType::Bf16,
        )
    };
    match kind {
        ReferenceTranscendentalKind::Silu | ReferenceTranscendentalKind::Tanh => {
            ReferenceDeviceCommand::unary_activation_tensor(ReferenceUnaryLaunch {
                kind: if kind == ReferenceTranscendentalKind::Silu {
                    ReferenceUnaryKind::Silu
                } else {
                    ReferenceUnaryKind::Tanh
                },
                input: region(ResolvedValueRole::Input, 0)?,
                output: region(ResolvedValueRole::Output, 0)?,
                shape,
            })
        }
        ReferenceTranscendentalKind::SiluMul => {
            ReferenceDeviceCommand::silu_mul_tensor(ReferenceSiluMulLaunch {
                gate: region(ResolvedValueRole::Input, 0)?,
                up: region(ResolvedValueRole::Input, 1)?,
                output: region(ResolvedValueRole::Output, 0)?,
                shape,
            })
        }
    }
    .map_err(|error| error.to_string())
}

fn geometry(
    kind: ReferenceTranscendentalKind,
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Bf16SameShapeGeometry, String> {
    if !attributes.is_empty() || bindings.len() != kind.input_count() as usize + 1 {
        return Err("reference activation requires exact port count and no attributes".into());
    }
    let input = binding(bindings, ResolvedValueRole::Input, 0)?.tensor();
    if kind == ReferenceTranscendentalKind::SiluMul
        && binding(bindings, ResolvedValueRole::Input, 1)?.tensor() != input
    {
        return Err("reference SiLU-mul up tensor must exactly match its gate tensor".into());
    }
    let geometry = Bf16SameShapeGeometry::new(
        input,
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|error| error.to_string())?;
    // Estimation must reject shapes the selected backend cannot encode,
    // including host-size conversion and the declared logical work count.
    backend_shape(&geometry)?
        .logical_work(kind.operations_per_element())
        .map_err(|error| error.to_string())?;
    Ok(geometry)
}

fn backend_shape(geometry: &Bf16SameShapeGeometry) -> Result<ReferenceActivationShape, String> {
    Ok(ReferenceActivationShape {
        rows: usize::try_from(geometry.rows())
            .map_err(|_| "reference activation rows exceed usize")?,
        width: usize::try_from(geometry.width())
            .map_err(|_| "reference activation width exceeds usize")?,
    })
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.activation.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("reference provider failure metadata is bounded and static")
}

fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
