//! Explicit immutable metadata producers, bound by the existing Core prepare path.
//! No plan cache, payload lookup map, device-list bypass, or alternate submitter.
use super::super::vnext_runtime::CudaTensorNativeModuleRetention;
use super::*;
use crate::native_ops::sage_vsa_fp4::{
    self, Bindings, CheckedHostVsaPattern, Device, DeviceSpan, PreparedSageVsaFp4, Request,
    SageVsaFp4Config, WorkspaceLayout,
};
use ferrum_runtime::{
    sparse_fp4_attention_checked_host_contract, NodeWorkContract, SparseFp4AttentionGeometry,
    TensorOperationInvocation, SPARSE_FP4_ATTENTION_CHECKED_HOST_CAPABILITY_ID,
    SPARSE_FP4_ATTENTION_CHECKED_HOST_OPERATION_ID,
};
use std::collections::BTreeMap;
mod profile;
use profile::{Execution, Pattern};
const ESTIMATOR: &str = "resource-estimator.cuda.sparse_fp4_attention.checked_host.sm120";

pub(super) mod module;
pub(super) use module::source_fingerprint;

pub(super) struct Provider {
    descriptor: OperationProviderDescriptor,
    device: Device,
    owner: Arc<CudaTensorNativeModuleRetention>,
    pattern: Pattern,
}
struct Prepared {
    execution: Execution,
    geometry: SparseFp4AttentionGeometry,
    pattern: Pattern,
    owner: Arc<CudaTensorNativeModuleRetention>,
    fingerprint: String,
    arithmetic: u64,
}
impl Provider {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        config: &SageVsaFp4Config,
        pattern: Arc<CheckedHostVsaPattern>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::for_pattern(runtime, config, Pattern::Host(pattern))
    }
    pub(super) fn regions(
        runtime: &CudaDeviceRuntime,
        config: &SageVsaFp4Config,
        pattern: Arc<sage_vsa_fp4::CheckedVsaRegions>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if !sage_vsa_fp4::regions::kernel_upload_compiled() {
            return Err(CudaDeviceRuntimeError::contract(
                "VSA region producer requires its locked kernel-upload exports",
            ));
        }
        Self::for_pattern(runtime, config, Pattern::Regions(pattern))
    }
    fn for_pattern(
        runtime: &CudaDeviceRuntime,
        config: &SageVsaFp4Config,
        pattern: Pattern,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        config
            .validate()
            .map_err(CudaDeviceRuntimeError::contract)?;
        let capabilities = BTreeSet::from([
            CapabilityId::new(pattern.capability()).map_err(contract_error)?,
            CapabilityId::new(pattern.native_capability()).map_err(contract_error)?,
        ]);
        if !capabilities.is_subset(&runtime.descriptor().capabilities) {
            return Err(CudaDeviceRuntimeError::contract(
                "VSA requires its explicit checked Host profile capabilities",
            ));
        }
        let module::Module {
            device,
            owner,
            fingerprint: implementation,
        } = module::Module::acquire(runtime, config)?;
        let s = pattern.shape();
        Request::new(
            s.batches(),
            s.heads(),
            s.query_blocks(),
            s.key_blocks(),
            1.0,
            device,
        )
        .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        // All original allocations (including their physical padding)
        // already carry Host account charges. Registration allocates no payload.
        if matches!(&pattern, Pattern::Host(_)) {
            runtime.register_tensor_metadata_host(pattern.payloads())?;
        }
        let contract = pattern.contract().map_err(contract_error)?;
        let bindings = super::transformer::contiguous_bindings(pattern.inputs() as u32);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(pattern.provider()).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation.clone(),
            ferrum_runtime::ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            capabilities,
            BTreeSet::new(),
            BTreeSet::new(),
            bindings,
            ESTIMATOR,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]),
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self {
            descriptor,
            device,
            owner,
            pattern,
        })
    }
    fn prepare(&self, geometry: SparseFp4AttentionGeometry) -> Result<Arc<Prepared>, String> {
        let s = self.pattern.shape();
        if geometry.pattern_sha256() != self.pattern.digest()
            || [
                geometry.batches(),
                geometry.heads(),
                geometry.query_blocks(),
                geometry.key_blocks(),
            ] != [s.batches(), s.heads(), s.query_blocks(), s.key_blocks()]
        {
            return Err("VSA operation digest/shape differs from explicitly supplied immutable Host payload".into());
        }
        let request = Request::new(
            s.batches(),
            s.heads(),
            s.query_blocks(),
            s.key_blocks(),
            geometry.softmax_scale(),
            self.device,
        )
        .map_err(|e| e.to_string())?;
        self.owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let execution = Execution::prepare(&self.pattern, request)?;
        let arithmetic = self
            .pattern
            .selected_tiles()
            .checked_mul(128 * 128 * 128 * 4)
            .ok_or("VSA tile arithmetic overflow")?;
        let fingerprint = implementation_fingerprint(&[
            self.descriptor
                .provider_implementation_fingerprint()
                .as_bytes(),
            self.pattern.numerical_identity().as_bytes(),
            self.pattern.digest().as_bytes(),
            execution.fingerprint().as_bytes(),
            &execution.bytes().to_le_bytes(),
        ]);
        Ok(Arc::new(Prepared {
            execution,
            geometry,
            pattern: self.pattern.clone(),
            owner: Arc::clone(&self.owner),
            fingerprint,
            arithmetic,
        }))
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    pattern: &Pattern,
) -> Result<SparseFp4AttentionGeometry, String> {
    if values.len() != pattern.inputs() + 1 {
        return Err("VSA input/output port count differs from producer profile".into());
    }
    let [q, k, v, sq, sk, sv] = [0, 1, 2, 3, 4, 5]
        .map(|i| binding(values, ResolvedValueRole::Input, i).map(|v| v.tensor()));
    let y = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    let operands = [q?, k?, v?, sq?, sk?, sv?];
    match pattern {
        Pattern::Host(_) => SparseFp4AttentionGeometry::new(operands, [y], attributes),
        Pattern::Regions(_) => {
            let scores = binding(values, ResolvedValueRole::Input, 6)?.tensor();
            let [q, k, v, sq, sk, sv] = operands;
            SparseFp4AttentionGeometry::with_region_scores(
                [q, k, v, sq, sk, sv, scores],
                [y],
                attributes,
            )
        }
    }
    .map_err(|e| e.to_string())
}
impl OperationResourceEstimator for Provider {
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
            return Err(invalid("VSA estimator received another operation"));
        }
        let prepared = self
            .prepare(
                geometry(request.values(), request.attributes(), &self.pattern).map_err(invalid)?,
            )
            .map_err(invalid)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(prepared.execution.bytes())?,
            16,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            Some(scratch),
            None,
        )
        .with_prepared_implementation_fingerprint(prepared.fingerprint.clone())?
        .with_prepared_implementation_retention(prepared)
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
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "cuda.sparse-fp4.encode",
            "VSA requires fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.sparse-fp4.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("VSA requires one fixed participant".into());
        };
        if invocation.operation().id.as_str() != self.pattern.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("VSA requires its fixed tensor contract".into());
        }
        let geometry = geometry(
            participant.bindings(),
            participant.attributes(),
            &self.pattern,
        )?;
        let prepared = invocation
            .prepared_implementation::<Prepared>()
            .ok_or("VSA immutable prepared owner is absent")?;
        if prepared.geometry != geometry
            || !prepared.pattern.same_owner(&self.pattern)
            || !Arc::ptr_eq(&prepared.owner, &self.owner)
            || invocation.prepared_implementation_fingerprint()
                != Some(prepared.fingerprint.as_str())
        {
            return Err("VSA prepared geometry/source/owner differs from this registry".into());
        }
        let mut regions = Vec::with_capacity(self.pattern.inputs() + 2);
        for (role, index) in (0..self.pattern.inputs())
            .map(|i| (ResolvedValueRole::Input, i))
            .chain((0..1).map(|i| (ResolvedValueRole::Output, i)))
        {
            let value = binding(participant.bindings(), role, index as u32)?;
            regions.push(super::tensor_storage::single_component(
                participant,
                value,
                16,
            )?);
        }
        regions.push(super::tensor_storage::scratch(
            participant,
            prepared.execution.bytes(),
            16,
        )?);
        if regions
            .iter()
            .any(|r| !Arc::ptr_eq(r.context(), self.owner.context()))
        {
            return Err("VSA binding belongs to another context".into());
        }
        self.owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        // SAFETY: the private prepared owner retains the original immutable
        // registered Host allocations; all device ranges came from Core views.
        unsafe { prepared.execution.validate(&regions, &prepared.pattern) }?;
        let elements = u64::from(geometry.batches())
            * u64::from(geometry.heads())
            * u64::from(geometry.query_rows())
            * 128;
        let arithmetic = prepared.arithmetic;
        let (computes, transfers) = prepared.execution.dispatch_counts();
        CudaDeviceCommand::operation(
            sage_vsa_fp4::SAGE_VSA_FP4_OPERATOR,
            regions,
            move |stream, regions| {
                prepared
                    .owner
                    .context()
                    .bind_to_thread()
                    .map_err(|e| CudaDeviceRuntimeError::driver("VSA context bind", e))?;
                // This command owns the exact prepared payload, registration,
                // module and device ranges through ordinary completion/quarantine.
                unsafe {
                    prepared
                        .execution
                        .run(regions, &prepared.pattern, stream.cu_stream().cast())
                }
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
            },
        )
        .and_then(|command| command.with_tensor_work(elements, arithmetic, computes, transfers))
        .map_err(|e| e.to_string())
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
