//! Explicit SM120 W8A8 dense provider. Device temporaries belong to invocation
//! scratch; the static CUDART fatbin has an independent opaque claim, retained
//! conservatively until process exit because this bridge has no unload receipt.

use super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError, CudaTensorCutlassRetention,
};
use super::{binding, contract_error, implementation_fingerprint, provider_failure};
use crate::native_ops::cutlass_fp8::{
    self, Bindings, Device, DeviceSpan, PreparedCutlassFp8, Request,
};
use ferrum_runtime::{
    dense_linear_fp8_bf16_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DenseLinearFp8Bf16Geometry, DeviceRuntime, DynamicStorageRequirement,
    ElementType, EncodedDeviceOperation, NodeWorkContract, OperationContract, OperationFailure,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProviderExecutionSemantics,
    ProviderId, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, QuantizationFormatId, ResolvedValueBinding, ResolvedValueRole,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue,
    TensorOperationInvocation, VNextError, WeightFormatId, DENSE_LINEAR_FP8_BF16_CAPABILITY_ID,
    DENSE_LINEAR_FP8_BF16_OPERATION_ID, FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

mod probe;
#[cfg(test)]
mod tests;
mod weights;
pub use probe::{
    CudaCutlassFp8Probe, CudaCutlassFp8ProbeReceipt, CudaCutlassFp8ProbeTicket,
    QUANTIZATION_PROBE_PROFILE_ID,
};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cutlass.sm120.w8a8.bf16.rowwise_e4m3fn.v1";
const PROVIDER: &str = cutlass_fp8::CUTLASS_FP8_PROVIDER_ID;
const ESTIMATOR: &str = "resource-estimator.cuda.dense_linear_fp8.bf16.cutlass.sm120";

/// Explicit, unqualified native-memory declaration. It is charged in addition
/// to context bootstrap and conservatively retained until process exit (native
/// module unload/primary-context destruction is not yet acknowledged). Plan records
/// are bounded Host metadata; no input/weight-sized Host scratch is allocated.
#[derive(Clone, Debug)]
pub struct CudaCutlassFp8Config {
    pub opaque_memory_bytes: u64,
    pub maximum_plans: usize,
    pub opaque_memory_profile: String,
}

impl CudaCutlassFp8Config {
    pub fn validate(&self) -> Result<(), CudaDeviceRuntimeError> {
        if self.opaque_memory_bytes == 0
            || self.maximum_plans == 0
            || self
                .maximum_plans
                .checked_mul(std::mem::size_of::<PreparedCutlassFp8>())
                .is_none_or(|bytes| bytes > isize::MAX as usize)
            || self.opaque_memory_profile.trim().is_empty()
            || self.opaque_memory_profile.len() > 256
            || !self.opaque_memory_profile.is_ascii()
            || self
                .opaque_memory_profile
                .bytes()
                .any(|b| b.is_ascii_control())
        {
            return Err(CudaDeviceRuntimeError::contract("CUTLASS FP8 requires positive finite opaque/plan limits and a versioned ASCII profile"));
        }
        Ok(())
    }
}

pub(super) fn source_fingerprint() -> Result<String, String> {
    if !cutlass_fp8::compiled() {
        return Err(
            "CUTLASS FP8 requires the explicitly built cuda-cutlass-fp8 native artifact".into(),
        );
    }
    let mut artifacts = crate::native_ops::compiled_native_operator_artifacts()
        .iter()
        .filter(|artifact| artifact.operator == cutlass_fp8::CUTLASS_FP8_OPERATOR);
    let artifact = artifacts
        .next()
        .ok_or("CUTLASS FP8 artifact identity absent")?;
    if artifacts.next().is_some() {
        return Err("CUTLASS FP8 artifact identity is ambiguous".into());
    }
    let identity = serde_json::to_vec(artifact).map_err(|e| e.to_string())?;
    Ok(implementation_fingerprint(&[
        cutlass_fp8::source_declaration_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        cutlass_fp8::CUTLASS_FP8_NUMERICAL_IDENTITY.as_bytes(),
        &identity,
        include_bytes!("tensor_dense_fp8_bf16.rs"),
        include_bytes!("tensor_dense_fp8_bf16/weights.rs"),
        include_bytes!("../../../native_ops/cutlass_fp8.rs"),
        include_bytes!("../vnext_runtime/tensor_eager/cutlass_owner.rs"),
    ]))
}

pub(super) struct Provider {
    descriptor: OperationProviderDescriptor,
    device: Device,
    plans: Mutex<BTreeMap<(u32, u32, u32), Arc<PreparedCutlassFp8>>>,
    maximum_plans: usize,
    probe: Option<Arc<CudaCutlassFp8Probe>>,
    // Retained after all plan records. Commands also keep this owner alive.
    owner: Arc<CudaTensorCutlassRetention>,
}

impl Provider {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        config: &CudaCutlassFp8Config,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_with_probe(runtime, config, None)
    }

    pub(super) fn new_with_probe(
        runtime: &CudaDeviceRuntime,
        config: &CudaCutlassFp8Config,
        probe: Option<Arc<CudaCutlassFp8Probe>>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        config.validate()?;
        let capability =
            CapabilityId::new(DENSE_LINEAR_FP8_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "FP8 requires the explicit tensor runtime capability config",
            ));
        }
        let source = source_fingerprint().map_err(CudaDeviceRuntimeError::contract)?;
        let implementation = implementation_fingerprint(&[
            source.as_bytes(),
            config.opaque_memory_profile.as_bytes(),
            &config.opaque_memory_bytes.to_le_bytes(),
            &(config.maximum_plans as u64).to_le_bytes(),
        ]);
        // Claim before any CUDART/native planning call. Querying driver device
        // attributes below neither loads this module nor creates another context.
        let owner =
            runtime.reserve_tensor_cutlass_module(&implementation, config.opaque_memory_bytes)?;
        let implementation = if probe.is_some() {
            implementation_fingerprint(&[
                implementation.as_bytes(),
                QUANTIZATION_PROBE_PROFILE_ID.as_bytes(),
                include_bytes!("tensor_dense_fp8_bf16/probe.rs"),
            ])
        } else {
            implementation
        };
        use cudarc::driver::sys::CUdevice_attribute::*;
        let attr = |key| {
            owner
                .context()
                .attribute(key)
                .map_err(|e| CudaDeviceRuntimeError::driver("CUTLASS FP8 device attribute", e))
        };
        let nonnegative = |value: i32| {
            u32::try_from(value)
                .map_err(|_| CudaDeviceRuntimeError::contract("negative CUDA device attribute"))
        };
        let major = nonnegative(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?)?;
        let minor = nonnegative(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?)?;
        let device = Device {
            ordinal: runtime.descriptor().ordinal,
            architecture: major
                .checked_mul(100)
                .and_then(|n| minor.checked_mul(10).and_then(|m| n.checked_add(m)))
                .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA architecture overflows"))?,
            multiprocessors: nonnegative(attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)?,
            maximum_dynamic_shared_bytes: u64::from(nonnegative(attr(
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?)?),
        };
        Request::new(1, 16, 16, device)
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        let contract = dense_linear_fp8_bf16_contract().map_err(contract_error)?;
        let estimator =
            implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation,
            // Limited to the selected artifact/device profile, not cross-device
            // or deployment parity. Actual repeatability still needs validation.
            ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(super::DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?
            ]),
            BTreeSet::from([
                QuantizationFormatId::new(FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID)
                    .map_err(contract_error)?,
            ]),
            super::transformer::contiguous_bindings(2),
            ESTIMATOR,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self {
            descriptor,
            device,
            plans: Mutex::new(BTreeMap::new()),
            maximum_plans: config.maximum_plans,
            probe,
            owner,
        })
    }

    fn plan(
        &self,
        g: DenseLinearFp8Bf16Geometry,
        prepare: bool,
    ) -> Result<Arc<PreparedCutlassFp8>, String> {
        let cvt = |n| u32::try_from(n).map_err(|_| "FP8 geometry exceeds native u32".to_owned());
        let key = (cvt(g.rows())?, cvt(g.inputs())?, cvt(g.outputs())?);
        let request = Request::new(key.0, key.1, key.2, self.device).map_err(|e| e.to_string())?;
        let mut cache = self.plans.lock().map_err(|_| "FP8 plan cache poisoned")?;
        if let Some(plan) = cache.get(&key) {
            return Ok(Arc::clone(plan));
        }
        if !prepare {
            return Err("FP8 plan was not prepared by the admitted estimator".into());
        }
        if cache.len() >= self.maximum_plans {
            return Err("FP8 prepared plan capacity exhausted".into());
        }
        self.owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let plan = Arc::new(PreparedCutlassFp8::new(request).map_err(|e| e.to_string())?);
        cache.insert(key, Arc::clone(&plan));
        Ok(plan)
    }

    fn prepared_fingerprint(&self, plan: &PreparedCutlassFp8) -> String {
        implementation_fingerprint(&[
            self.descriptor
                .provider_implementation_fingerprint()
                .as_bytes(),
            &plan.fingerprint_material(),
        ])
    }
}

fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<DenseLinearFp8Bf16Geometry, String> {
    if values.len() != 3 || attrs.len() != 2 {
        return Err("FP8 dense requires three ports and exactly two shape attributes".into());
    }
    let x = binding(values, ResolvedValueRole::Input, 0)?;
    let w = binding(values, ResolvedValueRole::Input, 1)?;
    let y = binding(values, ResolvedValueRole::Output, 0)?;
    let geometry = DenseLinearFp8Bf16Geometry::new(
        x.tensor(),
        w.tensor(),
        y.tensor(),
        super::unsigned_attribute(attrs, "in_features")?,
        super::unsigned_attribute(attrs, "out_features")?,
    )
    .map_err(|e| e.to_string())?;
    weights::validate(w)?;
    Ok(geometry)
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
            return Err(invalid("FP8 estimator received another operation"));
        }
        let geometry = geometry(request.values(), request.attributes()).map_err(invalid)?;
        let plan = self.plan(geometry, true).map_err(invalid)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(plan.scratch_bytes())?,
            plan.alignment(),
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
        .with_prepared_implementation_fingerprint(self.prepared_fingerprint(&plan))
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
            "cuda.dense.fp8.encode",
            "FP8 dense requires fixed tensor work".into(),
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
            .map_err(|error| provider_failure(identity, "cuda.dense.fp8.encode", error))
    }
}

impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("FP8 dense requires one participant".into());
        };
        if invocation.operation().id.as_str() != DENSE_LINEAR_FP8_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("FP8 dense requires its fixed tensor contract".into());
        }
        let g = geometry(participant.bindings(), participant.attributes())?;
        let plan = self.plan(g, false)?;
        if invocation.prepared_implementation_fingerprint()
            != Some(self.prepared_fingerprint(&plan).as_str())
        {
            return Err("FP8 native plan differs from admitted prepared implementation".into());
        }
        let input = weights::plain(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            plan.plan()
                .request
                .input_bytes()
                .map_err(|e| e.to_string())?,
        )?;
        let [weight, scale] = weights::retain(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        let output = weights::plain(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            plan.plan()
                .request
                .output_bytes()
                .map_err(|e| e.to_string())?,
        )?;
        let scratch = weights::scratch(participant, plan.scratch_bytes(), plan.alignment())?;
        let regions = vec![input, weight, scale, output, scratch];
        if regions
            .iter()
            .any(|region| !Arc::ptr_eq(region.context(), self.owner.context()))
        {
            return Err("FP8 binding belongs to another CUDA context".into());
        }
        let bindings = spans(&regions)?;
        self.owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        plan.validate(&bindings).map_err(|e| e.to_string())?;
        let capture = self
            .probe
            .as_ref()
            .map(|probe| probe.claim(plan.plan().request.rows, plan.plan().request.inputs))
            .transpose()
            .map_err(|e| e.to_string())?;
        let transfers = if capture.is_some() { 2 } else { 0 };
        let owner = Arc::clone(&self.owner);
        CudaDeviceCommand::operation(
            cutlass_fp8::CUTLASS_FP8_OPERATOR,
            regions,
            move |stream, regions| {
                owner
                    .context()
                    .bind_to_thread()
                    .map_err(|e| CudaDeviceRuntimeError::driver("FP8 context bind", e))?;
                let bindings = spans(regions).map_err(CudaDeviceRuntimeError::contract)?;
                // All five exact spans and the context/module quota are retained by
                // this command through completion, including partial enqueue errors.
                unsafe { plan.run(&bindings, stream.cu_stream().cast()) }
                    .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
                if let Some(capture) = &capture {
                    capture.read(stream, &regions[4], &plan)?;
                }
                Ok(())
            },
        )
        .and_then(|command| {
            command.with_tensor_work(
                g.output_elements(),
                g.logical_arithmetic_operations(),
                2,
                transfers,
            )
        })
        .map_err(|e| e.to_string())
    }
}

fn spans(regions: &[CudaBufferRegion]) -> Result<Bindings, String> {
    let [input, weight, scale, output, scratch] = regions else {
        return Err("FP8 command must retain five ranges".into());
    };
    let span = |region: &CudaBufferRegion| DeviceSpan {
        address: region.device_ptr(),
        bytes: region.length_bytes(),
    };
    Ok(Bindings {
        input: span(input),
        weight: span(weight),
        weight_scale: span(scale),
        output: span(output),
        scratch: span(scratch),
    })
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
