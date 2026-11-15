//! Packed NVFP4 operation on the existing tensor runtime/submission path.
//! Each encoded command retains the admitted operands, output, workspace and
//! common native module owner through completion, including partial errors.
use super::super::vnext_runtime::{CudaTensorNativeModuleRetention, StaticNativeModule};
use super::*;
use crate::native_ops::cutlass_nvfp4::{
    self, Bindings, Device, DeviceSpan, PreparedCutlassNvfp4, Request,
};
pub use cutlass_nvfp4::CutlassNvfp4Config;
use ferrum_runtime::{
    nvfp4_matmul_bf16_contract, NodeWorkContract, Nvfp4MatmulBf16Geometry,
    ProviderExecutionSemantics, TensorOperationInvocation, NVFP4_E2M1_E4M3_128X4_FORMAT_ID,
    NVFP4_MATMUL_BF16_CAPABILITY_ID, NVFP4_MATMUL_BF16_OPERATION_ID,
};
use std::collections::BTreeMap;
const ESTIMATOR: &str = "resource-estimator.cuda.nvfp4_matmul.sm120.k128";
mod quantize;

pub(super) fn providers(
    runtime: &CudaDeviceRuntime,
    config: &CutlassNvfp4Config,
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    for capability in [
        ferrum_runtime::NVFP4_QUANTIZE_BF16_CAPABILITY_ID,
        cutlass_nvfp4::QUANTIZE_CAPABILITY_ID,
    ] {
        if !runtime
            .descriptor()
            .capabilities
            .contains(&CapabilityId::new(capability).map_err(contract_error)?)
        {
            return Err(CudaDeviceRuntimeError::contract(
                "NVFP4 quantization capability absent",
            ));
        }
    }
    let shared = Arc::new(Provider::new(runtime, config)?);
    Ok(vec![
        Box::new(shared.as_ref().clone()),
        Box::new(quantize::Provider::new(runtime, shared)?),
    ])
}

pub(super) fn source_fingerprint() -> Result<String, String> {
    if !cutlass_nvfp4::compiled() {
        return Err("NVFP4 requires locked cuda-cutlass-nvfp4 artifact".into());
    }
    let mut artifacts = crate::native_ops::compiled_native_operator_artifacts()
        .iter()
        .filter(|a| a.operator == cutlass_nvfp4::CUTLASS_NVFP4_OPERATOR);
    let artifact = artifacts.next().ok_or("NVFP4 artifact absent")?;
    if artifacts.next().is_some() {
        return Err("NVFP4 artifact identity ambiguous".into());
    }
    Ok(implementation_fingerprint(&[
        cutlass_nvfp4::source_declaration_fingerprint().as_bytes(),
        &serde_json::to_vec(artifact).map_err(|e| e.to_string())?,
    ]))
}
#[derive(Clone)]
pub(super) struct Provider {
    descriptor: OperationProviderDescriptor,
    device: Device,
    owner: Arc<CudaTensorNativeModuleRetention>,
}
impl Provider {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        config: &CutlassNvfp4Config,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        config
            .validate()
            .map_err(CudaDeviceRuntimeError::contract)?;
        let capabilities = BTreeSet::from([
            CapabilityId::new(NVFP4_MATMUL_BF16_CAPABILITY_ID).map_err(contract_error)?,
            CapabilityId::new(cutlass_nvfp4::CAPABILITY_ID).map_err(contract_error)?,
        ]);
        if !capabilities.is_subset(&runtime.descriptor().capabilities) {
            return Err(CudaDeviceRuntimeError::contract(
                "NVFP4 requires explicit SM120 K128 capability configuration",
            ));
        }
        let implementation = implementation_fingerprint(&[
            source_fingerprint()
                .map_err(CudaDeviceRuntimeError::contract)?
                .as_bytes(),
            config.opaque_memory_profile.as_bytes(),
            &config.opaque_memory_bytes.to_le_bytes(),
        ]);
        let owner = runtime.reserve_tensor_native_module(
            StaticNativeModule::CutlassNvfp4,
            &implementation,
            config.opaque_memory_bytes,
        )?;
        use cudarc::driver::sys::CUdevice_attribute::*;
        let attr = |key| -> Result<u32, CudaDeviceRuntimeError> {
            let value = owner
                .context()
                .attribute(key)
                .map_err(|e| CudaDeviceRuntimeError::driver("NVFP4 device attribute", e))?;
            u32::try_from(value)
                .map_err(|_| CudaDeviceRuntimeError::contract("negative NVFP4 device attribute"))
        };
        let major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let device = Device {
            ordinal: runtime.descriptor().ordinal,
            architecture: major
                .checked_mul(100)
                .and_then(|v| minor.checked_mul(10).and_then(|m| v.checked_add(m)))
                .ok_or_else(|| CudaDeviceRuntimeError::contract("invalid NVFP4 architecture"))?,
            multiprocessors: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            maximum_dynamic_shared_bytes: attr(
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?
            .into(),
        };
        Request::new(1, 128, 8, device)
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        let contract = nvfp4_matmul_bf16_contract().map_err(contract_error)?;
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(cutlass_nvfp4::PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation.clone(),
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            capabilities,
            BTreeSet::from([
                WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?
            ]),
            BTreeSet::from([QuantizationFormatId::new(NVFP4_E2M1_E4M3_128X4_FORMAT_ID)
                .map_err(contract_error)?]),
            super::transformer::contiguous_bindings(5),
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
        })
    }
    fn plan(&self, geometry: Nvfp4MatmulBf16Geometry) -> Result<PreparedCutlassNvfp4, String> {
        let u32_shape =
            |value| u32::try_from(value).map_err(|_| "NVFP4 native shape exceeds u32".to_owned());
        let request = Request::new(
            u32_shape(geometry.a().rows())?,
            u32_shape(geometry.a().columns())?,
            u32_shape(geometry.b().rows())?,
            self.device,
        )
        .map_err(|e| e.to_string())?;
        self.owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        PreparedCutlassNvfp4::new(request).map_err(|e| e.to_string())
    }
    fn prepared_fingerprint(&self, plan: &PreparedCutlassNvfp4) -> String {
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
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Nvfp4MatmulBf16Geometry, String> {
    if values.len() != 6 || !attributes.is_empty() {
        return Err("NVFP4 requires five input ports, one output, no attributes".into());
    }
    let inputs =
        [0, 1, 2, 3, 4].map(|i| binding(values, ResolvedValueRole::Input, i).map(|v| v.tensor()));
    let [a, b, sa, sb, alpha] = inputs;
    let output = binding(values, ResolvedValueRole::Output, 0)?;
    Nvfp4MatmulBf16Geometry::new([a?, b?, sa?, sb?, alpha?], output.tensor())
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
            return Err(invalid("NVFP4 estimator received another contract"));
        }
        let plan = self
            .plan(geometry(request.values(), request.attributes()).map_err(invalid)?)
            .map_err(invalid)?;
        let scratch = if plan.workspace_bytes() == 0 {
            None
        } else {
            Some(ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::fixed(plan.workspace_bytes())?,
                plan.alignment(),
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?)
        };
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            scratch,
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
            "cuda.nvfp4.encode",
            "NVFP4 requires fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.nvfp4.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("NVFP4 requires one fixed participant".into());
        };
        if invocation.operation().id.as_str() != NVFP4_MATMUL_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("NVFP4 requires its fixed tensor contract".into());
        }
        let geometry = geometry(participant.bindings(), participant.attributes())?;
        let plan = self.plan(geometry)?;
        if invocation.prepared_implementation_fingerprint()
            != Some(self.prepared_fingerprint(&plan).as_str())
        {
            return Err("NVFP4 prepared plan differs from admitted implementation".into());
        }
        let mut regions = Vec::with_capacity(7);
        for (role, index) in (0..5)
            .map(|i| (ResolvedValueRole::Input, i))
            .chain([(ResolvedValueRole::Output, 0)])
        {
            let value = binding(participant.bindings(), role, index)?;
            regions.push(super::tensor_storage::single_component(
                participant,
                value,
                if index == 4 && role == ResolvedValueRole::Input {
                    4
                } else {
                    16
                },
            )?);
        }
        if plan.workspace_bytes() > 0 {
            regions.push(super::tensor_storage::scratch(
                participant,
                plan.workspace_bytes(),
                plan.alignment(),
            )?);
        }
        if regions
            .iter()
            .any(|r| !Arc::ptr_eq(r.context(), self.owner.context()))
        {
            return Err("NVFP4 binding belongs to another CUDA context".into());
        }
        plan.validate(&spans(&regions)?)
            .map_err(|e| e.to_string())?;
        let owner = Arc::clone(&self.owner);
        CudaDeviceCommand::operation(
            cutlass_nvfp4::CUTLASS_NVFP4_OPERATOR,
            regions,
            move |stream, regions| {
                owner
                    .context()
                    .bind_to_thread()
                    .map_err(|e| CudaDeviceRuntimeError::driver("NVFP4 context bind", e))?;
                let bindings = spans(regions).map_err(CudaDeviceRuntimeError::contract)?;
                // All admitted buffers and this module owner are captured by the
                // command, retained by ordinary runtime completion/quarantine.
                unsafe { plan.run(&bindings, stream.cu_stream().cast()) }
                    .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
            },
        )
        .and_then(|command| {
            command.with_tensor_work(
                geometry.a().rows() * geometry.b().rows(),
                geometry.logical_arithmetic_operations(),
                1,
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}
fn spans(regions: &[CudaBufferRegion]) -> Result<Bindings, String> {
    if !(6..=7).contains(&regions.len()) {
        return Err("NVFP4 requires six operand ranges and optional workspace".into());
    }
    let span = |i: usize| DeviceSpan {
        address: regions[i].device_ptr(),
        bytes: regions[i].length_bytes(),
    };
    Ok(Bindings {
        input: span(0),
        weight: span(1),
        input_scales: span(2),
        weight_scales: span(3),
        alpha: span(4),
        output: span(5),
        workspace: if regions.len() == 7 {
            span(6)
        } else {
            DeviceSpan::default()
        },
    })
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
