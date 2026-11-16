//! Explicit diagnostic provider. Only tensor storage/submission is shared with
//! the candidate; quantization executes an independently pinned upstream DSO.
use super::super::vnext_runtime::{CudaTensorNativeModuleRetention, StaticNativeModule};
use super::*;
use crate::native_ops::nvfp4_reference::{CudaNvfp4ReferenceConfig, NVFP4_REFERENCE_PROVIDER_ID};
use ferrum_native_ops::cutlass_nvfp4::{
    CutlassNvfp4Device, CutlassNvfp4Span, Nvfp4QuantizeBf16BindingsV1, Nvfp4QuantizeBf16PlanV1,
    Nvfp4QuantizeBf16RequestV1,
};
use ferrum_runtime::{
    nvfp4_quantize_bf16_contract, NodeWorkContract, Nvfp4QuantizeBf16Geometry,
    ProviderExecutionSemantics, TensorOperationInvocation, NVFP4_QUANTIZE_BF16_CAPABILITY_ID,
    NVFP4_QUANTIZE_BF16_OPERATION_ID,
};
use std::collections::BTreeMap;

type Run = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *mut u8,
    *mut u8,
    u32,
    u32,
    *mut std::ffi::c_void,
) -> i32;
const ESTIMATOR: &str = "resource-estimator.cuda.diagnostic.flashinfer_nvfp4.v1";
struct Artifact {
    run: Run,
    // Unload only after every captured command is terminal; context drops last.
    _library: libloading::Library,
    owner: Arc<CudaTensorNativeModuleRetention>,
}
pub(super) struct Provider {
    descriptor: OperationProviderDescriptor,
    artifact: Arc<Artifact>,
    device: CutlassNvfp4Device,
}
impl Provider {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        config: &CudaNvfp4ReferenceConfig,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let capability =
            CapabilityId::new(NVFP4_QUANTIZE_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "reference requires explicit NVFP4 quantization capability",
            ));
        }
        let verified = config.verify().map_err(CudaDeviceRuntimeError::contract)?;
        let implementation = implementation_fingerprint(&[
            verified.identity.as_bytes(),
            include_bytes!("nvfp4_reference.rs"),
            include_bytes!("../../../native_ops/nvfp4_reference.rs"),
        ]);
        let owner = runtime.reserve_tensor_native_module(
            StaticNativeModule::Nvfp4QualificationReference,
            &implementation,
            config.opaque_memory_bytes,
        )?;
        owner
            .context()
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("reference CUDA context", e))?;
        use cudarc::driver::sys::CUdevice_attribute::*;
        let attr = |key| {
            owner
                .context()
                .attribute(key)
                .map_err(|e| CudaDeviceRuntimeError::driver("reference CUDA attribute", e))
        };
        let architecture = (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 100
            + attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? * 10)
            as u32;
        if architecture != 1200 {
            return Err(CudaDeviceRuntimeError::contract(
                "reference profile requires SM120",
            ));
        }
        let device = CutlassNvfp4Device {
            ordinal: runtime.descriptor().ordinal,
            architecture,
            multiprocessors: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32,
            maximum_dynamic_shared_bytes: 0,
        };
        // Loading is authorized only by this explicit diagnostic configuration.
        // Pins are checked before executing any library initializer and the
        // capacity claim predates loading. Artifacts must remain immutable.
        let library = unsafe { libloading::Library::new(&verified.library) }.map_err(|e| {
            CudaDeviceRuntimeError::contract(format!("reference library load: {e}"))
        })?;
        let run = *unsafe { library.get::<Run>(b"ferrum_reference_nvfp4_quantize_bf16_v1\0") }
            .map_err(|e| CudaDeviceRuntimeError::contract(format!("reference ABI: {e}")))?;
        let contract = nvfp4_quantize_bf16_contract().map_err(contract_error)?;
        let mut bindings = transformer::contiguous_bindings(2);
        bindings.push(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            1,
            DynamicStorageRequirement::contiguous(),
        ));
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(NVFP4_REFERENCE_PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation.clone(),
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?
            ]),
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
            artifact: Arc::new(Artifact {
                run,
                _library: library,
                owner,
            }),
            device,
        })
    }
    fn plan(
        &self,
        values: &[ResolvedValueBinding],
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Nvfp4QuantizeBf16PlanV1, String> {
        if values.len() != 4 {
            return Err("reference requires two inputs and two outputs".into());
        }
        let geometry = Nvfp4QuantizeBf16Geometry::new(
            binding(values, ResolvedValueRole::Input, 0)?.tensor(),
            binding(values, ResolvedValueRole::Input, 1)?.tensor(),
            binding(values, ResolvedValueRole::Output, 0)?.tensor(),
            binding(values, ResolvedValueRole::Output, 1)?.tensor(),
            attrs,
        )
        .map_err(|e| e.to_string())?;
        // Only shape/span validation is reused. This does not invoke or load the
        // candidate's numerical kernel or its C ABI planning function.
        Nvfp4QuantizeBf16RequestV1::new(
            u32::try_from(geometry.matrix().rows()).map_err(|_| "reference rows exceed ABI")?,
            u32::try_from(geometry.matrix().columns())
                .map_err(|_| "reference width exceeds ABI")?,
            self.device,
        )
        .and_then(|r| r.plan())
        .map_err(|e| e.to_string())
    }
    fn fingerprint(&self, plan: &Nvfp4QuantizeBf16PlanV1) -> String {
        implementation_fingerprint(&[
            self.descriptor
                .provider_implementation_fingerprint()
                .as_bytes(),
            &plan.fingerprint_material(),
        ])
    }
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("reference needs one fixed participant".into());
        };
        if invocation.operation().id.as_str() != NVFP4_QUANTIZE_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("reference requires fixed quantization tensor work".into());
        }
        let plan = self.plan(participant.bindings(), participant.attributes())?;
        if invocation.prepared_implementation_fingerprint()
            != Some(self.fingerprint(&plan).as_str())
        {
            return Err("reference plan differs from admission".into());
        }
        let mut regions = Vec::with_capacity(4);
        for (role, index) in [
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Input, 1),
            (ResolvedValueRole::Output, 0),
            (ResolvedValueRole::Output, 1),
        ] {
            regions.push(tensor_storage::single_component(
                participant,
                binding(participant.bindings(), role, index)?,
                if role == ResolvedValueRole::Input && index == 1 {
                    4
                } else {
                    16
                },
            )?);
        }
        if regions
            .iter()
            .any(|r| !Arc::ptr_eq(r.context(), self.artifact.owner.context()))
        {
            return Err("reference bindings belong to another CUDA context".into());
        }
        spans(&regions)?
            .validate(&plan)
            .map_err(|e| e.to_string())?;
        let artifact = Arc::clone(&self.artifact);
        CudaDeviceCommand::operation(
            "ferrum.cuda.diagnostic.flashinfer_nvfp4",
            regions,
            move |stream, regions| {
                artifact
                    .owner
                    .context()
                    .bind_to_thread()
                    .map_err(|e| CudaDeviceRuntimeError::driver("reference context", e))?;
                let s = spans(regions).map_err(CudaDeviceRuntimeError::contract)?;
                // Retained regions/DSO/module claim survive completion or quarantine.
                let status = unsafe {
                    (artifact.run)(
                        s.input.address as *const u16,
                        s.global_scale.address as *const f32,
                        s.packed.address as *mut u8,
                        s.scales.address as *mut u8,
                        plan.request.rows,
                        plan.request.inputs,
                        stream.cu_stream().cast(),
                    )
                };
                if status == 0 {
                    Ok(())
                } else {
                    Err(CudaDeviceRuntimeError::contract(format!(
                        "reference CUDA status {status}"
                    )))
                }
            },
        )
        .and_then(|c| {
            c.with_tensor_work(
                u64::from(plan.request.rows) * u64::from(plan.request.inputs),
                0,
                1,
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}
fn spans(regions: &[CudaBufferRegion]) -> Result<Nvfp4QuantizeBf16BindingsV1, String> {
    if regions.len() != 4 {
        return Err("reference requires four exact spans".into());
    }
    let span = |i: usize| CutlassNvfp4Span {
        address: regions[i].device_ptr(),
        bytes: regions[i].length_bytes(),
    };
    Ok(Nvfp4QuantizeBf16BindingsV1 {
        input: span(0),
        global_scale: span(1),
        packed: span(2),
        scales: span(3),
    })
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
            return Err(invalid("reference estimator received another contract"));
        }
        let plan = self
            .plan(request.values(), request.attributes())
            .map_err(invalid)?;
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            None,
        )
        .with_prepared_implementation_fingerprint(self.fingerprint(&plan))
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
            "cuda.reference.nvfp4.encode",
            "reference needs fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.reference.nvfp4.encode", e))
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
