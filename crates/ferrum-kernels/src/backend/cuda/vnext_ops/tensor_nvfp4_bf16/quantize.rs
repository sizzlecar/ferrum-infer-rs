//! Pure activation producer. Quantization and GEMM retain one module owner.
use super::*;
use cutlass_nvfp4::{PreparedNvfp4QuantizeBf16, QuantizeBindings, QuantizeRequest};
use ferrum_runtime::{
    nvfp4_quantize_bf16_contract, Nvfp4QuantizeBf16Geometry, ProviderStorageBindingRequirement,
    NVFP4_QUANTIZE_BF16_CAPABILITY_ID, NVFP4_QUANTIZE_BF16_OPERATION_ID,
};

const ESTIMATOR: &str = "resource-estimator.cuda.nvfp4_quantize.bf16.rcp-approx-ftz.sm120";
pub(super) struct Provider {
    descriptor: OperationProviderDescriptor,
    shared: Arc<super::Provider>,
}
impl Provider {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        shared: Arc<super::Provider>,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let capabilities = BTreeSet::from([
            CapabilityId::new(NVFP4_QUANTIZE_BF16_CAPABILITY_ID).map_err(contract_error)?,
            CapabilityId::new(cutlass_nvfp4::QUANTIZE_CAPABILITY_ID).map_err(contract_error)?,
        ]);
        if !capabilities.is_subset(&runtime.descriptor().capabilities) {
            return Err(CudaDeviceRuntimeError::contract(
                "explicit NVFP4 activation profile capability absent",
            ));
        }
        let contract = nvfp4_quantize_bf16_contract().map_err(contract_error)?;
        let implementation = implementation_fingerprint(&[
            shared
                .descriptor
                .provider_implementation_fingerprint()
                .as_bytes(),
            cutlass_nvfp4::NVFP4_QUANTIZE_BF16_NUMERICAL_IDENTITY.as_bytes(),
            include_bytes!("quantize.rs"),
        ]);
        let mut bindings = super::super::transformer::contiguous_bindings(2);
        bindings.push(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            1,
            DynamicStorageRequirement::contiguous(),
        ));
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(cutlass_nvfp4::QUANTIZE_PROVIDER_ID).map_err(contract_error)?,
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
            BTreeSet::new(),
            bindings,
            ESTIMATOR,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]),
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor, shared })
    }
    fn plan(
        &self,
        geometry: Nvfp4QuantizeBf16Geometry,
    ) -> Result<PreparedNvfp4QuantizeBf16, String> {
        let matrix = geometry.matrix();
        let request = QuantizeRequest::new(
            u32::try_from(matrix.rows()).map_err(|_| "NVFP4 quantize rows exceed native ABI")?,
            u32::try_from(matrix.columns())
                .map_err(|_| "NVFP4 quantize width exceeds native ABI")?,
            self.shared.device,
        )
        .map_err(|e| e.to_string())?;
        self.shared
            .owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        PreparedNvfp4QuantizeBf16::new(request).map_err(|e| e.to_string())
    }
    fn fingerprint(&self, plan: &PreparedNvfp4QuantizeBf16) -> String {
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
            return Err("NVFP4 quantize needs one fixed participant".into());
        };
        if invocation.operation().id.as_str() != NVFP4_QUANTIZE_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("NVFP4 quantize requires its fixed tensor contract".into());
        }
        let geometry = geometry(participant.bindings(), participant.attributes())?;
        let plan = self.plan(geometry)?;
        if invocation.prepared_implementation_fingerprint()
            != Some(self.fingerprint(&plan).as_str())
        {
            return Err("NVFP4 quantize plan differs from admission".into());
        }
        let mut regions = Vec::with_capacity(4);
        for (role, index) in [
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Input, 1),
            (ResolvedValueRole::Output, 0),
            (ResolvedValueRole::Output, 1),
        ] {
            let value = binding(participant.bindings(), role, index)?;
            regions.push(super::super::tensor_storage::single_component(
                participant,
                value,
                if role == ResolvedValueRole::Input && index == 1 {
                    4
                } else {
                    16
                },
            )?);
        }
        if regions
            .iter()
            .any(|r| !Arc::ptr_eq(r.context(), self.shared.owner.context()))
        {
            return Err("NVFP4 quantize binding belongs to another CUDA context".into());
        }
        plan.validate(&spans(&regions)?)
            .map_err(|e| e.to_string())?;
        let shared = Arc::clone(&self.shared);
        CudaDeviceCommand::operation(
            "ferrum.cuda.nvfp4_quantize.bf16",
            regions,
            move |stream, regions| {
                shared
                    .owner
                    .context()
                    .bind_to_thread()
                    .map_err(|e| CudaDeviceRuntimeError::driver("NVFP4 quantize context", e))?;
                let bindings = spans(regions).map_err(CudaDeviceRuntimeError::contract)?;
                // All regions and the common admitted module owner stay captured
                // through ordinary completion, including partial-error quarantine.
                unsafe { plan.run(&bindings, stream.cu_stream().cast()) }
                    .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
            },
        )
        .and_then(|c| c.with_tensor_work(geometry.matrix().elements(), 0, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Nvfp4QuantizeBf16Geometry, String> {
    if values.len() != 4 {
        return Err("NVFP4 quantize requires two inputs and two outputs".into());
    }
    Nvfp4QuantizeBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Input, 1)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 1)?.tensor(),
        attributes,
    )
    .map_err(|e| e.to_string())
}
fn spans(regions: &[CudaBufferRegion]) -> Result<QuantizeBindings, String> {
    if regions.len() != 4 {
        return Err("NVFP4 quantize requires four exact spans".into());
    }
    let s = |i: usize| DeviceSpan {
        address: regions[i].device_ptr(),
        bytes: regions[i].length_bytes(),
    };
    Ok(QuantizeBindings {
        input: s(0),
        global_scale: s(1),
        packed: s(2),
        scales: s(3),
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
            return Err(invalid(
                "NVFP4 quantize estimator received another contract",
            ));
        }
        let plan = self
            .plan(geometry(request.values(), request.attributes()).map_err(invalid)?)
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
            "cuda.nvfp4.quantize.encode",
            "NVFP4 quantize needs fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.nvfp4.quantize.encode", e))
    }
}
