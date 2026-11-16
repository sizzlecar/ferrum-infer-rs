//! Sage attention quantization shares the attention module's admitted owner.
use super::super::vnext_runtime::CudaTensorNativeModuleRetention;
use super::tensor_sparse_fp4_attention::module::Module;
use super::*;
use crate::native_ops::sage_vsa_fp4::{self as sage, Device, DeviceSpan, SageVsaFp4Config};
use ferrum_runtime::{
    NodeWorkContract, PhysicalWeightLayout, ProviderExecutionSemantics,
    SparseFp4QkvQuantizeGeometry, TensorOperationInvocation,
    SPARSE_FP4_QKV_QUANTIZE_BF16_CAPABILITY_ID, SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID,
};
use sage::quantize::{self as native, Bindings, PreparedSageVsaFp4Quantize, Request};
use std::collections::BTreeMap;

const ESTIMATOR: &str = "resource-estimator.cuda.sparse_fp4_qkv_quantize.sage_group16";
pub(super) struct Provider {
    descriptor: OperationProviderDescriptor,
    device: Device,
    owner: Arc<CudaTensorNativeModuleRetention>,
}
struct Prepared {
    execution: PreparedSageVsaFp4Quantize,
    geometry: SparseFp4QkvQuantizeGeometry,
    owner: Arc<CudaTensorNativeModuleRetention>,
    fingerprint: String,
    arithmetic: u64,
}
impl Provider {
    pub(super) fn new(
        runtime: &CudaDeviceRuntime,
        config: &SageVsaFp4Config,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let capabilities = BTreeSet::from([
            CapabilityId::new(SPARSE_FP4_QKV_QUANTIZE_BF16_CAPABILITY_ID)
                .map_err(contract_error)?,
            CapabilityId::new(sage::QUANTIZE_CAPABILITY_ID).map_err(contract_error)?,
        ]);
        if !native::compiled() || !capabilities.is_subset(&runtime.descriptor().capabilities) {
            return Err(CudaDeviceRuntimeError::contract(
                "Sage QKV quantization requires its explicit locked native capability",
            ));
        }
        let Module {
            device,
            owner,
            fingerprint: module_identity,
        } = Module::acquire(runtime, config)?;
        // The module admission identity stays identical to attention. This
        // descriptor additionally identifies the selected numerical producer.
        let implementation = implementation_fingerprint(&[
            module_identity.as_bytes(),
            SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID.as_bytes(),
            native::NUMERICAL_IDENTITY.as_bytes(),
            include_bytes!("tensor_sparse_fp4_quantize.rs"),
        ]);
        let contract =
            ferrum_runtime::sparse_fp4_qkv_quantize_bf16_contract().map_err(contract_error)?;
        let mut bindings = super::transformer::contiguous_bindings(3);
        for ordinal in 1..6 {
            bindings.push(ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Output,
                ordinal,
                DynamicStorageRequirement::contiguous(),
            ));
        }
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(sage::QUANTIZE_PROVIDER_ID).map_err(contract_error)?,
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
        Ok(Self {
            descriptor,
            device,
            owner,
        })
    }
    fn prepare(&self, geometry: SparseFp4QkvQuantizeGeometry) -> Result<Arc<Prepared>, String> {
        let request = Request::new(
            geometry.batches(),
            geometry.heads(),
            geometry.query_rows(),
            geometry.key_rows(),
            self.device,
        )
        .map_err(|e| e.to_string())?;
        self.owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let execution = PreparedSageVsaFp4Quantize::new(request).map_err(|e| e.to_string())?;
        let elements = geometry.input_elements().map_err(|e| e.to_string())?;
        // Logical abs/max/normalize per scalar, two scale divisions per group;
        // conversions, swizzles and padding-free byte packing are not FLOPs.
        let arithmetic = elements
            .checked_mul(3)
            .and_then(|n| {
                (elements / 16)
                    .checked_mul(2)
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or("quantization arithmetic overflows")?;
        let fingerprint = implementation_fingerprint(&[
            self.descriptor
                .provider_implementation_fingerprint()
                .as_bytes(),
            &execution.plan().fingerprint_material(),
        ]);
        Ok(Arc::new(Prepared {
            execution,
            geometry,
            owner: Arc::clone(&self.owner),
            fingerprint,
            arithmetic,
        }))
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<SparseFp4QkvQuantizeGeometry, String> {
    if values.len() != 9
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err(
            "QKV quantization requires exactly three BF16 inputs and six packed byte outputs"
                .into(),
        );
    }
    let [q, k, v] =
        [0, 1, 2].map(|n| binding(values, ResolvedValueRole::Input, n).map(|b| b.tensor()));
    let [pq, pk, pv, sq, sk, sv] = [0, 1, 2, 3, 4, 5]
        .map(|n| binding(values, ResolvedValueRole::Output, n).map(|b| b.tensor()));
    SparseFp4QkvQuantizeGeometry::new([q?, k?, v?], [pq?, pk?, pv?, sq?, sk?, sv?], attrs)
        .map_err(|e| e.to_string())
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
            return Err(invalid("QKV estimator received another operation"));
        }
        let prepared = self
            .prepare(geometry(r.values(), r.attributes()).map_err(invalid)?)
            .map_err(invalid)?;
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            16,
            None,
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
        i: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            i.participants()[0].identity().clone(),
            "cuda.sparse_fp4_qkv_quantize.encode",
            "requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(identity, "cuda.sparse_fp4_qkv_quantize.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = i.participants() else {
            return Err("QKV quantization requires one fixed participant".into());
        };
        if i.operation().id.as_str() != SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires QKV fixed tensor contract".into());
        }
        let geometry = geometry(participant.bindings(), participant.attributes())?;
        let prepared = i
            .prepared_implementation::<Prepared>()
            .ok_or("QKV prepared owner absent")?;
        if prepared.geometry != geometry
            || !Arc::ptr_eq(&prepared.owner, &self.owner)
            || i.prepared_implementation_fingerprint() != Some(prepared.fingerprint.as_str())
        {
            return Err("QKV prepared geometry/module owner/identity differs".into());
        }
        let mut regions = Vec::with_capacity(9);
        for (role, ordinal) in (0..3)
            .map(|n| (ResolvedValueRole::Input, n))
            .chain((0..6).map(|n| (ResolvedValueRole::Output, n)))
        {
            regions.push(super::tensor_storage::single_component(
                participant,
                binding(participant.bindings(), role, ordinal)?,
                16,
            )?);
        }
        if regions
            .iter()
            .any(|r| !Arc::ptr_eq(r.context(), self.owner.context()))
        {
            return Err("QKV span belongs to another context".into());
        }
        let bindings = bindings(&regions)?;
        self.owner
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        // SAFETY: all spans came from retained Core views in this owner context.
        unsafe { prepared.execution.validate(&bindings) }.map_err(|e| e.to_string())?;
        let elements = geometry.input_elements().map_err(|e| e.to_string())?;
        let arithmetic = prepared.arithmetic;
        CudaDeviceCommand::operation(
            sage::SAGE_VSA_FP4_OPERATOR,
            regions,
            move |stream, regions| {
                prepared
                    .owner
                    .context()
                    .bind_to_thread()
                    .map_err(|e| CudaDeviceRuntimeError::driver("QKV context bind", e))?;
                let bindings = bindings_for_command(regions)?;
                // SAFETY: the command retains this exact plan/module and every span.
                // Partial-submit failures retain them through ordinary quarantine.
                unsafe { prepared.execution.run(&bindings, stream.cu_stream().cast()) }
                    .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))
            },
        )
        .and_then(|c| c.with_tensor_work(elements, arithmetic, 3, 0))
        .map_err(|e| e.to_string())
    }
}
fn bindings(regions: &[CudaBufferRegion]) -> Result<Bindings, String> {
    let [q, k, v, pq, pk, pv, sq, sk, sv] = regions else {
        return Err("QKV quantization span count differs".into());
    };
    let span = |r: &CudaBufferRegion| DeviceSpan {
        address: r.device_ptr(),
        bytes: r.length_bytes(),
    };
    Ok(Bindings {
        query_input: span(q),
        key_input: span(k),
        value_input: span(v),
        query: span(pq),
        key: span(pk),
        value: span(pv),
        query_scales: span(sq),
        key_scales: span(sk),
        value_scales: span(sv),
    })
}
fn bindings_for_command(regions: &[CudaBufferRegion]) -> Result<Bindings, CudaDeviceRuntimeError> {
    bindings(regions).map_err(CudaDeviceRuntimeError::contract)
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
