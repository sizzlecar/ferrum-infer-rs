use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    HeadFp8RoundtripBf16Geometry, PhysicalWeightLayout, WeightFormatId,
    HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID, INDEXED_HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID,
};
pub const NUMERICAL_PROFILE: &str =
    "cuda.head_fp8_roundtrip_bf16.amax_div448_divfull_zero_inverse_e4m3fn_rne_bf16.gradual.v1";
const ESTIMATOR: &str = "resource-estimator.cuda.head_fp8_roundtrip_bf16";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, false)
}
pub(super) fn descriptor_for(
    device: &DeviceDescriptor,
    indexed: bool,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(if indexed {
        INDEXED_HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID
    } else {
        HEAD_FP8_ROUNDTRIP_BF16_CAPABILITY_ID
    })
    .map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "missing head FP8 roundtrip capability",
        ));
    }
    let contract = if indexed {
        ferrum_runtime::indexed_head_fp8_roundtrip_bf16_contract()
    } else {
        ferrum_runtime::head_fp8_roundtrip_bf16_contract()
    }
    .map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("head_fp8_roundtrip_bf16.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_head_fp8_roundtrip_bf16.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(if indexed {
            "provider.cuda.indexed_head_fp8_roundtrip.bf16"
        } else {
            "provider.cuda.head_fp8_roundtrip.bf16"
        })
        .map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(if indexed { 2 } else { 1 }),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
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
            return Err(invalid_plan(
                "head FP8 roundtrip estimator received another operation",
            ));
        }
        geometry(r.values(), r.attributes()).map_err(invalid_plan)?;
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
            "cuda.head_fp8_roundtrip_bf16.encode",
            "requires fixed tensor execution".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.head_fp8_roundtrip_bf16.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one tensor participant".into());
        };
        if i.operation().id != *self.descriptor.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires head FP8 roundtrip fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(2);
        for (role, ordinal, bytes) in [
            (ResolvedValueRole::Input, 0, g.source_bytes()),
            (ResolvedValueRole::Output, 0, g.data_bytes()),
        ] {
            let b = binding(p.bindings(), role, ordinal)?;
            let [c] = b.storage().components() else {
                return Err("head FP8 roundtrip requires one component per port".into());
            };
            if c.element_type() != ElementType::Bf16 || c.length_bytes() != bytes {
                return Err("head FP8 roundtrip physical dtype/extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        if let Some(gather) = g.gather() {
            let b = binding(p.bindings(), ResolvedValueRole::Input, 1)?;
            let [c] = b.storage().components() else {
                return Err("indexed head FP8 requires one map component".into());
            };
            if c.element_type() != ElementType::I32 || c.length_bytes() != gather.index_bytes() {
                return Err("indexed head FP8 map dtype/extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        for i in 0..regions.len() {
            for j in i + 1..regions.len() {
                dense::disjoint(
                    regions[i].device_ptr(),
                    regions[i].length_bytes(),
                    regions[j].device_ptr(),
                    regions[j].length_bytes(),
                )?;
            }
        }
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(g.operation_id(), regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let y = regions[1].device_ptr();
            let groups = g.groups();
            let head_dim = g.head_dim();
            let mut builder = stream.launch_builder(kernels.head_fp8_roundtrip_bf16_function());
            let (indices, source_rows, output_rows, columns) =
                g.gather().map_or((0u64, 0, 0, 0), |gather| {
                    (
                        regions[2].device_ptr(),
                        gather.source_rows(),
                        gather.output_rows(),
                        gather.columns(),
                    )
                });
            builder
                .arg(&x)
                .arg(&y)
                .arg(&groups)
                .arg(&head_dim)
                .arg(&indices)
                .arg(&source_rows)
                .arg(&output_rows)
                .arg(&columns);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (
                        if head_dim <= 256 {
                            groups.div_ceil(4)
                        } else {
                            groups
                        }
                        .min(65535) as u32,
                        1,
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("head FP8 roundtrip launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), g.arithmetic(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<HeadFp8RoundtripBf16Geometry, String> {
    if ![2, 3].contains(&b.len())
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("head FP8 roundtrip requires two dense ports".into());
    }
    let x = binding(b, ResolvedValueRole::Input, 0)?.tensor();
    let y = binding(b, ResolvedValueRole::Output, 0)?.tensor();
    if b.len() == 3 {
        HeadFp8RoundtripBf16Geometry::indexed(
            x,
            binding(b, ResolvedValueRole::Input, 1)?.tensor(),
            y,
            attrs,
        )
    } else {
        HeadFp8RoundtripBf16Geometry::new(x, y, attrs)
    }
    .map_err(|e| e.to_string())
}
