use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    MaskedTileMeanBf16F32Geometry, PhysicalWeightLayout, WeightFormatId,
    MASKED_TILE_MEAN_BF16_F32_CAPABILITY_ID, MASKED_TILE_MEAN_BF16_F32_OPERATION_ID,
};
pub const NUMERICAL_PROFILE: &str = "cuda.masked_tile_mean.f32_add_rn_div_rn.gradual.v1";
const ESTIMATOR: &str = "resource-estimator.cuda.masked_tile_mean";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(MASKED_TILE_MEAN_BF16_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "missing masked tile mean capability",
        ));
    }
    let contract = ferrum_runtime::masked_tile_mean_bf16_f32_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("masked_tile_mean.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_masked_tile_mean_bf16_f32.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.masked_tile_mean.bf16_f32").map_err(contract_error)?,
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
        super::super::transformer::contiguous_bindings(2),
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
                "masked mean estimator received another operation",
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
            "cuda.masked_tile_mean.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.masked_tile_mean.encode", e))
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
        if i.operation().id.as_str() != MASKED_TILE_MEAN_BF16_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires masked tile mean fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(3);
        for (role, ordinal, dtype, bytes) in [
            (
                ResolvedValueRole::Input,
                0,
                ElementType::Bf16,
                g.input_bytes(),
            ),
            (
                ResolvedValueRole::Input,
                1,
                ElementType::I32,
                g.count_bytes(),
            ),
            (
                ResolvedValueRole::Output,
                0,
                ElementType::F32,
                g.output_bytes(),
            ),
        ] {
            let b = binding(p.bindings(), role, ordinal)?;
            let [c] = b.storage().components() else {
                return Err("requires one component per port".into());
            };
            if c.element_type() != dtype || c.length_bytes() != bytes {
                return Err("masked mean physical dtype/extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        for r in &regions[..2] {
            dense::disjoint(
                r.device_ptr(),
                r.length_bytes(),
                regions[2].device_ptr(),
                regions[2].length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            MASKED_TILE_MEAN_BF16_F32_OPERATION_ID,
            regions,
            move |stream, regions| {
                let x = regions[0].device_ptr();
                let c = regions[1].device_ptr();
                let y = regions[2].device_ptr();
                let elements = g.output_elements();
                let rows = g.source_rows();
                let columns = g.columns();
                let heads = g.heads();
                let tiles = g.tiles();
                let tile_rows = g.tile_rows();
                let dim = g.head_dim();
                let mut builder = stream.launch_builder(kernels.masked_tile_mean_function());
                builder
                    .arg(&x)
                    .arg(&c)
                    .arg(&y)
                    .arg(&elements)
                    .arg(&rows)
                    .arg(&columns)
                    .arg(&heads)
                    .arg(&tiles)
                    .arg(&tile_rows)
                    .arg(&dim);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| CudaDeviceRuntimeError::driver("masked tile mean launch", e))
            },
        )
        .and_then(|c| c.with_tensor_work(g.output_elements(), g.arithmetic_operations(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<MaskedTileMeanBf16F32Geometry, String> {
    if b.len() != 3
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("masked tile mean requires three dense ports".into());
    }
    MaskedTileMeanBf16F32Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Input, 1)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
