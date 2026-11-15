//! IEEE conversion using admitted owners and no global scratch or workspace.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    HeadAxisPackGeometry, HeadAxisPackKind, PhysicalWeightLayout, WeightFormatId,
};
pub const NUMERICAL_PROFILE: &str = "cuda.pack_head_axis.f32_to_bf16.ieee_rne.gradual.v1";
pub const BITCOPY_NUMERICAL_PROFILE: &str = "cuda.pack_head_axis.bf16.exact_bits.v1";
const ESTIMATOR: &str = "resource-estimator.cuda.head_axis_pack";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: HeadAxisPackKind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks selected head-axis pack capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let profile = match kind {
        HeadAxisPackKind::F32ToBf16 => NUMERICAL_PROFILE,
        HeadAxisPackKind::Bf16 => BITCOPY_NUMERICAL_PROFILE,
    };
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("head_axis_pack.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_head_axis_pack.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}", kind.operation_id()))
            .map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(1),
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
    kind: HeadAxisPackKind,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kind,
        kernels: Arc::clone(kernels),
    })
}
struct Provider {
    kind: HeadAxisPackKind,
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
            return Err(invalid_plan("cast estimator received another operation"));
        }
        geometry(self.kind, r.values(), r.attributes()).map_err(invalid_plan)?;
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
            "cuda.head_axis_pack.encode",
            "head-axis pack requires fixed tensor execution".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.head_axis_pack.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("head-axis pack requires one tensor participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("head-axis pack requires its selected fixed tensor contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(2);
        for (role, dtype, size) in [
            (
                ResolvedValueRole::Input,
                self.kind.input_type(),
                self.kind.input_bytes(),
            ),
            (
                ResolvedValueRole::Output,
                self.kind.output_type(),
                self.kind.output_bytes(),
            ),
        ] {
            let b = binding(p.bindings(), role, 0)?;
            let [c] = b.storage().components() else {
                return Err("head-axis pack requires one component per port".into());
            };
            if c.element_type() != dtype || g.elements().checked_mul(size) != Some(c.length_bytes())
            {
                return Err("cast physical dtype or exact byte extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        dense::disjoint(
            regions[0].device_ptr(),
            regions[0].length_bytes(),
            regions[1].device_ptr(),
            regions[1].length_bytes(),
        )?;
        let kernels = Arc::clone(&self.kernels);
        let kind = self.kind;
        CudaDeviceCommand::operation(kind.operation_id(), regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let y = regions[1].device_ptr();
            let elements = g.elements();
            let mut builder = stream.launch_builder(kernels.head_axis_pack_function());
            let direction = match kind {
                HeadAxisPackKind::F32ToBf16 => 1i32,
                HeadAxisPackKind::Bf16 => 0,
            };
            let heads = g.heads();
            let rows = g.rows();
            let head_dim = g.head_dim();
            builder
                .arg(&x)
                .arg(&y)
                .arg(&elements)
                .arg(&heads)
                .arg(&rows)
                .arg(&head_dim)
                .arg(&direction);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("head-axis pack launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), g.arithmetic_operations(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: HeadAxisPackKind,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<HeadAxisPackGeometry, String> {
    if b.len() != 2
        || !attrs.is_empty()
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("head-axis pack requires two dense ports and no attributes".into());
    }
    HeadAxisPackGeometry::new(
        kind,
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}
