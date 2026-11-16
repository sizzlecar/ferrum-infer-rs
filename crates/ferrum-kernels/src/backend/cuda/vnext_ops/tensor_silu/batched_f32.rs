//! Explicit batch/group F32 products using the existing lane BLAS and PTX owner.
use super::super::tensor_dense_bf16 as dense;
use super::*;
use ferrum_runtime::{BatchedF32Geometry, BatchedF32Op, PhysicalWeightLayout, WeightFormatId};
mod launch;
pub const NUMERICAL_PROFILE: &str =
    "cuda.batched_f32.cublas_pedantic.scale_rn.softmax_exp_tree256.v1";
const IDENTITY: &str = concat!(
    "F32-two-explicit-batch-axes-no-broadcast;matmul=CUBLAS_COMPUTE_32F_PEDANTIC;",
    "alpha1-beta0;no-TF32;atomics-not-allowed;workspace=null0;vendor-reduction;",
    "scale=separate-RN32-multiply-finite-attribute;softmax=expf-rn-sub-div;",
    "256-thread-strided-local-sums-halving-tree;negative-infinity-masked;",
    "NaN-positive-infinity-all-negative-infinity=whole-row-NaN;gradual-underflow;",
    "no-BF16-boundary;NoAlias;no-Torch-bit-parity-claim"
);
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    op: BatchedF32Op,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(op.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "missing batched F32 capability",
        ));
    }
    let contract = op.contract().map_err(contract_error)?;
    let name = format!(
        "provider.cuda.{}",
        op.operation_id().trim_start_matches("operation.")
    );
    let estimator = format!(
        "resource-estimator.cuda.{}",
        op.operation_id().trim_start_matches("operation.")
    );
    let version = if op == BatchedF32Op::Matmul {
        dense::cublas_version()?
    } else {
        0
    };
    let implementation = implementation_fingerprint(&[
        name.as_bytes(),
        NUMERICAL_PROFILE.as_bytes(),
        IDENTITY.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("batched_f32.rs"),
        include_bytes!("batched_f32/launch.rs"),
        format!("cublas.version={version}").as_bytes(),
    ]);
    let estimate = implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(name).map_err(contract_error)?,
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
        super::super::transformer::contiguous_bindings(op.input_count()),
        estimator,
        ContractVersion::new(1, 0),
        estimate,
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
    op: BatchedF32Op,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        op,
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    op: BatchedF32Op,
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
                "batched F32 estimator received another operation",
            ));
        }
        geometry(self.op, r.values(), r.attributes()).map_err(invalid_plan)?;
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
            "cuda.batched_f32.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.batched_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one fixed participant".into());
        };
        if i.operation().id.as_str() != self.op.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires its fixed tensor contract".into());
        }
        let g = geometry(self.op, p.bindings(), p.attributes())?;
        let mut regions = (0..self.op.input_count())
            .map(|n| {
                retained_typed_region(
                    p,
                    binding(p.bindings(), ResolvedValueRole::Input, n)?,
                    g.input_bytes(n as usize),
                    ElementType::F32,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        let output = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            g.output_bytes(),
            ElementType::F32,
        )?;
        for input in &regions {
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let kernels = Arc::clone(&self.kernels);
        let command = if self.op == BatchedF32Op::Matmul {
            CudaDeviceCommand::operation_with_blas(
                "batched_matmul_f32",
                regions,
                move |stream, blas, regions| {
                    let _owner = &kernels;
                    launch::matmul(stream, blas, regions, g)
                },
            )
        } else {
            CudaDeviceCommand::operation(self.op.operation_id(), regions, move |stream, regions| {
                launch::pointwise(stream, &kernels, regions, g)
            })
        };
        let launches = if self.op == BatchedF32Op::Matmul {
            g.batch() * g.group()
        } else {
            1
        };
        command
            .and_then(|c| {
                c.with_tensor_work(
                    g.output_elements(),
                    g.arithmetic_operations().map_err(contract_error)?,
                    launches,
                    0,
                )
            })
            .map_err(|e| e.to_string())
    }
}
fn geometry(
    op: BatchedF32Op,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<BatchedF32Geometry, String> {
    if b.len() != op.input_count() as usize + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact dense ports".into());
    }
    let inputs = (0..op.input_count())
        .map(|n| binding(b, ResolvedValueRole::Input, n).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, String>>()?;
    let g = BatchedF32Geometry::new(
        op,
        &inputs,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())?;
    if op == BatchedF32Op::Matmul && [g.m(), g.n(), g.k()].iter().any(|&d| d > i32::MAX as u64) {
        return Err("cuBLAS matrix axes must fit positive i32".into());
    }
    Ok(g)
}
