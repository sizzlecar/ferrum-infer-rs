//! F32 affine followed by BF16 conversion, with only one admitted row chunk.
//! Uses the original lane SGEMM, row-bias and cast kernels. No private allocator
//! or second module owner; all three launches and every chunk share one fence.
use super::*;
use ferrum_runtime::{
    affine_linear_f32_to_bf16_contract, DynamicStorageRequirement, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    AFFINE_LINEAR_F32_TO_BF16_CAPABILITY_ID, AFFINE_LINEAR_F32_TO_BF16_OPERATION_ID,
};

// Native implementation tiling, included in its fingerprint and exact estimate.
// It does not bound the logical number of rows or split the reduction axis.
const CHUNK_ROWS: u64 = 4096;
const ESTIMATOR: &str = "resource-estimator.cuda.affine_linear.f32_to_bf16.chunked";

pub(in crate::backend::cuda::vnext_ops) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability =
        CapabilityId::new(AFFINE_LINEAR_F32_TO_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks rounded F32 affine",
        ));
    }
    let contract = affine_linear_f32_to_bf16_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        b"chunked-rows;sgemm-pedantic-f32-dot-store;row-bias-fadd-rn;bf16-rne;no-reduction-split;no-Torch-bitwise-equivalence",
        &CHUNK_ROWS.to_le_bytes(),
        &cublas_version()?.to_le_bytes(),
        source_fingerprint().as_bytes(),
        RetainedF32RowBias::fingerprint().as_bytes(),
        include_bytes!("rounded.rs"),
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/vnext_float_cast.cuh")),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.affine_linear.f32_to_bf16.chunked")
            .map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(3),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}

pub(in crate::backend::cuda::vnext_ops) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: RetainedF32RowBias,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels,
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: RetainedF32RowBias,
}

fn shape(
    bindings: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<AffineLinearF32Geometry, String> {
    if bindings.len() != 4
        || attrs.len() != 2
        || bindings.iter().any(|b| {
            b.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err(
            "rounded F32 affine requires three dense inputs, one output and two shape attributes"
                .into(),
        );
    }
    AffineLinearF32Geometry::rounded_bf16(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        super::super::unsigned_attribute(attrs, "in_features")?,
        super::super::unsigned_attribute(attrs, "out_features")?,
    )
    .map_err(|e| e.to_string())
}
fn scratch_bytes(g: AffineLinearF32Geometry) -> u64 {
    g.rows().min(CHUNK_ROWS) * g.outputs() * 4
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
                "rounded affine estimator received another operation",
            ));
        }
        let g = shape(r.values(), r.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            ALIGNMENT,
            Some(ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::fixed(scratch_bytes(g))?,
                ALIGNMENT,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?),
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
            "cuda.rounded_affine.encode",
            "fixed tensor execution required".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.rounded_affine.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("rounded affine requires one participant".into());
        };
        if i.operation().id.as_str() != AFFINE_LINEAR_F32_TO_BF16_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("rounded affine requires its fixed tensor contract".into());
        }
        let g = shape(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(5);
        for (role, ordinal, bytes, scalar) in [
            (
                ResolvedValueRole::Input,
                0,
                g.rows() * g.inputs() * 4,
                Scalar::F32,
            ),
            (
                ResolvedValueRole::Input,
                1,
                g.outputs() * g.inputs() * 4,
                Scalar::F32,
            ),
            (ResolvedValueRole::Input, 2, g.outputs() * 4, Scalar::F32),
            (
                ResolvedValueRole::Output,
                0,
                g.output_elements() * 2,
                Scalar::Bf16,
            ),
        ] {
            regions.push(retained_region_for(
                p,
                binding(p.bindings(), role, ordinal)?,
                bytes,
                scalar,
            )?);
        }
        for r in &regions[..3] {
            disjoint(
                r.device_ptr(),
                r.length_bytes(),
                regions[3].device_ptr(),
                regions[3].length_bytes(),
            )?;
        }
        let translated = p
            .scratch_view()
            .ok_or("rounded affine lacks admitted scratch")?
            .translate(0, scratch_bytes(g))
            .map_err(|e| e.to_string())?;
        let mut pieces = translated.iter();
        let piece = pieces.next().ok_or("rounded affine scratch is empty")?;
        if pieces.next().is_some() {
            return Err("rounded affine scratch must be contiguous".into());
        }
        let (buffer, range, retention) = piece.buffer_and_physical_range();
        let scratch = buffer
            .retained_region(range, retention)
            .map_err(|e| e.to_string())?;
        if scratch.element_type() != ElementType::U8
            || scratch.length_bytes() != scratch_bytes(g)
            || scratch.device_ptr() % ALIGNMENT != 0
        {
            return Err("rounded affine scratch dtype, extent or alignment differs".into());
        }
        for r in &regions {
            disjoint(
                r.device_ptr(),
                r.length_bytes(),
                scratch.device_ptr(),
                scratch.length_bytes(),
            )?;
        }
        regions.push(scratch);
        self.kernels.validate_regions(&regions)?;
        let kernels = self.kernels.clone();
        let arithmetic = g
            .logical_arithmetic_operations()
            .checked_add(g.output_elements())
            .ok_or("rounded affine logical work overflows")?;
        CudaDeviceCommand::operation_with_blas(
            "vnext_affine_linear_f32_to_bf16_chunked",
            regions,
            move |stream, blas, r| {
                for first in (0..g.rows()).step_by(CHUNK_ROWS as usize) {
                    let rows = (g.rows() - first).min(CHUNK_ROWS);
                    let chunk = Shape::new_for(rows, g.inputs(), g.outputs(), Scalar::F32Affine)
                        .map_err(CudaDeviceRuntimeError::contract)?;
                    // Complete extents were proved above. Only rows are sliced;
                    // the original K axis, weight, bias and numerical policy remain.
                    launch_pointers(
                        stream,
                        blas,
                        r[0].device_ptr() + first * g.inputs() * 4,
                        r[1].device_ptr(),
                        r[4].device_ptr(),
                        chunk,
                        Scalar::F32Affine,
                    )?;
                    kernels.launch(stream, &r[2], &r[4], rows, g.outputs())?;
                    kernels.narrow(
                        stream,
                        r[4].device_ptr(),
                        r[3].device_ptr() + first * g.outputs() * 2,
                        rows * g.outputs(),
                    )?;
                }
                Ok(())
            },
        )
        .and_then(|c| {
            c.with_tensor_work(
                g.output_elements(),
                arithmetic,
                3 * g.rows().div_ceil(CHUNK_ROWS),
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}
