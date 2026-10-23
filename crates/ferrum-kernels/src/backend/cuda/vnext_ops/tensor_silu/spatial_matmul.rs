//! Single-head full-spatial attention using bounded query tiles and lane BLAS.
//! Every query still sees every key. No image tiling or head decomposition.
use super::*;
use ferrum_runtime::{
    spatial_attention_bf16_contract, DeviceCapacityRetention, DynamicStorageRequirement,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, SpatialAttentionBf16Geometry,
    SPATIAL_ATTENTION_BF16_CAPABILITY_ID, SPATIAL_ATTENTION_BF16_OPERATION_ID,
};

mod launch;
mod layout;
#[cfg(test)]
mod tests;
use layout::{geometry, Shape, SCRATCH_ALIGNMENT};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cublas.spatial_attention.bf16_f32.query256.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "single-head-full-spatial;QKV[N,3C,H,W];Y[N,C,H,W];C=1..2048;",
    "query-tile=min(S,256);all-S-keys-visible;batch-sequential-workspace-reuse;",
    "QK=BF16-GEMM-F32-output-COMPUTE_32F-DISALLOW_REDUCED_PRECISION_REDUCTION;alpha=1;",
    "scale=separate-RN32-multiply;softmax=F32-row-max-expf-sum-div;",
    "256-thread-strided-key-local-reductions-halving-tree;no-BF16-probabilities;",
    "V=exact-BF16-to-F32;PV=F32-GEMM-COMPUTE_32F_PEDANTIC-no-TF32;",
    "atomics=NOT_ALLOWED;workspace=null-0;host-alpha1-beta0;",
    "Y=final-BF16-RNE;nonfinite-scaled-score=whole-query-NaN;",
    "V-nonfinite=component-wise-F32-GEMM-arithmetic;NaN-class-only;",
    "vendor-reduction-order-and-zero-sign-no-old-profile-bitwise-claim;",
    "one-admitted-U8-workspace=aligned-F32-V+F32-S-by-tile+F32-tile-by-C;",
    "NoAlias;no-unmanaged-payload-or-extra-handle;no-implicit-workspace-retry"
);
const PROVIDER: &str = "provider.cuda.spatial_attention.bf16.cublas.query256";
const ESTIMATOR: &str = "resource-estimator.cuda.spatial_attention.bf16.cublas.query256";

pub(super) fn provider(
    runtime: &CudaDeviceRuntime,
    kernels: &Arc<RetainedFunctions>,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let capability =
        CapabilityId::new(SPATIAL_ATTENTION_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !runtime.descriptor().capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks spatial attention capability",
        ));
    }
    let contract = spatial_attention_bf16_contract().map_err(contract_error)?;
    let version = super::super::tensor_dense_bf16::cublas_version()?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        format!("cublas.version={version}").as_bytes(),
        include_bytes!("spatial_matmul.rs"),
        include_bytes!("spatial_matmul/layout.rs"),
        include_bytes!("spatial_matmul/launch.rs"),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(PROVIDER).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        runtime.descriptor().id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::new(),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(1),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        bootstrap: runtime.tensor_bootstrap_retention()?,
    }))
}

struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    bootstrap: DeviceCapacityRetention,
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
            return Err(invalid_plan(
                "spatial GEMM estimator received another operation",
            ));
        }
        let shape = geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(shape.scratch_bytes)?,
            SCRATCH_ALIGNMENT,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            SCRATCH_ALIGNMENT,
            Some(scratch),
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
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "cuda.spatial_gemm.encode",
            "spatial GEMM requires fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.spatial_gemm.encode", e))
    }
}

impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("spatial GEMM requires one participant".into());
        };
        if invocation.operation().id.as_str() != SPATIAL_ATTENTION_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("spatial GEMM requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        let input = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            shape.input_bytes,
        )?;
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.output_bytes,
        )?;
        let view = participant
            .scratch_view()
            .ok_or("spatial GEMM has no admitted scratch")?;
        let translated = view
            .translate(0, shape.scratch_bytes)
            .map_err(|e| e.to_string())?;
        let mut pieces = translated.iter();
        let piece = pieces.next().ok_or("spatial GEMM scratch is empty")?;
        if pieces.next().is_some() {
            return Err("spatial GEMM scratch must be contiguous".into());
        }
        let (buffer, range, retention) = piece.buffer_and_physical_range();
        let scratch = buffer
            .retained_region(range, retention)
            .map_err(|e| e.to_string())?;
        if scratch.element_type() != ElementType::U8
            || scratch.length_bytes() != shape.scratch_bytes
            || scratch.device_ptr() % SCRATCH_ALIGNMENT != 0
        {
            return Err("spatial GEMM scratch dtype/extent/alignment mismatch".into());
        }
        let regions = vec![input, output, scratch];
        for (i, left) in regions.iter().enumerate() {
            for right in &regions[i + 1..] {
                nonoverlapping_ranges(
                    left.device_ptr(),
                    left.length_bytes(),
                    right.device_ptr(),
                    right.length_bytes(),
                )?;
            }
        }
        let kernels = Arc::clone(&self.kernels);
        let bootstrap = self.bootstrap.clone();
        CudaDeviceCommand::operation_with_blas(
            "vnext_spatial_attention_bf16_cublas_query256",
            regions,
            move |stream, blas, regions| {
                let _bootstrap = &bootstrap;
                launch::enqueue(stream, blas, &kernels, regions, shape)
            },
        )
        .and_then(|command| {
            command.with_tensor_work(
                shape.output_bytes / 2,
                shape.operations,
                shape.launches(),
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}
