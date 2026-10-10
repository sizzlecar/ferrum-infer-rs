//! Explicit MarkerV2 SwiGLU provider using the locked native artifact.
use std::sync::{Arc, Mutex};

use super::super::native_blocks::{
    q8act::Q8ActKernels,
    upstream_linear::{validate_parts, UpstreamPlanFactory, WeightValidationRegistry},
    weights, CudaNativeBlockKernels,
};
use super::*;
use ferrum_interfaces::vnext::{
    EncodedReusableExecutionBindings, PreparedProjectionNumerics, UpstreamMarkerV2Profile,
    UpstreamScratchEstimate,
};

mod encode;

pub(in crate::backend::cuda::vnext_ops) struct CudaUpstreamSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    profile: UpstreamMarkerV2Profile,
    native: CudaNativeBlockKernels,
    g32: Option<Q8ActKernels>,
    plans: Arc<UpstreamPlanFactory>,
    validation: Arc<WeightValidationRegistry>,
    arithmetic: ferrum_interfaces::vnext::CompositeNumericalArithmetic,
    silu: CudaFunction,
    scratch: Mutex<BTreeMap<String, UpstreamScratchEstimate>>,
}

impl CudaUpstreamSwiGluProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
        profile: UpstreamMarkerV2Profile,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if !matches!(
            profile,
            UpstreamMarkerV2Profile::SwiGlu
                | UpstreamMarkerV2Profile::SwiGluPrefill
                | UpstreamMarkerV2Profile::SwiGluG32MmqPrefill
                | UpstreamMarkerV2Profile::SwiGluExtraPrefill
                | UpstreamMarkerV2Profile::SwiGluExtraLargePrefill
                | UpstreamMarkerV2Profile::SwiGluExtraAllRows
                | UpstreamMarkerV2Profile::SwiGluQ6F16
        ) {
            return Err(CudaDeviceRuntimeError::contract("not a SwiGLU profile"));
        }
        let contract = profile.contract().map_err(contract_error)?;
        let fingerprint = implementation_fingerprint(&[
            include_bytes!("upstream_swiglu.rs"),
            include_bytes!("upstream_swiglu/encode.rs"),
            include_bytes!("replay_encoding.rs"),
            include_bytes!("../native_blocks/upstream_linear.rs"),
            include_bytes!("../native_blocks/upstream_linear/preparation.rs"),
            include_bytes!("../native_blocks/upstream_linear/native_plan.rs"),
            include_bytes!("../../../../native_ops/upstream_q6_f16_linear.rs"),
            include_bytes!("../../../../native_ops/upstream_q6_f16_linear/ffi.rs"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../ferrum-native-ops/src/upstream_q6_f16_linear.rs"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-q6-f32-linear/abi.h"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-q6-f32-linear/mmq.cu"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-q6-f32-linear/f16_adapter.cuh"
            )),
            include_bytes!("../native_blocks/upstream_linear/weight_validation.rs"),
            include_bytes!("segment_bindings.rs"),
            include_bytes!("../native_blocks/weights.rs"),
            include_bytes!("../native_blocks/q8act.rs"),
            include_bytes!("../../../../native_ops/upstream_linear.rs"),
            include_bytes!("../../../../native_ops/upstream_linear/ffi.rs"),
            include_bytes!("../../../../native_ops/upstream_linear/dispatch.rs"),
            include_bytes!("../../../../native_ops/upstream_linear/extra_ffi.rs"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../ferrum-native-ops/src/upstream_extra_linear.rs"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../ferrum-native-ops/src/upstream_extra_linear/prefill.rs"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-extra-linear/mmq.cu"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-extra-linear/mmvq.cu"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-extra-linear/marker.cuh"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-extra-linear/format.h"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-extra-linear/abi.h"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-extra-linear/boundary.h"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-extra-linear/boundary.cu"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-linear/mmq.cu"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-linear/mmvq.cu"
            )),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../native-operators/cuda/upstream-linear/marker.cuh"
            )),
            crate::ptx::VNEXT_GGUF.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
        ]);
        let descriptor = provider_descriptor_with_formats(
            runtime,
            &contract,
            &profile.provider_id(),
            profile.capability_id(),
            &profile.estimator_id(),
            contiguous_bindings(3),
            BTreeSet::from([
                WeightFormatId::new("weight-format.gguf.native-block").map_err(contract_error)?
            ]),
            native_linear::quantization_formats().map_err(contract_error)?,
            fingerprint.clone(),
        )?;
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("upstream SiLU module", error))?;
        Ok(Self {
            silu: module
                .load_function(SILU_MUL_FUNCTION_NAME)
                .map_err(|error| CudaDeviceRuntimeError::driver("upstream SiLU function", error))?,
            native: CudaNativeBlockKernels::load(runtime.context())?,
            g32: profile
                .hybrid()
                .then(|| Q8ActKernels::load_attention(runtime.context()))
                .transpose()?,
            plans: Arc::new(
                UpstreamPlanFactory::new(runtime.context(), fingerprint)
                    .map_err(CudaDeviceRuntimeError::contract)?,
            ),
            validation: Arc::new(WeightValidationRegistry::default()),
            arithmetic: profile.arithmetic(),
            descriptor,
            profile,
            scratch: Mutex::new(BTreeMap::new()),
        })
    }

    fn scratch_bytes(
        &self,
        prepared: &PreparedProjectionNumerics,
    ) -> Result<UpstreamScratchEstimate, String> {
        let mut cache = self
            .scratch
            .lock()
            .map_err(|_| "upstream scratch cache poisoned")?;
        if let Some(&bytes) = cache.get(prepared.fingerprint()) {
            return Ok(bytes);
        }
        let mut bytes = self.plans.maximum_scratch(prepared)?;
        if self.profile.hybrid() {
            bytes.bytes_per_row =
                bytes
                    .bytes_per_row
                    .max(super::super::native_blocks::q8act::workspace_per_token(
                        prepared,
                    )?);
        }
        cache.insert(prepared.fingerprint().to_owned(), bytes);
        Ok(bytes)
    }
}

impl OperationResourceEstimator for CudaUpstreamSwiGluProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(&self.descriptor, &request, self.profile.operation_id())?;
        let prepared = PreparedProjectionNumerics::prepare(&self.arithmetic, request.values())
            .map_err(invalid_plan)?;
        for projection in prepared.projections() {
            let value = binding(
                request.values(),
                ResolvedValueRole::Input,
                projection.weight_input_ordinal(),
            )
            .map_err(invalid_plan)?;
            let parts = weights::matrix_parts(
                value
                    .weight()
                    .ok_or_else(|| invalid_plan("upstream weight is absent"))?,
                value.tensor().dimensions(),
            )
            .map_err(invalid_plan)?;
            // This checks retained formats/IDs/dimensions and strict transform ABI,
            // without applying G32 arithmetic to any staged leaf.
            validate_parts(self.profile, projection, &parts).map_err(invalid_plan)?;
        }
        let intermediate =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let extra = self.scratch_bytes(&prepared).map_err(invalid_plan)?;
        let transforms =
            super::super::native_blocks::hadamard::workspace_bytes_per_token(request.values())
                .map_err(invalid_plan)?;
        let per_token = intermediate
            .checked_mul(6)
            .and_then(|bytes| bytes.checked_add(transforms))
            .and_then(|bytes| bytes.checked_add(extra.bytes_per_row))
            .ok_or_else(|| invalid_plan("upstream SwiGLU scratch overflows"))?;
        let fixed = extra
            .fixed_bytes
            .checked_add(15)
            .ok_or_else(|| invalid_plan("upstream scratch alignment overflows"))?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(fixed, 0, per_token)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        // MMQ and MMVQ consume different coefficient expressions. Each leaf has
        // two independently owned validation flags; no flag is used as scratch.
        let flags = flag_bytes(&prepared).map_err(invalid_plan)?;
        let persistent = ProviderWorkspaceRequirement::new(
            flags,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Plan,
            ProviderWorkspaceReusePolicy::Preserve,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            Some(scratch),
            Some(persistent),
        )
        .with_projection_numerics(prepared))
    }
}

fn flag_bytes(prepared: &PreparedProjectionNumerics) -> Result<u64, String> {
    prepared
        .projections()
        .iter()
        .try_fold(0_u64, |n, p| {
            n.checked_add(p.leaves().len() as u64)
                .ok_or("leaf count overflows")
        })?
        .checked_mul(8)
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| "validation flag extent overflows".into())
}

impl OperationProvider<CudaDeviceRuntime> for CudaUpstreamSwiGluProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(
            &request,
            3,
            &[
                CapturedProviderWorkspace::Scratch,
                CapturedProviderWorkspace::Persistent,
            ],
        )
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode::encode(
            self,
            invocation,
            replay_encoding::EncodingTarget::Full,
            None,
        )
        .and_then(replay_encoding::Encoding::full)
        .map_err(|message| {
            provider_failure(
                identity,
                "cuda.dense_swiglu.upstream-marker-v2.encode",
                message,
            )
        })
    }
    fn encode_selected_with_segment_declaration(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<
        (
            EncodedDeviceOperation<CudaDeviceCommand>,
            Option<ferrum_interfaces::vnext::SegmentBindingDeclaration>,
        ),
        OperationFailure,
    > {
        let mut declaration = None;
        let identity = invocation.participants()[0].identity().clone();
        encode::encode(
            self,
            invocation,
            replay_encoding::EncodingTarget::Full,
            Some(&mut declaration),
        )
        .and_then(|encoded| encoded.full().map(|operation| (operation, declaration)))
        .map_err(|message| {
            provider_failure(
                identity,
                "cuda.dense_swiglu.upstream-marker-v2.encode",
                message,
            )
        })
    }
    fn encode_reusable_execution_bindings(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedReusableExecutionBindings<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        // FFN has no state binding slot of its own. Core calls this hook for
        // its retained dependency prefix inside an actual resident segment.
        encode::encode(
            self,
            invocation,
            replay_encoding::EncodingTarget::BindingsOnly,
            None,
        )
        .and_then(replay_encoding::Encoding::bindings)
        .map_err(|message| {
            provider_failure(
                identity,
                "cuda.dense_swiglu.upstream-marker-v2.bindings",
                message,
            )
        })
    }
}
