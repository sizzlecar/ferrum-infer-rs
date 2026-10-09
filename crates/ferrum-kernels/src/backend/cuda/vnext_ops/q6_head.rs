//! Independent last-token Q6/F32 D4 provider. No gather or model-name routing.
use super::native_blocks::{upstream_linear::WeightValidationRegistry, weights};
use super::*;
use crate::native_ops::upstream_linear::{Device, DeviceSpan};
use crate::native_ops::upstream_q6_f32_linear::PreparedQ6F32Linear;
use ferrum_interfaces::vnext::{
    Q6MmqF32Policy, Q6MmqF32Route, LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID as CAPABILITY,
    LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID as OPERATION,
};
use std::collections::BTreeMap;
use std::sync::Mutex;

mod encode;

pub struct CudaQ6MmqF32LastTokenProvider {
    descriptor: OperationProviderDescriptor,
    strict: native_blocks::CudaNativeBlockKernels,
    device: Device,
    context: Arc<cudarc::driver::CudaContext>,
    plans: Mutex<BTreeMap<(u32, u32, u32), Arc<PreparedQ6F32Linear>>>,
    validation: WeightValidationRegistry,
    policy: Q6MmqF32Policy,
}
impl CudaQ6MmqF32LastTokenProvider {
    pub fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        use cudarc::driver::sys::CUdevice_attribute::*;
        let attr = |key| {
            runtime
                .context()
                .attribute(key)
                .map_err(|e| CudaDeviceRuntimeError::driver("Q6 device attribute", e))
        };
        let device = Device {
            architecture: (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? as u32) * 100
                + (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? as u32) * 10,
            multiprocessors: u32::try_from(attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?,
            maximum_dynamic_shared_bytes: u64::try_from(attr(
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?)
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?,
        };
        let contract = ferrum_interfaces::vnext::last_token_dense_linear_q6_mmq_f32_contract()
            .map_err(contract_error)?;
        let descriptor = native_io::descriptor_with_fingerprint(
            runtime,
            &contract,
            "provider.cuda.last_token_dense_linear.f32.q6-d4-mmq-v1",
            CAPABILITY,
            "resource-estimator.cuda.last_token_dense_linear.f32.q6-d4-mmq-v1",
            fingerprint(),
        )?;
        Ok(Self {
            descriptor,
            strict: native_blocks::CudaNativeBlockKernels::load(runtime.context())?,
            device,
            context: runtime.context().clone(),
            plans: Mutex::new(BTreeMap::new()),
            validation: WeightValidationRegistry::default(),
            policy: Q6MmqF32Policy::new(),
        })
    }
    fn plan(
        &self,
        rows: u32,
        part: &weights::MatrixPart,
    ) -> Result<Arc<PreparedQ6F32Linear>, String> {
        let mut cache = self.plans.lock().map_err(|_| "Q6 plan cache poisoned")?;
        let key = (rows, part.columns, part.rows);
        if let Some(plan) = cache.get(&key) {
            return Ok(plan.clone());
        }
        // This is host preparation, outside the captured command. An eligible
        // route must fail here if its independently locked artifact is absent.
        self.context.bind_to_thread().map_err(|e| e.to_string())?;
        let plan = Arc::new(
            PreparedQ6F32Linear::new(rows, part.columns, part.rows, self.device)
                .map_err(|e| e.to_string())?,
        );
        cache.insert(key, plan.clone());
        Ok(plan)
    }
    fn maximum_scratch(&self, parts: &[weights::MatrixPart]) -> Result<u64, String> {
        let mut bytes = 16;
        for part in parts.iter().filter(|part| eligible(part)) {
            for rows in 1..=32 {
                bytes = bytes.max(self.plan(rows, part)?.workspace_bytes());
            }
        }
        Ok(bytes)
    }
}
fn eligible(part: &weights::MatrixPart) -> bool {
    part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q6K)
        && part.columns > 0
        && part.columns % 256 == 0
        && part.rows > 0
        && part.transform.is_none()
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn scratch_workspace(
    fixed_bytes: u64,
    transform_bytes_per_token: u64,
) -> Result<ProviderWorkspaceRequirement, VNextError> {
    // Plain Q6 has no token-dependent transform storage. An affine formula
    // with both variable coefficients zero is deliberately non-canonical.
    let formula = if transform_bytes_per_token == 0 {
        ProviderWorkspaceSizeFormula::fixed(fixed_bytes)?
    } else {
        ProviderWorkspaceSizeFormula::affine(fixed_bytes, 0, transform_bytes_per_token)?
    };
    ProviderWorkspaceRequirement::from_formula(
        formula,
        16,
        ProviderWorkspaceScope::Invocation,
        ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
        DynamicStorageRequirement::contiguous(),
    )
}
impl OperationResourceEstimator for CudaQ6MmqF32LastTokenProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        transformer::ensure_estimator_request(&self.descriptor, &request, OPERATION)?;
        let hidden = unsigned_attribute(request.attributes(), "hidden_size").map_err(invalid)?;
        let outputs = unsigned_attribute(request.attributes(), "out_features").map_err(invalid)?;
        let value = binding(request.values(), ResolvedValueRole::Input, 1).map_err(invalid)?;
        let parts = weights::matrix_parts(
            value
                .weight()
                .ok_or_else(|| invalid("Q6 head weight absent"))?,
            &[outputs, hidden],
        )
        .map_err(invalid)?;
        let bytes = self.maximum_scratch(&parts).map_err(invalid)?;
        let transforms = native_blocks::hadamard::workspace_bytes_per_token(request.values())
            .map_err(invalid)?;
        let scratch = scratch_workspace(bytes, transforms)?;
        let flags = (parts.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("Q6 flags overflow"))?
            .max(4);
        let persistent = ProviderWorkspaceRequirement::new(
            flags,
            16,
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
            16,
            Some(scratch),
            Some(persistent),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{ResourceWorkShape, TokenSpanWork};

    fn work() -> ResourceWorkShape {
        ResourceWorkShape::from_token_spans(vec![
            TokenSpanWork::from_token_ids_with_fit(&[1, 2, 3], 0..3, 9).unwrap(),
            TokenSpanWork::from_token_ids(&[4, 5], 1..2).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn plain_q6_scratch_is_fixed_and_admissible() {
        let workspace = scratch_workspace(1024, 0).unwrap();
        assert_eq!(workspace.fixed_bytes(), Some(1024));
        assert_eq!(workspace.evaluate_bytes(&work()).unwrap(), 1024);
        assert_eq!(workspace.evaluate_fit_bytes(&work()).unwrap(), 1024);
        assert_eq!(workspace.scope(), ProviderWorkspaceScope::Invocation);
        assert_eq!(
            workspace.reuse_policy(),
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead
        );
        assert!(scratch_workspace(0, 0).is_err());
        assert!(scratch_workspace(u64::MAX, 0).is_err());
    }

    #[test]
    fn transformed_q6_scratch_preserves_token_and_fit_capacity() {
        let workspace = scratch_workspace(1024, 64).unwrap();
        assert_eq!(workspace.evaluate_bytes(&work()).unwrap(), 1280);
        assert_eq!(workspace.evaluate_fit_bytes(&work()).unwrap(), 1728);
        assert!(scratch_workspace(u64::MAX - 15, 16).is_err());
    }
}
impl OperationProvider<CudaDeviceRuntime> for CudaQ6MmqF32LastTokenProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        if request.scratch_reusable_address_scope()?.is_none()
            || request
                .persistent_workspace_reusable_address_scope()?
                .is_none()
        {
            return Ok(ReusableExecutionTopology::EagerBoundary);
        }
        reusable_token_topology(&request, b"ferrum.cuda.q6-f32-head.topology.v1\0")
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode::encode(self, invocation)
            .map_err(|e| provider_failure(identity, "cuda.q6_f32_head.encode", e))
    }
    fn encode_reusable_execution_bindings(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<
        ferrum_interfaces::vnext::EncodedReusableExecutionBindings<CudaDeviceCommand>,
        OperationFailure,
    > {
        self.encode_selected(invocation)
            .map(ferrum_interfaces::vnext::EncodedReusableExecutionBindings::from_operation)
    }
}
pub(super) fn fingerprint() -> String {
    implementation_fingerprint(&[
        include_bytes!("q6_head.rs"),
        include_bytes!("q6_head/encode.rs"),
        include_bytes!("native_io.rs"),
        include_bytes!("../vnext_ops.rs"),
        include_bytes!("native_blocks.rs"),
        include_bytes!("native_blocks/weights.rs"),
        include_bytes!("native_blocks/hadamard.rs"),
        include_bytes!("native_blocks/upstream_linear/weight_validation.rs"),
        include_bytes!("../../../native_ops/upstream_linear.rs"),
        include_bytes!("../../../native_ops/upstream_q6_f32_linear.rs"),
        include_bytes!("../../../native_ops/upstream_q6_f32_linear/ffi.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ferrum-native-ops/src/upstream_q6_f32_linear.rs"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ferrum-interfaces/src/vnext/numerical/q6_mmq_f32.rs"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/mmq.cu"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/marker.cuh"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/abi.h"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/boundary.h"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/boundary.cu"
        )),
        crate::ptx::VNEXT_GGUF.as_bytes(),
    ])
}
