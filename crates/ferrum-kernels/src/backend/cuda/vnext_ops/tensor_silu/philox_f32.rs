//! Explicit-state eager cuRAND Philox normal generation into admitted storage.
use super::*;
use ferrum_runtime::{
    philox_normal_f32_contract, PhiloxNormalF32Geometry, PHILOX_NORMAL_F32_CAPABILITY_ID,
    PHILOX_NORMAL_F32_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str =
    "cuda.curand.philox_normal.f32.native_geometry.unqualified.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "native-CUDA-cuRAND-Philox4x32-10;contiguous32bit-geometry;B1;block256;unroll4;",
    "grid=min(ceil(N/256),SMcount*floor(maxThreadsPerSM/256));",
    "curand_init(seed,linear_thread,offset);normal4;output[thread+iteration*4G+kG];",
    "direct-F32;RN32(RN32(normal*1)+0);no-BF16-intermediate;explicit-U32[seedlo,seedhi,offsetlo,offsethi];",
    "all-u64-offsets-defined;input-state-not-mutated;fresh-offset0-first-image;",
    "ftz=false;prec-div=true;prec-sqrt=true;fmad=false;no-fast-math;",
    "one-thread-local-curand-state;no-device-state-array;zero-global-scratch;",
    "logical-work=one-normal-draw-per-output-not-hardware-FLOPs;",
    "eager-only;not-Torch-CPU-seed-stream;Torch-CUDA-byte-parity-NOT-VERIFIED;device-and-toolkit-dependent"
);
pub const BUILD_IDENTITY_SHA256: &str = env!("FERRUM_PHILOX_BUILD_IDENTITY_SHA256");
pub(super) const FUNCTION: &str = "ferrum_vnext_philox_normal_f32";
const PROVIDER_ID: &str = "provider.cuda.philox_normal.f32";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.philox_normal.f32";

/// Keeps the exact fingerprinted geometry attached to its provider descriptor.
/// Preparation is metadata-only; native module ownership is injected once the
/// complete selection has been checked and the shared bundle has been loaded.
pub(super) struct PreparedProvider {
    descriptor: OperationProviderDescriptor,
    policy: CudaPhiloxDeviceGeometry,
}

impl PreparedProvider {
    pub(super) fn new(
        device: &DeviceDescriptor,
        policy: CudaPhiloxDeviceGeometry,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Ok(Self {
            descriptor: descriptor(device, policy)?,
            policy,
        })
    }

    pub(super) fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    pub(super) fn into_provider(
        self,
        kernels: &Arc<RetainedFunctions>,
    ) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
        Box::new(PhiloxProvider {
            descriptor: self.descriptor,
            policy: self.policy,
            kernels: Arc::clone(kernels),
        })
    }
}
fn descriptor(
    device: &DeviceDescriptor,
    policy: CudaPhiloxDeviceGeometry,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(PHILOX_NORMAL_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks Philox normal capability",
        ));
    }
    let contract = philox_normal_f32_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        BUILD_IDENTITY_SHA256.as_bytes(),
        &policy.multiprocessors().to_le_bytes(),
        &policy.max_threads_per_multiprocessor().to_le_bytes(),
    ]);
    let estimator =
        implementation_fingerprint(&[ESTIMATOR_ID.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::new(),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(1),
        ESTIMATOR_ID,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)
    .map(|d| d.with_fixed_tensor_work(true))
}
struct PhiloxProvider {
    descriptor: OperationProviderDescriptor,
    policy: CudaPhiloxDeviceGeometry,
    kernels: Arc<RetainedFunctions>,
}
impl OperationResourceEstimator for PhiloxProvider {
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
                "Philox estimator received a different operation",
            ));
        }
        let geometry = geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        self.policy
            .launch_f32(geometry.element_count())
            .map_err(|e| invalid_plan(e.to_string()))?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            ALIGNMENT,
            None,
            None,
        ))
    }
}
impl OperationProvider<CudaDeviceRuntime> for PhiloxProvider {
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
            "cuda.philox_normal.f32.encode",
            "Philox requires fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.philox_normal.f32.encode", e))
    }
}
impl PhiloxProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("Philox requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != PHILOX_NORMAL_F32_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("Philox requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        let launch = self
            .policy
            .launch_f32(shape.element_count())
            .map_err(|e| e.to_string())?;
        let state = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            16,
            ElementType::U32,
        )?;
        let output = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.output_bytes(),
            ElementType::F32,
        )?;
        nonoverlapping_ranges(
            state.device_ptr(),
            16,
            output.device_ptr(),
            shape.output_bytes(),
        )?;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(FUNCTION, vec![state, output], move |stream, regions| {
            let state = regions[0].device_ptr();
            let output = regions[1].device_ptr();
            let elements = launch.element_count();
            let rounds = launch.rounds();
            let mut builder = stream.launch_builder(kernels.philox_f32_function());
            builder.arg(&state).arg(&output).arg(&elements).arg(&rounds);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (launch.blocks(), 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 Philox normal launch", e))
        })
        .and_then(|c| {
            c.with_tensor_work(
                shape.element_count(),
                shape.logical_arithmetic_operations(),
                1,
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<PhiloxNormalF32Geometry, String> {
    if values.len() != 2 || !attrs.is_empty() {
        return Err("Philox requires exactly state/output and no attributes".into());
    }
    PhiloxNormalF32Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}
