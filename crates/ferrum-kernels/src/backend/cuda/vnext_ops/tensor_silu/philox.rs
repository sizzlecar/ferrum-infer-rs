//! Explicit-state eager cuRAND Philox normal generation into admitted storage.
use super::*;
use cudarc::driver::sys::CUdevice_attribute;
use ferrum_runtime::{
    philox_normal_bf16_contract, PhiloxNormalBf16Geometry, PHILOX_NORMAL_BF16_CAPABILITY_ID,
    PHILOX_NORMAL_BF16_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str =
    "cuda.curand.philox_normal.bf16.torch213_geometry.unqualified.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "Torch-v2.13.0-DistributionTemplates-contiguous32bit-geometry;B1;block256;unroll4;",
    "grid=min(ceil(N/256),SMcount*floor(maxThreadsPerSM/256));",
    "curand_init(seed,linear_thread,offset);normal4;output[thread+iteration*4G+kG];",
    "B=integer-RNE;RN32(RN32(normal*1)+0);explicit-U32[seedlo,seedhi,offsetlo,offsethi];",
    "all-u64-offsets-defined;input-state-not-mutated;fresh-offset0-first-image;",
    "ftz=false;prec-div=true;prec-sqrt=true;fmad=false;no-fast-math;",
    "one-thread-local-curand-state;no-device-state-array;zero-global-scratch;",
    "logical-work=one-normal-draw-per-output-not-hardware-FLOPs;",
    "eager-only;Torch-cu130-byte-parity-NOT-VERIFIED;device-and-toolkit-dependent"
);
pub const BUILD_IDENTITY_SHA256: &str = env!("FERRUM_PHILOX_BUILD_IDENTITY_SHA256");
pub(super) const FUNCTION: &str = "ferrum_vnext_philox_normal_bf16";
const PROVIDER_ID: &str = "provider.cuda.philox_normal.bf16";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.philox_normal.bf16";

/// Actual device launch limits, not an authority to allocate or execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaPhiloxDeviceGeometry {
    multiprocessors: u32,
    max_threads_per_multiprocessor: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaPhiloxLaunchGeometry {
    blocks: u32,
    elements: u64,
    rounds: u64,
}
impl CudaPhiloxDeviceGeometry {
    pub fn query(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let attr = |kind| {
            runtime
                .context()
                .attribute(kind)
                .map_err(|e| CudaDeviceRuntimeError::driver("Philox CUDA device attribute", e))
        };
        Self::new(
            attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
        )
    }
    pub(super) fn new(multiprocessors: i32, threads: i32) -> Result<Self, CudaDeviceRuntimeError> {
        if multiprocessors <= 0 || threads < THREADS as i32 {
            return Err(CudaDeviceRuntimeError::contract(
                "Philox requires positive SM count and at least 256 threads per SM",
            ));
        }
        let result = Self {
            multiprocessors: multiprocessors as u32,
            max_threads_per_multiprocessor: threads as u32,
        };
        result
            .multiprocessors
            .checked_mul(result.max_threads_per_multiprocessor / THREADS)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Philox device grid capacity overflows")
            })?;
        Ok(result)
    }
    pub const fn multiprocessors(self) -> u32 {
        self.multiprocessors
    }
    pub const fn max_threads_per_multiprocessor(self) -> u32 {
        self.max_threads_per_multiprocessor
    }
    pub fn launch(self, elements: u64) -> Result<CudaPhiloxLaunchGeometry, CudaDeviceRuntimeError> {
        if elements == 0 || elements > (i32::MAX as u64 / 2) {
            return Err(CudaDeviceRuntimeError::contract(
                "Philox requires positive 32-bit BF16 byte extent",
            ));
        }
        let capacity = u64::from(self.multiprocessors)
            * u64::from(self.max_threads_per_multiprocessor / THREADS);
        let blocks = elements.div_ceil(u64::from(THREADS)).min(capacity) as u32;
        let rounds = elements.div_ceil(u64::from(blocks) * u64::from(THREADS) * 4);
        Ok(CudaPhiloxLaunchGeometry {
            blocks,
            elements,
            rounds,
        })
    }
}
impl CudaPhiloxLaunchGeometry {
    pub const fn blocks(self) -> u32 {
        self.blocks
    }
    pub const fn threads_per_block(self) -> u32 {
        THREADS
    }
    pub const fn element_count(self) -> u64 {
        self.elements
    }
    /// Per-thread generator offset advance, including unused tail draws.
    pub const fn offset_advance(self) -> u64 {
        self.rounds * 4
    }
    pub const fn rounds(self) -> u64 {
        self.rounds
    }
}

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

    #[cfg(test)]
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
    let capability = CapabilityId::new(PHILOX_NORMAL_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks Philox normal capability",
        ));
    }
    let contract = philox_normal_bf16_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        BUILD_IDENTITY_SHA256.as_bytes(),
        &policy.multiprocessors.to_le_bytes(),
        &policy.max_threads_per_multiprocessor.to_le_bytes(),
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
            .launch(geometry.element_count())
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
            "cuda.philox_normal.bf16.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.philox_normal.bf16.encode", e))
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
        if invocation.operation().id.as_str() != PHILOX_NORMAL_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("Philox requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        let launch = self
            .policy
            .launch(shape.element_count())
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
            ElementType::Bf16,
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
            let mut builder = stream.launch_builder(kernels.philox_function());
            builder
                .arg(&state)
                .arg(&output)
                .arg(&launch.elements)
                .arg(&launch.rounds);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (launch.blocks, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("BF16 Philox normal launch", e))
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
) -> Result<PhiloxNormalBf16Geometry, String> {
    if values.len() != 2 || !attrs.is_empty() {
        return Err("Philox requires exactly state/output and no attributes".into());
    }
    PhiloxNormalBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn launch_grid_matches_bounded_four_wide_mapping() {
        let p = CudaPhiloxDeviceGeometry::new(2, 768).unwrap();
        for (n, blocks, advance) in [
            (1, 1, 4),
            (256, 1, 4),
            (257, 2, 4),
            (1536, 6, 4),
            (6144, 6, 4),
            (6145, 6, 8),
        ] {
            let l = p.launch(n).unwrap();
            assert_eq!((l.blocks(), l.offset_advance()), (blocks, advance));
            let stride = u64::from(l.blocks()) * 256;
            let mut visits = vec![0u8; n as usize];
            for thread in 0..stride {
                for round in 0..l.rounds() {
                    for component in 0..4 {
                        let index = thread + round * stride * 4 + component * stride;
                        if index < n {
                            visits[index as usize] += 1;
                        }
                    }
                }
            }
            assert!(visits.iter().all(|&count| count == 1));
        }
    }
    #[test]
    fn invalid_device_and_large_byte_geometry_fail_before_native_work() {
        for (sms, threads) in [(0, 2048), (-1, 2048), (1, 255), (i32::MAX, i32::MAX)] {
            assert!(CudaPhiloxDeviceGeometry::new(sms, threads).is_err());
        }
        let p = CudaPhiloxDeviceGeometry::new(1, 256).unwrap();
        assert!(p.launch(0).is_err());
        assert!(p.launch(1 << 30).is_err());
        assert_eq!(p.launch((1 << 30) - 1).unwrap().offset_advance(), 4194304);
        assert!(nonoverlapping_ranges(0x1000, 16, 0x1008, 32).is_err());
        assert!(nonoverlapping_ranges(0x1000, 16, 0x1010, 32).is_ok());
    }
}
