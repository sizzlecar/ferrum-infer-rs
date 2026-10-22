//! Experimental bounded CUDA tensor construction. Opaque allowances are
//! caller-declared admission budgets, not measured CUDA memory profiles.

use std::sync::atomic::AtomicUsize;

use ferrum_runtime::{DeviceCapacityOwner, DeviceCapacityRetention, DeviceOpaqueMemoryKind};

use super::*;

/// Default bound, including uploads, initialization and compute commands.
/// The actual bound is explicitly configured and enters runtime identity.
pub const CUDA_TENSOR_MAX_COMMANDS_PER_SUBMISSION: usize = 64;

pub struct CudaTensorRuntimeConfig {
    pub device: CudaDeviceRuntimeConfig,
    pub usable_memory_bytes: u64,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    /// All commands, not just compute nodes. Fixed command record counts are
    /// checked across all admitted flights. This is not a bound on captured
    /// closure allocations or total Host RSS, which remain outside this profile.
    pub maximum_commands_per_submission: usize,
    /// Versioned declaration, e.g. `experimental-unqualified-smoke-v1`.
    /// This identity never certifies that its opaque allowances are sufficient.
    pub opaque_memory_profile: String,
}

pub(super) struct TensorEagerState {
    capacity: DeviceCapacityOwner,
    _bootstrap: DeviceCapacityRetention,
    stream_bytes: u64,
    maximum_streams: usize,
    maximum_in_flight: usize,
    maximum_commands: usize,
    streams: AtomicUsize,
    provider_live: AtomicBool,
}

pub(crate) struct CudaTensorProviderRetention {
    owner: Arc<TensorEagerState>,
}

impl Drop for CudaTensorProviderRetention {
    fn drop(&mut self) {
        self.owner.provider_live.store(false, Ordering::Release);
    }
}

pub(super) struct TensorLaneRetention {
    owner: Arc<TensorEagerState>,
    _capacity: DeviceCapacityRetention,
    flights: AtomicUsize,
}

impl Drop for TensorLaneRetention {
    fn drop(&mut self) {
        self.owner.streams.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(super) struct TensorFlightRetention {
    lane: Arc<TensorLaneRetention>,
}

impl Drop for TensorFlightRetention {
    fn drop(&mut self) {
        self.lane.flights.fetch_sub(1, Ordering::AcqRel);
    }
}

fn claim_count(
    counter: &AtomicUsize,
    limit: usize,
    label: &str,
) -> Result<(), CudaDeviceRuntimeError> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            (current < limit).then(|| current + 1)
        })
        .map(|_| ())
        .map_err(|_| {
            CudaDeviceRuntimeError::contract(format!("CUDA tensor {label} capacity exhausted"))
        })
}

impl TensorEagerState {
    pub(super) fn validate_submission(
        &self,
        commands: usize,
        timing: DeviceTimingMode,
        facts: ExecutionFactsCapture,
    ) -> Result<(), CudaDeviceRuntimeError> {
        validate_submission_policy(self.maximum_commands, commands, timing, facts)
    }

    pub(super) fn reserve_lane(
        self: &Arc<Self>,
    ) -> Result<Arc<TensorLaneRetention>, CudaDeviceRuntimeError> {
        claim_count(&self.streams, self.maximum_streams, "stream")?;
        let capacity = match self
            .capacity
            .reserve(DeviceOpaqueMemoryKind::Stream, self.stream_bytes)
        {
            Ok(capacity) => capacity,
            Err(error) => {
                self.streams.fetch_sub(1, Ordering::AcqRel);
                return Err(CudaDeviceRuntimeError::capacity(error));
            }
        };
        Ok(Arc::new(TensorLaneRetention {
            owner: Arc::clone(self),
            _capacity: capacity,
            flights: AtomicUsize::new(0),
        }))
    }
}

/// Bounds fixed command records across the configured stream/flight limits.
/// Heap data retained by individual records is additional and is not claimed
/// to be charged to a HostMemoryAccount by this experimental runtime.
fn validate_command_capacity(
    streams: usize,
    flights: usize,
    commands: usize,
) -> Result<usize, CudaDeviceRuntimeError> {
    if streams == 0 || flights == 0 || commands == 0 || commands > u32::MAX as usize {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA tensor stream/flight limits must be positive and command limit must be in 1..=u32::MAX",
        ));
    }
    streams
        .checked_mul(flights)
        .and_then(|count| count.checked_mul(commands))
        .and_then(|count| {
            count.checked_mul(std::mem::size_of::<DeviceCommandEntry<CudaDeviceCommand>>())
        })
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "CUDA tensor fixed command record capacity exceeds addressable memory",
            )
        })
}

fn validate_submission_policy(
    maximum_commands: usize,
    commands: usize,
    timing: DeviceTimingMode,
    facts: ExecutionFactsCapture,
) -> Result<(), CudaDeviceRuntimeError> {
    if commands == 0
        || commands > maximum_commands
        || timing != DeviceTimingMode::Off
        || facts != ExecutionFactsCapture::Off
    {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "CUDA tensor eager profile requires timing/facts off and 1..={maximum_commands} total commands"
        )));
    }
    Ok(())
}

impl TensorLaneRetention {
    pub(super) fn reserve_flight(
        self: &Arc<Self>,
    ) -> Result<Arc<TensorFlightRetention>, CudaDeviceRuntimeError> {
        claim_count(&self.flights, self.owner.maximum_in_flight, "in-flight")?;
        Ok(Arc::new(TensorFlightRetention {
            lane: Arc::clone(self),
        }))
    }
}

/// Native constructors may partially allocate before returning an error. Until
/// their cleanup can be proven, retain the quota rather than fabricate recovery.
/// The retained quota is deliberately fail-closed for this process lifetime.
pub(super) fn create_native<T, O>(
    owner: Arc<O>,
    create: impl FnOnce() -> Result<T, CudaDeviceRuntimeError>,
) -> Result<(T, Arc<O>), CudaDeviceRuntimeError> {
    match catch_unwind(AssertUnwindSafe(create)) {
        Ok(Ok(native)) => Ok((native, owner)),
        Ok(Err(error)) => {
            std::mem::forget(owner);
            Err(error)
        }
        Err(payload) => {
            std::mem::forget(owner);
            resume_unwind(payload)
        }
    }
}

fn driver_status(
    status: cudarc::driver::sys::CUresult,
    operation: &'static str,
) -> Result<(), CudaDeviceRuntimeError> {
    if status == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaDeviceRuntimeError::driver(
            operation,
            DriverError(status),
        ))
    }
}

impl CudaDeviceRuntime {
    pub fn new_tensor_eager(
        mut config: CudaTensorRuntimeConfig,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if config.device.device_id.as_str() != format!("device.cuda.{}", config.device.ordinal) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor runtime requires the canonical physical device identity",
            ));
        }
        if config.maximum_streams == 0
            || config.maximum_in_flight_per_stream == 0
            || config.bootstrap_memory_bytes == 0
            || config.stream_memory_bytes == 0
            || config.opaque_memory_profile.is_empty()
            || config.opaque_memory_profile.len() > 256
            || !config.opaque_memory_profile.is_ascii()
            || config
                .opaque_memory_profile
                .bytes()
                .any(|byte| byte.is_ascii_control())
        {
            return Err(CudaDeviceRuntimeError::contract("CUDA tensor budgets require finite positive limits and a versioned profile identity"));
        }
        if config.device.capabilities.is_empty()
            || !config.device.capabilities.iter().all(|capability| {
                matches!(
                    capability.as_str(),
                    ferrum_runtime::SILU_BF16_CAPABILITY_ID
                        | ferrum_runtime::DENSE_LINEAR_BF16_CAPABILITY_ID
                        | ferrum_runtime::SILU_MUL_BF16_CAPABILITY_ID
                        | ferrum_runtime::TANH_BF16_CAPABILITY_ID
                        | ferrum_runtime::MODULATION_SCALE_BF16_CAPABILITY_ID
                        | ferrum_runtime::GATED_RESIDUAL_BF16_CAPABILITY_ID
                        | ferrum_runtime::LAYER_NORM_BF16_CAPABILITY_ID
                        | ferrum_runtime::HEAD_RMS_NORM_BF16_CAPABILITY_ID
                        | ferrum_runtime::ADJACENT_ROPE_BF16_CAPABILITY_ID
                        | ferrum_runtime::STRUCTURED_ATTENTION_BF16_CAPABILITY_ID
                        | ferrum_runtime::ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID
                        | ferrum_runtime::GELU_TANH_BF16_CAPABILITY_ID
                        | ferrum_runtime::ROW_GATHER_BF16_CAPABILITY_ID
                        | ferrum_runtime::BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID
                        | ferrum_runtime::PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID
                        | ferrum_runtime::HALF_SPLIT_ROPE_BF16_CAPABILITY_ID
                        | ferrum_runtime::STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID
                        | ferrum_runtime::CONV2D_BF16_CAPABILITY_ID
                        | ferrum_runtime::CHANNEL_L2_NORM_BF16_CAPABILITY_ID
                        | ferrum_runtime::CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID
                        | ferrum_runtime::SPATIAL_ATTENTION_BF16_CAPABILITY_ID
                        | ferrum_runtime::SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID
                        | ferrum_runtime::SILU_NCHW_BF16_CAPABILITY_ID
                        | ferrum_runtime::PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID
                )
            })
        {
            return Err(CudaDeviceRuntimeError::contract(
                "this CUDA tensor eager profile admits only its fixed BF16 pointwise, normalization, rotary and Dense providers",
            ));
        }
        let streams = u64::try_from(config.maximum_streams)
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA stream limit exceeds u64"))?;
        let overhead = config
            .stream_memory_bytes
            .checked_mul(streams)
            .and_then(|bytes| bytes.checked_add(config.bootstrap_memory_bytes))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("CUDA tensor opaque budget overflows")
            })?;
        if overhead >= config.usable_memory_bytes {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor opaque budget leaves no usable storage",
            ));
        }
        validate_command_capacity(
            config.maximum_streams,
            config.maximum_in_flight_per_stream,
            config.maximum_commands_per_submission,
        )?;
        // Driver device queries do not construct a CUDA context. Charge its
        // opaque allowance before the existing constructor can create one.
        let ordinal = i32::try_from(config.device.ordinal)
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA ordinal exceeds i32"))?;
        let mut device = 0;
        let mut total = 0_usize;
        unsafe {
            driver_status(cudarc::driver::sys::cuInit(0), "driver initialization")?;
            driver_status(
                cudarc::driver::sys::cuDeviceGet(&mut device, ordinal),
                "device query",
            )?;
            driver_status(
                cudarc::driver::sys::cuDeviceTotalMem_v2(&mut total, device),
                "device memory query",
            )?;
        }
        // Numeric policy and resource policy jointly identify this runtime.
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        for part in [
            "ferrum-cuda-tensor-eager-v1".to_owned(),
            config.device.runtime_implementation_fingerprint.clone(),
            config.opaque_memory_profile.clone(),
            config.usable_memory_bytes.to_string(),
            config.bootstrap_memory_bytes.to_string(),
            config.stream_memory_bytes.to_string(),
            config.maximum_streams.to_string(),
            config.maximum_in_flight_per_stream.to_string(),
            config.maximum_commands_per_submission.to_string(),
        ] {
            hash.update((part.len() as u64).to_le_bytes());
            hash.update(part.as_bytes());
        }
        config.device.runtime_implementation_fingerprint = format!("{:x}", hash.finalize());
        let descriptor = DeviceDescriptor {
            id: config.device.device_id.clone(),
            class: DeviceClass::Accelerator,
            ordinal: u32::try_from(config.device.ordinal)
                .map_err(|_| CudaDeviceRuntimeError::contract("CUDA ordinal exceeds u32"))?,
            total_memory_bytes: u64::try_from(total)
                .map_err(|_| CudaDeviceRuntimeError::contract("CUDA memory exceeds u64"))?,
            runtime_implementation_fingerprint: config
                .device
                .runtime_implementation_fingerprint
                .clone(),
            capabilities: config.device.capabilities.clone(),
            dynamic_storage_profiles: config.device.dynamic_storage_profiles.clone(),
        };
        let capacity = DeviceCapacityOwner::open(&descriptor, config.usable_memory_bytes)
            .map_err(CudaDeviceRuntimeError::capacity)?;
        let bootstrap = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.bootstrap_memory_bytes,
            )
            .map_err(CudaDeviceRuntimeError::capacity)?;
        let state = Arc::new(TensorEagerState {
            capacity,
            _bootstrap: bootstrap,
            stream_bytes: config.stream_memory_bytes,
            maximum_streams: config.maximum_streams,
            maximum_in_flight: config.maximum_in_flight_per_stream,
            maximum_commands: config.maximum_commands_per_submission,
            streams: AtomicUsize::new(0),
            provider_live: AtomicBool::new(false),
        });
        let (mut runtime, state) = create_native(state, || Self::new(config.device))?;
        if runtime.descriptor() != &descriptor {
            std::mem::forget(state);
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor device descriptor changed during context creation",
            ));
        }
        runtime.tensor_eager = Some(state);
        Ok(runtime)
    }

    /// Only one provider module may occupy this profile's bootstrap allowance.
    /// Place the returned owner after every native function/module field.
    pub(crate) fn reserve_tensor_provider(
        &self,
    ) -> Result<CudaTensorProviderRetention, CudaDeviceRuntimeError> {
        let owner = self.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "CUDA tensor provider requires a bounded tensor runtime",
            )
        })?;
        owner
            .provider_live
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                CudaDeviceRuntimeError::contract("CUDA tensor provider module capacity exhausted")
            })?;
        Ok(CudaTensorProviderRetention {
            owner: Arc::clone(owner),
        })
    }

    /// Retains existing bootstrap capacity for providers that reuse a lane's
    /// native handles and create no additional modules, handles or contexts.
    pub(crate) fn tensor_bootstrap_retention(
        &self,
    ) -> Result<DeviceCapacityRetention, CudaDeviceRuntimeError> {
        self.tensor_eager
            .as_ref()
            .map(|owner| owner._bootstrap.clone())
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA tensor provider requires a bounded tensor runtime",
                )
            })
    }

    /// Charges each additional native-library owner independently. Merely
    /// cloning the bootstrap retention does not authorize another handle.
    pub(crate) fn reserve_tensor_opaque(
        &self,
        bytes: u64,
    ) -> Result<DeviceCapacityRetention, CudaDeviceRuntimeError> {
        if bytes == 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "native owner requires a positive opaque allowance",
            ));
        }
        let owner = self.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract("native owner requires bounded tensor runtime")
        })?;
        owner
            .capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, bytes)
            .map_err(CudaDeviceRuntimeError::capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{DynamicStorageAllocator, DynamicStorageView};

    #[test]
    fn command_capacity_checks_fixed_records_across_every_stream_and_flight() {
        let record = std::mem::size_of::<DeviceCommandEntry<CudaDeviceCommand>>();
        assert_eq!(
            validate_command_capacity(2, 3, 1024).unwrap(),
            2 * 3 * 1024 * record
        );
        for (streams, flights, commands) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (usize::MAX, 2, 1),
            (1, 1, usize::MAX),
        ] {
            assert!(validate_command_capacity(streams, flights, commands).is_err());
        }
    }

    #[test]
    fn total_command_limit_applies_before_enqueue_and_preserves_timing_restrictions() {
        let check = |maximum, count, timing, facts| {
            validate_submission_policy(maximum, count, timing, facts)
        };
        assert!(check(64, 64, DeviceTimingMode::Off, ExecutionFactsCapture::Off).is_ok());
        assert!(check(64, 65, DeviceTimingMode::Off, ExecutionFactsCapture::Off).is_err());
        assert!(check(
            1024,
            1024,
            DeviceTimingMode::Off,
            ExecutionFactsCapture::Off
        )
        .is_ok());
        assert!(check(
            1024,
            1025,
            DeviceTimingMode::Off,
            ExecutionFactsCapture::Off
        )
        .is_err());
        assert!(check(1024, 0, DeviceTimingMode::Off, ExecutionFactsCapture::Off).is_err());
        assert!(check(
            1024,
            1,
            DeviceTimingMode::Completion,
            ExecutionFactsCapture::Off
        )
        .is_err());
        assert!(check(1024, 1, DeviceTimingMode::Off, ExecutionFactsCapture::On).is_err());
    }

    #[test]
    fn final_flight_owner_holds_lane_limit_and_opaque_capacity() {
        let identity = NEXT_RUNTIME_INSTANCE.fetch_add(1, Ordering::Relaxed);
        let descriptor = DeviceDescriptor {
            id: DeviceId::new(format!("device.test.cuda-tensor-retention-{identity}")).unwrap(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: 128,
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        };
        let capacity = DeviceCapacityOwner::open(&descriptor, 128).unwrap();
        let state = Arc::new(TensorEagerState {
            _bootstrap: capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 16)
                .unwrap(),
            capacity: capacity.clone(),
            stream_bytes: 32,
            maximum_streams: 1,
            maximum_in_flight: 1,
            maximum_commands: CUDA_TENSOR_MAX_COMMANDS_PER_SUBMISSION,
            streams: AtomicUsize::new(0),
            provider_live: AtomicBool::new(false),
        });
        let lane = state.reserve_lane().unwrap();
        let flight = lane.reserve_flight().unwrap();
        assert!(lane.reserve_flight().is_err());
        assert!(state.reserve_lane().is_err());
        assert!(capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 96)
            .is_err());
        drop(lane);
        assert!(state.reserve_lane().is_err());
        drop(flight);
        assert_eq!(state.streams.load(Ordering::Acquire), 0);
        let full_remainder = capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 112)
            .unwrap();
        assert_eq!(full_remainder.bytes(), 112);
        drop(full_remainder);
        assert!(state.reserve_lane().is_ok());
    }
}
