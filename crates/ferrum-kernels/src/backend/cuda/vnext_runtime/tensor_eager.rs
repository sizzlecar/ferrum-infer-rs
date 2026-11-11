//! Experimental bounded CUDA tensor construction. Opaque allowances are
//! caller-declared admission budgets, not measured CUDA memory profiles.

use std::sync::atomic::AtomicUsize;

pub use crate::backend::cuda_tensor_causal_policy::{
    CudaDenseCausalProfile, CudaDenseF16CausalProfile, CudaTensorExecutionProfile,
};
use crate::backend::cuda_tensor_memory_policy::{AutoMemoryPolicy, RetainedNative};
pub use crate::backend::cuda_tensor_memory_policy::{
    CudaMemoryUtilization, CudaTensorMemoryObservation, CUDA_TENSOR_AUTO_MEMORY_POLICY_VERSION,
};
use ferrum_runtime::{DeviceCapacityOwner, DeviceCapacityRetention, DeviceOpaqueMemoryKind};

use super::*;
mod causal_commands;
mod convrot_fused_owner;
mod cutlass_owner;
mod fused_f32_owner;
pub(crate) use convrot_fused_owner::CudaTensorConvRotFusedRetention;
pub(crate) use cutlass_owner::CudaTensorCutlassRetention;
pub(crate) use fused_f32_owner::CudaTensorFusedF32Retention;

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

/// Automatically bounds the same process-wide account used by all tensor
/// plans. There is no second explicit usable-memory budget in this input.
/// The optional maximum only tightens the automatic ceiling.
pub struct CudaTensorRuntimeAutoConfig {
    pub device: CudaDeviceRuntimeConfig,
    pub utilization: CudaMemoryUtilization,
    pub maximum_memory_bytes: Option<u64>,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
}

/// Shares this exact runtime/context/account with the closed dense-F16 causal
/// set. Stream count remains explicit; a retained tensor actor, causal lane and
/// static initialization can require three simultaneously live streams.
pub struct CudaTensorAndCausalRuntimeConfig {
    pub tensor: CudaTensorRuntimeConfig,
    pub causal: CudaDenseF16CausalProfile,
}
pub struct CudaTensorAndCausalRuntimeAutoConfig {
    pub tensor: CudaTensorRuntimeAutoConfig,
    pub causal: CudaDenseF16CausalProfile,
}

pub(super) struct TensorEagerState {
    // Drop the private cache before its capacity/bootstrap owners. Each tensor
    // buffer retains this state until after its queued allocation-stream free.
    pub(super) allocation_pool: Option<allocation::TensorAllocationPool>,
    // Bootstrap may retain the provisional owner. Drop its claim before the
    // final lower ceiling, which must govern all native owners until cleanup.
    _bootstrap: DeviceCapacityRetention,
    capacity: DeviceCapacityOwner,
    memory_observation: Option<CudaTensorMemoryObservation>,
    stream_bytes: u64,
    maximum_streams: usize,
    maximum_in_flight: usize,
    maximum_commands: usize,
    streams: AtomicUsize,
    provider_live: AtomicBool,
    cutlass_module: Mutex<Option<cutlass_owner::ContextModuleQuota>>,
    cutlass_provider_live: AtomicBool,
    fused_f32_module: Mutex<Option<fused_f32_owner::FusedF32ModuleQuota>>,
    fused_f32_provider_live: AtomicBool,
    convrot_fused_module: Mutex<Option<convrot_fused_owner::ConvRotFusedModuleQuota>>,
    convrot_fused_provider_live: AtomicBool,
    causal_profile: Option<CudaDenseF16CausalProfile>,
    causal_modules: Mutex<std::sync::Weak<CudaCausalProviderRetention>>,
}

/// Native functions must be destroyed before this independently charged owner.
pub(crate) struct CudaCausalProviderRetention {
    _capacity: DeviceCapacityRetention,
    _owner: Arc<TensorEagerState>,
}

pub(crate) struct CudaTensorProviderRetention {
    owner: Arc<TensorEagerState>,
}

impl Drop for CudaTensorProviderRetention {
    fn drop(&mut self) {
        self.owner.provider_live.store(false, Ordering::Release);
    }
}

pub(crate) struct TensorLaneRetention {
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
    pub(super) fn decode_replay(
        &self,
    ) -> Option<crate::backend::cuda_tensor_causal_policy::CudaCausalDecodeReplay> {
        self.causal_profile
            .as_ref()
            .and_then(CudaDenseF16CausalProfile::decode_replay)
    }

    pub(super) fn execution_profile(&self) -> CudaTensorExecutionProfile {
        self.causal_profile.as_ref().map_or(
            CudaTensorExecutionProfile::TensorOnly,
            CudaDenseF16CausalProfile::execution_profile,
        )
    }

    pub(super) fn causal_retention(&self) -> Option<Arc<CudaCausalProviderRetention>> {
        self.causal_modules
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .upgrade()
    }
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
        let stream_bytes = self
            .stream_bytes
            .checked_add(
                self.decode_replay()
                    .map_or(0, |p| p.memory_bytes_per_stream()),
            )
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("causal graph stream allowance overflows")
            })?;
        claim_count(&self.streams, self.maximum_streams, "stream")?;
        let capacity = match self
            .capacity
            .reserve(DeviceOpaqueMemoryKind::Stream, stream_bytes)
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
    // Isolated diagnostic opt-in: Kernel uses the existing eager command path
    // and creates at most two events per command plus submission start/fence.
    // Commands, live flights and lanes are already bounded before enqueue; the
    // fence retains its flight/lane (and declared stream opaque allowance) until
    // its event owners are destroyed. No extra per-event byte claim is invented:
    // native event storage is within the caller-declared bootstrap/stream opaque
    // allowances, which remain experimental and unqualified. This diagnostic
    // allowance is not evidence that the production memory profile is qualified.
    // Off remains the default; reusable capture and execution facts stay disabled.
    if commands == 0
        || commands > maximum_commands
        || !matches!(timing, DeviceTimingMode::Off | DeviceTimingMode::Kernel)
        || facts != ExecutionFactsCapture::Off
    {
        return Err(CudaDeviceRuntimeError::contract(format!(
            "CUDA tensor eager profile requires timing Off or diagnostic Kernel, facts Off, and 1..={maximum_commands} total commands"
        )));
    }
    Ok(())
}

impl TensorLaneRetention {
    pub(super) fn blas_workspace_bytes(&self) -> u64 {
        self.owner.decode_replay().map_or(0, |_| {
            crate::backend::cuda_tensor_causal_policy::CudaCausalDecodeReplay::BLAS_WORKSPACE_BYTES
        })
    }

    pub(crate) fn replay_limit(&self) -> Option<usize> {
        self.owner.decode_replay().map(|p| p.maximum_executables)
    }
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
pub(crate) fn create_native<T, O>(
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

fn query_device_total(ordinal: usize) -> Result<u64, CudaDeviceRuntimeError> {
    let ordinal = i32::try_from(ordinal)
        .map_err(|_| CudaDeviceRuntimeError::contract("CUDA ordinal exceeds i32"))?;
    let mut device = 0;
    let mut total = 0_usize;
    // These driver device queries do not create/retain a CUDA context.
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
    u64::try_from(total).map_err(|_| CudaDeviceRuntimeError::contract("CUDA memory exceeds u64"))
}

impl CudaDeviceRuntime {
    pub fn new_tensor_eager(
        config: CudaTensorRuntimeConfig,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_with_native_lane_reuse(config, CudaNativeLaneReuse::Off)
    }

    pub fn new_tensor_and_causal_eager(
        config: CudaTensorAndCausalRuntimeConfig,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_and_causal_eager_with_native_lane_reuse(config, CudaNativeLaneReuse::Off)
    }

    pub fn new_tensor_and_causal_eager_auto(
        config: CudaTensorAndCausalRuntimeAutoConfig,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_and_causal_eager_auto_with_native_lane_reuse(
            config,
            CudaNativeLaneReuse::Off,
        )
    }

    /// Creates one runtime/context and queries free memory on that context.
    /// The conservative total ceiling is min(observed free, floor(total *
    /// utilization), optional maximum). Bootstrap remains charged within it;
    /// its declared allowance is never added back to observed free memory.
    pub fn new_tensor_eager_auto(
        config: CudaTensorRuntimeAutoConfig,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_auto_with_options(config, CudaTensorRuntimeOptions::default())
    }

    /// Retain only fully retired native lanes within the original stream quota.
    pub fn new_tensor_eager_with_native_lane_reuse(
        config: CudaTensorRuntimeConfig,
        reuse: CudaNativeLaneReuse,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_with_options(
            config,
            CudaTensorRuntimeOptions {
                native_lane_reuse: reuse,
                ..Default::default()
            },
        )
    }

    pub fn new_tensor_eager_auto_with_native_lane_reuse(
        config: CudaTensorRuntimeAutoConfig,
        reuse: CudaNativeLaneReuse,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_auto_with_options(
            config,
            CudaTensorRuntimeOptions {
                native_lane_reuse: reuse,
                ..Default::default()
            },
        )
    }

    pub fn new_tensor_and_causal_eager_with_native_lane_reuse(
        config: CudaTensorAndCausalRuntimeConfig,
        reuse: CudaNativeLaneReuse,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_and_causal_eager_with_options(
            config,
            CudaTensorRuntimeOptions {
                native_lane_reuse: reuse,
                ..Default::default()
            },
        )
    }

    pub fn new_tensor_and_causal_eager_auto_with_native_lane_reuse(
        config: CudaTensorAndCausalRuntimeAutoConfig,
        reuse: CudaNativeLaneReuse,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_and_causal_eager_auto_with_options(
            config,
            CudaTensorRuntimeOptions {
                native_lane_reuse: reuse,
                ..Default::default()
            },
        )
    }

    pub fn new_tensor_eager_with_options(
        config: CudaTensorRuntimeConfig,
        options: CudaTensorRuntimeOptions,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_with_observation(config, None, None, options)
    }

    pub fn new_tensor_eager_auto_with_options(
        config: CudaTensorRuntimeAutoConfig,
        options: CudaTensorRuntimeOptions,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_auto_with_profile(config, None, options)
    }

    pub fn new_tensor_and_causal_eager_with_options(
        config: CudaTensorAndCausalRuntimeConfig,
        options: CudaTensorRuntimeOptions,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_with_observation(config.tensor, None, Some(config.causal), options)
    }

    pub fn new_tensor_and_causal_eager_auto_with_options(
        config: CudaTensorAndCausalRuntimeAutoConfig,
        options: CudaTensorRuntimeOptions,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_tensor_eager_auto_with_profile(config.tensor, Some(config.causal), options)
    }

    fn new_tensor_eager_auto_with_profile(
        config: CudaTensorRuntimeAutoConfig,
        causal: Option<CudaDenseF16CausalProfile>,
        options: CudaTensorRuntimeOptions,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        options.native_lane_reuse.validate(config.maximum_streams)?;
        options.native_allocation_reuse.validate()?;
        if let Some(policy) = options.pinned_weight_upload {
            policy.validate()?;
        }
        let policy = AutoMemoryPolicy::new(config.utilization, config.maximum_memory_bytes)
            .map_err(CudaDeviceRuntimeError::contract)?;
        let total = query_device_total(config.device.ordinal)?;
        let usable_memory_bytes = policy
            .provisional_ceiling(total)
            .map_err(CudaDeviceRuntimeError::contract)?;
        Self::new_tensor_eager_with_observation(
            CudaTensorRuntimeConfig {
                device: config.device,
                usable_memory_bytes,
                bootstrap_memory_bytes: config.bootstrap_memory_bytes,
                stream_memory_bytes: config.stream_memory_bytes,
                maximum_streams: config.maximum_streams,
                maximum_in_flight_per_stream: config.maximum_in_flight_per_stream,
                maximum_commands_per_submission: config.maximum_commands_per_submission,
                opaque_memory_profile: config.opaque_memory_profile,
            },
            Some((policy, total)),
            causal,
            options,
        )
    }

    pub fn tensor_memory_observation(&self) -> Option<&CudaTensorMemoryObservation> {
        self.tensor_eager
            .as_ref()
            .and_then(|state| state.memory_observation.as_ref())
    }

    fn new_tensor_eager_with_observation(
        mut config: CudaTensorRuntimeConfig,
        automatic: Option<(AutoMemoryPolicy, u64)>,
        causal: Option<CudaDenseF16CausalProfile>,
        options: CudaTensorRuntimeOptions,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        options.native_lane_reuse.validate(config.maximum_streams)?;
        options.native_allocation_reuse.validate()?;
        if let Some(policy) = options.pinned_weight_upload {
            policy.validate()?;
        }
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
        if causal.is_some()
            && config.device.attention_execution_policy != AttentionExecutionPolicy::Portable
        {
            return Err(CudaDeviceRuntimeError::contract(
                "bounded dense-F16 causal construction requires explicit Portable attention",
            ));
        }
        crate::backend::cuda_tensor_causal_policy::validate_profile_capabilities(
            &config.device.capabilities,
            causal.as_ref(),
        )
        .map_err(CudaDeviceRuntimeError::contract)?;
        let streams = u64::try_from(config.maximum_streams)
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA stream limit exceeds u64"))?;
        let stream_bytes = config
            .stream_memory_bytes
            .checked_add(
                causal
                    .as_ref()
                    .and_then(CudaDenseF16CausalProfile::decode_replay)
                    .map_or(0, |p| p.memory_bytes_per_stream()),
            )
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("causal graph stream allowance overflows")
            })?;
        let overhead = stream_bytes
            .checked_mul(streams)
            .and_then(|bytes| bytes.checked_add(config.bootstrap_memory_bytes))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("CUDA tensor opaque budget overflows")
            })?;
        let overhead = crate::backend::cuda_tensor_causal_policy::add_module_allowance(
            overhead,
            causal.as_ref(),
        )
        .map_err(CudaDeviceRuntimeError::contract)?;
        let overhead = overhead
            .checked_add(options.native_allocation_reuse.maximum_idle_bytes())
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("native allocation cache allowance overflows")
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
        let total = match automatic {
            Some((_, total)) => total,
            None => query_device_total(config.device.ordinal)?,
        };
        // Numeric policy and resource policy jointly identify this runtime.
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        for part in [
            if automatic.is_some() {
                "ferrum-cuda-tensor-eager-auto-v1"
            } else {
                "ferrum-cuda-tensor-eager-v1"
            }
            .to_owned(),
            config.device.runtime_implementation_fingerprint.clone(),
            config.opaque_memory_profile.clone(),
            automatic.map_or_else(
                || config.usable_memory_bytes.to_string(),
                |(policy, _)| policy.identity(),
            ),
            config.bootstrap_memory_bytes.to_string(),
            config.stream_memory_bytes.to_string(),
            config.maximum_streams.to_string(),
            config.maximum_in_flight_per_stream.to_string(),
            config.maximum_commands_per_submission.to_string(),
        ] {
            hash.update((part.len() as u64).to_le_bytes());
            hash.update(part.as_bytes());
        }
        if let CudaNativeLaneReuse::Bounded { maximum_idle_lanes } = options.native_lane_reuse {
            hash.update(b"retired-native-lanes-v1");
            hash.update((maximum_idle_lanes.get() as u64).to_le_bytes());
            hash.update(include_bytes!("idle_lanes.rs"));
        }
        if let CudaNativeAllocationReuse::Bounded {
            maximum_idle_bytes,
            maximum_entry_bytes,
            maximum_entries,
        } = options.native_allocation_reuse
        {
            hash.update(b"retired-native-allocations-largest-exact-v1");
            hash.update(maximum_idle_bytes.get().to_le_bytes());
            hash.update(maximum_entry_bytes.get().to_le_bytes());
            hash.update((maximum_entries.get() as u64).to_le_bytes());
            hash.update(include_bytes!("idle_allocations.rs"));
        }
        if let Some(policy) = options.pinned_weight_upload {
            hash.update(b"cuda.pinned-managed-weights.v1\0");
            hash.update((policy.maximum_allocation_bytes.get() as u64).to_le_bytes());
            hash.update((policy.maximum_registered_bytes.get() as u64).to_le_bytes());
            hash.update(include_bytes!("pinned_host.rs"));
        }
        // Allocator ownership/recovery is part of the executable runtime
        // contract, including for explicitly bounded (non-auto) construction.
        for implementation in [
            include_bytes!("allocation.rs").as_slice(),
            include_bytes!("allocation/pool.rs").as_slice(),
        ] {
            hash.update((implementation.len() as u64).to_le_bytes());
            hash.update(implementation);
        }
        // The old constructor adds no causal declaration to its identity.
        if let Some(profile) = &causal {
            for part in [
                "ferrum-cuda-tensor-and-dense-f16-causal-v1".to_owned(),
                profile.module_memory_bytes().to_string(),
                profile.opaque_memory_profile().to_owned(),
                format!("visual-prompt-v1={}", profile.is_visual_prompt_v1()),
                format!("decode-replay-v1={:?}", profile.decode_replay_limits()),
            ] {
                hash.update((part.len() as u64).to_le_bytes());
                hash.update(part.as_bytes());
            }
            for implementation in [
                include_bytes!("tensor_eager.rs").as_slice(),
                include_bytes!("tensor_eager/causal_policy.rs").as_slice(),
                include_bytes!("tensor_eager/causal_commands.rs").as_slice(),
                include_bytes!("blas.rs").as_slice(),
            ] {
                hash.update((implementation.len() as u64).to_le_bytes());
                hash.update(implementation);
            }
        }
        if automatic.is_some() {
            let implementation = include_bytes!("tensor_eager/memory_policy.rs");
            hash.update((implementation.len() as u64).to_le_bytes());
            hash.update(implementation);
        }
        config.device.runtime_implementation_fingerprint = format!("{:x}", hash.finalize());
        let descriptor = DeviceDescriptor {
            id: config.device.device_id.clone(),
            class: DeviceClass::Accelerator,
            ordinal: u32::try_from(config.device.ordinal)
                .map_err(|_| CudaDeviceRuntimeError::contract("CUDA ordinal exceeds u32"))?,
            total_memory_bytes: total,
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
        let allocation_cache_capacity = if options.native_allocation_reuse.maximum_idle_bytes() == 0
        {
            None
        } else {
            Some(
                capacity
                    .reserve(
                        DeviceOpaqueMemoryKind::Bootstrap,
                        options.native_allocation_reuse.maximum_idle_bytes(),
                    )
                    .map_err(CudaDeviceRuntimeError::capacity)?,
            )
        };
        let state = Arc::new(TensorEagerState {
            allocation_pool: None,
            capacity,
            _bootstrap: bootstrap,
            memory_observation: None,
            stream_bytes: config.stream_memory_bytes,
            maximum_streams: config.maximum_streams,
            maximum_in_flight: config.maximum_in_flight_per_stream,
            maximum_commands: config.maximum_commands_per_submission,
            streams: AtomicUsize::new(0),
            provider_live: AtomicBool::new(false),
            cutlass_module: Mutex::new(None),
            cutlass_provider_live: AtomicBool::new(false),
            fused_f32_module: Mutex::new(None),
            fused_f32_provider_live: AtomicBool::new(false),
            convrot_fused_module: Mutex::new(None),
            convrot_fused_provider_live: AtomicBool::new(false),
            causal_profile: causal,
            causal_modules: Mutex::new(std::sync::Weak::new()),
        });
        let (runtime, state) = create_native(state, || Self::new(config.device))?;
        let mut guard = RetainedNative::new(runtime, state);
        if guard.native().descriptor() != &descriptor {
            guard.retain_owner();
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor device descriptor changed during context creation",
            ));
        }
        if let Some((policy, _)) = automatic {
            let (free, observed_total) =
                guard.native().context.mem_get_info().map_err(|error| {
                    CudaDeviceRuntimeError::driver("context free-memory query", error)
                })?;
            let free = u64::try_from(free)
                .map_err(|_| CudaDeviceRuntimeError::contract("CUDA free memory exceeds u64"))?;
            let observed_total = u64::try_from(observed_total).map_err(|_| {
                CudaDeviceRuntimeError::contract("CUDA observed total memory exceeds u64")
            })?;
            if observed_total != total {
                return Err(CudaDeviceRuntimeError::contract(
                    "CUDA context total memory differs from its pre-context descriptor",
                ));
            }
            let observation = policy
                .observe(observed_total, free, overhead)
                .map_err(CudaDeviceRuntimeError::contract)?;
            // Opening a second owner only lowers the shared ceiling; it does
            // not duplicate the bootstrap claim. Both owners use the original
            // immutable descriptor. No registry has observed this runtime yet.
            let state = Arc::get_mut(guard.owner_mut()).ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA constructor quota unexpectedly escaped before publication",
                )
            })?;
            let final_capacity =
                DeviceCapacityOwner::open(&descriptor, observation.effective_ceiling_bytes())
                    .map_err(CudaDeviceRuntimeError::capacity)?;
            state.capacity = final_capacity;
            state.memory_observation = Some(observation);
        }
        if guard.native().context.has_async_alloc() {
            let stream = Arc::clone(&guard.native().allocation_stream);
            let state = Arc::get_mut(guard.owner_mut()).ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "CUDA allocator owner escaped before private pool construction",
                )
            })?;
            let release_threshold_bytes = state
                .memory_observation
                .as_ref()
                .map_or(
                    config.usable_memory_bytes,
                    CudaTensorMemoryObservation::effective_ceiling_bytes,
                )
                .checked_sub(overhead)
                .filter(|bytes| *bytes > 0)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "CUDA tensor cache ceiling leaves no managed storage",
                    )
                })?;
            state.allocation_pool = Some(
                allocation::TensorAllocationPool::new(
                    stream,
                    release_threshold_bytes,
                    state._bootstrap.clone(),
                    state.capacity.clone(),
                )
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("private tensor allocation pool creation", error)
                })?,
            );
        }
        let (mut runtime, state) = guard.into_parts();
        runtime.tensor_eager = Some(state);
        if let Some(policy) = options.pinned_weight_upload {
            runtime.pinned_weight_upload = Some((
                pinned_host::CudaPinnedHostRegistry::new(
                    Arc::clone(&runtime.context),
                    policy.maximum_registered_bytes.get(),
                )?,
                policy.maximum_allocation_bytes.get(),
            ));
        }
        runtime.idle_native_lanes = idle_lanes::IdleNativeLanes::new(options.native_lane_reuse);
        runtime.idle_native_allocations = idle_allocations::IdleAllocations::new(
            options.native_allocation_reuse,
            &runtime,
            allocation_cache_capacity,
        )?;
        Ok(runtime)
    }

    pub fn tensor_execution_profile(&self) -> Option<CudaTensorExecutionProfile> {
        self.tensor_eager
            .as_ref()
            .map(|state| state.execution_profile())
    }

    pub(crate) fn reserve_causal_provider(
        &self,
    ) -> Result<Arc<CudaCausalProviderRetention>, CudaDeviceRuntimeError> {
        let state = self.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "dense-F16 causal module admission requires a bounded runtime",
            )
        })?;
        let profile = state.causal_profile.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract("tensor-only runtime does not admit causal modules")
        })?;
        let mut live = state
            .causal_modules
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("causal module owner lock poisoned"))?;
        if live.upgrade().is_some() {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA causal provider module capacity exhausted",
            ));
        }
        let capacity = self.reserve_tensor_opaque(profile.module_memory_bytes())?;
        let retention = Arc::new(CudaCausalProviderRetention {
            _capacity: capacity,
            _owner: Arc::clone(state),
        });
        *live = Arc::downgrade(&retention);
        Ok(retention)
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
        let retention = owner
            .capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, bytes)
            .map_err(CudaDeviceRuntimeError::capacity)?;
        self.prepare_native_allocator()?;
        Ok(retention)
    }

    /// Returns only idle pool backing before an already admitted native owner
    /// allocates outside the managed pool. This does not grant new capacity.
    pub(super) fn prepare_native_allocator(&self) -> Result<(), CudaDeviceRuntimeError> {
        let owner = self.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract("native owner requires bounded tensor runtime")
        })?;
        // cuDNN/SDPA/CUTLASS allocate outside our managed pool. Return its
        // idle physical backing before a newly admitted native owner is built;
        // logical admission alone does not make those cached pages available.
        // Do not change the threshold: repeated native-owner replacement must
        // not accumulate historical allowances or disable future cache reuse.
        if let Some(pool) = owner.allocation_pool.as_ref() {
            self.allocation_stream.synchronize().map_err(|error| {
                CudaDeviceRuntimeError::driver("native owner allocation synchronization", error)
            })?;
            pool.trim_idle().map_err(|error| {
                CudaDeviceRuntimeError::driver("native owner allocation pool trim", error)
            })?;
        }
        Ok(())
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
        assert!(check(64, 64, DeviceTimingMode::Kernel, ExecutionFactsCapture::Off).is_ok());
        assert!(check(64, 65, DeviceTimingMode::Kernel, ExecutionFactsCapture::Off).is_err());
        assert!(check(64, 64, DeviceTimingMode::Kernel, ExecutionFactsCapture::On).is_err());
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
            allocation_pool: None,
            _bootstrap: capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 16)
                .unwrap(),
            capacity: capacity.clone(),
            memory_observation: None,
            stream_bytes: 32,
            maximum_streams: 1,
            maximum_in_flight: 1,
            maximum_commands: CUDA_TENSOR_MAX_COMMANDS_PER_SUBMISSION,
            streams: AtomicUsize::new(0),
            provider_live: AtomicBool::new(false),
            cutlass_module: Mutex::new(None),
            cutlass_provider_live: AtomicBool::new(false),
            fused_f32_module: Mutex::new(None),
            fused_f32_provider_live: AtomicBool::new(false),
            convrot_fused_module: Mutex::new(None),
            convrot_fused_provider_live: AtomicBool::new(false),
            causal_profile: None,
            causal_modules: Mutex::new(std::sync::Weak::new()),
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

    fn causal_module_test_owners() -> (
        DeviceCapacityOwner,
        Arc<TensorEagerState>,
        Arc<CudaCausalProviderRetention>,
    ) {
        let identity = NEXT_RUNTIME_INSTANCE.fetch_add(1, Ordering::Relaxed);
        let descriptor = DeviceDescriptor {
            id: DeviceId::new(format!("device.test.causal-module-retention-{identity}")).unwrap(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: 128,
            runtime_implementation_fingerprint: "b".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        };
        let capacity = DeviceCapacityOwner::open(&descriptor, 128).unwrap();
        let state = Arc::new(TensorEagerState {
            allocation_pool: None,
            _bootstrap: capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 16)
                .unwrap(),
            capacity: capacity.clone(),
            memory_observation: None,
            stream_bytes: 8,
            maximum_streams: 3,
            maximum_in_flight: 1,
            maximum_commands: 8,
            streams: AtomicUsize::new(0),
            provider_live: AtomicBool::new(false),
            cutlass_module: Mutex::new(None),
            cutlass_provider_live: AtomicBool::new(false),
            fused_f32_module: Mutex::new(None),
            fused_f32_provider_live: AtomicBool::new(false),
            convrot_fused_module: Mutex::new(None),
            convrot_fused_provider_live: AtomicBool::new(false),
            causal_profile: Some(
                CudaDenseF16CausalProfile::new(32, "test-declaration-v1").unwrap(),
            ),
            causal_modules: Mutex::new(std::sync::Weak::new()),
        });
        let quota = Arc::new(CudaCausalProviderRetention {
            _capacity: capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 32)
                .unwrap(),
            _owner: Arc::clone(&state),
        });
        *state.causal_modules.lock().unwrap() = Arc::downgrade(&quota);
        (capacity, state, quota)
    }

    #[test]
    fn causal_command_native_fields_drop_before_the_last_independent_module_grant() {
        // Metadata/real ledger test: no context, module, stream or fake device
        // computation is created. The captured sentinel observes destruction
        // at the exact position occupied by native-function closure owners.
        struct NativeDrop(DeviceCapacityOwner);
        impl Drop for NativeDrop {
            fn drop(&mut self) {
                assert!(
                    self.0
                        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 81)
                        .is_err(),
                    "the additional 32-byte grant must still be live during native-field cleanup"
                );
            }
        }
        let (capacity, state, quota) = causal_module_test_owners();
        let native = NativeDrop(capacity.clone());
        let executable = CudaCommandExecutable {
            regions: Vec::new(),
            host_storage: Vec::new(),
            host_upload_layout: None,
            work_declaration: CudaWorkDeclaration::Static,
            enqueue: Mutex::new(CudaEnqueueAction::Static(Box::new(move |_, _, _, _| {
                let _retained_native_field = &native;
                Ok(())
            }))),
            _causal_modules: Some(Arc::clone(&quota)),
        };
        drop(quota);
        assert!(state.causal_retention().is_some());
        assert!(capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 81)
            .is_err());
        drop(executable);
        assert!(state.causal_retention().is_none());
        drop(
            capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 112)
                .unwrap(),
        );
    }

    // These are admission/ownership tests, not device-copy execution tests.
    // The existing metadata command helper creates no native context or buffer;
    // its module allowance is nevertheless an actual DeviceCapacityOwner grant.
    fn transfer_only_command(quota: Option<Arc<CudaCausalProviderRetention>>) -> CudaDeviceCommand {
        let mut command = super::super::tests::command("test_device_row_concat")
            .with_work_attribution(DeviceBatchingForm::ParticipantLoop, 2, 3, 0, 6)
            .unwrap();
        Arc::get_mut(command.executable.as_mut().unwrap())
            .unwrap()
            ._causal_modules = quota;
        command
    }

    #[test]
    fn bounded_causal_device_copies_preserve_real_work_and_module_ownership() {
        let (capacity, state, quota) = causal_module_test_owners();
        let mut command = transfer_only_command(Some(quota));
        command
            .prepare_bounded_causal_phase(DeviceCommandPhase::Compute, false, None)
            .unwrap();
        assert_eq!(command.compute_dispatch_count, 0);
        assert_eq!(command.transfer_command_count, 6);
        assert_eq!(command.participant_count, 2);
        assert_eq!(command.token_count, 3);
        assert!(state.causal_retention().is_some());
        assert!(capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 81)
            .is_err());
        drop(command);
        assert!(state.causal_retention().is_none());
        drop(
            capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 112)
                .unwrap(),
        );
        drop(state);
        drop(
            capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 128)
                .unwrap(),
        );
    }

    #[test]
    fn bounded_causal_copy_allowance_does_not_admit_empty_unowned_or_host_work() {
        let (capacity, state, quota) = causal_module_test_owners();
        let mut empty = transfer_only_command(Some(Arc::clone(&quota)));
        // A malformed provider must not use its valid module grant to submit
        // a compute-phase command with no physical work at all.
        empty.transfer_command_count = 0;
        assert!(empty
            .prepare_bounded_causal_phase(DeviceCommandPhase::Compute, false, None)
            .is_err());
        drop(empty);

        let mut unowned = transfer_only_command(None);
        assert!(unowned
            .prepare_bounded_causal_phase(DeviceCommandPhase::Compute, false, None)
            .is_err());
        drop(unowned);

        let mut upload = transfer_only_command(Some(Arc::clone(&quota)));
        let executable = Arc::get_mut(upload.executable.as_mut().unwrap()).unwrap();
        executable
            .host_storage
            .push(vec![0u8; 4].into_boxed_slice());
        executable.host_upload_layout = Some(HostTransferLayout::new(ElementType::F16, 2).unwrap());
        assert!(upload
            .prepare_bounded_causal_phase(DeviceCommandPhase::Compute, false, None)
            .is_err());
        drop(upload);
        drop(quota);
        assert!(state.causal_retention().is_none());
        drop(state);
        drop(
            capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 128)
                .unwrap(),
        );
    }

    #[test]
    fn bounded_causal_copy_allowance_keeps_reusable_work_out_of_eager_mode() {
        use crate::backend::cpu::vnext_runtime::CpuDeviceRuntime;
        use ferrum_runtime::{
            DeviceReusableExecutionProgramId, DeviceReusableExecutionSegment, ExecutionLane,
            ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
        };
        let (capacity, state, quota) = causal_module_test_owners();
        // Mint the typed lane identity through a real CPU execution lane. No
        // CUDA catalog or device execution is fabricated for this rejected call.
        let identity = NEXT_RUNTIME_INSTANCE.fetch_add(1, Ordering::Relaxed);
        let runtime = Arc::new(
            CpuDeviceRuntime::new(DeviceDescriptor {
                id: DeviceId::new(format!("device.test.causal-copy-lane-{identity}")).unwrap(),
                class: DeviceClass::Host,
                ordinal: 0,
                total_memory_bytes: 128,
                runtime_implementation_fingerprint: "c".repeat(64),
                capabilities: BTreeSet::new(),
                dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                    DynamicStorageAllocator::LinearArena,
                    DynamicStorageView::Contiguous,
                )
                .unwrap()]),
            })
            .unwrap(),
        );
        let lane = ExecutionLane::create(runtime).unwrap();
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("test.causal-copy").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap();
        let program = DeviceReusableExecutionProgramId::new(
            serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
            "c".repeat(64),
            lane.id(),
            bucket.bucket_id().clone(),
            "d".repeat(64),
            "e".repeat(64),
            0,
            1,
            1,
            1,
        )
        .unwrap();
        let invocation = DeviceReusableExecutionInvocation::new(
            program,
            DeviceReusableExecutionSegment::new(0, 0, 1, 1).unwrap(),
            1,
            1,
        )
        .unwrap();
        let mut command = CudaDeviceCommand::reusable_execution(identity, invocation);
        Arc::get_mut(command.executable.as_mut().unwrap())
            .unwrap()
            ._causal_modules = Some(quota);
        assert!(command
            .prepare_bounded_causal_phase(DeviceCommandPhase::Compute, false, None)
            .is_err());
        drop(command);
        assert!(state.causal_retention().is_none());
        drop(state);
        drop(
            capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, 128)
                .unwrap(),
        );
        drop(lane);
    }
}
