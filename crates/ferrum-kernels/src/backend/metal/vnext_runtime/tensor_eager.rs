//! Fixed-tensor runtime ownership. Opaque budgets are declarations, not measured peaks.
use super::*;
use ferrum_runtime::{DeviceCapacityOwner, DeviceCapacityRetention, DeviceOpaqueMemoryKind};
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicUsize;
mod memory;
pub use memory::{MetalTensorMemoryObservation, MetalTensorRuntimeAutoConfig};

#[derive(Debug, Clone)]
pub struct MetalTensorRuntimeConfig {
    pub device: MetalDeviceRuntimeConfig,
    /// One total ceiling shared with every plan and opaque reservation.
    pub usable_memory_bytes: u64,
    pub bootstrap_memory_bytes: u64,
    pub stream_memory_bytes: u64,
    pub maximum_streams: usize,
    pub maximum_in_flight_per_stream: usize,
    pub maximum_commands_per_submission: usize,
    pub opaque_memory_profile: String,
}

impl MetalTensorRuntimeConfig {
    pub fn validate(&self) -> Result<(), MetalDeviceRuntimeError> {
        if self.device.device_id.as_str() != "device.metal.0" {
            return Err(invalid(
                "Metal tensor runtime requires the canonical default device identity",
            ));
        }
        if self.maximum_streams == 0
            || self.maximum_in_flight_per_stream == 0
            || self.maximum_commands_per_submission == 0
            || self.maximum_commands_per_submission > u32::MAX as usize
            || self.bootstrap_memory_bytes == 0
            || self.stream_memory_bytes == 0
            || self.opaque_memory_profile.is_empty()
            || self.opaque_memory_profile.len() > 256
            || !self.opaque_memory_profile.is_ascii()
            || self
                .opaque_memory_profile
                .bytes()
                .any(|byte| byte.is_ascii_control())
        {
            return Err(invalid(
                "Metal tensor budgets require positive bounds and an explicit profile identity",
            ));
        }
        let overhead = self
            .stream_memory_bytes
            .checked_mul(self.maximum_streams as u64)
            .and_then(|bytes| bytes.checked_add(self.bootstrap_memory_bytes))
            .ok_or_else(|| invalid("Metal tensor opaque budget overflows"))?;
        if overhead >= self.usable_memory_bytes {
            return Err(invalid(
                "Metal opaque reservations leave no capacity for tensors",
            ));
        }
        self.maximum_streams
            .checked_mul(self.maximum_in_flight_per_stream)
            .and_then(|n| n.checked_mul(self.maximum_commands_per_submission))
            .and_then(|n| n.checked_mul(std::mem::size_of::<MetalDeviceCommand>()))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid("Metal tensor command capacity exceeds addressable memory"))?;
        Ok(())
    }
}

pub(super) struct TensorEagerState {
    _bootstrap: DeviceCapacityRetention,
    capacity: DeviceCapacityOwner,
    usable_bytes: u64,
    stream_bytes: u64,
    maximum_streams: usize,
    maximum_in_flight: usize,
    maximum_commands: usize,
    streams: AtomicUsize,
    provider_live: AtomicBool,
    native_memory: memory::NativeAllocationObservations,
}

pub(crate) struct MetalTensorProviderRetention {
    owner: Arc<TensorEagerState>,
}
impl Drop for MetalTensorProviderRetention {
    fn drop(&mut self) {
        self.owner.provider_live.store(false, Ordering::Release);
    }
}

struct LaneSlot {
    owner: Arc<TensorEagerState>,
}
impl Drop for LaneSlot {
    fn drop(&mut self) {
        self.owner.streams.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(super) struct TensorLaneRetention {
    // Native queue and capacity drop before returning the lane slot.
    queue: CommandQueue,
    _capacity: DeviceCapacityRetention,
    slot: LaneSlot,
    flights: AtomicUsize,
}
impl TensorLaneRetention {
    pub(super) fn queue(&self) -> &CommandQueue {
        &self.queue
    }
    pub(super) fn reserve_flight(
        self: &Arc<Self>,
    ) -> Result<TensorFlightRetention, MetalDeviceRuntimeError> {
        claim_count(
            &self.flights,
            self.slot.owner.maximum_in_flight,
            "in-flight",
        )?;
        Ok(TensorFlightRetention {
            lane: Arc::clone(self),
        })
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

impl TensorEagerState {
    pub(super) fn observe_memory(
        &self,
        device: &metal::DeviceRef,
    ) -> ferrum_runtime::DeviceNativeMemorySnapshot {
        self.native_memory.observe(device.current_allocated_size())
    }

    pub(super) fn create_lane(
        self: &Arc<Self>,
        device: &metal::Device,
    ) -> Result<Arc<TensorLaneRetention>, MetalDeviceRuntimeError> {
        claim_count(&self.streams, self.maximum_streams, "stream")?;
        let slot = LaneSlot {
            owner: Arc::clone(self),
        };
        let capacity = self
            .capacity
            .reserve(DeviceOpaqueMemoryKind::Stream, self.stream_bytes)
            .map_err(MetalDeviceRuntimeError::Capacity)?;
        let queue = device.new_command_queue();
        self.observe_memory(device);
        Ok(Arc::new(TensorLaneRetention {
            queue,
            _capacity: capacity,
            slot,
            flights: AtomicUsize::new(0),
        }))
    }

    pub(super) fn validate_submission(
        &self,
        commands: usize,
        timing: DeviceTimingMode,
    ) -> Result<(), MetalDeviceRuntimeError> {
        if commands == 0 || commands > self.maximum_commands || timing != DeviceTimingMode::Off {
            return Err(invalid(
                "Metal tensor eager submissions require timing off and the admitted command bound",
            ));
        }
        Ok(())
    }
}

impl MetalDeviceRuntime {
    /// Uses a device handle directly; never initializes legacy global providers.
    pub fn new_tensor_eager(
        config: MetalTensorRuntimeConfig,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        config.validate()?;
        let device =
            metal::Device::system_default().ok_or_else(|| invalid("no default Metal device"))?;
        Self::new_tensor_eager_on_device(config, device)
    }

    fn new_tensor_eager_on_device(
        mut config: MetalTensorRuntimeConfig,
        device: metal::Device,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        config.validate()?;
        let mut hash = Sha256::new();
        for part in [
            "ferrum-metal-tensor-eager-v1".to_owned(),
            config.device.runtime_implementation_fingerprint.clone(),
            device.registry_id().to_string(),
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
        hash.update(include_bytes!("tensor_eager.rs"));
        hash.update(include_bytes!("../vnext_runtime.rs"));
        hash.update(include_bytes!("managed_tensor.rs"));
        hash.update(include_bytes!("tensor_eager/memory.rs"));
        config.device.runtime_implementation_fingerprint = format!("{:x}", hash.finalize());
        let descriptor = DeviceDescriptor {
            id: config.device.device_id.clone(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: device.recommended_max_working_set_size(),
            runtime_implementation_fingerprint: config
                .device
                .runtime_implementation_fingerprint
                .clone(),
            capabilities: config.device.capabilities.clone(),
            dynamic_storage_profiles: config.device.dynamic_storage_profiles.clone(),
        };
        let capacity = DeviceCapacityOwner::open(&descriptor, config.usable_memory_bytes)
            .map_err(MetalDeviceRuntimeError::Capacity)?;
        let bootstrap = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.bootstrap_memory_bytes,
            )
            .map_err(MetalDeviceRuntimeError::Capacity)?;
        let state = Arc::new(TensorEagerState {
            _bootstrap: bootstrap,
            capacity,
            usable_bytes: config.usable_memory_bytes,
            stream_bytes: config.stream_memory_bytes,
            maximum_streams: config.maximum_streams,
            maximum_in_flight: config.maximum_in_flight_per_stream,
            maximum_commands: config.maximum_commands_per_submission,
            streams: AtomicUsize::new(0),
            provider_live: AtomicBool::new(false),
            native_memory: memory::NativeAllocationObservations::default(),
        });
        state.observe_memory(&device);
        Self::new_on_device(config.device, device, Some(state))
    }

    pub fn tensor_capacity_bytes(&self) -> Option<u64> {
        self.tensor_eager.as_ref().map(|state| state.usable_bytes)
    }

    /// Account for the complete page-aligned backing needed by no-copy inputs.
    pub fn tensor_host_account(
        &self,
        limits: ferrum_runtime::HostMemoryLimits,
    ) -> Result<ferrum_runtime::HostMemoryAccount, MetalDeviceRuntimeError> {
        if self.tensor_eager.is_none() {
            return Err(invalid(
                "Metal tensor host memory requires an admitted tensor runtime",
            ));
        }
        let alignment = crate::backend::metal::retained_memory::native_page_size()
            .map_err(|error| invalid(error.to_string()))?;
        ferrum_runtime::HostMemoryAccount::with_allocation_alignment(limits, alignment)
            .map_err(|error| invalid(error.to_string()))
    }

    pub(crate) fn retain_tensor_provider(
        &self,
    ) -> Result<MetalTensorProviderRetention, MetalDeviceRuntimeError> {
        let owner = self.tensor_eager.as_ref().ok_or_else(|| {
            invalid("Metal fixed tensor providers require an admitted tensor runtime")
        })?;
        owner
            .provider_live
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid("Metal tensor runtime already owns its native provider bundle"))?;
        Ok(MetalTensorProviderRetention {
            owner: Arc::clone(owner),
        })
    }
}

fn claim_count(
    counter: &AtomicUsize,
    limit: usize,
    label: &str,
) -> Result<(), MetalDeviceRuntimeError> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < limit).then(|| n + 1)
        })
        .map(|_| ())
        .map_err(|_| invalid(format!("Metal tensor {label} capacity exhausted")))
}
fn invalid(message: impl Into<String>) -> MetalDeviceRuntimeError {
    MetalDeviceRuntimeError::contract(message)
}

#[cfg(test)]
mod tests;
