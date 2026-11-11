//! Bounded, unbound native backing. Every checkout still consumes a new Core
//! permit and must initialize its complete contents through the original path.
use super::*;
use ferrum_runtime::DeviceCapacityRetention;
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaTensorRuntimeOptions {
    pub native_lane_reuse: CudaNativeLaneReuse,
    pub native_allocation_reuse: CudaNativeAllocationReuse,
    /// In-place registration policy fixed before device capacity admission.
    pub pinned_weight_upload: Option<CudaPinnedWeightUploadConfig>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CudaNativeAllocationReuse {
    #[default]
    Off,
    Bounded {
        maximum_idle_bytes: NonZeroU64,
        maximum_entry_bytes: NonZeroU64,
        maximum_entries: NonZeroUsize,
    },
}

impl CudaNativeAllocationReuse {
    pub fn validate(self) -> Result<(), CudaDeviceRuntimeError> {
        if let Self::Bounded {
            maximum_idle_bytes,
            maximum_entry_bytes,
            maximum_entries,
        } = self
        {
            if maximum_entry_bytes > maximum_idle_bytes
                || maximum_idle_bytes.get() > isize::MAX as u64
                || maximum_entries.get()
                    > isize::MAX as usize / std::mem::size_of::<NativeBacking>()
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "native allocation reuse limits are inconsistent or unaddressable",
                ));
            }
        }
        Ok(())
    }

    pub const fn maximum_idle_bytes(self) -> u64 {
        match self {
            Self::Off => 0,
            Self::Bounded {
                maximum_idle_bytes, ..
            } => maximum_idle_bytes.get(),
        }
    }
}

struct BackingBudget {
    // An idle backing always retains its original stream, private pool and
    // fixed reservation. TensorEagerState does not point back to this cache.
    stream: Arc<CudaStream>,
    _tensor_owner: Arc<tensor_eager::TensorEagerState>,
    _capacity: DeviceCapacityRetention,
    poisoned: AtomicBool,
}

struct NativeBacking {
    // Taking the pointer marks retirement as possibly enqueued BEFORE calling
    // the driver. An error or panic must never retry the same free.
    pointer: Option<cudarc::driver::sys::CUdeviceptr>,
    bytes: usize,
    alignment: u64,
    owner: Arc<BackingBudget>,
}

impl NativeBacking {
    fn into_slice(mut self) -> CudaSlice<u8> {
        let pointer = self
            .pointer
            .take()
            .expect("live cache entry owns its pointer");
        // SAFETY: the exact original allocation and allocation stream are
        // retained; this runtime disables cudarc event tracking. No new CUDA
        // allocation, hidden view, event, or copy occurs here.
        unsafe { self.owner.stream.upgrade_device_ptr(pointer, self.bytes) }
    }

    fn retire(&mut self) -> Result<(), CudaDeviceRuntimeError> {
        self.retire_with(|pointer, stream| {
            stream.context().bind_to_thread()?;
            // SAFETY: only this cache entry owns this unbound native pointer;
            // final CudaAllocation retirement excluded all execution aliases.
            unsafe {
                cudarc::driver::result::free_async(pointer, stream.cu_stream())?;
            }
            stream.synchronize()
        })
    }

    fn retire_with(
        &mut self,
        release: impl FnOnce(cudarc::driver::sys::CUdeviceptr, &CudaStream) -> Result<(), DriverError>,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let Some(pointer) = self.pointer.take() else {
            return Ok(());
        };
        let result = catch_unwind(AssertUnwindSafe(|| release(pointer, &self.owner.stream)));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.retain_unknown();
                Err(CudaDeviceRuntimeError::driver(
                    "cached allocation retirement",
                    error,
                ))
            }
            Err(payload) => {
                self.retain_unknown();
                resume_unwind(payload)
            }
        }
    }

    fn retain_unknown(&self) {
        self.owner.poisoned.store(true, Ordering::Release);
        std::mem::forget(Arc::clone(&self.owner));
    }
}

impl Drop for NativeBacking {
    fn drop(&mut self) {
        if let Err(error) = self.retire() {
            tracing::error!(error = %error, allocation_bytes = self.bytes,
                "CUDA cached allocation retirement indeterminate; original pool and budget retained");
        }
    }
}

pub(super) struct IdleAllocations {
    maximum_bytes: u64,
    maximum_entry_bytes: u64,
    maximum_entries: usize,
    entries: Mutex<Vec<NativeBacking>>,
    owner: Arc<BackingBudget>,
}

impl IdleAllocations {
    pub(super) fn new(
        policy: CudaNativeAllocationReuse,
        runtime: &CudaDeviceRuntime,
        capacity: Option<DeviceCapacityRetention>,
    ) -> Result<Option<Arc<Self>>, CudaDeviceRuntimeError> {
        let CudaNativeAllocationReuse::Bounded {
            maximum_idle_bytes,
            maximum_entry_bytes,
            maximum_entries,
        } = policy
        else {
            debug_assert!(capacity.is_none());
            return Ok(None);
        };
        policy.validate()?;
        let stream = Arc::clone(&runtime.allocation_stream);
        let tensor_owner = Arc::clone(runtime.tensor_eager.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "native allocation reuse requires bounded tensor ownership",
            )
        })?);
        if !stream.context().has_async_alloc()
            || stream.context().is_event_tracking()
            || runtime.tensor_allocation_pool().is_none()
        {
            return Err(CudaDeviceRuntimeError::contract("native allocation reuse requires the original private async pool with event tracking disabled"));
        }
        let capacity = capacity.ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "native allocation reuse lacks its original reservation",
            )
        })?;
        Ok(Some(Arc::new(Self {
            maximum_bytes: maximum_idle_bytes.get(),
            maximum_entry_bytes: maximum_entry_bytes.get(),
            maximum_entries: maximum_entries.get(),
            entries: Mutex::new(Vec::new()),
            owner: Arc::new(BackingBudget {
                stream,
                _tensor_owner: tensor_owner,
                _capacity: capacity,
                poisoned: AtomicBool::new(false),
            }),
        })))
    }

    pub(super) fn take(
        &self,
        bytes: usize,
        alignment: u64,
    ) -> Result<Option<CudaSlice<u8>>, CudaDeviceRuntimeError> {
        let started = tracing::enabled!(tracing::Level::DEBUG).then(Instant::now);
        let mut entries = self.entries.lock().map_err(|_| {
            self.owner.poisoned.store(true, Ordering::Release);
            CudaDeviceRuntimeError::contract("native allocation reuse cache lock poisoned")
        })?;
        let mutex_wait_us = started.map_or(0, |started| {
            u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
        });
        if self.owner.poisoned.load(Ordering::Acquire) {
            return Err(CudaDeviceRuntimeError::contract(
                "native allocation reuse cache is poisoned",
            ));
        }
        let result = entries
            .iter()
            .position(|entry| entry.bytes == bytes && entry.alignment == alignment)
            .map(|index| entries.remove(index).into_slice());
        drop(entries);
        if started.is_some() {
            tracing::debug!(
                allocation_bytes = bytes,
                mutex_wait_us,
                reused = result.is_some(),
                "CUDA native backing checkout timing"
            );
        }
        Ok(result)
    }

    pub(super) fn put(&self, base: CudaSlice<u8>, alignment: u64) {
        let bytes = base.len();
        if bytes == 0 || bytes as u64 > self.maximum_entry_bytes {
            return;
        }
        let Ok(mut entries) = self.entries.lock() else {
            self.owner.poisoned.store(true, Ordering::Release);
            return;
        };
        if self.owner.poisoned.load(Ordering::Acquire) {
            return;
        }
        // Keep the largest retired blocks; equal sizes prefer older entries.
        // Decide the complete victim set before freeing anything. The policy
        // knows only exact native extents, never a model/resource identity.
        let Some(victims) = victims_for(&entries, bytes, self.maximum_bytes, self.maximum_entries)
        else {
            return;
        };
        let eviction_count = victims.len();
        // Host wall time, including retirement's stream wait under this mutex;
        // this is not a GPU event or a kernel execution duration.
        let eviction_started = tracing::enabled!(tracing::Level::DEBUG).then(Instant::now);
        for index in victims.into_iter().rev() {
            let mut evicted = entries.remove(index);
            if let Err(error) = evicted.retire() {
                tracing::error!(error = %error, "CUDA allocation cache eviction failed; cache poisoned");
                // Incoming bytes still follow their original CudaSlice path;
                // they were never published against the cache reservation.
                return;
            }
        }
        let eviction_retirement_us = eviction_started.map_or(0, |started| {
            u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
        });
        // The private pool has event tracking disabled, so leak transfers
        // ownership without recording/waiting on implicit cudarc events.
        let pointer = base.leak();
        entries.push(NativeBacking {
            pointer: Some(pointer),
            bytes,
            alignment,
            owner: Arc::clone(&self.owner),
        });
        tracing::debug!(
            native_allocation_returned = true,
            allocation_bytes = bytes,
            idle_entries = entries.len(),
            idle_bytes = entries.iter().map(|e| e.bytes as u64).sum::<u64>(),
            eviction_count,
            eviction_retirement_us,
            policy = "largest_retired_exact_v1",
            "CUDA native allocation retirement"
        );
    }
}

fn victims_for(
    entries: &[NativeBacking],
    incoming: usize,
    maximum_bytes: u64,
    maximum_entries: usize,
) -> Option<Vec<usize>> {
    let incoming_index = entries.len();
    let mut sizes = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.bytes as u64, i))
        .collect::<Vec<_>>();
    sizes.push((incoming as u64, incoming_index));
    let mut bytes = sizes
        .iter()
        .try_fold(0_u64, |sum, (n, _)| sum.checked_add(*n))?;
    let mut count = sizes.len();
    sizes.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    let mut victims = Vec::new();
    for (size, index) in sizes {
        if bytes <= maximum_bytes && count <= maximum_entries {
            break;
        }
        if index == incoming_index {
            return None;
        }
        victims.push(index);
        bytes -= size;
        count -= 1;
    }
    victims.sort_unstable();
    Some(victims)
}

pub(super) struct AllocationReturn {
    pub(super) cache: std::sync::Weak<IdleAllocations>,
    pub(super) alignment: u64,
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        let Some(base) = self._base.take() else {
            return;
        };
        if let Some((cache, alignment)) = self.idle_allocation_return.as_ref().and_then(|origin| {
            origin
                .cache
                .upgrade()
                .map(|cache| (cache, origin.alignment))
        }) {
            cache.put(base, alignment);
        }
        // A late return after runtime drop follows the original CudaSlice
        // retirement. Its original TensorEagerState remains live through this
        // destructor; no cache ownership or successful recycle is fabricated.
    }
}

#[cfg(test)]
mod tests;
