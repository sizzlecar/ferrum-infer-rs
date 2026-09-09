use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrum_interfaces::vnext::{BufferDescriptor, DeviceBufferRetention, ElementType};

use super::CpuRuntimeError;

/// Includes resident allocations and owned uploads awaiting submission.
pub(super) struct MemoryBudget {
    limit: u64,
    used: AtomicU64,
    peak: AtomicU64,
}

impl MemoryBudget {
    pub(super) fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        })
    }

    pub(super) fn reserve(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<MemoryReservation, CpuRuntimeError> {
        let before = self.used.fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(bytes).filter(|next| *next <= self.limit)
        }).map_err(|used| CpuRuntimeError::new(format!(
            "CPU memory capacity exceeded: requested {bytes} bytes, {used} in use, {} available to this runtime", self.limit
        )))?;
        self.peak.fetch_max(before + bytes, Ordering::Relaxed);
        Ok(MemoryReservation {
            budget: Arc::clone(self),
            bytes,
        })
    }

    pub(super) fn used(&self) -> u64 {
        self.used.load(Ordering::Acquire)
    }
    pub(super) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

pub(super) struct MemoryReservation {
    budget: Arc<MemoryBudget>,
    bytes: u64,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        let prior = self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(prior >= self.bytes);
    }
}

/// The allocation's alignment is part of its actual allocator layout.
pub(super) struct Storage {
    pointer: NonNull<u8>,
    layout: Layout,
    _reservation: MemoryReservation,
}

// Storage owns its allocation. Shared buffer access is protected by a Mutex;
// uploads are immutable after encoding and move into a single command batch.
unsafe impl Send for Storage {}

impl Storage {
    pub(super) fn new(
        size: u64,
        alignment: u64,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, CpuRuntimeError> {
        let size = usize::try_from(size)
            .map_err(|_| CpuRuntimeError::new("CPU allocation exceeds usize"))?;
        let alignment = usize::try_from(alignment)
            .map_err(|_| CpuRuntimeError::new("CPU alignment exceeds usize"))?;
        if size == 0 {
            return Err(CpuRuntimeError::new("CPU allocation must be nonempty"));
        }
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|error| CpuRuntimeError::new(error.to_string()))?;
        let reservation = budget.reserve(size as u64)?;
        // SAFETY: Layout is nonempty and validated above. A null pointer is a
        // recoverable allocation failure; no slice is formed before this check.
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or_else(|| {
            CpuRuntimeError::new(format!("CPU allocation of {size} bytes failed"))
        })?;
        Ok(Self {
            pointer,
            layout,
            _reservation: reservation,
        })
    }

    pub(super) fn bytes(&self) -> &[u8] {
        // SAFETY: Storage owns this initialized allocation until Drop.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
    }

    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: An exclusive Storage borrow excludes all other slice access.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        // SAFETY: The pointer was allocated with this exact Layout and has not
        // been freed. The budget reservation drops only after deallocation.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

pub struct CpuDeviceBuffer {
    pub(super) descriptor: BufferDescriptor,
    pub(super) runtime_instance: u64,
    storage: Arc<Mutex<Storage>>,
}

impl CpuDeviceBuffer {
    pub(super) fn allocate(
        descriptor: BufferDescriptor,
        runtime_instance: u64,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, CpuRuntimeError> {
        let storage = Storage::new(descriptor.size_bytes, descriptor.alignment_bytes, budget)?;
        Ok(Self {
            descriptor,
            runtime_instance,
            storage: Arc::new(Mutex::new(storage)),
        })
    }

    pub(crate) fn retained_region(
        &self,
        range: Range<u64>,
        retention: DeviceBufferRetention,
    ) -> Result<CpuBufferRegion, CpuRuntimeError> {
        self.region_inner(range, Some(retention))
    }

    pub(super) fn region(&self, range: Range<u64>) -> Result<CpuBufferRegion, CpuRuntimeError> {
        self.region_inner(range, None)
    }

    fn region_inner(
        &self,
        range: Range<u64>,
        retention: Option<DeviceBufferRetention>,
    ) -> Result<CpuBufferRegion, CpuRuntimeError> {
        if range.start >= range.end || range.end > self.descriptor.size_bytes {
            return Err(CpuRuntimeError::new(
                "CPU buffer region is empty or exceeds its admitted allocation",
            ));
        }
        Ok(CpuBufferRegion {
            storage: Arc::clone(&self.storage),
            runtime_instance: self.runtime_instance,
            range: usize::try_from(range.start)
                .map_err(|_| CpuRuntimeError::new("CPU offset exceeds usize"))?
                ..usize::try_from(range.end)
                    .map_err(|_| CpuRuntimeError::new("CPU end exceeds usize"))?,
            element_type: self.descriptor.element_type,
            _retention: retention,
        })
    }
}

#[derive(Clone)]
pub(crate) struct CpuBufferRegion {
    storage: Arc<Mutex<Storage>>,
    runtime_instance: u64,
    range: Range<usize>,
    element_type: ElementType,
    _retention: Option<DeviceBufferRetention>,
}

impl CpuBufferRegion {
    fn lock(&self) -> Result<MutexGuard<'_, Storage>, CpuRuntimeError> {
        self.storage
            .lock()
            .map_err(|_| CpuRuntimeError::new("CPU buffer was poisoned by a failed computation"))
    }

    pub(crate) fn validate_runtime(&self, instance: u64) -> Result<(), CpuRuntimeError> {
        if self.runtime_instance == instance {
            Ok(())
        } else {
            Err(CpuRuntimeError::new(
                "CPU buffer belongs to another runtime",
            ))
        }
    }

    pub(crate) fn length_bytes(&self) -> usize {
        self.range.len()
    }
    pub(crate) fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub(crate) fn same_physical_region(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
            && self.range == other.range
            && self.element_type == other.element_type
    }

    /// Lock each distinct allocation once, in address order. Several value or
    /// weight regions may share one arena; nested per-region locks would then
    /// deadlock even when their byte ranges are disjoint.
    pub(crate) fn with_regions<T>(
        regions: &[Self],
        operation: impl FnOnce(&mut CpuRegionSet<'_>) -> Result<T, CpuRuntimeError>,
    ) -> Result<T, CpuRuntimeError> {
        let mut allocations = regions
            .iter()
            .map(|region| &region.storage)
            .collect::<Vec<_>>();
        allocations.sort_unstable_by_key(|storage| Arc::as_ptr(storage));
        allocations.dedup_by(|left, right| Arc::ptr_eq(left, right));
        let indices = regions
            .iter()
            .map(|region| {
                allocations
                    .binary_search_by_key(&Arc::as_ptr(&region.storage), |storage| {
                        Arc::as_ptr(storage)
                    })
                    .expect("every region belongs to the collected allocations")
            })
            .collect();
        let guards = allocations
            .iter()
            .map(|storage| {
                storage.lock().map_err(|_| {
                    CpuRuntimeError::new("CPU buffer was poisoned by a failed computation")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        operation(&mut CpuRegionSet {
            regions,
            indices,
            guards,
        })
    }

    pub(crate) fn with_read<T>(&self, read: impl FnOnce(&[u8]) -> T) -> Result<T, CpuRuntimeError> {
        let storage = self.lock()?;
        Ok(read(&storage.bytes()[self.range.clone()]))
    }

    pub(crate) fn write(&self, bytes: &[u8]) -> Result<(), CpuRuntimeError> {
        if bytes.len() != self.range.len() {
            return Err(CpuRuntimeError::new(
                "CPU write length differs from destination region",
            ));
        }
        self.lock()?.bytes_mut()[self.range.clone()].copy_from_slice(bytes);
        Ok(())
    }

    pub(super) fn zero(&self) -> Result<(), CpuRuntimeError> {
        self.lock()?.bytes_mut()[self.range.clone()].fill(0);
        Ok(())
    }

    pub(super) fn copy_to(&self, destination: &Self) -> Result<(), CpuRuntimeError> {
        if self.range.len() != destination.range.len() {
            return Err(CpuRuntimeError::new("CPU copy regions differ in length"));
        }
        if Arc::ptr_eq(&self.storage, &destination.storage) {
            self.lock()?
                .bytes_mut()
                .copy_within(self.range.clone(), destination.range.start);
        } else if Arc::as_ptr(&self.storage) < Arc::as_ptr(&destination.storage) {
            let source = self.lock()?;
            let mut target = destination.lock()?;
            target.bytes_mut()[destination.range.clone()]
                .copy_from_slice(&source.bytes()[self.range.clone()]);
        } else {
            let mut target = destination.lock()?;
            let source = self.lock()?;
            target.bytes_mut()[destination.range.clone()]
                .copy_from_slice(&source.bytes()[self.range.clone()]);
        }
        Ok(())
    }
}

/// Temporarily borrowed views of admitted allocations, never copied weights.
pub(crate) struct CpuRegionSet<'a> {
    regions: &'a [CpuBufferRegion],
    indices: Vec<usize>,
    guards: Vec<MutexGuard<'a, Storage>>,
}

impl CpuRegionSet<'_> {
    pub(crate) fn read(&self, index: usize) -> &[u8] {
        &self.guards[self.indices[index]].bytes()[self.regions[index].range.clone()]
    }

    #[cfg(test)]
    pub(crate) fn write(&mut self, index: usize) -> &mut [u8] {
        &mut self.guards[self.indices[index]].bytes_mut()[self.regions[index].range.clone()]
    }

    /// Give a kernel simultaneous input/output slices only after proving that
    /// writes are disjoint from every read and other write. Read aliases are
    /// allowed. In-place kernels use sequential read/write access instead.
    pub(crate) fn with_io<const I: usize, const O: usize, T>(
        &mut self,
        reads: [usize; I],
        writes: [usize; O],
        operation: impl FnOnce([&[u8]; I], [&mut [u8]; O]) -> Result<T, CpuRuntimeError>,
    ) -> Result<T, CpuRuntimeError> {
        let pointers = self.io_pointers(&reads, &writes)?;
        // SAFETY: All allocations are exclusively locked for this callback.
        // Regions were range-checked when retained. The overlap checks above
        // prove all mutable slices are mutually disjoint and have no read
        // aliases, including when they share one allocation. The callback
        // cannot return a reference borrowed from these temporary slices.
        let inputs = reads.map(|index| unsafe {
            std::slice::from_raw_parts(
                pointers[self.indices[index]].add(self.regions[index].range.start),
                self.regions[index].range.len(),
            )
        });
        let outputs = writes.map(|index| unsafe {
            std::slice::from_raw_parts_mut(
                pointers[self.indices[index]].add(self.regions[index].range.start),
                self.regions[index].range.len(),
            )
        });
        operation(inputs, outputs)
    }
    fn io_pointers(
        &mut self,
        reads: &[usize],
        writes: &[usize],
    ) -> Result<Vec<*mut u8>, CpuRuntimeError> {
        if reads
            .iter()
            .chain(writes)
            .any(|&index| index >= self.regions.len())
        {
            return Err(CpuRuntimeError::new("CPU kernel region index is invalid"));
        }
        let overlaps = |left: usize, right: usize| {
            self.indices[left] == self.indices[right]
                && self.regions[left].range.start < self.regions[right].range.end
                && self.regions[right].range.start < self.regions[left].range.end
        };
        for (position, &write) in writes.iter().enumerate() {
            if reads
                .iter()
                .chain(&writes[..position])
                .any(|&other| overlaps(write, other))
            {
                return Err(CpuRuntimeError::new(
                    "CPU kernel output aliases another live binding",
                ));
            }
        }
        Ok(self
            .guards
            .iter_mut()
            .map(|storage| storage.bytes_mut().as_mut_ptr())
            .collect::<Vec<_>>())
    }

    /// Dynamic page counts use the same alias proof as fixed-arity operators.
    pub(crate) fn with_io_slices<T>(
        &mut self,
        reads: &[usize],
        writes: &[usize],
        operation: impl FnOnce(&[&[u8]], &mut [&mut [u8]]) -> Result<T, CpuRuntimeError>,
    ) -> Result<T, CpuRuntimeError> {
        let pointers = self.io_pointers(reads, writes)?;
        // SAFETY: io_pointers validates every retained range and excludes all
        // mutable aliases while every owner allocation remains exclusively
        // locked. Callback borrows cannot escape this scope.
        let inputs = reads
            .iter()
            .map(|&index| unsafe {
                std::slice::from_raw_parts(
                    pointers[self.indices[index]].add(self.regions[index].range.start),
                    self.regions[index].range.len(),
                )
            })
            .collect::<Vec<_>>();
        let mut outputs = writes
            .iter()
            .map(|&index| unsafe {
                std::slice::from_raw_parts_mut(
                    pointers[self.indices[index]].add(self.regions[index].range.start),
                    self.regions[index].range.len(),
                )
            })
            .collect::<Vec<_>>();
        operation(&inputs, &mut outputs)
    }
}
