//! Shared admission and ownership of explicitly managed host byte allocations.
//!
//! Product composition injects one account into all cooperating engines. Limits
//! cover physical backing bytes (including configured padding) and reservations,
//! not allocator metadata, page cache, network buffers, or total process RSS.
//! A transfer classification does not prove page locking: this module currently
//! allocates pageable memory only.

use std::{
    alloc::Layout,
    collections::BTreeMap,
    ops::Range,
    sync::{Arc, Mutex},
};

use thiserror::Error;

mod scope;
pub use scope::HostMemoryScope;
use scope::{ScopeAuthority, ScopeCredit};

/// A quota dimension; it does not describe allocation or DMA capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemoryClass {
    InputPreprocess,
    TransferStaging,
    Output,
    WeightBacking,
}

impl HostMemoryClass {
    const fn index(self) -> usize {
        match self {
            Self::InputPreprocess => 0,
            Self::TransferStaging => 1,
            Self::Output => 2,
            Self::WeightBacking => 3,
        }
    }
}

/// Zero disables a class. Class ceilings are shared by every account clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryClassLimits {
    pub input_preprocess_bytes: usize,
    pub transfer_staging_bytes: usize,
    pub output_bytes: usize,
    pub weight_backing_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryLimits {
    total_bytes: usize,
    classes: [usize; 4],
}

impl HostMemoryLimits {
    pub fn new(
        total_bytes: usize,
        classes: HostMemoryClassLimits,
    ) -> Result<Self, HostMemoryError> {
        let classes = [
            classes.input_preprocess_bytes,
            classes.transfer_staging_bytes,
            classes.output_bytes,
            classes.weight_backing_bytes,
        ];
        if total_bytes == 0 || classes.iter().any(|&limit| limit > total_bytes) {
            return Err(HostMemoryError::InvalidLimits);
        }
        Ok(Self {
            total_bytes,
            classes,
        })
    }

    pub const fn total_bytes(self) -> usize {
        self.total_bytes
    }

    pub const fn class_bytes(self, class: HostMemoryClass) -> usize {
        self.classes[class.index()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemoryPressureScope {
    Total,
    Class(HostMemoryClass),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HostMemoryError {
    #[error("host total limit must be non-zero and class limits must not exceed it")]
    InvalidLimits,
    #[error("host reservation requires non-empty allocations within the addressable byte range")]
    InvalidReservation,
    #[error("host byte accounting overflows the address space")]
    ArithmeticOverflow,
    #[error("host {scope:?} capacity exhausted: requested {requested}, committed {committed}, limit {limit}")]
    Capacity {
        scope: HostMemoryPressureScope,
        requested: usize,
        committed: usize,
        limit: usize,
    },
    #[error("host allocation of {bytes} bytes failed")]
    AllocationFailed { bytes: usize },
    #[error("host memory account is fail-closed")]
    Poisoned,
    #[error("host byte slice is inverted or exceeds its retained owner")]
    InvalidSlice,
    #[error("host reservation scope is closed")]
    ScopeClosed,
    #[error(
        "host {class:?} scoped capacity exhausted: requested {requested}, available {available}"
    )]
    ScopeCapacity {
        class: HostMemoryClass,
        requested: usize,
        available: usize,
    },
    #[error("host reservation scopes cannot be nested")]
    NestedScope,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Counters {
    total: usize,
    classes: [usize; 4],
}

impl Counters {
    fn add(self, other: Self) -> Option<Self> {
        let mut next = self;
        next.total = next.total.checked_add(other.total)?;
        for (value, amount) in next.classes.iter_mut().zip(other.classes) {
            *value = value.checked_add(amount)?;
        }
        Some(next)
    }

    fn subtract(self, other: Self) -> Option<Self> {
        let mut next = self;
        next.total = next.total.checked_sub(other.total)?;
        for (value, amount) in next.classes.iter_mut().zip(other.classes) {
            *value = value.checked_sub(amount)?;
        }
        Some(next)
    }

    fn single(class: HostMemoryClass, bytes: usize) -> Self {
        let mut classes = [0; 4];
        classes[class.index()] = bytes;
        Self {
            total: bytes,
            classes,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    committed: Counters,
    allocated: Counters,
    fail_closed: bool,
    next_scope: u64,
    scopes: BTreeMap<u64, ScopeCredit>,
}

#[derive(Debug)]
struct AccountInner {
    limits: HostMemoryLimits,
    allocation_alignment: usize,
    state: Mutex<State>,
}

/// Clone this authority across engines instead of creating per-engine totals.
#[derive(Debug, Clone)]
pub struct HostMemoryAccount(Arc<AccountInner>, Option<Arc<ScopeAuthority>>);

impl HostMemoryAccount {
    pub fn new(limits: HostMemoryLimits) -> Self {
        Self(
            Arc::new(AccountInner {
                limits,
                allocation_alignment: 1,
                state: Mutex::new(State::default()),
            }),
            None,
        )
    }

    /// Align payloads and pad their enclosing backing for native no-copy views.
    /// Every padding byte is reserved before allocation, in the same class and
    /// total account. This remains pageable memory; alignment is not pinning.
    pub fn with_allocation_alignment(
        limits: HostMemoryLimits,
        alignment: usize,
    ) -> Result<Self, HostMemoryError> {
        if !alignment.is_power_of_two() || alignment > isize::MAX as usize {
            return Err(HostMemoryError::InvalidReservation);
        }
        Ok(Self(
            Arc::new(AccountInner {
                limits,
                allocation_alignment: alignment,
                state: Mutex::new(State::default()),
            }),
            None,
        ))
    }

    pub fn allocation_alignment(&self) -> usize {
        self.0.allocation_alignment
    }

    /// Physical capacity required for a logical allocation, including alignment.
    /// This metadata calculation neither reserves nor allocates memory.
    pub fn backing_bytes(&self, bytes: usize) -> Result<usize, HostMemoryError> {
        if bytes == 0 || bytes > isize::MAX as usize {
            return Err(HostMemoryError::InvalidReservation);
        }
        let padding = self.0.allocation_alignment - 1;
        bytes
            .checked_add(padding)
            .map(|n| n & !padding)
            .and_then(|n| n.checked_add(padding))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(HostMemoryError::ArithmeticOverflow)
    }

    pub fn limits(&self) -> HostMemoryLimits {
        self.0.limits
    }

    /// Check provenance against the parent injected by product composition.
    /// Equal numeric limits do not make independently created accounts equal.
    pub fn owns(&self, bytes: &ManagedHostBytes) -> bool {
        Arc::ptr_eq(&self.0, &bytes.backing.charge.account.0)
    }

    /// Shared backing-budget and provenance identity, regardless of a view's
    /// scoped admission credit. Equal numeric limits do not establish ownership.
    pub fn same_parent(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn usage(&self) -> Result<HostMemoryUsage, HostMemoryError> {
        let state = self.0.state.lock().map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        Ok(HostMemoryUsage {
            committed: state.committed,
            allocated: state.allocated,
        })
    }

    pub fn reserve(
        &self,
        class: HostMemoryClass,
        bytes: usize,
    ) -> Result<HostMemoryReservation, HostMemoryError> {
        let mut reservations = self.reserve_many(&[(class, bytes)])?;
        Ok(reservations.remove(0))
    }

    /// Atomically reserve a whole set of independent backing allocations.
    /// Failure changes neither the parent nor any class counter. This is a
    /// non-waiting API: orchestration can reserve output before acquiring DMA
    /// slots instead of holding one scarce resource while waiting for another.
    pub fn reserve_many(
        &self,
        requests: &[(HostMemoryClass, usize)],
    ) -> Result<Vec<HostMemoryReservation>, HostMemoryError> {
        if requests.is_empty() {
            return Err(HostMemoryError::InvalidReservation);
        }
        let mut demand = Counters::default();
        for &(class, bytes) in requests {
            demand = demand
                .add(Counters::single(class, self.backing_bytes(bytes)?))
                .ok_or(HostMemoryError::ArithmeticOverflow)?;
        }
        // Allocate reservation bookkeeping before charging. These small control
        // objects, like Arc/Mutex metadata, are outside the backing-byte limit.
        let mut reservations = Vec::new();
        reservations
            .try_reserve_exact(requests.len())
            .map_err(|_| HostMemoryError::AllocationFailed {
                bytes: requests
                    .len()
                    .saturating_mul(std::mem::size_of::<HostMemoryReservation>()),
            })?;
        let mut state = self.0.state.lock().map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        let scoped = self.scoped_demand(&state, demand)?;
        let unreserved = demand.subtract(scoped).expect("scoped demand is a subset");
        let next = self.check_admission(&state, unreserved)?;
        self.consume_scope(&mut state, scoped);
        state.committed = next;
        // Capacity is already allocated, and these operations cannot fail.
        for &(class, bytes) in requests {
            reservations.push(HostMemoryReservation {
                logical_bytes: bytes,
                charge: Charge {
                    account: self.clone(),
                    class,
                    bytes: self
                        .backing_bytes(bytes)
                        .expect("validated before admission"),
                    allocated: false,
                },
            });
        }
        Ok(reservations)
    }

    pub fn allocate_zeroed(
        &self,
        class: HostMemoryClass,
        bytes: usize,
    ) -> Result<ManagedHostBuffer, HostMemoryError> {
        self.reserve(class, bytes)?.allocate_zeroed()
    }
}

/// A coherent snapshot; committed includes reservations and live allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryUsage {
    committed: Counters,
    allocated: Counters,
}

impl HostMemoryUsage {
    pub const fn committed_bytes(self) -> usize {
        self.committed.total
    }
    pub const fn allocated_bytes(self) -> usize {
        self.allocated.total
    }
    pub const fn reserved_bytes(self) -> usize {
        self.committed.total - self.allocated.total
    }
    pub const fn class_committed_bytes(self, class: HostMemoryClass) -> usize {
        self.committed.classes[class.index()]
    }
    pub const fn class_allocated_bytes(self, class: HostMemoryClass) -> usize {
        self.allocated.classes[class.index()]
    }
}

#[derive(Debug)]
struct Charge {
    account: HostMemoryAccount,
    class: HostMemoryClass,
    bytes: usize,
    allocated: bool,
}

impl Charge {
    fn mark_allocated(&mut self) -> Result<(), HostMemoryError> {
        let mut state = self
            .account
            .0
            .state
            .lock()
            .map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        let next = state
            .allocated
            .add(Counters::single(self.class, self.bytes))
            .ok_or(HostMemoryError::ArithmeticOverflow)?;
        if self.allocated
            || next.total > state.committed.total
            || next
                .classes
                .iter()
                .zip(state.committed.classes)
                .any(|(&allocated, committed)| allocated > committed)
        {
            state.fail_closed = true;
            return Err(HostMemoryError::Poisoned);
        }
        state.allocated = next;
        self.allocated = true;
        Ok(())
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        let Ok(mut state) = self.account.0.state.lock() else {
            return;
        };
        if state.fail_closed {
            return;
        }
        let amount = Counters::single(self.class, self.bytes);
        let allocated = if self.allocated {
            state.allocated.subtract(amount)
        } else {
            Some(state.allocated)
        };
        if self.account.refund_scope(&mut state, amount) {
            if let Some(allocated) = allocated {
                state.allocated = allocated;
            } else {
                state.fail_closed = true;
            }
            return;
        }
        let committed = state.committed.subtract(amount);
        match (committed, allocated) {
            (Some(committed), Some(allocated)) => {
                state.committed = committed;
                state.allocated = allocated;
            }
            _ => state.fail_closed = true,
        }
    }
}

/// Non-cloneable, unallocated capacity. Drop rolls it back; allocation consumes
/// it so callers cannot separate the returned backing from its charge.
#[derive(Debug)]
pub struct HostMemoryReservation {
    charge: Charge,
    logical_bytes: usize,
}

impl HostMemoryReservation {
    pub fn bytes(&self) -> usize {
        self.logical_bytes
    }
    /// Physical backing demand, including an account's explicit alignment.
    pub fn backing_bytes(&self) -> usize {
        self.charge.bytes
    }
    pub fn class(&self) -> HostMemoryClass {
        self.charge.class
    }

    /// Allocate and fully initialize an admitted backing from an exact source.
    /// No uninitialized buffer or byte slice is exposed, including on failure.
    /// The source and its owners remain independently charged while copying.
    pub(crate) fn allocate_copy(self, source: &[u8]) -> Result<ManagedHostBuffer, HostMemoryError> {
        let length = self.bytes();
        if length == 0 || source.len() != length {
            return Err(HostMemoryError::InvalidReservation);
        }
        self.allocate(Some(source))
    }

    pub fn allocate_zeroed(self) -> Result<ManagedHostBuffer, HostMemoryError> {
        self.allocate(None)
    }

    fn allocate(mut self, source: Option<&[u8]>) -> Result<ManagedHostBuffer, HostMemoryError> {
        let length = self.charge.bytes;
        let layout =
            Layout::array::<u8>(length).map_err(|_| HostMemoryError::InvalidReservation)?;
        // Padding is initialized too: a native no-copy view can include it.
        let zeroed = source.is_none() || self.charge.account.allocation_alignment() > 1;
        // SAFETY: the complete, nonzero allocation was admitted, including padding.
        let pointer = unsafe {
            if zeroed {
                std::alloc::alloc_zeroed(layout)
            } else {
                std::alloc::alloc(layout)
            }
        };
        if pointer.is_null() {
            return Err(HostMemoryError::AllocationFailed { bytes: length });
        }
        let alignment = self.charge.account.allocation_alignment();
        let offset = (pointer as usize).wrapping_neg() & (alignment - 1);
        let range = offset..offset + self.logical_bytes;
        if let Some(source) = source {
            // SAFETY: admission includes alignment-1 leading bytes plus the
            // rounded payload. The fresh destination cannot alias the source.
            unsafe {
                std::ptr::copy_nonoverlapping(source.as_ptr(), pointer.add(offset), source.len())
            };
        }
        // SAFETY: all bytes are initialized. Box uses the original u8 layout.
        let bytes = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pointer, length)) };
        self.charge.mark_allocated()?;
        Ok(ManagedHostBuffer {
            bytes,
            range,
            charge: self.charge,
        })
    }
}

/// Exclusively writable, already admitted pageable backing. Fields are ordered
/// so the actual bytes are freed before their capacity charge is returned.
pub struct ManagedHostBuffer {
    bytes: Box<[u8]>,
    range: Range<usize>,
    charge: Charge,
}

impl ManagedHostBuffer {
    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    pub fn class(&self) -> HostMemoryClass {
        self.charge.class
    }

    pub fn freeze(self) -> ManagedHostBytes {
        let range = self.range.clone();
        ManagedHostBytes {
            backing: Arc::new(self),
            range,
        }
    }
}

impl AsRef<[u8]> for ManagedHostBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.bytes[self.range.clone()]
    }
}
impl AsMut<[u8]> for ManagedHostBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[self.range.clone()]
    }
}

impl std::fmt::Debug for ManagedHostBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ManagedHostBuffer")
            .field("bytes", &self.len())
            .field("backing_bytes", &self.bytes.len())
            .field("class", &self.charge.class)
            .finish_non_exhaustive()
    }
}

/// Immutable shared backing suitable for retaining through transfers or output
/// bodies. Clones and slices retain one allocation and one physical charge.
#[derive(Clone)]
pub struct ManagedHostBytes {
    backing: Arc<ManagedHostBuffer>,
    range: Range<usize>,
}

impl ManagedHostBytes {
    /// Retains the full initialized backing, including accounted padding, while
    /// exposing only this logical slice as the payload. A backend must still
    /// validate its own native page and alignment requirements.
    pub fn retained_region(&self) -> Result<crate::RetainedHostMemoryRegion, crate::VNextError> {
        crate::RetainedHostMemoryRegion::new(
            Arc::new(FrozenHostAllocation(Arc::clone(&self.backing))),
            self.range.start,
            self.range.len(),
        )
    }
    pub fn bytes(&self) -> &[u8] {
        &self.backing.bytes[self.range.clone()]
    }
    pub fn len(&self) -> usize {
        self.range.len()
    }
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
    /// Physical capacity retained by this owner, including alignment padding.
    /// A clone or slice (even an empty slice) retains the entire allocation's
    /// charge; its logical `len()` is not the retained memory footprint.
    pub fn backing_bytes(&self) -> usize {
        self.backing.charge.bytes
    }
    pub fn class(&self) -> HostMemoryClass {
        self.backing.charge.class
    }

    /// Whether both handles retain exactly the same allocation and byte range.
    /// Equal contents or equal parent budgets do not establish this identity.
    /// This check neither scans bytes nor reveals an allocation address.
    pub fn same_region(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.backing, &other.backing) && self.range == other.range
    }

    pub fn slice(&self, range: Range<usize>) -> Result<Self, HostMemoryError> {
        if range.start > range.end || range.end > self.len() {
            return Err(HostMemoryError::InvalidSlice);
        }
        Ok(Self {
            backing: self.backing.clone(),
            range: self.range.start + range.start..self.range.start + range.end,
        })
    }
}

struct FrozenHostAllocation(Arc<ManagedHostBuffer>);

// SAFETY: constructed only after freeze. The Arc is private, never converted
// back into a writable buffer, and retains the full allocation and its charge.
unsafe impl crate::StableHostMemory for FrozenHostAllocation {
    fn stable_bytes(&self) -> &[u8] {
        &self.0.bytes
    }

    fn managed_memory_class(&self) -> Option<HostMemoryClass> {
        Some(self.0.class())
    }
}

impl std::fmt::Debug for ManagedHostBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ManagedHostBytes")
            .field("range", &self.range)
            .field("class", &self.class())
            .finish_non_exhaustive()
    }
}

impl AsRef<[u8]> for ManagedHostBytes {
    fn as_ref(&self) -> &[u8] {
        self.bytes()
    }
}

// SAFETY: freeze removes the sole mutable owner. Arc retains the byte allocation
// and its charge, and a view's private range never changes during its lifetime.
unsafe impl crate::StableHostMemory for ManagedHostBytes {
    fn stable_bytes(&self) -> &[u8] {
        self.bytes()
    }

    fn managed_memory_class(&self) -> Option<HostMemoryClass> {
        Some(self.class())
    }
}

#[cfg(test)]
mod tests;
