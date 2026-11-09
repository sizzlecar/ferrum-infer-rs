//! Native registration stays with the original accounted allocation, including
//! when a reusable streaming slot is thawed after proven device completion.

use super::*;
use std::any::TypeId;

/// Backend ownership attached to an admitted Host allocation.
///
/// # Safety
/// The registration must concern only the allocation supplied to its factory.
/// It must not retain the same `ManagedHostBytes` (which would create a cycle),
/// expose mutation, or permit native access after the last allocation owner is
/// released. Commands must retain the allocation through their original native
/// completion/recovery owner. `release` may return true only after all native
/// registration ownership is retired; it must be idempotent and must not panic.
/// A false result retains the registration, payload and capacity indefinitely.
pub unsafe trait NativeHostMemoryRegistration: Send + Sync + 'static {
    fn release(&mut self) -> bool;
}

pub(super) struct Registration {
    kind: TypeId,
    key: u64,
    native: Box<dyn NativeHostMemoryRegistration>,
}

impl ManagedHostBytes {
    /// Register this original allocation once for a backend-specific key.
    /// Slices share the registration; the factory receives the complete physical
    /// backing and original logical range, never only the caller's slice.
    ///
    /// The factory runs while registration is serialized. It must undo native
    /// changes before returning an error. Keys must distinguish registrations
    /// that cannot be shared. No extra Host allocation or capacity is granted.
    pub fn try_register_native<T, E>(
        &self,
        key: u64,
        register: impl FnOnce(&[u8], Range<usize>) -> Result<T, E>,
    ) -> Result<(), E>
    where
        T: NativeHostMemoryRegistration,
    {
        let mut registrations = self
            .backing
            .registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let kind = TypeId::of::<T>();
        if registrations.iter().any(|r| r.kind == kind && r.key == key) {
            return Ok(());
        }
        let native = register(&self.backing.bytes, self.backing.range.clone())?;
        registrations.push(Registration {
            kind,
            key,
            native: Box::new(native),
        });
        Ok(())
    }
}

impl Drop for ManagedHostBuffer {
    fn drop(&mut self) {
        let registrations = self
            .registrations
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for registration in registrations {
            if !registration.native.release() {
                // Native retirement is unproven. Keep the exact allocation,
                // its quota and all registration/context owners alive. Closing
                // admission makes this visible even after the last view drops.
                self.charge
                    .account
                    .0
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .fail_closed = true;
                self.charge.account.0.capacity_changed.notify();
                tracing::error!(
                    physical_bytes = self.bytes.len(),
                    class = ?self.charge.class,
                    "Host native registration retirement failed; allocation and quota retained"
                );
                return;
            }
        }
        // SAFETY: all native registrations retired successfully and this is the
        // final allocation owner. Retire contexts, then bytes, then their quota.
        unsafe {
            ManuallyDrop::drop(&mut self.registrations);
            ManuallyDrop::drop(&mut self.bytes);
            ManuallyDrop::drop(&mut self.charge);
        }
    }
}

#[cfg(test)]
mod tests;
