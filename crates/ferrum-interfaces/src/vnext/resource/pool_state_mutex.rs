//! Pool-local invalidation accompanies every writable state borrow, including
//! rollback and poison recovery. Stamps are observations, not read-side locks.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LockResult, Mutex, MutexGuard, PoisonError};

const DISABLED: u64 = u64::MAX;

pub(super) struct PoolStateMutex<T> {
    raw: Mutex<T>,
    version: AtomicU64,
}

pub(super) struct PoolStateGuard<'a, T> {
    raw: MutexGuard<'a, T>,
    owner: &'a PoolStateMutex<T>,
    dirty: bool,
}

impl<T: std::fmt::Debug> std::fmt::Debug for PoolStateGuard<'_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.raw.fmt(formatter)
    }
}

/// The borrowed mutex is the provenance; a matching numeric stamp alone never
/// authorizes another pool. This carries no resource authority of its own.
pub(super) struct PoolReadStamp<'a, T> {
    owner: &'a PoolStateMutex<T>,
    version: u64,
}

impl<T> Copy for PoolReadStamp<'_, T> {}
impl<T> Clone for PoolReadStamp<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PoolStateMutex<T> {
    pub(super) fn new(value: T) -> Self {
        Self {
            raw: Mutex::new(value),
            version: AtomicU64::new(0),
        }
    }

    pub(super) fn lock(&self) -> LockResult<PoolStateGuard<'_, T>> {
        match self.raw.lock() {
            Ok(raw) => Ok(PoolStateGuard {
                raw,
                owner: self,
                dirty: false,
            }),
            Err(error) => {
                self.version.store(DISABLED, Ordering::Release);
                Err(PoisonError::new(PoolStateGuard {
                    raw: error.into_inner(),
                    owner: self,
                    dirty: false,
                }))
            }
        }
    }
}

impl<'a, T> PoolStateGuard<'a, T> {
    /// Call only after the protected predicates succeed. A writer never mints
    /// a proof even if its mutations eventually restore identical state.
    pub(super) fn read_stamp(&self) -> Option<PoolReadStamp<'a, T>> {
        let version = self.owner.version.load(Ordering::Acquire);
        (!self.dirty && version != DISABLED && version % 2 == 0).then_some(PoolReadStamp {
            owner: self.owner,
            version,
        })
    }
}

impl<T> PoolReadStamp<'_, T> {
    pub(super) fn matches(&self, owner: &PoolStateMutex<T>) -> bool {
        std::ptr::eq(self.owner, owner) && owner.version.load(Ordering::Acquire) == self.version
    }

    #[cfg(test)]
    fn matches_guard(&self, guard: &PoolStateGuard<'_, T>) -> bool {
        !guard.dirty && self.matches(guard.owner)
    }
}

impl<T> Deref for PoolStateGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

impl<T> DerefMut for PoolStateGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if !self.dirty {
            let version = self.owner.version.load(Ordering::Relaxed);
            // The lock serializes writers. Exhaustion is permanent: an old
            // retained proof must never become current after an ABA wrap.
            let next = version
                .checked_add(2)
                .filter(|next| *next < DISABLED)
                .map_or(DISABLED, |_| version + 1);
            self.owner.version.store(next, Ordering::Release);
            self.dirty = true;
        }
        &mut self.raw
    }
}

impl<T> Drop for PoolStateGuard<'_, T> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // This runs before raw MutexGuard::drop publishes std poison and
            // unlocks, including a panic while holding only a read borrow.
            self.owner.version.store(DISABLED, Ordering::Release);
        } else if self.dirty {
            let version = self.owner.version.load(Ordering::Relaxed);
            if version != DISABLED {
                self.owner.version.store(version + 1, Ordering::Release);
            }
        }
    }
}

#[cfg(test)]
mod tests;
