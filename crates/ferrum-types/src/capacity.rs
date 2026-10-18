//! Process-local identities for observations of actual allocator capacity.

use std::{
    num::NonZeroU64,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// Identity of one actual allocator. Its owner creates this once; observations
/// clone it. It owns no blocks and grants no allocation or release authority.
/// Diagnostic serialization cannot be used to reconstruct this identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocatorCapacityDomain(Arc<AllocatorCapacityIdentity>);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AllocatorCapacityIdentity {
    diagnostic_id: NonZeroU64,
}

impl AllocatorCapacityDomain {
    pub fn new() -> Self {
        static NEXT_DOMAIN: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_DOMAIN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("allocator capacity domain identity space exhausted");
        Self(Arc::new(AllocatorCapacityIdentity {
            diagnostic_id: NonZeroU64::new(id).expect("allocator capacity domain is nonzero"),
        }))
    }
}

impl Default for AllocatorCapacityDomain {
    fn default() -> Self {
        Self::new()
    }
}

impl serde::Serialize for AllocatorCapacityDomain {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0.diagnostic_id.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn cloned_domain_keeps_identity_and_distinct_allocators_do_not_alias() {
        let first = AllocatorCapacityDomain::new();
        let retained = first.clone();
        let second = AllocatorCapacityDomain::new();
        assert_eq!(first, retained);
        assert_ne!(first, second);
        assert_eq!(
            HashSet::from([first.clone(), retained.clone(), second.clone()]).len(),
            2
        );
        assert_eq!(
            BTreeSet::from([first.clone(), retained.clone(), second.clone()]).len(),
            2
        );
        assert_ne!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        drop(first);
        assert_ne!(retained, second);
    }
}
