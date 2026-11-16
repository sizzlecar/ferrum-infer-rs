//! Bounded static CUDART module ownership, testable without CUDA. All modules
//! charge the runtime's existing device account. No independent memory ledger.
use ferrum_runtime::{
    DeviceCapacityOwner, DeviceCapacityRetention, DeviceOpaqueMemoryKind, VNextError,
};
use std::sync::{Arc, Mutex, Weak};

/// Sharing is part of the installed policy; an exclusive admission cannot be
/// converted into a shared one while its CUDART module remains installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StaticNativeModuleSharing {
    Exclusive,
    Shared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StaticNativeModule {
    CutlassFp8,
    CutlassNvfp4,
    CutlassFusedF32Attention,
    CutlassInt8FusedOutput,
    #[cfg(feature = "cuda-nvfp4-diagnostics")]
    Nvfp4QualificationReference,
    SageVsaFp4,
}
impl StaticNativeModule {
    const COUNT: usize = 6;
    const fn index(self) -> usize {
        match self {
            Self::CutlassFp8 => 0,
            Self::CutlassNvfp4 => 1,
            Self::CutlassFusedF32Attention => 2,
            Self::CutlassInt8FusedOutput => 3,
            #[cfg(feature = "cuda-nvfp4-diagnostics")]
            Self::Nvfp4QualificationReference => 4,
            Self::SageVsaFp4 => 5,
        }
    }
}
struct InstalledModule {
    identity: String,
    capacity: Option<DeviceCapacityRetention>,
    sharing: StaticNativeModuleSharing,
    live: Weak<LiveModule>,
}
impl Drop for InstalledModule {
    fn drop(&mut self) {
        // No explicit fatbin unload/primary-context destruction receipt exists
        // at this boundary. Dropping cudarc's wrapper cannot reclaim this claim.
        if let Some(capacity) = self.capacity.take() {
            std::mem::forget(capacity);
        }
    }
}
pub(crate) struct StaticNativeModuleCapacity {
    slots: Mutex<[Option<InstalledModule>; StaticNativeModule::COUNT]>,
}
impl Default for StaticNativeModuleCapacity {
    fn default() -> Self {
        Self {
            slots: Mutex::new(std::array::from_fn(|_| None)),
        }
    }
}
pub(crate) struct StaticNativeModuleLease {
    _live: Arc<LiveModule>,
}
struct LiveModule;
#[derive(Debug)]
pub(crate) enum StaticNativeModuleCapacityError {
    Contract(&'static str),
    Capacity(VNextError),
}
impl From<&'static str> for StaticNativeModuleCapacityError {
    fn from(reason: &'static str) -> Self {
        Self::Contract(reason)
    }
}
impl StaticNativeModuleCapacity {
    #[cfg(any(test, feature = "cuda-nvfp4-diagnostics"))]
    pub(crate) fn claim(
        &self,
        capacity: &DeviceCapacityOwner,
        module: StaticNativeModule,
        identity: &str,
        bytes: u64,
    ) -> Result<StaticNativeModuleLease, StaticNativeModuleCapacityError> {
        self.claim_with_sharing(
            capacity,
            module,
            identity,
            bytes,
            StaticNativeModuleSharing::Exclusive,
        )
    }

    pub(crate) fn claim_with_sharing(
        &self,
        capacity: &DeviceCapacityOwner,
        module: StaticNativeModule,
        identity: &str,
        bytes: u64,
        sharing: StaticNativeModuleSharing,
    ) -> Result<StaticNativeModuleLease, StaticNativeModuleCapacityError> {
        if bytes == 0
            || identity.is_empty()
            || identity.len() > 256
            || !identity.is_ascii()
            || identity.bytes().any(|b| b.is_ascii_control())
        {
            return Err(
                "static native module requires bounded identity and positive allowance".into(),
            );
        }
        let mut slots = self
            .slots
            .lock()
            .map_err(|_| "static native module owner poisoned")?;
        let slot = &mut slots[module.index()];
        if let Some(installed) = slot {
            let retained = installed
                .capacity
                .as_ref()
                .ok_or("static module capacity absent")?;
            if installed.identity != identity
                || retained.bytes() != bytes
                || !capacity.owns(retained)
                || installed.sharing != sharing
            {
                return Err(
                    "static native module policy or budget owner differs from installed policy"
                        .into(),
                );
            }
            if let Some(live) = installed.live.upgrade() {
                return match sharing {
                    StaticNativeModuleSharing::Exclusive => {
                        Err("static native module already has a live provider".into())
                    }
                    StaticNativeModuleSharing::Shared => {
                        Ok(StaticNativeModuleLease { _live: live })
                    }
                };
            }
            let live = Arc::new(LiveModule);
            installed.live = Arc::downgrade(&live);
            return Ok(StaticNativeModuleLease { _live: live });
        }
        let retained = capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, bytes)
            .map_err(StaticNativeModuleCapacityError::Capacity)?;
        let live = Arc::new(LiveModule);
        *slot = Some(InstalledModule {
            identity: identity.into(),
            capacity: Some(retained),
            sharing,
            live: Arc::downgrade(&live),
        });
        Ok(StaticNativeModuleLease { _live: live })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{
        DeviceClass, DeviceDescriptor, DeviceId, DynamicStorageAllocator, DynamicStorageProfile,
        DynamicStorageView,
    };
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicU64, Ordering};
    fn account() -> (DeviceDescriptor, DeviceCapacityOwner) {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let descriptor = DeviceDescriptor {
            id: DeviceId::new(format!(
                "device.test.native-module-{}",
                NEXT.fetch_add(1, Ordering::Relaxed)
            ))
            .unwrap(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: 128,
            runtime_implementation_fingerprint: "c".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        };
        let owner = DeviceCapacityOwner::open(&descriptor, 128).unwrap();
        (descriptor, owner)
    }
    #[test]
    fn same_module_reopens_without_recharge_and_rejects_live_or_policy_conflicts() {
        let (descriptor, owner) = account();
        let slots = StaticNativeModuleCapacity::default();
        let lease = slots
            .claim(&owner, StaticNativeModule::CutlassFp8, "fp8.v1", 80)
            .unwrap();
        assert!(slots
            .claim(&owner, StaticNativeModule::CutlassFp8, "fp8.v1", 80)
            .is_err());
        drop(lease);
        assert!(slots
            .claim(&owner, StaticNativeModule::CutlassFp8, "changed", 80)
            .is_err());
        assert!(slots
            .claim(&owner, StaticNativeModule::CutlassFp8, "fp8.v1", 81)
            .is_err());
        let other_owner = DeviceCapacityOwner::open(&descriptor, 128).unwrap();
        assert!(slots
            .claim(&other_owner, StaticNativeModule::CutlassFp8, "fp8.v1", 80)
            .is_err());
        let reopened = slots
            .claim(&owner, StaticNativeModule::CutlassFp8, "fp8.v1", 80)
            .unwrap();
        let remaining = owner.reserve(DeviceOpaqueMemoryKind::Stream, 48).unwrap();
        assert!(owner.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        drop(remaining);
        drop(reopened);
        drop(slots);
        // The module's lack of an unload receipt continues to charge 80 bytes.
        assert!(owner.reserve(DeviceOpaqueMemoryKind::Stream, 49).is_err());
    }
    #[test]
    fn shared_module_retains_one_charge_and_last_live_lease_without_policy_changes() {
        let (descriptor, owner) = account();
        let slots = StaticNativeModuleCapacity::default();
        let shared = |account: &DeviceCapacityOwner, identity, bytes| {
            slots.claim_with_sharing(
                account,
                StaticNativeModule::SageVsaFp4,
                identity,
                bytes,
                StaticNativeModuleSharing::Shared,
            )
        };
        let first = shared(&owner, "sage.v1", 80).unwrap();
        let second = shared(&owner, "sage.v1", 80).unwrap();
        assert!(Arc::ptr_eq(&first._live, &second._live));
        assert!(slots
            .claim(&owner, StaticNativeModule::SageVsaFp4, "sage.v1", 80)
            .is_err());
        assert!(shared(&owner, "different", 80).is_err());
        assert!(shared(&owner, "sage.v1", 81).is_err());
        let other = DeviceCapacityOwner::open(&descriptor, 128).unwrap();
        assert!(shared(&other, "sage.v1", 80).is_err());
        let observed = Arc::downgrade(&first._live);
        drop(first);
        assert!(observed.upgrade().is_some());
        let third = shared(&owner, "sage.v1", 80).unwrap();
        assert!(Arc::ptr_eq(&second._live, &third._live));
        drop(second);
        drop(third);
        assert!(observed.upgrade().is_none());
        let reopened = shared(&owner, "sage.v1", 80).unwrap();
        assert!(observed.upgrade().is_none());
        let remaining = owner.reserve(DeviceOpaqueMemoryKind::Stream, 48).unwrap();
        assert!(owner.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        drop(remaining);
        drop(reopened);
        drop(slots);
        assert!(owner.reserve(DeviceOpaqueMemoryKind::Stream, 49).is_err());
    }
    #[test]
    fn installed_exclusive_module_cannot_be_reopened_with_shared_policy() {
        let (_, owner) = account();
        let slots = StaticNativeModuleCapacity::default();
        let original = slots
            .claim(&owner, StaticNativeModule::CutlassNvfp4, "nvfp4.v1", 80)
            .unwrap();
        let share = || {
            slots.claim_with_sharing(
                &owner,
                StaticNativeModule::CutlassNvfp4,
                "nvfp4.v1",
                80,
                StaticNativeModuleSharing::Shared,
            )
        };
        assert!(share().is_err());
        drop(original);
        assert!(share().is_err());
        assert!(slots
            .claim(&owner, StaticNativeModule::CutlassNvfp4, "nvfp4.v1", 80)
            .is_ok());
    }
    #[test]
    fn distinct_modules_share_capacity_and_failed_admission_leaves_slot_reusable() {
        let (_, owner) = account();
        let slots = StaticNativeModuleCapacity::default();
        let _fp8 = slots
            .claim(&owner, StaticNativeModule::CutlassFp8, "shared-name", 64)
            .unwrap();
        let temporary = owner.reserve(DeviceOpaqueMemoryKind::Stream, 32).unwrap();
        assert!(slots
            .claim(&owner, StaticNativeModule::CutlassNvfp4, "shared-name", 64)
            .is_err());
        drop(temporary);
        let _nvfp4 = slots
            .claim(&owner, StaticNativeModule::CutlassNvfp4, "shared-name", 64)
            .unwrap();
        assert!(slots
            .claim(
                &owner,
                StaticNativeModule::CutlassFusedF32Attention,
                "f32",
                1
            )
            .is_err());
        assert!(slots
            .claim(
                &owner,
                StaticNativeModule::CutlassInt8FusedOutput,
                "int8",
                1
            )
            .is_err());
        assert!(slots
            .claim(&owner, StaticNativeModule::CutlassInt8FusedOutput, "", 0)
            .is_err());
    }
}
