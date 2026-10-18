use ferrum_types::Result;
use serde::{Deserialize, Serialize};

/// Point-in-time memory evidence emitted by the shared plan runtime.
///
/// Static model allocations are separated from dynamic request resources so
/// product telemetry never reports model weights as KV or recurrent-state
/// usage. Process-wide claims are included because another live plan can
/// consume capacity visible to this runtime. Dynamic free bytes remain
/// reusable by this plan; quarantined and other claimed bytes do not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanRuntimeResourceSnapshot {
    device_capacity_bytes: u64,
    usable_capacity_bytes: u64,
    process_claimed_bytes: u64,
    plan_claimed_bytes: u64,
    static_bytes: u64,
    dynamic_resident_bytes: u64,
    dynamic_free_bytes: u64,
    pending_growth_bytes: u64,
    quarantined_bytes: u64,
}

impl PlanRuntimeResourceSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_capacity_bytes: u64,
        usable_capacity_bytes: u64,
        process_claimed_bytes: u64,
        plan_claimed_bytes: u64,
        static_bytes: u64,
        dynamic_resident_bytes: u64,
        dynamic_free_bytes: u64,
        pending_growth_bytes: u64,
        quarantined_bytes: u64,
    ) -> Result<Self> {
        let snapshot = Self {
            device_capacity_bytes,
            usable_capacity_bytes,
            process_claimed_bytes,
            plan_claimed_bytes,
            static_bytes,
            dynamic_resident_bytes,
            dynamic_free_bytes,
            pending_growth_bytes,
            quarantined_bytes,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// Revalidates deserialized evidence before it crosses a trusted failure or
    /// profile boundary.
    pub fn validate(&self) -> Result<()> {
        if self.usable_capacity_bytes > self.device_capacity_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime usable capacity {} exceeds device capacity {}",
                self.usable_capacity_bytes, self.device_capacity_bytes
            )));
        }
        if self.process_claimed_bytes > self.usable_capacity_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime process claims {} exceed usable capacity {}",
                self.process_claimed_bytes, self.usable_capacity_bytes
            )));
        }
        if self.plan_claimed_bytes > self.process_claimed_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime plan claims {} exceed process claims {}",
                self.plan_claimed_bytes, self.process_claimed_bytes
            )));
        }
        if self.dynamic_free_bytes > self.dynamic_resident_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime dynamic free bytes {} exceed resident bytes {}",
                self.dynamic_free_bytes, self.dynamic_resident_bytes
            )));
        }
        let minimum_plan_claim = self
            .static_bytes
            .checked_add(self.dynamic_resident_bytes)
            .and_then(|bytes| bytes.checked_add(self.quarantined_bytes))
            .ok_or_else(|| {
                ferrum_types::FerrumError::internal(
                    "plan runtime static, resident, and quarantined bytes overflow u64",
                )
            })?;
        if minimum_plan_claim > self.plan_claimed_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime accounted plan bytes {minimum_plan_claim} exceed plan claims {}",
                self.plan_claimed_bytes
            )));
        }
        Ok(())
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn usable_capacity_bytes(&self) -> u64 {
        self.usable_capacity_bytes
    }

    pub const fn process_claimed_bytes(&self) -> u64 {
        self.process_claimed_bytes
    }

    pub const fn plan_claimed_bytes(&self) -> u64 {
        self.plan_claimed_bytes
    }

    pub const fn static_bytes(&self) -> u64 {
        self.static_bytes
    }

    pub const fn dynamic_resident_bytes(&self) -> u64 {
        self.dynamic_resident_bytes
    }

    pub const fn dynamic_free_bytes(&self) -> u64 {
        self.dynamic_free_bytes
    }

    pub const fn dynamic_used_bytes(&self) -> u64 {
        self.dynamic_resident_bytes - self.dynamic_free_bytes
    }

    pub const fn pending_growth_bytes(&self) -> u64 {
        self.pending_growth_bytes
    }

    pub const fn quarantined_bytes(&self) -> u64 {
        self.quarantined_bytes
    }

    /// Capacity immediately reusable by this plan without reclaiming another
    /// plan: process-wide unclaimed bytes plus free extents already resident
    /// in this plan's dynamic pools.
    pub fn available_bytes(&self) -> Result<u64> {
        self.usable_capacity_bytes
            .checked_sub(self.process_claimed_bytes)
            .and_then(|bytes| bytes.checked_add(self.dynamic_free_bytes))
            .ok_or_else(|| {
                ferrum_types::FerrumError::internal(
                    "plan runtime available capacity calculation overflowed",
                )
            })
    }

    pub fn used_bytes(&self) -> Result<u64> {
        self.available_bytes().and_then(|available| {
            self.usable_capacity_bytes
                .checked_sub(available)
                .ok_or_else(|| {
                    ferrum_types::FerrumError::internal(
                        "plan runtime available bytes exceed usable capacity",
                    )
                })
        })
    }
}

#[cfg(test)]
mod plan_runtime_resource_snapshot_tests {
    use super::PlanRuntimeResourceSnapshot;

    #[test]
    fn separates_static_and_dynamic_usage() {
        let snapshot =
            PlanRuntimeResourceSnapshot::new(1_000, 900, 710, 710, 400, 300, 200, 20, 10).unwrap();

        assert_eq!(snapshot.available_bytes().unwrap(), 390);
        assert_eq!(snapshot.used_bytes().unwrap(), 510);
        assert_eq!(snapshot.dynamic_resident_bytes(), 300);
        assert_eq!(snapshot.dynamic_used_bytes(), 100);
        assert_eq!(snapshot.dynamic_free_bytes(), 200);
        assert_eq!(snapshot.pending_growth_bytes(), 20);
        assert_eq!(snapshot.quarantined_bytes(), 10);
    }

    #[test]
    fn rejects_incoherent_capacity_evidence() {
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 1_001, 0, 0, 0, 0, 0, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 901, 0, 0, 0, 0, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 500, 501, 0, 0, 0, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 100, 100, 0, 100, 101, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 500, 500, 400, 100, 0, 0, 1).is_err());
    }
}
