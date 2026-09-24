use super::*;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

/// Independent cost clock. It never changes request ingress or the SLO clock.
pub(in crate::continuous_engine) struct EngineCostClock {
    epoch: Instant,
}
impl Default for EngineCostClock {
    fn default() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }
}
impl EngineCostClock {
    pub fn at_ns(&self, instant: Instant) -> Option<u64> {
        u64::try_from(instant.checked_duration_since(self.epoch)?.as_nanos()).ok()
    }
}
impl CostObservationClock for EngineCostClock {
    fn now_ns(&self) -> Option<u64> {
        self.at_ns(Instant::now())
    }
}

/// Local evidence IDs only. Exhaustion is sticky and cannot wrap into another
/// request's incarnation. No ID here authorizes device or scheduler work.
pub(in crate::continuous_engine) struct EngineCostIds {
    next_call: AtomicU64,
    next_incarnation: AtomicU64,
}
impl Default for EngineCostIds {
    fn default() -> Self {
        Self {
            next_call: AtomicU64::new(1),
            next_incarnation: AtomicU64::new(1),
        }
    }
}
impl EngineCostIds {
    fn take(counter: &AtomicU64) -> Result<NonZeroU64, CostCallRejection> {
        let value = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| CostCallRejection::IdExhausted)?;
        NonZeroU64::new(value).ok_or(CostCallRejection::IdExhausted)
    }
    pub fn next_call(&self) -> Result<NonZeroU64, CostCallRejection> {
        Self::take(&self.next_call)
    }
    pub fn new_frontier(&self) -> Result<CostFrontier, CostCallRejection> {
        Ok(CostFrontier {
            owner_incarnation: Self::take(&self.next_incarnation)?,
            work_generation: NonZeroU64::MIN,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine) struct CostFrontier {
    pub owner_incarnation: NonZeroU64,
    pub work_generation: NonZeroU64,
}
impl CostFrontier {
    /// Call after every actual state/frontier transition, including restore and
    /// recomputation. On error the owner must disable this evidence frontier;
    /// keeping the old generation would permit an ABA match.
    pub fn advanced(self) -> Result<Self, CostCallRejection> {
        Ok(Self {
            work_generation: self
                .work_generation
                .checked_add(1)
                .ok_or(CostCallRejection::IdExhausted)?,
            ..self
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn clock_rejects_pre_epoch_ingress_instead_of_resetting_it() {
        let epoch = Instant::now();
        let clock = EngineCostClock { epoch };
        assert_eq!(clock.at_ns(epoch), Some(0));
        assert_eq!(
            clock.at_ns(epoch.checked_sub(Duration::from_nanos(1)).unwrap()),
            None
        );
        assert_eq!(clock.at_ns(epoch + Duration::from_nanos(7)), Some(7));
    }

    #[test]
    fn identifiers_and_generations_exhaust_without_wrapping() {
        let ids = EngineCostIds {
            next_call: AtomicU64::new(u64::MAX - 1),
            next_incarnation: AtomicU64::new(u64::MAX),
        };
        assert_eq!(ids.next_call().unwrap().get(), u64::MAX - 1);
        assert_eq!(ids.next_call(), Err(CostCallRejection::IdExhausted));
        assert_eq!(ids.next_call(), Err(CostCallRejection::IdExhausted));
        assert_eq!(ids.new_frontier(), Err(CostCallRejection::IdExhausted));
        let frontier = CostFrontier {
            owner_incarnation: NonZeroU64::MIN,
            work_generation: NonZeroU64::new(u64::MAX).unwrap(),
        };
        assert_eq!(frontier.advanced(), Err(CostCallRejection::IdExhausted));
    }
}
