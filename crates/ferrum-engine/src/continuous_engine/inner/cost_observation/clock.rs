use super::*;
use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

mod system;

/// Independent cost clock. It never changes request ingress or the SLO clock.
pub(in crate::continuous_engine) struct EngineCostClock {
    source: ClockSource,
}
enum ClockSource {
    SameBoot(system::SystemCostClock),
    ProcessLocal(Instant),
}
impl Default for EngineCostClock {
    fn default() -> Self {
        Self::from_system(system::SystemCostClock::capture())
    }
}
impl EngineCostClock {
    fn from_system(system: Option<system::SystemCostClock>) -> Self {
        Self {
            source: match system {
                Some(clock) => ClockSource::SameBoot(clock),
                None => ClockSource::ProcessLocal(Instant::now()),
            },
        }
    }
}
fn process_ns_at(epoch: Instant, instant: Instant) -> Option<u64> {
    u64::try_from(instant.checked_duration_since(epoch)?.as_nanos()).ok()
}
impl CostObservationClock for EngineCostClock {
    fn now_ns(&self) -> Option<u64> {
        match &self.source {
            // A failed OS read cannot switch origin to the process-local
            // fallback: all existing observations use the selected clock.
            ClockSource::SameBoot(clock) => clock.now_ns(),
            ClockSource::ProcessLocal(epoch) => process_ns_at(*epoch, Instant::now()),
        }
    }
    fn monotonic_domain(&self) -> Option<&CostMonotonicDomainV1> {
        match &self.source {
            ClockSource::SameBoot(clock) => Some(clock.domain()),
            ClockSource::ProcessLocal(_) => None,
        }
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
        assert_eq!(process_ns_at(epoch, epoch), Some(0));
        assert_eq!(
            process_ns_at(epoch, epoch.checked_sub(Duration::from_nanos(1)).unwrap()),
            None
        );
        assert_eq!(
            process_ns_at(epoch, epoch + Duration::from_nanos(7)),
            Some(7)
        );
    }

    #[test]
    fn unavailable_system_identity_preserves_local_clock_without_reuse_domain() {
        let clock = EngineCostClock::from_system(None);
        assert!(clock.monotonic_domain().is_none());
        let first = clock.now_ns().unwrap();
        assert!(clock.now_ns().unwrap() >= first);
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
