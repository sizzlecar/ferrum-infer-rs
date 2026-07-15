//! Capacity-driven waiting admission for the vNext execution runtime.
//!
//! This queue owns policy only. Physical and logical capacity remain owned by
//! `ferrum-interfaces`; callers probe those authorities and feed the typed
//! result back here. A deferred request stays queued and is not probed again
//! until release, capacity, or product-policy evidence changes.

use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroU64;

use ferrum_interfaces::vnext::{
    AdmissionDeferred, CapacityEpochs, DeferredAction, DynamicBackingDeferred,
};

/// Typed product policy for waiting-request bypass fairness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicAdmissionQueuePolicy {
    max_bypass_release_epochs: u64,
}

impl DynamicAdmissionQueuePolicy {
    pub fn new(max_bypass_release_epochs: u64) -> Result<Self, DynamicAdmissionQueueError> {
        if max_bypass_release_epochs == 0 {
            return Err(DynamicAdmissionQueueError::InvalidPolicy(
                "max_bypass_release_epochs must be non-zero",
            ));
        }
        Ok(Self {
            max_bypass_release_epochs,
        })
    }

    pub const fn max_bypass_release_epochs(self) -> u64 {
        self.max_bypass_release_epochs
    }
}

impl Default for DynamicAdmissionQueuePolicy {
    fn default() -> Self {
        Self {
            max_bypass_release_epochs: 8,
        }
    }
}

/// Exact wake evidence observed by one plan-local scheduler queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionWakeEpochs {
    coordinator_id: NonZeroU64,
    release_epoch: u64,
    capacity_epoch: u64,
    policy_epoch: u64,
}

impl AdmissionWakeEpochs {
    pub const fn new(
        coordinator_id: NonZeroU64,
        release_epoch: u64,
        capacity_epoch: u64,
        policy_epoch: u64,
    ) -> Self {
        Self {
            coordinator_id,
            release_epoch,
            capacity_epoch,
            policy_epoch,
        }
    }

    pub fn from_capacity(epochs: CapacityEpochs, policy_epoch: u64) -> Self {
        Self {
            coordinator_id: NonZeroU64::new(epochs.coordinator_id().get())
                .expect("core-issued admission coordinator ids are non-zero"),
            release_epoch: epochs.release_epoch(),
            capacity_epoch: epochs.capacity_epoch(),
            policy_epoch,
        }
    }

    pub const fn coordinator_id(self) -> NonZeroU64 {
        self.coordinator_id
    }

    pub const fn release_epoch(self) -> u64 {
        self.release_epoch
    }

    pub const fn capacity_epoch(self) -> u64 {
        self.capacity_epoch
    }

    pub const fn policy_epoch(self) -> u64 {
        self.policy_epoch
    }

    fn changed_since(self, observed: Self) -> bool {
        self.release_epoch > observed.release_epoch
            || self.capacity_epoch > observed.capacity_epoch
            || self.policy_epoch > observed.policy_epoch
    }

    fn is_monotonic_after(self, prior: Self) -> bool {
        self.coordinator_id == prior.coordinator_id
            && self.release_epoch >= prior.release_epoch
            && self.capacity_epoch >= prior.capacity_epoch
            && self.policy_epoch >= prior.policy_epoch
    }
}

/// Scheduler-facing deferral evidence. It contains no capacity authority and
/// cannot manufacture an admission; it only suppresses blind retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionDeferral {
    action: DeferredAction,
    observed: AdmissionWakeEpochs,
}

impl AdmissionDeferral {
    pub const fn new(action: DeferredAction, observed: AdmissionWakeEpochs) -> Self {
        Self { action, observed }
    }

    pub fn from_admission(deferred: &AdmissionDeferred, policy_epoch: u64) -> Self {
        Self::new(
            deferred.action(),
            AdmissionWakeEpochs::from_capacity(deferred.epochs(), policy_epoch),
        )
    }

    pub fn from_backing(deferred: &DynamicBackingDeferred, policy_epoch: u64) -> Self {
        Self::new(
            DeferredAction::AwaitBackingGrowth,
            AdmissionWakeEpochs::from_capacity(deferred.epochs(), policy_epoch),
        )
    }

    pub const fn action(self) -> DeferredAction {
        self.action
    }

    pub const fn observed(self) -> AdmissionWakeEpochs {
        self.observed
    }
}

/// One plan-local stable waiting identity. It is not a request, sequence, or
/// capacity authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaitingAdmissionTicket(NonZeroU64);

impl WaitingAdmissionTicket {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Result of probing the real vNext resource authority for one waiting item.
pub enum AdmissionProbeOutcome<A, R, E> {
    Admitted(A),
    Deferred(AdmissionDeferral),
    PermanentRejected(R),
    Faulted(E),
}

/// Ownership changes emitted by one bounded scheduling tick.
pub enum AdmissionQueueEvent<T, A, R, E> {
    Admitted {
        ticket: WaitingAdmissionTicket,
        request: T,
        admission: A,
    },
    PermanentRejected {
        ticket: WaitingAdmissionTicket,
        request: T,
        rejection: R,
    },
    Faulted {
        ticket: WaitingAdmissionTicket,
        request: T,
        error: E,
    },
    /// The request remains queued. The runtime may choose a costed victim set,
    /// then advance policy/capacity evidence and probe again.
    PreemptionRequested {
        ticket: WaitingAdmissionTicket,
        deferral: AdmissionDeferral,
    },
    /// The request remains queued while the executor performs one bounded
    /// backing-maintenance attempt outside the queue lock.
    BackingGrowthRequested {
        ticket: WaitingAdmissionTicket,
        deferral: AdmissionDeferral,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionTickReceipt {
    probed: usize,
    skipped_unchanged: usize,
    admitted: usize,
    deferred: usize,
    permanent_rejected: usize,
    faulted: usize,
    preemption_requested: usize,
    backing_growth_requested: usize,
    waiting_after: usize,
    fairness_barrier: Option<WaitingAdmissionTicket>,
}

impl AdmissionTickReceipt {
    pub const fn probed(self) -> usize {
        self.probed
    }

    pub const fn skipped_unchanged(self) -> usize {
        self.skipped_unchanged
    }

    pub const fn admitted(self) -> usize {
        self.admitted
    }

    pub const fn deferred(self) -> usize {
        self.deferred
    }

    pub const fn permanent_rejected(self) -> usize {
        self.permanent_rejected
    }

    pub const fn faulted(self) -> usize {
        self.faulted
    }

    pub const fn preemption_requested(self) -> usize {
        self.preemption_requested
    }

    pub const fn backing_growth_requested(self) -> usize {
        self.backing_growth_requested
    }

    pub const fn waiting_after(self) -> usize {
        self.waiting_after
    }

    pub const fn fairness_barrier(self) -> Option<WaitingAdmissionTicket> {
        self.fairness_barrier
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicAdmissionQueueError {
    InvalidPolicy(&'static str),
    IdentityExhausted,
    DuplicateTicket(WaitingAdmissionTicket),
    ForeignCoordinator { expected: u64, actual: u64 },
    EpochRegression,
}

impl fmt::Display for DynamicAdmissionQueueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy(reason) => write!(formatter, "invalid admission policy: {reason}"),
            Self::IdentityExhausted => {
                formatter.write_str("waiting admission identity space is exhausted")
            }
            Self::DuplicateTicket(ticket) => write!(
                formatter,
                "waiting admission ticket {} is already queued",
                ticket.get()
            ),
            Self::ForeignCoordinator { expected, actual } => write!(
                formatter,
                "admission wake belongs to coordinator {actual}, expected {expected}"
            ),
            Self::EpochRegression => formatter.write_str("admission wake epochs regressed"),
        }
    }
}

impl std::error::Error for DynamicAdmissionQueueError {}

struct WaitingEntry<T> {
    ticket: WaitingAdmissionTicket,
    request: T,
    deferral: Option<AdmissionDeferral>,
    bypass_release_epochs: u64,
    last_bypass_release_epoch: Option<u64>,
}

/// Plan-local waiting queue with bounded probing and release-epoch fairness.
/// Active decode work is deliberately outside this type, so a fairness barrier
/// can pause new bypass admissions without stopping already admitted work.
pub struct DynamicAdmissionQueue<T> {
    policy: DynamicAdmissionQueuePolicy,
    waiting: VecDeque<WaitingEntry<T>>,
    next_ticket: Option<NonZeroU64>,
    last_wake: Option<AdmissionWakeEpochs>,
}

impl<T> DynamicAdmissionQueue<T> {
    pub fn new(policy: DynamicAdmissionQueuePolicy) -> Self {
        Self::with_capacity(policy, 0)
    }

    pub fn with_capacity(policy: DynamicAdmissionQueuePolicy, capacity: usize) -> Self {
        Self {
            policy,
            waiting: VecDeque::with_capacity(capacity),
            next_ticket: NonZeroU64::new(1),
            last_wake: None,
        }
    }

    pub const fn policy(&self) -> DynamicAdmissionQueuePolicy {
        self.policy
    }

    pub fn set_policy(
        &mut self,
        policy: DynamicAdmissionQueuePolicy,
    ) -> DynamicAdmissionQueuePolicy {
        std::mem::replace(&mut self.policy, policy)
    }

    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.waiting.iter().map(|entry| &entry.request)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.waiting.iter_mut().map(|entry| &mut entry.request)
    }

    pub fn request_mut(&mut self, ticket: WaitingAdmissionTicket) -> Option<&mut T> {
        self.waiting
            .iter_mut()
            .find(|entry| entry.ticket == ticket)
            .map(|entry| &mut entry.request)
    }

    pub fn position(&self, predicate: impl FnMut(&T) -> bool) -> Option<usize> {
        self.waiting
            .iter()
            .map(|entry| &entry.request)
            .position(predicate)
    }

    pub fn find_mut(&mut self, mut predicate: impl FnMut(&T) -> bool) -> Option<&mut T> {
        self.waiting
            .iter_mut()
            .find(|entry| predicate(&entry.request))
            .map(|entry| &mut entry.request)
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        self.waiting.remove(index).map(|entry| entry.request)
    }

    /// Return a previously admitted lifecycle item to this queue without
    /// minting a second waiting identity.
    pub fn requeue(
        &mut self,
        ticket: WaitingAdmissionTicket,
        request: T,
    ) -> Result<(), (DynamicAdmissionQueueError, T)> {
        if self.waiting.iter().any(|entry| entry.ticket == ticket) {
            return Err((DynamicAdmissionQueueError::DuplicateTicket(ticket), request));
        }
        self.waiting.push_back(WaitingEntry {
            ticket,
            request,
            deferral: None,
            bypass_release_epochs: 0,
            last_bypass_release_epoch: None,
        });
        Ok(())
    }

    pub fn enqueue(
        &mut self,
        request: T,
    ) -> Result<WaitingAdmissionTicket, DynamicAdmissionQueueError> {
        let raw = self
            .next_ticket
            .ok_or(DynamicAdmissionQueueError::IdentityExhausted)?;
        self.next_ticket = raw.get().checked_add(1).and_then(NonZeroU64::new);
        let ticket = WaitingAdmissionTicket(raw);
        self.waiting.push_back(WaitingEntry {
            ticket,
            request,
            deferral: None,
            bypass_release_epochs: 0,
            last_bypass_release_epoch: None,
        });
        Ok(ticket)
    }

    pub fn cancel(&mut self, ticket: WaitingAdmissionTicket) -> Option<T> {
        let index = self
            .waiting
            .iter()
            .position(|entry| entry.ticket == ticket)?;
        self.waiting.remove(index).map(|entry| entry.request)
    }

    fn validate_wake(&self, wake: AdmissionWakeEpochs) -> Result<(), DynamicAdmissionQueueError> {
        let Some(prior) = self.last_wake else {
            return Ok(());
        };
        if wake.coordinator_id != prior.coordinator_id {
            return Err(DynamicAdmissionQueueError::ForeignCoordinator {
                expected: prior.coordinator_id.get(),
                actual: wake.coordinator_id.get(),
            });
        }
        if !wake.is_monotonic_after(prior) {
            return Err(DynamicAdmissionQueueError::EpochRegression);
        }
        Ok(())
    }

    /// Probes at most `maximum_probes` entries and admits at most
    /// `maximum_admissions`. The caller supplies reusable event storage, so a
    /// steady scheduling tick does not require a new result allocation.
    pub fn schedule_into<A, R, E>(
        &mut self,
        wake: AdmissionWakeEpochs,
        maximum_probes: usize,
        maximum_admissions: usize,
        events: &mut Vec<AdmissionQueueEvent<T, A, R, E>>,
        mut probe: impl FnMut(&mut T) -> AdmissionProbeOutcome<A, R, E>,
    ) -> Result<AdmissionTickReceipt, DynamicAdmissionQueueError> {
        events.clear();
        self.validate_wake(wake)?;
        self.last_wake = Some(wake);

        let initial_len = self.waiting.len();
        let mut receipt = AdmissionTickReceipt {
            probed: 0,
            skipped_unchanged: 0,
            admitted: 0,
            deferred: 0,
            permanent_rejected: 0,
            faulted: 0,
            preemption_requested: 0,
            backing_growth_requested: 0,
            waiting_after: 0,
            fairness_barrier: None,
        };
        let mut admission_closed = maximum_probes == 0 || maximum_admissions == 0;
        let mut newest_admitted_ticket = None;

        for _ in 0..initial_len {
            let mut entry = self
                .waiting
                .pop_front()
                .expect("initial waiting length remains exact during one queue scan");
            let aged = entry.bypass_release_epochs >= self.policy.max_bypass_release_epochs;
            if admission_closed
                || receipt.probed >= maximum_probes
                || receipt.admitted >= maximum_admissions
            {
                if aged && receipt.fairness_barrier.is_none() {
                    receipt.fairness_barrier = Some(entry.ticket);
                }
                self.waiting.push_back(entry);
                continue;
            }

            if let Some(deferral) = entry.deferral {
                if deferral.observed.coordinator_id != wake.coordinator_id {
                    self.waiting.push_front(entry);
                    return Err(DynamicAdmissionQueueError::ForeignCoordinator {
                        expected: wake.coordinator_id.get(),
                        actual: deferral.observed.coordinator_id.get(),
                    });
                }
                if !wake.changed_since(deferral.observed) {
                    receipt.skipped_unchanged += 1;
                    if aged {
                        receipt.fairness_barrier = Some(entry.ticket);
                        admission_closed = true;
                    }
                    self.waiting.push_back(entry);
                    continue;
                }
            }

            receipt.probed += 1;
            match probe(&mut entry.request) {
                AdmissionProbeOutcome::Admitted(admission) => {
                    receipt.admitted += 1;
                    newest_admitted_ticket = Some(entry.ticket);
                    events.push(AdmissionQueueEvent::Admitted {
                        ticket: entry.ticket,
                        request: entry.request,
                        admission,
                    });
                }
                AdmissionProbeOutcome::PermanentRejected(rejection) => {
                    receipt.permanent_rejected += 1;
                    events.push(AdmissionQueueEvent::PermanentRejected {
                        ticket: entry.ticket,
                        request: entry.request,
                        rejection,
                    });
                }
                AdmissionProbeOutcome::Faulted(error) => {
                    receipt.faulted += 1;
                    events.push(AdmissionQueueEvent::Faulted {
                        ticket: entry.ticket,
                        request: entry.request,
                        error,
                    });
                }
                AdmissionProbeOutcome::Deferred(deferral) => {
                    if deferral.observed.coordinator_id != wake.coordinator_id {
                        self.waiting.push_front(entry);
                        return Err(DynamicAdmissionQueueError::ForeignCoordinator {
                            expected: wake.coordinator_id.get(),
                            actual: deferral.observed.coordinator_id.get(),
                        });
                    }
                    entry.deferral = Some(deferral);
                    receipt.deferred += 1;
                    if deferral.action == DeferredAction::PreemptAndRecompute {
                        receipt.preemption_requested += 1;
                        events.push(AdmissionQueueEvent::PreemptionRequested {
                            ticket: entry.ticket,
                            deferral,
                        });
                    } else if deferral.action == DeferredAction::AwaitBackingGrowth {
                        receipt.backing_growth_requested += 1;
                        events.push(AdmissionQueueEvent::BackingGrowthRequested {
                            ticket: entry.ticket,
                            deferral,
                        });
                    }
                    if aged {
                        receipt.fairness_barrier = Some(entry.ticket);
                        admission_closed = true;
                    }
                    self.waiting.push_back(entry);
                }
            }
        }

        if let Some(admitted) = newest_admitted_ticket {
            for entry in &mut self.waiting {
                if entry.ticket < admitted
                    && entry.deferral.is_some()
                    && entry.last_bypass_release_epoch != Some(wake.release_epoch)
                {
                    entry.bypass_release_epochs = entry.bypass_release_epochs.saturating_add(1);
                    entry.last_bypass_release_epoch = Some(wake.release_epoch);
                }
            }
        }
        if receipt.fairness_barrier.is_none() {
            receipt.fairness_barrier = self
                .waiting
                .iter()
                .find(|entry| {
                    entry.deferral.is_some()
                        && entry.bypass_release_epochs >= self.policy.max_bypass_release_epochs
                })
                .map(|entry| entry.ticket);
        }
        receipt.waiting_after = self.waiting.len();
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wake(release: u64, capacity: u64, policy: u64) -> AdmissionWakeEpochs {
        AdmissionWakeEpochs::new(NonZeroU64::new(7).unwrap(), release, capacity, policy)
    }

    fn wait_at(wake: AdmissionWakeEpochs) -> AdmissionDeferral {
        AdmissionDeferral::new(DeferredAction::WaitForRelease, wake)
    }

    #[test]
    fn deferred_head_does_not_block_an_eligible_smaller_request() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(10_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        let mut events = Vec::with_capacity(2);
        let receipt = queue
            .schedule_into(wake(0, 0, 0), 2, 2, &mut events, |request| {
                if *request == 10 {
                    AdmissionProbeOutcome::<u64, (), ()>::Deferred(wait_at(wake(0, 0, 0)))
                } else {
                    AdmissionProbeOutcome::Admitted(*request)
                }
            })
            .unwrap();

        assert_eq!(receipt.admitted(), 1);
        assert_eq!(receipt.deferred(), 1);
        assert_eq!(receipt.waiting_after(), 1);
        assert!(matches!(events[0], AdmissionQueueEvent::Admitted { .. }));
    }

    #[test]
    fn unchanged_evidence_suppresses_blind_retries() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        let mut events = Vec::new();
        queue
            .schedule_into(wake(3, 4, 0), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(3, 4, 0)))
            })
            .unwrap();
        let mut probes = 0;
        let receipt = queue
            .schedule_into(wake(3, 4, 0), 1, 1, &mut events, |_| {
                probes += 1;
                AdmissionProbeOutcome::<(), (), ()>::Admitted(())
            })
            .unwrap();

        assert_eq!(probes, 0);
        assert_eq!(receipt.skipped_unchanged(), 1);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn release_epoch_wakes_and_admits_a_deferred_request() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        let mut events = Vec::new();
        queue
            .schedule_into(wake(1, 1, 0), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(1, 1, 0)))
            })
            .unwrap();
        let receipt = queue
            .schedule_into(wake(2, 1, 0), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Admitted(())
            })
            .unwrap();

        assert_eq!(receipt.admitted(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn aging_barrier_stops_new_bypass_without_owning_active_work() {
        let policy = DynamicAdmissionQueuePolicy::new(2).unwrap();
        let mut queue = DynamicAdmissionQueue::new(policy);
        let head = queue.enqueue(100_u64).unwrap();
        queue.enqueue(1_u64).unwrap();
        let mut events = Vec::new();
        queue
            .schedule_into(wake(1, 0, 0), 2, 2, &mut events, |request| {
                if *request == 100 {
                    AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(1, 0, 0)))
                } else {
                    AdmissionProbeOutcome::Admitted(())
                }
            })
            .unwrap();
        queue.enqueue(2_u64).unwrap();
        queue
            .schedule_into(wake(2, 0, 0), 2, 2, &mut events, |request| {
                if *request == 100 {
                    AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(2, 0, 0)))
                } else {
                    AdmissionProbeOutcome::Admitted(())
                }
            })
            .unwrap();
        queue.enqueue(3_u64).unwrap();
        let mut probes = Vec::new();
        let receipt = queue
            .schedule_into(wake(3, 0, 0), 3, 3, &mut events, |request| {
                probes.push(*request);
                AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(3, 0, 0)))
            })
            .unwrap();

        assert_eq!(receipt.fairness_barrier(), Some(head));
        assert_eq!(probes, vec![100]);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn preemption_is_signalled_once_per_unchanged_epoch() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        let ticket = queue.enqueue(9_u64).unwrap();
        let mut events = Vec::new();
        let preempt = AdmissionDeferral::new(DeferredAction::PreemptAndRecompute, wake(4, 1, 0));
        let first = queue
            .schedule_into(wake(4, 1, 0), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(preempt)
            })
            .unwrap();
        assert_eq!(first.preemption_requested(), 1);
        assert!(matches!(
            events[0],
            AdmissionQueueEvent::PreemptionRequested { ticket: actual, .. } if actual == ticket
        ));

        let second = queue
            .schedule_into(wake(4, 1, 0), 1, 1, &mut events, |_| {
                panic!("unchanged capacity must not re-probe")
            })
            .unwrap();
        assert_eq!(second.preemption_requested(), 0);
        assert!(events.is_empty());
    }

    #[test]
    fn backing_growth_is_signalled_once_per_unchanged_epoch() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        let ticket = queue.enqueue(11_u64).unwrap();
        let mut events = Vec::new();
        let growth = AdmissionDeferral::new(DeferredAction::AwaitBackingGrowth, wake(5, 2, 0));
        let first = queue
            .schedule_into(wake(5, 2, 0), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(growth)
            })
            .unwrap();
        assert_eq!(first.backing_growth_requested(), 1);
        assert!(matches!(
            events[0],
            AdmissionQueueEvent::BackingGrowthRequested { ticket: actual, .. }
                if actual == ticket
        ));

        let second = queue
            .schedule_into(wake(5, 2, 0), 1, 1, &mut events, |_| {
                panic!("unchanged capacity must not request duplicate growth")
            })
            .unwrap();
        assert_eq!(second.backing_growth_requested(), 0);
        assert!(events.is_empty());
    }

    #[test]
    fn permanent_rejection_and_fault_leave_no_waiting_ownership() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        let mut events = Vec::new();
        let receipt = queue
            .schedule_into(wake(0, 0, 0), 2, 2, &mut events, |request| {
                if *request == 1 {
                    AdmissionProbeOutcome::<(), _, &str>::PermanentRejected("impossible")
                } else {
                    AdmissionProbeOutcome::Faulted("contract")
                }
            })
            .unwrap();

        assert_eq!(receipt.permanent_rejected(), 1);
        assert_eq!(receipt.faulted(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn coordinator_and_epoch_regressions_fail_closed_before_probing() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        let mut events = Vec::<AdmissionQueueEvent<u64, (), (), ()>>::new();
        queue
            .schedule_into(wake(3, 2, 1), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::Deferred(wait_at(wake(3, 2, 1)))
            })
            .unwrap();

        let foreign = AdmissionWakeEpochs::new(NonZeroU64::new(8).unwrap(), 4, 2, 1);
        assert!(matches!(
            queue.schedule_into(foreign, 1, 1, &mut events, |_| unreachable!()),
            Err(DynamicAdmissionQueueError::ForeignCoordinator { .. })
        ));
        assert!(matches!(
            queue.schedule_into(wake(2, 2, 1), 1, 1, &mut events, |_| unreachable!()),
            Err(DynamicAdmissionQueueError::EpochRegression)
        ));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn cancellation_returns_the_exact_waiting_request() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        let first = queue.enqueue("first").unwrap();
        let second = queue.enqueue("second").unwrap();

        assert_eq!(queue.cancel(first), Some("first"));
        assert_eq!(queue.cancel(first), None);
        assert_eq!(queue.cancel(second), Some("second"));
        assert!(queue.is_empty());
    }
}
