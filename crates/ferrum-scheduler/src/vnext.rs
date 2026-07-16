//! Capacity-driven waiting admission for the vNext execution runtime.
//!
//! This queue owns policy only. Physical and logical capacity remain owned by
//! `ferrum-interfaces`; callers probe those authorities and feed the typed
//! result back here. A deferred request stays queued and is not probed again
//! until release, capacity, or product-policy evidence changes.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::num::NonZeroU64;

use ferrum_interfaces::vnext::{
    AdmissionDeferred, CapacityAvailabilityEpoch, CapacityEpochs, CapacityWaitCondition,
    DeferredAction, DynamicBackingDeferred, LogicalAdmissionCoordinatorId,
};

/// Typed product policy for waiting-request bypass fairness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicAdmissionQueuePolicy {
    max_relevant_bypass_events: u64,
}

impl DynamicAdmissionQueuePolicy {
    pub fn new(max_relevant_bypass_events: u64) -> Result<Self, DynamicAdmissionQueueError> {
        if max_relevant_bypass_events == 0 {
            return Err(DynamicAdmissionQueueError::InvalidPolicy(
                "max_relevant_bypass_events must be non-zero",
            ));
        }
        Ok(Self {
            max_relevant_bypass_events,
        })
    }

    pub const fn max_relevant_bypass_events(self) -> u64 {
        self.max_relevant_bypass_events
    }
}

impl Default for DynamicAdmissionQueuePolicy {
    fn default() -> Self {
        Self {
            max_relevant_bypass_events: 8,
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

    fn is_monotonic_after(self, prior: Self) -> bool {
        self.coordinator_id == prior.coordinator_id
            && self.release_epoch >= prior.release_epoch
            && self.capacity_epoch >= prior.capacity_epoch
            && self.policy_epoch >= prior.policy_epoch
    }
}

/// One allocation-free scheduler view over the current global audit epochs
/// and the canonical per-source availability generations.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionWakeSnapshot<'a> {
    epochs: AdmissionWakeEpochs,
    availability: &'a [CapacityAvailabilityEpoch],
}

impl<'a> AdmissionWakeSnapshot<'a> {
    pub const fn new(
        epochs: AdmissionWakeEpochs,
        availability: &'a [CapacityAvailabilityEpoch],
    ) -> Self {
        Self {
            epochs,
            availability,
        }
    }

    pub const fn epochs(self) -> AdmissionWakeEpochs {
        self.epochs
    }

    pub const fn availability(self) -> &'a [CapacityAvailabilityEpoch] {
        self.availability
    }
}

/// Scheduler-facing deferral evidence. It contains no capacity authority and
/// cannot manufacture an admission; it only suppresses blind retries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionDeferral {
    action: DeferredAction,
    observed: AdmissionWakeEpochs,
    wait_condition: CapacityWaitCondition,
}

impl AdmissionDeferral {
    pub const fn new(
        action: DeferredAction,
        observed: AdmissionWakeEpochs,
        wait_condition: CapacityWaitCondition,
    ) -> Self {
        Self {
            action,
            observed,
            wait_condition,
        }
    }

    pub fn from_admission(deferred: &AdmissionDeferred, policy_epoch: u64) -> Self {
        Self::new(
            deferred.action(),
            AdmissionWakeEpochs::from_capacity(deferred.epochs(), policy_epoch),
            deferred.wait_condition().clone(),
        )
    }

    pub fn from_backing(deferred: &DynamicBackingDeferred, policy_epoch: u64) -> Self {
        Self::new(
            DeferredAction::AwaitBackingGrowth,
            AdmissionWakeEpochs::from_capacity(deferred.epochs(), policy_epoch),
            deferred.wait_condition().clone(),
        )
    }

    pub const fn action(&self) -> DeferredAction {
        self.action
    }

    pub const fn observed(&self) -> AdmissionWakeEpochs {
        self.observed
    }

    pub fn wait_condition(&self) -> &CapacityWaitCondition {
        &self.wait_condition
    }

    fn refresh_observed(&mut self, observed: AdmissionWakeEpochs) {
        self.observed = observed;
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
    ContractFaulted {
        ticket: WaitingAdmissionTicket,
        request: T,
        error: DynamicAdmissionQueueError,
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
    ForeignCoordinator {
        expected: u64,
        actual: u64,
    },
    EpochRegression,
    InvalidWakeCondition(String),
    InvalidDeferralTransition {
        expected: DeferredAction,
        actual: Option<DeferredAction>,
    },
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
            Self::InvalidWakeCondition(reason) => {
                write!(formatter, "invalid admission wake condition: {reason}")
            }
            Self::InvalidDeferralTransition { expected, actual } => write!(
                formatter,
                "waiting admission deferral transition expected {expected:?}, found {actual:?}"
            ),
        }
    }
}

impl std::error::Error for DynamicAdmissionQueueError {}

struct WaitingEntry<T> {
    ticket: WaitingAdmissionTicket,
    request: T,
    deferral: Option<AdmissionDeferral>,
    relevant_bypass_events: u64,
    fairness_opportunity: bool,
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

    /// Returns one union predicate only when every queued request is passively
    /// waiting for an exact capacity release. A missing deferral or an active
    /// maintenance action means the caller must run another scheduler tick
    /// instead of parking the engine.
    ///
    /// When several requests observe the same source at different generations,
    /// the oldest generation is retained so a change relevant to any request
    /// wakes the queue.
    pub fn passive_wait_condition(
        &self,
    ) -> Result<Option<CapacityWaitCondition>, DynamicAdmissionQueueError> {
        if self.waiting.is_empty() {
            return Ok(None);
        }

        let mut coordinator_id: Option<LogicalAdmissionCoordinatorId> = None;
        let mut observed_by_source = BTreeMap::new();
        for entry in &self.waiting {
            let Some(deferral) = entry.deferral.as_ref() else {
                return Ok(None);
            };
            if deferral.action() != DeferredAction::WaitForRelease {
                return Ok(None);
            }

            let actual = deferral.wait_condition().coordinator_id();
            if let Some(expected) = coordinator_id {
                if expected != actual {
                    return Err(DynamicAdmissionQueueError::ForeignCoordinator {
                        expected: expected.get(),
                        actual: actual.get(),
                    });
                }
            } else {
                coordinator_id = Some(actual);
            }
            for observed in deferral.wait_condition().observed() {
                observed_by_source
                    .entry(observed.source())
                    .and_modify(|epoch: &mut u64| *epoch = (*epoch).min(observed.epoch()))
                    .or_insert(observed.epoch());
            }
        }

        let observed = observed_by_source
            .into_iter()
            .map(|(source, epoch)| {
                CapacityAvailabilityEpoch::new(source, epoch).map_err(|error| {
                    DynamicAdmissionQueueError::InvalidWakeCondition(error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        CapacityWaitCondition::new(
            coordinator_id.expect("non-empty passive queue has one coordinator"),
            observed,
        )
        .map(Some)
        .map_err(|error| DynamicAdmissionQueueError::InvalidWakeCondition(error.to_string()))
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
            relevant_bypass_events: 0,
            fairness_opportunity: false,
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
            relevant_bypass_events: 0,
            fairness_opportunity: false,
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

    /// Convert a completed backing-growth attempt into an epoch-driven wait
    /// without moving the request or minting a new fairness identity.
    pub fn wait_for_release_after_backing_pressure(
        &mut self,
        mut predicate: impl FnMut(&T) -> bool,
        observed: AdmissionWakeEpochs,
        wait_condition: CapacityWaitCondition,
    ) -> Result<bool, DynamicAdmissionQueueError> {
        self.validate_wake(observed)?;
        let Some(entry) = self
            .waiting
            .iter_mut()
            .find(|entry| predicate(&entry.request))
        else {
            return Ok(false);
        };
        let actual = entry.deferral.as_ref().map(AdmissionDeferral::action);
        if actual != Some(DeferredAction::AwaitBackingGrowth) {
            return Err(DynamicAdmissionQueueError::InvalidDeferralTransition {
                expected: DeferredAction::AwaitBackingGrowth,
                actual,
            });
        }
        let prior = entry
            .deferral
            .as_ref()
            .expect("validated backing-growth deferral remains present")
            .observed();
        if !observed.is_monotonic_after(prior) {
            return Err(DynamicAdmissionQueueError::EpochRegression);
        }
        entry.deferral = Some(AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            observed,
            wait_condition,
        ));
        Ok(true)
    }

    /// Clears one stale backing-growth deferral after the executor observed
    /// that the physical allocator state already changed. This is a one-shot
    /// retry permission, not a capacity grant: the next scheduler pass must
    /// still perform authoritative admission for the same ticket.
    pub fn retry_after_backing_recheck(
        &mut self,
        mut predicate: impl FnMut(&T) -> bool,
        observed: AdmissionWakeEpochs,
    ) -> Result<bool, DynamicAdmissionQueueError> {
        self.validate_wake(observed)?;
        let Some(entry) = self
            .waiting
            .iter_mut()
            .find(|entry| predicate(&entry.request))
        else {
            return Ok(false);
        };
        let actual = entry.deferral.as_ref().map(AdmissionDeferral::action);
        if actual != Some(DeferredAction::AwaitBackingGrowth) {
            return Err(DynamicAdmissionQueueError::InvalidDeferralTransition {
                expected: DeferredAction::AwaitBackingGrowth,
                actual,
            });
        }
        let prior = entry
            .deferral
            .as_ref()
            .expect("validated backing-growth deferral remains present")
            .observed();
        if !observed.is_monotonic_after(prior) {
            return Err(DynamicAdmissionQueueError::EpochRegression);
        }
        entry.deferral = None;
        Ok(true)
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
        wake: AdmissionWakeSnapshot<'_>,
        maximum_probes: usize,
        maximum_admissions: usize,
        events: &mut Vec<AdmissionQueueEvent<T, A, R, E>>,
        probe: impl FnMut(&mut T) -> AdmissionProbeOutcome<A, R, E>,
    ) -> Result<AdmissionTickReceipt, DynamicAdmissionQueueError> {
        self.schedule_into_observed(
            wake,
            maximum_probes,
            maximum_admissions,
            events,
            |_, _, _| {},
            probe,
        )
    }

    /// Variant of [`Self::schedule_into`] that exposes unchanged-epoch skips
    /// without moving or cloning queued state. The observer must remain
    /// non-blocking because it runs while this queue is exclusively borrowed.
    pub fn schedule_into_observed<A, R, E>(
        &mut self,
        wake: AdmissionWakeSnapshot<'_>,
        maximum_probes: usize,
        maximum_admissions: usize,
        events: &mut Vec<AdmissionQueueEvent<T, A, R, E>>,
        mut observe_skipped_unchanged: impl FnMut(&T, WaitingAdmissionTicket, &AdmissionDeferral),
        mut probe: impl FnMut(&mut T) -> AdmissionProbeOutcome<A, R, E>,
    ) -> Result<AdmissionTickReceipt, DynamicAdmissionQueueError> {
        events.clear();
        let wake_epochs = wake.epochs();
        self.validate_wake(wake_epochs)?;
        self.last_wake = Some(wake_epochs);

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
        let mut maximum_admitted_ticket: Option<WaitingAdmissionTicket> = None;

        for _ in 0..initial_len {
            let mut entry = self
                .waiting
                .pop_front()
                .expect("initial waiting length remains exact during one queue scan");
            let first_probe = entry.deferral.is_none();
            entry.fairness_opportunity = false;
            let aged = entry.relevant_bypass_events >= self.policy.max_relevant_bypass_events;
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

            if let Some(deferral) = entry.deferral.as_ref() {
                let contract_error =
                    if deferral.observed.coordinator_id != wake_epochs.coordinator_id {
                        Some(DynamicAdmissionQueueError::ForeignCoordinator {
                            expected: wake_epochs.coordinator_id.get(),
                            actual: deferral.observed.coordinator_id.get(),
                        })
                    } else if deferral.wait_condition.coordinator_id().get()
                        != wake_epochs.coordinator_id.get()
                    {
                        Some(DynamicAdmissionQueueError::ForeignCoordinator {
                            expected: wake_epochs.coordinator_id.get(),
                            actual: deferral.wait_condition.coordinator_id().get(),
                        })
                    } else if !wake_epochs.is_monotonic_after(deferral.observed) {
                        Some(DynamicAdmissionQueueError::EpochRegression)
                    } else {
                        deferral
                            .wait_condition
                            .changed_since(wake.availability())
                            .err()
                            .map(|error| {
                                DynamicAdmissionQueueError::InvalidWakeCondition(error.to_string())
                            })
                    };
                if let Some(error) = contract_error {
                    receipt.faulted += 1;
                    events.push(AdmissionQueueEvent::ContractFaulted {
                        ticket: entry.ticket,
                        request: entry.request,
                        error,
                    });
                    continue;
                }
            }
            if let Some(deferral) = entry.deferral.as_mut() {
                let policy_changed = wake_epochs.policy_epoch > deferral.observed.policy_epoch;
                let audit_changed = wake_epochs.release_epoch > deferral.observed.release_epoch
                    || wake_epochs.capacity_epoch > deferral.observed.capacity_epoch;
                let availability_changed = deferral
                    .wait_condition
                    .changed_since(wake.availability())
                    .expect("existing deferral was validated before retry selection");
                if !policy_changed && !availability_changed {
                    if audit_changed {
                        deferral.refresh_observed(wake_epochs);
                    }
                    receipt.skipped_unchanged += 1;
                    observe_skipped_unchanged(&entry.request, entry.ticket, deferral);
                    if aged {
                        receipt.fairness_barrier = Some(entry.ticket);
                        admission_closed = true;
                    }
                    self.waiting.push_back(entry);
                    continue;
                }
                entry.fairness_opportunity = availability_changed;
            }

            entry.fairness_opportunity |= first_probe;
            receipt.probed += 1;
            match probe(&mut entry.request) {
                AdmissionProbeOutcome::Admitted(admission) => {
                    receipt.admitted += 1;
                    maximum_admitted_ticket = Some(
                        maximum_admitted_ticket
                            .map_or(entry.ticket, |current| current.max(entry.ticket)),
                    );
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
                    let contract_error = if deferral.observed.coordinator_id
                        != wake_epochs.coordinator_id
                    {
                        Some(DynamicAdmissionQueueError::ForeignCoordinator {
                            expected: wake_epochs.coordinator_id.get(),
                            actual: deferral.observed.coordinator_id.get(),
                        })
                    } else if deferral.wait_condition.coordinator_id().get()
                        != wake_epochs.coordinator_id.get()
                    {
                        Some(DynamicAdmissionQueueError::ForeignCoordinator {
                            expected: wake_epochs.coordinator_id.get(),
                            actual: deferral.wait_condition.coordinator_id().get(),
                        })
                    } else {
                        deferral
                            .wait_condition
                            .validate_sources_present(wake.availability())
                            .err()
                            .map(|error| {
                                DynamicAdmissionQueueError::InvalidWakeCondition(error.to_string())
                            })
                    };
                    if let Some(error) = contract_error {
                        receipt.faulted += 1;
                        events.push(AdmissionQueueEvent::ContractFaulted {
                            ticket: entry.ticket,
                            request: entry.request,
                            error,
                        });
                        continue;
                    }
                    let action = deferral.action;
                    entry.deferral = Some(deferral.clone());
                    receipt.deferred += 1;
                    if action == DeferredAction::PreemptAndRecompute {
                        receipt.preemption_requested += 1;
                        events.push(AdmissionQueueEvent::PreemptionRequested {
                            ticket: entry.ticket,
                            deferral,
                        });
                    } else if action == DeferredAction::AwaitBackingGrowth {
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

        for entry in &mut self.waiting {
            if maximum_admitted_ticket.is_some_and(|admitted| entry.ticket < admitted)
                && entry.deferral.is_some()
                && entry.fairness_opportunity
            {
                entry.relevant_bypass_events = entry.relevant_bypass_events.saturating_add(1);
            }
            entry.fairness_opportunity = false;
        }
        if receipt.fairness_barrier.is_none() {
            receipt.fairness_barrier = self
                .waiting
                .iter()
                .find(|entry| {
                    entry.deferral.is_some()
                        && entry.relevant_bypass_events >= self.policy.max_relevant_bypass_events
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
    use ferrum_interfaces::vnext::{CapacityAvailabilitySource, CapacityWaitCondition};

    #[derive(Clone, Copy)]
    struct TestWake {
        epochs: AdmissionWakeEpochs,
        availability: [CapacityAvailabilityEpoch; 1],
    }

    impl TestWake {
        fn epochs(self) -> AdmissionWakeEpochs {
            self.epochs
        }

        fn snapshot(&self) -> AdmissionWakeSnapshot<'_> {
            AdmissionWakeSnapshot::new(self.epochs, &self.availability)
        }

        fn condition(self) -> CapacityWaitCondition {
            CapacityWaitCondition::from_observation(
                self.epochs.coordinator_id().get(),
                self.availability.to_vec(),
            )
            .unwrap()
        }
    }

    fn wake_for(coordinator_id: u64, release: u64, capacity: u64, policy: u64) -> TestWake {
        TestWake {
            epochs: AdmissionWakeEpochs::new(
                NonZeroU64::new(coordinator_id).unwrap(),
                release,
                capacity,
                policy,
            ),
            availability: [CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::ActiveSequenceSlots,
                release
                    .checked_add(capacity)
                    .and_then(|epoch| epoch.checked_add(1))
                    .unwrap(),
            )
            .unwrap()],
        }
    }

    fn wake(release: u64, capacity: u64, policy: u64) -> TestWake {
        wake_for(7, release, capacity, policy)
    }

    fn deferral(action: DeferredAction, wake: TestWake) -> AdmissionDeferral {
        AdmissionDeferral::new(action, wake.epochs(), wake.condition())
    }

    fn wait_at(wake: TestWake) -> AdmissionDeferral {
        deferral(DeferredAction::WaitForRelease, wake)
    }

    fn condition_at(
        coordinator_id: u64,
        source: CapacityAvailabilitySource,
        epoch: u64,
    ) -> CapacityWaitCondition {
        CapacityWaitCondition::from_observation(
            coordinator_id,
            vec![CapacityAvailabilityEpoch::new(source, epoch).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn passive_wait_condition_unions_sources_and_keeps_oldest_generation() {
        let source = CapacityAvailabilitySource::ActiveSequenceSlots;
        let domain = CapacityAvailabilitySource::Domain(
            ferrum_interfaces::vnext::CapacityDomainId::new(1).unwrap(),
        );
        let observed = wake_for(7, 0, 0, 0).epochs();
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        queue.waiting[0].deferral = Some(AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            observed,
            CapacityWaitCondition::from_observation(
                7,
                vec![
                    CapacityAvailabilityEpoch::new(source, 5).unwrap(),
                    CapacityAvailabilityEpoch::new(domain, 9).unwrap(),
                ],
            )
            .unwrap(),
        ));
        queue.waiting[1].deferral = Some(AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            observed,
            condition_at(7, source, 3),
        ));

        let condition = queue.passive_wait_condition().unwrap().unwrap();
        assert_eq!(condition.coordinator_id().get(), 7);
        assert_eq!(
            condition.observed(),
            &[
                CapacityAvailabilityEpoch::new(domain, 9).unwrap(),
                CapacityAvailabilityEpoch::new(source, 3).unwrap(),
            ]
        );
    }

    #[test]
    fn active_or_unprobed_waiting_work_prevents_passive_parking() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        assert!(queue.passive_wait_condition().unwrap().is_none());

        queue.waiting[0].deferral = Some(AdmissionDeferral::new(
            DeferredAction::AwaitBackingGrowth,
            wake_for(7, 0, 0, 0).epochs(),
            condition_at(7, CapacityAvailabilitySource::ActiveSequenceSlots, 1),
        ));
        assert!(queue.passive_wait_condition().unwrap().is_none());
    }

    #[test]
    fn passive_wait_condition_rejects_mixed_coordinators() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        for (index, coordinator_id) in [7_u64, 8].into_iter().enumerate() {
            queue.waiting[index].deferral = Some(AdmissionDeferral::new(
                DeferredAction::WaitForRelease,
                wake_for(coordinator_id, 0, 0, 0).epochs(),
                condition_at(
                    coordinator_id,
                    CapacityAvailabilitySource::ActiveSequenceSlots,
                    1,
                ),
            ));
        }

        assert!(matches!(
            queue.passive_wait_condition(),
            Err(DynamicAdmissionQueueError::ForeignCoordinator {
                expected: 7,
                actual: 8,
            })
        ));
    }

    #[test]
    fn deferred_head_does_not_block_an_eligible_smaller_request() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(10_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        let mut events = Vec::with_capacity(2);
        let receipt = queue
            .schedule_into(wake(0, 0, 0).snapshot(), 2, 2, &mut events, |request| {
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
            .schedule_into(wake(3, 4, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(3, 4, 0)))
            })
            .unwrap();
        let mut probes = 0;
        let mut skipped = Vec::new();
        let receipt = queue
            .schedule_into_observed(
                wake(3, 4, 0).snapshot(),
                1,
                1,
                &mut events,
                |request, ticket, deferral| skipped.push((*request, ticket, deferral.clone())),
                |_| {
                    probes += 1;
                    AdmissionProbeOutcome::<(), (), ()>::Admitted(())
                },
            )
            .unwrap();

        assert_eq!(probes, 0);
        assert_eq!(receipt.skipped_unchanged(), 1);
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].0, 1);
        assert_eq!(skipped[0].2.observed(), wake(3, 4, 0).epochs());
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn invalid_existing_wake_is_evented_without_losing_queue_ownership() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        let domain_source = CapacityAvailabilitySource::Domain(
            ferrum_interfaces::vnext::CapacityDomainId::new(1).unwrap(),
        );
        queue.waiting[1].deferral = Some(AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            wake(0, 0, 0).epochs(),
            CapacityWaitCondition::from_observation(
                7,
                vec![CapacityAvailabilityEpoch::new(domain_source, 1).unwrap()],
            )
            .unwrap(),
        ));
        let mut events = Vec::new();
        let mut probes = 0;

        let receipt = queue
            .schedule_into(wake(0, 1, 0).snapshot(), 2, 2, &mut events, |_| {
                probes += 1;
                AdmissionProbeOutcome::<(), (), ()>::Admitted(())
            })
            .unwrap();

        assert_eq!(probes, 1);
        assert_eq!(receipt.admitted(), 1);
        assert_eq!(receipt.faulted(), 1);
        assert!(queue.is_empty());
        assert!(matches!(events[0], AdmissionQueueEvent::Admitted { .. }));
        assert!(matches!(
            &events[1],
            AdmissionQueueEvent::ContractFaulted { request: 2, error, .. }
                if matches!(error, DynamicAdmissionQueueError::InvalidWakeCondition(_))
        ));
    }

    #[test]
    fn malformed_new_deferral_is_evented_after_an_earlier_admission() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        let domain_source = CapacityAvailabilitySource::Domain(
            ferrum_interfaces::vnext::CapacityDomainId::new(1).unwrap(),
        );
        let malformed = AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            wake(0, 0, 0).epochs(),
            CapacityWaitCondition::from_observation(
                7,
                vec![CapacityAvailabilityEpoch::new(domain_source, 1).unwrap()],
            )
            .unwrap(),
        );
        let mut events = Vec::new();

        let receipt = queue
            .schedule_into(wake(0, 0, 0).snapshot(), 2, 2, &mut events, |request| {
                if *request == 1 {
                    AdmissionProbeOutcome::<(), (), ()>::Admitted(())
                } else {
                    AdmissionProbeOutcome::Deferred(malformed.clone())
                }
            })
            .unwrap();

        assert_eq!(receipt.admitted(), 1);
        assert_eq!(receipt.faulted(), 1);
        assert!(queue.is_empty());
        assert!(matches!(events[0], AdmissionQueueEvent::Admitted { .. }));
        assert!(matches!(
            &events[1],
            AdmissionQueueEvent::ContractFaulted { request: 2, error, .. }
                if matches!(error, DynamicAdmissionQueueError::InvalidWakeCondition(_))
        ));
    }

    #[test]
    fn unrelated_capacity_source_change_refreshes_audit_without_probing() {
        let coordinator_id = NonZeroU64::new(7).unwrap();
        let domain_1 = CapacityAvailabilitySource::Domain(
            ferrum_interfaces::vnext::CapacityDomainId::new(1).unwrap(),
        );
        let domain_2 = CapacityAvailabilitySource::Domain(
            ferrum_interfaces::vnext::CapacityDomainId::new(2).unwrap(),
        );
        let initial_epochs = AdmissionWakeEpochs::new(coordinator_id, 0, 0, 0);
        let initial_availability = [
            CapacityAvailabilityEpoch::new(domain_1, 1).unwrap(),
            CapacityAvailabilityEpoch::new(domain_2, 1).unwrap(),
        ];
        let wait_condition = CapacityWaitCondition::from_observation(
            coordinator_id.get(),
            vec![initial_availability[0]],
        )
        .unwrap();
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        let mut events = Vec::new();
        queue
            .schedule_into(
                AdmissionWakeSnapshot::new(initial_epochs, &initial_availability),
                1,
                1,
                &mut events,
                |_| {
                    AdmissionProbeOutcome::<(), (), ()>::Deferred(AdmissionDeferral::new(
                        DeferredAction::WaitForRelease,
                        initial_epochs,
                        wait_condition.clone(),
                    ))
                },
            )
            .unwrap();

        let unrelated_epochs = AdmissionWakeEpochs::new(coordinator_id, 0, 1, 0);
        let unrelated_availability = [
            CapacityAvailabilityEpoch::new(domain_1, 1).unwrap(),
            CapacityAvailabilityEpoch::new(domain_2, 2).unwrap(),
        ];
        let mut skipped = Vec::new();
        let unrelated = queue
            .schedule_into_observed(
                AdmissionWakeSnapshot::new(unrelated_epochs, &unrelated_availability),
                1,
                1,
                &mut events,
                |_, _, deferral| skipped.push(deferral.clone()),
                |_| panic!("an unrelated domain change must not re-probe admission"),
            )
            .unwrap();
        assert_eq!(unrelated.skipped_unchanged(), 1);
        assert_eq!(skipped[0].observed(), unrelated_epochs);

        let repeated = queue
            .schedule_into(
                AdmissionWakeSnapshot::new(unrelated_epochs, &unrelated_availability),
                1,
                1,
                &mut events,
                |_| panic!("a repeated unrelated wake must remain suppressed"),
            )
            .unwrap();
        assert_eq!(repeated.skipped_unchanged(), 1);

        let relevant_epochs = AdmissionWakeEpochs::new(coordinator_id, 0, 2, 0);
        let relevant_availability = [
            CapacityAvailabilityEpoch::new(domain_1, 2).unwrap(),
            CapacityAvailabilityEpoch::new(domain_2, 2).unwrap(),
        ];
        let relevant = queue
            .schedule_into(
                AdmissionWakeSnapshot::new(relevant_epochs, &relevant_availability),
                1,
                1,
                &mut events,
                |_| AdmissionProbeOutcome::<(), (), ()>::Admitted(()),
            )
            .unwrap();
        assert_eq!(relevant.probed(), 1);
        assert_eq!(relevant.admitted(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn unrelated_release_events_do_not_age_the_fairness_barrier() {
        let coordinator_id = NonZeroU64::new(7).unwrap();
        let domain_1 = CapacityAvailabilitySource::Domain(
            ferrum_interfaces::vnext::CapacityDomainId::new(1).unwrap(),
        );
        let domain_2 = CapacityAvailabilitySource::Domain(
            ferrum_interfaces::vnext::CapacityDomainId::new(2).unwrap(),
        );
        let initial_epochs = AdmissionWakeEpochs::new(coordinator_id, 0, 0, 0);
        let initial_availability = [
            CapacityAvailabilityEpoch::new(domain_1, 1).unwrap(),
            CapacityAvailabilityEpoch::new(domain_2, 1).unwrap(),
        ];
        let wait_condition = CapacityWaitCondition::from_observation(
            coordinator_id.get(),
            vec![initial_availability[0]],
        )
        .unwrap();
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::new(2).unwrap());
        let head = queue.enqueue(100_u64).unwrap();
        queue.enqueue(1_u64).unwrap();
        let mut events = Vec::new();
        let initial = queue
            .schedule_into(
                AdmissionWakeSnapshot::new(initial_epochs, &initial_availability),
                2,
                2,
                &mut events,
                |request| {
                    if *request == 100 {
                        AdmissionProbeOutcome::<u64, (), ()>::Deferred(AdmissionDeferral::new(
                            DeferredAction::WaitForRelease,
                            initial_epochs,
                            wait_condition.clone(),
                        ))
                    } else {
                        AdmissionProbeOutcome::Admitted(*request)
                    }
                },
            )
            .unwrap();
        assert_eq!(initial.admitted(), 1);
        assert_eq!(initial.fairness_barrier(), None);

        queue.enqueue(2_u64).unwrap();
        let unrelated_epochs = AdmissionWakeEpochs::new(coordinator_id, 1, 0, 0);
        let unrelated_availability = [
            CapacityAvailabilityEpoch::new(domain_1, 1).unwrap(),
            CapacityAvailabilityEpoch::new(domain_2, 2).unwrap(),
        ];
        let mut probes = Vec::new();
        let unrelated = queue
            .schedule_into(
                AdmissionWakeSnapshot::new(unrelated_epochs, &unrelated_availability),
                2,
                2,
                &mut events,
                |request| {
                    probes.push(*request);
                    AdmissionProbeOutcome::<u64, (), ()>::Admitted(*request)
                },
            )
            .unwrap();

        assert_eq!(probes, vec![2]);
        assert_eq!(unrelated.admitted(), 1);
        assert_eq!(unrelated.fairness_barrier(), None);
        assert_eq!(queue.position(|request| *request == 100), Some(0));
        assert_eq!(head.get(), 1);
    }

    #[test]
    fn release_epoch_wakes_and_admits_a_deferred_request() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        let mut events = Vec::new();
        queue
            .schedule_into(wake(1, 1, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(1, 1, 0)))
            })
            .unwrap();
        let receipt = queue
            .schedule_into(wake(2, 1, 0).snapshot(), 1, 1, &mut events, |_| {
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
            .schedule_into(wake(1, 0, 0).snapshot(), 2, 2, &mut events, |request| {
                if *request == 100 {
                    AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(1, 0, 0)))
                } else {
                    AdmissionProbeOutcome::Admitted(())
                }
            })
            .unwrap();
        queue.enqueue(2_u64).unwrap();
        queue
            .schedule_into(wake(2, 0, 0).snapshot(), 2, 2, &mut events, |request| {
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
            .schedule_into(wake(3, 0, 0).snapshot(), 3, 3, &mut events, |request| {
                probes.push(*request);
                AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(3, 0, 0)))
            })
            .unwrap();

        assert_eq!(receipt.fairness_barrier(), Some(head));
        assert_eq!(probes, vec![100]);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn requeue_order_does_not_hide_a_newer_ticket_bypass() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        let oldest = queue.enqueue(1_u64).unwrap();
        let blocked = queue.enqueue(2_u64).unwrap();
        let newer = queue.enqueue(3_u64).unwrap();
        let oldest_request = queue.cancel(oldest).unwrap();
        queue.requeue(oldest, oldest_request).unwrap();
        queue.waiting[0].deferral = Some(wait_at(wake(0, 0, 0)));
        let mut events = Vec::new();

        queue
            .schedule_into(wake(1, 0, 0).snapshot(), 3, 3, &mut events, |request| {
                if *request == 2 {
                    AdmissionProbeOutcome::<(), (), ()>::Deferred(wait_at(wake(1, 0, 0)))
                } else {
                    AdmissionProbeOutcome::Admitted(())
                }
            })
            .unwrap();

        let blocked_entry = queue
            .waiting
            .iter()
            .find(|entry| entry.ticket == blocked)
            .unwrap();
        assert_eq!(blocked_entry.relevant_bypass_events, 1);
        assert!(events.iter().any(|event| matches!(
            event,
            AdmissionQueueEvent::Admitted { ticket, .. } if *ticket == newer
        )));
    }

    #[test]
    fn preemption_is_signalled_once_per_unchanged_epoch() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        let ticket = queue.enqueue(9_u64).unwrap();
        let mut events = Vec::new();
        let preempt = deferral(DeferredAction::PreemptAndRecompute, wake(4, 1, 0));
        let first = queue
            .schedule_into(wake(4, 1, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(preempt.clone())
            })
            .unwrap();
        assert_eq!(first.preemption_requested(), 1);
        assert!(matches!(
            events[0],
            AdmissionQueueEvent::PreemptionRequested { ticket: actual, .. } if actual == ticket
        ));

        let second = queue
            .schedule_into(wake(4, 1, 0).snapshot(), 1, 1, &mut events, |_| {
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
        let growth = deferral(DeferredAction::AwaitBackingGrowth, wake(5, 2, 0));
        let first = queue
            .schedule_into(wake(5, 2, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(growth.clone())
            })
            .unwrap();
        assert_eq!(first.backing_growth_requested(), 1);
        assert!(matches!(
            events[0],
            AdmissionQueueEvent::BackingGrowthRequested { ticket: actual, .. }
                if actual == ticket
        ));

        let second = queue
            .schedule_into(wake(5, 2, 0).snapshot(), 1, 1, &mut events, |_| {
                panic!("unchanged capacity must not request duplicate growth")
            })
            .unwrap();
        assert_eq!(second.backing_growth_requested(), 0);
        assert!(events.is_empty());
    }

    #[test]
    fn backing_pressure_preserves_ticket_and_waits_for_release_epoch() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        let ticket = queue.enqueue(11_u64).unwrap();
        let mut events = Vec::new();
        let growth = deferral(DeferredAction::AwaitBackingGrowth, wake(5, 2, 0));
        queue
            .schedule_into(wake(5, 2, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(growth.clone())
            })
            .unwrap();

        assert!(queue
            .wait_for_release_after_backing_pressure(
                |request| *request == 11,
                wake(5, 2, 0).epochs(),
                wake(5, 2, 0).condition(),
            )
            .unwrap());
        let unchanged = queue
            .schedule_into(wake(5, 2, 0).snapshot(), 1, 1, &mut events, |_| {
                panic!("device pressure must not spin at unchanged epochs")
            })
            .unwrap();
        assert_eq!(unchanged.skipped_unchanged(), 1);

        let released = queue
            .schedule_into(wake(6, 2, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Admitted(())
            })
            .unwrap();
        assert_eq!(released.admitted(), 1);
        assert!(queue.is_empty());
        assert!(matches!(
            events.as_slice(),
            [AdmissionQueueEvent::Admitted { ticket: actual, .. }] if *actual == ticket
        ));
    }

    #[test]
    fn backing_recheck_forces_one_probe_without_waiting_for_an_epoch() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        let ticket = queue.enqueue(11_u64).unwrap();
        let mut events = Vec::new();
        let growth = deferral(DeferredAction::AwaitBackingGrowth, wake(5, 2, 0));
        queue
            .schedule_into(wake(5, 2, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Deferred(growth.clone())
            })
            .unwrap();

        assert!(queue
            .retry_after_backing_recheck(|request| *request == 11, wake(5, 2, 0).epochs())
            .unwrap());
        let retried = queue
            .schedule_into(wake(5, 2, 0).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::<(), (), ()>::Admitted(())
            })
            .unwrap();

        assert_eq!(retried.probed(), 1);
        assert_eq!(retried.admitted(), 1);
        assert!(queue.is_empty());
        assert!(matches!(
            events.as_slice(),
            [AdmissionQueueEvent::Admitted { ticket: actual, .. }] if *actual == ticket
        ));
    }

    #[test]
    fn permanent_rejection_and_fault_leave_no_waiting_ownership() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(1_u64).unwrap();
        queue.enqueue(2_u64).unwrap();
        let mut events = Vec::new();
        let receipt = queue
            .schedule_into(wake(0, 0, 0).snapshot(), 2, 2, &mut events, |request| {
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
            .schedule_into(wake(3, 2, 1).snapshot(), 1, 1, &mut events, |_| {
                AdmissionProbeOutcome::Deferred(wait_at(wake(3, 2, 1)))
            })
            .unwrap();

        let foreign = wake_for(8, 4, 2, 1);
        assert!(matches!(
            queue.schedule_into(foreign.snapshot(), 1, 1, &mut events, |_| unreachable!()),
            Err(DynamicAdmissionQueueError::ForeignCoordinator { .. })
        ));
        assert!(matches!(
            queue.schedule_into(
                wake(2, 2, 1).snapshot(),
                1,
                1,
                &mut events,
                |_| unreachable!(),
            ),
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
