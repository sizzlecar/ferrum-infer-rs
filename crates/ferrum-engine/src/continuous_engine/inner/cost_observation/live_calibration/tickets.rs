//! Private pre-preparation tickets. FIFO acceptance alone cannot attest a
//! complete window: even its last lost/abandoned ticket remains in this ledger.
use std::sync::{
    atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
    Arc, OnceLock,
};

mod failure;
pub(in crate::continuous_engine::inner::cost_observation) use failure::TicketFailureAudit;

const RESERVED: u8 = 1;
const ACCEPTED: u8 = 2;
const CONSUMED: u8 = 3;
const FAILED: u8 = 4;
const OUTSIDE: u8 = 5;
const NO_SUBMISSION: u8 = 6;

// A closed/failed window and its issued population share one linearization
// point. Separate closed and issued atomics allow an in-flight producer to
// issue after the worker has observed a failed window as completely retired.
const ISSUANCE_FAILED: usize = 1 << (usize::BITS - 1);
const ISSUANCE_CLOSED: usize = 1 << (usize::BITS - 2);
const ISSUANCE_COUNT: usize = ISSUANCE_CLOSED - 1;

#[derive(Debug, Default)]
struct Slot {
    state: AtomicU8,
    issued_at_ns: AtomicU64,
    call_id: AtomicU64,
    fifo: AtomicU64,
}

#[derive(Debug)]
pub(super) struct Window {
    pub generation: u64,
    pub phase: usize,
    slots: Box<[Slot]>,
    issuance: AtomicUsize,
    retired: AtomicUsize,
    deadline_ns: u64,
    /// Immutable before the first reserve. A result cannot enroll another
    /// source or change its original ticket offset, identity, or clock.
    enrollments: Box<[SourceEnrollment]>,
    rolling: bool,
    worker: OnceLock<std::thread::Thread>,
    first_ticket_failure: OnceLock<TicketFailureAudit>,
    first_route_failure:
        OnceLock<super::super::route_population::diagnostic::CallRouteFailureAudit>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct SourceEnrollment {
    pub generation: u64,
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub block: usize,
    pub offered_offset: u64,
    pub deadline_ns: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct WindowAudit {
    pub generation: u64,
    pub phase: usize,
    pub declared_offers: usize,
    pub issued: usize,
    pub retired: usize,
    pub eligible_route: usize,
    pub outside_declared_route: usize,
    pub no_submission: usize,
    pub failed: bool,
    pub closed: bool,
    pub first_ticket_failure: Option<TicketFailureAudit>,
    /// DEBUG-only scalar evidence; never consulted by population validation.
    pub first_route_failure:
        Option<super::super::route_population::diagnostic::CallRouteFailureAudit>,
}

/// Worker-only diagnostic data. These numbers cannot recreate a private ticket.
pub(super) struct FailedTicketAudit {
    pub ticket: u64,
    pub issued_at_ns: u64,
    pub call_id: u64,
}

impl Window {
    /// Includes the boxed ledger and Arc counters; no issued-ticket count can
    /// make this allocation smaller while the generation is retained.
    pub fn retained_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(2 * std::mem::size_of::<usize>())?
            .checked_add(self.slots.len().checked_mul(std::mem::size_of::<Slot>())?)?
            .checked_add(
                self.enrollments
                    .len()
                    .checked_mul(std::mem::size_of::<SourceEnrollment>())?,
            )
    }

    pub fn new(generation: u64, phase: usize, offers: usize, deadline_ns: u64) -> Arc<Self> {
        Self::build(generation, phase, offers, deadline_ns, Vec::new(), false)
    }

    pub fn with_enrollments(
        generation: u64,
        phase: usize,
        offers: usize,
        deadline_ns: u64,
        enrollments: Vec<SourceEnrollment>,
    ) -> Arc<Self> {
        Self::build(generation, phase, offers, deadline_ns, enrollments, true)
    }

    fn build(
        generation: u64,
        phase: usize,
        offers: usize,
        deadline_ns: u64,
        enrollments: Vec<SourceEnrollment>,
        rolling: bool,
    ) -> Arc<Self> {
        assert!(
            offers <= ISSUANCE_COUNT,
            "ticket population exceeds counter capacity"
        );
        Arc::new(Self {
            generation,
            phase,
            slots: (0..offers).map(|_| Slot::default()).collect(),
            issuance: AtomicUsize::new(0),
            retired: AtomicUsize::new(0),
            deadline_ns,
            enrollments: enrollments.into_boxed_slice(),
            rolling,
            worker: OnceLock::new(),
            first_ticket_failure: OnceLock::new(),
            first_route_failure: OnceLock::new(),
        })
    }

    pub fn enrollment(&self, generation: u64) -> Option<&SourceEnrollment> {
        self.enrollments
            .iter()
            .find(|source| source.generation == generation)
    }

    pub fn enrolled(&self) -> &[SourceEnrollment] {
        &self.enrollments
    }
    pub fn is_rolling(&self) -> bool {
        self.rolling
    }

    pub fn reserve(self: &Arc<Self>, now_ns: u64) -> Option<Ticket> {
        if now_ns > self.deadline_ns {
            self.fail();
        }
        let index = self
            .issuance
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                let issued = state & ISSUANCE_COUNT;
                if state & ISSUANCE_CLOSED != 0 || issued >= self.slots.len() {
                    return None;
                }
                let next = issued + 1;
                Some(if next == self.slots.len() {
                    next | ISSUANCE_CLOSED
                } else {
                    next
                })
            })
            .ok()?
            & ISSUANCE_COUNT;
        let slot = &self.slots[index];
        slot.issued_at_ns.store(now_ns, Ordering::Relaxed);
        slot.state.store(RESERVED, Ordering::Release);
        Some(Ticket {
            window: Arc::clone(self),
            index,
            retired: false,
            no_submission: None,
            route_population: ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
        })
    }

    pub fn attach_worker(&self, worker: std::thread::Thread) {
        let _ = self.worker.set(worker);
    }
    fn wake(&self) {
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
    }
    /// Stop new reservations without changing a completed population's result.
    /// Incomplete shutdown windows are marked failed by worker finalization.
    pub fn close(&self) {
        let prior = self.issuance.fetch_or(ISSUANCE_CLOSED, Ordering::AcqRel);
        if prior & ISSUANCE_CLOSED == 0 {
            self.wake();
        }
    }
    pub fn fail(&self) {
        let prior = self
            .issuance
            .fetch_or(ISSUANCE_FAILED | ISSUANCE_CLOSED, Ordering::AcqRel);
        let first = prior & ISSUANCE_FAILED == 0;
        if first {
            self.wake();
        }
    }

    pub fn audit(&self) -> WindowAudit {
        let issuance = self.issuance.load(Ordering::Acquire);
        WindowAudit {
            generation: self.generation,
            phase: self.phase,
            declared_offers: self.slots.len(),
            issued: issuance & ISSUANCE_COUNT,
            retired: self.retired.load(Ordering::Acquire),
            eligible_route: self
                .slots
                .iter()
                .filter(|s| s.state.load(Ordering::Acquire) == CONSUMED)
                .count(),
            outside_declared_route: self
                .slots
                .iter()
                .filter(|s| s.state.load(Ordering::Acquire) == OUTSIDE)
                .count(),
            no_submission: self
                .slots
                .iter()
                .filter(|s| s.state.load(Ordering::Acquire) == NO_SUBMISSION)
                .count(),
            failed: issuance & ISSUANCE_FAILED != 0,
            closed: issuance & ISSUANCE_CLOSED != 0,
            first_ticket_failure: self.first_ticket_failure.get().copied(),
            first_route_failure: self.first_route_failure.get().copied(),
        }
    }

    /// Iterate after all issued tickets retired to persist even failed offers
    /// that never entered the FIFO. This does not allocate on producers.
    pub fn failed_tickets(&self) -> impl Iterator<Item = FailedTicketAudit> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            (slot.state.load(Ordering::Acquire) == FAILED).then(|| FailedTicketAudit {
                ticket: index as u64 + 1,
                issued_at_ns: slot.issued_at_ns.load(Ordering::Relaxed),
                call_id: slot.call_id.load(Ordering::Relaxed),
            })
        })
    }

    /// Called even when the original FIFO is empty. A lost final offer cannot
    /// hide behind an otherwise contiguous accepted FIFO.
    pub fn complete(&self, now_ns: u64) -> bool {
        if now_ns > self.deadline_ns {
            self.fail();
        }
        let a = self.audit();
        a.closed
            && !a.failed
            && a.issued == a.declared_offers
            && a.retired == a.issued
            && self.slots.iter().all(|s| {
                matches!(
                    s.state.load(Ordering::Acquire),
                    CONSUMED | OUTSIDE | NO_SUBMISSION
                )
            })
    }
}

/// No public constructor, Clone, deserializer, or independent numeric authority.
/// Ownership travels from preparation through the original sample FIFO exactly once.
#[derive(Debug)]
pub(in crate::continuous_engine::inner::cost_observation) struct Ticket {
    pub(super) window: Arc<Window>,
    index: usize,
    retired: bool,
    no_submission: Option<Arc<super::NoSubmissionReceipt>>,
    route_population: ferrum_types::SloCalibrationRoutePopulationV1,
}
impl Ticket {
    pub(in crate::continuous_engine::inner::cost_observation) fn record_route_failure(
        &self,
        mut diagnostic: super::super::route_population::diagnostic::CallRouteFailureAudit,
    ) {
        diagnostic.generation = self.window.generation;
        diagnostic.phase = self.phase();
        diagnostic.ticket = self.ordinal();
        // The first route gate diagnostic is distinct from the first dropped
        // ticket: parallel calls may settle in another order. Both retain ids.
        if self.window.first_route_failure.set(diagnostic).is_ok() {
            tracing::debug!(
                target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                ?diagnostic,
                "first live route settlement rejection in offered window"
            );
        }
    }

    pub(super) fn with_population(
        mut self,
        policy: ferrum_types::SloCalibrationRoutePopulationV1,
    ) -> Self {
        self.route_population = policy;
        self
    }
    pub fn route_population(&self) -> ferrum_types::SloCalibrationRoutePopulationV1 {
        self.route_population
    }
    pub fn complete_outside(mut self) {
        let slot = &self.window.slots[self.index];
        if self.route_population.is_all_attempts()
            || slot
                .state
                .compare_exchange(ACCEPTED, OUTSIDE, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            self.fail();
        }
        self.retired = true;
        self.window.retired.fetch_add(1, Ordering::AcqRel);
        self.window.wake();
    }

    pub fn no_submission(&self) -> Option<&Arc<super::NoSubmissionReceipt>> {
        self.no_submission.as_ref()
    }
    pub fn bind_no_submission(&mut self, receipt: Arc<super::NoSubmissionReceipt>) {
        self.no_submission = Some(receipt);
    }
    pub fn complete_no_submission(mut self) {
        let slot = &self.window.slots[self.index];
        if !self.route_population.allows_no_submission()
            || self.no_submission.is_none()
            || slot
                .state
                .compare_exchange(ACCEPTED, NO_SUBMISSION, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            self.fail();
        }
        self.retired = true;
        self.window.retired.fetch_add(1, Ordering::AcqRel);
        self.window.wake();
    }

    fn fail(&self) {
        self.record_unretired_failure();
        self.window.slots[self.index]
            .state
            .store(FAILED, Ordering::Release);
        self.window.fail();
    }

    pub fn bind_call(&mut self, call_id: u64) {
        let slot = &self.window.slots[self.index];
        if call_id == 0
            || slot
                .call_id
                .compare_exchange(0, call_id, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            self.fail();
        }
    }
    pub fn accepted(&self, ordinal: u64) {
        let slot = &self.window.slots[self.index];
        if ordinal == 0 || slot.call_id.load(Ordering::Acquire) == 0 {
            self.fail();
            return;
        }
        slot.fifo.store(ordinal, Ordering::Relaxed);
        if slot
            .state
            .compare_exchange(RESERVED, ACCEPTED, Ordering::Release, Ordering::Acquire)
            .is_err()
        {
            self.fail();
        }
    }
    pub fn ordinal(&self) -> u64 {
        self.index as u64 + 1
    }
    pub fn phase(&self) -> usize {
        self.window.phase
    }
    pub fn matches(&self, call_id: u64, fifo: u64, preparation_ns: Option<u64>) -> bool {
        let slot = &self.window.slots[self.index];
        slot.state.load(Ordering::Acquire) == ACCEPTED
            && slot.call_id.load(Ordering::Relaxed) == call_id
            && slot.fifo.load(Ordering::Relaxed) == fifo
            && preparation_ns == Some(slot.issued_at_ns.load(Ordering::Relaxed))
    }
    pub fn complete(mut self) {
        let slot = &self.window.slots[self.index];
        if slot
            .state
            .compare_exchange(ACCEPTED, CONSUMED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.fail();
        }
        self.retired = true;
        self.window.retired.fetch_add(1, Ordering::Release);
    }
}
impl Drop for Ticket {
    fn drop(&mut self) {
        if !self.retired {
            self.fail();
            self.window.retired.fetch_add(1, Ordering::Release);
            self.window.wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_lost_ticket_closes_window_without_next_fifo_entry_or_replacement() {
        let window = Window::new(1, 0, 2, 100);
        let mut first = window.reserve(10).unwrap();
        let last = window.reserve(11).unwrap();
        first.bind_call(7);
        first.accepted(3);
        assert!(first.matches(7, 3, Some(10)));
        first.complete();
        drop(last);
        assert!(!window.complete(20));
        assert!(window.reserve(21).is_none());
        let a = window.audit();
        assert_eq!((a.issued, a.retired), (2, 2));
        assert!(a.failed && a.closed);
    }

    #[test]
    fn phase_ticket_cannot_rebind_call_or_refresh_original_clock() {
        let window = Window::new(2, 1, 1, 100);
        let mut ticket = window.reserve(40).unwrap();
        ticket.bind_call(9);
        ticket.accepted(12);
        assert!(ticket.matches(9, 12, Some(40)));
        assert!(!ticket.matches(10, 12, Some(40)));
        assert!(!ticket.matches(9, 12, Some(41)));
        ticket.bind_call(10);
        assert!(!ticket.matches(10, 12, Some(40)));
        drop(ticket);
        assert!(!window.complete(50));
    }

    #[test]
    fn completed_population_remains_expired_and_never_reopens_slots() {
        let window = Window::new(1, 2, 1, 10);
        let mut ticket = window.reserve(1).unwrap();
        ticket.bind_call(1);
        ticket.accepted(1);
        ticket.complete();
        assert!(window.complete(10));
        assert!(!window.complete(11));
        assert!(window.reserve(2).is_none());
    }

    #[test]
    fn closing_issuance_keeps_a_completed_population_successful() {
        let window = Window::new(1, 2, 1, 100);
        let mut ticket = window.reserve(1).unwrap();
        ticket.bind_call(7);
        ticket.accepted(3);
        ticket.complete();
        window.close();
        assert!(window.complete(2));
        assert!(!window.audit().failed);
        assert!(window.reserve(3).is_none());
    }

    #[test]
    fn failed_window_rejects_a_reservation_based_on_a_stale_open_population() {
        let window = Window::new(1, 0, 2, 100);
        // Pause the producer between its read and the reservation CAS. This
        // is the old race in which the worker could retire this generation
        // before the producer incremented the independent issued counter.
        let producer_observed = window.issuance.load(Ordering::Acquire);
        window.fail();
        let retired = window.audit();
        assert!(retired.failed && retired.closed);
        assert_eq!(retired.issued, retired.retired);
        assert!(window
            .issuance
            .compare_exchange(
                producer_observed,
                producer_observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err());
        assert!(window.reserve(1).is_none());
        assert_eq!(window.audit().issued, 0);
    }

    #[test]
    fn reservation_before_failure_remains_in_the_retirement_population() {
        let window = Window::new(1, 0, 2, 100);
        let mut ticket = window.reserve(4).unwrap();
        ticket.bind_call(9);
        window.fail();
        let pending = window.audit();
        assert_eq!((pending.issued, pending.retired), (1, 0));
        assert!(window.reserve(5).is_none());
        drop(ticket);
        let retired = window.audit();
        assert_eq!((retired.issued, retired.retired), (1, 1));
        let failed: Vec<_> = window.failed_tickets().collect();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].ticket, 1);
        assert_eq!(failed[0].issued_at_ns, 4);
        assert_eq!(failed[0].call_id, 9);
    }

    #[test]
    fn invalid_acceptance_cannot_retire_as_a_successful_ticket() {
        let window = Window::new(1, 0, 1, 100);
        let ticket = window.reserve(4).unwrap();
        ticket.accepted(9); // No call was bound before FIFO acceptance.
        ticket.complete();
        assert!(!window.complete(5));
        let failed: Vec<_> = window.failed_tickets().collect();
        assert_eq!(failed.len(), 1);
        assert_eq!((failed[0].ticket, failed[0].call_id), (1, 0));
        assert_eq!(window.audit().retired, 1);
    }
}
