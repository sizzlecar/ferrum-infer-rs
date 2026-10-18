//! Split selection/application of the same waiting queue. A selected request
//! stays in place while the caller consults an executor outside the queue lock.
use super::*;

pub(crate) struct AdmissionQueueScan<'a> {
    id: NonZeroU64,
    wake: Option<AdmissionWakeSnapshot<'a>>,
    release_epoch: u64,
    maximum_probes: usize,
    maximum_admissions: usize,
    initial_last_ticket: u64,
    pending: Option<WaitingAdmissionTicket>,
    admission_closed: bool,
    maximum_admitted_ticket: Option<WaitingAdmissionTicket>,
    receipt: AdmissionTickReceipt,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_scan_unwind_preserves_request_and_allows_the_next_scan() {
        let mut queue = DynamicAdmissionQueue::new(DynamicAdmissionQueuePolicy::default());
        queue.enqueue(7_u32).unwrap();
        let mut events: Vec<AdmissionQueueEvent<u32, (), (), ()>> = Vec::new();
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            queue.schedule_with_capacity_observed(
                None,
                0,
                1,
                1,
                &mut events,
                |_, _| AdmissionQueueEligibility::Eligible,
                |_, _| {},
                |_, _, _| {},
                |_| panic!("injected synchronous admission probe panic"),
            )
        }));
        assert!(failed.is_err());
        assert_eq!(queue.iter().copied().collect::<Vec<_>>(), [7]);
        let receipt = queue
            .schedule_with_capacity_observed(
                None,
                0,
                1,
                1,
                &mut events,
                |_, _| AdmissionQueueEligibility::Eligible,
                |_, _| {},
                |_, _, _| {},
                |_| AdmissionProbeOutcome::Admitted(()),
            )
            .unwrap();
        assert_eq!(receipt.admitted(), 1);
        assert!(matches!(
            &events[..],
            [AdmissionQueueEvent::Admitted { request: 7, .. }]
        ));
        assert!(queue.is_empty());
    }
}

impl AdmissionQueueScan<'_> {
    pub(crate) fn id(&self) -> u64 {
        self.id.get()
    }
}

pub(crate) enum QueueProbeApplication<A, R, E> {
    Applied,
    Unapplied {
        outcome: AdmissionProbeOutcome<A, R, E>,
        error: Option<DynamicAdmissionQueueError>,
    },
}

impl<T> DynamicAdmissionQueue<T> {
    pub(crate) fn scan_active(&self) -> bool {
        self.active_scan.is_some()
    }

    pub(crate) fn begin_scan<'a>(
        &mut self,
        wake: Option<AdmissionWakeSnapshot<'a>>,
        release_epoch: u64,
        maximum_probes: usize,
        maximum_admissions: usize,
    ) -> Result<AdmissionQueueScan<'a>, DynamicAdmissionQueueError> {
        if self.active_scan.is_some() {
            return Err(DynamicAdmissionQueueError::InvalidScan(
                "another scan is active",
            ));
        }
        if let Some(wake) = wake {
            self.validate_wake(wake.epochs())?;
        }
        let id = self
            .next_scan
            .ok_or(DynamicAdmissionQueueError::IdentityExhausted)?;
        self.next_scan = id.get().checked_add(1).and_then(NonZeroU64::new);
        self.active_scan = Some(id);
        if let Some(wake) = wake {
            self.last_wake = Some(wake.epochs());
        }
        Ok(AdmissionQueueScan {
            id,
            wake,
            release_epoch,
            maximum_probes,
            maximum_admissions,
            initial_last_ticket: self.next_ticket.map_or(u64::MAX, |next| next.get() - 1),
            pending: None,
            admission_closed: maximum_probes == 0 || maximum_admissions == 0,
            maximum_admitted_ticket: None,
            receipt: AdmissionTickReceipt {
                probed: 0,
                held: 0,
                skipped_unchanged: 0,
                admitted: 0,
                deferred: 0,
                permanent_rejected: 0,
                faulted: 0,
                preemption_requested: 0,
                backing_growth_requested: 0,
                waiting_after: self.waiting.len(),
                fairness_barrier: None,
            },
        })
    }

    pub(crate) fn next_scan_candidate<A, R, E>(
        &mut self,
        scan: &mut AdmissionQueueScan<'_>,
        events: &mut Vec<AdmissionQueueEvent<T, A, R, E>>,
        mut eligibility: impl FnMut(&mut T, WaitingAdmissionTicket) -> AdmissionQueueEligibility,
        mut observe_held: impl FnMut(&T, WaitingAdmissionTicket),
        mut observe_skipped: impl FnMut(&T, WaitingAdmissionTicket, &AdmissionDeferral),
    ) -> Result<Option<WaitingAdmissionTicket>, DynamicAdmissionQueueError> {
        if self.active_scan != Some(scan.id) || scan.pending.is_some() {
            return Err(DynamicAdmissionQueueError::InvalidScan(
                "scan is stale or has an unapplied probe",
            ));
        }
        while let Some(index) = self.waiting.iter().position(|entry| {
            entry.scan_seen != scan.id.get() && entry.ticket.get() <= scan.initial_last_ticket
        }) {
            let entry = &mut self.waiting[index];
            entry.scan_seen = scan.id.get();
            entry.fairness_opportunity = false;
            if eligibility(&mut entry.request, entry.ticket) == AdmissionQueueEligibility::Held {
                scan.receipt.held += 1;
                observe_held(&entry.request, entry.ticket);
                continue;
            }
            let first_probe = entry.deferral.is_none() && entry.release_deferred_at.is_none();
            let aged = entry.relevant_bypass_events >= self.policy.max_relevant_bypass_events;
            if scan.admission_closed
                || scan.receipt.probed >= scan.maximum_probes
                || scan.receipt.admitted >= scan.maximum_admissions
            {
                if aged && scan.receipt.fairness_barrier.is_none() {
                    scan.receipt.fairness_barrier = Some(entry.ticket);
                }
                continue;
            }
            if let Some(observed) = entry.release_deferred_at {
                if scan.release_epoch <= observed {
                    scan.receipt.skipped_unchanged += 1;
                    if aged {
                        scan.receipt.fairness_barrier = Some(entry.ticket);
                        scan.admission_closed = true;
                    }
                    continue;
                }
                entry.fairness_opportunity = true;
            }
            if let Some(deferral) = entry.deferral.as_ref() {
                let contract_error = match scan.wake {
                    None => Some(DynamicAdmissionQueueError::InvalidWakeCondition(
                        "native deferral requires its actual capacity snapshot".into(),
                    )),
                    Some(wake) => {
                        let epochs = wake.epochs();
                        if deferral.observed.coordinator_id != epochs.coordinator_id {
                            Some(DynamicAdmissionQueueError::ForeignCoordinator {
                                expected: epochs.coordinator_id.get(),
                                actual: deferral.observed.coordinator_id.get(),
                            })
                        } else if deferral.wait_condition.coordinator_id().get()
                            != epochs.coordinator_id.get()
                        {
                            Some(DynamicAdmissionQueueError::ForeignCoordinator {
                                expected: epochs.coordinator_id.get(),
                                actual: deferral.wait_condition.coordinator_id().get(),
                            })
                        } else if !epochs.is_monotonic_after(deferral.observed) {
                            Some(DynamicAdmissionQueueError::EpochRegression)
                        } else {
                            deferral
                                .wait_condition
                                .changed_since(wake.availability())
                                .err()
                                .map(|error| {
                                    DynamicAdmissionQueueError::InvalidWakeCondition(
                                        error.to_string(),
                                    )
                                })
                        }
                    }
                };
                if let Some(error) = contract_error {
                    scan.receipt.faulted += 1;
                    let entry = self
                        .waiting
                        .remove(index)
                        .expect("contract-faulted entry exists");
                    events.push(AdmissionQueueEvent::ContractFaulted {
                        ticket: entry.ticket,
                        request: entry.request,
                        error,
                    });
                    continue;
                }
            }
            if let Some(deferral) = entry.deferral.as_mut() {
                let wake = scan.wake.expect("validated native snapshot");
                let epochs = wake.epochs();
                let policy_changed = epochs.policy_epoch > deferral.observed.policy_epoch;
                let audit_changed = epochs.release_epoch > deferral.observed.release_epoch
                    || epochs.capacity_epoch > deferral.observed.capacity_epoch;
                let availability_changed = deferral
                    .wait_condition
                    .changed_since(wake.availability())
                    .expect("validated native wait condition");
                if !policy_changed && !availability_changed {
                    if audit_changed {
                        deferral.refresh_observed(epochs);
                    }
                    scan.receipt.skipped_unchanged += 1;
                    observe_skipped(&entry.request, entry.ticket, deferral);
                    if aged {
                        scan.receipt.fairness_barrier = Some(entry.ticket);
                        scan.admission_closed = true;
                    }
                    continue;
                }
                entry.fairness_opportunity = availability_changed;
            }
            entry.fairness_opportunity |= first_probe;
            entry.in_flight = Some(scan.id);
            scan.receipt.probed += 1;
            scan.pending = Some(entry.ticket);
            let ticket = entry.ticket;
            return Ok(Some(ticket));
        }
        Ok(None)
    }

    pub(crate) fn apply_scan_probe<A, R, E>(
        &mut self,
        scan: &mut AdmissionQueueScan<'_>,
        ticket: WaitingAdmissionTicket,
        outcome: AdmissionProbeOutcome<A, R, E>,
        events: &mut Vec<AdmissionQueueEvent<T, A, R, E>>,
    ) -> QueueProbeApplication<A, R, E> {
        if self.active_scan != Some(scan.id) || scan.pending != Some(ticket) {
            return QueueProbeApplication::Unapplied {
                outcome,
                error: Some(DynamicAdmissionQueueError::InvalidScan(
                    "probe does not belong to this scan",
                )),
            };
        }
        scan.pending = None;
        let Some(index) = self
            .waiting
            .iter()
            .position(|entry| entry.ticket == ticket && entry.in_flight == Some(scan.id))
        else {
            return QueueProbeApplication::Unapplied {
                outcome,
                error: None,
            };
        };
        let mut entry = self
            .waiting
            .remove(index)
            .expect("exact selected entry exists");
        entry.in_flight = None;
        let aged = entry.relevant_bypass_events >= self.policy.max_relevant_bypass_events;
        match outcome {
            AdmissionProbeOutcome::Admitted(admission) => {
                scan.receipt.admitted += 1;
                scan.maximum_admitted_ticket = Some(
                    scan.maximum_admitted_ticket
                        .map_or(ticket, |current| current.max(ticket)),
                );
                events.push(AdmissionQueueEvent::Admitted {
                    ticket,
                    request: entry.request,
                    admission,
                });
            }
            AdmissionProbeOutcome::PermanentRejected(rejection) => {
                scan.receipt.permanent_rejected += 1;
                events.push(AdmissionQueueEvent::PermanentRejected {
                    ticket,
                    request: entry.request,
                    rejection,
                });
            }
            AdmissionProbeOutcome::Faulted(error) => {
                scan.receipt.faulted += 1;
                events.push(AdmissionQueueEvent::Faulted {
                    ticket,
                    request: entry.request,
                    error,
                });
            }
            AdmissionProbeOutcome::CapacityDeferred => {
                entry.deferral = None;
                entry.release_deferred_at = Some(scan.release_epoch);
                scan.receipt.deferred += 1;
                if aged {
                    scan.receipt.fairness_barrier = Some(ticket);
                    scan.admission_closed = true;
                }
                self.waiting.insert(index, entry);
            }
            AdmissionProbeOutcome::Deferred(deferral) => {
                let contract_error = match scan.wake {
                    None => Some(DynamicAdmissionQueueError::InvalidWakeCondition(
                        "native deferral requires its actual capacity snapshot".into(),
                    )),
                    Some(wake) => {
                        let epochs = wake.epochs();
                        if deferral.observed.coordinator_id != epochs.coordinator_id {
                            Some(DynamicAdmissionQueueError::ForeignCoordinator {
                                expected: epochs.coordinator_id.get(),
                                actual: deferral.observed.coordinator_id.get(),
                            })
                        } else if deferral.wait_condition.coordinator_id().get()
                            != epochs.coordinator_id.get()
                        {
                            Some(DynamicAdmissionQueueError::ForeignCoordinator {
                                expected: epochs.coordinator_id.get(),
                                actual: deferral.wait_condition.coordinator_id().get(),
                            })
                        } else {
                            deferral
                                .wait_condition
                                .validate_sources_present(wake.availability())
                                .err()
                                .map(|error| {
                                    DynamicAdmissionQueueError::InvalidWakeCondition(
                                        error.to_string(),
                                    )
                                })
                        }
                    }
                };
                if let Some(error) = contract_error {
                    scan.receipt.faulted += 1;
                    events.push(AdmissionQueueEvent::ContractFaulted {
                        ticket,
                        request: entry.request,
                        error,
                    });
                    return QueueProbeApplication::Applied;
                }
                let action = deferral.action;
                entry.deferral = Some(deferral.clone());
                entry.release_deferred_at = None;
                scan.receipt.deferred += 1;
                if action == DeferredAction::PreemptAndRecompute {
                    scan.receipt.preemption_requested += 1;
                    events.push(AdmissionQueueEvent::PreemptionRequested { ticket, deferral });
                } else if action == DeferredAction::AwaitBackingGrowth {
                    scan.receipt.backing_growth_requested += 1;
                    events.push(AdmissionQueueEvent::BackingGrowthRequested { ticket, deferral });
                }
                if aged {
                    scan.receipt.fairness_barrier = Some(ticket);
                    scan.admission_closed = true;
                }
                self.waiting.insert(index, entry);
            }
        }
        QueueProbeApplication::Applied
    }

    /// Ends selection, including caller cancellation. Pending entries remain
    /// in their original position with their previous real deferral evidence.
    pub(crate) fn finish_scan(
        &mut self,
        mut scan: AdmissionQueueScan<'_>,
    ) -> Result<AdmissionTickReceipt, DynamicAdmissionQueueError> {
        if self.active_scan != Some(scan.id) {
            return Err(DynamicAdmissionQueueError::InvalidScan(
                "scan is no longer active",
            ));
        }
        self.active_scan = None;
        for entry in &mut self.waiting {
            if entry.in_flight == Some(scan.id) {
                entry.in_flight = None;
            }
            if entry.scan_seen == scan.id.get()
                && scan
                    .maximum_admitted_ticket
                    .is_some_and(|admitted| entry.ticket < admitted)
                && (entry.deferral.is_some() || entry.release_deferred_at.is_some())
                && entry.fairness_opportunity
            {
                entry.relevant_bypass_events = entry.relevant_bypass_events.saturating_add(1);
            }
            entry.fairness_opportunity = false;
        }
        if scan.receipt.fairness_barrier.is_none() {
            scan.receipt.fairness_barrier = self
                .waiting
                .iter()
                .find(|entry| {
                    (entry.deferral.is_some() || entry.release_deferred_at.is_some())
                        && entry.relevant_bypass_events >= self.policy.max_relevant_bypass_events
                })
                .map(|entry| entry.ticket);
        }
        scan.receipt.waiting_after = self.waiting.len();
        Ok(scan.receipt)
    }
}
