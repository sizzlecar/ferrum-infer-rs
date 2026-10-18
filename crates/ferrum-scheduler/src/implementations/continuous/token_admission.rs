//! Executor admission may await device preparation. Only selection and exact
//! outcome publication take scheduler locks; the original request stays queued.
use super::*;
use crate::vnext::{AdmissionQueueScan, QueueProbeApplication};

#[derive(Debug)]
pub struct TokenAdmissionCandidate {
    key: PrefixRequestKey,
    scan_id: u64,
    request: InferenceRequest,
}

impl TokenAdmissionCandidate {
    pub fn request(&self) -> &InferenceRequest {
        &self.request
    }
    pub fn key(&self) -> &PrefixRequestKey {
        &self.key
    }
}

/// A logical queue owner, not executor capacity or submission authority.
#[derive(Debug, Clone)]
pub struct TokenAdmissionAccepted {
    key: PrefixRequestKey,
}

impl TokenAdmissionAccepted {
    pub fn key(&self) -> &PrefixRequestKey {
        &self.key
    }
}

#[must_use = "unapplied executor outcomes must remain owned by the admission caller"]
pub enum TokenAdmissionApply {
    Applied {
        accepted: Option<TokenAdmissionAccepted>,
    },
    /// Cancellation/replacement has no error. Invalid usage retains both the
    /// original executor outcome and the scheduler error, without publishing it.
    Unapplied {
        outcome: ExecutorAdmissionProbeOutcome,
        error: Option<FerrumError>,
    },
}

/// RAII reservation of one scheduler scan, with no mutex/RwLock guard. Drop
/// withdraws only this scan's in-flight marks; it neither drops the queued
/// request nor pretends that executor resources have been released.
pub struct TokenAdmissionRound<'a> {
    scheduler: &'a ContinuousBatchScheduler,
    scan: Option<AdmissionQueueScan<'a>>,
    wake: Option<AdmissionWakeSnapshot<'a>>,
}

impl TokenAdmissionRound<'_> {
    pub(super) fn run_sync(
        mut self,
        probe: &mut dyn FnMut(&InferenceRequest) -> ExecutorAdmissionProbeOutcome,
        observer: &mut dyn FnMut(ExecutorAdmissionQueueObservation),
    ) -> Result<AdmissionTickReceipt> {
        while let Some(candidate) = self.next(observer)? {
            let outcome = probe(candidate.request());
            match self.apply(candidate, outcome) {
                TokenAdmissionApply::Applied { .. } => {}
                TokenAdmissionApply::Unapplied { error, .. } => {
                    return Err(error.unwrap_or_else(|| {
                        FerrumError::scheduler("request was cancelled during synchronous admission")
                    }));
                }
            }
        }
        self.finish()
    }

    pub fn next(
        &mut self,
        observer: &mut dyn FnMut(ExecutorAdmissionQueueObservation),
    ) -> Result<Option<TokenAdmissionCandidate>> {
        let scan = self.scan.as_mut().expect("live admission round");
        let mut waiting = self.scheduler.waiting_queue.write();
        let mut events = self.scheduler.dynamic_admission_events.lock();
        events.clear();
        let observer = std::cell::RefCell::new(observer);
        let selected = waiting
            .next_scan_candidate(
                scan,
                &mut events,
                |request, ticket| {
                    self.scheduler.token_admission_eligibility(
                        request,
                        ticket,
                        &mut **observer.borrow_mut(),
                    )
                },
                |_, _| {},
                |request, ticket, deferral| {
                    observer.borrow_mut()(ExecutorAdmissionQueueObservation::SkippedUnchanged {
                        request_id: request.inner.request.id.clone(),
                        ticket: ticket.get(),
                        deferral: deferral.clone(),
                        current: self
                            .wake
                            .expect("native deferral has a real snapshot")
                            .epochs(),
                    });
                },
            )
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        // The waiting lock spans removal and publication of a contract failure,
        // so concurrent cancel/submit cannot observe an unowned lifecycle gap.
        self.scheduler.apply_token_admission_events(&mut events);
        selected
            .map(|ticket| {
                let request = waiting
                    .request_mut(ticket)
                    .expect("selected ticket remains queued");
                let key = PrefixRequestKey::from_request(request)
                    .expect("queued request retains its original waiting identity");
                Ok(TokenAdmissionCandidate {
                    key,
                    scan_id: scan.id(),
                    request: request.inner.request.clone(),
                })
            })
            .transpose()
    }

    pub fn apply(
        &mut self,
        candidate: TokenAdmissionCandidate,
        outcome: ExecutorAdmissionProbeOutcome,
    ) -> TokenAdmissionApply {
        let scan = self.scan.as_mut().expect("live admission round");
        if scan.id() != candidate.scan_id {
            return TokenAdmissionApply::Unapplied {
                outcome,
                error: Some(FerrumError::scheduler(
                    "admission candidate belongs to another round",
                )),
            };
        }
        if let AdmissionProbeOutcome::Admitted(receipt) = &outcome {
            if receipt.request_id != *candidate.key.request_id() {
                return TokenAdmissionApply::Unapplied {
                    outcome,
                    error: Some(FerrumError::scheduler(
                        "admission receipt belongs to another request",
                    )),
                };
            }
        }
        let mut waiting = self.scheduler.waiting_queue.write();
        let ticket = candidate.key.ticket();
        if waiting
            .request_mut(ticket)
            .is_some_and(|request| !candidate.key.matches(request))
        {
            return TokenAdmissionApply::Unapplied {
                outcome,
                error: Some(FerrumError::scheduler(
                    "admission queue incarnation changed",
                )),
            };
        }
        let admitted = matches!(&outcome, AdmissionProbeOutcome::Admitted(_));
        let mut events = self.scheduler.dynamic_admission_events.lock();
        events.clear();
        match waiting.apply_scan_probe(scan, ticket, outcome, &mut events) {
            QueueProbeApplication::Unapplied { outcome, error } => TokenAdmissionApply::Unapplied {
                outcome,
                error: error.map(|error| FerrumError::scheduler(error.to_string())),
            },
            QueueProbeApplication::Applied => {
                // The original queue item (including concurrent priority
                // updates) is promoted, never the cloned probe input.
                self.scheduler.apply_token_admission_events(&mut events);
                TokenAdmissionApply::Applied {
                    accepted: admitted.then_some(TokenAdmissionAccepted { key: candidate.key }),
                }
            }
        }
    }

    pub fn finish(mut self) -> Result<AdmissionTickReceipt> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> Result<AdmissionTickReceipt> {
        let scan = self.scan.take().expect("admission round finishes once");
        let mut waiting = self.scheduler.waiting_queue.write();
        // Also publish any queue-owned fault already collected before a
        // caller observer unwound. Clearing the shared event storage must not
        // silently drop a request that was removed from the waiting queue.
        self.scheduler
            .apply_token_admission_events(&mut self.scheduler.dynamic_admission_events.lock());
        let receipt = waiting
            .finish_scan(scan)
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        self.scheduler.record_token_admission_tick(receipt);
        Ok(receipt)
    }
}

impl Drop for TokenAdmissionRound<'_> {
    fn drop(&mut self) {
        if self.scan.is_some() {
            if let Err(error) = self.finish_inner() {
                tracing::error!(%error, "failed to withdraw scheduler admission scan");
            }
        }
    }
}

impl ContinuousBatchScheduler {
    pub fn begin_token_admission<'a>(
        &'a self,
        maximum_admissions: usize,
        wake: Option<AdmissionWakeSnapshot<'a>>,
    ) -> Result<TokenAdmissionRound<'a>> {
        let mut waiting = self.waiting_queue.write();
        let active_capacity = self
            .config
            .max_running_requests
            .saturating_sub(self.active_count());
        let decode_capacity = self
            .cb_config
            .max_decode_batch
            .saturating_sub(self.decoding_count());
        let slots = active_capacity.min(decode_capacity);
        let slots = self
            .capacity_backpressure_admit_limit()
            .map_or(slots, |limit| slots.min(limit))
            .min(maximum_admissions);
        let scan = waiting
            .begin_scan(
                wake,
                self.capacity_release_epoch.load(Ordering::Acquire),
                slots,
                slots,
            )
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        self.dynamic_admission_ticks.fetch_add(1, Ordering::Relaxed);
        Ok(TokenAdmissionRound {
            scheduler: self,
            scan: Some(scan),
            wake,
        })
    }

    pub(super) fn begin_admission_round<'a>(
        &'a self,
        maximum_probes: usize,
        maximum_admissions: usize,
        wake: Option<AdmissionWakeSnapshot<'a>>,
    ) -> Result<TokenAdmissionRound<'a>> {
        let mut waiting = self.waiting_queue.write();
        let probes = waiting.len().min(maximum_probes);
        let scan = waiting
            .begin_scan(
                wake,
                self.capacity_release_epoch.load(Ordering::Acquire),
                probes,
                maximum_admissions,
            )
            .map_err(|error| FerrumError::scheduler(error.to_string()))?;
        self.dynamic_admission_ticks.fetch_add(1, Ordering::Relaxed);
        Ok(TokenAdmissionRound {
            scheduler: self,
            scan: Some(scan),
            wake,
        })
    }

    /// Roll back scheduler acceptance when the engine cannot publish the same
    /// request incarnation. Executor closure/physical release remain the
    /// caller's responsibility. A replacement request is never cancelled.
    pub fn cancel_token_admission(&self, accepted: &TokenAdmissionAccepted) -> Result<bool> {
        let _waiting = self.waiting_queue.write();
        let mut prefill = self.prefill_queue.write();
        let Some(index) = prefill
            .iter()
            .position(|request| accepted.key.matches(request))
        else {
            return Ok(false);
        };
        let request = prefill
            .remove(index)
            .expect("exact admitted request exists");
        let id = &request.inner.request.id;
        self.request_index.write().remove(id);
        self.record_pressure_frontier_terminal(id);
        self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn record_token_admission_tick(&self, receipt: AdmissionTickReceipt) {
        self.dynamic_admission_probes
            .fetch_add(receipt.probed() as u64, Ordering::Relaxed);
        self.dynamic_admission_skipped_unchanged
            .fetch_add(receipt.skipped_unchanged() as u64, Ordering::Relaxed);
        self.dynamic_admission_deferred
            .fetch_add(receipt.deferred() as u64, Ordering::Relaxed);
        self.dynamic_backing_growth_requested
            .fetch_add(receipt.backing_growth_requested() as u64, Ordering::Relaxed);
    }

    fn apply_token_admission_events(&self, events: &mut Vec<ExecutorAdmissionQueueEvent>) {
        for event in events.drain(..) {
            match event {
                AdmissionQueueEvent::Admitted {
                    request, admission, ..
                } => {
                    self.promote_admitted_request(request, &admission);
                }
                AdmissionQueueEvent::PermanentRejected {
                    request, rejection, ..
                } => {
                    self.fail_typed_admission(
                        request,
                        FerrumError::request_validation(format!(
                            "request cannot fit the vNext runtime: {rejection:?}",
                        )),
                    );
                }
                AdmissionQueueEvent::Faulted { request, error, .. } => {
                    self.fail_typed_admission(request, error)
                }
                AdmissionQueueEvent::ContractFaulted { request, error, .. } => {
                    self.fail_typed_admission(request, FerrumError::scheduler(error.to_string()));
                }
                AdmissionQueueEvent::PreemptionRequested { .. }
                | AdmissionQueueEvent::BackingGrowthRequested { .. } => {}
            }
        }
    }

    fn token_admission_eligibility(
        &self,
        request: &mut ContinuousBatchRequest,
        ticket: WaitingAdmissionTicket,
        observer: &mut dyn FnMut(ExecutorAdmissionQueueObservation),
    ) -> AdmissionQueueEligibility {
        if request.prefix_rendezvous.held() {
            return AdmissionQueueEligibility::Held;
        }
        let request_id = &request.inner.request.id;
        if !self.pressure_active.load(Ordering::Acquire) {
            return AdmissionQueueEligibility::Eligible;
        }
        let hold = self.pressure_coordinator.lock().hold_status(request_id);
        match hold {
            PressureHoldStatus::Held { .. } => AdmissionQueueEligibility::Held,
            PressureHoldStatus::OwnerAdmissionEligible { .. } | PressureHoldStatus::None => {
                AdmissionQueueEligibility::Eligible
            }
            PressureHoldStatus::Released {
                episode_id,
                progress_owner_id,
                progress_baseline,
                progress_current,
                reason,
                ordinal,
                previous_wait_condition,
                current_wait_condition,
            } => {
                {
                    let mut coordinator = self.pressure_coordinator.lock();
                    if let Err(error) = coordinator.consume_released_hold(request_id) {
                        warn!(%request_id, %error, "Pressure coordinator rejected terminal hold release");
                        return AdmissionQueueEligibility::Held;
                    }
                    self.pressure_active
                        .store(coordinator.has_records(), Ordering::Release);
                }
                observer(ExecutorAdmissionQueueObservation::PressureHoldReleased {
                    episode_id,
                    transition_ordinal: ordinal,
                    request_id: request_id.clone(),
                    progress_owner_id,
                    progress_baseline,
                    progress_current,
                    reason,
                    previous_wait_condition,
                    current_wait_condition,
                    ticket: ticket.get(),
                });
                AdmissionQueueEligibility::Eligible
            }
        }
    }
}
