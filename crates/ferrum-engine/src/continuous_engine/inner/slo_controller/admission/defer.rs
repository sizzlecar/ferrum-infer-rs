//! Time holds are separate from physical admission deferrals. No synthetic
//! capacity epoch/permit is created here; the existing queue probes and waits
//! remain authoritative for every permitted owner.
use super::*;
use std::collections::HashSet;

#[derive(Debug)]
pub(super) struct TimeAdmissionReview {
    pub review_at: Option<Instant>,
    pub queue_iteration: u64,
    pub wake_epochs: AdmissionWakeEpochs,
    pub obligations: usize,
    pub model_version: u64,
    pub capacity_availability: Vec<ferrum_interfaces::vnext::CapacityAvailabilityEpoch>,
    pub queue_owners: Vec<(PlanningRequestKey, PlanningQueueKind)>,
}

impl TimeAdmissionReview {
    pub(super) fn pending(&self, captured: &ControllerSnapshot, now: Instant) -> bool {
        self.review_at.is_some_and(|at| now < at)
            && self.queue_iteration == captured.queue.iteration()
            && self.wake_epochs == captured.queue.wake_epochs()
            && self.obligations == captured.snapshot.requests.len()
            && self.model_version == captured.model.model_version()
            && self.capacity_availability == captured.queue.capacity_availability()
            && self.queue_owners.len() == captured.queue.requests().len()
            && self
                .queue_owners
                .iter()
                .zip(captured.queue.requests())
                .all(|((key, queue), row)| *key == row.key && *queue == row.queue)
    }
}

struct OverflowOwner {
    request_id: RequestId,
    owner: Arc<()>,
}

#[derive(Default)]
pub(in crate::continuous_engine::inner::slo_controller) struct TimeActivationState {
    held: HashSet<RequestId>,
    review_at: Option<Instant>,
    /// At most one overdue fresh owner bypasses the time ceiling. This owner
    /// retains the slot through maintenance/NotSubmitted/capacity yield until
    /// it leaves or fits below the normal ceiling. A fast loop cannot mint an
    /// unbounded sequence of timeout exceptions.
    overflow: Option<OverflowOwner>,
}

pub(in crate::continuous_engine::inner) struct TimeActivationGate {
    pub maximum: usize,
    held: HashSet<RequestId>,
    overflow_available: bool,
}
impl TimeActivationGate {
    pub(in crate::continuous_engine::inner) fn eligible(&self, request: &InferenceRequest) -> bool {
        !self.held.contains(&request.id)
    }
}

impl EngineInner {
    /// Called under the ingress/iteration lock, before any physical probe.
    /// Queue seals are read evidence only; admission still obtains its real
    /// prefix, pressure, capacity and executor permissions in the normal path.
    pub(in crate::continuous_engine::inner) fn prepare_time_activation(
        &self,
        maximum: usize,
        wake: AdmissionWakeSnapshot<'_>,
    ) -> TimeActivationGate {
        if self.config.scheduler.slo.mode != ferrum_types::SloMode::Enforce {
            return TimeActivationGate {
                maximum,
                held: HashSet::new(),
                overflow_available: false,
            };
        }
        let now = slo_clock_now();
        let queue = self
            .scheduler
            .planning_state(NonZeroUsize::new(4096).unwrap(), wake);
        let mut sequences = self.sequences.write();
        let mut controller = self.slo_controller.lock();
        let activation = &mut controller.time_activation;
        let Ok(queue) = queue else {
            // This admission-only retry does not park already active service.
            activation.review_at = now.checked_add(std::time::Duration::from_millis(
                self.config.scheduler.slo.planner.retry_backoff_ms.get(),
            ));
            return TimeActivationGate {
                maximum: 0,
                held: activation.held.clone(),
                overflow_available: false,
            };
        };
        if queue
            .requests()
            .iter()
            .any(|row| !sequences.contains_key(&row.key.request_id))
        {
            activation.review_at = now.checked_add(std::time::Duration::from_millis(
                self.config.scheduler.slo.planner.retry_backoff_ms.get(),
            ));
            return TimeActivationGate {
                maximum: 0,
                held: activation.held.clone(),
                overflow_available: false,
            };
        }
        let active_limit = self
            .config
            .scheduler
            .slo
            .admission
            .max_active_requests
            .get();
        let mut activated = 0usize;
        for row in queue.requests() {
            let Some(sequence) = sequences.get_mut(&row.key.request_id) else {
                continue;
            };
            let continuing = matches!(
                row.queue,
                PlanningQueueKind::Prefill
                    | PlanningQueueKind::Decode
                    | PlanningQueueKind::Preempted
            ) || row.recompute_target_tokens.is_some()
                || sequence
                    .time_admission
                    .as_ref()
                    .is_some_and(|state| state.activated);
            if continuing {
                activated += 1;
                if let Some(state) = sequence.time_admission.as_mut() {
                    state.activated = true;
                }
            }
        }
        if activation.overflow.as_ref().is_some_and(|owner| {
            sequences.get(&owner.request_id).is_none_or(|sequence| {
                !Arc::ptr_eq(&owner.owner, &sequence.stream_projection_identity)
            }) || !queue
                .requests()
                .iter()
                .any(|row| row.key.request_id == owner.request_id)
                || activated < active_limit
        }) {
            activation.overflow = None;
        }
        // If the previous exception is now one of at most `active_limit`
        // ordinary active owners, there is no longer an excess slot to retain.
        if activated <= active_limit
            && activation.overflow.as_ref().is_some_and(|owner| {
                sequences.get(&owner.request_id).is_some_and(|sequence| {
                    sequence
                        .time_admission
                        .as_ref()
                        .is_some_and(|state| state.activated)
                })
            })
        {
            activation.overflow = None;
        }
        let room = active_limit.saturating_sub(activated);
        activation.held.clear();
        activation.review_at = None;
        let mut continuations = 0usize;
        let overflow_available =
            self.completion_allowed() && activation.overflow.is_none() && activated == active_limit;
        let mut overdue = false;
        for row in queue
            .requests()
            .iter()
            .filter(|row| row.queue == PlanningQueueKind::Waiting)
        {
            let Some(sequence) = sequences.get_mut(&row.key.request_id) else {
                activation.held.insert(row.key.request_id.clone());
                continue;
            };
            if row.recompute_target_tokens.is_some()
                || sequence
                    .time_admission
                    .as_ref()
                    .is_some_and(|state| state.activated)
            {
                continuations += 1;
                continue;
            }
            if room > 0 {
                continue;
            }
            activation.held.insert(row.key.request_id.clone());
            let expiry = sequence.time_admission.as_ref().and_then(|state| {
                state
                    .ingress
                    .checked_add(self.config.scheduler.slo.admission.max_wait())
            });
            if let Some(at) = expiry.filter(|at| *at > now) {
                activation.review_at = Some(activation.review_at.map_or(at, |old| old.min(at)));
            } else if overflow_available {
                // Missing original timing cannot manufacture a time promise or
                // justify an indefinite time hold. It shares the one bounded
                // completion-only slot, without changing physical permission.
                overdue = true;
                activation.held.remove(&row.key.request_id);
            }
            if let Some(state) = sequence.time_admission.as_mut() {
                if !state.last_assessment.is_some_and(|assessment| {
                    assessment.kind
                        == TimeAdmissionAssessmentKind::Deferred(
                            TimeAdmissionDeferReason::ActiveLimit,
                        )
                }) {
                    counter!("ferrum.engine.slo_time_admission_assessments_total", "decision" => "deferred_time_promise").increment(1);
                    tracing::trace!(request_id = %row.key.request_id, active_limit, review_at = ?expiry, "time active limit retains fresh waiting owner");
                }
                state.last_assessment = Some(TimeAdmissionAssessment {
                    at: now,
                    snapshot_generation: queue.iteration(),
                    model_version: 0,
                    kind: TimeAdmissionAssessmentKind::Deferred(
                        TimeAdmissionDeferReason::ActiveLimit,
                    ),
                });
            }
        }
        TimeActivationGate {
            // Continuations consume no new logical slot, but share this
            // conservative per-turn limit. A successful admission drives
            // another iteration. Even when every fresh waiter is overdue,
            // only one can claim the extra slot, through an actual Admitted.
            maximum: maximum.min(if room > 0 {
                room
            } else if overdue {
                1
            } else {
                continuations
            }),
            held: activation.held.clone(),
            overflow_available,
        }
    }

    pub(in crate::continuous_engine::inner) fn record_time_activation(
        &self,
        request_id: &RequestId,
        gate: &TimeActivationGate,
    ) {
        let mut sequences = self.sequences.write();
        let Some(sequence) = sequences.get_mut(request_id) else {
            return;
        };
        let continuing = sequence
            .time_admission
            .as_ref()
            .is_some_and(|state| state.activated);
        if !continuing && gate.overflow_available {
            let mut controller = self.slo_controller.lock();
            if controller.time_activation.overflow.is_none() {
                controller.time_activation.overflow = Some(OverflowOwner {
                    request_id: request_id.clone(),
                    owner: Arc::clone(&sequence.stream_projection_identity),
                });
                tracing::trace!(%request_id, "overdue waiting owner physically admitted for completion without time promise");
                counter!("ferrum.engine.slo_time_admission_overdue_slots_total").increment(1);
            }
        }
        if let Some(state) = sequence.time_admission.as_mut() {
            state.activated = true;
            if !continuing && gate.overflow_available {
                if let Some(assessment) = state.last_assessment.as_mut() {
                    assessment.kind = TimeAdmissionAssessmentKind::OverdueCompletion;
                }
            }
        }
    }

    pub(in crate::continuous_engine::inner::slo_controller) fn time_admission_capacity_wait(
        &self,
    ) -> Result<Option<ferrum_interfaces::vnext::CapacityWaitCondition>> {
        let held = self.slo_controller.lock().time_activation.held.clone();
        self.scheduler
            .passive_capacity_wait_condition_with_eligibility(&|request| {
                !held.contains(&request.id)
            })
    }

    /// Independent from transient controller backoff: waiting for a new time
    /// admission must never delay the currently active set's execution.
    pub(in crate::continuous_engine) async fn wait_for_slo_time_admission(&self) {
        if self.config.scheduler.slo.mode != ferrum_types::SloMode::Enforce {
            return std::future::pending().await;
        }
        let mut at = self.slo_controller.lock().time_activation.review_at;
        for sequence in self.sequences.read().values() {
            if let Some(review) = sequence
                .time_admission
                .as_ref()
                .and_then(|state| state.deferred.as_ref())
            {
                if let Some(review_at) = review.review_at {
                    at = Some(at.map_or(review_at, |old| old.min(review_at)));
                }
            }
        }
        let Some(at) = at else {
            return std::future::pending().await;
        };
        if at > slo_clock_now() {
            tokio::time::sleep_until(tokio::time::Instant::from_std(at)).await;
        }
        let now = slo_clock_now();
        // A review can become due before the idle select is registered. It
        // still wakes once. Consume the actual elapsed timestamps rather than
        // dropping them or leaving a permanent ready future behind.
        let mut sequences = self.sequences.write();
        for sequence in sequences.values_mut() {
            if let Some(review) = sequence
                .time_admission
                .as_mut()
                .and_then(|state| state.deferred.as_mut())
            {
                if review.review_at.is_some_and(|at| at <= now) {
                    review.review_at = None;
                }
            }
        }
        let mut state = self.slo_controller.lock();
        if state.time_activation.review_at.is_some_and(|at| at <= now) {
            state.time_activation.review_at = None;
        }
    }
}

#[cfg(test)]
mod tests;
