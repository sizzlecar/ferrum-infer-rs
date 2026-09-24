//! Persistent service debt is an owner fact. Search and rejected submissions
//! cannot repay it. Returned numeric snapshots retain no physical authority.
use super::*;

pub(super) struct RecoveryPeer {
    pub id: RequestId,
    pub owner: Arc<()>,
    pub incarnation: u64,
    pub generation: u64,
    pub debt: RecoveryServiceDebt,
    pub ingress: Instant,
    pub eligible: bool,
    pub recovering: bool,
}

impl EngineInner {
    pub(super) fn capture_recovery_peers(
        &self,
        queue: &PlanningQueueSnapshot,
        budget: &ControllerBudget,
        protection: Option<&PlanningObligationSet>,
    ) -> ControllerResult<Vec<RecoveryPeer>> {
        let unavailable = |reason| Unavailable {
            reason,
            obligations: queue.requests().len(),
            retry: None,
        };
        let sequences = self.sequences.try_read().ok_or_else(|| {
            unavailable("recovery_owner_busy").retry(retry::ControllerRetryReason::SnapshotBusy)
        })?;
        Self::recovery_peers_locked(queue, &sequences, budget, protection)
    }

    pub(super) fn recovery_peers_locked(
        queue: &PlanningQueueSnapshot,
        sequences: &HashMap<RequestId, SequenceState>,
        budget: &ControllerBudget,
        protection: Option<&PlanningObligationSet>,
    ) -> ControllerResult<Vec<RecoveryPeer>> {
        let unavailable = |reason| Unavailable {
            reason,
            obligations: queue.requests().len(),
            retry: None,
        };
        let now = slo_clock_now();
        let mut peers = Vec::with_capacity(queue.requests().len());
        for row in queue.requests() {
            if !budget.poll() {
                return Err(unavailable("compute_budget_exhausted")
                    .retry(retry::ControllerRetryReason::ComputeBudget));
            }
            let sequence = sequences
                .get(&row.key.request_id)
                .ok_or_else(|| unavailable("recovery_owner_changed"))?;
            let Some(state) = &sequence.time_admission else {
                continue;
            };
            let Some(frontier) = sequence.cost_frontier else {
                return Err(unavailable("recovery_frontier_unknown"));
            };
            let output_ready = sequence.credited_output.as_ref().is_some_and(|output| {
                output.failure.is_none() && !output.port.consumer_closed() && output.grant.is_none()
                    && output.port.planning_snapshot().future_capacity.is_some()
                    && !matches!(output.port.planning_snapshot().readiness,
                        crate::continuous_engine::output_flow_runtime::OutputPlanningCreditView::OutputBlocked(_))
            });
            let timed_miss = sequence.slo.as_ref().is_some_and(|slo| {
                let misses = slo.violations();
                misses.ttft
                    || misses.tpot
                    || misses.itl
                    || slo.next_deadline().is_ok_and(|deadline| deadline < now)
            });
            let control_miss = protection.is_some_and(|scope| {
                scope.rows().iter().any(|obligation| {
                    obligation.key.request_id == row.key.request_id
                        && obligation.key.incarnation == frontier.owner_incarnation.get()
                        && obligation.key.work_generation == row.key.generation
                        && !obligation.expired_control_milestones.is_empty()
                })
            });
            peers.push(RecoveryPeer {
                id: row.key.request_id.clone(),
                owner: Arc::clone(&sequence.stream_projection_identity),
                incarnation: frontier.owner_incarnation.get(),
                generation: frontier.work_generation.get(),
                debt: state.recovery_service,
                ingress: sequence
                    .slo
                    .as_ref()
                    .map_or(sequence.start_time, |slo| slo.ingress()),
                eligible: row.readiness.ready()
                    && output_ready
                    && sequence.generated_tokens.len() < sequence.sampling_params.max_tokens,
                recovering: state.recovery_seen || timed_miss || control_miss,
            });
        }
        Ok(peers)
    }

    /// Exactly once per durable flight, after a real Submitted outcome and
    /// before host commit advances any selected owner's frontier.
    pub(super) fn record_recovery_submission(&self, work: &owner::ControllerWork) {
        let mut sequences = self.sequences.write();
        for peer in &work.proof.recovery_peers {
            let Some(sequence) = sequences.get_mut(&peer.id) else {
                continue;
            };
            if !Arc::ptr_eq(&peer.owner, &sequence.stream_projection_identity)
                || sequence.cost_frontier.is_none_or(|frontier| {
                    frontier.owner_incarnation.get() != peer.incarnation
                        || frontier.work_generation.get() != peer.generation
                })
            {
                continue;
            }
            let Some(state) = &mut sequence.time_admission else {
                continue;
            };
            state.recovery_seen |= peer.recovering;
            if peer.eligible && peer.recovering && !work.rows().any(|row| row.request_id == peer.id)
            {
                state.recovery_service.bypass();
                counter!("ferrum.engine.slo_recovery_eligible_bypasses_total").increment(1);
            }
        }
    }
}

pub(super) fn required_peer(peers: &[RecoveryPeer]) -> Option<&RecoveryPeer> {
    peers
        .iter()
        .filter(|peer| peer.eligible && peer.recovering && peer.debt.due())
        .min_by_key(|peer| {
            (
                std::cmp::Reverse(peer.debt.eligible_bypasses()),
                peer.ingress,
                peer.incarnation,
            )
        })
}
