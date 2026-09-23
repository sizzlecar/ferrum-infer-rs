use super::*;

impl ContinuousBatchScheduler {
    /// Consume an empty fairness turn only after the controller attempted to
    /// choose work and found none to publish. Legacy next_batch advances this
    /// same clock even on an empty turn; pure snapshots deliberately do not.
    ///
    /// Recheck every obligation and wake source while holding only try-locks.
    /// No request, ticket, cursor, frontier, capacity epoch or resource changes.
    /// A fresh snapshot and ordinary selection are still required afterwards.
    pub fn try_consume_maintenance_fairness_turn(
        &self,
        expected: &PlanningQueueSnapshot,
        wake: AdmissionWakeSnapshot<'_>,
    ) -> PlanningResult<PlanningMaintenanceFairnessOutcome> {
        use PlanningMaintenanceFairnessOutcome::*;
        if !Arc::ptr_eq(&expected.owner, &self.planning_owner) {
            return Ok(Stale);
        }
        let waiting = self
            .waiting_queue
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let prefill = self
            .prefill_queue
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let decode = self
            .decode_queue
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let preempted = self
            .preempted_requests
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let index = self
            .request_index
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let pressure = self
            .pressure_coordinator
            .try_lock()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let actual = self.project_planning_state(
            expected.maximum_requests,
            wake,
            &Queues {
                waiting: &waiting,
                prefill: &prefill,
                decode: &decode,
                preempted: &preempted,
                index: &index,
                pressure: &pressure,
            },
        )?;
        if !expected.matches(&actual) {
            return Ok(Stale);
        }
        if !actual
            .requests
            .iter()
            .any(|row| row.readiness.maintenance_blocked)
        {
            return Ok(NoPending);
        }
        let previous_iteration = actual.seal.iteration;
        let next_iteration = previous_iteration
            .checked_add(1)
            .ok_or(PlanningStateUnavailable::CounterExhausted)?;
        let matured_tickets = actual
            .requests
            .iter()
            .filter(|row| {
                row.readiness.maintenance_blocked
                    && row
                        .seal
                        .maintenance
                        .is_some_and(|ticket| ticket.not_before_iteration == next_iteration)
            })
            .count();
        if matured_tickets == 0 {
            return Err(PlanningStateUnavailable::InvalidState(
                "maintenance fairness ticket exceeds one empty turn",
            ));
        }
        if self
            .current_iteration
            .compare_exchange(
                previous_iteration,
                next_iteration,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return Ok(Stale);
        }
        self.metrics_tracker.record_iteration();
        Ok(Advanced {
            previous_iteration,
            next_iteration,
            matured_tickets,
        })
    }

    /// Publish only the exact selected logical work after rechecking the entire
    /// obligation set. The caller must serialize execution iterations and must
    /// separately revalidate engine/output/resource evidence before submission.
    pub fn try_select_planned_wave(
        &self,
        expected: &PlanningQueueSnapshot,
        selected: &[PlanningWorkSelection],
        hint: &BatchHint,
        wake: AdmissionWakeSnapshot<'_>,
    ) -> PlanningResult<PlanningSelectionOutcome> {
        if !Arc::ptr_eq(&expected.owner, &self.planning_owner) {
            return Ok(PlanningSelectionOutcome::Stale);
        }
        if selected.is_empty()
            || selected.len() > expected.maximum_requests.get()
            || selected.len() > hint.max_batch_size
        {
            return Ok(PlanningSelectionOutcome::Rejected("invalid wave width"));
        }
        // Some legacy paths take these locks in different orders. Never wait
        // with a partial set: failed try_lock drops every earlier guard.
        let waiting = self
            .waiting_queue
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let mut prefill = self
            .prefill_queue
            .try_write()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let mut decode = self
            .decode_queue
            .try_write()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let preempted = self
            .preempted_requests
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let index = self
            .request_index
            .try_read()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let pressure = self
            .pressure_coordinator
            .try_lock()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let actual = self.project_planning_state(
            expected.maximum_requests,
            wake,
            &Queues {
                waiting: &waiting,
                prefill: &prefill,
                decode: &decode,
                preempted: &preempted,
                index: &index,
                pressure: &pressure,
            },
        )?;
        if !expected.matches(&actual) {
            return Ok(PlanningSelectionOutcome::Stale);
        }

        let mut work = Vec::new();
        work.try_reserve_exact(selected.len())
            .map_err(|_| PlanningStateUnavailable::AllocationFailed)?;
        let mut seen = HashSet::new();
        seen.try_reserve(selected.len())
            .map_err(|_| PlanningStateUnavailable::AllocationFailed)?;
        let mut total_tokens = 0usize;
        let mut receipt_rows = Vec::new();
        receipt_rows
            .try_reserve_exact(selected.len())
            .map_err(|_| PlanningStateUnavailable::AllocationFailed)?;
        let mut decode_count = 0usize;
        let mut prefill_count = 0usize;
        for choice in selected {
            if !seen.insert(&choice.key.request_id) {
                return Ok(PlanningSelectionOutcome::Rejected(
                    "duplicate selected request",
                ));
            }
            let Some(view) = actual.requests.iter().find(|r| r.key == choice.key) else {
                return Ok(PlanningSelectionOutcome::Stale);
            };
            if !view.readiness.ready() {
                return Ok(PlanningSelectionOutcome::Rejected(
                    "selected request is not ready",
                ));
            }
            let (request, offset, count) = match choice.action {
                PlanningWorkAction::Decode if view.queue == PlanningQueueKind::Decode => {
                    decode_count += 1;
                    let request = decode.requests.get(&choice.key.request_id).unwrap();
                    // decode_tokens is cumulative output, including tokens
                    // already replayed by a later prefill. The sealed physical
                    // KV frontier is the next decode input position.
                    (request, view.computed_tokens, 1)
                }
                PlanningWorkAction::Prefill { offset, count }
                    if view.queue == PlanningQueueKind::Prefill =>
                {
                    prefill_count += 1;
                    let Some(target) = view.prefill_context_tokens else {
                        return Ok(PlanningSelectionOutcome::Rejected(
                            "unknown full prefill boundary",
                        ));
                    };
                    let Some(end) = offset.checked_add(count.get()) else {
                        return Ok(PlanningSelectionOutcome::Rejected("prefill span overflow"));
                    };
                    if offset != view.prefill_offset
                        || end > target
                        || view
                            .prefill_chunk_ceiling
                            .is_some_and(|ceiling| count.get() > ceiling)
                        || !prefix_span_valid(view.seal.prefix_rendezvous.1, offset, count.get())
                    {
                        return Ok(PlanningSelectionOutcome::Rejected(
                            "invalid selected prefill span",
                        ));
                    }
                    (
                        prefill
                            .iter()
                            .find(|r| r.inner.request.id == choice.key.request_id)
                            .unwrap(),
                        offset,
                        count.get(),
                    )
                }
                _ => {
                    return Ok(PlanningSelectionOutcome::Rejected(
                        "selected phase mismatch",
                    ))
                }
            };
            total_tokens = total_tokens
                .checked_add(count)
                .ok_or(PlanningStateUnavailable::CounterExhausted)?;
            view.computed_tokens
                .checked_add(count)
                .ok_or(PlanningStateUnavailable::CounterExhausted)?;
            if total_tokens > hint.max_tokens {
                return Ok(PlanningSelectionOutcome::Rejected(
                    "selected token budget exceeded",
                ));
            }
            let mut scheduled = request.inner.clone();
            scheduled.tokens_processed = offset;
            scheduled.tokens_to_process = Some(count);
            work.push(scheduled);
            let mut projected = request.logical_work_frontier.clone();
            projected.mark_scheduled(count);
            receipt_rows.push((choice.key.clone(), projected));
        }
        if decode_count > self.cb_config.max_decode_batch
            || decode_count > actual.seal.decode_limit
            // The existing mixed path spends the remaining hint slots on
            // prefills; max_prefill_batch bounds a prefill-only cohort.
            || (decode_count == 0 && prefill_count > self.cb_config.max_prefill_batch)
        {
            return Ok(PlanningSelectionOutcome::Rejected(
                "scheduler width ceiling exceeded",
            ));
        }

        let next_iteration = actual
            .seal
            .iteration
            .checked_add(1)
            .ok_or(PlanningStateUnavailable::CounterExhausted)?;
        // This consumes the existing scheduler iteration, not a new global
        // revision required on every legacy mutator. Replaying the snapshot
        // cannot publish the same logical wave twice.
        if self
            .current_iteration
            .compare_exchange(
                actual.seal.iteration,
                next_iteration,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return Ok(PlanningSelectionOutcome::Stale);
        }
        for choice in selected {
            let request = match choice.action {
                PlanningWorkAction::Decode => {
                    decode.requests.get_mut(&choice.key.request_id).unwrap()
                }
                PlanningWorkAction::Prefill { .. } => prefill
                    .iter_mut()
                    .find(|r| r.inner.request.id == choice.key.request_id)
                    .unwrap(),
            };
            let count = match choice.action {
                PlanningWorkAction::Decode => 1,
                PlanningWorkAction::Prefill { count, .. } => count.get(),
            };
            request.logical_work_frontier.mark_scheduled(count);
            request.last_iteration = actual.seal.iteration;
            // The matching snapshot established that these old gates are open.
            request.execution_capacity_deferral = None;
            request.execution_readiness_block = None;
            request.execution_maintenance_retry = None;
        }
        if let Some(last) = selected
            .iter()
            .rev()
            .find(|choice| choice.action == PlanningWorkAction::Decode)
        {
            let index = decode.requests.get_index_of(&last.key.request_id).unwrap();
            decode.selection_cursor = decode
                .requests
                .get_index((index + 1) % decode.requests.len())
                .map(|(id, _)| id.clone());
        }
        self.metrics_tracker.record_iteration();
        let max_sequence_length = work
            .iter()
            .map(|r| r.request.sampling_params.max_tokens)
            .max()
            .unwrap_or(0);
        Ok(PlanningSelectionOutcome::Published {
            batch: BatchPlan {
                batch_id: BatchId::new(),
                requests: work,
                max_sequence_length,
                estimated_time_ms: None,
                // Estimates here cannot replace the executor's real admission.
                resource_requirements: BatchResourceRequirements::default(),
                created_at: chrono::Utc::now(),
            },
            receipt: PlanningPublicationReceipt {
                owner: Arc::clone(&self.planning_owner),
                iteration: actual.seal.iteration,
                rows: receipt_rows,
            },
        })
    }

    /// Reopen logical scheduling after the caller's typed executor result
    /// proved that no work was submitted. Does not release/restore any physical
    /// state. The caller retains the receipt on Busy so contention cannot strand
    /// a proven-unsubmitted wave. Exact frontiers make repeated release harmless.
    /// Never call for execution errors.
    pub fn release_unsubmitted_planned_wave(
        &self,
        receipt: &PlanningPublicationReceipt,
    ) -> PlanningResult<PlanningPublicationRelease> {
        if !Arc::ptr_eq(&receipt.owner, &self.planning_owner) {
            return Err(PlanningStateUnavailable::InvalidState(
                "foreign publication receipt",
            ));
        }
        let mut prefill = self
            .prefill_queue
            .try_write()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let mut decode = self
            .decode_queue
            .try_write()
            .ok_or(PlanningStateUnavailable::Busy)?;
        let mut outcome = PlanningPublicationRelease {
            released_rows: 0,
            superseded_rows: 0,
        };
        for (key, frontier) in &receipt.rows {
            let request = if let Some(request) = prefill
                .iter_mut()
                .find(|r| r.inner.request.id == key.request_id)
            {
                Some(request)
            } else {
                decode.requests.get_mut(&key.request_id)
            };
            if let Some(request) = request.filter(|request| {
                request.waiting_admission_ticket == Some(key.ticket)
                    && request.logical_work_frontier.progress_generation() == key.generation
                    && &request.logical_work_frontier == frontier
                    && request.last_iteration == receipt.iteration
            }) {
                request.logical_work_frontier.mark_scheduled(0);
                outcome.released_rows += 1;
            } else {
                outcome.superseded_rows += 1;
            }
        }
        Ok(outcome)
    }
}

pub(super) fn prefix_span_valid(
    source: Option<(usize, u64, u64)>,
    offset: usize,
    count: usize,
) -> bool {
    let Some((boundary, alignment, minimum)) = source else {
        return true;
    };
    let (Ok(alignment), Ok(minimum)) = (usize::try_from(alignment), usize::try_from(minimum))
    else {
        return false;
    };
    if alignment == 0 || minimum == 0 {
        return false;
    }
    let Some(rounded) = minimum
        .checked_add(alignment - 1)
        .and_then(|n| n.checked_div(alignment))
        .and_then(|n| n.checked_mul(alignment))
    else {
        return false;
    };
    let Some(remaining) = boundary.checked_sub(offset) else {
        return false;
    };
    if remaining < rounded
        || !remaining.is_multiple_of(alignment)
        || count == 0
        || count > remaining
        || !count.is_multiple_of(alignment)
    {
        return false;
    }
    count == remaining || (count >= rounded && remaining - count >= rounded)
}
