use super::*;

impl ContinuousBatchScheduler {
    pub(super) fn project_planning_state(
        &self,
        maximum_requests: NonZeroUsize,
        wake: AdmissionWakeSnapshot<'_>,
        queues: &Queues<'_>,
    ) -> PlanningResult<PlanningQueueSnapshot> {
        if maximum_requests.get() > MAX_PLANNING_REQUESTS {
            return Err(PlanningStateUnavailable::Unsupported(
                "planning request limit exceeds hard bound",
            ));
        }
        let count = queues
            .waiting
            .len()
            .checked_add(queues.prefill.len())
            .and_then(|n| n.checked_add(queues.decode.requests.len()))
            .and_then(|n| n.checked_add(queues.preempted.len()))
            .ok_or(PlanningStateUnavailable::CounterExhausted)?;
        if count > maximum_requests.get() || queues.index.len() > maximum_requests.get() {
            return Err(PlanningStateUnavailable::TooManyRequests {
                actual: count.max(queues.index.len()),
                maximum: maximum_requests.get(),
            });
        }
        if queues.index.len() != count {
            return Err(PlanningStateUnavailable::Unsupported(
                "terminal cleanup or queue transition pending",
            ));
        }
        if wake.availability().len() > MAX_PLANNING_SOURCES {
            return Err(PlanningStateUnavailable::TooManyCapacitySources);
        }
        if wake
            .availability()
            .windows(2)
            .any(|w| w[0].source() >= w[1].source())
        {
            return Err(PlanningStateUnavailable::InvalidWake);
        }
        let iteration = self.current_iteration.load(Ordering::Acquire);
        let mut requests = Vec::new();
        requests
            .try_reserve_exact(count)
            .map_err(|_| PlanningStateUnavailable::AllocationFailed)?;
        let mut seen = HashSet::new();
        seen.try_reserve(count)
            .map_err(|_| PlanningStateUnavailable::AllocationFailed)?;
        let mut source_count = wake.availability().len();
        let mut append = |request: &ContinuousBatchRequest, queue: PlanningQueueKind| {
            if !seen.insert(request.inner.request.id.clone())
                || queues.index.get(&request.inner.request.id) != Some(&request.phase)
            {
                return Err(PlanningStateUnavailable::InvalidState(
                    "duplicate or mismatched obligation",
                ));
            }
            requests.push(project_request(
                request,
                queue,
                requests.len(),
                iteration,
                wake,
                queues.pressure,
                &mut source_count,
            )?);
            Ok(())
        };
        for request in queues.waiting.iter() {
            append(request, PlanningQueueKind::Waiting)?;
        }
        for request in queues.prefill.iter() {
            append(request, PlanningQueueKind::Prefill)?;
        }
        let decode_start = queues
            .decode
            .selection_cursor
            .as_ref()
            .and_then(|id| queues.decode.requests.get_index_of(id))
            .unwrap_or(0);
        for offset in 0..queues.decode.requests.len() {
            let (_, request) = queues
                .decode
                .requests
                .get_index((decode_start + offset) % queues.decode.requests.len())
                .unwrap();
            append(request, PlanningQueueKind::Decode)?;
        }
        let mut preempted = Vec::new();
        preempted
            .try_reserve_exact(queues.preempted.len())
            .map_err(|_| PlanningStateUnavailable::AllocationFailed)?;
        preempted.extend(queues.preempted.values());
        preempted.sort_unstable_by_key(|request| request.waiting_admission_ticket);
        for request in preempted {
            append(request, PlanningQueueKind::Preempted)?;
        }
        let mut availability = Vec::new();
        availability
            .try_reserve_exact(wake.availability().len())
            .map_err(|_| PlanningStateUnavailable::AllocationFailed)?;
        availability.extend_from_slice(wake.availability());
        Ok(PlanningQueueSnapshot {
            owner: Arc::clone(&self.planning_owner),
            maximum_requests,
            requests,
            seal: QueueSeal {
                iteration,
                pressure_revision: queues.pressure.planning_revision(),
                decode_cursor: queues.decode.selection_cursor.clone(),
                wake: wake.epochs(),
                availability,
                capacity_release: self.capacity_release_epoch.load(Ordering::Acquire),
                mixed_epoch: self.capacity_mixed_recompute_epoch.load(Ordering::Acquire),
                decode_limit: self
                    .decode_capacity_backpressure_limit
                    .load(Ordering::Acquire)
                    .min(self.cb_config.max_decode_batch),
                prefill_limit: self
                    .capacity_backpressure_limit
                    .load(Ordering::Acquire)
                    .min(self.cb_config.max_prefill_batch),
            },
        })
    }
}

fn project_request(
    request: &ContinuousBatchRequest,
    queue: PlanningQueueKind,
    fairness_rank: usize,
    iteration: u64,
    wake: AdmissionWakeSnapshot<'_>,
    pressure: &PressureCoordinator,
    source_count: &mut usize,
) -> PlanningResult<PlanningRequestState> {
    let expected_phase = match queue {
        PlanningQueueKind::Waiting => RequestPhase::Waiting,
        PlanningQueueKind::Prefill => RequestPhase::Prefilling,
        PlanningQueueKind::Decode => RequestPhase::Decoding,
        PlanningQueueKind::Preempted => RequestPhase::Preempted,
    };
    if request.phase != expected_phase {
        return Err(PlanningStateUnavailable::InvalidState(
            "queue phase mismatch",
        ));
    }
    let ticket = request
        .waiting_admission_ticket
        .ok_or(PlanningStateUnavailable::InvalidState(
            "missing admission identity",
        ))?;
    let generation = request.logical_work_frontier.progress_generation();
    if generation.get() == u64::MAX {
        return Err(PlanningStateUnavailable::CounterExhausted);
    }
    let capacity_blocked = if let Some(deferral) = &request.execution_capacity_deferral {
        *source_count = source_count
            .checked_add(deferral.wait_condition().observed().len())
            .ok_or(PlanningStateUnavailable::CounterExhausted)?;
        if *source_count > MAX_PLANNING_SOURCES {
            return Err(PlanningStateUnavailable::TooManyCapacitySources);
        }
        let old = deferral.observed();
        let now = wake.epochs();
        if old.coordinator_id() != now.coordinator_id()
            || deferral.wait_condition().coordinator_id().get() != now.coordinator_id().get()
            || now.release_epoch() < old.release_epoch()
            || now.capacity_epoch() < old.capacity_epoch()
            || now.policy_epoch() < old.policy_epoch()
        {
            return Err(PlanningStateUnavailable::InvalidWake);
        }
        let changed = deferral
            .wait_condition()
            .changed_since(wake.availability())
            .map_err(|_| PlanningStateUnavailable::InvalidWake)?;
        !changed && old.policy_epoch() == now.policy_epoch()
    } else {
        false
    };
    let readiness = request
        .execution_readiness_block
        .as_ref()
        .map(|block| (block.ticket_id, block.status()));
    let execution_blocked = readiness.is_some_and(|(_, status)| {
        !matches!(
            status,
            EXECUTION_READINESS_READY | EXECUTION_READINESS_CANCELLED
        )
    });
    if let Some(ticket) = request.execution_maintenance_retry {
        if request.last_execution_maintenance_capacity_epoch != Some(ticket.latest_capacity_epoch) {
            return Err(PlanningStateUnavailable::InvalidState(
                "maintenance receipt mismatch",
            ));
        }
    }
    let prefix_restore = request.prefix_restore.planning_state();
    let prefix_rendezvous = request.prefix_rendezvous.planning_state();
    let (computed, resident, scheduled, committed, recompute) =
        request.logical_work_frontier.planning_counters();
    let original_prompt_tokens = request
        .inner
        .request
        .metadata
        .get(PROMPT_TOKENS_METADATA_KEY)
        .and_then(|value| value.as_u64())
        .and_then(|n| usize::try_from(n).ok());
    let prompt_tokens = if request.prefill_tokens != 0 {
        Some(request.prefill_tokens)
    } else {
        original_prompt_tokens
    };
    // Product ingress overwrites this metadata from its actual token vector.
    // Completed-output receipts carry the only additional logical tokens.
    // A previous recompute's prefill_tokens must not be added a second time.
    let prefill_context_tokens = if queue == PlanningQueueKind::Decode {
        None
    } else {
        prefill_context(original_prompt_tokens, committed)?
    };
    Ok(PlanningRequestState {
        key: PlanningRequestKey {
            request_id: request.inner.request.id.clone(),
            ticket,
            generation,
        },
        queue,
        phase: request.phase,
        priority: request.inner.request.priority,
        fairness_rank,
        readiness: PlanningReadiness {
            output_limit_reached: committed >= request.inner.request.sampling_params.max_tokens,
            unfinished_work: scheduled != computed,
            admitted: matches!(
                queue,
                PlanningQueueKind::Prefill | PlanningQueueKind::Decode
            ) && request.inner.state == RequestState::Running,
            pressure_held: pressure.planning_is_held(&request.inner.request.id),
            execution_blocked,
            capacity_blocked,
            maintenance_blocked: request
                .execution_maintenance_retry
                .is_some_and(|ticket| iteration < ticket.not_before_iteration),
            prefix_blocked: prefix_restore.0 || prefix_restore.2 || prefix_rendezvous.0,
        },
        computed_tokens: computed,
        resident_tokens: resident,
        scheduled_tokens: scheduled,
        committed_output_tokens: committed,
        recompute_target_tokens: recompute,
        prompt_tokens,
        prefill_context_tokens,
        prefill_offset: request.prefill_chunk_offset,
        maximum_output_tokens: request.inner.request.sampling_params.max_tokens,
        prefill_chunk_ceiling: request.prefill_execution_chunk_ceiling,
        restored_tokens: prefix_restore.1,
        seal: RequestSeal {
            frontier: request.logical_work_frontier.clone(),
            state: request.inner.state,
            prefill_tokens: request.prefill_tokens,
            decode_tokens: request.decode_tokens,
            readiness,
            deferral: request.execution_capacity_deferral.clone(),
            maintenance: request.execution_maintenance_retry,
            last_maintenance_epoch: request.last_execution_maintenance_capacity_epoch,
            capacity_deferred_until: request.capacity_deferred_until_release_epoch,
            mixed_attempt: request.capacity_deferred_mixed_attempt_epoch,
            empty_retry: request.capacity_deferred_empty_retry_epoch,
            from_decode: request.capacity_deferred_from_decode,
            last_iteration: request.last_iteration,
            prefix_restore,
            prefix_rendezvous,
        },
    })
}

pub(super) fn prefill_context(
    original_prompt_tokens: Option<usize>,
    committed_output_tokens: usize,
) -> PlanningResult<Option<usize>> {
    original_prompt_tokens
        .map(|original| {
            original
                .checked_add(committed_output_tokens)
                .ok_or(PlanningStateUnavailable::CounterExhausted)
        })
        .transpose()
}
