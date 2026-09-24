use super::*;
use crate::continuous_engine::output_flow_runtime::OutputPlanningCreditView;
use ferrum_scheduler::implementations::continuous::cost_model::{
    BatchOrderSemantics, WaveExecutionPath, WaveGraphState,
};

impl EngineInner {
    pub(super) fn capture_slo_controller_snapshot(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        controller_budget: Arc<ControllerBudget>,
    ) -> ControllerResult<ControllerSnapshot> {
        let fallback_count = self.scheduler.active_count() + self.scheduler.waiting_count();
        let unavailable = |reason| Unavailable {
            reason,
            obligations: fallback_count,
            retry: None,
        };
        if self.model_executor.execution_resource_authority()
            != ExecutionResourceAuthority::PlanRuntime
            || self.spec_config.is_some()
        {
            return Err(unavailable("unsupported_execution_authority"));
        }
        let mut budget = || controller_budget.poll();
        if !budget() {
            return Err(unavailable("compute_budget_exhausted"));
        }
        let mut availability = self
            .dynamic_admission_availability
            .try_lock()
            .ok_or_else(|| unavailable("capacity_snapshot_busy"))?;
        let epochs = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)
            .map_err(|_| unavailable("capacity_snapshot_unavailable"))?
            .ok_or_else(|| unavailable("capacity_epochs_unavailable"))?;
        let queue = self
            .scheduler
            .planning_state(
                NonZeroUsize::new(256).unwrap(),
                AdmissionWakeSnapshot::new(
                    AdmissionWakeEpochs::new(
                        epochs.coordinator_id,
                        epochs.release_epoch,
                        epochs.capacity_epoch,
                        0,
                    ),
                    &availability,
                ),
            )
            .map_err(|_| unavailable("scheduler_snapshot_unavailable"))?;
        drop(availability);
        let count = queue.requests().len();
        let unavailable = |reason| Unavailable {
            reason,
            obligations: count,
            retry: None,
        };
        if count == 0 {
            return Err(unavailable("no_work"));
        }
        let sequences = self
            .sequences
            .try_read()
            .ok_or_else(|| unavailable("sequence_snapshot_busy"))?;
        // A scheduler request missing its engine owner, or an additional live
        // owner omitted by the scheduler, invalidates the complete transaction.
        if sequences.len() != count
            || queue
                .requests()
                .iter()
                .any(|row| !sequences.contains_key(&row.key.request_id))
        {
            return Err(unavailable("obligation_owner_mismatch"));
        }
        let observed_at = slo_clock_now();
        let states = queue
            .requests()
            .iter()
            .map(|row| sequences[&row.key.request_id].slo.as_ref())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| unavailable("untrusted_timing"))?;
        let origin =
            PlanningTimeOrigin::from_ingress(observed_at, NonZeroUsize::new(256).unwrap(), states)
                .map_err(|_| unavailable("invalid_timing"))?;
        let runtime = self
            .cost_runtime
            .as_ref()
            .ok_or_else(|| unavailable("cost_runtime_unavailable"))?;
        let model = runtime
            .try_snapshot()
            .ok_or_else(|| unavailable("cost_snapshot_busy"))?
            .ok_or_else(|| unavailable("cost_unavailable"))?;
        let reference = self
            .prefill_reference_runtime
            .as_ref()
            .ok_or_else(|| unavailable("missing_reference_work"))?
            .calibration();
        let caps = self.model_executor.capabilities();
        // VNext owns recurrent state in PlanRuntime resources, not in the
        // legacy sequence handle. Match the logical per-owner state consumed
        // by both its actual observation and future route projection, even
        // before the first prefill has materialized that state. KV growth is a
        // separate context axis; other token-scaled state has no cost contract.
        let fixed_state_bytes = match caps.memory_requirements.typed_sequence_state {
            Some(state) if state.other_token_scaled_bytes_per_token != 0 => {
                return Err(unavailable("non_kv_token_scaled_state_unsupported"));
            }
            Some(state) => Some(state.fixed_bytes_per_sequence),
            None => None,
        };
        let mut reference_points = 0_usize;
        let mut milestone_count = 0_usize;
        let mut reference_chunks = std::collections::BTreeSet::new();
        let mut fences = Vec::with_capacity(count);
        let mut requests = Vec::with_capacity(count);
        let mut engine_resource_requests = Vec::with_capacity(count);
        for row in queue.requests() {
            if !budget() {
                return Err(unavailable("compute_budget_exhausted"));
            }
            let sequence = &sequences[&row.key.request_id];
            // Do not copy an unbounded sampling history into a planning view.
            // More policies can join when their exact bounded route is known.
            if sequence.sampling_params.repetition_penalty != 1.0 {
                return Err(unavailable("sampling_route_unsupported"));
            }
            let frontier = sequence
                .cost_frontier
                .ok_or_else(|| unavailable("engine_frontier_unknown"))?;
            let is_decode = row.queue == PlanningQueueKind::Decode;
            if is_decode != sequence.prefill_complete
                || row.committed_output_tokens != sequence.generated_tokens.len()
                || (!is_decode
                    && (row.prefill_offset != sequence.prefill_tokens_processed
                        || row.computed_tokens != row.prefill_offset))
            {
                return Err(unavailable("frontier_mismatch"));
            }
            let maximum = u32::try_from(sequence.sampling_params.max_tokens)
                .ok()
                .and_then(NonZeroU32::new)
                .ok_or_else(|| unavailable("output_limit_unrepresentable"))?;
            let timing = origin
                .project_request(sequence.slo.as_ref().unwrap(), maximum)
                .map_err(|_| unavailable("invalid_timing"))?;
            let output = sequence
                .credited_output
                .as_ref()
                .ok_or_else(|| unavailable("unbounded_output"))?;
            let view = output.port.planning_snapshot();
            let credit = match view.future_capacity {
                Some(credit) => credit
                    .try_into()
                    .map_err(|_| unavailable("output_capacity_overflow"))?,
                None => OutputCreditView {
                    available_token_commands: 0,
                    byte_backing: OutputByteBacking::Incremental {
                        available_bytes: 0,
                        bytes_per_token_upper_bound: None,
                    },
                },
            };
            let context = sequence
                .model_kv
                .as_ref()
                .map_or(0, |kv| kv.handle().num_tokens());
            let context_tokens =
                u32::try_from(context).map_err(|_| unavailable("context_overflow"))?;
            let cache_id = sequence.model_cache_id();
            if is_decode && cache_id.is_none() {
                return Err(unavailable("cache_identity_missing"));
            }
            let output_policy_signature = sequence
                .cost_policy_signature
                .ok_or_else(|| unavailable("host_policy_unknown"))?;
            let recurrent_state_bytes = match fixed_state_bytes {
                Some(bytes) => bytes,
                None => u64::try_from(
                    sequence
                        .recurrent_state
                        .as_ref()
                        .map_or(0, |state| state.handle().state_bytes()),
                )
                .map_err(|_| unavailable("state_size_overflow"))?,
            };
            let readiness = if matches!(view.readiness, OutputPlanningCreditView::OutputBlocked(_))
            {
                RequestReadiness::OutputBlocked
            } else if view.future_capacity.is_none() {
                RequestReadiness::StateBlocked
            } else if row.readiness.ready() {
                RequestReadiness::Ready
            } else if row.readiness.capacity_blocked || row.readiness.pressure_held {
                RequestReadiness::ResourceBlocked
            } else {
                RequestReadiness::StateBlocked
            };
            let phase = if is_decode {
                RequestPhaseView::Decode
            } else {
                let total = NonZeroU32::new(
                    u32::try_from(sequence.prefill_context_len())
                        .map_err(|_| unavailable("context_overflow"))?,
                )
                .ok_or_else(|| unavailable("empty_prefill"))?;
                // This is the exact upcoming input, not the historical KV
                // frontier evicted by capacity yield. The reference below
                // still requires its original calibrated prompt length.
                if row.prefill_context_tokens != Some(total.get() as usize) {
                    return Err(unavailable("prefill_boundary_mismatch"));
                }
                let offset = u32::try_from(row.prefill_offset)
                    .map_err(|_| unavailable("context_overflow"))?;
                let executable_until = row
                    .prefill_chunk_ceiling
                    .and_then(|limit| row.prefill_offset.checked_add(limit))
                    .unwrap_or(total.get() as usize)
                    .min(total.get() as usize) as u32;
                let binding = sequence
                    .prefill_reference
                    .as_ref()
                    .ok_or_else(|| unavailable("missing_reference_work"))?
                    .known()
                    .map_err(|_| unavailable("missing_reference_work"))?;
                // Physical state can return to zero after eviction. Useful
                // reference credit belongs to the original request binding
                // and must survive replay without being credited twice.
                let high_water = binding.binding().logical_high_water();
                if binding.identity() != reference.identity()
                    || binding.tau_ref_ns() != reference.tau_ref_ns()
                    || binding.total_prompt_tokens() != total
                    || offset > high_water
                    || high_water > total.get()
                    || row
                        .recompute_target_tokens
                        .is_some_and(|target| target > high_water as usize)
                {
                    return Err(unavailable("reference_frontier_mismatch"));
                }
                let points = &binding.binding().reference().points;
                reference_points = reference_points
                    .checked_add(binding.binding().reference().evidence_point_count())
                    .filter(|count| *count <= 8192)
                    .ok_or_else(|| unavailable("reference_point_budget"))?;
                milestone_count = milestone_count
                    .checked_add(points.len().saturating_sub(1))
                    .filter(|count| *count <= 4096)
                    .ok_or_else(|| unavailable("reference_milestone_budget"))?;
                let base = binding
                    .project_progress(&origin, offset, executable_until, &[])
                    .map_err(|_| unavailable("missing_reference_work"))?;
                let deadline = timing
                    .ingress_at_ns
                    .checked_add(timing.budgets.ttft_ns.get())
                    .ok_or_else(|| unavailable("invalid_timing"))?;
                let checkpoints = calibrated_checkpoints(&base, deadline)
                    .ok_or_else(|| unavailable("invalid_reference_milestones"))?;
                let legal_granule = self
                    .model_executor
                    .guarded_prefill_granularity()
                    .and_then(|n| u32::try_from(n.get()).ok())
                    .and_then(NonZeroU32::new);
                if reference.piecewise_domain().is_some() && legal_granule.is_none() {
                    return Err(unavailable("missing_reference_granule"));
                }
                let progress = binding
                    .project_progress_with_granule(
                        &origin,
                        offset,
                        executable_until,
                        &checkpoints,
                        legal_granule,
                    )
                    .map_err(|_| unavailable("missing_reference_work"))?;
                // Adjacent calibrated segments remain available after the
                // first selected wave; current legal merged endpoints enrich
                // the finite candidate set without interpolating unknown work.
                for pair in points
                    .windows(2)
                    .filter(|_| reference.piecewise_domain().is_none())
                {
                    if !budget() {
                        return Err(unavailable("compute_budget_exhausted"));
                    }
                    let count = pair[1].prompt_tokens - pair[0].prompt_tokens;
                    if count as usize <= hint.max_tokens {
                        reference_chunks.insert(NonZeroU32::new(count).unwrap());
                    }
                }
                let maximum_tokens =
                    NonZeroU32::new(u32::try_from(hint.max_tokens.min(u32::MAX as usize)).unwrap())
                        .ok_or_else(|| unavailable("invalid_wave_capacity"))?;
                let chunks = reference.legal_chunks(total, offset,
                    ferrum_scheduler::implementations::continuous::prefill_reference::ReferenceChunkLimits {
                        maximum_tokens, alignment: if reference.piecewise_domain().is_some() { legal_granule.unwrap() } else { NonZeroU32::MIN }, allow_final_short_chunk: true,
                        maximum_candidates: NonZeroUsize::new(64).unwrap(),
                    }).map_err(|_| unavailable("missing_reference_work"))?;
                reference_chunks.extend(chunks);
                if reference_chunks.len() > 64 {
                    return Err(unavailable("reference_candidate_budget"));
                }
                RequestPhaseView::Prefill(progress)
            };
            requests.push(RequestSchedulingView {
                key: RequestWorkKey {
                    request_id: row.key.request_id.clone(),
                    incarnation: frontier.owner_incarnation.get(),
                    work_generation: row.key.generation,
                },
                timing,
                phase,
                readiness,
                context_tokens,
                recurrent_state_bytes,
                output_credit: credit,
                output_policy_signature,
                fairness_rank: row.fairness_rank as u64,
                recovery_service: sequence.time_admission.as_ref().map_or_else(
                    || {
                        RecoveryServiceDebt::new(
                            self.config.scheduler.slo.admission.max_active_requests,
                        )
                    },
                    |state| state.recovery_service,
                ),
                ranking_service_cost_ns: None,
                optimistic_next_service: None,
            });
            let host_features = super::super::cost_observation::participant_host_features(sequence);
            let future_greedy_policy = host_features
                .is_some_and(|host| host.supports_empirical_plain_text_content())
                .then(
                    || ferrum_interfaces::model_executor::LogitsReturnPolicy::GreedyArgmax {
                        token_mask: sequence.argmax_token_mask.clone(),
                        repetition_penalty: None,
                    },
                );
            fences.push(EngineFence {
                key: row.key.clone(),
                incarnation: frontier.owner_incarnation.get(),
                generation: frontier.work_generation.get(),
                generated: sequence.generated_tokens.len(),
                context,
                output: view,
                cache_id: cache_id.map(str::to_owned),
                prefill_complete: sequence.prefill_complete,
                prefill_tokens_processed: sequence.prefill_tokens_processed,
                prefill_total: sequence.prefill_context_len(),
                logits_policy: sequence.model_decode_logits_policy(),
                future_greedy_policy,
                host_features,
            });
            engine_resource_requests.push(ExecutorResourcePlanningRequest {
                request_id: &sequence.request_id,
                cache_id: if is_decode { cache_id } else { None },
            });
        }
        let limits = ResourcePlanningLimits {
            maximum_participants: 256,
            maximum_projected_waves: self.config.scheduler.slo.planner.lookahead_waves.get(),
            ..Default::default()
        };
        let route = match self.model_executor.execution_cost_route_view(
            &engine_resource_requests,
            limits,
            &mut budget,
        ) {
            ferrum_interfaces::vnext::ExecutionCostRouteAvailability::Known(value) => value,
            ferrum_interfaces::vnext::ExecutionCostRouteAvailability::Unknown(
                ferrum_interfaces::vnext::ExecutionCostRouteUnknown::Resource(reason),
            ) => {
                return Err(unavailable(unknown_label(resources::resource_reason(
                    reason,
                ))))
            }
            ferrum_interfaces::vnext::ExecutionCostRouteAvailability::Unknown(
                ferrum_interfaces::vnext::ExecutionCostRouteUnknown::BudgetExhausted,
            ) => return Err(unavailable("compute_budget_exhausted")),
            ferrum_interfaces::vnext::ExecutionCostRouteAvailability::Unknown(_) => {
                return Err(unavailable("shape_unavailable"))
            }
        };
        // Route and resource transitions must start in the same captured epoch.
        // The complete route view already contains the lane-bearing resource
        // evidence; a second capture has a different private state fence even
        // when its public counters happen to match. This is numeric evidence,
        // not a resource reservation. Publication still revalidates both views.
        let resources = route.resource_view().clone();
        if !budget() {
            return Err(unavailable("compute_budget_exhausted"));
        }
        drop(engine_resource_requests);
        drop(sequences);
        let before = slo_clock_now();
        let cost_at = runtime
            .clock
            .now_ns()
            .ok_or_else(|| unavailable("cost_clock_unavailable"))?;
        let after = slo_clock_now();
        let anchor = PlanningCostClockAnchor::conservative_read(&origin, before, cost_at, after)
            .map_err(|_| unavailable("cost_clock_mismatch"))?;
        let maximum_rows = hint.max_batch_size.min(caps.max_batch_size).min(256);
        let max_wave_rows =
            NonZeroUsize::new(maximum_rows).ok_or_else(|| unavailable("invalid_wave_capacity"))?;
        let maximum_context_tokens = u32::try_from(caps.max_sequence_length)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| unavailable("context_limit_unrepresentable"))?;
        // A past deadline must not place the entire recovery horizon in the
        // past. Keep the first future obligation, or one finite reference span
        // when every owner is already late; no request clock is rewritten.
        let classified_at_ns = origin
            .at_ns(slo_clock_now())
            .map_err(|_| unavailable("clock_mismatch"))?;
        let horizon_end_ns = requests
            .iter()
            .filter(|row| !row.timing.completed())
            .filter_map(|row| row.timing.next_deadline_ns())
            .filter(|deadline| *deadline >= classified_at_ns)
            .min()
            .or_else(|| {
                reference
                    .tau_ref_ns()
                    .get()
                    .checked_mul(self.config.scheduler.slo.planner.lookahead_waves.get() as u64)
                    .and_then(|span| classified_at_ns.checked_add(span))
            })
            .ok_or_else(|| unavailable("horizon_overflow"))?;
        let snapshot = SchedulerSnapshot {
            observed_at_ns: origin.observed_at_ns(),
            generation: queue.iteration(),
            cost_model_version: model.model_version(),
            fingerprint: model.fingerprint().clone(),
            requests,
            capabilities: BackendPlanningCapabilities {
                work_policy: self.controller_work_policy(hint, &queue),
                path: WaveExecutionPath::PlanRuntime,
                graph_state: WaveGraphState::Disabled,
                order: BatchOrderSemantics::Ordered,
                decode_batch_sizes: (1..=maximum_rows.min(64))
                    .filter_map(NonZeroUsize::new)
                    .collect(),
                prefill_batch_sizes: (1..=maximum_rows.min(64))
                    .filter_map(NonZeroUsize::new)
                    .collect(),
                prefill_chunk_sizes: reference_chunks.into_iter().collect(),
                prefill_alignment: if reference.piecewise_domain().is_some() {
                    self.model_executor
                        .guarded_prefill_granularity()
                        .and_then(|n| u32::try_from(n.get()).ok())
                        .and_then(NonZeroU32::new)
                        .unwrap_or(NonZeroU32::MIN)
                } else {
                    NonZeroU32::MIN
                },
                allow_final_short_chunk: true,
                native_mixed: caps.supports_dynamic_batching,
                max_wave_rows,
                max_prefill_tokens_per_wave: NonZeroU64::new(hint.max_tokens as u64)
                    .ok_or_else(|| unavailable("invalid_wave_capacity"))?,
                workspace_bytes_upper_bound: 0,
            },
            capacity: CapacityReadView {
                evidence_known: false,
                available_kv_tokens: 0,
                maximum_context_tokens,
                available_workspace_bytes: 0,
                available_output_bytes: 0,
            },
            scope: PlanningScope {
                horizon_end_ns,
                reference_decode_token_ns: reference.tau_ref_ns(),
                reference_work_version: reference.identity().revision.get(),
            },
            has_unmodeled_maintenance: queue
                .requests()
                .iter()
                .any(|row| row.readiness.maintenance_blocked || row.readiness.prefix_blocked),
        };
        if !budget() {
            return Err(unavailable("compute_budget_exhausted"));
        }
        let protection = Arc::new(
            PlanningObligationSet::capture_with_budget(&snapshot, classified_at_ns, &mut || {
                if controller_budget.poll() {
                    Ok(())
                } else {
                    Err(PlanningUnknownReason::ComputeBudgetExhausted)
                }
            })
            .map_err(|_| unavailable("recovery_scope_unavailable"))?,
        );
        let recovery_peers =
            self.capture_recovery_peers(&queue, &controller_budget, Some(&protection))?;
        Ok(ControllerSnapshot {
            protection,
            recovery_peers,
            budget: controller_budget,
            queue,
            snapshot,
            origin,
            anchor,
            resources,
            route,
            fences,
            model,
        })
    }
}

/// Fixed points partition the original promise interval in proportion to
/// remaining reference work. Snapshot time cannot move these checkpoints.
fn calibrated_checkpoints(progress: &PrefillProgressView, deadline: u64) -> Option<Vec<u64>> {
    let window = deadline.checked_sub(progress.admitted_at_ns)?;
    let baseline = progress.reference_work_at_admission_ns;
    let remaining = progress
        .reference
        .points
        .last()?
        .cumulative_work_ns
        .checked_sub(baseline)?;
    if remaining == 0 || window == 0 {
        return None;
    }
    let mut result = Vec::new();
    for point in &progress.reference.points {
        let Some(work) = point
            .cumulative_work_ns
            .checked_sub(baseline)
            .filter(|work| *work > 0)
        else {
            continue;
        };
        let elapsed =
            u64::try_from((u128::from(window) * u128::from(work)).div_ceil(u128::from(remaining)))
                .ok()?;
        let at = progress.admitted_at_ns.checked_add(elapsed)?;
        if result.last() != Some(&at) {
            result.push(at);
        }
    }
    Some(result)
}
