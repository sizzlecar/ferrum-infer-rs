//! Compare legal first-wave edits against the actual static wave. Resource
//! owners retain admission/commit authority; unsupported projections keep it.

mod trajectory;

use super::*;
use crate::{cost_prediction::*, joint_policy::*};
use ferrum_interfaces::request_observation::{
    CompletionQuery, RequestStoppingObservation, StoppingForecast, StoppingQuery,
};
use trajectory::{CohortTrajectory, FirstAction, RequestShape};

struct Cohort {
    requests: Vec<SimulatedRequest>,
    facts: Vec<RequestStoppingObservation>,
    trajectory: CohortTrajectory,
    horizon: usize,
}

impl ContinuousBatchScheduler {
    pub(super) fn select_joint_first_wave(
        &self,
        hint: &BatchHint,
        batch: &mut [ScheduledRequest],
        round: &mut TimePlanningRound,
        now: Instant,
    ) {
        if round.config.strategy != ferrum_types::SloStrategy::Joint {
            return;
        }
        let started = Instant::now();
        let prior_elapsed = round.prediction_elapsed;
        round.statistics.joint_comparisons += 1;
        let outcome = self.compare_joint_first_wave(hint, batch, round, now);
        round.prediction_elapsed = prior_elapsed.saturating_add(started.elapsed());
        match outcome {
            Ok(Some(action)) => {
                if self.apply_joint_first_wave(batch, round, action) {
                    round.statistics.joint_interventions += 1;
                } else {
                    round.statistics.static_fallbacks += 1;
                }
            }
            Ok(None) => {}
            Err(Unsupported::WaitingResources | Unsupported::UnknownKvCapacity) => {
                round.statistics.joint_unknown_resources += 1;
                round.statistics.static_fallbacks += 1;
            }
            Err(_) => {
                round.statistics.joint_unresolved += 1;
                round.statistics.static_fallbacks += 1;
            }
        }
    }

    fn compare_joint_first_wave(
        &self,
        hint: &BatchHint,
        batch: &[ScheduledRequest],
        round: &mut TimePlanningRound,
        now: Instant,
    ) -> std::result::Result<Option<FirstAction>, Unsupported> {
        let (decision, alternatives, _) =
            self.evaluate_joint_action_set(hint, batch, round, now)?;
        match decision.choice {
            PolicyChoice::Alternative(index) => Ok(alternatives.get(index).copied()),
            PolicyChoice::Static => Ok(None),
            PolicyChoice::AllKnownInfeasible => {
                round.statistics.joint_known_infeasible += 1;
                Ok(None)
            }
            PolicyChoice::Unresolved => Err(Unsupported::Cost(CostUnknownReason::NoCoverage)),
        }
    }

    pub(super) fn evaluate_joint_admission(
        &self,
        hint: &BatchHint,
        round: &mut TimePlanningRound,
        now: Instant,
    ) -> std::result::Result<ScenarioPolicyDecision, Unsupported> {
        let started = Instant::now();
        let result = self
            .evaluate_joint_action_set(hint, &[], round, now)
            .and_then(|(decision, _, missing_stopping)| {
                if decision.choice == PolicyChoice::Unresolved && missing_stopping {
                    Err(Unsupported::StoppingCoverage)
                } else {
                    Ok(decision)
                }
            });
        round.prediction_elapsed = round.prediction_elapsed.saturating_add(started.elapsed());
        result
    }

    fn evaluate_joint_action_set(
        &self,
        hint: &BatchHint,
        batch: &[ScheduledRequest],
        round: &mut TimePlanningRound,
        now: Instant,
    ) -> std::result::Result<(ScenarioPolicyDecision, Vec<FirstAction>, bool), Unsupported> {
        let started = Instant::now();
        let remaining_time = Duration::from_micros(round.config.planning.max_elapsed_us)
            .checked_sub(round.prediction_elapsed)
            .ok_or(Unsupported::TimeBudget)?;
        let mut cohort = self.joint_cohort(hint, batch, round, now)?;
        // Both candidates pay this round's declared maximum control delay.
        // These are model-relative timing proxies; future host/transport gaps
        // still require same-hardware client measurements and calibration.
        for request in &mut cohort.requests {
            request.age = request
                .age
                .checked_add(remaining_time)
                .ok_or(Unsupported::Arithmetic)?;
        }
        let first_wave = if cohort.trajectory.first_wave.is_empty() {
            cohort.trajectory.natural_members(&cohort.requests)?
        } else {
            cohort.trajectory.first_wave.clone()
        };
        let largest = first_wave
            .iter()
            .filter(|member| {
                member.work.kind == ferrum_execution::execution_attempt::AttemptWorkKind::Prefill
            })
            .map(|member| member.work.requested.as_range().len())
            .max()
            .unwrap_or(0);
        let static_action = FirstAction {
            prefill_limit: None,
        };
        let mut alternatives = if largest >= 2 {
            vec![FirstAction {
                prefill_limit: Some(largest.div_ceil(2)),
            }]
        } else {
            Vec::new()
        };
        let quarter = largest.div_ceil(4);
        if quarter < largest.div_ceil(2) {
            alternatives.push(FirstAction {
                prefill_limit: Some(quarter),
            });
        }
        if alternatives.len() + 1 > round.config.planning.max_candidates {
            return Err(Unsupported::CandidateBudget);
        }
        let forecast_bytes = cohort
            .facts
            .iter()
            .try_fold(0usize, |bytes, fact| {
                fact.cap
                    .checked_mul(std::mem::size_of::<
                        ferrum_interfaces::request_observation::StoppingMass,
                    >())
                    .and_then(|size| bytes.checked_add(size))
            })
            .ok_or(Unsupported::ParticipantBudget)?;
        if forecast_bytes > round.config.planning.max_request_fact_bytes {
            return Err(Unsupported::ParticipantBudget);
        }
        let mut forecasts = Vec::with_capacity(cohort.facts.len());
        let mut success = Vec::with_capacity(cohort.facts.len());
        let mut missing_stopping = false;
        for fact in &cohort.facts {
            let forecast = round.stopping_forecast(&StoppingQuery {
                rules: &fact.rules,
                verified_survival_through: fact.verified_survival_through,
                cap: fact.cap,
                horizon: fact.cap,
                now,
            });
            // Validate mass even when an optional third-party provider supplies
            // the forecast. Unsupported tails never acquire a cap endpoint.
            if forecast
                .stops
                .windows(2)
                .any(|pair| pair[0].generated_tokens >= pair[1].generated_tokens)
                || forecast
                    .stops
                    .iter()
                    .any(|stop| !stop.mass.is_finite() || stop.mass < 0.0)
                || !forecast.unresolved_mass.is_finite()
                || forecast.unresolved_mass < 0.0
                || !forecast.known_survival_mass.is_finite()
                || forecast.known_survival_mass < 0.0
                || (forecast.resolved_mass()
                    + forecast.unresolved_mass
                    + forecast.known_survival_mass
                    - 1.0)
                    .abs()
                    > 1e-9
            {
                return Err(Unsupported::InvalidState);
            }
            missing_stopping |= forecast.resolved_mass() < 1.0 - 1e-9;
            forecasts.push(forecast);
            let completion = round.completion_forecast(&CompletionQuery {
                rules: &fact.rules,
                cap: fact.cap,
                now,
                maturity: Duration::from_millis(round.config.joint.completion_cohort_maturity_ms),
            });
            let valid = completion.samples > 0
                && completion.approximation.is_some()
                && [
                    completion.succeeded_mass,
                    completion.failed_mass,
                    completion.unresolved_mass,
                ]
                .iter()
                .all(|mass| mass.is_finite() && *mass >= 0.0 && *mass <= 1.0)
                && (completion.succeeded_mass
                    + completion.failed_mass
                    + completion.unresolved_mass
                    - 1.0)
                    .abs()
                    <= 1e-9;
            success.push(valid.then_some(ProbabilityInterval {
                lower: completion.succeeded_mass,
                upper: completion.succeeded_mass + completion.unresolved_mass,
            }));
            missing_stopping |= !valid || completion.unresolved_mass > 0.0;
        }
        let mut config = round.config;
        config.planning.max_queries = config
            .planning
            .max_queries
            .checked_sub(round.query_count)
            .filter(|count| *count > 0)
            .ok_or(Unsupported::QueryBudget)?;
        config.planning.max_elapsed_us = remaining_time
            .checked_sub(started.elapsed())
            .and_then(|time| u64::try_from(time.as_micros()).ok())
            .filter(|time| *time > 0)
            .ok_or(Unsupported::TimeBudget)?;
        let mut aggregate =
            ScenarioRiskAccumulator::new(&cohort.requests, alternatives.len(), &config, &success)?;
        let mut aggregate_error = None;
        let summary = SloPolicy::evaluate_scenarios_with_requests(
            EvaluationInput {
                snapshot: round.snapshot.as_ref(),
                now,
                config: &config,
                execution: round.execution,
                requests: &cohort.requests,
                resources: ResourceSupport {
                    waiting_requests: cohort.trajectory.admission.waiting_requests(),
                    kv_reserved_through_scenario_end: true,
                },
                adapter: &cohort.trajectory,
                static_action: &static_action,
                alternatives: &alternatives,
                static_window_rounds: cohort.horizon,
            },
            |rank, requests| {
                for (request, forecast) in requests.iter_mut().zip(&forecasts) {
                    request.scenario_end_tokens = endpoint_at_rank(
                        forecast,
                        rank,
                        request.output_tokens,
                        request.output_cap,
                    )?;
                }
                Ok(())
            },
            |rank, comparison| {
                if aggregate_error.is_none() {
                    aggregate_error = aggregate.observe(rank, comparison).err();
                }
            },
        );
        // A failed evaluation cannot renew a sub-round's exhausted query budget.
        round.query_count = round.query_count.saturating_add(
            summary
                .as_ref()
                .map_or(config.planning.max_queries, |summary| summary.queries),
        );
        let summary = summary?;
        round.statistics.joint_cost_snapshot_queries = round
            .statistics
            .joint_cost_snapshot_queries
            .saturating_add(summary.queries as u64);
        round.statistics.joint_cost_cache_hits = round
            .statistics
            .joint_cost_cache_hits
            .saturating_add(summary.cache_hits as u64);
        round.statistics.joint_cost_cache_peak_bytes = round
            .statistics
            .joint_cost_cache_peak_bytes
            .max(summary.cache_retained_bytes as u64);
        if let Some(error) = aggregate_error {
            return Err(error);
        }
        let decision = aggregate.finish(summary, &vec![None; alternatives.len()])?;
        if started.elapsed() > remaining_time {
            return Err(Unsupported::TimeBudget);
        }
        Ok((decision, alternatives, missing_stopping))
    }

    fn apply_joint_first_wave(
        &self,
        batch: &mut [ScheduledRequest],
        round: &mut TimePlanningRound,
        action: FirstAction,
    ) -> bool {
        let Some(limit) = action.prefill_limit else {
            return false;
        };
        let mut prefill = self.prefill_queue.write();
        let mut updates = Vec::new();
        for (index, scheduled) in batch.iter().enumerate() {
            let Some(request) = prefill
                .iter()
                .find(|request| request.inner.request.id == scheduled.request.id)
            else {
                continue;
            };
            if scheduled.tokens_processed != request.prefill_chunk_offset {
                return false;
            }
            let Some(original) = scheduled.tokens_to_process else {
                return false;
            };
            let Some((work, true)) = self.cost_work(
                request,
                original.min(limit),
                false,
                round.prefill_shape_policy,
            ) else {
                return false;
            };
            if work.requested.as_range().len() > original {
                return false;
            }
            updates.push((index, work));
        }
        let changed = updates.iter().any(|(index, work)| {
            batch[*index].tokens_to_process != Some(work.requested.as_range().len())
        });
        if !changed {
            return false;
        }
        for (index, work) in updates {
            let scheduled = &mut batch[index];
            let request = prefill
                .iter_mut()
                .find(|request| request.inner.request.id == scheduled.request.id)
                .expect("validated locked prefill member");
            let tokens = work.requested.as_range().len();
            request.logical_work_frontier.mark_scheduled(tokens);
            scheduled.tokens_to_process = Some(tokens);
            round.selected[index] = work;
            round.statistics.shrunk_prefills += 1;
        }
        true
    }

    fn joint_cohort(
        &self,
        hint: &BatchHint,
        batch: &[ScheduledRequest],
        round: &TimePlanningRound,
        now: Instant,
    ) -> std::result::Result<Cohort, Unsupported> {
        if self.pressure_active.load(Ordering::Acquire)
            || self.runtime_config.prefill_first_until_active.is_some()
            || self.config.prefix_rendezvous_max_wait_ms.is_some()
            || self.decode_capacity_backpressure_limit().is_some()
        {
            return Err(Unsupported::UnknownKvCapacity);
        }
        let facts = round.request_facts().ok_or(Unsupported::InvalidState)?;
        let clocks = round.request_clocks().ok_or(Unsupported::InvalidState)?;
        let limit = hint
            .max_batch_size
            .min(self.cb_config.max_decode_batch)
            .min(self.config.max_running_requests)
            .min(round.config.planning.max_participants);
        if facts.is_empty()
            || facts.len() > round.config.planning.max_participants
            || limit.min(facts.len()) > hint.max_tokens
        {
            return Err(Unsupported::ParticipantBudget);
        }
        let capacity = round.capacity().ok_or(Unsupported::UnknownKvCapacity)?;
        let mut blocks = 0usize;
        let mut horizon = 0usize;
        let mut requests = Vec::with_capacity(facts.len());
        let mut ordered_facts = Vec::with_capacity(facts.len());
        let mut shapes = Vec::with_capacity(facts.len());
        let mut initial_admission = Vec::with_capacity(facts.len());
        let waiting = self.waiting_queue.read();
        let prefill = self.prefill_queue.read();
        let decode = self.decode_queue.read();
        if prefill.len() + decode.requests.len() + waiting.len() != facts.len() {
            return Err(Unsupported::InvalidState);
        }
        if !waiting.is_empty()
            && (self.config.max_running_requests > limit || waiting.has_capacity_deferrals())
        {
            return Err(Unsupported::WaitingResources);
        }
        if !waiting.is_empty() {
            // Capacity preceded physical admission. Preserve the whole initial
            // demand as members move to residency. Without an exact baseline,
            // conservatively charge every cohort member.
            let new_sequences = round
                .capacity_initial_new_sequences
                .unwrap_or(facts.len())
                .max(waiting.len());
            match capacity.admission_coverage {
                ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
                    max_new_sequences,
                    max_sequence_tokens,
                    max_batch_sequences,
                } if new_sequences <= max_new_sequences && limit <= max_batch_sequences => {
                    if facts.iter().any(|(id, fact)| {
                        prefill
                            .iter()
                            .chain(decode.requests.values())
                            .chain(waiting.iter())
                            .find(|request| request.inner.request.id == *id)
                            .and_then(|request| self.prompt_token_estimate(request))
                            .and_then(|prompt| prompt.checked_add(fact.cap))
                            .is_none_or(|tokens| tokens > max_sequence_tokens)
                    }) {
                        return Err(Unsupported::UnknownKvCapacity);
                    }
                }
                _ => return Err(Unsupported::UnknownKvCapacity),
            }
        }
        if waiting.is_empty() && facts.len() > limit {
            return Err(Unsupported::ParticipantBudget);
        }
        let residents = prefill.len() + decode.requests.len();
        for (index, request) in prefill
            .iter()
            .chain(decode.requests.values())
            .chain(waiting.iter())
            .enumerate()
        {
            if request.capacity_deferred_from_decode
                || request.prefix_restore.is_pending()
                || request.execution_capacity_deferral.is_some()
                || request.execution_readiness_block.is_some()
                || request.execution_maintenance_retry.is_some()
                || request.output_readiness.is_some()
                || request.capacity_deferred_until_release_epoch > 0
                || request.prefix_rendezvous.held()
            {
                return Err(Unsupported::InvalidState);
            }
            let id = &request.inner.request.id;
            let fact = &facts
                .iter()
                .find(|(candidate, _)| candidate == id)
                .ok_or(Unsupported::InvalidState)?
                .1;
            let clock = clocks
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, clock)| clock)
                .ok_or(Unsupported::InvalidState)?;
            if fact.terminal.is_some()
                || clock.generated_tokens != fact.generated_tokens
                || fact.generated_tokens >= fact.cap
                || clock.captured_at > now
                || clock.itl_threshold
                    != Duration::try_from_secs_f64(round.config.itl_ms / 1000.0).ok()
                || (fact.generated_tokens > 0 && clock.first_text.is_none())
            {
                return Err(Unsupported::InvalidState);
            }
            if !clock.streaming
                || (fact.generated_tokens > 0
                    && clock.nonempty_text_events != fact.generated_tokens as u64)
            {
                return Err(Unsupported::HistoricalTextCoalescing);
            }
            let prompt = self
                .prompt_token_estimate(request)
                .ok_or(Unsupported::InvalidState)?;
            let prefilled = if request.phase == RequestPhase::Decoding {
                prompt
            } else {
                request.prefill_chunk_offset
            };
            let context_cap = prompt
                .checked_add(fact.cap)
                .ok_or(Unsupported::Arithmetic)?;
            // Charge the entire prompt+cap again against genuinely free blocks:
            // no guessed retained capacity, shared-prefix credit or early EOS.
            blocks = blocks
                .checked_add(context_cap.div_ceil(capacity.block_size))
                .ok_or(Unsupported::Arithmetic)?;
            horizon = horizon
                .checked_add(prompt.saturating_sub(prefilled))
                .and_then(|value| value.checked_add(fact.cap - fact.generated_tokens))
                .ok_or(Unsupported::Arithmetic)?;
            let count = usize::try_from(clock.nonempty_text_events.saturating_sub(1))
                .map_err(|_| Unsupported::Arithmetic)?;
            let violating = usize::try_from(clock.violating_text_intervals)
                .map_err(|_| Unsupported::Arithmetic)?;
            let clock_value = RequestClock {
                first_output: clock.first_text,
                last_output: clock.last_text,
                maximum_output_interval: clock.max_text_interval.unwrap_or_default(),
                interval_count: count,
                violating_interval_count: violating,
                completed: None,
            };
            requests.push(SimulatedRequest {
                id: id.clone(),
                prompt_tokens: prompt,
                prefill_tokens: prefilled,
                output_tokens: fact.generated_tokens,
                output_cap: fact.cap,
                // Placeholder for supported discrete rank assignments below;
                // it never substitutes cap for unknown stopping mass.
                scenario_end_tokens: fact.cap,
                age: clock
                    .age
                    .checked_add(now.duration_since(clock.captured_at))
                    .ok_or(Unsupported::Arithmetic)?,
                central_clock: clock_value,
                risk_clock: clock_value,
            });
            ordered_facts.push(fact.clone());
            shapes.push(RequestShape {
                restored_tokens: request.prefix_restore.restored_tokens(),
                execution_ceiling: request.prefill_execution_chunk_ceiling,
            });
            initial_admission.push(if index < residents {
                InitialAdmission::Resident
            } else {
                InitialAdmission::Waiting {
                    order: index - residents,
                }
            });
        }
        if blocks > capacity.free_blocks {
            return Err(Unsupported::UnknownKvCapacity);
        }
        if !batch.is_empty() && !round.cost_selection_is_exact(batch.len()) {
            return Err(Unsupported::Cost(CostUnknownReason::InexactWork));
        }
        let first_wave = batch
            .iter()
            .zip(&round.selected)
            .map(|(scheduled, work)| {
                let index = requests
                    .iter()
                    .position(|request| request.id == scheduled.request.id)
                    .ok_or(Unsupported::InvalidState)?;
                Ok(GroupMember {
                    request: index,
                    work: *work,
                })
            })
            .collect::<std::result::Result<Vec<_>, Unsupported>>()?;
        Ok(Cohort {
            requests,
            facts: ordered_facts,
            horizon,
            trajectory: CohortTrajectory {
                first_wave,
                shapes,
                admission: AdmissionSlotPlan::new(
                    &initial_admission,
                    limit,
                    round.config.planning.max_participants,
                )?,
                execution: round.execution,
                shape_policy: round.prefill_shape_policy,
                max_batch: limit,
                max_tokens: hint.max_tokens,
                prefill_step_chunk: self.runtime_config.prefill_step_chunk,
                active_decode_prefill_chunk: self.runtime_config.active_decode_prefill_chunk,
                active_decode_prefill_budget: self
                    .runtime_config
                    .active_decode_prefill_token_budget,
                decode_pressure_threshold: self.decode_pressure_prefill_cap_threshold(hint),
                default_active_chunk: self.default_active_decode_prefill_chunk(),
                chunking: self.cb_config.enable_chunked_prefill,
            },
        })
    }
}

/// A finite rank approximates the known discrete CDF. An overlap with missing
/// mass invalidates the whole rank; surviving ranks are never renormalized.
fn endpoint_at_rank(
    forecast: &StoppingForecast,
    rank: ScenarioRank,
    current: usize,
    cap: usize,
) -> std::result::Result<usize, Unsupported> {
    if rank.quantile_upper > forecast.resolved_mass() + 1e-12 {
        return Err(Unsupported::Cost(CostUnknownReason::NoCoverage));
    }
    let mut cumulative = 0.0;
    for stop in &forecast.stops {
        cumulative += stop.mass;
        if cumulative + 1e-12 >= rank.quantile_upper {
            if stop.generated_tokens <= current || stop.generated_tokens > cap {
                return Err(Unsupported::InvalidState);
            }
            return Ok(stop.generated_tokens);
        }
    }
    Err(Unsupported::Cost(CostUnknownReason::NoCoverage))
}
