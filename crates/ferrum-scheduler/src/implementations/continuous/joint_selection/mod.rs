//! Compare legal first-wave edits against the actual static wave. Resource
//! owners retain admission/commit authority; unsupported projections keep it.

mod prefill_phase;
mod trajectory;

#[cfg(test)]
mod recoverability_tests;

#[cfg(test)]
mod planning_profile;

use super::*;
use crate::{cost_prediction::*, joint_policy::*};
use ferrum_interfaces::request_observation::{
    CompletionQuery, RequestStoppingObservation, StoppingForecast, StoppingQuery,
};
use std::num::NonZeroUsize;
use trajectory::{CohortTrajectory, FirstAction, RequestShape};

struct Cohort {
    requests: Vec<SimulatedRequest>,
    facts: Vec<RequestStoppingObservation>,
    trajectory: CohortTrajectory,
    wave_bindings: Vec<WaveBinding>,
    cap_bindings: Vec<PrefillCapTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AdmissionCapMode {
    Propose,
    ExistingOnly,
}

#[derive(Debug)]
pub(super) struct JointAdmissionEvaluation {
    pub(super) decision: ScenarioPolicyDecision,
    pub(super) prefill_cap: Option<PrefillCapProposal>,
}

#[derive(Debug, Clone)]
pub(super) struct PrefillCapProposal {
    pub(super) targets: Vec<PrefillCapTarget>,
    pub(super) limit: NonZeroUsize,
    pub(super) valid_until: Instant,
}

#[derive(Debug, Clone)]
pub(super) struct PrefillCapTarget {
    pub(super) request: usize,
    pub(super) key: PrefixRequestKey,
    pub(super) phase: RequestPhase,
    pub(super) generation: LogicalWorkGeneration,
    pub(super) prompt_tokens: usize,
    pub(super) prefilled_tokens: usize,
    pub(super) prior_cap: Option<PrefillPhaseCap>,
}

struct JointActionEvaluation {
    decision: ScenarioPolicyDecision,
    alternatives: Vec<FirstAction>,
    missing_stopping: bool,
    wave_bindings: Vec<WaveBinding>,
    prefill_cap: Option<PrefillCapProposal>,
}

#[derive(Clone)]
struct WaveBinding {
    id: RequestId,
    ticket: WaitingAdmissionTicket,
    generation: LogicalWorkGeneration,
    prompt_tokens: usize,
    offset: usize,
    work: CostWork,
}

pub(super) struct SelectedJointAction {
    action: FirstAction,
    wave_bindings: Vec<WaveBinding>,
}

impl ContinuousBatchScheduler {
    pub(super) fn select_joint_first_wave(
        &self,
        hint: &BatchHint,
        batch: &mut Vec<ScheduledRequest>,
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
                let decode_only = matches!(action.action, FirstAction::DecodeOnly);
                let phase = matches!(action.action, FirstAction::PrefillPhaseLimit(_));
                if self.apply_joint_first_wave(batch, round, action) {
                    round.statistics.joint_interventions += 1;
                    round.statistics.joint_decode_only_interventions += u64::from(decode_only);
                    round.statistics.joint_prefill_phase_interventions += u64::from(phase);
                } else {
                    round.statistics.static_fallbacks += 1;
                }
            }
            Ok(None) => {}
            Err(reason) => {
                round.statistics.record_joint_unsupported(reason);
                round.statistics.static_fallbacks += 1;
            }
        }
    }

    pub(super) fn compare_joint_first_wave(
        &self,
        hint: &BatchHint,
        batch: &[ScheduledRequest],
        round: &mut TimePlanningRound,
        now: Instant,
    ) -> std::result::Result<Option<SelectedJointAction>, Unsupported> {
        let JointActionEvaluation {
            decision,
            alternatives,
            wave_bindings,
            ..
        } = self.evaluate_joint_action_set(
            hint,
            batch,
            round,
            now,
            ComparisonKind::Dispatch,
            AdmissionCapMode::ExistingOnly,
            0,
        )?;
        let mut record = |outcome| {
            round.statistics.joint_decision.consume(
                ComparisonKind::Dispatch,
                outcome,
                &decision.observation,
            );
        };
        match decision.choice {
            PolicyChoice::Alternative(index) => {
                record(DecisionConsumerOutcome::Alternative);
                Ok(alternatives
                    .get(index)
                    .copied()
                    .map(|action| SelectedJointAction {
                        action,
                        wave_bindings,
                    }))
            }
            PolicyChoice::Static => {
                record(DecisionConsumerOutcome::StaticFeasible);
                Ok(None)
            }
            PolicyChoice::AllKnownInfeasible => {
                record(DecisionConsumerOutcome::AllKnownInfeasible);
                round.statistics.joint_known_infeasible += 1;
                Ok(None)
            }
            // Several kinds of unresolved scenario mass can reach this point.
            // Do not relabel an aggregate outcome as a cost-model failure.
            PolicyChoice::Unresolved => {
                record(DecisionConsumerOutcome::FeasibilityUnresolved);
                Err(Unsupported::FeasibilityUnresolved)
            }
        }
    }

    pub(super) fn evaluate_joint_admission(
        &self,
        hint: &BatchHint,
        round: &mut TimePlanningRound,
        now: Instant,
        mode: AdmissionCapMode,
        captured_member_bytes: usize,
    ) -> std::result::Result<JointAdmissionEvaluation, Unsupported> {
        let started = Instant::now();
        let result = self
            .evaluate_joint_action_set(
                hint,
                &[],
                round,
                now,
                ComparisonKind::Admission,
                mode,
                captured_member_bytes,
            )
            .and_then(|evaluation| {
                let JointActionEvaluation {
                    decision,
                    missing_stopping,
                    prefill_cap,
                    ..
                } = evaluation;
                if decision.choice == PolicyChoice::Unresolved && missing_stopping {
                    round.statistics.joint_decision.consume(
                        ComparisonKind::Admission,
                        DecisionConsumerOutcome::StoppingCoverage,
                        &decision.observation,
                    );
                    Err(Unsupported::StoppingCoverage)
                } else {
                    Ok(JointAdmissionEvaluation {
                        decision,
                        prefill_cap,
                    })
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
        comparison_kind: ComparisonKind,
        cap_mode: AdmissionCapMode,
        captured_member_bytes: usize,
    ) -> std::result::Result<JointActionEvaluation, Unsupported> {
        let started = Instant::now();
        let remaining_time = Duration::from_micros(round.config.planning.max_elapsed_us)
            .checked_sub(round.prediction_elapsed)
            .ok_or(Unsupported::TimeBudget)?;
        let mut cohort =
            self.joint_cohort(hint, batch, round, now, cap_mode, captured_member_bytes)?;
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
        if cap_mode == AdmissionCapMode::Propose {
            // Fix the target set before any cap changes packing. The resulting
            // preview may fill the released token budget with other members,
            // but those members do not inherit the commitment.
            cohort.cap_bindings.retain(|binding| {
                first_wave.iter().any(|member| {
                    member.request == binding.request
                        && member.work.kind
                            == ferrum_execution::execution_attempt::AttemptWorkKind::Prefill
                })
            });
            cohort
                .trajectory
                .cap_commit_targets
                .try_reserve_exact(cohort.cap_bindings.len())
                .map_err(|_| Unsupported::ParticipantBudget)?;
            cohort
                .trajectory
                .cap_commit_targets
                .extend(cohort.cap_bindings.iter().map(|binding| PrefillTarget {
                    request: binding.request,
                    prompt_tokens: binding.prompt_tokens,
                }));
        }
        let static_action = FirstAction::Static;
        let mut alternatives = if largest >= 2 {
            vec![FirstAction::PrefillLimit(
                NonZeroUsize::new(largest.div_ceil(2)).expect("positive prefill limit"),
            )]
        } else {
            Vec::new()
        };
        let quarter = largest.div_ceil(4);
        if quarter < largest.div_ceil(2) {
            alternatives.push(FirstAction::PrefillLimit(
                NonZeroUsize::new(quarter).expect("positive prefill limit"),
            ));
        }
        if cohort.trajectory.allows_decode_only(&first_wave) {
            alternatives.push(FirstAction::DecodeOnly);
        }
        // Dispatch phases bind its already selected resident first wave.
        // Admission commitments below instead preview natural repacking.
        if !batch.is_empty() && cohort.wave_bindings.len() == batch.len() {
            for limit in [largest.div_ceil(2), quarter] {
                let Some(limit) = NonZeroUsize::new(limit) else {
                    continue;
                };
                if !alternatives.iter().any(|action| {
                    matches!(action, FirstAction::PrefillPhaseLimit(previous) if *previous == limit)
                }) && cohort.trajectory.allows_prefill_phase(&cohort.requests, limit)
                {
                    alternatives.push(FirstAction::PrefillPhaseLimit(limit));
                }
            }
        }
        if cap_mode == AdmissionCapMode::Propose {
            for limit in [largest.div_ceil(2), quarter] {
                let Some(limit) = NonZeroUsize::new(limit) else {
                    continue;
                };
                if cohort.trajectory.allows_cap_commit(&cohort.requests, limit)
                    && !alternatives.iter().any(|action| matches!(action, FirstAction::PrefillCapCommit(previous) if *previous == limit))
                {
                    alternatives.push(FirstAction::PrefillCapCommit(limit));
                }
            }
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
        if alternatives
            .iter()
            .any(|action| matches!(action, FirstAction::PrefillPhaseLimit(_)))
        {
            let phase_bytes = ContinuationPhase::prefill_targets_bytes_bound(cohort.requests.len())
                .ok_or(Unsupported::ParticipantBudget)?;
            let clocks_bytes = round
                .request_clocks()
                .ok_or(Unsupported::InvalidClock)?
                .len()
                .checked_mul(std::mem::size_of::<(RequestId, RequestClockFacts)>())
                .ok_or(Unsupported::ParticipantBudget)?;
            let binding_bytes = cohort
                .wave_bindings
                .len()
                .checked_mul(std::mem::size_of::<WaveBinding>())
                // The later apply uses smaller per-member update tuples.
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or(Unsupported::ParticipantBudget)?;
            let bytes = round
                .request_fact_bytes
                .checked_add(clocks_bytes)
                .and_then(|bytes| bytes.checked_add(forecast_bytes))
                .and_then(|bytes| bytes.checked_add(binding_bytes))
                .and_then(|bytes| bytes.checked_add(phase_bytes))
                .ok_or(Unsupported::ParticipantBudget)?;
            if bytes > round.config.planning.max_request_fact_bytes {
                return Err(Unsupported::ParticipantBudget);
            }
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
        config.planning.max_queries = round.remaining_queries();
        if config.planning.max_queries == 0 {
            return Err(Unsupported::QueryBudget);
        }
        config.planning.max_elapsed_us = remaining_time
            .checked_sub(started.elapsed())
            .and_then(|time| u64::try_from(time.as_micros()).ok())
            .filter(|time| *time > 0)
            .ok_or(Unsupported::TimeBudget)?;
        let mut aggregate =
            ScenarioRiskAccumulator::new(&cohort.requests, alternatives.len(), &config, &success)?;
        let mut aggregate_error = None;
        let mut diagnostics = ScenarioDiagnostics::default();
        let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
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
                horizon: EvaluationHorizon::ScenarioCompletion,
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
            &mut diagnostics,
        );
        round
            .statistics
            .joint_evaluation
            .record(comparison_kind, diagnostics, summary.is_ok());
        // Early Unknown, invalid distributions and panics still made only the
        // recorded original-provider calls. A genuine query-budget boundary is
        // sticky across subrounds, independently of the actual call counter.
        if !round.charge_scenario_queries(&diagnostics) {
            return Err(Unsupported::QueryBudget);
        }
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
        round.statistics.joint_decision.computed(comparison_kind);
        if started.elapsed() > remaining_time {
            round
                .statistics
                .joint_decision
                .post_finish_time_rejected(comparison_kind);
            return Err(Unsupported::TimeBudget);
        }
        round.statistics.joint_decision.returned(comparison_kind);
        let prefill_cap = match decision.choice {
            PolicyChoice::Alternative(index) => match alternatives.get(index) {
                Some(FirstAction::PrefillCapCommit(limit)) => Some(PrefillCapProposal {
                    targets: cohort.cap_bindings,
                    limit: *limit,
                    valid_until: decision.valid_until,
                }),
                _ => None,
            },
            _ => None,
        };
        Ok(JointActionEvaluation {
            decision,
            alternatives,
            missing_stopping,
            wave_bindings: cohort.wave_bindings,
            prefill_cap,
        })
    }

    pub(super) fn apply_joint_first_wave(
        &self,
        batch: &mut Vec<ScheduledRequest>,
        round: &mut TimePlanningRound,
        selected: SelectedJointAction,
    ) -> bool {
        let limit = match selected.action {
            FirstAction::Static => return false,
            // Admission commits policy before a fresh gate; this is never an
            // edit of an already selected dispatch batch.
            FirstAction::PrefillCapCommit(_) => return false,
            FirstAction::DecodeOnly => return self.apply_joint_decode_only(batch, round),
            FirstAction::PrefillLimit(limit) => limit.get(),
            FirstAction::PrefillPhaseLimit(limit) => {
                return self.apply_joint_prefill_phase(
                    batch,
                    round,
                    limit,
                    &selected.wave_bindings,
                );
            }
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

    fn apply_joint_decode_only(
        &self,
        batch: &mut Vec<ScheduledRequest>,
        round: &mut TimePlanningRound,
    ) -> bool {
        use ferrum_execution::execution_attempt::AttemptWorkKind;

        if !round.cost_selection_is_exact(batch.len()) {
            return false;
        }
        let mut prefill = self.prefill_queue.write();
        let decode = self.decode_queue.read();
        let mut prefill_count = 0;
        let mut decode_count = 0;
        // Validate the entire original wave before touching its reservations.
        // Nominal prefill ranges can differ from exact projected cost ranges;
        // recompute that projection instead of treating them as interchangeable.
        for (index, (scheduled, selected)) in batch.iter().zip(&round.selected).enumerate() {
            if batch[..index]
                .iter()
                .any(|previous| previous.request.id == scheduled.request.id)
            {
                return false;
            }
            let Some(tokens) = scheduled.tokens_to_process else {
                return false;
            };
            let (request, is_decode) = match selected.kind {
                AttemptWorkKind::Prefill => {
                    let Some(request) = prefill
                        .iter()
                        .find(|request| request.inner.request.id == scheduled.request.id)
                    else {
                        return false;
                    };
                    if request.phase != RequestPhase::Prefilling
                        || scheduled.tokens_processed != request.prefill_chunk_offset
                        || request
                            .joint_decode_only_deferred_generation
                            .is_some_and(|last| {
                                request.logical_work_frontier.progress_generation() <= last
                            })
                    {
                        return false;
                    }
                    prefill_count += 1;
                    (request, false)
                }
                AttemptWorkKind::Decode => {
                    let Some(request) = decode.requests.get(&scheduled.request.id) else {
                        return false;
                    };
                    if request.phase != RequestPhase::Decoding
                        || scheduled.tokens_processed != request.total_tokens()
                        || tokens != 1
                    {
                        return false;
                    }
                    decode_count += 1;
                    (request, true)
                }
            };
            if request.capacity_deferred_from_decode
                || request.prefix_restore.is_pending()
                || request.execution_capacity_deferral.is_some()
                || request.execution_readiness_block.is_some()
                || request.execution_maintenance_retry.is_some()
                || request.output_readiness.is_some()
                || request.capacity_deferred_until_release_epoch > 0
                || request.prefix_rendezvous.held()
                || self.cost_work(request, tokens, is_decode, round.prefill_shape_policy)
                    != Some((*selected, true))
            {
                return false;
            }
        }
        if prefill_count == 0 || decode_count == 0 {
            return false;
        }
        // The skipped prefill remains admitted and queued. Only the pending
        // schedule is withdrawn, without publishing progress or releasing KV.
        for (scheduled, selected) in batch.iter().zip(&round.selected) {
            if selected.kind == AttemptWorkKind::Prefill {
                let request = prefill
                    .iter_mut()
                    .find(|request| request.inner.request.id == scheduled.request.id)
                    .expect("validated locked prefill member");
                request.joint_decode_only_deferred_generation =
                    Some(request.logical_work_frontier.progress_generation());
                request.logical_work_frontier.mark_scheduled(0);
            }
        }
        let mut index = 0;
        batch.retain(|_| {
            let keep = round.selected[index].kind == AttemptWorkKind::Decode;
            index += 1;
            keep
        });
        round
            .selected
            .retain(|work| work.kind == AttemptWorkKind::Decode);
        true
    }

    fn joint_cohort(
        &self,
        hint: &BatchHint,
        batch: &[ScheduledRequest],
        round: &TimePlanningRound,
        now: Instant,
        cap_mode: AdmissionCapMode,
        captured_member_bytes: usize,
    ) -> std::result::Result<Cohort, Unsupported> {
        if self.pressure_active.load(Ordering::Acquire) {
            return Err(Unsupported::PressureActive);
        }
        // The first actual static decode disarms fill-first for this busy
        // cohort before this comparison. From that point the copied
        // decode-first continuation is exact, including later arrivals.
        // Only an empty-to-new-cohort transition can arm it again. Keep
        // the unmodeled cold fill phase conservative.
        if self.runtime_config.prefill_first_until_active.is_some()
            && self.fill_first_initial_cohort_armed.load(Ordering::Acquire)
        {
            return Err(Unsupported::ColdFillFirst);
        }
        if self.config.prefix_rendezvous_max_wait_ms.is_some() {
            return Err(Unsupported::PrefixRendezvous);
        }
        if self.decode_capacity_backpressure_limit().is_some() {
            return Err(Unsupported::DecodeBackpressure);
        }
        let facts = round.request_facts().ok_or(Unsupported::InvalidState)?;
        let clocks = round.request_clocks().ok_or(Unsupported::InvalidState)?;
        if cap_mode == AdmissionCapMode::Propose || captured_member_bytes > 0 {
            // Include the caller's still-live frozen cohort. Bounds are checked
            // before allocating proposal bindings or the phase target payload.
            // Two copies cover a retained proposal/target set alongside core
            // phase scratch; the implementation normally moves the proposal.
            let forecast_bytes = facts
                .iter()
                .try_fold(0usize, |bytes, (_, fact)| {
                    fact.cap
                        .checked_mul(std::mem::size_of::<
                            ferrum_interfaces::request_observation::StoppingMass,
                        >())
                        .and_then(|next| bytes.checked_add(next))
                })
                .ok_or(Unsupported::ParticipantBudget)?;
            let proposal_bytes = if cap_mode == AdmissionCapMode::Propose {
                facts
                    .len()
                    .checked_mul(std::mem::size_of::<PrefillCapTarget>())
                    .and_then(|bindings| {
                        ContinuationPhase::prefill_targets_bytes_bound(facts.len())
                            .and_then(|targets| bindings.checked_add(targets))
                    })
                    .and_then(|bytes| bytes.checked_mul(2))
                    .ok_or(Unsupported::ParticipantBudget)?
            } else {
                0
            };
            let bytes = clocks
                .len()
                .checked_mul(std::mem::size_of::<(RequestId, RequestClockFacts)>())
                .and_then(|bytes| bytes.checked_add(round.request_fact_bytes))
                .and_then(|bytes| bytes.checked_add(captured_member_bytes))
                .and_then(|bytes| {
                    bytes.checked_add(
                        round
                            .persistent_capacity()
                            .map_or(0, |proof| proof.retained_bytes()),
                    )
                })
                .and_then(|bytes| bytes.checked_add(forecast_bytes))
                .and_then(|bytes| bytes.checked_add(proposal_bytes))
                .ok_or(Unsupported::ParticipantBudget)?;
            if bytes > round.config.planning.max_request_fact_bytes {
                return Err(Unsupported::ParticipantBudget);
            }
        }
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
        let capacity = round.capacity();
        let mut blocks = 0usize;
        let mut requests = Vec::with_capacity(facts.len());
        let mut ordered_facts = Vec::with_capacity(facts.len());
        let mut shapes = Vec::with_capacity(facts.len());
        let mut initial_admission = Vec::with_capacity(facts.len());
        let mut cap_bindings = Vec::new();
        if cap_mode == AdmissionCapMode::Propose {
            cap_bindings
                .try_reserve_exact(facts.len())
                .map_err(|_| Unsupported::ParticipantBudget)?;
        }
        let waiting = self.waiting_queue.read();
        let prefill = self.prefill_queue.read();
        let decode = self.decode_queue.read();
        if prefill.len() + decode.requests.len() + waiting.len() != facts.len() {
            return Err(Unsupported::InvalidState);
        }
        let resident_proof = waiting.is_empty()
            && round.resident_capacity().is_some_and(|proof| {
                proof.participants == facts.len() && proof.max_batch_sequences >= facts.len()
            });
        let persistent_proof = round.persistent_capacity().filter(|proof| {
            proof.max_live_sequences() >= self.config.max_running_requests
                && proof.max_wave_sequences() >= limit
        });
        let complete_proof = resident_proof || persistent_proof.is_some();
        if !complete_proof && capacity.is_none() {
            return Err(Unsupported::MissingCapacity);
        }
        if !waiting.is_empty()
            && (self.config.max_running_requests > limit || waiting.has_capacity_deferrals())
        {
            return Err(Unsupported::WaitingResources);
        }
        if !waiting.is_empty() && persistent_proof.is_none() {
            let capacity = capacity.ok_or(Unsupported::MissingCapacity)?;
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
                        return Err(Unsupported::AdmissionCapacity);
                    }
                }
                _ => return Err(Unsupported::AdmissionCapacity),
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
                return Err(Unsupported::PendingRequestState);
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
            if fact.terminal.is_some() || fact.generated_tokens >= fact.cap {
                return Err(Unsupported::InvalidState);
            }
            if clock.generated_tokens != fact.generated_tokens
                || clock.captured_at > now
                || clock.itl_threshold
                    != Duration::try_from_secs_f64(round.config.itl_ms / 1000.0).ok()
                || (fact.generated_tokens > 0 && clock.first_text.is_none())
            {
                return Err(Unsupported::InvalidClock);
            }
            if !clock.streaming {
                return Err(Unsupported::NonStreamingClock);
            }
            if fact.generated_tokens > 0
                && clock.nonempty_text_events != fact.generated_tokens as u64
            {
                return Err(Unsupported::HistoricalTextCoalescing);
            }
            let prompt = self
                .prompt_token_estimate(request)
                .ok_or(Unsupported::InvalidState)?;
            if request
                .joint_prefill_phase_cap
                .is_some_and(|cap| cap.prompt_tokens != prompt)
            {
                return Err(Unsupported::InvalidState);
            }
            let prefilled = if request.phase == RequestPhase::Decoding {
                prompt
            } else {
                request.prefill_chunk_offset
            };
            let context_cap = prompt
                .checked_add(fact.cap)
                .ok_or(Unsupported::Arithmetic)?;
            if let Some(proof) = persistent_proof {
                let member = proof
                    .members()
                    .iter()
                    .find(|member| member.request_id == *id && member.lifecycle == fact.lifecycle)
                    .ok_or(Unsupported::AdmissionCapacity)?;
                if matches!(
                    member.state,
                    ferrum_execution::model_executor::PersistentCapacityState::Resident { .. }
                ) != (index < residents)
                {
                    return Err(Unsupported::AdmissionCapacity);
                }
                // The final sampled token is not consumed. Replay may have
                // required a higher original admission ceiling, which remains
                // conservative for this trajectory's prompt + output demand.
                if member.through_tokens < context_cap.saturating_sub(1) {
                    return Err(Unsupported::AdmissionCapacity);
                }
            }
            if !complete_proof {
                // Without exact complete-cohort ownership evidence, charge the
                // entire prompt+cap again. No guessed retained or prefix credit.
                let capacity = capacity.ok_or(Unsupported::MissingCapacity)?;
                blocks = blocks
                    .checked_add(context_cap.div_ceil(capacity.block_size))
                    .ok_or(Unsupported::Arithmetic)?;
            }
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
                phase_cap: request.joint_prefill_phase_cap.map(|cap| cap.limit),
                decode_only_allowed: request
                    .joint_decode_only_deferred_generation
                    .is_none_or(|last| request.logical_work_frontier.progress_generation() > last),
            });
            if cap_mode == AdmissionCapMode::Propose
                && matches!(
                    request.phase,
                    RequestPhase::Waiting | RequestPhase::Prefilling
                )
                && prefilled < prompt
            {
                cap_bindings.push(PrefillCapTarget {
                    request: index,
                    key: PrefixRequestKey::from_request(request)
                        .ok_or(Unsupported::InvalidState)?,
                    phase: request.phase,
                    generation: request.logical_work_frontier.progress_generation(),
                    prompt_tokens: prompt,
                    prefilled_tokens: prefilled,
                    prior_cap: request.joint_prefill_phase_cap,
                });
            }
            initial_admission.push(if index < residents {
                InitialAdmission::Resident
            } else {
                InitialAdmission::Waiting {
                    order: index - residents,
                }
            });
        }
        if !complete_proof && capacity.is_none_or(|capacity| blocks > capacity.free_blocks) {
            return Err(Unsupported::InsufficientCapacity);
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
        let binding_count = if first_wave.iter().any(|member| {
            member.work.kind == ferrum_execution::execution_attempt::AttemptWorkKind::Prefill
                && shapes[member.request]
                    .phase_cap
                    .is_none_or(|cap| cap.get() > 1)
                && requests[member.request]
                    .prompt_tokens
                    .saturating_sub(requests[member.request].prefill_tokens)
                    > 1
        }) {
            batch.len()
        } else {
            0
        };
        let binding_bytes = binding_count
            .checked_mul(std::mem::size_of::<WaveBinding>())
            .and_then(|bytes| bytes.checked_add(round.request_fact_bytes))
            .and_then(|bytes| {
                clocks
                    .len()
                    .checked_mul(std::mem::size_of::<(RequestId, RequestClockFacts)>())
                    .and_then(|clocks| bytes.checked_add(clocks))
            })
            .ok_or(Unsupported::ParticipantBudget)?;
        if binding_bytes > round.config.planning.max_request_fact_bytes {
            return Err(Unsupported::ParticipantBudget);
        }
        let mut wave_bindings = Vec::new();
        wave_bindings
            .try_reserve_exact(binding_count)
            .map_err(|_| Unsupported::ParticipantBudget)?;
        for (scheduled, member) in batch.iter().zip(&first_wave).take(binding_count) {
            let request = prefill
                .iter()
                .chain(decode.requests.values())
                .find(|request| request.inner.request.id == scheduled.request.id)
                .ok_or(Unsupported::InvalidState)?;
            let Some(ticket) = request.waiting_admission_ticket else {
                // Legacy externally constructed fixtures can still use the
                // original actions, but cannot install a lifecycle commitment.
                wave_bindings.clear();
                break;
            };
            wave_bindings.push(WaveBinding {
                id: scheduled.request.id.clone(),
                ticket,
                generation: request.logical_work_frontier.progress_generation(),
                prompt_tokens: requests[member.request].prompt_tokens,
                offset: scheduled.tokens_processed,
                work: member.work,
            });
        }
        Ok(Cohort {
            requests,
            facts: ordered_facts,
            wave_bindings,
            cap_bindings,
            trajectory: CohortTrajectory {
                first_wave,
                cap_commit_targets: Vec::new(),
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
