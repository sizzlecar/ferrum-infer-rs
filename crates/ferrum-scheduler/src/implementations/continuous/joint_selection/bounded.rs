//! Receding finite-event control. The result is a local ranking proxy, never
//! a complete-request feasibility certificate or authority to reject admission.
use super::*;
use crate::joint_policy::{BoundedAccumulator, BoundedTiming};

impl ContinuousBatchScheduler {
    pub(super) fn select_bounded_first_wave(
        &self,
        hint: &BatchHint,
        batch: &mut Vec<ScheduledRequest>,
        round: &mut TimePlanningRound,
        now: Instant,
    ) {
        let started = Instant::now();
        let prior_elapsed = round.prediction_elapsed;
        round.statistics.bounded.comparisons += 1;
        match self.compare_bounded_first_wave(hint, batch, round, now) {
            Ok(Some(action)) => {
                round.statistics.bounded.selected += 1;
                let decode_only = matches!(action.action, FirstAction::DecodeOnly);
                if self.apply_joint_first_wave(batch, round, action) {
                    round.statistics.bounded.applied += 1;
                    round.statistics.joint_interventions += 1;
                    round.statistics.joint_decode_only_interventions += u64::from(decode_only);
                } else {
                    round.statistics.bounded.invalidated += 1;
                    round.statistics.static_fallbacks += 1;
                }
            }
            Ok(None) => round.statistics.bounded.retained += 1,
            Err(reason) => {
                round.statistics.bounded.unknown += 1;
                round.statistics.record_joint_unsupported(reason);
                round.statistics.static_fallbacks += 1;
            }
        }
        round.prediction_elapsed = prior_elapsed.saturating_add(started.elapsed());
    }

    fn compare_bounded_first_wave(
        &self,
        hint: &BatchHint,
        batch: &[ScheduledRequest],
        round: &mut TimePlanningRound,
        now: Instant,
    ) -> std::result::Result<Option<SelectedJointAction>, Unsupported> {
        let started = Instant::now();
        let remaining_time = Duration::from_micros(round.config.planning.max_elapsed_us)
            .checked_sub(round.prediction_elapsed)
            .filter(|duration| !duration.is_zero())
            .ok_or(Unsupported::TimeBudget)?;
        let mut cohort =
            self.joint_cohort(hint, batch, round, now, AdmissionCapMode::ExistingOnly, 0)?;
        // Charge the declared control-delay envelope to every compared action.
        for request in &mut cohort.requests {
            request.age = request
                .age
                .checked_add(remaining_time)
                .ok_or(Unsupported::Arithmetic)?;
        }
        let first_wave = &cohort.trajectory.first_wave;
        let largest = first_wave
            .iter()
            .filter(|member| {
                member.work.kind == ferrum_execution::execution_attempt::AttemptWorkKind::Prefill
            })
            .map(|member| member.work.requested.as_range().len())
            .max()
            .unwrap_or(0);
        let mut alternatives = Vec::new();
        if largest > 1 {
            alternatives.push(FirstAction::PrefillLimit(
                NonZeroUsize::new(largest.div_ceil(2)).unwrap(),
            ));
            if largest.div_ceil(4) < largest.div_ceil(2) {
                alternatives.push(FirstAction::PrefillLimit(
                    NonZeroUsize::new(largest.div_ceil(4)).unwrap(),
                ));
            }
        }
        if cohort.trajectory.allows_decode_only(first_wave) {
            alternatives.push(FirstAction::DecodeOnly);
        }
        if alternatives.is_empty() {
            return Ok(None);
        }
        if alternatives.len() + 1 > round.config.planning.max_candidates {
            return Err(Unsupported::CandidateBudget);
        }
        // These allocations coexist: captured facts/clocks, lifecycle bindings,
        // and the per-action aggregate. Bound them together before allocating
        // the aggregate; separate per-allocation limits are not a total bound.
        let retained_bytes = round
            .request_clocks()
            .ok_or(Unsupported::InvalidClock)?
            .len()
            .checked_mul(std::mem::size_of::<(RequestId, RequestClockFacts)>())
            .and_then(|bytes| bytes.checked_add(round.request_fact_bytes))
            .and_then(|bytes| {
                cohort
                    .wave_bindings
                    .len()
                    .checked_mul(std::mem::size_of::<WaveBinding>())
                    // Applying an action also retains per-member update tuples.
                    .and_then(|bindings| bindings.checked_mul(2))
                    .and_then(|bindings| bytes.checked_add(bindings))
            })
            .and_then(|bytes| {
                BoundedAccumulator::retained_bytes_bound(alternatives.len())
                    .and_then(|aggregate| bytes.checked_add(aggregate))
            })
            .and_then(|bytes| {
                BoundedTiming::retained_bytes_bound(
                    cohort.requests.len(),
                    alternatives.len().checked_add(1)?,
                )
                .and_then(|timing| bytes.checked_add(timing))
            })
            .ok_or(Unsupported::ParticipantBudget)?;
        if retained_bytes > round.config.planning.max_request_fact_bytes {
            return Err(Unsupported::ParticipantBudget);
        }
        let mut config = round.config;
        config.planning.max_queries = round.remaining_queries();
        if config.planning.max_queries == 0 {
            return Err(Unsupported::QueryBudget);
        }
        config.planning.max_elapsed_us = remaining_time
            .checked_sub(started.elapsed())
            .and_then(|duration| u64::try_from(duration.as_micros()).ok())
            .filter(|micros| *micros > 0)
            .ok_or(Unsupported::TimeBudget)?;
        let mut aggregate = BoundedAccumulator::new(&cohort.requests, alternatives.len(), &config)?;
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
                static_action: &FirstAction::Static,
                alternatives: &alternatives,
                horizon: EvaluationHorizon::BoundedRounds(config.planning.max_lookahead_rounds),
            },
            // Explicit survival-to-cap scenario for a finite horizon. No future
            // EOS is invented and no full-cap stopping-CDF allocation is made.
            |_, _| Ok(()),
            |rank, comparison| {
                if aggregate_error.is_none() {
                    aggregate_error = aggregate.observe(rank, comparison).err();
                }
            },
            &mut diagnostics,
        );
        round.statistics.bounded.provider_calls = round
            .statistics
            .bounded
            .provider_calls
            .saturating_add(diagnostics.provider_calls);
        round.statistics.bounded.simulation_queries = round
            .statistics
            .bounded
            .simulation_queries
            .saturating_add(diagnostics.simulation_queries);
        round.statistics.bounded.cache_hits = round
            .statistics
            .bounded
            .cache_hits
            .saturating_add(diagnostics.cache_hits);
        if !round.charge_scenario_queries(&diagnostics) {
            return Err(Unsupported::QueryBudget);
        }
        let summary = summary?;
        if let Some(error) = aggregate_error {
            return Err(error);
        }
        let decision = aggregate.finish(summary)?;
        round.statistics.bounded.evaluated += 1;
        round.statistics.bounded.last_static_score = Some(decision.static_score);
        round.statistics.bounded.last_selected_score = decision.selected_score;
        // The model's cost support expires independently of the CPU budget.
        let apply_at = now
            .checked_add(started.elapsed())
            .ok_or(Unsupported::Arithmetic)?;
        if started.elapsed() >= remaining_time || apply_at >= decision.valid_until {
            return Err(Unsupported::TimeBudget);
        }
        Ok(decision
            .selected
            .and_then(|index| alternatives.get(index).copied())
            .map(|action| SelectedJointAction {
                action,
                wave_bindings: cohort.wave_bindings,
            }))
    }
}
