//! Bounded read-only time selection; resource epochs and executor authority stay
//! in the original admission/commit protocol.
use super::*;
use crate::cost_prediction::*;
use ferrum_execution::execution_attempt::{AttemptPrefixRestore, AttemptWorkKind};
use ferrum_execution::model_executor::{PrefillChunk, PrefillShapePolicy};
use ferrum_execution::token_execution::ExecutionRange;
use ferrum_types::{CostFallback, UnknownCostPolicy};

pub(super) enum TimeDecision {
    Allow(usize),
    Hold,
    Reject(FerrumError),
}

impl TimePlanningRound {
    fn fits_query(&self, count: usize) -> bool {
        CostQuery::fits(count, self.query_limits)
    }

    fn reserve_query(&mut self) -> bool {
        if self.remaining_queries() == 0
            || self.prediction_elapsed >= Duration::from_micros(self.config.planning.max_elapsed_us)
        {
            return false;
        }
        self.query_count += 1;
        true
    }

    fn predict(&mut self, work: &[CostWork], exact: bool, now: Instant) -> CostPrediction {
        let prediction = if !self.reserve_query() {
            CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
        } else if !self.fits_query(work.len()) {
            CostPrediction::Unknown(CostUnknownReason::QueryLimit)
        } else if !exact {
            CostPrediction::Unknown(CostUnknownReason::InexactWork)
        } else {
            let started = Instant::now();
            let prediction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.snapshot.predict(
                    &CostQuery {
                        work,
                        work_is_exact: exact,
                        execution: self.execution,
                    },
                    now,
                )
            }))
            .unwrap_or(CostPrediction::Unknown(
                CostUnknownReason::UnsupportedDomain,
            ));
            self.prediction_elapsed += started.elapsed();
            prediction
        };
        self.validate_prediction(prediction, now)
    }

    fn validate_prediction(&mut self, prediction: CostPrediction, now: Instant) -> CostPrediction {
        let prediction = match prediction {
            CostPrediction::Known { expires_at, .. } if now >= expires_at => {
                CostPrediction::Unknown(CostUnknownReason::Stale)
            }
            CostPrediction::Known {
                central_service_time,
                upper_service_time,
                risk,
                samples,
                observed_at,
                expires_at,
                ..
            } if central_service_time.is_zero()
                || upper_service_time.is_zero()
                || samples == 0
                || observed_at > now
                || expires_at <= observed_at
                || matches!(risk, CostRisk::CalibratedQuantile { quantile, .. }
                    if !quantile.is_finite() || quantile <= 0.0 || quantile >= 1.0) =>
            {
                CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain)
            }
            other => other,
        };
        match prediction {
            CostPrediction::Known { .. } => self.statistics.known_queries += 1,
            CostPrediction::Unknown(_) => self.statistics.unknown_queries += 1,
        }
        prediction
    }

    fn fallback(
        &mut self,
        fallback: CostFallback,
        tokens: usize,
        reason: CostUnknownReason,
    ) -> TimeDecision {
        match fallback {
            CostFallback::StaticFallback => {
                self.statistics.static_fallbacks += 1;
                TimeDecision::Allow(tokens)
            }
            CostFallback::Reject => {
                self.statistics.rejected_candidates += 1;
                TimeDecision::Reject(FerrumError::request_validation(format!(
                    "scheduler cost coverage unavailable: {reason:?}"
                )))
            }
        }
    }

    fn unknown(
        &mut self,
        reason: CostUnknownReason,
        request: &ContinuousBatchRequest,
        tokens: usize,
        now: Instant,
    ) -> TimeDecision {
        self.unknown_policy(self.config.unknown_cost, reason, request, tokens, now)
    }

    pub(super) fn unknown_policy(
        &mut self,
        policy: UnknownCostPolicy,
        reason: CostUnknownReason,
        request: &ContinuousBatchRequest,
        tokens: usize,
        now: Instant,
    ) -> TimeDecision {
        // Cost expiry cannot strand an already admitted/started request. The
        // logical state survives KV release/recompute and includes partial
        // prefill; an offset alone misses admitted zero-offset and replay work.
        let admitted = request.logical_work_frontier.work_kind()
            != pressure::LogicalWorkKind::Waiting
            || request.logical_work_frontier.progress_generation() != LogicalWorkGeneration::ZERO;
        if (matches!(
            self.config.strategy,
            ferrum_types::SloStrategy::Joint | ferrum_types::SloStrategy::Bounded
        ) && admitted)
            || request.capacity_deferred_from_decode
            || request.slo_first_token_emitted
        {
            return self.fallback(CostFallback::StaticFallback, tokens, reason);
        }
        match policy {
            UnknownCostPolicy::StaticFallback => {
                self.fallback(CostFallback::StaticFallback, tokens, reason)
            }
            UnknownCostPolicy::Reject => self.fallback(CostFallback::Reject, tokens, reason),
            UnknownCostPolicy::Wait {
                max_wait_ms,
                on_timeout,
            } => {
                let deadline = request
                    .inner
                    .ingress()
                    .instant()
                    .checked_add(Duration::from_millis(max_wait_ms));
                if let Some(deadline) = deadline.filter(|deadline| *deadline > now) {
                    self.retry_at = Some(self.retry_at.map_or(deadline, |old| old.min(deadline)));
                    self.time_blocked = true;
                    self.statistics.held_candidates += 1;
                    TimeDecision::Hold
                } else {
                    self.fallback(on_timeout, tokens, reason)
                }
            }
        }
    }

    pub(super) fn reset_selection(&mut self) {
        self.bounded_decision = None;
        self.selected.clear();
        self.selected_exact = true;
        self.execution_already_projected = true;
        self.selected_over_limit = false;
    }

    pub(super) fn selected_work(&mut self, work: Option<(CostWork, bool)>) {
        let Some((work, exact)) = work else {
            self.selected_exact = false;
            return;
        };
        if !self.fits_query(self.selected.len().saturating_add(1)) {
            self.selected_over_limit = true;
            return;
        }
        self.selected.push(work);
        self.selected_exact &= exact;
    }

    fn candidate_prediction(
        &mut self,
        work: CostWork,
        exact: bool,
        now: Instant,
    ) -> CostPrediction {
        if !self.reserve_query() {
            return self.validate_prediction(
                CostPrediction::Unknown(CostUnknownReason::PlanningBudget),
                now,
            );
        }
        if self.selected_over_limit || !self.fits_query(self.selected.len().saturating_add(1)) {
            return self
                .validate_prediction(CostPrediction::Unknown(CostUnknownReason::QueryLimit), now);
        }
        if !self.selected_exact || !exact {
            return self
                .validate_prediction(CostPrediction::Unknown(CostUnknownReason::InexactWork), now);
        }
        self.selected.push(work);
        let started = Instant::now();
        let prediction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.snapshot.predict(
                &CostQuery {
                    work: &self.selected,
                    work_is_exact: true,
                    execution: self.execution,
                },
                now,
            )
        }))
        .unwrap_or(CostPrediction::Unknown(
            CostUnknownReason::UnsupportedDomain,
        ));
        self.prediction_elapsed += started.elapsed();
        self.selected.pop();
        self.validate_prediction(prediction, now)
    }
}

impl ContinuousBatchScheduler {
    pub(super) fn cost_work(
        &self,
        request: &ContinuousBatchRequest,
        tokens: usize,
        decode: bool,
        shape_policy: PrefillShapePolicy,
    ) -> Option<(CostWork, bool)> {
        let original_prompt = self.prompt_token_estimate(request);
        let (start, context_tokens, mut exact) = if decode {
            // Prefill samples the first output before the scheduler records a
            // decode count. Later counts include that first sampled token.
            // Engine's attempt context includes the current input token, while
            // its execution range starts one token before that context's end.
            if request.decode_tokens == 0 {
                (
                    request.prefill_tokens,
                    request.prefill_tokens.checked_add(1)?,
                    request.prefill_tokens > 0,
                )
            } else if let Some(prompt) = original_prompt {
                let context = prompt.checked_add(request.decode_tokens)?;
                (context.checked_sub(1)?, context, true)
            } else {
                // A recompute context already includes old generated tokens;
                // adding the whole output count would double count them.
                (request.prefill_tokens, request.prefill_tokens, false)
            }
        } else {
            let context = (request.prefill_tokens > 0)
                .then_some(request.prefill_tokens)
                .or(original_prompt);
            (
                request.prefill_chunk_offset,
                context.unwrap_or_else(|| self.initial_prefill_token_estimate(request)),
                context.is_some()
                    && !(request.capacity_deferred_from_decode && request.prefill_tokens == 0),
            )
        };
        let mut requested = ExecutionRange::new(start, tokens).ok()?;
        exact &= decode || requested.end() <= context_tokens;
        if !decode && exact {
            // Recompute from the current queue frontier each time. A restore,
            // re-admission or completed chunk can change this scope while the
            // same immutable cost snapshot remains in use. An estimated prompt
            // length cannot authorize projecting a checkpoint boundary.
            // A capacity victim may still be an unsampled prefill. Queue
            // re-entry is not evidence that its causal context has outputs.
            let original_prompt = !request.slo_first_token_emitted;
            let projected = shape_policy
                .project(
                    PrefillChunk::new(start, tokens, context_tokens).ok()?,
                    original_prompt,
                )
                .ok()?;
            let chunk = projected.chunk();
            requested =
                ExecutionRange::new(chunk.tokens_processed(), chunk.tokens_to_process()).ok()?;
            exact &= projected.is_exact();
        }
        let restored = request.prefix_restore.restored_tokens();
        Some((
            CostWork {
                kind: if decode {
                    AttemptWorkKind::Decode
                } else {
                    AttemptWorkKind::Prefill
                },
                requested,
                context_tokens,
                prefix_restore: if restored == 0 {
                    AttemptPrefixRestore::NoneObserved
                } else {
                    AttemptPrefixRestore::Restored { tokens: restored }
                },
                internally_reused_tokens: None,
            },
            exact,
        ))
    }

    pub(super) fn admission_time_decision(
        &self,
        request: &ContinuousBatchRequest,
        planning: &mut TimePlanningRound,
        now: Instant,
    ) -> TimeDecision {
        if planning.config.strategy == ferrum_types::SloStrategy::Bounded {
            // A finite control horizon is not an admission feasibility proof.
            // Keep ordinary resource-owner queueing/probes authoritative.
            return TimeDecision::Allow(self.remaining_prefill_tokens(request).max(1));
        }
        if planning.config.strategy == ferrum_types::SloStrategy::Joint {
            return self.joint_admission_time_decision(request, planning, now);
        }
        let maximum = self.remaining_prefill_tokens(request).max(1);
        let projected = self.cost_work(request, maximum, false, planning.prefill_shape_policy);
        let tokens = projected.map_or(maximum, |(work, _)| work.requested.as_range().len());
        let prediction = projected.map_or(
            CostPrediction::Unknown(CostUnknownReason::InexactWork),
            |(work, exact)| planning.predict(&[work], exact, now),
        );
        match prediction {
            // Final batch selection can shrink this candidate with its actual
            // peers and current static limits; admission is not a time permit.
            CostPrediction::Known { .. } => TimeDecision::Allow(tokens),
            CostPrediction::Unknown(reason) => planning.unknown(reason, request, tokens, now),
        }
    }

    pub(super) fn prefill_time_decision(
        &self,
        request: &ContinuousBatchRequest,
        maximum: usize,
        planning: &mut TimePlanningRound,
        now: Instant,
    ) -> TimeDecision {
        if matches!(
            planning.config.strategy,
            ferrum_types::SloStrategy::Joint | ferrum_types::SloStrategy::Bounded
        ) {
            // Preserve the static scheduler's nominal token/quota spending.
            // Exact projected device work is recorded separately below.
            return TimeDecision::Allow(maximum);
        }
        let projected_maximum = self
            .cost_work(request, maximum, false, planning.prefill_shape_policy)
            .map_or(maximum, |(work, _)| work.requested.as_range().len());
        let mut tokens = projected_maximum;
        let has_decode = planning
            .selected
            .iter()
            .any(|work| work.kind == AttemptWorkKind::Decode);
        let ttft_remaining =
            (planning.config.ttft_ms - request.inner.age_at(now).as_secs_f64() * 1000.0).max(0.0);
        let target_ms = if has_decode {
            planning.config.itl_ms
        } else if request.slo_first_token_emitted || request.capacity_deferred_from_decode {
            planning.config.itl_ms
        } else {
            ttft_remaining
        };
        let target = Duration::try_from_secs_f64(target_ms / 1000.0).unwrap_or(Duration::ZERO);
        loop {
            let projected = self.cost_work(request, tokens, false, planning.prefill_shape_policy);
            // Halving starts from actual projected work, never an unexecuted
            // nominal span beyond a checkpoint. Projection is idempotent.
            tokens = projected.map_or(tokens, |(work, _)| work.requested.as_range().len());
            let prediction = projected.map_or(
                CostPrediction::Unknown(CostUnknownReason::InexactWork),
                |(work, exact)| planning.candidate_prediction(work, exact, now),
            );
            match prediction {
                CostPrediction::Known {
                    upper_service_time, ..
                } if upper_service_time <= target => {
                    planning.statistics.shrunk_prefills += u64::from(tokens < projected_maximum);
                    return TimeDecision::Allow(tokens);
                }
                CostPrediction::Known { .. } if tokens > 1 => {
                    // Count each observed over-budget query, independently of
                    // whether a smaller candidate is eventually permitted.
                    planning.statistics.budget_misses += 1;
                    tokens = tokens.div_ceil(2).min(tokens - 1);
                }
                CostPrediction::Known { .. } => {
                    planning.statistics.budget_misses += 1;
                    // There is no finite evidence that waiting improves a
                    // singleton's cost. Keep one-token progress; when peers
                    // already run, defer only this addition to their next turn.
                    if planning.selected.is_empty() && !planning.selected_over_limit {
                        planning.statistics.shrunk_prefills += u64::from(projected_maximum > 1);
                        return TimeDecision::Allow(1);
                    }
                    planning.time_blocked = true;
                    planning.statistics.held_candidates += 1;
                    return TimeDecision::Hold;
                }
                CostPrediction::Unknown(reason) => {
                    // A known oversized combination does not establish the
                    // smaller combination's cost. Static fallback can defer
                    // this addition while its already-selected peers advance;
                    // explicit Reject/Wait retain their declared semantics.
                    if tokens < projected_maximum
                        && !planning.selected.is_empty()
                        && matches!(
                            planning.config.unknown_cost,
                            UnknownCostPolicy::StaticFallback
                        )
                    {
                        planning.time_blocked = true;
                        planning.statistics.held_candidates += 1;
                        return TimeDecision::Hold;
                    }
                    let decision = planning.unknown(reason, request, tokens, now);
                    if matches!(decision, TimeDecision::Allow(allowed) if allowed < projected_maximum)
                    {
                        planning.statistics.shrunk_prefills += 1;
                    }
                    return decision;
                }
            }
        }
    }

    pub(super) fn prioritize_ttft(
        &self,
        queue: &mut VecDeque<ContinuousBatchRequest>,
        now: Instant,
        planning: Option<&mut TimePlanningRound>,
    ) {
        let Some(planning) = planning else {
            return;
        };
        if matches!(
            planning.config.strategy,
            ferrum_types::SloStrategy::Joint | ferrum_types::SloStrategy::Bounded
        ) {
            return;
        }
        let urgency_ms = planning.config.tpot_ms.min(planning.config.itl_ms);
        queue.make_contiguous().sort_by_cached_key(|request| {
            if request.slo_first_token_emitted || request.capacity_deferred_from_decode {
                return (true, i128::MAX);
            }
            let tokens = self.remaining_prefill_tokens(request).max(1);
            let estimate = self
                .cost_work(request, tokens, false, planning.prefill_shape_policy)
                .and_then(
                    |(work, exact)| match planning.predict(&[work], exact, now) {
                        CostPrediction::Known {
                            upper_service_time, ..
                        } => Some(upper_service_time),
                        CostPrediction::Unknown(_) => None,
                    },
                )
                .unwrap_or_default();
            let slack_ms = planning.config.ttft_ms
                - request.inner.age_at(now).as_secs_f64() * 1000.0
                - estimate.as_secs_f64() * 1000.0;
            if slack_ms > urgency_ms {
                (true, i128::MAX)
            } else {
                (false, (slack_ms * 1_000_000.0) as i128)
            }
        });
    }
}
