use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
mod cache;
use cache::ScenarioCache;

#[cfg(test)]
mod tests;

/// Complete traversal of the declared empirical rank support. The callback's
/// partial aggregates must be discarded if traversal returns an error. Each
/// comparison can still contain unsupported alternatives; those remain unknown.
#[derive(Debug, Clone, Copy)]
pub struct ScenarioEvaluation {
    /// Earliest expiry of any conditional cost distribution actually queried.
    pub valid_until: Instant,
    pub dependence: CostScenarioDependence,
    pub ranks: usize,
    pub queries: usize,
    pub cache_hits: usize,
    pub cache_retained_bytes: usize,
    pub resolved_mass: f64,
    pub unresolved_mass: f64,
    pub unresolved_ranks: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ScenarioRank {
    pub index: usize,
    pub count: usize,
    pub weight: f64,
    pub quantile_lower: f64,
    pub quantile_upper: f64,
}

impl SloPolicy {
    /// Visit one rank at a time without retaining rank × candidate × cohort
    /// copies. A single global query/time budget covers seed and every rank.
    /// Shared residual rank is a sensitivity assumption, not joint calibration
    /// coverage, and these differences never substitute for paired evidence.
    pub fn evaluate_scenarios<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        mut visit: impl FnMut(usize, f64, &Comparison),
    ) -> Result<ScenarioEvaluation, Unsupported> {
        Self::evaluate_scenarios_with_requests(
            input,
            |_, _| Ok(()),
            |rank, comparison| {
                visit(rank.index, rank.weight, comparison);
            },
        )
    }

    /// Assign discrete stopping endpoints for the same global empirical cost
    /// rank. Only `scenario_end_tokens` may change. Unsupported assignments
    /// retain their entire weight as unresolved; they are never filled at cap
    /// or removed by renormalizing the remaining rank weights.
    pub fn evaluate_scenarios_with_requests<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        mut assign: impl FnMut(ScenarioRank, &mut [SimulatedRequest]) -> Result<(), Unsupported>,
        mut visit: impl FnMut(ScenarioRank, &Comparison),
    ) -> Result<ScenarioEvaluation, Unsupported> {
        if input.requests.is_empty() {
            return Err(Unsupported::EmptyCohort);
        }
        if input.config.planning.max_queries == 0 {
            return Err(Unsupported::QueryBudget);
        }
        if input.requests.len() > input.config.planning.max_participants {
            return Err(Unsupported::ParticipantBudget);
        }
        admission::validate_admission(&input)?;
        if !input.resources.kv_reserved_through_scenario_end {
            return Err(Unsupported::UnknownKvCapacity);
        }
        let started = Instant::now();
        let limits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            input.snapshot.query_limits()
        }))
        .map_err(|_| Unsupported::Cost(CostUnknownReason::UnsupportedDomain))?;
        let mut plan = PlanBuilder {
            groups: Vec::new(),
            participant_limit: input.config.planning.max_participants,
            query_limits: input.config.query.intersect(limits),
            execution: input.execution,
            active: input
                .adapter
                .admission_slots()
                .map(|slots| slots.active_mask(input.requests))
                .transpose()?,
        };
        input
            .adapter
            .plan(Some(input.static_action), input.requests, &mut plan)?;
        let group = plan.groups.first().ok_or(Unsupported::NoProgress)?;
        for member in group {
            validate_member(input.requests, *member)?;
        }
        let work = group.iter().map(|member| member.work).collect::<Vec<_>>();
        let seed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            input.snapshot.cost_scenarios(
                &CostQuery {
                    work: &work,
                    work_is_exact: true,
                    execution: CostExecution::SingleInvocation,
                },
                input.now,
                input.config.planning.max_scenarios,
            )
        }))
        .map_err(|_| Unsupported::Cost(CostUnknownReason::UnsupportedDomain))?
        .validated(input.now, input.config.planning.max_scenarios);
        let (weights, dependence, seed_expiry) = match &seed {
            CostScenarioPrediction::Known {
                scenarios,
                dependence,
                expires_at,
                ..
            } => (
                scenarios
                    .iter()
                    .map(|scenario| scenario.weight)
                    .collect::<Vec<_>>(),
                *dependence,
                *expires_at,
            ),
            CostScenarioPrediction::Unknown(reason) => return Err(Unsupported::Cost(*reason)),
        };
        let queries = AtomicUsize::new(1);
        let mut cache = ScenarioCache::new(
            input.now,
            input.config.planning.max_cost_cache_bytes,
            input.config.planning.max_queries,
        );
        cache.insert(
            &CostQuery {
                work: &work,
                work_is_exact: true,
                execution: CostExecution::SingleInvocation,
            },
            seed,
        );
        let cache = parking_lot::Mutex::new(cache);
        let valid_until = parking_lot::Mutex::new(seed_expiry);
        let mut resolved_mass = 0.0;
        let mut unresolved_mass = 0.0;
        let mut unresolved_ranks = 0;
        let mut quantile_lower = 0.0;
        for (rank, weight) in weights.iter().copied().enumerate() {
            if started.elapsed() >= Duration::from_micros(input.config.planning.max_elapsed_us) {
                return Err(Unsupported::TimeBudget);
            }
            let rank_info = ScenarioRank {
                index: rank,
                count: weights.len(),
                weight,
                quantile_lower,
                quantile_upper: (quantile_lower + weight).min(1.0),
            };
            quantile_lower = rank_info.quantile_upper;
            let mut requests = input.requests.to_vec();
            if assign(rank_info, &mut requests).is_err() {
                unresolved_mass += weight;
                unresolved_ranks += 1;
                continue;
            }
            if requests
                .iter()
                .zip(input.requests)
                .any(|(request, original)| {
                    request.id != original.id
                        || request.prompt_tokens != original.prompt_tokens
                        || request.prefill_tokens != original.prefill_tokens
                        || request.output_tokens != original.output_tokens
                        || request.output_cap != original.output_cap
                        || request.age != original.age
                        || request.central_clock != original.central_clock
                        || request.risk_clock != original.risk_clock
                })
            {
                return Err(Unsupported::InvalidState);
            }
            let snapshot = ConditionalSnapshot {
                original: input.snapshot,
                rank,
                weights: &weights,
                dependence,
                query_limits: limits,
                config: input.config,
                started,
                queries: &queries,
                cache: &cache,
                valid_until: &valid_until,
            };
            let comparison = Self::evaluate(EvaluationInput {
                snapshot: &snapshot,
                now: input.now,
                config: input.config,
                execution: input.execution,
                requests: &requests,
                resources: input.resources,
                adapter: input.adapter,
                static_action: input.static_action,
                alternatives: input.alternatives,
                static_window_rounds: input.static_window_rounds,
            })?;
            visit(rank_info, &comparison);
            resolved_mass += weight;
        }
        if started.elapsed() >= Duration::from_micros(input.config.planning.max_elapsed_us) {
            return Err(Unsupported::TimeBudget);
        }
        let valid_until = *valid_until.lock();
        let cache = cache.lock();
        Ok(ScenarioEvaluation {
            valid_until,
            dependence,
            ranks: weights.len(),
            queries: queries.load(Ordering::Relaxed),
            cache_hits: cache.hits(),
            cache_retained_bytes: cache.retained_bytes(),
            resolved_mass,
            unresolved_mass,
            unresolved_ranks,
        })
    }
}

struct ConditionalSnapshot<'a> {
    original: &'a dyn CostSnapshot,
    rank: usize,
    weights: &'a [f64],
    dependence: CostScenarioDependence,
    query_limits: CostQueryLimits,
    config: &'a SchedulerSloConfig,
    started: Instant,
    queries: &'a AtomicUsize,
    cache: &'a parking_lot::Mutex<ScenarioCache>,
    valid_until: &'a parking_lot::Mutex<Instant>,
}

impl CostSnapshot for ConditionalSnapshot<'_> {
    fn revision(&self) -> u64 {
        self.original.revision()
    }
    fn query_limits(&self) -> CostQueryLimits {
        self.query_limits
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        if self.started.elapsed() >= Duration::from_micros(self.config.planning.max_elapsed_us) {
            return CostPrediction::Unknown(CostUnknownReason::PlanningBudget);
        }
        // This mutex protects only comparison-local memoization, never the
        // estimator or execution. No lock is held across an original query.
        if let Some(prediction) = self.cache.lock().get(query, now, self.rank) {
            return prediction;
        }
        if self
            .queries
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |queries| {
                (queries < self.config.planning.max_queries).then(|| queries + 1)
            })
            .is_err()
        {
            return CostPrediction::Unknown(CostUnknownReason::PlanningBudget);
        }
        let distribution = self
            .original
            .cost_scenarios(query, now, self.config.planning.max_scenarios)
            .validated(now, self.config.planning.max_scenarios);
        match &distribution {
            CostScenarioPrediction::Unknown(reason) => CostPrediction::Unknown(*reason),
            CostScenarioPrediction::Known {
                scenarios,
                dependence,
                support,
                samples,
                observed_at,
                expires_at,
                ..
            } => {
                if *dependence != self.dependence
                    || scenarios.len() != self.weights.len()
                    || scenarios
                        .iter()
                        .zip(self.weights)
                        .any(|(scenario, weight)| (scenario.weight - weight).abs() > 1e-12)
                    || self.started.elapsed()
                        >= Duration::from_micros(self.config.planning.max_elapsed_us)
                {
                    return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
                }
                let service_time = scenarios[self.rank].service_time;
                let mut valid_until = self.valid_until.lock();
                *valid_until = (*valid_until).min(*expires_at);
                let prediction = CostPrediction::Known {
                    central_service_time: service_time,
                    upper_service_time: service_time,
                    risk: CostRisk::ConditionalScenario {
                        dependence: *dependence,
                    },
                    support: *support,
                    samples: *samples,
                    observed_at: *observed_at,
                    expires_at: *expires_at,
                };
                drop(valid_until);
                self.cache.lock().insert(query, distribution);
                prediction
            }
        }
    }
}
