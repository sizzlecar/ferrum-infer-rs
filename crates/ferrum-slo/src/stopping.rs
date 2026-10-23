//! Admission-ordered, bounded stopping evidence. Window maintenance runs off the
//! inference thread; immutable forecasts perform bounded query work. Unknown
//! suffixes are censored, never treated as active or complete.

use ferrum_interfaces::request_observation::*;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

mod observer;
pub use observer::{
    StoppingCollectorConfig, StoppingCollectorStats, StoppingEstimatorOwner, StoppingWorkerExit,
};

#[derive(Debug, Clone, Copy)]
pub struct StoppingConfig {
    pub window_requests: usize,
    pub max_domains: usize,
    pub max_rule_bytes: usize,
    pub min_risk_set: usize,
    pub max_forecast_positions: usize,
    /// Maximum fact age for stopping hazards and maximum ingress age for the
    /// completion cohort. Completion uses the inclusive [maturity, ttl] range
    /// for every outcome, including active and missing observations.
    pub ttl: Duration,
}

impl Default for StoppingConfig {
    fn default() -> Self {
        Self {
            window_requests: 512,
            max_domains: 32,
            max_rule_bytes: 4096,
            min_risk_set: 16,
            max_forecast_positions: 4096,
            ttl: Duration::from_secs(60),
        }
    }
}

impl StoppingConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.window_requests == 0
            || self.max_domains == 0
            || self.max_rule_bytes == 0
            || self.min_risk_set == 0
            || self.min_risk_set > self.window_requests
            || self.max_forecast_positions == 0
            || self.ttl.is_zero()
        {
            return Err("stopping capacities, support and TTL must be positive and consistent");
        }
        std::alloc::Layout::array::<Record>(self.window_requests)
            .map_err(|_| "stopping window exceeds addressable memory")?;
        std::alloc::Layout::array::<StoppingMass>(self.max_forecast_positions)
            .map_err(|_| "stopping forecast exceeds addressable memory")?;
        std::alloc::Layout::array::<RiskEvents>(self.max_forecast_positions)
            .map_err(|_| "stopping forecast scratch exceeds addressable memory")?;
        self.window_requests
            .checked_mul(self.max_rule_bytes)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or("stopping identity storage exceeds addressable memory")?;
        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct StoppingStats {
    pub admitted: u64,
    pub evicted: u64,
    pub late_retired: u64,
    pub invalid: u64,
    pub conflicts: u64,
    pub domain_limit: u64,
    pub stale_updates: u64,
    pub unobserved_lifetimes: usize,
}

#[derive(Clone)]
struct Record {
    lifecycle: RequestLifecycleId,
    admitted_at: Instant,
    // Domain-capacity misses retain only lifecycle membership. Keeping these
    // unknown entries in the same admission window allows old domains to age
    // out without retaining unbounded identities or favoring completed work.
    fact: Option<RequestStoppingObservation>,
    observed_at: Instant,
    conflicted: bool,
}

pub struct StoppingWindow {
    config: StoppingConfig,
    producer: Option<u64>,
    retired_through: u64,
    highest_admission: u64,
    records: VecDeque<Record>,
    admission_times: Vec<Option<(RequestLifecycleId, Instant)>>,
    stats: StoppingStats,
}

impl StoppingWindow {
    pub fn new(config: StoppingConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            producer: None,
            retired_through: 0,
            highest_admission: 0,
            records: VecDeque::new(),
            admission_times: vec![None; config.window_requests],
            stats: StoppingStats::default(),
        })
    }

    pub fn stats(&self) -> StoppingStats {
        StoppingStats {
            unobserved_lifetimes: self.unobserved_lifetimes(),
            ..self.stats
        }
    }
    pub fn retained_requests(&self) -> usize {
        self.records.len()
    }

    fn unobserved_lifetimes(&self) -> usize {
        (self.highest_admission - self.retired_through) as usize - self.records.len()
    }

    /// A serial range, not a completed/observed-only window. Large gaps cost
    /// constant metadata and cannot keep favorable old completions alive.
    pub fn advance_admission_watermark(&mut self, lifecycle: RequestLifecycleId) -> bool {
        if lifecycle.producer == 0
            || lifecycle.admission == 0
            || self
                .producer
                .is_some_and(|producer| producer != lifecycle.producer)
        {
            self.stats.invalid += 1;
            return false;
        }
        self.producer = Some(lifecycle.producer);
        if lifecycle.admission <= self.highest_admission {
            return false;
        }
        self.highest_admission = lifecycle.admission;
        self.retired_through = self
            .highest_admission
            .saturating_sub(self.config.window_requests as u64);
        while self
            .records
            .front()
            .is_some_and(|record| record.lifecycle.admission <= self.retired_through)
        {
            self.records.pop_front();
            self.stats.evicted += 1;
        }
        true
    }

    /// Exact producer ingress time for a serial, independently of its lossy
    /// cumulative facts. A newer serial does not imply anything about the
    /// ingress time of a missing older one: concurrent ingress can reorder.
    pub fn record_admission(
        &mut self,
        lifecycle: RequestLifecycleId,
        admitted_at: Instant,
    ) -> bool {
        self.advance_admission_watermark(lifecycle);
        if self.producer != Some(lifecycle.producer)
            || lifecycle.admission <= self.retired_through
            || lifecycle.admission > self.highest_admission
        {
            return false;
        }
        let slot = &mut self.admission_times
            [(lifecycle.admission % self.config.window_requests as u64) as usize];
        if let Some((existing, time)) = slot {
            if *existing == lifecycle {
                // An inconsistent duplicate cannot renew an old admission.
                let older = (*time).min(admitted_at);
                let changed = older != *time;
                *time = older;
                return changed;
            }
        }
        *slot = Some((lifecycle, admitted_at));
        true
    }

    /// Receives full cumulative facts. Ingress may reorder notices, therefore
    /// insertion and eviction use the producer's admission serial explicitly.
    pub fn observe(&mut self, observation: &RequestStoppingObservation, now: Instant) {
        if !observation.is_consistent()
            || observation.observed_at > now
            || observation.rules.identity().len() > self.config.max_rule_bytes
            || self
                .producer
                .is_some_and(|producer| producer != observation.lifecycle.producer)
        {
            self.stats.invalid += 1;
            return;
        }
        self.record_admission(observation.lifecycle, observation.admitted_at);
        if observation.lifecycle.admission <= self.retired_through {
            self.stats.late_retired += 1;
            return;
        }
        if let Some(record) = self
            .records
            .iter_mut()
            .find(|record| record.lifecycle == observation.lifecycle)
        {
            let Some(previous) = record.fact.as_mut() else {
                return;
            };
            if observation.revision < previous.revision {
                self.stats.stale_updates += 1;
                return;
            }
            if observation == previous {
                return;
            }
            if previous.terminal.is_some()
                && observation.terminal == previous.terminal
                && observation.admitted_at == previous.admitted_at
                && observation.rules == previous.rules
                && observation.cap == previous.cap
                && observation.generated_tokens == previous.generated_tokens
                && observation.verified_survival_through == previous.verified_survival_through
            {
                if observation.completion == previous.completion {
                    // An identical terminal retry cannot refresh stopping or
                    // completion freshness merely by arriving later.
                    previous.revision = previous.revision.max(observation.revision);
                    return;
                }
                if !record.conflicted
                    && previous.completion.is_none()
                    && observation.completion.is_some()
                    && observation.revision > previous.revision
                    && observation.observed_at >= previous.observed_at
                {
                    // Completion enrichment is distinct from natural stopping;
                    // preserve record.observed_at for the original hazard TTL.
                    *previous = observation.clone();
                    return;
                }
            }
            let consistent = observation.revision > previous.revision
                && observation.rules == previous.rules
                && observation.cap == previous.cap
                && observation.admitted_at == previous.admitted_at
                && observation.generated_tokens >= previous.generated_tokens
                && observation.observed_at >= previous.observed_at
                && observation.verified_survival_through >= previous.verified_survival_through
                && previous.terminal.is_none();
            if !consistent || record.conflicted {
                // Preserve only the common verified prefix. A contradiction
                // cannot be healed by a later favorable terminal notice.
                previous.verified_survival_through = previous
                    .verified_survival_through
                    .min(observation.verified_survival_through);
                if observation.rules != previous.rules || observation.cap != previous.cap {
                    previous.verified_survival_through = 0;
                }
                previous.terminal = None;
                previous.completion = None;
                previous.revision = previous.revision.max(observation.revision);
                if !record.conflicted {
                    self.stats.conflicts += 1;
                }
                record.conflicted = true;
                return;
            }
            *previous = observation.clone();
            record.observed_at = observation.observed_at;
            return;
        }
        // Count full identities, never an unverified hash match. This scan is
        // bounded by the single request window, including censored records.
        let mut domain_supported = true;
        if !self.records.iter().any(|record| {
            record
                .fact
                .as_ref()
                .is_some_and(|fact| fact.rules == observation.rules)
        }) {
            let mut domains: Vec<&EffectiveStoppingRules> = Vec::new();
            for record in &self.records {
                let Some(fact) = &record.fact else {
                    continue;
                };
                if !domains.contains(&fact.rules.as_ref()) {
                    domains.push(&fact.rules);
                }
            }
            if domains.len() >= self.config.max_domains {
                self.stats.domain_limit += 1;
                domain_supported = false;
            }
        }
        let position = self
            .records
            .iter()
            .position(|record| record.lifecycle.admission > observation.lifecycle.admission)
            .unwrap_or(self.records.len());
        self.records.insert(
            position,
            Record {
                lifecycle: observation.lifecycle,
                admitted_at: observation.admitted_at,
                fact: domain_supported.then(|| observation.clone()),
                observed_at: observation.observed_at,
                conflicted: false,
            },
        );
        self.stats.admitted += 1;
    }

    pub fn snapshot(&self) -> StoppingTable {
        let mut unobserved_admission_times = Vec::with_capacity(self.unobserved_lifetimes());
        for admission in self.retired_through.saturating_add(1)..=self.highest_admission {
            if self
                .records
                .iter()
                .any(|record| record.lifecycle.admission == admission)
            {
                continue;
            }
            let time = self.admission_times
                [(admission % self.config.window_requests as u64) as usize]
                .filter(|(lifecycle, _)| lifecycle.admission == admission)
                .map(|(_, time)| time);
            unobserved_admission_times.push(time);
        }
        StoppingTable {
            config: self.config,
            records: self.records.iter().cloned().collect(),
            unobserved_admission_times,
        }
    }
}

#[derive(Clone)]
pub struct StoppingTable {
    config: StoppingConfig,
    records: Vec<Record>,
    unobserved_admission_times: Vec<Option<Instant>>,
}

/// One bounded query's integer event counts, discarded after forecasting.
/// A record contributes to the risk set through its verified prefix, including
/// the natural endpoint itself when one was observed.
#[derive(Clone, Copy, Default)]
struct RiskEvents {
    risk_ends: usize,
    natural_ends: usize,
}

impl StoppingSnapshot for StoppingTable {
    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        if query.cap == 0
            || query.maturity.is_zero()
            || query.maturity >= self.config.ttl
            || query.rules.identity().len() > self.config.max_rule_bytes
        {
            return CompletionForecast::unknown();
        }
        // Select the cohort by immutable ingress age before examining outcomes.
        // Otherwise expired completed requests would remain unknown forever in
        // a low-arrival window while recent successes entered its denominator.
        // Missing timestamps cannot prove either age boundary and stay unknown.
        let missing_mature = self
            .unobserved_admission_times
            .iter()
            .filter(|time| {
                time.and_then(|time| query.now.checked_duration_since(time))
                    .is_none_or(|age| age >= query.maturity && age <= self.config.ttl)
            })
            .count();
        let (mut matched, mut succeeded, mut failed, mut unresolved) =
            (0usize, 0usize, 0usize, missing_mature);
        for record in &self.records {
            if query
                .now
                .checked_duration_since(record.admitted_at)
                .is_none_or(|age| age < query.maturity || age > self.config.ttl)
            {
                continue;
            }
            let Some(fact) = &record.fact else {
                // The unsupported domain may match this query; it cannot be
                // silently removed from a favorable completion denominator.
                unresolved += 1;
                continue;
            };
            if fact.rules.as_ref() != query.rules || fact.cap != query.cap {
                continue;
            }
            matched += 1;
            if record.conflicted
                || query
                    .now
                    .checked_duration_since(fact.observed_at)
                    .is_none_or(|age| age > self.config.ttl)
            {
                unresolved += 1;
                continue;
            }
            match fact.completion {
                Some(RequestCompletionOutcome::Succeeded) => succeeded += 1,
                Some(RequestCompletionOutcome::Failed) => failed += 1,
                None => unresolved += 1,
            }
        }
        if matched < self.config.min_risk_set {
            return CompletionForecast::unknown();
        }
        let samples = succeeded + failed + unresolved;
        CompletionForecast {
            succeeded_mass: succeeded as f64 / samples as f64,
            failed_mass: failed as f64 / samples as f64,
            unresolved_mass: unresolved as f64 / samples as f64,
            samples,
            approximation: Some(CompletionApproximation::EmpiricalMaturedCohort),
        }
    }

    fn forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        if query.cap == 0
            || query.verified_survival_through >= query.cap
            || query.horizon <= query.verified_survival_through
            || query.rules.identity().len() > self.config.max_rule_bytes
        {
            return StoppingForecast::unknown();
        }
        let horizon = query.horizon.min(query.cap);
        if query.rules.mechanism() == StoppingMechanism::CapOnly
            || query.verified_survival_through.checked_add(1) == Some(query.cap)
        {
            return if horizon == query.cap {
                StoppingForecast {
                    stops: vec![StoppingMass {
                        generated_tokens: query.cap,
                        mass: 1.0,
                    }],
                    known_survival_mass: 0.0,
                    unresolved_mass: 0.0,
                    tpot_undefined_mass: if query.cap < 2 { 1.0 } else { 0.0 },
                }
            } else {
                StoppingForecast {
                    stops: Vec::new(),
                    known_survival_mass: 1.0,
                    unresolved_mass: 0.0,
                    tpot_undefined_mass: 0.0,
                }
            };
        }
        // Only statistical positions need scratch: the cap remains an explicit
        // endpoint and is handled before the work-limit check below. Subtraction
        // avoids forming cap + 1 or horizon + 1, including usize::MAX queries.
        let positions = horizon
            .min(query.cap - 1)
            .saturating_sub(query.verified_survival_through)
            .min(self.config.max_forecast_positions);
        let mut events = vec![RiskEvents::default(); positions].into_boxed_slice();
        let mut unknown_members = self.unobserved_admission_times.len();
        let mut known_members = 0usize;
        let mut risk = 0usize;
        // Rule equality and freshness are invariant throughout this query.
        // Filter once, then sweep integer endpoints instead of rescanning every
        // retained request at every forecast position.
        for record in &self.records {
            let Some(fact) = &record.fact else {
                // Preserve missing-domain membership even when it has no usable
                // observation timestamp; it was never a freshness-filtered fact.
                unknown_members += 1;
                continue;
            };
            if fact.rules.as_ref() != query.rules
                || query
                    .now
                    .checked_duration_since(record.observed_at)
                    .is_none_or(|age| age > self.config.ttl)
            {
                continue;
            }
            let terminal_stop = matches!(
                fact.terminal,
                Some(RequestTerminalReason::Eos | RequestTerminalReason::Stop)
            );
            if !(terminal_stop && fact.generated_tokens <= query.verified_survival_through) {
                known_members += 1;
            }
            let natural_end = !record.conflicted && terminal_stop;
            // All facts enter through is_consistent(): an EOS/Stop at g has
            // verified survival g-1. Conflict handling clears the terminal and
            // retains only a common verified prefix; completion enrichment does
            // not change either boundary. Thus this is one continuous risk
            // interval, exactly the old natural-end OR verified-prefix test.
            let risk_through = if natural_end {
                fact.verified_survival_through.max(fact.generated_tokens)
            } else {
                fact.verified_survival_through
            };
            if let Some(distance) = risk_through
                .checked_sub(query.verified_survival_through)
                .filter(|distance| *distance > 0)
            {
                risk += 1;
                if let Some(event) = events.get_mut(distance - 1) {
                    event.risk_ends += 1;
                }
            }
            if natural_end {
                if let Some(distance) = fact
                    .generated_tokens
                    .checked_sub(query.verified_survival_through)
                    .filter(|distance| *distance > 0)
                {
                    if let Some(event) = events.get_mut(distance - 1) {
                        event.natural_ends += 1;
                    }
                }
            }
        }
        let unknown_mass = if unknown_members == 0 {
            0.0
        } else {
            unknown_members as f64 / (known_members + unknown_members) as f64
        };
        let mut forecast = StoppingForecast {
            stops: Vec::new(),
            known_survival_mass: 1.0 - unknown_mass,
            unresolved_mass: unknown_mass,
            tpot_undefined_mass: 0.0,
        };
        for (position, generated) in ((query.verified_survival_through + 1)..=horizon).enumerate() {
            // Cap is a declared endpoint, independent of statistical support.
            // An earlier gap already moved survivor mass to unknown and broke.
            if generated == query.cap {
                push_stop(&mut forecast, generated, 1.0);
                break;
            }
            if position >= self.config.max_forecast_positions {
                forecast.unresolved_mass += forecast.known_survival_mass;
                forecast.known_survival_mass = 0.0;
                break;
            }
            let event = events[position];
            if risk < self.config.min_risk_set {
                forecast.unresolved_mass += forecast.known_survival_mass;
                forecast.known_survival_mass = 0.0;
                break;
            }
            if event.natural_ends > 0 {
                push_stop(
                    &mut forecast,
                    generated,
                    event.natural_ends as f64 / risk as f64,
                );
            }
            if forecast.known_survival_mass <= 0.0 {
                break;
            }
            risk -= event.risk_ends;
        }
        forecast
    }
}

fn push_stop(forecast: &mut StoppingForecast, generated_tokens: usize, hazard: f64) {
    let mass = forecast.known_survival_mass * hazard;
    forecast.known_survival_mass -= mass;
    if mass > 0.0 {
        forecast.stops.push(StoppingMass {
            generated_tokens,
            mass,
        });
        if generated_tokens < 2 {
            forecast.tpot_undefined_mass += mass;
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod forecast_scan_tests;
