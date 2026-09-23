//! Trusted in-process ingress context and constant-size token timing state.
//!
//! These types deliberately have no serde implementation. Wire metadata cannot
//! supply a monotonic clock or grant a service class. Product entrypoints capture
//! ingress before parsing, rendering or tokenization and pass it unchanged.

use ferrum_types::{FerrumError, Result, SloConfig, SloLatencyBudgets, SloMode};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct InferenceRequestContext {
    ingress: Instant,
    service_class: Option<String>,
}

impl InferenceRequestContext {
    pub fn capture() -> Self {
        Self::from_ingress(Instant::now())
    }

    /// For a trusted transport boundary or an explicit virtual-clock fixture.
    pub fn from_ingress(ingress: Instant) -> Self {
        Self {
            ingress,
            service_class: None,
        }
    }

    /// The caller must resolve this class using server-owned routing policy.
    /// Never populate it from arbitrary request metadata or a client timestamp.
    pub fn with_service_class(mut self, service_class: String) -> Self {
        self.service_class = Some(service_class);
        self
    }

    pub fn ingress(&self) -> Instant {
        self.ingress
    }

    pub fn service_class(&self) -> Option<&str> {
        self.service_class.as_deref()
    }

    /// Resolve only the engine's configured policy, without granting resource
    /// authority or a time-admission promise. Validate the policy at startup.
    pub fn resolve_slo(&self, config: &SloConfig) -> Result<Option<RequestSloState>> {
        if config.mode == SloMode::Off {
            return Ok(None);
        }
        let class = self
            .service_class()
            .or(config.default_service_class.as_deref())
            .ok_or_else(|| FerrumError::config("active SLO mode requires a service class"))?;
        let service = config
            .services
            .iter()
            .find(|service| service.id == class)
            .ok_or_else(|| {
                FerrumError::invalid_request(format!("unknown SLO service class {class:?}"))
            })?;
        RequestSloState::new(self.ingress, class.to_owned(), service.server_token_commit).map(Some)
    }
}

/// Sticky failures at the internal ingress/token-commit boundary. Client-visible
/// SSE intervals are measured separately and cannot be inferred from these bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SloTimingViolations {
    pub ttft: bool,
    pub tpot: bool,
    pub itl: bool,
}

impl SloTimingViolations {
    pub fn any(self) -> bool {
        self.ttft || self.tpot || self.itl
    }
}

/// The internal token obligation whose first unrecorded violation needs a
/// timer. This is separate from capacity and output readiness notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SloTimingBoundary {
    FirstToken,
    InterToken,
    TokenPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SloViolationWake {
    pub at: Instant,
    pub boundary: SloTimingBoundary,
}

/// Lightweight controller state. Request identity, incarnation, readiness and
/// physical permits remain owned by the existing request/executor lifecycle.
#[derive(Debug, Clone)]
pub struct RequestSloState {
    ingress: Instant,
    service_class: String,
    budgets: SloLatencyBudgets,
    first_commit: Option<Instant>,
    last_commit: Option<Instant>,
    max_token_gap: Option<Duration>,
    committed_tokens: u64,
    violations: SloTimingViolations,
    last_observation: Instant,
    invalid_observation: bool,
}

impl RequestSloState {
    pub fn new(
        ingress: Instant,
        service_class: String,
        budgets: SloLatencyBudgets,
    ) -> Result<Self> {
        budgets.validate().map_err(FerrumError::config)?;
        ingress
            .checked_add(budgets.ttft())
            .ok_or_else(|| FerrumError::config("first-token deadline overflows monotonic time"))?;
        Ok(Self {
            ingress,
            service_class,
            budgets,
            first_commit: None,
            last_commit: None,
            max_token_gap: None,
            committed_tokens: 0,
            violations: SloTimingViolations::default(),
            last_observation: ingress,
            invalid_observation: false,
        })
    }

    pub fn ingress(&self) -> Instant {
        self.ingress
    }
    pub fn service_class(&self) -> &str {
        &self.service_class
    }
    pub fn budgets(&self) -> SloLatencyBudgets {
        self.budgets
    }
    pub fn first_commit(&self) -> Option<Instant> {
        self.first_commit
    }
    pub fn last_commit(&self) -> Option<Instant> {
        self.last_commit
    }
    pub fn committed_tokens(&self) -> u64 {
        self.committed_tokens
    }
    pub fn max_token_gap(&self) -> Option<Duration> {
        self.max_token_gap
    }
    pub fn violations(&self) -> SloTimingViolations {
        self.violations
    }
    /// Arithmetic/clock failures invalidate the state for planning permanently.
    pub fn is_trusted(&self) -> bool {
        !self.invalid_observation
    }

    pub fn first_deadline(&self) -> Instant {
        // Construction checks this immutable pair once.
        self.ingress
            .checked_add(self.budgets.ttft())
            .expect("validated first deadline")
    }

    fn decode_deadlines(&self) -> Result<Option<(Instant, Instant)>> {
        let (Some(first), Some(last)) = (self.first_commit, self.last_commit) else {
            return Ok(None);
        };
        let cumulative_ms = self
            .budgets
            .tpot_ms
            .get()
            .checked_mul(self.committed_tokens)
            .ok_or_else(|| FerrumError::resource_exhausted("cumulative TPOT duration overflow"))?;
        let tpot = first
            .checked_add(Duration::from_millis(cumulative_ms))
            .ok_or_else(|| FerrumError::resource_exhausted("cumulative TPOT deadline overflow"))?;
        let itl = last
            .checked_add(self.budgets.itl())
            .ok_or_else(|| FerrumError::resource_exhausted("ITL deadline overflow"))?;
        Ok(Some((itl, tpot)))
    }

    /// For n already committed tokens, the next token is due by
    /// min(last + ITL, first + n * TPOT). An empty sequence uses the TTFT deadline.
    pub fn next_deadline(&self) -> Result<Instant> {
        if !self.is_trusted() {
            return Err(FerrumError::internal("SLO timing state is untrusted"));
        }
        Ok(match self.decode_deadlines()? {
            Some((itl, tpot)) => itl.min(tpot),
            None => self.first_deadline(),
        })
    }

    /// Earliest still-unrecorded timing violation for a live request. Equality
    /// with a deadline is allowed, so wake one nanosecond after it. Once a bit
    /// is sticky, do not keep waking on its past deadline. The lifecycle owner
    /// must stop asking when no further token is owed.
    pub fn next_violation_wake(&self) -> Result<Option<SloViolationWake>> {
        if !self.is_trusted() {
            return Err(FerrumError::internal("SLO timing state is untrusted"));
        }
        let next = match self.decode_deadlines()? {
            None => (!self.violations.ttft)
                .then_some((self.first_deadline(), SloTimingBoundary::FirstToken)),
            Some((itl, tpot)) => {
                let interval =
                    (!self.violations.itl).then_some((itl, SloTimingBoundary::InterToken));
                let prefix =
                    (!self.violations.tpot).then_some((tpot, SloTimingBoundary::TokenPrefix));
                match (interval, prefix) {
                    (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
                    (left, right) => left.or(right),
                }
            }
        };
        next.map(|(deadline, boundary)| {
            let at = deadline
                .checked_add(Duration::from_nanos(1))
                .ok_or_else(|| {
                    FerrumError::resource_exhausted("SLO violation wake overflows monotonic time")
                })?;
            Ok(SloViolationWake { at, boundary })
        })
        .transpose()
    }

    /// Lifecycle-owner form of the query. A deadline that cannot be represented
    /// invalidates this timing state, rather than silently removing its timer
    /// while leaving apparently trusted planning evidence behind.
    pub fn arm_violation_wake(&mut self) -> Result<Option<SloViolationWake>> {
        let result = self.next_violation_wake();
        self.invalid_observation |= result.is_err();
        result
    }

    /// Call for a still-live obligation, including while resource/output blocked.
    /// Completion/cancellation lifecycle is intentionally outside this state.
    pub fn observe_wait(&mut self, now: Instant) -> Result<()> {
        let result = self.observe_wait_inner(now);
        self.invalid_observation |= result.is_err();
        result
    }

    fn observe_wait_inner(&mut self, now: Instant) -> Result<()> {
        if now < self.last_observation {
            return Err(FerrumError::internal(
                "SLO observation precedes ingress or previous observation",
            ));
        }
        match self.decode_deadlines()? {
            Some((itl, tpot)) => {
                self.violations.itl |= now > itl;
                self.violations.tpot |= now > tpot;
            }
            None => self.violations.ttft |= now > self.first_deadline(),
        }
        self.last_observation = now;
        Ok(())
    }

    pub fn record_commit(&mut self, committed_at: Instant) -> Result<()> {
        let Some(next_count) = self.committed_tokens.checked_add(1) else {
            self.invalid_observation = true;
            return Err(FerrumError::resource_exhausted("SLO token count overflow"));
        };
        self.observe_wait(committed_at)?;
        if let Some(last) = self.last_commit {
            let gap = committed_at.duration_since(last);
            self.max_token_gap = Some(self.max_token_gap.map_or(gap, |previous| previous.max(gap)));
        }
        self.first_commit.get_or_insert(committed_at);
        self.last_commit = Some(committed_at);
        self.committed_tokens = next_count;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
