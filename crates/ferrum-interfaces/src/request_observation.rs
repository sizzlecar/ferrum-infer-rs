//! Cumulative facts from the product request's verified stopping boundary.
//!
//! An execution incarnation or a public RequestId is not a request lifetime.
//! Producers assign admission numbers once, before resource admission, and keep
//! them through preemption and KV reconstruction. A missing terminal notice
//! leaves an unknown suffix; it never establishes that a request is still live.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestLifecycleId {
    pub producer: u64,
    pub admission: u64,
}

/// Only the product stopping owner may assert `CapOnly`: model EOS, user stops,
/// grammar completion and response boundaries must all be absent as terminators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoppingMechanism {
    Observed,
    CapOnly,
}

/// Complete canonical identity, not a hash. Model, tokenizer and execution
/// configuration are fixed by the instance. Cap is carried separately; shorter
/// caps censor the natural-stop table and never become natural-stop evidence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EffectiveStoppingRules {
    identity: Box<[u8]>,
    mechanism: StoppingMechanism,
}

impl EffectiveStoppingRules {
    pub fn new(identity: Vec<u8>, mechanism: StoppingMechanism, max_bytes: usize) -> Option<Self> {
        (!identity.is_empty() && identity.len() <= max_bytes).then(|| Self {
            identity: identity.into_boxed_slice(),
            mechanism,
        })
    }

    pub fn identity(&self) -> &[u8] {
        &self.identity
    }

    pub fn mechanism(&self) -> StoppingMechanism {
        self.mechanism
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestTerminalReason {
    Eos,
    Stop,
    Length,
    Cancelled,
    Error,
}

/// Engine product completion after semantic validation, execution closure and
/// delivery into all requested engine output channels. Client receipt and HTTP
/// protocol validity are later boundaries and are not asserted here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestCompletionOutcome {
    Succeeded,
    Failed,
}

/// A cumulative snapshot, so later complete facts repair intermediate loss.
/// `verified_survival_through = g` means the natural stopping rules were checked
/// after output g and did not fire. The separately declared cap is excluded:
/// Length at g proves natural survival through g because cap is checked last.
/// EOS/Stop at g proves survival only through g - 1.
/// Generated count includes engine-owned terminal tokens, as usage does; it
/// does not count SSE messages and cannot establish client latency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestStoppingObservation {
    pub lifecycle: RequestLifecycleId,
    /// Original ingress clock, immutable across every cumulative notice.
    pub admitted_at: Instant,
    pub revision: u64,
    /// Producer monotonic clock, captured when this cumulative fact was checked.
    pub observed_at: Instant,
    pub rules: Arc<EffectiveStoppingRules>,
    pub cap: usize,
    pub generated_tokens: usize,
    pub verified_survival_through: usize,
    pub terminal: Option<RequestTerminalReason>,
    pub completion: Option<RequestCompletionOutcome>,
}

impl RequestStoppingObservation {
    pub fn is_consistent(&self) -> bool {
        self.lifecycle.producer != 0
            && self.lifecycle.admission != 0
            && self.revision != 0
            && self.cap != 0
            && self.admitted_at <= self.observed_at
            && self.generated_tokens <= self.cap
            && self.verified_survival_through <= self.generated_tokens
            && match self.completion {
                None => true,
                Some(RequestCompletionOutcome::Succeeded) => matches!(
                    self.terminal,
                    Some(
                        RequestTerminalReason::Eos
                            | RequestTerminalReason::Stop
                            | RequestTerminalReason::Length
                    )
                ),
                Some(RequestCompletionOutcome::Failed) => self.terminal.is_some(),
            }
            && match self.terminal {
                Some(RequestTerminalReason::Eos | RequestTerminalReason::Stop) => {
                    self.generated_tokens > 0
                        && self.verified_survival_through == self.generated_tokens - 1
                        && self.rules.mechanism() == StoppingMechanism::Observed
                }
                Some(RequestTerminalReason::Length) => {
                    self.generated_tokens == self.cap && self.verified_survival_through == self.cap
                }
                _ => true,
            }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestObservationLimits {
    pub max_rule_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestObservationDrop {
    ProjectionLimit,
    QueueFull,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestObservationDelivery {
    Recorded,
    Dropped(RequestObservationDrop),
}

/// Must neither block nor train. Losing a fact must never retry inference.
pub trait RequestStoppingObserver: Send + Sync {
    fn limits(&self) -> RequestObservationLimits;
    /// Bounded, nonblocking admission watermark, published before any lossy
    /// rule projection or queue send. Missing cumulative notices remain unseen
    /// lifetimes rather than disappearing from the observation denominator.
    /// Preserve the original ingress time for this exact lifecycle; serial
    /// order does not imply time order when ingress publication is concurrent.
    fn record_admission(&self, _lifecycle: RequestLifecycleId, _admitted_at: Instant) {}
    fn try_record(&self, observation: &RequestStoppingObservation) -> RequestObservationDelivery;
    fn record_drop(&self, reason: RequestObservationDrop);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoppingMass {
    pub generated_tokens: usize,
    pub mass: f64,
}

/// Conditional on the verified prefix and successful future execution.
/// Informative cancellation/censoring remains a statistical assumption, not a
/// guarantee. Cost/stop dependence must be declared by the consuming policy.
#[derive(Debug, Clone, PartialEq)]
pub struct StoppingForecast {
    pub stops: Vec<StoppingMass>,
    pub known_survival_mass: f64,
    pub unresolved_mass: f64,
    /// Subset of resolved stops for which TPOT has no denominator (length < 2).
    /// This is not extra mass and must not be included in a TPOT pass numerator.
    pub tpot_undefined_mass: f64,
}

impl StoppingForecast {
    pub fn unknown() -> Self {
        Self {
            stops: Vec::new(),
            known_survival_mass: 0.0,
            unresolved_mass: 1.0,
            tpot_undefined_mass: 0.0,
        }
    }

    pub fn resolved_mass(&self) -> f64 {
        self.stops.iter().map(|stop| stop.mass).sum()
    }

    /// Without a justified dependency assumption, unresolved mass contributes
    /// anywhere in [0, U] to the probability of exceeding a latency bound.
    pub fn exceedance_interval(&self, resolved_exceedance: f64) -> Option<(f64, f64)> {
        (resolved_exceedance.is_finite()
            && resolved_exceedance >= 0.0
            && resolved_exceedance <= self.resolved_mass() + f64::EPSILON)
            .then(|| {
                (
                    resolved_exceedance,
                    (resolved_exceedance + self.unresolved_mass + self.known_survival_mass)
                        .min(1.0),
                )
            })
    }
}

pub struct StoppingQuery<'a> {
    pub rules: &'a EffectiveStoppingRules,
    pub verified_survival_through: usize,
    pub cap: usize,
    /// Inclusive prediction frontier; outstanding survival remains explicit.
    pub horizon: usize,
    pub now: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionApproximation {
    /// Same-instance, full stopping-domain and exact-cap empirical arrival
    /// proportion in a fixed minimum-age cohort. Recent successes and pending
    /// requests are excluded equally; every matured missing outcome is unknown.
    /// Applying this to current arrivals assumes cohort exchangeability, not
    /// finite-sample, conditional-on-progress or joint coverage certainty.
    EmpiricalMaturedCohort,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompletionForecast {
    pub succeeded_mass: f64,
    pub failed_mass: f64,
    pub unresolved_mass: f64,
    pub samples: usize,
    pub approximation: Option<CompletionApproximation>,
}

impl CompletionForecast {
    pub fn unknown() -> Self {
        Self {
            succeeded_mass: 0.0,
            failed_mass: 0.0,
            unresolved_mass: 1.0,
            samples: 0,
            approximation: None,
        }
    }
}

pub struct CompletionQuery<'a> {
    pub rules: &'a EffectiveStoppingRules,
    pub cap: usize,
    /// Positive minimum admission age, not a success deadline. Must be less
    /// than the observation TTL to provide useful completed support.
    pub maturity: Duration,
    pub now: Instant,
}

pub trait StoppingSnapshot: Send + Sync {
    fn forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast;
    fn completion_forecast(&self, _query: &CompletionQuery<'_>) -> CompletionForecast {
        CompletionForecast::unknown()
    }
}

pub trait StoppingSnapshotSource: Send + Sync {
    fn snapshot(&self) -> Arc<dyn StoppingSnapshot>;
}
