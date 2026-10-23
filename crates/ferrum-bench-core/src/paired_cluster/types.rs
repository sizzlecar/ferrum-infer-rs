//! Typed paired-run observations and explicit analysis limits.
use std::time::Duration;

pub type Digest = [u8; 32];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestKey {
    pub index: usize,
    pub episode: u32,
    pub source: Digest,
    pub prompt: Digest,
    pub scheduled_ms: u64,
    pub output_cap: u64,
}
#[derive(Clone, Debug)]
pub struct Workload {
    /// Hash of declared backend/model/precision/schedule/initialization policy.
    /// Candidate executable identity is bound separately, not equalized here.
    pub cell: Digest,
    pub inputs: Digest,
    pub repeat_ids: Vec<u64>,
    pub requests: Vec<RequestKey>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    Rejected,
    Timeout,
    Cancelled,
    Transport,
    InvalidOutput,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failed(FailureKind),
    /// Expected arrival retained, but its terminal record is unavailable.
    Missing,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub prompt: u64,
    pub completion: u64,
    pub total: u64,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub key: RequestKey,
    pub outcome: Outcome,
    pub usage: Option<Usage>,
    pub finish: Option<Finish>,
    pub done: bool,
    /// All timestamps share the client run's original scheduled-time origin.
    pub terminal_ms: Option<f64>,
    pub ttft_from_scheduled_ms: Option<f64>,
    /// Last nonempty visible text minus first; never the terminal timestamp.
    pub visible_span_ms: Option<f64>,
    /// Independently recorded total visible-event count for this stream. None
    /// means unknown; never derive a complete count from a partial gap vector.
    pub visible_text_events: Option<usize>,
    /// Individually verified, distinct successive visible-event gaps, including
    /// zero. Missing/invalid entries do not invalidate other independently
    /// verified entries. The reader must establish this provenance, not merely
    /// treat every finite number in an untrusted record as a valid gap.
    pub visible_gaps_ms: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct Run {
    pub repeat_id: u64,
    pub cell: Digest,
    pub inputs: Digest,
    /// Hash of original server PID/start incarnation receipt, not PID alone.
    pub server_incarnation: Digest,
    pub client_exit: Option<i32>,
    pub requests: Vec<Request>,
}
#[derive(Clone, Debug)]
pub struct RunPair {
    pub baseline: Run,
    pub candidate: Run,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fraction {
    pub numerator: u32,
    pub denominator: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct AbsoluteLimits {
    pub quantile: Fraction,
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub visible_itl_ms: f64,
    pub request_deadline: Duration,
}
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    pub feasible_min_throughput_ratio: f64,
    pub feasible_max_latency_ratio: f64,
    pub rescue_min_throughput_ratio: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateKind {
    Controller,
    Observe,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prng {
    SplitMix64V1,
}
#[derive(Clone, Copy, Debug)]
pub struct Bootstrap {
    pub prng: Prng,
    pub seed: u64,
    pub replicates: usize,
    /// Bounds even rejection-sampling PRNG draws; no probabilistic loop bound.
    pub max_rng_draws: usize,
    pub confidence: Fraction,
    /// Optional paired run/episode sensitivity only. Never an acceptance gate.
    pub episode_sensitivity: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub max_runs: usize,
    pub max_requests: usize,
    pub max_events: usize,
    pub max_replicates: usize,
    pub max_sample_visits: usize,
    /// Logical temporary Vec capacities; excludes caller-owned input, allocator
    /// overhead, returned report, and OS/RSS. Not a hard process memory promise.
    pub max_scratch_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub absolute: AbsoluteLimits,
    pub tolerances: Tolerances,
    pub kind: CandidateKind,
    pub bootstrap: Bootstrap,
    pub budget: Budget,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Issue {
    InvalidSpec,
    Budget,
    Overflow,
    InsufficientRunClusters,
    PairMismatch,
    WorkloadMismatch,
    DuplicateServerIncarnation,
    MissingTerminal,
    MissingClientExit,
    FailedRequest,
    FailedClient,
    InvalidUsage,
    InvalidTiming,
    UndefinedMetric,
    NonpositiveRatio,
    MixedStaticFeasibility,
    UncertainStaticFeasibility,
    IntervalCrossesTolerance,
    Numerical,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Pass,
    Fail,
    Unresolved(Issue),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Percentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}
#[derive(Clone, Debug)]
pub struct RunSummary {
    pub repeat_id: u64,
    pub client_exit: Option<i32>,
    pub arrivals: usize,
    pub successes: usize,
    pub failures: usize,
    pub missing: usize,
    pub protocol_violations: usize,
    pub deadline_violations: usize,
    pub tpot_undefined: usize,
    pub tpot_unknown: usize,
    pub visible_intervals: usize,
    /// Invalid/missing event evidence; observed valid gaps remain descriptive.
    pub itl_unknown_requests: usize,
    pub successful_usage_tokens: Option<u64>,
    pub finite_drain_seconds: Option<f64>,
    pub throughput_tps: Option<f64>,
    /// Descriptive distributions of independently observed valid samples.
    /// Only target_quantiles represents complete metric populations.
    pub ttft: Option<Percentiles>,
    pub tpot: Option<Percentiles>,
    pub visible_itl: Option<Percentiles>,
    /// TTFT, TPOT, visible ITL; None when the population/evidence is incomplete.
    pub target_quantiles: [Option<f64>; 3],
    /// A partial metric can prove Fail using a known population upper bound,
    /// but can never establish Pass. No missing values are imputed.
    pub latency_gates: [Gate; 3],
    pub absolute: Gate,
    pub inference_issue: Option<Issue>,
}
#[derive(Clone, Debug)]
pub struct PairSummary {
    pub baseline: RunSummary,
    pub candidate: RunSummary,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticClass {
    AllFeasible,
    AllInfeasible,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Ratios {
    pub throughput: Interval,
    pub ttft: Interval,
    pub tpot: Interval,
    pub visible_itl: Interval,
}
#[derive(Clone, Debug)]
pub struct Primary {
    pub static_class: StaticClass,
    pub ratios: Ratios,
    pub tolerance_gate: Gate,
}
#[derive(Clone, Debug)]
pub struct Sensitivity {
    /// Run-level throughput deliberately has no episode-resampled counterpart.
    pub latency_ratios: [Interval; 3],
    pub interpretation: &'static str,
}
#[derive(Clone, Debug)]
pub struct Report {
    pub pairs: Vec<PairSummary>,
    pub candidate_absolute: Gate,
    pub primary: Result<Primary, Issue>,
    pub episode_sensitivity: Option<Result<Sensitivity, Issue>>,
    /// Observed candidate Fail dominates any unavailable paired inference.
    pub development_gate: Gate,
    pub method_status: &'static str,
    pub interpretation: &'static str,
    pub bootstrap: Bootstrap,
}
