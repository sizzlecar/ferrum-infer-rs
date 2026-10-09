//! Constant-space attribution of existing host intervals to dispatch outcomes.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Disposition {
    Hit,
    NoResidentProgram,
    NoCachedRecipe,
    CapabilityUnavailable,
    UnsupportedEncoder,
    Other,
    BeforeDecision,
    Retried,
}

impl Disposition {
    const ALL: [Self; 8] = [
        Self::Hit,
        Self::NoResidentProgram,
        Self::NoCachedRecipe,
        Self::CapabilityUnavailable,
        Self::UnsupportedEncoder,
        Self::Other,
        Self::BeforeDecision,
        Self::Retried,
    ];

    const fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::NoResidentProgram => "no_resident_program",
            Self::NoCachedRecipe => "no_cached_recipe",
            Self::CapabilityUnavailable => "immutable_capability_unavailable",
            Self::UnsupportedEncoder => "unsupported_encoder",
            Self::Other => "other",
            Self::BeforeDecision => "before_decision",
            Self::Retried => "retried",
        }
    }

    fn from_stats(stats: Option<InvocationPreparationStats>, outcome: Outcome) -> Self {
        let Some(stats) = stats else {
            return Self::BeforeDecision;
        };
        let causes = [
            (stats.segment_hits, Self::Hit),
            (stats.segment_no_resident_program, Self::NoResidentProgram),
            (stats.segment_no_cached_recipe, Self::NoCachedRecipe),
            (
                stats.segment_immutable_capability_unavailable,
                Self::CapabilityUnavailable,
            ),
            (stats.segment_unsupported_encoder, Self::UnsupportedEncoder),
        ];
        let mut present = causes.into_iter().filter(|(count, _)| *count != 0);
        match (present.next(), present.next()) {
            (Some((1, disposition)), None) => disposition,
            (None, None) if stats.segment_misses == 0 && outcome == Outcome::Error => {
                Self::BeforeDecision
            }
            _ => Self::Other,
        }
    }
}

const DECISION_SEEN: u8 = 1 << 4;
const UNDECIDED: u8 = 1 << 5;
const INCOMPLETE: u8 = 1 << 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Success,
    Error,
}

#[derive(Default)]
struct OutcomeBucket {
    observations: AtomicU64,
    incomplete_declarations: AtomicU64,
    elapsed: AtomicDurationMetrics,
}

impl OutcomeBucket {
    fn record(&self, incomplete: bool, elapsed: Option<Duration>) {
        self.observations.fetch_add(1, Ordering::Relaxed);
        self.incomplete_declarations
            .fetch_add(u64::from(incomplete), Ordering::Relaxed);
        if let Some(elapsed) = elapsed {
            self.elapsed.record(elapsed);
        }
    }

    fn reset(&self) {
        self.observations.store(0, Ordering::Relaxed);
        self.incomplete_declarations.store(0, Ordering::Relaxed);
        self.elapsed.reset();
    }

    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "observations": self.observations.load(Ordering::Relaxed),
            "incomplete_declarations": self.incomplete_declarations.load(Ordering::Relaxed),
            "elapsed": self.elapsed.snapshot(),
        })
    }
}

#[derive(Default)]
struct OutcomeTable([[OutcomeBucket; 2]; 8]);

impl OutcomeTable {
    fn record(
        &self,
        disposition: Disposition,
        outcome: Outcome,
        incomplete: bool,
        elapsed: Option<Duration>,
    ) {
        self.0[disposition as usize][outcome as usize].record(incomplete, elapsed);
    }

    fn reset(&self) {
        for row in &self.0 {
            for bucket in row {
                bucket.reset();
            }
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        Disposition::ALL
            .into_iter()
            .map(|disposition| {
                let row = &self.0[disposition as usize];
                (
                    disposition.as_str().to_owned(),
                    serde_json::json!({
                        "success": row[Outcome::Success as usize].snapshot(),
                        "error": row[Outcome::Error as usize].snapshot(),
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into()
    }
}

#[derive(Default)]
pub(super) struct OutcomeMetrics {
    provider_attempts: OutcomeTable,
    whole_host: OutcomeTable,
}

impl OutcomeMetrics {
    pub(super) fn reset(&self) {
        self.provider_attempts.reset();
        self.whole_host.reset();
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "collection": "host_timing_enabled_only",
            "provider_attempts": self.provider_attempts.snapshot(),
            "whole_host": self.whole_host.snapshot(),
            "limitations": [
                "success means dispatch returned Submitted, not terminal or GPU success; error includes definitely-not-submitted and possibly-submitted failures",
                "provider_attempts observes each returned core dispatch; elapsed is present only if ProviderNodeEncode was entered, including partial ordinary errors",
                "whole_host reuses host_encode_submit including token-mask preparation and all dispatch retries, excluding completion and postprocess",
                "any attempted retry puts the whole host interval in retried, including failure before the next core dispatch; before_decision has no classified segment outcome",
                "incomplete_declarations is an overlapping count-only flag, never a second elapsed attribution; multiple primary causes and successful non-segment paths are other",
                "provider_attempts and whole_host overlap; phase groups use the existing prefill/decode/mixed classification; snapshots are non-atomic",
                "no_cached_recipe retains the existing combined cache, owner, entry, epoch and availability meaning; no new cache-cause inference is made",
                "unwinding does not finalize these observations; elapsed is host wall time, not pure CPU or device time"
            ],
        })
    }
}

/// No clocks or allocations: the underlying dispatch timers own elapsed time.
pub(super) struct AttemptObservation<'a, T, P> {
    timing: &'a T,
    preparation: &'a P,
    // The sink trait is Send + Sync. These atomics are private to this attempt,
    // not snapshots of concurrently accumulated executor counters.
    decision: AtomicU8,
    provider_ns: AtomicU64,
    provider_seen: AtomicBool,
}

impl<'a, T: DeviceSubmissionTimingSink, P> AttemptObservation<'a, T, P> {
    pub(super) fn new(timing: &'a T, preparation: &'a P) -> Option<Self> {
        T::ENABLED.then(|| Self {
            timing,
            preparation,
            decision: AtomicU8::new(0),
            provider_ns: AtomicU64::new(0),
            provider_seen: AtomicBool::new(false),
        })
    }

    pub(super) fn finish(self, outcome: Outcome) -> AttemptSummary {
        let decision = self.decision.load(Ordering::Relaxed);
        let disposition = if decision & DECISION_SEEN == 0
            || (decision & UNDECIDED != 0 && outcome == Outcome::Error)
        {
            Disposition::BeforeDecision
        } else {
            Disposition::ALL[usize::from(decision & 0x0f)]
        };
        AttemptSummary {
            disposition,
            outcome,
            incomplete_declarations: decision & INCOMPLETE != 0,
            provider_elapsed: self
                .provider_seen
                .load(Ordering::Acquire)
                .then(|| Duration::from_nanos(self.provider_ns.load(Ordering::Relaxed))),
        }
    }
}

impl<T, P: InvocationPreparationSink> InvocationPreparationSink for AttemptObservation<'_, T, P> {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.preparation.record_preparation(stats);
        let mut decision =
            DECISION_SEEN | Disposition::from_stats(Some(stats), Outcome::Success) as u8;
        if Disposition::from_stats(Some(stats), Outcome::Error) == Disposition::BeforeDecision {
            decision |= UNDECIDED;
        }
        if stats.segment_incomplete_declarations != 0 {
            decision |= INCOMPLETE;
        }
        self.decision.store(decision, Ordering::Relaxed);
    }
}

impl<T: DeviceSubmissionTimingSink, P: Sync> DeviceSubmissionTimingSink
    for AttemptObservation<'_, T, P>
{
    const ENABLED: bool = T::ENABLED;

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration) {
        self.timing.record_device_submission(stage, elapsed);
    }

    fn record_reusable_execution(&self, observation: DeviceReusableExecutionObservation) {
        self.timing.record_reusable_execution(observation);
    }
}

impl<T: SubmissionWaveDispatchTimingSink, P: Sync> SubmissionWaveDispatchTimingSink
    for AttemptObservation<'_, T, P>
{
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration) {
        self.timing.record(stage, elapsed);
        if stage == SubmissionWaveDispatchStage::ProviderNodeEncode {
            self.provider_ns.store(
                elapsed.as_nanos().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
            self.provider_seen.store(true, Ordering::Release);
        }
    }
}

pub(super) struct AttemptSummary {
    disposition: Disposition,
    outcome: Outcome,
    incomplete_declarations: bool,
    provider_elapsed: Option<Duration>,
}

#[derive(Default)]
pub(super) struct WaveObservation {
    attempts: u32,
    retried: bool,
    last_disposition: Option<Disposition>,
    incomplete_declarations: bool,
}

impl WaveObservation {
    pub(super) fn record_attempt(
        &mut self,
        summary: AttemptSummary,
        aggregate: &OutcomeMetrics,
        phase: &OutcomeMetrics,
    ) {
        self.attempts = self.attempts.saturating_add(1);
        self.last_disposition = Some(summary.disposition);
        self.incomplete_declarations |= summary.incomplete_declarations;
        for metrics in [aggregate, phase] {
            metrics.provider_attempts.record(
                summary.disposition,
                summary.outcome,
                summary.incomplete_declarations,
                summary.provider_elapsed,
            );
        }
    }

    pub(super) fn mark_retry(&mut self) {
        self.retried = true;
    }

    pub(super) fn finish(
        self,
        outcome: Outcome,
        elapsed: Duration,
        aggregate: &OutcomeMetrics,
        phase: &OutcomeMetrics,
    ) {
        let disposition = if self.retried || self.attempts > 1 {
            Disposition::Retried
        } else {
            self.last_disposition.unwrap_or(Disposition::BeforeDecision)
        };
        for metrics in [aggregate, phase] {
            metrics.whole_host.record(
                disposition,
                outcome,
                self.incomplete_declarations,
                Some(elapsed),
            );
        }
    }
}

#[cfg(test)]
mod tests;
