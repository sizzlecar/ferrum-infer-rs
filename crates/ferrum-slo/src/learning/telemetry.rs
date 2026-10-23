//! Live prediction diagnostics. Each prediction is frozen before its invocation;
//! completed labels never cause a second lookup against a newer model.

use super::snapshot::{Snapshot, Source};
use crate::{
    model::{FeatureDomain, ModelFamily, Phase, ReuseKnowledge, WorkFeatures},
    worker::{sample_work, validate_sample},
};
use ferrum_execution::execution_attempt::*;
use ferrum_scheduler::cost_prediction::CostUnknownReason;
use serde::Serialize;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

const DOMAIN_COUNT: usize = 9;

/// Totals cover actual invocations, not scheduler candidate queries. Error
/// denominators are `base_known` and `calibrated_known`, respectively. All
/// durations/errors are host executor wall seconds, not client-visible latency.
/// Atomic fields are individually sampled; a concurrent health read is not a
/// transaction. Exact totals are available after owner shutdown.
#[derive(Debug, Serialize)]
pub struct LearningPredictionStats {
    pub family: ModelFamily,
    pub calibrated_quantile: f64,
    pub unprepared_completions: u64,
    pub invalid_projection: u64,
    pub domains: Vec<DomainPredictionStats>,
}

#[derive(Debug, Serialize)]
pub struct DomainPredictionStats {
    pub domain: FeatureDomain,
    pub prepared: u64,
    pub completed: u64,
    pub eligible: u64,
    pub participants: u64,
    pub confirmed_prefix_restore_participants: u64,
    pub confirmed_prefix_restore_tokens: u64,
    pub known_internal_reuse_participants: u64,
    pub known_internal_reuse_tokens: u64,
    pub unknown_internal_reuse_participants: u64,
    pub excluded: u64,
    pub mismatched_or_noncausal: u64,
    pub base_known: u64,
    pub calibrated_known: u64,
    pub no_coverage: u64,
    pub stale: u64,
    pub unsupported: u64,
    pub query_limit: u64,
    pub central_signed_error_seconds: f64,
    pub central_absolute_error_seconds: f64,
    pub central_squared_error_seconds: f64,
    pub risk_base_exceedances: u64,
    pub calibrated_exceedances: u64,
    pub calibrated_excess_seconds: f64,
    pub min_model_revision: Option<u64>,
    pub max_model_revision: Option<u64>,
    pub preparation_total_ns: u64,
    pub preparation_max_ns: u64,
    pub feedback_delay_samples: u64,
    pub feedback_delay_total_ns: u64,
    pub feedback_delay_max_ns: u64,
    pub forwarded: u64,
    pub forwarding_dropped: u64,
}

#[derive(Default)]
struct FloatSum(AtomicU64);
impl FloatSum {
    fn add(&self, value: f64) {
        // Labels and model outputs are finite and bounded; saturate eventual
        // lifetime aggregate overflow without emitting non-finite JSON.
        let _ = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                Some(
                    (f64::from_bits(bits) + value)
                        .clamp(-f64::MAX, f64::MAX)
                        .to_bits(),
                )
            });
    }
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

#[derive(Default)]
struct DomainCounters {
    prepared: AtomicU64,
    completed: AtomicU64,
    eligible: AtomicU64,
    participants: AtomicU64,
    confirmed_prefix_restore_participants: AtomicU64,
    confirmed_prefix_restore_tokens: AtomicU64,
    known_internal_reuse_participants: AtomicU64,
    known_internal_reuse_tokens: AtomicU64,
    unknown_internal_reuse_participants: AtomicU64,
    excluded: AtomicU64,
    mismatched_or_noncausal: AtomicU64,
    base_known: AtomicU64,
    calibrated_known: AtomicU64,
    no_coverage: AtomicU64,
    stale: AtomicU64,
    unsupported: AtomicU64,
    query_limit: AtomicU64,
    central_signed_error_seconds: FloatSum,
    central_absolute_error_seconds: FloatSum,
    central_squared_error_seconds: FloatSum,
    risk_base_exceedances: AtomicU64,
    calibrated_exceedances: AtomicU64,
    calibrated_excess_seconds: FloatSum,
    // Zero means no known model; real model revisions start at one.
    min_model_revision: AtomicU64,
    max_model_revision: AtomicU64,
    preparation_total_ns: AtomicU64,
    preparation_max_ns: AtomicU64,
    feedback_delay_samples: AtomicU64,
    feedback_delay_total_ns: AtomicU64,
    feedback_delay_max_ns: AtomicU64,
    forwarded: AtomicU64,
    forwarding_dropped: AtomicU64,
}

pub(super) struct Telemetry {
    family: ModelFamily,
    quantile: f64,
    unprepared: AtomicU64,
    invalid_projection: AtomicU64,
    domains: [DomainCounters; DOMAIN_COUNT],
}

fn index(domain: FeatureDomain) -> usize {
    let phase = match domain.phase {
        Phase::Prefill => 0,
        Phase::Decode => 1,
        Phase::Mixed => 2,
    };
    let reuse = match domain.reuse {
        ReuseKnowledge::Known => 0,
        ReuseKnowledge::Unknown => 1,
        ReuseKnowledge::Partial => 2,
    };
    phase * 3 + reuse
}
fn domain(index: usize) -> FeatureDomain {
    FeatureDomain {
        phase: [Phase::Prefill, Phase::Decode, Phase::Mixed][index / 3],
        reuse: [
            ReuseKnowledge::Known,
            ReuseKnowledge::Unknown,
            ReuseKnowledge::Partial,
        ][index % 3],
    }
}
fn add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
        Some(old.saturating_add(value))
    });
}
fn duration(total: &AtomicU64, max: &AtomicU64, value: Duration) {
    let nanos = value.as_nanos().min(u64::MAX as u128) as u64;
    add(total, nanos);
    max.fetch_max(nanos, Ordering::Relaxed);
}

impl Telemetry {
    pub fn new(family: ModelFamily, quantile: f64) -> Self {
        Self {
            family,
            quantile,
            unprepared: AtomicU64::new(0),
            invalid_projection: AtomicU64::new(0),
            domains: std::array::from_fn(|_| DomainCounters::default()),
        }
    }
    pub fn stats(&self) -> LearningPredictionStats {
        LearningPredictionStats {
            family: self.family,
            calibrated_quantile: self.quantile,
            unprepared_completions: self.unprepared.load(Ordering::Relaxed),
            invalid_projection: self.invalid_projection.load(Ordering::Relaxed),
            domains: self
                .domains
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    macro_rules! read {
                        ($f:ident) => {
                            d.$f.load(Ordering::Relaxed)
                        };
                    }
                    DomainPredictionStats {
                        domain: domain(i),
                        prepared: read!(prepared),
                        completed: read!(completed),
                        eligible: read!(eligible),
                        participants: read!(participants),
                        confirmed_prefix_restore_participants: read!(
                            confirmed_prefix_restore_participants
                        ),
                        confirmed_prefix_restore_tokens: read!(confirmed_prefix_restore_tokens),
                        known_internal_reuse_participants: read!(known_internal_reuse_participants),
                        known_internal_reuse_tokens: read!(known_internal_reuse_tokens),
                        unknown_internal_reuse_participants: read!(
                            unknown_internal_reuse_participants
                        ),
                        excluded: read!(excluded),
                        mismatched_or_noncausal: read!(mismatched_or_noncausal),
                        base_known: read!(base_known),
                        calibrated_known: read!(calibrated_known),
                        no_coverage: read!(no_coverage),
                        stale: read!(stale),
                        unsupported: read!(unsupported),
                        query_limit: read!(query_limit),
                        central_signed_error_seconds: d.central_signed_error_seconds.get(),
                        central_absolute_error_seconds: d.central_absolute_error_seconds.get(),
                        central_squared_error_seconds: d.central_squared_error_seconds.get(),
                        risk_base_exceedances: read!(risk_base_exceedances),
                        calibrated_exceedances: read!(calibrated_exceedances),
                        calibrated_excess_seconds: d.calibrated_excess_seconds.get(),
                        min_model_revision: (read!(min_model_revision) != 0)
                            .then(|| read!(min_model_revision)),
                        max_model_revision: (read!(max_model_revision) != 0)
                            .then(|| read!(max_model_revision)),
                        preparation_total_ns: read!(preparation_total_ns),
                        preparation_max_ns: read!(preparation_max_ns),
                        feedback_delay_samples: read!(feedback_delay_samples),
                        feedback_delay_total_ns: read!(feedback_delay_total_ns),
                        feedback_delay_max_ns: read!(feedback_delay_max_ns),
                        forwarded: read!(forwarded),
                        forwarding_dropped: read!(forwarding_dropped),
                    }
                })
                .collect(),
        }
    }
    pub fn feedback(&self, features: WorkFeatures, completed_at: Instant, now: Instant) {
        if let Some(delay) = now.checked_duration_since(completed_at) {
            let d = &self.domains[index(features.domain())];
            add(&d.feedback_delay_samples, 1);
            duration(&d.feedback_delay_total_ns, &d.feedback_delay_max_ns, delay);
        }
    }
}

#[derive(Clone, Copy)]
struct Prediction {
    revision: u64,
    central: f64,
    risk_base: f64,
    upper: Option<f64>,
}

/// One fixed-size private ticket. No old snapshot or model is retained.
struct Completion {
    target: Arc<dyn ExecutionAttemptObserver>,
    telemetry: Arc<Telemetry>,
    features: WorkFeatures,
    frozen_at: Instant,
    prediction: Option<Prediction>,
    unknown: Option<CostUnknownReason>,
}

fn freeze(
    snapshot: &Snapshot,
    features: WorkFeatures,
    now: Instant,
) -> (Option<Prediction>, Option<CostUnknownReason>) {
    let Some(domain) = snapshot
        .domains
        .iter()
        .find(|d| d.model.domain() == features.domain())
    else {
        return (None, Some(CostUnknownReason::NoCoverage));
    };
    if now >= domain.model_expires_at {
        return (None, Some(CostUnknownReason::Stale));
    }
    let Ok(base) = domain.model.predict(&snapshot.identity, &features) else {
        return (None, Some(CostUnknownReason::UnsupportedDomain));
    };
    let Ok(central) = Duration::try_from_secs_f64(base.central_seconds) else {
        return (None, Some(CostUnknownReason::UnsupportedDomain));
    };
    let bucket = snapshot
        .boundaries
        .partition_point(|limit| (*limit as f64) < features.work_tokens()) as u32;
    let correction =
        domain
            .calibration
            .correction(domain.model_revision, &features.domain(), bucket, now);
    let (upper, unknown) = match correction {
        None => (None, Some(CostUnknownReason::Stale)),
        Some(correction) => match Duration::try_from_secs_f64(
            (base.risk_base_seconds + correction.signed_seconds).max(0.0),
        ) {
            Ok(upper) if !central.is_zero() && !upper.is_zero() => {
                (Some(upper.as_secs_f64()), None)
            }
            Ok(_) => (None, Some(CostUnknownReason::UnsupportedDomain)),
            Err(_) => (None, Some(CostUnknownReason::UnsupportedDomain)),
        },
    };
    (
        Some(Prediction {
            revision: domain.model_revision,
            central: central.as_secs_f64(),
            risk_base: base.risk_base_seconds,
            upper,
        }),
        unknown,
    )
}

pub(super) struct Observer {
    pub target: Arc<dyn ExecutionAttemptObserver>,
    pub source: Arc<Source>,
    pub telemetry: Arc<Telemetry>,
}

impl ExecutionAttemptObserver for Observer {
    fn limits(&self) -> ExecutionAttemptLimits {
        self.target.limits()
    }
    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        self.target.record_drop(reason);
    }
    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        // Direct/manual producers without prepare remain valid learning data,
        // but cannot manufacture pre-execution prediction-error observations.
        add(&self.telemetry.unprepared, 1);
        self.target.try_record(summary)
    }
    fn prepare(
        &self,
        participants: &[ExecutionAttemptParticipant],
    ) -> Option<Box<dyn ExecutionAttemptCompletion>> {
        let started = Instant::now();
        let limits = self.target.limits();
        if !limits.accepts(participants.len()) {
            return None;
        }
        let Ok(features) = WorkFeatures::extract_iter(
            participants.iter().map(sample_work),
            limits.max_participants,
        ) else {
            add(&self.telemetry.invalid_projection, 1);
            return None;
        };
        let snapshot = self.source.load_typed();
        let (prediction, unknown) = if ferrum_scheduler::cost_prediction::CostQuery::fits(
            participants.len(),
            snapshot.query,
        ) {
            freeze(&snapshot, features, started)
        } else {
            (None, Some(CostUnknownReason::QueryLimit))
        };
        let d = &self.telemetry.domains[index(features.domain())];
        add(&d.prepared, 1);
        let completion = Box::new(Completion {
            target: self.target.clone(),
            telemetry: self.telemetry.clone(),
            features,
            frozen_at: started,
            prediction,
            unknown,
        });
        drop(snapshot);
        duration(
            &d.preparation_total_ns,
            &d.preparation_max_ns,
            started.elapsed(),
        );
        Some(completion)
    }
}

impl ExecutionAttemptCompletion for Completion {
    fn try_record(self: Box<Self>, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        self.record_at(summary, Instant::now())
    }
}

impl Completion {
    fn record_at(
        self,
        summary: &ExecutionAttemptSummary,
        now: Instant,
    ) -> ExecutionAttemptDelivery {
        let d = &self.telemetry.domains[index(self.features.domain())];
        add(&d.completed, 1);
        let features = WorkFeatures::extract_iter(
            summary.participants.iter().map(sample_work),
            self.target.limits().max_participants,
        );
        if summary.started_at < self.frozen_at || features.ok() != Some(self.features) {
            add(&d.mismatched_or_noncausal, 1);
        } else if let Ok(wall) = validate_sample(summary, now) {
            if wall.as_secs_f64() <= 86400.0 {
                add(&d.eligible, 1);
                for participant in &summary.participants {
                    add(&d.participants, 1);
                    if let AttemptPrefixRestore::Restored { tokens } = participant.prefix_restore {
                        add(&d.confirmed_prefix_restore_participants, 1);
                        add(&d.confirmed_prefix_restore_tokens, tokens as u64);
                    }
                    match participant.internally_reused_tokens {
                        Some(tokens) => {
                            add(&d.known_internal_reuse_participants, 1);
                            add(&d.known_internal_reuse_tokens, tokens as u64);
                        }
                        None => add(&d.unknown_internal_reuse_participants, 1),
                    }
                }
                let actual = wall.as_secs_f64();
                if let Some(prediction) = self.prediction {
                    add(&d.base_known, 1);
                    let error = actual - prediction.central;
                    d.central_signed_error_seconds.add(error);
                    d.central_absolute_error_seconds.add(error.abs());
                    d.central_squared_error_seconds.add(error * error);
                    add(
                        &d.risk_base_exceedances,
                        u64::from(actual > prediction.risk_base),
                    );
                    let _ = d.min_model_revision.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |old| {
                            Some(if old == 0 {
                                prediction.revision
                            } else {
                                old.min(prediction.revision)
                            })
                        },
                    );
                    d.max_model_revision
                        .fetch_max(prediction.revision, Ordering::Relaxed);
                    if let Some(upper) = prediction.upper {
                        add(&d.calibrated_known, 1);
                        add(&d.calibrated_exceedances, u64::from(actual > upper));
                        d.calibrated_excess_seconds.add((actual - upper).max(0.0));
                    }
                }
                if let Some(reason) = self.unknown {
                    add(
                        match reason {
                            CostUnknownReason::NoCoverage => &d.no_coverage,
                            CostUnknownReason::Stale => &d.stale,
                            CostUnknownReason::QueryLimit => &d.query_limit,
                            _ => &d.unsupported,
                        },
                        1,
                    );
                }
            } else {
                add(&d.excluded, 1);
            }
        } else {
            add(&d.excluded, 1);
        }
        let result = self.target.try_record(summary);
        add(
            if result == ExecutionAttemptDelivery::Recorded {
                &d.forwarded
            } else {
                &d.forwarding_dropped
            },
            1,
        );
        result
    }
}

#[cfg(test)]
mod tests;
