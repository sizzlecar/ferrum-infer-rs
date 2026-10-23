use crate::{
    calibration::{CalibrationSnapshot, Correction},
    model::{
        BasePrediction, ExecutionIdentity, FeatureDomain, FittedModel, PredictionRejection,
        WorkFeatures,
    },
};
use arc_swap::ArcSwap;
use ferrum_execution::execution_attempt::AttemptWorkKind;
use ferrum_scheduler::cost_prediction::*;
use ferrum_types::CostQueryLimits;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::watch;

pub(super) struct Domain {
    pub model: Arc<FittedModel>,
    pub model_revision: u64,
    pub model_expires_at: Instant,
    pub calibration: CalibrationSnapshot<FeatureDomain>,
}

pub(super) struct Snapshot {
    pub revision: u64,
    pub query: CostQueryLimits,
    pub identity: Arc<ExecutionIdentity>,
    pub boundaries: Arc<[usize]>,
    pub quantile: f64,
    pub domains: Vec<Domain>,
    // Publication attaches the source's one shared slot; training/restore
    // snapshots do not create or persist another recorder.
    pub rejections: Option<Arc<super::rejection::Recorder>>,
}

struct PreparedPrediction<'a> {
    domain: &'a Domain,
    base: BasePrediction,
    bucket: u32,
    correction: Correction,
    central_service_time: Duration,
    upper_service_time: Duration,
}

fn unsupported(detail: CostUnsupportedDetail) -> CostUnknownReason {
    CostUnknownReason::UnsupportedWithDetail(detail)
}

fn model_rejection(reason: PredictionRejection) -> CostUnknownReason {
    unsupported(match reason {
        PredictionRejection::ExecutionIdentityMismatch => {
            CostUnsupportedDetail::ExecutionIdentityMismatch
        }
        PredictionRejection::FeatureDomainMismatch => CostUnsupportedDetail::FeatureDomainMismatch,
        PredictionRejection::OutsideObservedSupport => {
            CostUnsupportedDetail::OutsideObservedSupport
        }
        PredictionRejection::InvalidWorkFeatures(_) => CostUnsupportedDetail::InvalidWorkProjection,
        PredictionRejection::InvalidBasePrediction(_) => {
            CostUnsupportedDetail::InvalidBasePrediction
        }
    })
}

impl Snapshot {
    /// One immutable query preparation shared by scalar and scenario results.
    /// Diagnostics describe the rejecting branch; they never run a second fit
    /// or prediction, refresh evidence, or change any support/expiry rule.
    fn prepare(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
    ) -> Result<PreparedPrediction<'_>, CostUnknownReason> {
        if !CostQuery::fits(query.work.len(), self.query) {
            return Err(CostUnknownReason::QueryLimit);
        }
        if !query.work_is_exact {
            return Err(CostUnknownReason::InexactWork);
        }
        if query.execution == CostExecution::SequentialPrefillThenDecode
            && query
                .work
                .iter()
                .any(|w| w.kind == AttemptWorkKind::Prefill)
            && query.work.iter().any(|w| w.kind == AttemptWorkKind::Decode)
        {
            // No inferred joint quantile from pure-phase or mixed-invocation fits.
            return Err(unsupported(CostUnsupportedDetail::ExecutionComposition));
        }
        let features = WorkFeatures::extract(query.work, self.query.max_participants)
            .map_err(|_| unsupported(CostUnsupportedDetail::InvalidWorkProjection))?;
        let Some(domain) = self
            .domains
            .iter()
            .find(|d| d.model.domain() == features.domain())
        else {
            return Err(CostUnknownReason::NoCoverage);
        };
        if now >= domain.model_expires_at {
            return Err(CostUnknownReason::Stale);
        }
        let base = match domain.model.predict_detailed(&self.identity, &features) {
            Ok(base) => base,
            Err(reason) => {
                if reason == PredictionRejection::OutsideObservedSupport {
                    if let Some(recorder) = &self.rejections {
                        recorder.record(self.revision, domain, &features, query, now);
                    }
                }
                return Err(model_rejection(reason));
            }
        };
        let bucket = self
            .boundaries
            .partition_point(|limit| (*limit as f64) < features.work_tokens())
            as u32;
        let Some(correction) =
            domain
                .calibration
                .correction(domain.model_revision, &features.domain(), bucket, now)
        else {
            // Every installed version had qualified same-domain calibration.
            return Err(CostUnknownReason::Stale);
        };
        let Ok(central_service_time) = Duration::try_from_secs_f64(base.central_seconds) else {
            return Err(unsupported(CostUnsupportedDetail::UnrepresentableDuration));
        };
        let Ok(upper_service_time) = Duration::try_from_secs_f64(
            (base.risk_base_seconds + correction.signed_seconds).max(0.0),
        ) else {
            return Err(unsupported(CostUnsupportedDetail::UnrepresentableDuration));
        };
        Ok(PreparedPrediction {
            domain,
            base,
            bucket,
            correction,
            central_service_time,
            upper_service_time,
        })
    }
}

impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        self.revision
    }
    fn query_limits(&self) -> CostQueryLimits {
        self.query
    }

    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        let prepared = match self.prepare(query, now) {
            Ok(prepared) => prepared,
            Err(reason) => return CostPrediction::Unknown(reason),
        };
        CostPrediction::Known {
            central_service_time: prepared.central_service_time,
            upper_service_time: prepared.upper_service_time,
            risk: CostRisk::CalibratedQuantile {
                quantile: self.quantile,
                approximation: CostRiskApproximation::EmpiricalConditional,
            },
            support: CostSupport::ValidatedFeatureDomain,
            samples: prepared.correction.samples,
            observed_at: prepared.correction.observed_at,
            expires_at: prepared
                .correction
                .expires_at
                .min(prepared.domain.model_expires_at),
        }
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        max_scenarios: usize,
    ) -> CostScenarioPrediction {
        if max_scenarios == 0 {
            return CostScenarioPrediction::Unknown(CostUnknownReason::QueryLimit);
        }
        let prepared = match self.prepare(query, now) {
            Ok(prepared) => prepared,
            Err(reason) => return CostScenarioPrediction::Unknown(reason),
        };
        let domain = prepared.domain;
        let count = max_scenarios.min(32);
        let Some(residuals) = domain.calibration.scenarios(
            domain.model_revision,
            &domain.model.domain(),
            prepared.bucket,
            now,
            count,
        ) else {
            return CostScenarioPrediction::Unknown(CostUnknownReason::Stale);
        };
        let mut scenarios = Vec::with_capacity(count);
        for (rank, residual) in residuals.signed_seconds.into_iter().enumerate() {
            let Ok(service_time) =
                Duration::try_from_secs_f64((prepared.base.risk_base_seconds + residual).max(0.0))
            else {
                return CostScenarioPrediction::Unknown(unsupported(
                    CostUnsupportedDetail::UnrepresentableDuration,
                ));
            };
            scenarios.push(CostScenario {
                rank,
                weight: 1.0 / count as f64,
                // Event simulation needs positive progress; nanoseconds are
                // Duration's resolution, not another fitted safety margin.
                service_time: service_time.max(Duration::from_nanos(1)),
            });
        }
        CostScenarioPrediction::Known {
            central_service_time: prepared.central_service_time,
            scenarios,
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: residuals.correction.samples,
            observed_at: residuals.correction.observed_at,
            expires_at: residuals.correction.expires_at.min(domain.model_expires_at),
        }
    }
}

pub(super) struct Source {
    snapshots: ArcSwap<Snapshot>,
    revisions: watch::Receiver<u64>,
    rejections: Arc<super::rejection::Recorder>,
}

impl Source {
    pub fn load_typed(&self) -> Arc<Snapshot> {
        self.snapshots.load_full()
    }
    pub fn rejection_stats(&self) -> super::LearningQueryRejectionStats {
        self.rejections.stats()
    }
    pub fn new(
        query: CostQueryLimits,
        identity: ExecutionIdentity,
        boundaries: Vec<usize>,
    ) -> (Self, watch::Sender<u64>) {
        let (sender, revisions) = watch::channel(0);
        let rejections = Arc::new(super::rejection::Recorder::new(
            query.max_participants,
            Instant::now(),
        ));
        (
            Self {
                snapshots: ArcSwap::from_pointee(Snapshot {
                    revision: 0,
                    query,
                    identity: Arc::new(identity),
                    boundaries: boundaries.into(),
                    quantile: 0.95,
                    domains: Vec::new(),
                    rejections: Some(rejections.clone()),
                }),
                revisions,
                rejections,
            },
            sender,
        )
    }
    pub fn install(&self, mut snapshot: Snapshot) {
        snapshot.rejections = Some(self.rejections.clone());
        self.snapshots.store(Arc::new(snapshot));
    }
}

impl CostSnapshotSource for Source {
    fn load(&self) -> Arc<dyn CostSnapshot> {
        self.snapshots.load_full()
    }
    fn subscribe(&self) -> watch::Receiver<u64> {
        self.revisions.clone()
    }
}
