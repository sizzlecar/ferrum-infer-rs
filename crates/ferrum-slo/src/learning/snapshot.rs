use crate::{
    calibration::CalibrationSnapshot,
    model::{ExecutionIdentity, FeatureDomain, FittedModel, WorkFeatures},
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
}

impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        self.revision
    }
    fn query_limits(&self) -> CostQueryLimits {
        self.query
    }

    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        if !CostQuery::fits(query.work.len(), self.query) {
            return CostPrediction::Unknown(CostUnknownReason::QueryLimit);
        }
        if !query.work_is_exact {
            return CostPrediction::Unknown(CostUnknownReason::InexactWork);
        }
        if query.execution == CostExecution::SequentialPrefillThenDecode
            && query
                .work
                .iter()
                .any(|w| w.kind == AttemptWorkKind::Prefill)
            && query.work.iter().any(|w| w.kind == AttemptWorkKind::Decode)
        {
            // No inferred joint quantile from pure-phase or mixed-invocation fits.
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        }
        let Ok(features) = WorkFeatures::extract(query.work, self.query.max_participants) else {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        };
        let Some(domain) = self
            .domains
            .iter()
            .find(|d| d.model.domain() == features.domain())
        else {
            return CostPrediction::Unknown(CostUnknownReason::NoCoverage);
        };
        if now >= domain.model_expires_at {
            return CostPrediction::Unknown(CostUnknownReason::Stale);
        }
        let Ok(base) = domain.model.predict(&self.identity, &features) else {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
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
            return CostPrediction::Unknown(CostUnknownReason::Stale);
        };
        let Ok(central_service_time) = Duration::try_from_secs_f64(base.central_seconds) else {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        };
        let Ok(upper_service_time) = Duration::try_from_secs_f64(
            (base.risk_base_seconds + correction.signed_seconds).max(0.0),
        ) else {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        };
        CostPrediction::Known {
            central_service_time,
            upper_service_time,
            risk: CostRisk::CalibratedQuantile {
                quantile: self.quantile,
                approximation: CostRiskApproximation::EmpiricalConditional,
            },
            support: CostSupport::ValidatedFeatureDomain,
            samples: correction.samples,
            observed_at: correction.observed_at,
            expires_at: correction.expires_at.min(domain.model_expires_at),
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
        let (central_service_time, support) = match self.predict(query, now) {
            CostPrediction::Known {
                central_service_time,
                support,
                ..
            } => (central_service_time, support),
            CostPrediction::Unknown(reason) => return CostScenarioPrediction::Unknown(reason),
        };
        let Ok(features) = WorkFeatures::extract(query.work, self.query.max_participants) else {
            return CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        };
        let Some(domain) = self
            .domains
            .iter()
            .find(|d| d.model.domain() == features.domain())
        else {
            return CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage);
        };
        let Ok(base) = domain.model.predict(&self.identity, &features) else {
            return CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        };
        let bucket = self
            .boundaries
            .partition_point(|limit| (*limit as f64) < features.work_tokens())
            as u32;
        let count = max_scenarios.min(32);
        let Some(residuals) = domain.calibration.scenarios(
            domain.model_revision,
            &features.domain(),
            bucket,
            now,
            count,
        ) else {
            return CostScenarioPrediction::Unknown(CostUnknownReason::Stale);
        };
        let mut scenarios = Vec::with_capacity(count);
        for (rank, residual) in residuals.signed_seconds.into_iter().enumerate() {
            let Ok(service_time) =
                Duration::try_from_secs_f64((base.risk_base_seconds + residual).max(0.0))
            else {
                return CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
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
            central_service_time,
            scenarios,
            dependence: CostScenarioDependence::SharedResidualRank,
            support,
            samples: residuals.correction.samples,
            observed_at: residuals.correction.observed_at,
            expires_at: residuals.correction.expires_at.min(domain.model_expires_at),
        }
    }
}

pub(super) struct Source {
    snapshots: ArcSwap<Snapshot>,
    revisions: watch::Receiver<u64>,
}

impl Source {
    pub fn load_typed(&self) -> Arc<Snapshot> {
        self.snapshots.load_full()
    }
    pub fn new(
        query: CostQueryLimits,
        identity: ExecutionIdentity,
        boundaries: Vec<usize>,
    ) -> (Self, watch::Sender<u64>) {
        let (sender, revisions) = watch::channel(0);
        (
            Self {
                snapshots: ArcSwap::from_pointee(Snapshot {
                    revision: 0,
                    query,
                    identity: Arc::new(identity),
                    boundaries: boundaries.into(),
                    quantile: 0.95,
                    domains: Vec::new(),
                }),
                revisions,
            },
            sender,
        )
    }
    pub fn install(&self, snapshot: Snapshot) {
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
