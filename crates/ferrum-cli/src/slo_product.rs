//! Shared run/serve assembly and ownership of optional execution-cost feedback.

mod identity;
mod snapshot;

use ferrum_engine::{builder::ExecutionFeedbackBinding, EngineBuilder};
use ferrum_execution::execution_attempt::{ExecutionAttemptLimits, ExecutionAttemptSummary};
use ferrum_models::vnext::ProductionModelSourceBundle;
use ferrum_scheduler::cost_prediction::CostQuery;
use ferrum_slo::{
    learning::{LearningConfig, LearningEstimator, LearningEstimatorOwner},
    model::ModelFamily,
    stopping::{StoppingCollectorConfig, StoppingEstimatorOwner},
    ReferenceEstimator, ReferenceEstimatorOwner,
};
use ferrum_types::{EngineConfig, FerrumError, PredictorFamily, Result, SloConfig, SloMode};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

enum CostOwner {
    Reference(ReferenceEstimatorOwner),
    Learning(LearningEstimatorOwner),
}

impl CostOwner {
    fn observer(&self) -> Arc<dyn ferrum_execution::execution_attempt::ExecutionAttemptObserver> {
        match self {
            Self::Reference(owner) => owner.observer(),
            Self::Learning(owner) => owner.observer(),
        }
    }
    fn source(&self) -> Arc<dyn ferrum_scheduler::cost_prediction::CostSnapshotSource> {
        match self {
            Self::Reference(owner) => owner.source(),
            Self::Learning(owner) => owner.source(),
        }
    }
    fn stats(&self) -> ferrum_slo::ReferenceEstimatorStats {
        match self {
            Self::Reference(owner) => owner.stats(),
            Self::Learning(owner) => owner.stats(),
        }
    }
    async fn shutdown(&self, path: Option<&Path>) -> Result<()> {
        match self {
            Self::Reference(owner) => owner
                .shutdown()
                .await
                .map(|_| ())
                .map_err(|error| FerrumError::internal(error.to_string())),
            Self::Learning(owner) => {
                if let Some(path) = path {
                    let (_, report) = owner
                        .shutdown_and_save(path)
                        .await
                        .map_err(|error| FerrumError::internal(error.to_string()))?;
                    tracing::info!(?report, path = %path.display(), "SLO learned state shutdown");
                    Ok(())
                } else {
                    owner
                        .shutdown()
                        .await
                        .map(|_| ())
                        .map_err(|error| FerrumError::internal(error.to_string()))
                }
            }
        }
    }
}

pub(crate) struct ProductExecutionFeedback {
    owners: Arc<OnceLock<Owners>>,
    mode: SloMode,
    slo: Option<SloConfig>,
    state_path: Option<PathBuf>,
}

struct Owners {
    cost: CostOwner,
    stopping: StoppingEstimatorOwner,
    binding: Option<identity::BindingScope>,
    persistence: PersistenceStatus,
    state_path: Option<PathBuf>,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "status")]
enum PersistenceStatus {
    Disabled,
    Bound,
    SkippedUnbound { reason: String },
}

impl Owners {
    fn feedback(&self, mode: SloMode) -> ExecutionFeedbackBinding {
        ExecutionFeedbackBinding {
            execution_observer: self.cost.observer(),
            cost_source: (mode == SloMode::LearnAndControl).then(|| {
                Arc::new(snapshot::ProductSnapshotSource {
                    cost: self.cost.source(),
                    stopping: self.stopping.source(),
                }) as Arc<dyn ferrum_scheduler::cost_prediction::CostSnapshotSource>
            }),
            stopping_observer: Some(self.stopping.observer()),
        }
    }
}

/// Convert the old observation representation only at the assembly boundary.
/// The scheduler receives a neutral count/byte limit after this conversion.
pub(crate) fn resolve_product_slo(mut slo: SloConfig) -> Result<SloConfig> {
    slo.validate().map_err(FerrumError::config)?;
    if let Some((max_participants, max_summary_bytes)) = slo.legacy_query_observer_limits() {
        let limits = ExecutionAttemptLimits {
            max_participants,
            max_summary_bytes,
        };
        let mut low = 0;
        let mut high = max_participants;
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if limits.accepts(middle) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        if low == 0 {
            return Err(FerrumError::config(
                "SLO observation limits cannot fit one query participant",
            ));
        }
        slo.resolve_query_limits(ferrum_types::CostQueryLimits {
            max_participants: low,
            max_query_bytes: CostQuery::projected_bytes(low)
                .ok_or_else(|| FerrumError::config("SLO query size overflows"))?,
        });
    }
    if !CostQuery::fits(1, slo.policy.query) {
        return Err(FerrumError::config(
            "SLO query limits cannot fit one participant",
        ));
    }
    Ok(slo)
}

impl ProductExecutionFeedback {
    /// Off allocates no learning queue, worker, source, or request observer.
    pub(crate) fn prepare(
        config: &mut EngineConfig,
        slo: Option<SloConfig>,
        sources: Option<&ProductionModelSourceBundle>,
    ) -> Result<Self> {
        Self::prepare_with_state(config, slo, sources, None)
    }

    pub(crate) fn prepare_with_state(
        config: &mut EngineConfig,
        slo: Option<SloConfig>,
        _sources: Option<&ProductionModelSourceBundle>,
        state_path: Option<&Path>,
    ) -> Result<Self> {
        if state_path.is_some()
            && slo.is_none_or(|slo| {
                slo.mode == SloMode::Off || slo.predictor == PredictorFamily::ReferenceDiagnostic
            })
        {
            return Err(FerrumError::config(
                "SLO state requires an enabled learned predictor",
            ));
        }
        let Some(slo) = slo.map(resolve_product_slo).transpose()? else {
            config.scheduler.slo = None;
            return Ok(Self {
                owners: Arc::new(OnceLock::new()),
                mode: SloMode::Off,
                slo: None,
                state_path: None,
            });
        };
        config.scheduler.slo = slo.scheduler_policy();
        if slo.mode == SloMode::Off {
            return Ok(Self {
                owners: Arc::new(OnceLock::new()),
                mode: SloMode::Off,
                slo: None,
                state_path: None,
            });
        }
        let owners = Arc::new(OnceLock::new());
        match slo.predictor {
            PredictorFamily::ReferenceDiagnostic => {
                let limits = ExecutionAttemptLimits {
                    max_participants: slo.reference.max_participants,
                    max_summary_bytes: slo
                        .reference
                        .max_summary_bytes
                        .min(slo.reference.max_queue_bytes),
                };
                if !limits.accepts(slo.policy.query.max_participants) {
                    return Err(FerrumError::config(
                        "SLO query exceeds reference observation capacity",
                    ));
                }
                let cost = ReferenceEstimator::spawn(slo.reference)
                    .map(CostOwner::Reference)
                    .map_err(|e| FerrumError::config(e.to_string()))?;
                let stopping = StoppingEstimatorOwner::spawn(StoppingCollectorConfig::default())
                    .map_err(|e| FerrumError::config(e.to_string()))?;
                let _ = owners.set(Owners {
                    cost,
                    stopping,
                    binding: None,
                    persistence: PersistenceStatus::Disabled,
                    state_path: None,
                });
            }
            _ => {
                let learning = LearningConfig::default();
                if slo.policy.query.max_participants > learning.collection.max_participants
                    || !ExecutionAttemptSummary::projected_bytes(slo.policy.query.max_participants)
                        .is_some_and(|bytes| bytes <= learning.collection.max_summary_bytes)
                {
                    return Err(FerrumError::config(
                        "SLO query exceeds learning observation capacity",
                    ));
                }
            }
        }
        Ok(Self {
            owners,
            mode: slo.mode,
            slo: Some(slo),
            state_path: state_path.map(Path::to_path_buf),
        })
    }

    pub(crate) fn configure(&self, builder: EngineBuilder) -> EngineBuilder {
        let Some(slo) = self.slo else {
            return builder;
        };
        if let Some(owners) = self
            .owners
            .get()
            .filter(|_| slo.predictor == PredictorFamily::ReferenceDiagnostic)
        {
            let binding = owners.feedback(self.mode);
            let builder = builder.with_request_stopping_observer(
                binding.stopping_observer.expect("stopping observer"),
            );
            return match binding.cost_source {
                Some(source) => builder.with_execution_feedback(binding.execution_observer, source),
                None => builder.with_execution_attempt_observer(binding.execution_observer),
            };
        }
        let owners = Arc::clone(&self.owners);
        let requested_path = self.state_path.clone();
        builder.with_execution_feedback_factory(move |config, executor| {
            if owners.get().is_some() {
                return Err(FerrumError::config("feedback assembly already bound to an executor"));
            }
            let bound = identity::bind(config, executor)?;
            let (state_path, persistence) = match (&requested_path, &bound.scope) {
                (None, _) => (None, PersistenceStatus::Disabled),
                (Some(path), identity::BindingScope::Reusable) => (Some(path.clone()), PersistenceStatus::Bound),
                (Some(_), identity::BindingScope::InstanceScoped { reason }) => {
                    tracing::warn!(%reason, "SLO persistent reuse skipped: execution binding is instance scoped");
                    (None, PersistenceStatus::SkippedUnbound { reason: reason.clone() })
                }
            };
            let learning = LearningConfig { query: slo.policy.query, ..LearningConfig::default() };
            let learner = match slo.predictor {
                PredictorFamily::Auto => LearningEstimator::spawn_auto_with_state(learning, bound.identity, state_path.as_deref()),
                PredictorFamily::QuantileGbdt => LearningEstimator::spawn_with_state(learning, bound.identity, ModelFamily::Gbdt, state_path.as_deref()),
                _ => LearningEstimator::spawn_with_state(learning, bound.identity, ModelFamily::Physical, state_path.as_deref()),
            }.map_err(|e| FerrumError::config(e.to_string()))?;
            tracing::info!(requested = ?slo.predictor, resolution = ?learner.family_resolution(), state_load = ?learner.load_report(), "SLO predictor selection");
            let cost = CostOwner::Learning(learner);
            let stopping = StoppingEstimatorOwner::spawn(StoppingCollectorConfig::default())
                .map_err(|e| FerrumError::config(e.to_string()))?;
            owners.set(Owners { cost, stopping, binding: Some(bound.scope), persistence, state_path })
                .map_err(|_| FerrumError::config("feedback assembly was bound twice"))?;
            Ok(owners.get().expect("initialized feedback").feedback(slo.mode))
        })
    }

    pub(crate) fn into_owner(self) -> Option<Arc<Self>> {
        self.slo.is_some().then(|| Arc::new(self))
    }

    /// Health serialization is a cold path, separate from inference.
    pub(crate) fn stats(&self) -> Option<serde_json::Value> {
        self.owners.get().and_then(|owners| {
            serde_json::to_value(owners.cost.stats())
                .ok()
                .map(|mut value| {
                    value["mode"] = serde_json::json!(self.mode);
                    value["stopping"] = serde_json::json!(owners.stopping.stats());
                    value["execution_binding"] = serde_json::json!(owners.binding);
                    value["persistence"] = serde_json::json!(owners.persistence);
                    if let CostOwner::Learning(owner) = &owners.cost {
                        value["state_load"] = serde_json::json!(owner.load_report());
                        value["live_prediction"] = serde_json::json!(owner.prediction_stats());
                        value["predictor_selection"] = serde_json::json!(owner.family_resolution());
                    }
                    value
                })
        })
    }

    #[cfg(test)]
    pub(crate) fn revisions_for_test(&self) -> tokio::sync::watch::Receiver<u64> {
        self.owners
            .get()
            .expect("bound feedback")
            .cost
            .source()
            .subscribe()
    }

    /// Every cancelled shutdown wait may be retried against the original join.
    pub(crate) async fn shutdown(&self) -> Result<()> {
        let Some(owners) = self.owners.get() else {
            return Ok(());
        };
        let cost = owners.cost.shutdown(owners.state_path.as_deref());
        let stopping = async {
            owners
                .stopping
                .shutdown()
                .await
                .map(|_| ())
                .map_err(|error| FerrumError::internal(error.to_string()))
        };
        let (cost, stopping) = tokio::join!(cost, stopping);
        cost.and(stopping)
            .map_err(|error| FerrumError::internal(format!("execution feedback shutdown: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_explicit_off_need_no_runtime_or_worker() {
        let mut engine = EngineConfig::default();
        assert!(ProductExecutionFeedback::prepare(&mut engine, None, None)
            .unwrap()
            .into_owner()
            .is_none());
        let mut slo: SloConfig = "ttft:200,tpot:15,itl:50".parse().unwrap();
        slo.mode = SloMode::Off;
        assert!(
            ProductExecutionFeedback::prepare(&mut engine, Some(slo), None)
                .unwrap()
                .into_owner()
                .is_none()
        );
        assert!(engine.scheduler.slo.is_none());
    }

    #[tokio::test]
    async fn shared_feedback_owner_stops_the_original_worker() {
        for mode in [SloMode::Observe, SloMode::LearnAndControl] {
            let mut engine = EngineConfig::default();
            let mut slo: SloConfig = "ttft:200,tpot:15,itl:50".parse().unwrap();
            slo.mode = mode;
            slo.predictor = PredictorFamily::ReferenceDiagnostic;
            let feedback = ProductExecutionFeedback::prepare(&mut engine, Some(slo), None).unwrap();
            assert_eq!(
                engine.scheduler.slo.is_some(),
                mode == SloMode::LearnAndControl
            );
            feedback.shutdown().await.unwrap();
            feedback.shutdown().await.unwrap();
        }
    }

    #[test]
    fn legacy_bytes_become_neutral_participant_limits() {
        let bytes = ExecutionAttemptSummary::projected_bytes(3).unwrap();
        let raw = serde_json::json!({"ttft_ms":200,"tpot_ms":15,"itl_ms":50,
            "reference":{"max_participants":12,"max_summary_bytes":bytes}});
        let slo = resolve_product_slo(serde_json::from_value(raw).unwrap()).unwrap();
        assert_eq!(slo.policy.query.max_participants, 3);
        assert_eq!(
            slo.policy.query.max_query_bytes,
            CostQuery::projected_bytes(3).unwrap()
        );
        assert!(slo.legacy_query_observer_limits().is_none());
    }

    #[tokio::test]
    async fn actual_engine_assembly_starts_each_learned_predictor_once() {
        for predictor in [
            PredictorFamily::Auto,
            PredictorFamily::Physical,
            PredictorFamily::QuantileGbdt,
        ] {
            for mode in [SloMode::Observe, SloMode::LearnAndControl] {
                let mut engine = EngineConfig::default();
                let mut slo: SloConfig = "ttft:200,tpot:15,itl:50".parse().unwrap();
                slo.predictor = predictor;
                slo.mode = mode;
                let feedback =
                    ProductExecutionFeedback::prepare(&mut engine, Some(slo), None).unwrap();
                assert_eq!(
                    engine.scheduler.slo.is_some(),
                    mode == SloMode::LearnAndControl
                );
                assert!(
                    feedback.stats().is_none(),
                    "no learner before actual executor binding"
                );
                let engine = feedback
                    .configure(EngineBuilder::new(engine))
                    .build()
                    .await
                    .unwrap();
                assert!(feedback.stats().is_some());
                assert_eq!(
                    feedback.stats().unwrap()["execution_binding"]["scope"],
                    "InstanceScoped"
                );
                let selected = &feedback.stats().unwrap()["predictor_selection"];
                assert_eq!(
                    selected["resolution"],
                    if predictor == PredictorFamily::Auto {
                        "AutoPhysicalFallback"
                    } else {
                        "Explicit"
                    }
                );
                if predictor == PredictorFamily::Auto {
                    assert_eq!(selected["reason"], "NoBoundStateProvided");
                }
                engine.shutdown().await.unwrap();
                engine.wait_for_shutdown_completion().await.unwrap();
                feedback.shutdown().await.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn unbound_state_path_is_visible_and_never_reads_or_overwrites_an_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        std::fs::write(&path, b"invalid previous artifact").unwrap();
        let mut config = EngineConfig::default();
        let slo: SloConfig = "ttft:200,tpot:15,itl:50".parse().unwrap();
        let feedback =
            ProductExecutionFeedback::prepare_with_state(&mut config, Some(slo), None, Some(&path))
                .unwrap();
        let engine = feedback
            .configure(EngineBuilder::new(config))
            .build()
            .await
            .unwrap();
        let stats = feedback.stats().unwrap();
        assert_eq!(stats["persistence"]["status"], "SkippedUnbound");
        assert!(stats["persistence"]["reason"].as_str().unwrap().len() > 0);
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();
        feedback.shutdown().await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"invalid previous artifact");
    }
}
