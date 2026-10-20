//! Shared run/serve assembly and ownership of optional execution-cost feedback.

use ferrum_engine::EngineBuilder;
use ferrum_slo::{ReferenceEstimator, ReferenceEstimatorOwner};
use ferrum_types::{EngineConfig, FerrumError, Result};
use std::sync::Arc;

pub(crate) struct ProductExecutionFeedback {
    owner: Option<ReferenceEstimatorOwner>,
}

impl ProductExecutionFeedback {
    /// None is a real off path: no estimator, worker, queue or subscription.
    pub(crate) fn prepare(config: &EngineConfig) -> Result<Self> {
        let owner = config
            .scheduler
            .slo
            .map(|slo| {
                slo.validate().map_err(FerrumError::config)?;
                ReferenceEstimator::spawn(slo.reference)
                    .map_err(|error| FerrumError::config(error.to_string()))
            })
            .transpose()?;
        Ok(Self { owner })
    }

    pub(crate) fn configure(&self, builder: EngineBuilder) -> EngineBuilder {
        match &self.owner {
            Some(owner) => builder.with_execution_feedback(owner.observer(), owner.source()),
            None => builder,
        }
    }

    pub(crate) fn into_owner(self) -> Option<Arc<Self>> {
        self.owner.is_some().then(|| Arc::new(self))
    }

    /// Health reporting is a cold path; no serialization runs with inference.
    pub(crate) fn stats(&self) -> Option<serde_json::Value> {
        self.owner
            .as_ref()
            .and_then(|owner| serde_json::to_value(owner.stats()).ok())
    }

    #[cfg(test)]
    pub(crate) fn revisions_for_test(&self) -> tokio::sync::watch::Receiver<u64> {
        self.owner
            .as_ref()
            .expect("enabled feedback")
            .source()
            .subscribe()
    }

    /// Observe the original worker, including after an earlier wait was
    /// cancelled. Its owner also stops and retains its join on startup Drop.
    pub(crate) async fn shutdown(&self) -> Result<()> {
        if let Some(owner) = &self.owner {
            let _exit = owner.shutdown().await.map_err(|error| {
                FerrumError::internal(format!("execution feedback shutdown: {error}"))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_off_needs_no_runtime_or_worker_owner() {
        let feedback = ProductExecutionFeedback::prepare(&EngineConfig::default()).unwrap();
        assert!(feedback.into_owner().is_none());
    }

    #[tokio::test]
    async fn shared_feedback_owner_stops_the_original_worker() {
        let mut config = EngineConfig::default();
        config.scheduler.slo = Some("ttft:200,tpot:15,itl:50".parse().unwrap());
        let feedback = ProductExecutionFeedback::prepare(&config).unwrap();
        assert!(feedback.owner.is_some());
        feedback.shutdown().await.unwrap();
        feedback.shutdown().await.unwrap();
    }
}
