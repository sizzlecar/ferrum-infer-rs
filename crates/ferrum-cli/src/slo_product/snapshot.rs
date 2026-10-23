//! One planning turn captures both independently published observation streams.
//! Base model and calibration are already atomic inside the cost snapshot.

use ferrum_interfaces::request_observation::{
    CompletionForecast, CompletionQuery, StoppingForecast, StoppingQuery, StoppingSnapshot,
    StoppingSnapshotSource,
};
use ferrum_scheduler::cost_prediction::{
    CostPrediction, CostQuery, CostScenarioPrediction, CostSnapshot, CostSnapshotSource,
};
use ferrum_types::CostQueryLimits;
use std::{sync::Arc, time::Instant};
use tokio::sync::watch;

pub(super) struct ProductSnapshotSource {
    pub cost: Arc<dyn CostSnapshotSource>,
    pub stopping: Arc<dyn StoppingSnapshotSource>,
}

struct ProductSnapshot {
    cost: Arc<dyn CostSnapshot>,
    stopping: Arc<dyn StoppingSnapshot>,
}

impl CostSnapshot for ProductSnapshot {
    fn revision(&self) -> u64 {
        self.cost.revision()
    }
    fn query_limits(&self) -> CostQueryLimits {
        self.cost.query_limits()
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        self.cost.predict(query, now)
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        max_scenarios: usize,
    ) -> CostScenarioPrediction {
        self.cost.cost_scenarios(query, now, max_scenarios)
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.stopping.forecast(query)
    }
    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        self.stopping.completion_forecast(query)
    }
}

impl CostSnapshotSource for ProductSnapshotSource {
    fn load(&self) -> Arc<dyn CostSnapshot> {
        Arc::new(ProductSnapshot {
            cost: self.cost.load(),
            stopping: self.stopping.snapshot(),
        })
    }
    fn subscribe(&self) -> watch::Receiver<u64> {
        // Unknown stopping Wait is bounded by original ingress. Cost revision
        // wakeups are advisory; no request relies on a stopping notice to progress.
        self.cost.subscribe()
    }
}
