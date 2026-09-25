//! Observe existing product result labels only during this synchronous query.
//! No global recorder, replacement predictor, or production diagnostic API.
use metrics::{
    Counter, CounterFn, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Default)]
pub(super) struct QueryMetrics(Arc<Mutex<BTreeMap<String, u64>>>);
impl QueryMetrics {
    pub(super) fn snapshot(&self) -> BTreeMap<String, u64> {
        self.0.lock().unwrap().clone()
    }
}
struct Count {
    reason: String,
    values: Arc<Mutex<BTreeMap<String, u64>>>,
}
impl CounterFn for Count {
    fn increment(&self, value: u64) {
        let mut values = self.values.lock().unwrap();
        let count = values.entry(self.reason.clone()).or_default();
        *count = count.checked_add(value).unwrap();
    }
    fn absolute(&self, value: u64) {
        let mut values = self.values.lock().unwrap();
        let count = values.entry(self.reason.clone()).or_default();
        *count = (*count).max(value);
    }
}
impl Recorder for QueryMetrics {
    fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> Counter {
        if key.name() != "ferrum.engine.structured_v2_cost_queries_total" {
            return Counter::noop();
        }
        let reason = key.labels().find(|l| l.key() == "reason").unwrap().value();
        Counter::from_arc(Arc::new(Count {
            reason: reason.into(),
            values: Arc::clone(&self.0),
        }))
    }
    fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> Gauge {
        Gauge::noop()
    }
    fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
        Histogram::noop()
    }
}
