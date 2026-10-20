//! Observe the actual request gauge without installing a process-global recorder.

use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Default)]
struct Value(Mutex<f64>);

impl ::metrics::GaugeFn for Value {
    fn increment(&self, value: f64) {
        *self.0.lock() += value;
    }
    fn decrement(&self, value: f64) {
        *self.0.lock() -= value;
    }
    fn set(&self, value: f64) {
        *self.0.lock() = value;
    }
}

#[derive(Default)]
pub(super) struct ActiveRequests(Arc<Value>);

impl ActiveRequests {
    pub(super) fn value(&self) -> f64 {
        *self.0 .0.lock()
    }
}

impl ::metrics::Recorder for ActiveRequests {
    fn describe_counter(
        &self,
        _: ::metrics::KeyName,
        _: Option<::metrics::Unit>,
        _: ::metrics::SharedString,
    ) {
    }
    fn describe_gauge(
        &self,
        _: ::metrics::KeyName,
        _: Option<::metrics::Unit>,
        _: ::metrics::SharedString,
    ) {
    }
    fn describe_histogram(
        &self,
        _: ::metrics::KeyName,
        _: Option<::metrics::Unit>,
        _: ::metrics::SharedString,
    ) {
    }
    fn register_counter(
        &self,
        _: &::metrics::Key,
        _: &::metrics::Metadata<'_>,
    ) -> ::metrics::Counter {
        ::metrics::Counter::noop()
    }
    fn register_histogram(
        &self,
        _: &::metrics::Key,
        _: &::metrics::Metadata<'_>,
    ) -> ::metrics::Histogram {
        ::metrics::Histogram::noop()
    }
    fn register_gauge(
        &self,
        key: &::metrics::Key,
        _: &::metrics::Metadata<'_>,
    ) -> ::metrics::Gauge {
        if key.name() == "ferrum.engine.active_requests" {
            ::metrics::Gauge::from_arc(Arc::clone(&self.0))
        } else {
            ::metrics::Gauge::noop()
        }
    }
}
