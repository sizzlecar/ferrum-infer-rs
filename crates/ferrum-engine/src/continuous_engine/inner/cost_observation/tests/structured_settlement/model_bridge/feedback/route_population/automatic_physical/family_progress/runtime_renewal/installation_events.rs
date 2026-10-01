//! Thread-local capture of the actual product event, with no global subscriber.
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub(super) struct Installations(Arc<Mutex<Vec<serde_json::Value>>>);
impl Installations {
    pub(super) fn during<T>(&self, run: impl FnOnce() -> T) -> T {
        tracing::subscriber::with_default(self.clone(), run)
    }
    pub(super) fn records(&self) -> Vec<serde_json::Value> {
        self.0.lock().unwrap().clone()
    }
}

#[derive(Default)]
struct Fields {
    event: Option<String>,
    installation: Option<String>,
}
impl tracing::field::Visit for Fields {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        match field.name() {
            "event" => self.event = Some(value.into()),
            "installation" => self.installation = Some(value.into()),
            _ => {}
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "installation" {
            self.installation = Some(format!("{value:?}"));
        }
    }
}

impl tracing::Subscriber for Installations {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrum_engine::continuous_engine::inner::cost_observation::runtime"
    }
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        if fields.event.as_deref() == Some("structured_catalog_installed_v1") {
            self.0.lock().unwrap().push(
                serde_json::from_str(fields.installation.as_deref().expect("original payload"))
                    .expect("bounded typed installation JSON"),
            );
        }
        assert_ne!(
            fields.event.as_deref(),
            Some("structured_catalog_installation_diagnostic_incomplete_v1"),
            "the small real fixture must retain its complete original provenance"
        );
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}
