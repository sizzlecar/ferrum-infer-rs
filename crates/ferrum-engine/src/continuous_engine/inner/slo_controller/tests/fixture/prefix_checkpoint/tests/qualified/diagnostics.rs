//! Existing cold failure events from the real worker and block-close threads.
//! Enabling their bounded context scans and stderr output has logging overhead;
//! it does not rewrite observations or change the qualification rules.
use std::{fmt::Write, sync::Once};
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
    Event, Level, Metadata, Subscriber,
};

pub(super) fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // A thread-local subscriber would miss the existing std::thread
        // workers. Preserve any subscriber already installed by the harness.
        if let Err(error) = tracing::subscriber::set_global_default(OwnerDiagnostics) {
            eprintln!(
                "qualified startup owner diagnostics not installed; existing subscriber preserved: {error}"
            );
        }
    });
}

struct OwnerDiagnostics;
impl Subscriber for OwnerDiagnostics {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target() == "ferrum_scheduler::structured_owner_diagnostics"
            && *metadata.level() == Level::WARN
    }
    fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
        Some(tracing::metadata::LevelFilter::WARN)
    }
    fn register_callsite(&self, _: &'static Metadata<'static>) -> tracing::subscriber::Interest {
        // Other tests can still select their own thread-local subscriber.
        tracing::subscriber::Interest::sometimes()
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields(String::new());
        event.record(&mut fields);
        eprintln!("structured owner diagnostic: {}", fields.0);
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

struct Fields(String);
impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        let _ = write!(self.0, "{}={value:?}", field.name());
    }
}
