use crate::model_executor::PrefixRestoreObservation;
use ferrum_runtime::{ExecutionEventSink, ExecutionEventSinkError};

/// Product observations that borrow executor-owned restore evidence.
///
/// A product sink also consumes runtime events. An `Arc` can be upcast to the
/// runtime trait without allocating another sink or transport; runtime event
/// emission still requires the original sealed permits.
pub trait ProductExecutionEventSink: ExecutionEventSink {
    /// Prefix lookup precedes a per-request execution cursor. Opt-in sinks
    /// receive resource-only decisions through their existing trace transport.
    fn records_prefix_restore_decisions(&self) -> bool {
        false
    }

    fn record_prefix_restore_decision(
        &self,
        _observation: &PrefixRestoreObservation<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_executor::{PrefixRestoreDecision, PrefixRestoreSource};
    use ferrum_runtime::{EventEmissionPermit, ExecutionEventCapturePolicy, ExecutionEventKind};
    use ferrum_types::RequestId;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Default)]
    struct SharedSink {
        runtime_queries: AtomicUsize,
        prefix_decisions: AtomicUsize,
    }

    impl ExecutionEventSink for SharedSink {
        fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
            false
        }

        fn capture_policy(&self) -> ExecutionEventCapturePolicy {
            self.runtime_queries.fetch_add(1, Ordering::Relaxed);
            ExecutionEventCapturePolicy::AllFrames
        }

        fn record(&self, _permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
            Ok(())
        }
    }

    impl ProductExecutionEventSink for SharedSink {
        fn records_prefix_restore_decisions(&self) -> bool {
            true
        }

        fn record_prefix_restore_decision(
            &self,
            observation: &PrefixRestoreObservation<'_>,
        ) -> Result<(), ExecutionEventSinkError> {
            assert!(matches!(
                observation.decision,
                PrefixRestoreDecision::NoReusableEntry
            ));
            self.prefix_decisions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn product_and_runtime_views_share_one_sink() {
        let sink = Arc::new(SharedSink::default());
        let product: Arc<dyn ProductExecutionEventSink> = sink.clone();
        let runtime: Arc<dyn ExecutionEventSink> = product.clone();
        assert_eq!(
            Arc::as_ptr(&product) as *const (),
            Arc::as_ptr(&runtime) as *const ()
        );

        assert!(product.records_prefix_restore_decisions());
        product
            .record_prefix_restore_decision(&PrefixRestoreObservation {
                request_id: &RequestId::new(),
                source: PrefixRestoreSource::Index,
                decision: PrefixRestoreDecision::NoReusableEntry,
            })
            .unwrap();
        // Capture policy uses the same lower trait object; disabled runtime
        // events do not disable the independently opted-in product observation.
        let _ = runtime.capture_policy();
        assert_eq!(sink.runtime_queries.load(Ordering::Relaxed), 1);
        assert_eq!(sink.prefix_decisions.load(Ordering::Relaxed), 1);
    }
}
