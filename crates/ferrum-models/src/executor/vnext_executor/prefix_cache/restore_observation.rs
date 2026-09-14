use super::*;
use ferrum_interfaces::model_executor::PrefixRestoreObservation;

pub(super) struct PrefixRestoreObserver {
    sink: Option<Arc<dyn ExecutionEventSink>>,
    request_id: RequestId,
    source: PrefixRestoreSource,
}

impl PrefixRestoreObserver {
    pub(super) fn new(
        sink: Option<Arc<dyn ExecutionEventSink>>,
        request_id: RequestId,
        source: PrefixRestoreSource,
    ) -> Self {
        Self {
            sink: sink.filter(|sink| sink.records_prefix_restore_decisions()),
            request_id,
            source,
        }
    }

    pub(super) fn record(&self, decision: PrefixRestoreDecision<'_>) {
        let Some(sink) = &self.sink else { return };
        if let Err(error) = sink.record_prefix_restore_decision(&PrefixRestoreObservation {
            request_id: &self.request_id,
            source: self.source,
            decision,
        }) {
            // Observation failure cannot change an acknowledged publication or
            // convert a safe cold-prefill fallback into a request failure.
            tracing::warn!(request_id = %self.request_id, %error, "prefix restore observation failed");
        }
    }

    pub(super) fn capacity_ready(
        &self,
        candidate_prefix_tokens: usize,
        decision: &VNextExecutionCapacityDecision<()>,
    ) -> bool {
        match decision {
            VNextExecutionCapacityDecision::Ready(()) => true,
            VNextExecutionCapacityDecision::Deferred(capacity) => {
                self.record(PrefixRestoreDecision::SequenceCapacityNotReady {
                    candidate_prefix_tokens,
                    capacity,
                });
                false
            }
            VNextExecutionCapacityDecision::RequestStateDeferred(request_state) => {
                self.record(PrefixRestoreDecision::RequestStateNotReady {
                    candidate_prefix_tokens,
                    request_state,
                });
                false
            }
        }
    }

    pub(super) fn acknowledge(
        &self,
        metrics: &PrefixCacheMetrics,
        restored_tokens: usize,
        acknowledge: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        metrics.acknowledge_restore(restored_tokens, acknowledge)?;
        self.record(PrefixRestoreDecision::Restored {
            candidate_prefix_tokens: restored_tokens,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sink {
        enabled: bool,
        fail: bool,
        decisions: Mutex<Vec<serde_json::Value>>,
    }

    impl ExecutionEventSink for Sink {
        fn is_enabled(&self, _: ExecutionEventKind) -> bool {
            false
        }
        fn records_prefix_restore_decisions(&self) -> bool {
            self.enabled
        }
        fn record_prefix_restore_decision(
            &self,
            observation: &PrefixRestoreObservation<'_>,
        ) -> std::result::Result<(), ExecutionEventSinkError> {
            self.decisions
                .lock()
                .push(serde_json::to_value(observation).unwrap());
            if self.fail {
                Err(ExecutionEventSinkError::new("sink unavailable"))
            } else {
                Ok(())
            }
        }
        fn record(
            &self,
            _: EventEmissionPermit,
        ) -> std::result::Result<(), ExecutionEventSinkError> {
            panic!("prefix decisions must not manufacture execution cursor events")
        }
    }

    fn recording_sink(enabled: bool, fail: bool) -> Arc<Sink> {
        Arc::new(Sink {
            enabled,
            fail,
            decisions: Mutex::new(Vec::new()),
        })
    }

    #[test]
    fn prefix_restore_observation_requires_ack_and_does_not_change_publication_errors() {
        let sink = recording_sink(true, false);
        let metrics = Arc::new(PrefixCacheMetrics::default());
        let output = |cancelled: bool| {
            let request = RequestId::new();
            let observer = PrefixRestoreObserver::new(
                Some(sink.clone()),
                request.clone(),
                PrefixRestoreSource::Rendezvous,
            );
            let metrics = Arc::clone(&metrics);
            PlanRuntimePrefixRestoreOutput::new(
                request.clone(),
                3,
                4,
                Arc::new(ferrum_testkit::MockKvCacheHandle::new(request, 1, 3)),
                move || {
                    observer.acknowledge(&metrics, 3, || {
                        if cancelled {
                            Err(FerrumError::cancelled("target cancelled"))
                        } else {
                            Ok(())
                        }
                    })
                },
            )
            .unwrap()
        };
        drop(output(false));
        assert!(output(true).acknowledge().is_err());
        assert!(sink.decisions.lock().is_empty());
        assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);
        assert_eq!(output(false).acknowledge().unwrap().committed_tokens(), 3);
        assert_eq!(metrics.hits.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.saved_prefill_tokens.load(Ordering::Relaxed), 3);
        let decisions = sink.decisions.lock();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0]["source"], "rendezvous");
        assert_eq!(
            decisions[0]["decision"],
            serde_json::json!({"outcome":"restored","candidate_prefix_tokens":3})
        );
        drop(decisions);

        let failing = recording_sink(true, true);
        let observer =
            PrefixRestoreObserver::new(Some(failing), RequestId::new(), PrefixRestoreSource::Index);
        assert!(observer.acknowledge(&metrics, 2, || Ok(())).is_ok());
        assert_eq!(metrics.hits.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.saved_prefill_tokens.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn prefix_restore_observation_preserves_capacity_evidence_and_disabled_sink() {
        use std::num::NonZeroU64;
        let observed =
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 7)
                .unwrap();
        let capacity = ExecutorExecutionCapacityDeferral::from_backing_pressure(
            ExecutorAdmissionEpochs::new(NonZeroU64::new(19).unwrap(), 3, 5),
            CapacityWaitCondition::from_observation(19, vec![observed]).unwrap(),
            DeviceCapacityPressure::new(
                DeviceCapacityPressureScope::PlanBudget,
                "device.test".into(),
                32,
                64,
                64,
                64,
                64,
            )
            .unwrap()
            .into(),
            ExecutorExecutionCapacityStage::SequenceExtension,
        )
        .unwrap();
        let expected = serde_json::to_value(&capacity).unwrap();
        let sink = recording_sink(true, false);
        let request_id = RequestId::new();
        let observer = PrefixRestoreObserver::new(
            Some(sink.clone()),
            request_id.clone(),
            PrefixRestoreSource::Index,
        );
        assert!(observer.capacity_ready(128, &VNextExecutionCapacityDecision::Ready(())));
        assert!(sink.decisions.lock().is_empty());
        let decision = VNextExecutionCapacityDecision::Deferred(capacity);
        assert!(!observer.capacity_ready(128, &decision));
        let decisions = sink.decisions.lock();
        assert_eq!(decisions[0]["request_id"], serde_json::json!(request_id));
        assert_eq!(
            decisions[0]["decision"]["outcome"],
            "sequence_capacity_not_ready"
        );
        assert_eq!(decisions[0]["decision"]["candidate_prefix_tokens"], 128);
        assert_eq!(decisions[0]["decision"]["capacity"], expected);
        drop(decisions);
        let disabled = recording_sink(false, false);
        let observer = PrefixRestoreObserver::new(
            Some(disabled.clone()),
            request_id,
            PrefixRestoreSource::Index,
        );
        assert!(!observer.capacity_ready(128, &decision));
        observer.record(PrefixRestoreDecision::NoReusableEntry);
        assert!(disabled.decisions.lock().is_empty());
    }
}
