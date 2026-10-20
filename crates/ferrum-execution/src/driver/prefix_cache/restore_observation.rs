use super::*;
use crate::model_executor::{PrefixRestoreAcknowledgement, PrefixRestoreObservation};

pub(super) struct PrefixRestoreObserver {
    sink: Option<Arc<dyn ProductExecutionEventSink>>,
    request_id: RequestId,
    source: PrefixRestoreSource,
}

impl PrefixRestoreObserver {
    pub(super) fn new(
        sink: Option<Arc<dyn ProductExecutionEventSink>>,
        request_id: RequestId,
        source: PrefixRestoreSource,
    ) -> Self {
        Self {
            sink: sink.filter(|sink| sink.records_prefix_restore_decisions()),
            request_id,
            source,
        }
    }

    pub(super) fn record(&self, decision: PrefixRestoreDecision<'_>) -> Result<()> {
        let Some(sink) = &self.sink else {
            return Ok(());
        };
        let result = sink.record_prefix_restore_decision(&PrefixRestoreObservation {
            request_id: &self.request_id,
            source: self.source,
            decision,
        });
        crate::event_sink::resolve_event_delivery(sink.as_ref(), result)
            .map_err(|error| FerrumError::backend(format!("prefix restore observation: {error}")))
    }

    pub(super) fn capacity_ready(
        &self,
        candidate_prefix_tokens: usize,
        decision: &VNextExecutionCapacityDecision<()>,
    ) -> Result<bool> {
        match decision {
            VNextExecutionCapacityDecision::Ready(()) => Ok(true),
            VNextExecutionCapacityDecision::Deferred(capacity) => {
                self.record(PrefixRestoreDecision::SequenceCapacityNotReady {
                    candidate_prefix_tokens,
                    capacity,
                })?;
                Ok(false)
            }
            VNextExecutionCapacityDecision::RequestStateDeferred(request_state) => {
                self.record(PrefixRestoreDecision::RequestStateNotReady {
                    candidate_prefix_tokens,
                    request_state,
                })?;
                Ok(false)
            }
        }
    }

    pub(super) fn acknowledge(
        &self,
        metrics: &PrefixCacheMetrics,
        restored_tokens: usize,
        acknowledge: impl FnOnce() -> Result<crate::token_execution::ExecutionFrontier>,
    ) -> Result<PrefixRestoreAcknowledgement> {
        let frontier = metrics.acknowledge_restore(restored_tokens, acknowledge)?;
        let error = self
            .record(PrefixRestoreDecision::Restored {
                candidate_prefix_tokens: restored_tokens,
            })
            .err();
        Ok(PrefixRestoreAcknowledgement { frontier, error })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sink {
        enabled: bool,
        fail: bool,
        policy: ExecutionEventDeliveryPolicy,
        kind: ExecutionEventSinkErrorKind,
        decisions: Mutex<Vec<serde_json::Value>>,
    }

    impl ExecutionEventSink for Sink {
        fn delivery_policy(&self) -> ExecutionEventDeliveryPolicy {
            self.policy
        }
        fn is_enabled(&self, _: ExecutionEventKind) -> bool {
            false
        }
        fn record(
            &self,
            _: EventEmissionPermit,
        ) -> std::result::Result<(), ExecutionEventSinkError> {
            panic!("prefix decisions must not manufacture execution cursor events")
        }
    }

    impl ProductExecutionEventSink for Sink {
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
                Err(match self.kind {
                    ExecutionEventSinkErrorKind::Contract => {
                        ExecutionEventSinkError::contract("invalid prefix projection")
                    }
                    ExecutionEventSinkErrorKind::Transport => {
                        ExecutionEventSinkError::transport("sink unavailable")
                    }
                })
            } else {
                Ok(())
            }
        }
    }

    fn recording_sink(enabled: bool, fail: bool) -> Arc<Sink> {
        Arc::new(Sink {
            enabled,
            fail,
            policy: ExecutionEventDeliveryPolicy::OptionalTelemetry,
            kind: ExecutionEventSinkErrorKind::Transport,
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
            let issuer = crate::token_execution::ExecutorSequenceIssuer::default();
            let sequence = issuer.issue_sequence().unwrap();
            PlanRuntimePrefixRestoreOutput::new_tokens(request, 3, 4, sequence.clone(), move || {
                observer.acknowledge(&metrics, 3, || {
                    if cancelled {
                        Err(FerrumError::cancelled("target cancelled"))
                    } else {
                        issuer.frontier(&sequence, 3)
                    }
                })
            })
            .unwrap()
        };
        drop(output(false));
        assert!(output(true).acknowledge_tokens().is_err());
        assert!(sink.decisions.lock().is_empty());
        assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);
        assert_eq!(
            output(false)
                .acknowledge_tokens()
                .unwrap()
                .frontier
                .consumed_tokens(),
            3
        );
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
        let issuer = crate::token_execution::ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let acknowledged = observer
            .acknowledge(&metrics, 2, || issuer.frontier(&sequence, 2))
            .unwrap();
        assert!(acknowledged.error.is_none());
        assert_eq!(acknowledged.frontier.sequence(), &sequence);
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
        assert!(observer
            .capacity_ready(128, &VNextExecutionCapacityDecision::Ready(()))
            .unwrap());
        assert!(sink.decisions.lock().is_empty());
        let decision = VNextExecutionCapacityDecision::Deferred(capacity);
        assert!(!observer.capacity_ready(128, &decision).unwrap());
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
        assert!(!observer.capacity_ready(128, &decision).unwrap());
        observer
            .record(PrefixRestoreDecision::NoReusableEntry)
            .unwrap();
        assert!(disabled.decisions.lock().is_empty());
    }

    #[test]
    fn prefix_restore_post_ack_contract_and_required_transport_keep_actual_frontier() {
        for (policy, kind) in [
            (
                ExecutionEventDeliveryPolicy::OptionalTelemetry,
                ExecutionEventSinkErrorKind::Contract,
            ),
            (
                ExecutionEventDeliveryPolicy::RequiredEvidence,
                ExecutionEventSinkErrorKind::Transport,
            ),
        ] {
            let sink = Arc::new(Sink {
                enabled: true,
                fail: true,
                policy,
                kind,
                decisions: Mutex::new(Vec::new()),
            });
            let observer = PrefixRestoreObserver::new(
                Some(sink),
                RequestId::new(),
                PrefixRestoreSource::Index,
            );
            let metrics = PrefixCacheMetrics::default();
            let issuer = crate::token_execution::ExecutorSequenceIssuer::new();
            let sequence = issuer.issue_sequence().unwrap();
            let actual = issuer.frontier(&sequence, 3).unwrap();
            let acknowledged = observer
                .acknowledge(&metrics, 3, || Ok(actual.clone()))
                .unwrap();
            assert!(acknowledged.error.is_some());
            assert_eq!(acknowledged.frontier, actual);
            assert_eq!(metrics.hits.load(Ordering::Relaxed), 1);
            assert_eq!(metrics.saved_prefill_tokens.load(Ordering::Relaxed), 3);
            assert!(observer
                .record(PrefixRestoreDecision::NoReusableEntry)
                .is_err());
        }
    }
}
