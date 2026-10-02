use super::*;

#[path = "checkpoint_access_projection_tests.rs"]
mod checkpoint_access_projection_tests;

#[path = "checkpoint_access_guard_tests.rs"]
mod checkpoint_access_guard_tests;

fn observed_capture(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> SequenceCheckpoint<TestRuntime> {
    let mut checkpoint = access_capture(harness, lane, reaper);
    if let Some(publication) = checkpoint.take_publication_acknowledgement() {
        publication.acknowledge(&checkpoint).unwrap();
    }
    checkpoint
}

#[test]
fn checkpoint_statistical_domain_is_reusable_without_reusing_receipt_authority() {
    let harness = prefix_harness(Default::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let sink = observe_transfers(&reaper, 2);
    let first = observed_capture(&harness, &lane, &reaper);
    let second = observed_capture(&harness, &lane, &reaper);
    let (weak, foreign) = {
        let samples = sink.samples.lock().unwrap();
        assert_eq!(samples.len(), 2);
        assert!(!samples[0].identity().same_transfer(samples[1].identity()));
        let weak = samples[0].identity().downgrade();
        assert!(weak.matches(&samples[0].identity().clone()));
        assert!(!weak.matches(samples[1].identity()));
        assert!(!weak.is_expired());
        assert_eq!(samples[0].cost_domain(), samples[1].cost_domain());
        let mut changed = lane.descriptor().clone();
        changed
            .runtime_implementation_fingerprint
            .push_str(".changed");
        let byte_plan = harness
            .fixture
            .plan
            .checkpoint_byte_plan(first.completed_tokens() as u64)
            .unwrap();
        let projected = crate::vnext::NativeCheckpointTransferCostDomain::from_projection(
            &byte_plan,
            &changed,
            crate::vnext::NativeCheckpointTransferKind::Capture,
            *samples[0].geometry(),
        );
        assert_ne!(samples[0].cost_domain(), &projected);
        (weak, samples[1].identity().clone())
    };
    drop(first);
    drop(sink);
    assert!(
        weak.is_expired(),
        "a weak receipt must not retain the original transfer identity"
    );
    assert!(!weak.matches(&foreign));
    drop(weak);
    drop(foreign);
    drop(second);
    drop(reaper);
    drop(lane);
    harness.close();
}

pub(super) struct TransferObservationSink {
    capacity: usize,
    pub(super) samples: std::sync::Mutex<Vec<crate::vnext::NativeCheckpointTransferObservation>>,
}
impl crate::vnext::NativeCheckpointObservationSink for TransferObservationSink {
    fn try_record(&self, observation: crate::vnext::NativeCheckpointTransferObservation) -> bool {
        let Ok(mut samples) = self.samples.try_lock() else {
            return false;
        };
        if samples.len() == self.capacity {
            return false;
        }
        samples.push(observation);
        true
    }
}
pub(super) fn observe_transfers(
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    capacity: usize,
) -> Arc<TransferObservationSink> {
    let sink = Arc::new(TransferObservationSink {
        capacity,
        samples: std::sync::Mutex::new(Vec::with_capacity(capacity)),
    });
    let erased: Arc<dyn crate::vnext::NativeCheckpointObservationSink> = sink.clone();
    reaper
        .install_checkpoint_observation_sink(Arc::downgrade(&erased))
        .unwrap();
    sink
}

#[test]
fn checkpoint_transfer_observation_is_exact_and_restore_waits_for_acknowledgement() {
    use crate::vnext::{CheckpointTransferObservationStart, NativeCheckpointTransferKind};
    let harness = prefix_harness(Default::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let sink = observe_transfers(&reaper, 4);
    *harness.runtime.device_timing.lock().unwrap() =
        DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(37));
    let started = CheckpointTransferObservationStart::now();
    let mut transfer = access_submitted(
        reaper
            .try_capture_sequence_checkpoint_with_observation(
                &harness.fixture.plan,
                &harness.root.trusted_runtime_binding().unwrap(),
                Arc::clone(&harness.session),
                Arc::clone(&lane),
                DeviceTimingMode::Completion,
                started,
            )
            .unwrap(),
    );
    assert!(sink.samples.lock().unwrap().is_empty());
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Captured(mut checkpoint)) = transfer.take_result().unwrap()
    else {
        panic!("expected native capture");
    };
    assert!(
        sink.samples.lock().unwrap().is_empty(),
        "native outbox is not model publication"
    );
    checkpoint
        .take_publication_acknowledgement()
        .unwrap()
        .acknowledge(&checkpoint)
        .unwrap();
    let capture_identity = {
        let samples = sink.samples.lock().unwrap();
        assert_eq!(samples.len(), 1);
        let sample = &samples[0];
        assert_eq!(sample.kind(), NativeCheckpointTransferKind::Capture);
        assert_eq!(sample.identity().slot_id(), transfer.slot_id());
        assert_eq!(sample.identity().plan_hash(), checkpoint.plan_hash());
        assert_eq!(
            sample.identity().layout_fingerprint(),
            checkpoint.layout_fingerprint()
        );
        assert_eq!(sample.identity().lane_id(), lane.id());
        assert!(sample.source_capture_identity().is_none());
        assert_eq!(sample.geometry().copy_bytes(), checkpoint.logical_bytes());
        assert_eq!(sample.geometry().initialization_bytes(), 0);
        assert_eq!(sample.geometry().initialization_commands(), 0);
        let projected = crate::vnext::NativeCheckpointTransferCostDomain::from_projection(
            &harness
                .fixture
                .plan
                .checkpoint_byte_plan(checkpoint.completed_tokens() as u64)
                .unwrap(),
            lane.descriptor(),
            NativeCheckpointTransferKind::Capture,
            *sample.geometry(),
        );
        assert_eq!(sample.cost_domain(), &projected);
        assert_eq!(sample.device_timing().measured().unwrap().elapsed_ns(), 37);
        assert!(!sample.wall_elapsed().is_zero());
        sample.identity().clone()
    };
    drop(transfer);

    for outcome in 0..3 {
        let acknowledge = outcome == 2;
        let target = admitted_full_target(
            &harness,
            if acknowledge {
                "ack-target"
            } else if outcome == 1 {
                "cancel-target"
            } else {
                "drop-target"
            },
            &[19, 23],
        );
        harness.runtime.encoded_zero_bytes.lock().unwrap().clear();
        harness.runtime.encoded_copy_regions.lock().unwrap().clear();
        let mut transfer = access_submitted(
            reaper
                .try_restore_sequence_checkpoint_with_observation(
                    &harness.fixture.plan,
                    Arc::clone(&target),
                    &checkpoint,
                    Arc::from([19, 23]),
                    Arc::clone(&lane),
                    DeviceTimingMode::Completion,
                    CheckpointTransferObservationStart::now(),
                )
                .unwrap(),
        );
        assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
        let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
        else {
            panic!("expected gated native restore");
        };
        assert_eq!(sink.samples.lock().unwrap().len(), 1);
        assert_transfer_gate(&target);
        if acknowledge {
            publication.acknowledge().unwrap();
        } else if outcome == 1 {
            target.request_cancel().unwrap();
            assert!(publication.acknowledge().is_err());
        } else {
            drop(publication);
        }
        let samples = sink.samples.lock().unwrap();
        assert_eq!(samples.len(), if acknowledge { 2 } else { 1 });
        if acknowledge {
            let sample = &samples[1];
            assert_eq!(sample.kind(), NativeCheckpointTransferKind::Restore);
            assert!(!sample.identity().same_transfer(&capture_identity));
            assert!(sample
                .source_capture_identity()
                .unwrap()
                .same_transfer(&capture_identity));
            assert_eq!(sample.geometry().copy_bytes(), checkpoint.logical_bytes());
            assert_eq!(
                sample.geometry().copy_commands(),
                harness.runtime.encoded_copy_regions.lock().unwrap().len() as u64
            );
            let zeros = harness.runtime.encoded_zero_bytes.lock().unwrap();
            assert_eq!(
                sample.geometry().initialization_bytes(),
                zeros.iter().sum::<u64>()
            );
            assert_eq!(
                sample.geometry().initialization_commands(),
                zeros.len() as u64
            );
            assert!(sample.geometry().initialization_bytes() > 0);
        }
        drop(samples);
        drop(transfer);
        target.try_abort_if_quiescent().unwrap();
        drop(target);
    }
    drop(checkpoint);
    drop(sink);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_observation_full_or_contended_sink_never_changes_native_success() {
    for contended in [false, true] {
        let harness = prefix_harness(Default::default());
        let lane = harness.root.create_execution_lane().unwrap();
        let reaper = CompletionReaper::new();
        prove_prefix_source(&harness, &lane, &reaper);
        let sink = observe_transfers(&reaper, usize::from(contended));
        let held = contended.then(|| sink.samples.lock().unwrap());
        let checkpoint = observed_capture(&harness, &lane, &reaper);
        drop(held);
        assert!(sink.samples.lock().unwrap().is_empty());
        assert_eq!(reaper.retained_count(), 0);
        drop(checkpoint);
        drop(sink);
        drop(reaper);
        drop(lane);
        harness.close();
    }
}
