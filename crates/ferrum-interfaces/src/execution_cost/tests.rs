use super::*;

fn limits(waves: usize, rows: usize) -> CostRecorderLimits {
    CostRecorderLimits {
        max_waves: waves,
        max_rows_per_wave: rows,
        max_retained_rows: rows,
    }
}

fn recorder(waves: usize) -> BoundedWaveRecorder {
    BoundedWaveRecorder::new(NonZeroU64::new(7).unwrap(), limits(waves, 8)).unwrap()
}

fn decode() -> ActualWaveShape {
    ActualWaveShape {
        kind: ActualWaveKind::Decode,
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: [1; 32],
        output_policy_signature: [2; 32],
        numeric_features: None,
        host_content_features: None,
        row_multiset_features: None,
        rows: vec![ActualWaveRow {
            request_id: RequestId::new(),
            owner_incarnation: 4,
            work_generation: 9,
            input_index: 0,
            work: ActualRowWork::Decode { kv_tokens: 12 },
        }],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}

#[test]
fn recorder_charges_allocated_numeric_rows_in_addition_to_work_rows() {
    let mut shape = decode();
    shape.numeric_features = Some(CanonicalWaveCostFeatures {
        schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
        output_policy_signature: [3; 32],
        rows: vec![CostRowNumericFeatures {
            generated_tokens_before: 1,
            maximum_output_tokens: 8,
            sampling_history_tokens: 1,
            repetition_tokens: 0,
            decoded_prefix_tokens: 2,
            decoded_text_bytes_bound: 8,
            decode_scratch_bytes_bound: 0,
        }],
    });
    let mut recorder = BoundedWaveRecorder::new(NonZeroU64::new(8).unwrap(), limits(1, 1)).unwrap();
    assert_eq!(
        recorder
            .begin(
                shape.clone(),
                WaveObservationBoundary::IsolatedPreparationToCommit,
                0
            )
            .unwrap_err(),
        CostRecorderError::RowCapacity
    );
    let mut roomy = BoundedWaveRecorder::new(
        NonZeroU64::new(9).unwrap(),
        CostRecorderLimits {
            max_waves: 1,
            max_rows_per_wave: 1,
            max_retained_rows: 2,
        },
    )
    .unwrap();
    assert!(roomy
        .begin(
            shape.clone(),
            WaveObservationBoundary::IsolatedPreparationToCommit,
            0
        )
        .is_ok());
    shape.numeric_features.as_mut().unwrap().rows.reserve(8);
    let mut recorder = BoundedWaveRecorder::new(
        NonZeroU64::new(10).unwrap(),
        CostRecorderLimits {
            max_waves: 1,
            max_rows_per_wave: 1,
            max_retained_rows: 16,
        },
    )
    .unwrap();
    assert_eq!(
        recorder
            .begin(
                shape,
                WaveObservationBoundary::IsolatedPreparationToCommit,
                0
            )
            .unwrap_err(),
        CostRecorderError::RowCapacity
    );
}

#[test]
fn row_multiset_recorder_charges_static_storage_and_rejects_role_mismatch() {
    let mut shape = decode();
    shape.numeric_features = Some(CanonicalWaveCostFeatures {
        schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
        output_policy_signature: [3; 32],
        rows: vec![CostRowNumericFeatures {
            generated_tokens_before: 1,
            maximum_output_tokens: 8,
            sampling_history_tokens: 1,
            repetition_tokens: 0,
            decoded_prefix_tokens: 2,
            decoded_text_bytes_bound: 8,
            decode_scratch_bytes_bound: 0,
        }],
    });
    shape.row_multiset_features = Some(HostRowMultisetCostFeaturesV2 {
        schema_version: HOST_ROW_MULTISET_FEATURE_SCHEMA_V2,
        wave_policy_signature: [4; 32],
        rows: vec![HostRowStaticCostFeaturesV2 {
            role: HostRowRoleV2::Decode,
            categorical_signature: [5; 32],
        }],
    });
    assert_eq!(shape.validate(1), Ok(()));
    let make = |max_retained_rows| {
        BoundedWaveRecorder::new(
            NonZeroU64::new(11).unwrap(),
            CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 1,
                max_retained_rows,
            },
        )
        .unwrap()
    };
    assert_eq!(
        make(2)
            .begin(shape.clone(), WaveObservationBoundary::ExecutorOnly, 0)
            .unwrap_err(),
        CostRecorderError::RowCapacity
    );
    assert!(make(3)
        .begin(shape.clone(), WaveObservationBoundary::ExecutorOnly, 0)
        .is_ok());
    shape.row_multiset_features.as_mut().unwrap().rows[0].role = HostRowRoleV2::Prefill;
    assert_eq!(shape.validate(1), Err(CostRecorderError::InvalidShape));
    shape.row_multiset_features.as_mut().unwrap().rows[0].role = HostRowRoleV2::Decode;
    shape
        .row_multiset_features
        .as_mut()
        .unwrap()
        .rows
        .reserve(8);
    assert_eq!(
        make(32)
            .begin(shape, WaveObservationBoundary::ExecutorOnly, 0)
            .unwrap_err(),
        CostRecorderError::RowCapacity
    );
}

fn complete(
    recorder: &mut BoundedWaveRecorder,
    start: u64,
    boundary: WaveObservationBoundary,
) -> WaveObservationHandle {
    let handle = recorder.begin(decode(), boundary, start).unwrap();
    recorder.submission_started(&handle, start + 3).unwrap();
    recorder
        .terminal(
            &handle,
            ActualWaveOutcome::Completed,
            start + 8,
            NonZeroU64::new(4),
        )
        .unwrap();
    recorder.host_committed(&handle, start + 10).unwrap();
    handle
}

#[test]
fn complete_wall_includes_prepare_and_commit_not_just_device() {
    let mut recorder = recorder(1);
    let handle = complete(
        &mut recorder,
        20,
        WaveObservationBoundary::IsolatedPreparationToCommit,
    );
    assert_eq!(recorder.trainable_wall_ns(&handle).unwrap().get(), 10);
    assert_eq!(
        recorder.observations()[0].device_elapsed_ns.unwrap().get(),
        4
    );
    assert_eq!(recorder.coverage(), CostObservationCoverage::Complete);
}

#[test]
fn independent_recorders_reject_handles_even_with_identical_call_ids() {
    let mut first = recorder(1);
    let mut second = recorder(1);
    let handle = first
        .begin(decode(), WaveObservationBoundary::ExecutorOnly, 0)
        .unwrap();
    second
        .begin(decode(), WaveObservationBoundary::ExecutorOnly, 0)
        .unwrap();
    assert_eq!(
        second.submission_started(&handle, 1),
        Err(CostRecorderError::HandleMismatch)
    );
    assert!(second.observations()[0].submission_started_at_ns.is_none());
    assert!(matches!(
        second.coverage(),
        CostObservationCoverage::Unknown {
            reason: CostObservationUnknownReason::InvalidObservation,
            ..
        }
    ));
    first.submission_started(&handle, 1).unwrap();
}

#[test]
fn overflow_retains_existing_evidence_and_cannot_claim_full_coverage() {
    let mut recorder = recorder(1);
    let handle = complete(
        &mut recorder,
        0,
        WaveObservationBoundary::IsolatedPreparationToCommit,
    );
    assert_eq!(
        recorder
            .begin(decode(), WaveObservationBoundary::ExecutorOnly, 11)
            .unwrap_err(),
        CostRecorderError::WaveCapacity
    );
    assert_eq!(recorder.observations().len(), 1);
    assert_eq!(
        recorder.trainable_wall_ns(&handle),
        Err(CostObservationUnknownReason::LostObservations)
    );
    recorder.note_lost(u64::MAX);
    recorder.note_lost(1);
    assert_eq!(
        recorder.coverage(),
        CostObservationCoverage::Unknown {
            reason: CostObservationUnknownReason::LostObservations,
            lost_observations: u64::MAX
        }
    );
}

#[test]
fn total_row_retention_is_independent_of_wave_limit() {
    let mut recorder = BoundedWaveRecorder::new(NonZeroU64::new(1).unwrap(), limits(3, 1)).unwrap();
    complete(&mut recorder, 0, WaveObservationBoundary::ExecutorOnly);
    assert_eq!(
        recorder
            .begin(decode(), WaveObservationBoundary::ExecutorOnly, 11)
            .unwrap_err(),
        CostRecorderError::RowCapacity
    );
    assert_eq!(recorder.observations().len(), 1);
}

#[test]
fn spare_vector_capacity_cannot_bypass_retention_limit() {
    let mut recorder = BoundedWaveRecorder::new(NonZeroU64::new(1).unwrap(), limits(3, 1)).unwrap();
    let mut shape = decode();
    shape.rows.reserve(64);
    assert_eq!(
        recorder
            .begin(shape, WaveObservationBoundary::ExecutorOnly, 0)
            .unwrap_err(),
        CostRecorderError::RowCapacity
    );
    recorder
        .begin(decode(), WaveObservationBoundary::ExecutorOnly, 1)
        .unwrap();
    assert_eq!(recorder.observations()[0].physical_wave_ordinal, 1);
    assert!(matches!(
        recorder.coverage(),
        CostObservationCoverage::Unknown {
            reason: CostObservationUnknownReason::LostObservations,
            ..
        }
    ));
}

#[test]
fn cancellation_or_missing_terminal_remains_incomplete() {
    let mut recorder = recorder(1);
    let handle = recorder
        .begin(
            decode(),
            WaveObservationBoundary::IsolatedPreparationToCommit,
            0,
        )
        .unwrap();
    recorder.submission_started(&handle, 1).unwrap();
    assert_eq!(
        recorder.trainable_wall_ns(&handle),
        Err(CostObservationUnknownReason::IncompleteObservation)
    );
    recorder
        .terminal(&handle, ActualWaveOutcome::Completed, 5, None)
        .unwrap();
    assert_eq!(
        recorder.trainable_wall_ns(&handle),
        Err(CostObservationUnknownReason::IncompleteObservation)
    );
    recorder.host_committed(&handle, 6).unwrap();
    assert_eq!(recorder.trainable_wall_ns(&handle).unwrap().get(), 6);
}

#[test]
fn no_submit_and_failures_are_not_zero_cost_successes() {
    for outcome in [
        ActualWaveOutcome::NotSubmitted,
        ActualWaveOutcome::Deferred,
        ActualWaveOutcome::FailedAfterSubmit,
        ActualWaveOutcome::SubmissionIndeterminate,
        ActualWaveOutcome::PartiallyCompleted,
    ] {
        let mut recorder = recorder(1);
        let handle = recorder
            .begin(
                decode(),
                WaveObservationBoundary::IsolatedPreparationToCommit,
                0,
            )
            .unwrap();
        if !matches!(
            outcome,
            ActualWaveOutcome::NotSubmitted | ActualWaveOutcome::Deferred
        ) {
            recorder.submission_started(&handle, 1).unwrap();
        }
        recorder.terminal(&handle, outcome, 5, None).unwrap();
        assert_eq!(recorder.coverage(), CostObservationCoverage::Complete);
        assert_eq!(
            recorder.trainable_wall_ns(&handle),
            Err(CostObservationUnknownReason::OutcomeNotCompleted)
        );
    }
}

#[test]
fn definitely_not_submitted_can_follow_an_attempt_but_cannot_report_device_work() {
    let mut recorder = recorder(1);
    let handle = recorder
        .begin(decode(), WaveObservationBoundary::ExecutorOnly, 0)
        .unwrap();
    recorder.submission_started(&handle, 1).unwrap();
    assert_eq!(
        recorder.terminal(
            &handle,
            ActualWaveOutcome::NotSubmitted,
            5,
            NonZeroU64::new(1)
        ),
        Err(CostRecorderError::InvalidTransition)
    );
    assert!(recorder.observations()[0].outcome.is_none());
}

#[test]
fn duplicate_terminal_or_clock_reversal_cannot_repair_invalid_evidence() {
    let mut recorder = recorder(1);
    let handle = complete(
        &mut recorder,
        10,
        WaveObservationBoundary::IsolatedPreparationToCommit,
    );
    assert_eq!(
        recorder.terminal(&handle, ActualWaveOutcome::Completed, 30, None),
        Err(CostRecorderError::InvalidTransition)
    );
    assert_eq!(
        recorder.trainable_wall_ns(&handle),
        Err(CostObservationUnknownReason::InvalidObservation)
    );
    let mut reversed = self::recorder(1);
    let handle = reversed
        .begin(decode(), WaveObservationBoundary::ExecutorOnly, 10)
        .unwrap();
    assert_eq!(
        reversed.submission_started(&handle, 9),
        Err(CostRecorderError::InvalidTiming)
    );
    assert!(reversed.observations()[0]
        .submission_started_at_ns
        .is_none());
}

#[test]
fn overlapping_sibling_commits_do_not_train_as_isolated_waves() {
    let mut recorder = recorder(2);
    let first = recorder
        .begin(
            decode(),
            WaveObservationBoundary::IsolatedPreparationToCommit,
            0,
        )
        .unwrap();
    recorder.submission_started(&first, 1).unwrap();
    recorder
        .terminal(&first, ActualWaveOutcome::Completed, 3, None)
        .unwrap();
    let second = complete(
        &mut recorder,
        4,
        WaveObservationBoundary::IsolatedPreparationToCommit,
    );
    recorder.host_committed(&first, 15).unwrap();
    assert_eq!(
        recorder.trainable_wall_ns(&first),
        Err(CostObservationUnknownReason::OverlappingSiblingWave)
    );
    assert_eq!(
        recorder.trainable_wall_ns(&second),
        Err(CostObservationUnknownReason::OverlappingSiblingWave)
    );
}

#[test]
fn serial_children_preserve_actual_fallback_shapes_without_outer_total() {
    let mut recorder = recorder(3);
    for start in [0, 10, 20] {
        let mut shape = decode();
        shape.path = ActualWavePath::CapacityFallback;
        let handle = recorder
            .begin(
                shape,
                WaveObservationBoundary::IsolatedPreparationToCommit,
                start,
            )
            .unwrap();
        recorder.submission_started(&handle, start + 1).unwrap();
        recorder
            .terminal(&handle, ActualWaveOutcome::Completed, start + 8, None)
            .unwrap();
        recorder.host_committed(&handle, start + 10).unwrap();
        assert_eq!(recorder.trainable_wall_ns(&handle).unwrap().get(), 10);
    }
    assert_eq!(recorder.observations().len(), 3);
    assert!(recorder
        .observations()
        .iter()
        .all(|wave| wave.shape.as_ref().unwrap().path == ActualWavePath::CapacityFallback));
}

#[test]
fn explicit_composite_boundary_is_never_an_isolated_sample() {
    let mut recorder = recorder(1);
    let handle = complete(
        &mut recorder,
        0,
        WaveObservationBoundary::CompositeDeferredCommit,
    );
    assert_eq!(
        recorder.trainable_wall_ns(&handle),
        Err(CostObservationUnknownReason::BoundaryNotIsolated)
    );
}

#[test]
fn duplicate_rows_overflowing_prefill_and_wrong_kind_are_rejected() {
    let mut shape = decode();
    shape.rows.push(shape.rows[0].clone());
    assert_eq!(shape.validate(2), Err(CostRecorderError::InvalidShape));
    shape.rows.pop();
    shape.kind = ActualWaveKind::Prefill;
    shape.rows[0].work = ActualRowWork::Prefill {
        offset: u32::MAX,
        count: 2,
        total_prompt_tokens: u32::MAX,
    };
    assert_eq!(shape.validate(1), Err(CostRecorderError::InvalidShape));
    shape.rows[0].work = ActualRowWork::Prefill {
        offset: 4,
        count: 2,
        total_prompt_tokens: 6,
    };
    assert_eq!(shape.validate(1), Ok(()));
    shape.kind = ActualWaveKind::Mixed;
    assert_eq!(shape.validate(1), Err(CostRecorderError::InvalidShape));
}

#[test]
fn absent_device_timing_is_not_synthesized_and_impossible_duration_is_rejected() {
    let mut recorder = recorder(1);
    let handle = recorder
        .begin(decode(), WaveObservationBoundary::ExecutorOnly, 0)
        .unwrap();
    recorder.submission_started(&handle, 5).unwrap();
    assert_eq!(
        recorder.terminal(&handle, ActualWaveOutcome::Completed, 8, NonZeroU64::new(4)),
        Err(CostRecorderError::InvalidTiming)
    );
    assert!(recorder.observations()[0].device_elapsed_ns.is_none());
}

#[test]
fn retention_limits_have_hard_bounds_and_empty_trace_is_unknown() {
    for bad in [
        limits(0, 1),
        limits(4097, 1),
        limits(1, 1025),
        CostRecorderLimits {
            max_waves: 1,
            max_rows_per_wave: 2,
            max_retained_rows: 1,
        },
    ] {
        assert!(matches!(
            BoundedWaveRecorder::new(NonZeroU64::new(1).unwrap(), bad),
            Err(CostRecorderError::InvalidLimits)
        ));
    }
    assert!(matches!(
        recorder(1).coverage(),
        CostObservationCoverage::Unknown {
            reason: CostObservationUnknownReason::NoObservations,
            ..
        }
    ));
}

#[test]
fn missing_hardware_retains_three_digests_but_cannot_be_known() {
    let partial = ExecutorCostIdentityComponents {
        model_weights: Some([1; 32]),
        numerical_policy: Some([2; 32]),
        device_runtime: None,
        execution_config: Some([3; 32]),
    };
    match partial.clone().into_availability() {
        ExecutorCostIdentityAvailability::Unknown {
            reason,
            partial: Some(cached),
        } => {
            assert_eq!(reason, CostIdentityUnknownReason::MissingHardwareIdentity);
            assert_eq!(*cached, partial);
        }
        other => panic!("unexpected availability: {other:?}"),
    }
    let known = ExecutorCostIdentityComponents {
        device_runtime: Some([4; 32]),
        ..partial
    }
    .into_availability();
    assert!(matches!(known, ExecutorCostIdentityAvailability::Known(_)));
}

struct TestClock(std::sync::atomic::AtomicU64);
impl CostObservationClock for TestClock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }
}
fn participant(shape: &ActualWaveShape) -> CostObservationParticipant {
    let row = &shape.rows[0];
    CostObservationParticipant {
        request_id: row.request_id.clone(),
        owner_incarnation: row.owner_incarnation,
        work_generation: row.work_generation,
        input_index: row.input_index,
        output_policy_signature: Some([2; 32]),
        host_features: None,
    }
}

#[test]
fn context_requires_host_commit_after_real_terminal() {
    let mut recorder = recorder(1);
    let shape = decode();
    let participants = [participant(&shape)];
    let clock = TestClock(std::sync::atomic::AtomicU64::new(8));
    let handle = {
        let mut context = PlanRuntimeCostObservationContext::new(
            &mut recorder,
            &clock,
            &participants,
            Some(1),
            WaveObservationBoundary::IsolatedPreparationToCommit,
        );
        context.physical_wave(Ok(shape), Some(3));
        context.terminal(ActualWaveOutcome::Completed, None);
        context.finish_call(ObservedCallOutcome::Completed);
        assert_eq!(context.physical_wave_count(), 1);
        context.wave_handle().unwrap()
    };
    assert_eq!(
        recorder.trainable_wall_ns(&handle),
        Err(CostObservationUnknownReason::IncompleteObservation)
    );
    recorder.host_committed(&handle, 11).unwrap();
    assert_eq!(recorder.trainable_wall_ns(&handle).unwrap().get(), 10);
}

#[test]
fn context_no_submission_has_no_zero_cost_wave() {
    for outcome in [
        ObservedCallOutcome::Unsupported,
        ObservedCallOutcome::NotSubmitted,
        ObservedCallOutcome::Deferred,
    ] {
        let mut recorder = recorder(1);
        let clock = TestClock(std::sync::atomic::AtomicU64::new(8));
        {
            let mut context = PlanRuntimeCostObservationContext::new(
                &mut recorder,
                &clock,
                &[],
                Some(1),
                WaveObservationBoundary::ExecutorOnly,
            );
            context.finish_call(outcome);
            assert_eq!(context.call_outcome(), Some(outcome));
            assert_eq!(context.physical_wave_count(), 0);
        }
        assert!(recorder.observations().is_empty());
        assert!(matches!(
            recorder.coverage(),
            CostObservationCoverage::Unknown {
                reason: CostObservationUnknownReason::NoObservations,
                ..
            }
        ));
    }
}

#[test]
fn context_unknown_shape_retains_real_failure_without_training() {
    let mut recorder = recorder(1);
    let clock = TestClock(std::sync::atomic::AtomicU64::new(8));
    let handle = {
        let mut context = PlanRuntimeCostObservationContext::new(
            &mut recorder,
            &clock,
            &[],
            Some(1),
            WaveObservationBoundary::ExecutorOnly,
        );
        context.physical_wave(Err(ActualWaveEvidenceUnknown::GraphPath), Some(2));
        context.terminal(ActualWaveOutcome::SubmissionIndeterminate, None);
        context.finish_call(ObservedCallOutcome::Failed);
        context.wave_handle().unwrap()
    };
    assert!(recorder.observations()[0].shape.is_none());
    assert_eq!(
        recorder.observations()[0].outcome,
        Some(ActualWaveOutcome::SubmissionIndeterminate)
    );
    assert_eq!(
        recorder.trainable_wall_ns(&handle),
        Err(CostObservationUnknownReason::EvidenceUnavailable(
            ActualWaveEvidenceUnknown::GraphPath
        ))
    );
}

#[test]
fn context_siblings_downgrade_boundaries_and_overflow_cannot_reuse_old_handle() {
    for capacity in [1, 2] {
        let mut recorder = recorder(capacity);
        let shape = decode();
        let participants = [participant(&shape)];
        let clock = TestClock(std::sync::atomic::AtomicU64::new(8));
        {
            let mut context = PlanRuntimeCostObservationContext::new(
                &mut recorder,
                &clock,
                &participants,
                Some(1),
                WaveObservationBoundary::IsolatedPreparationToCommit,
            );
            context.physical_wave(Ok(shape.clone()), Some(2));
            context.terminal(ActualWaveOutcome::Completed, None);
            context.physical_wave(Ok(shape), Some(9));
            clock.0.store(12, std::sync::atomic::Ordering::Relaxed);
            context.terminal(ActualWaveOutcome::FailedAfterSubmit, None);
            assert_eq!(context.wave_handle().is_some(), capacity == 2);
        }
        assert_eq!(
            recorder.observations()[0].outcome,
            Some(ActualWaveOutcome::Completed)
        );
        assert!(recorder
            .observations()
            .iter()
            .all(|wave| wave.boundary == WaveObservationBoundary::CompositeDeferredCommit));
    }
}

#[test]
fn context_correlation_mismatch_and_cancellation_stay_unknown() {
    let mut recorder = recorder(1);
    let mut shape = decode();
    let participants = [participant(&shape)];
    shape.rows[0].work_generation += 1;
    let clock = TestClock(std::sync::atomic::AtomicU64::new(8));
    {
        let mut context = PlanRuntimeCostObservationContext::new(
            &mut recorder,
            &clock,
            &participants,
            Some(1),
            WaveObservationBoundary::IsolatedPreparationToCommit,
        );
        context.physical_wave(Ok(shape), Some(2));
        assert_eq!(
            context.unknown_reason(),
            Some(ActualWaveEvidenceUnknown::ParticipantCorrelation)
        );
        // Dropping an in-flight future must not fabricate a terminal outcome.
    }
    let wave = &recorder.observations()[0];
    assert_eq!(
        wave.shape_unknown,
        Some(ActualWaveEvidenceUnknown::ParticipantCorrelation)
    );
    assert!(wave.terminal_at_ns.is_none());
    assert!(wave.outcome.is_none());
}
