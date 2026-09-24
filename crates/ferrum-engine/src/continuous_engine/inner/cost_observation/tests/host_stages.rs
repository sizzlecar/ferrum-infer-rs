use super::*;
use ferrum_interfaces::model_executor::ExecutorCompletionWork;
use ferrum_types::{FinishReason, InferenceRequest, TokenId};

fn owner(row: &ActualWaveRow) -> crate::continuous_engine::SequenceState {
    let mut request = InferenceRequest::new("input", "model");
    request.id = row.request_id.clone();
    let mut sequence = crate::continuous_engine::SequenceState::new(request, vec![TokenId::new(1)]);
    sequence.cost_frontier = Some(CostFrontier {
        owner_incarnation: NonZeroU64::new(row.owner_incarnation).unwrap(),
        work_generation: NonZeroU64::new(row.work_generation + 1).unwrap(),
    });
    sequence
}

fn terminal() -> HostTerminalStageV1 {
    HostTerminalStageV1 {
        finish_reason: FinishReason::Length,
        generated_tokens: 3,
        through_output_ordinal: 3,
        output_failed: false,
        physical_failed: false,
        scheduler_failed: false,
        terminal_handoff_succeeded: true,
        pending_restore_removed: false,
        admission_cancellation_work: ExecutorCompletionWork::NoAdditionalWork,
        cache_completion_work: ExecutorCompletionWork::NoAdditionalWork,
        other_physical_resources: false,
        request_slot_closed: true,
        owner_matched: true,
    }
}

#[test]
fn host_settled_wall_preserves_actual_mixed_host_order_and_old_composite_rejection() {
    let mut shape = shape(&[
        ActualRowWork::Prefill {
            offset: 0,
            count: 2,
            total_prompt_tokens: 2,
        },
        ActualRowWork::Decode { kv_tokens: 7 },
    ]);
    shape.numeric_features = Some(CanonicalWaveCostFeatures {
        schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
        output_policy_signature: [31; 32],
        rows: [0, 2]
            .map(|generated| CostRowNumericFeatures {
                generated_tokens_before: generated,
                maximum_output_tokens: 4,
                sampling_history_tokens: generated,
                repetition_tokens: 0,
                decoded_prefix_tokens: generated + 1,
                decoded_text_bytes_bound: (generated + 1) * 4,
                decode_scratch_bytes_bound: (generated + 1) * 8,
            })
            .to_vec(),
    });
    shape.row_multiset_features = Some(HostRowMultisetCostFeaturesV2 {
        schema_version: HOST_ROW_MULTISET_FEATURE_SCHEMA_V2,
        wave_policy_signature: [32; 32],
        rows: vec![
            HostRowStaticCostFeaturesV2 {
                role: HostRowRoleV2::Prefill,
                categorical_signature: [33; 32],
            },
            HostRowStaticCostFeaturesV2 {
                role: HostRowRoleV2::Decode,
                categorical_signature: [34; 32],
            },
        ],
    });
    let sink = sink(4, 32);
    let capture = Arc::new(CostCalibrationCapture::default());
    let (mut call, clock) = begin(&shape, &sink);
    call.attach_calibration_capture(Arc::clone(&capture));
    execute(&mut call, &clock, shape.clone());
    for (ordinal, index) in [1, 0].into_iter().enumerate() {
        let row = &shape.rows[index];
        clock.set(10 + ordinal as u64 * 10);
        call.begin_host_row(&row.request_id);
        clock.set(11 + ordinal as u64 * 10);
        let evidence = committed(row, 0);
        call.note_host_token_commit(&evidence);
        let mut pending = call.host_publication(&evidence, true, true).unwrap();
        assert!(pending.matches_owner(&owner(row)));
        clock.set(15 + ordinal as u64 * 10);
        pending.terminal_handed_off();
        let mut terminal = terminal();
        if matches!(row.work, ActualRowWork::Prefill { .. }) {
            terminal.generated_tokens = 1;
        }
        call.record_settled(pending.settle(owner(row), terminal));
    }
    call.reject(CostCallRejection::Composite);
    clock.set(30);
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Composite)
    );
    let stages = capture.host_stages().unwrap();
    assert_eq!(
        stages.actual_shape.as_ref().unwrap().row_multiset_features,
        shape.row_multiset_features
    );
    assert_eq!(
        stages.actual_shape.as_ref().unwrap().numeric_features,
        shape.numeric_features
    );
    let receipt = capture.host_stage_queue().unwrap();
    assert_eq!(receipt.accepted_ordinal, Some(1));
    assert_eq!(receipt.disposition, HostStageQueueDisposition::Published);
    assert_eq!(stages.rows[0].host_processing_ordinal, Some(1));
    assert_eq!(stages.rows[1].host_processing_ordinal, Some(0));
    assert_eq!(stages.full_wall_ns, Some(24));
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    assert_eq!(stages.rows[0].settled_at_ns, Some(25));
    assert_eq!(sink.stats().rejected(CostCallRejection::Composite), 1);
}

#[test]
fn unknown_cleanup_failed_terminal_and_cancelled_owner_never_seal_a_full_wall() {
    for failure in 0..4 {
        let shape = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
        let sink = sink(4, 32);
        let (mut call, clock) = begin(&shape, &sink);
        execute(&mut call, &clock, shape.clone());
        clock.set(10);
        let row = &shape.rows[0];
        call.begin_host_row(&row.request_id);
        let evidence = committed(row, 0);
        call.note_host_token_commit(&evidence);
        let mut pending = call.host_publication(&evidence, true, true).unwrap();
        let mut result = terminal();
        match failure {
            0 => result.cache_completion_work = ExecutorCompletionWork::Unknown,
            1 => result.pending_restore_removed = true,
            2 => result.output_failed = true,
            _ => result.terminal_handoff_succeeded = false,
        }
        clock.set(20);
        pending.terminal_handed_off();
        call.record_settled(pending.settle(owner(row), result));
        let stages = call.make_host_stages().unwrap();
        assert_ne!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert_eq!(stages.full_wall_ns, None);
        assert!(stages.rows[0].terminal.is_some());
    }
}

#[test]
fn host_receipt_cannot_cross_calls_even_when_public_call_ids_and_rows_match() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
    let sink = sink(4, 32);
    let (mut first, clock) = begin(&shape, &sink);
    let (mut second, second_clock) = begin(&shape, &sink);
    assert_eq!(first.call_id, second.call_id); // distinct EngineCostIds instances
    execute(&mut first, &clock, shape.clone());
    execute(&mut second, &second_clock, shape.clone());
    let row = &shape.rows[0];
    clock.set(10);
    first.begin_host_row(&row.request_id);
    let evidence = committed(row, 0);
    first.note_host_token_commit(&evidence);
    let pending = first.host_publication(&evidence, true, true).unwrap();
    let mut replacement = owner(row);
    replacement
        .cost_frontier
        .as_mut()
        .unwrap()
        .owner_incarnation = NonZeroU64::new(999).unwrap();
    assert!(!pending.matches_owner(&replacement));
    clock.set(20);
    second.record_settled(pending.settle(owner(row), terminal()));
    assert_eq!(second.rejection, Some(CostCallRejection::FrontierMismatch));
    // The original pending footprint survives loss/cancellation of its receipt.
    assert!(first.make_host_stages().unwrap().full_wall_ns.is_none());
}

#[test]
fn partial_has_no_actor_handoff_and_late_rejection_stays_invalid_after_composite() {
    let shape = shape(&[ActualRowWork::Prefill {
        offset: 0,
        count: 2,
        total_prompt_tokens: 4,
    }]);
    let sink = sink(4, 32);
    let (mut call, clock) = begin(&shape, &sink);
    execute(&mut call, &clock, shape.clone());
    clock.set(10);
    let row = &shape.rows[0];
    call.begin_host_row(&row.request_id);
    let evidence = committed(row, 0);
    call.note_host_token_commit(&evidence);
    call.host_publication(&evidence, false, false);
    let stages = call.make_host_stages().unwrap();
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    assert_eq!(stages.rows[0].output_published_at_ns, None);
    assert_eq!(stages.full_wall_ns, Some(9));
    call.reject(CostCallRejection::Composite);
    call.reject(CostCallRejection::HostDuplicate);
    assert_eq!(call.rejection, Some(CostCallRejection::Composite));
    assert_eq!(call.make_host_stages().unwrap().full_wall_ns, None);
}

#[test]
fn auxiliary_drop_has_no_accepted_ordinal_and_missing_physical_row_cannot_be_complete() {
    let shape = shape(&[
        ActualRowWork::Decode { kv_tokens: 7 },
        ActualRowWork::Decode { kv_tokens: 9 },
    ]);
    let sink = sink(1, 32);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    );
    let capture = Arc::new(CostCalibrationCapture::default());
    let (mut call, clock) = begin(&shape, &sink);
    call.attach_calibration_capture(Arc::clone(&capture));
    let mut omitted = shape.clone();
    omitted.rows.pop();
    execute(&mut call, &clock, omitted);
    clock.set(10);
    let row = &shape.rows[0];
    call.begin_host_row(&row.request_id);
    let evidence = committed(row, 10);
    call.note_host_token_commit(&evidence);
    call.host_publication(&evidence, false, true);
    assert_eq!(call.make_host_stages().unwrap().full_wall_ns, None);
    call.reject(CostCallRejection::Composite);
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Composite)
    );
    assert_eq!(
        capture.host_stage_queue(),
        Some(HostStageQueueReceipt {
            accepted_ordinal: None,
            disposition: HostStageQueueDisposition::DroppedCapacity,
        })
    );
}
