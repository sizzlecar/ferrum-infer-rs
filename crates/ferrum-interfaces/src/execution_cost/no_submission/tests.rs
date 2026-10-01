use super::*;
use crate::model_executor::{ExecutorExecutionCapacityDeferral, ExecutorExecutionCapacityStage};
use crate::vnext::{
    CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
    DeviceCapacityPressure, DeviceCapacityPressureScope,
};
use crate::ExecutorAdmissionEpochs;

struct Clock(Option<u64>);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        self.0
    }
}
fn recorder() -> BoundedWaveRecorder {
    BoundedWaveRecorder::new(
        NonZeroU64::new(17).unwrap(),
        CostRecorderLimits {
            max_waves: 2,
            max_rows_per_wave: 8,
            max_retained_rows: 16,
        },
    )
    .unwrap()
}
fn participants() -> Vec<CostObservationParticipant> {
    (0..2)
        .map(|input_index| CostObservationParticipant {
            request_id: RequestId::new(),
            owner_incarnation: 4,
            work_generation: 7,
            input_index,
            output_policy_signature: None,
            host_features: None,
        })
        .collect()
}
fn deferral(
    stage: ExecutorExecutionCapacityStage,
) -> crate::model_executor::ExecutorExecutionDeferral {
    let observed =
        CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 7).unwrap();
    ExecutorExecutionCapacityDeferral::from_backing_pressure(
        ExecutorAdmissionEpochs::new(NonZeroU64::new(19).unwrap(), 3, 5),
        CapacityWaitCondition::from_observation(19, vec![observed]).unwrap(),
        DeviceCapacityPressure::new(
            DeviceCapacityPressureScope::PlanBudget,
            "no-submission-proof".to_owned(),
            1,
            1,
            1,
            1,
            1,
        )
        .unwrap()
        .into(),
        stage,
    )
    .unwrap()
    .into()
}
fn finish(
    recorder: &mut BoundedWaveRecorder,
    rows: &[CostObservationParticipant],
    stage: ExecutorExecutionCapacityStage,
) {
    let mut context = PlanRuntimeCostObservationContext::new(
        recorder,
        &Clock(Some(120)),
        rows,
        Some(100),
        WaveObservationBoundary::IsolatedPreparationToCommit,
    );
    context.finish_capacity_deferred(&deferral(stage));
}
fn shape(rows: &[CostObservationParticipant]) -> ActualWaveShape {
    ActualWaveShape {
        statistical_evidence: None,
        kind: ActualWaveKind::Decode,
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: [1; 32],
        output_policy_signature: [2; 32],
        numeric_features: None,
        host_content_features: None,
        row_multiset_features: None,
        rows: rows
            .iter()
            .map(|p| ActualWaveRow {
                request_id: p.request_id.clone(),
                owner_incarnation: p.owner_incarnation,
                work_generation: p.work_generation,
                input_index: p.input_index,
                work: ActualRowWork::Decode { kv_tokens: 12 },
            })
            .collect(),
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}

#[test]
fn no_submission_receipt_binds_real_capacity_stage_call_clock_and_original_frontier() {
    for stage in [
        ExecutorExecutionCapacityStage::SequenceExtension,
        ExecutorExecutionCapacityStage::StepAdmission,
        ExecutorExecutionCapacityStage::SubmissionWave,
    ] {
        let rows = participants();
        let mut recorder = recorder();
        finish(&mut recorder, &rows, stage);
        let proof = *recorder.no_submission().unwrap();
        assert_eq!(
            (
                proof.call_id(),
                proof.prepare_started_at_ns(),
                proof.returned_at_ns()
            ),
            (17, 100, 120)
        );
        assert!(proof.matches_participants(&rows));
        let wire = rows
            .iter()
            .map(CallNoSubmissionParticipantV1::from)
            .collect::<Vec<_>>();
        assert_eq!(
            no_submission_participant_signature(&wire),
            Some(*proof.participant_signature())
        );
        let mut changed = rows.clone();
        changed[0].work_generation += 1;
        assert!(!proof.matches_participants(&changed));
        changed = rows.clone();
        changed.reverse();
        assert!(!proof.matches_participants(&changed));
        let frozen = recorder.take_frozen();
        assert_eq!(frozen.no_submission(), Some(&proof));
        assert!(recorder.no_submission().is_none());
        assert!(frozen.observations().is_empty());
        assert!(matches!(
            frozen.coverage(),
            CostObservationCoverage::Unknown {
                reason: CostObservationUnknownReason::NoObservations,
                lost_observations: 0
            }
        ));
    }
}

#[test]
fn no_submission_generic_deferred_and_missing_bad_clock_never_mint() {
    let rows = participants();
    let mut r = recorder();
    PlanRuntimeCostObservationContext::new(
        &mut r,
        &Clock(Some(120)),
        &rows,
        Some(100),
        WaveObservationBoundary::IsolatedPreparationToCommit,
    )
    .finish_call(ObservedCallOutcome::Deferred);
    assert!(r.no_submission().is_none());
    for clock in [None, Some(99)] {
        let mut r = recorder();
        PlanRuntimeCostObservationContext::new(
            &mut r,
            &Clock(clock),
            &rows,
            Some(100),
            WaveObservationBoundary::IsolatedPreparationToCommit,
        )
        .finish_capacity_deferred(&deferral(ExecutorExecutionCapacityStage::StepAdmission));
        assert!(r.no_submission().is_none());
    }
}

#[test]
fn no_submission_receipt_cannot_cover_physical_work_loss_or_unknown() {
    let rows = participants();
    for fault in 0..3 {
        let mut r = recorder();
        match fault {
            0 => {
                let h = r
                    .begin(
                        shape(&rows),
                        WaveObservationBoundary::IsolatedPreparationToCommit,
                        100,
                    )
                    .unwrap();
                r.submission_started(&h, 105).unwrap();
                r.terminal(&h, ActualWaveOutcome::Completed, 110, None)
                    .unwrap();
            }
            1 => r.note_lost(1),
            _ => r.note_unknown_evidence(ActualWaveEvidenceUnknown::ProviderPath),
        }
        finish(&mut r, &rows, ExecutorExecutionCapacityStage::StepAdmission);
        assert!(r.no_submission().is_none());
    }
    let mut r = recorder();
    finish(
        &mut r,
        &rows,
        ExecutorExecutionCapacityStage::SubmissionWave,
    );
    assert!(r.no_submission().is_some());
    assert_eq!(
        r.begin(
            shape(&rows),
            WaveObservationBoundary::IsolatedPreparationToCommit,
            121
        )
        .unwrap_err(),
        CostRecorderError::InvalidTransition
    );
    assert!(r.no_submission().is_none());
}

#[test]
fn no_submission_duplicate_completion_or_invalid_participants_invalidate_receipt() {
    let rows = participants();
    let mut r = recorder();
    let mut c = PlanRuntimeCostObservationContext::new(
        &mut r,
        &Clock(Some(120)),
        &rows,
        Some(100),
        WaveObservationBoundary::IsolatedPreparationToCommit,
    );
    c.finish_capacity_deferred(&deferral(ExecutorExecutionCapacityStage::StepAdmission));
    c.finish_call(ObservedCallOutcome::Deferred);
    drop(c);
    assert!(r.no_submission().is_none());
    for fault in 0..4 {
        let mut rows = participants();
        match fault {
            0 => rows[0].owner_incarnation = 0,
            1 => rows[0].work_generation = 0,
            2 => rows[1].request_id = rows[0].request_id.clone(),
            _ => rows[1].input_index = rows[0].input_index,
        }
        let mut r = recorder();
        finish(&mut r, &rows, ExecutorExecutionCapacityStage::StepAdmission);
        assert!(r.no_submission().is_none());
    }
}

#[test]
fn no_submission_not_submitted_batch_preserves_its_control_outcome() {
    let rows = participants();
    let mut r = recorder();
    let mut c = PlanRuntimeCostObservationContext::new(
        &mut r,
        &Clock(Some(120)),
        &rows,
        Some(100),
        WaveObservationBoundary::IsolatedPreparationToCommit,
    );
    c.finish_capacity_not_submitted(&deferral(ExecutorExecutionCapacityStage::SubmissionWave));
    assert_eq!(c.call_outcome(), Some(ObservedCallOutcome::NotSubmitted));
    drop(c);
    assert!(r.no_submission().is_some());
}

#[test]
fn no_submission_passive_diagnostics_do_not_change_real_lifecycle_gates() {
    let rows = participants();
    for diagnostics in [false, true] {
        for fault in 0..5 {
            let mut recorder = recorder();
            if diagnostics {
                recorder.enable_route_diagnostics();
            }
            let mut context = PlanRuntimeCostObservationContext::new(
                &mut recorder,
                &Clock(Some(120)),
                &rows,
                Some(100),
                WaveObservationBoundary::IsolatedPreparationToCommit,
            );
            match fault {
                0 => {}
                1 => context.route_submission(None),
                2 => context.mark_unknown(ActualWaveEvidenceUnknown::ProviderPath),
                3 => context.physical_wave(Ok(shape(&rows)), Some(110)),
                _ => context.finish_call(ObservedCallOutcome::Deferred),
            }
            context
                .finish_capacity_deferred(&deferral(ExecutorExecutionCapacityStage::StepAdmission));
            drop(context);
            assert_eq!(recorder.no_submission().is_some(), fault == 0);
            assert_eq!(recorder.route_diagnostic().is_some(), diagnostics);
        }
    }
}
