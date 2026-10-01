//! Actual installed-policy settlements remain comparable after publication.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    profile::EngineCostSnapshot, selected_feedback::FeedbackObservation, structured_epoch,
    structured_feedback, trainer::structured_v2,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2;
use ferrum_types::{
    SloSelectedFeedbackSettingsV1, SloSelectedFeedbackStorageV1, SloStructuredFeedbackPolicy,
};
use std::num::{NonZeroU64, NonZeroUsize};

#[tokio::test]
async fn installed_terminal_feedback_compares_real_receipts_without_revoking() {
    let fixture = Fixture::new().await;
    let imported = replay(&collect(&fixture, Variation::default())).unwrap();
    let snapshot = EngineCostSnapshot::live_catalog(
        imported.children,
        fingerprint(),
        structured_epoch::View::initial(),
        500,
    )
    .unwrap();
    let mut monitor = snapshot
        .open_structured_feedback(&SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 {
            policy: SloSelectedFeedbackSettingsV1 {
                window_samples: NonZeroUsize::new(2).unwrap(),
                minimum_underestimates: NonZeroUsize::new(2).unwrap(),
                minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
                trigger_excess_ns: NonZeroU64::new(2).unwrap(),
                correction_padding_ns: 1,
                maximum_family_margin_ns: NonZeroU64::new(1000).unwrap(),
                maximum_consumption_lag_ns: NonZeroU64::new(10_000).unwrap(),
                maximum_uncomparable_observations: 0,
                maximum_failed_or_partial: 0,
                maximum_queue_drops: 0,
                maximum_state_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
            },
            storage: SloSelectedFeedbackStorageV1::MemoryOnly,
        })
        .unwrap()
        .unwrap();
    let ids = EngineCostIds::default();
    let queue = sink(2, 512);
    let cases = [
        (&fixture.continuing, None, 1000),
        (&fixture.continuing, Some(FinishReason::EOS), 1100),
        (&fixture.continuing, Some(FinishReason::Stop), 1100),
        (&fixture.at_length, Some(FinishReason::Length), 1130),
        (&fixture.at_length, Some(FinishReason::EOS), 1130),
        (&fixture.at_length, Some(FinishReason::Stop), 1130),
    ];
    for (index, (template, reason, wall)) in cases.into_iter().enumerate() {
        let (fifo, stages) = record_stages(&ids, &queue, template, index as u64 + 1, reason, wall);
        let (input, actual) = structured_v2::structured_serving_observation_v2(&stages).unwrap();
        assert_eq!(
            structured_v2::structured_discovery_input_v2(&stages).unwrap(),
            input,
            "discovery and feedback consume the same validated settled input"
        );
        let now = actual.observed_at_ns + 1;
        let prediction = snapshot
            .audit_structured_query_v2(&StructuredQueryV2::exact(input), now)
            .unwrap();
        let observation = structured_feedback::evaluate(
            &CostEvidenceEntry::StagesOnly {
                stages,
                legacy_rejection: CostCallRejection::Composite,
            },
            Some(snapshot.as_ref()),
            &VirtualClock(AtomicU64::new(now)),
        );
        let FeedbackObservation::Compared(comparison) = &observation else {
            panic!("original {reason:?} must reach retrospective comparison")
        };
        assert_eq!(comparison.actual_ns, wall);
        assert_eq!(comparison.base_planning_ns, prediction.planning_ns);
        assert!(comparison.base_planning_ns >= wall);
        monitor.observe_classified(fifo, observation);
    }
    let audit = monitor.audit();
    assert_eq!(audit.compared, 6);
    assert_eq!(audit.uncomparable_observations, 0);
    assert!(audit.revoked.is_none());
}

#[tokio::test]
async fn installed_feedback_rejects_below_capacity_length_and_changed_receipt() {
    let fixture = Fixture::new().await;
    let (_, invalid_length) = record_stages(
        &EngineCostIds::default(),
        &sink(2, 512),
        &fixture.continuing,
        1,
        Some(FinishReason::Length),
        1100,
    );
    assert!(matches!(
        structured_v2::structured_serving_observation_v2(&invalid_length),
        Err(StructuredUnknownV2::UnsupportedScope)
    ));
    let (_, original) = record_stages(
        &EngineCostIds::default(),
        &sink(2, 512),
        &fixture.continuing,
        2,
        Some(FinishReason::EOS),
        1100,
    );
    assert!(structured_v2::structured_serving_observation_v2(&original).is_ok());
    let mut changed = original.as_ref().clone();
    changed.full_wall_ns = Some(1);
    assert!(matches!(
        structured_v2::structured_serving_observation_v2(&Arc::new(changed)),
        Err(StructuredUnknownV2::InvalidSample)
    ));
}

#[test]
fn legacy_feedback_keeps_length_only_terminal_contract() {
    for maximum in [3, 8] {
        for reason in [FinishReason::EOS, FinishReason::Stop] {
            let original = super::super::stages(
                &[maximum],
                Some(HostTerminalStageV1 {
                    finish_reason: reason,
                    ..terminal()
                }),
                8,
            );
            assert!(original.structured_evidence.as_ref().unwrap().is_ok());
            assert!(matches!(
                structured_v2::structured_serving_observation_v2(&original),
                Err(StructuredUnknownV2::UnsupportedScope)
            ));
        }
    }
    assert!(
        structured_v2::structured_serving_observation_v2(&super::super::stages(
            &[3],
            Some(terminal()),
            8,
        ))
        .is_ok()
    );
}
