//! Real CPU submission and original private host settlement through the shared
//! run/serve automatic runtime. Virtual durations exercise feedback semantics;
//! they do not establish backend performance or a terminal latency guarantee.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    resolved::ResolvedCostEntry, selected_feedback::FeedbackObservation, structured_feedback,
};
use ferrum_interfaces::execution_cost::{
    HostContentDomainV1, PlainTextPolicyCapabilityV2, PlainTextSamplingRouteV2,
};
use ferrum_types::{
    FinishReason, SloAutomaticCalibrationNumericalStrategyV1, SloSelectedFeedbackStorageV1,
    SloStructuredFeedbackPolicy,
};

fn eos_capable_wave() -> Wave {
    let mut host = wave(A).host;
    host.state.maximum_output_tokens = 3;
    host.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextInstalledV2(
        PlainTextPolicyCapabilityV2 {
            sampling: PlainTextSamplingRouteV2::Greedy {
                repetition_penalty: false,
            },
            model_eos: true,
            user_stop: false,
        },
    ));
    wave_with_host_rows(A, 7, host, 2)
}

#[tokio::test]
async fn prospective_completion_global_runtime_compares_eos_and_corrects_then_revokes() {
    let f = Families::configured_with_settings(
        SloAutomaticCalibrationSettingsV1 {
            discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
            phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
            numerical_strategy:
                SloAutomaticCalibrationNumericalStrategyV1::IdentifiedFitGlobalResidualV1,
            ..Default::default()
        },
        true,
    );
    // Complete original Discovery/Fit/Residual/Qualification blocks. Every
    // physical row continues below its cap, while its installed policy permits
    // EOS. No EOS outcome or measured wall is supplied to the readiness rule.
    for phase in 0..4 {
        f.runtime.consume_samples();
        for _ in 0..OFFERS {
            let stages =
                fixture::record_cohort_route(&f.runtime, &f.clock, eos_capable_wave()).unwrap();
            assert!(stages.rows.iter().all(|row| row.terminal.is_none()));
            stages
                .structured_evidence
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap()
                .validate_host_stages(&stages)
                .unwrap();
            f.runtime.consume_samples();
        }
        let audit = f.live().audit();
        assert_eq!(audit.failed_generations, 0, "{audit:#?}");
        assert_eq!(
            audit.qualified_publications,
            u64::from(phase == 3),
            "{audit:#?}"
        );
    }
    let now = f.clock.now_ns().unwrap();
    let children = f.runtime.training.live_catalog_children(now).unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(
        children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|phase| phase.members),
        [OFFERS; 3]
    );
    let query_wave = eos_capable_wave();
    let query = StructuredQueryV2::from_future_with_domain(
        &query_wave.prepared.exact,
        &query_wave.prepared.selected,
        &query_wave.prepared.recipe,
        &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
        &f.domain,
    )
    .unwrap();
    let snapshot = f.runtime.snapshot().unwrap();
    let prediction = snapshot.audit_structured_query_v2(&query, now).unwrap();
    let family = snapshot.prospective_source(&query).unwrap().domain;
    let mut monitor = snapshot
        .open_structured_feedback(&SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 {
            policy: SloSelectedFeedbackSettingsV1 {
                window_samples: NonZeroUsize::new(2).unwrap(),
                minimum_underestimates: NonZeroUsize::new(2).unwrap(),
                minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
                trigger_excess_ns: NonZeroU64::new(2).unwrap(),
                correction_padding_ns: 1,
                maximum_family_margin_ns: NonZeroU64::new(64).unwrap(),
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
    let initial_view = monitor.current_view();
    assert!(initial_view.current());
    let mut corrected_view = None;
    let mut expected_margin = 0;
    let mut previous_fifo = None;
    for (index, extra) in [4u64, 4, 256, 256].into_iter().enumerate() {
        // Advance the original clock during host settlement, never mutate the
        // completed wall field. The second pair exceeds the declared margin.
        assert!(
            f.runtime.sink.pop_numbered_with_ticket().is_none(),
            "the completed Q population and prior feedback must leave no older FIFO entry"
        );
        let delay = prediction.planning_ns.checked_add(extra).unwrap();
        let stages =
            fixture::record_eos_feedback_cohort(&f.runtime, &f.clock, eos_capable_wave(), delay)
                .unwrap();
        assert!(stages
            .rows
            .iter()
            .all(|row| row.terminal.as_ref().unwrap().finish_reason == FinishReason::EOS));
        stages
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .validate_host_stages(&stages)
            .unwrap();
        let (fifo, entry, ticket) = f.runtime.sink.pop_numbered_with_ticket().unwrap();
        assert!(
            ticket.is_none(),
            "feedback cannot create calibration membership"
        );
        let CostEvidenceEntry::StagesOnly {
            stages: original, ..
        } = &entry
        else {
            panic!("original private settlement entry");
        };
        // The fixture calls make_host_stages for a preview. finish queues the
        // original call, and sealed::resolve builds a fresh Arc from it. Pointer
        // equality is not the call identity; the queued record is authoritative.
        if let Some(previous) = previous_fifo {
            assert_eq!(fifo, previous + 1, "original accepted FIFO order");
        }
        previous_fifo = Some(fifo);
        assert_eq!(original.call_id, stages.call_id);
        assert_eq!(original.fingerprint, stages.fingerprint);
        assert_eq!(original.prepare_started_at_ns, stages.prepare_started_at_ns);
        assert_eq!(
            original.executor_returned_at_ns,
            stages.executor_returned_at_ns
        );
        assert_eq!(original.finalized_at_ns, stages.finalized_at_ns);
        assert_eq!(original.full_wall_ns, stages.full_wall_ns);
        assert_eq!(original.completeness, stages.completeness);
        assert_eq!(original.rows.len(), stages.rows.len());
        for (queued, preview) in original.rows.iter().zip(&stages.rows) {
            assert_eq!(queued.request_id, preview.request_id);
            assert_eq!(queued.owner_incarnation, preview.owner_incarnation);
            assert_eq!(queued.work_generation, preview.work_generation);
            assert_eq!(queued.input_index, preview.input_index);
            assert_eq!(
                queued.host_processing_ordinal,
                preview.host_processing_ordinal
            );
            assert_eq!(queued.host_started_at_ns, preview.host_started_at_ns);
            assert_eq!(queued.token_committed_at_ns, preview.token_committed_at_ns);
            assert_eq!(
                queued.output_published_at_ns,
                preview.output_published_at_ns
            );
            assert_eq!(
                queued.completion_started_at_ns,
                preview.completion_started_at_ns
            );
            assert_eq!(queued.settled_at_ns, preview.settled_at_ns);
            assert_eq!(queued.completeness, preview.completeness);
        }
        // Includes the actual shape, complete terminal receipts and original
        // structured sidecar, which the legacy Serialize view omits.
        assert_eq!(
            serde_json::to_value(original.structured_diagnostic_view()).unwrap(),
            serde_json::to_value(stages.structured_diagnostic_view()).unwrap(),
        );
        original
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .validate_host_stages(original)
            .unwrap();
        let wall = original.full_wall_ns.unwrap();
        let resolved = ResolvedCostEntry::new_with_domain(entry, Some(&f.domain));
        let mut failure = None;
        let observation = structured_feedback::evaluate_resolved(
            &resolved,
            Some(snapshot.as_ref()),
            f.clock.as_ref(),
            &mut failure,
        );
        let FeedbackObservation::Compared(comparison) = &observation else {
            panic!("an actual EOS on the declared route must not become an excluded/unknown sample: {failure:?}");
        };
        assert_eq!(comparison.family, family);
        assert_eq!(comparison.actual_ns, wall);
        assert_eq!(
            comparison.base_planning_ns, prediction.planning_ns,
            "settled EOS uses the identical prospective completion features as its pre-submit query"
        );
        assert!(wall > comparison.base_planning_ns);
        let excess = wall - comparison.base_planning_ns;
        if index < 2 {
            expected_margin = expected_margin.max(excess + 1);
            assert!(expected_margin <= 64);
        } else {
            assert!(excess > 64);
        }
        monitor.observe_classified(fifo, observation);
        let audit = monitor.audit();
        assert_eq!(audit.compared, index as u64 + 1);
        assert_eq!(audit.uncomparable_observations, 0);
        assert_eq!(audit.outside_support_observations, 0);
        assert_eq!(audit.outside_catalog_observations, 0);
        match index {
            0 => {
                assert!(audit.revoked.is_none());
                assert_eq!(audit.corrections, 0);
                assert!(initial_view.current());
            }
            1 => {
                assert!(audit.revoked.is_none());
                assert_eq!(audit.corrections, 1);
                assert_eq!(audit.maximum_margin_ns, expected_margin);
                assert!(
                    !initial_view.current(),
                    "correction closes the old epoch first"
                );
                let view = monitor.publish(Some(0)).unwrap();
                view.activate();
                assert!(view.current());
                assert_eq!(view.margin(&family), expected_margin);
                corrected_view = Some(view);
            }
            2 => {
                assert!(audit.revoked.is_none());
                assert!(corrected_view.as_ref().unwrap().current());
            }
            3 => {
                assert!(audit.revoked.is_some());
                assert_eq!(
                    serde_json::to_value(&audit).unwrap()["revoked"],
                    "correction_limit"
                );
                assert!(!corrected_view.as_ref().unwrap().current());
                let revoked = monitor.publish(Some(0)).unwrap();
                revoked.activate();
                assert!(!revoked.current(), "revocation cannot reactivate an epoch");
            }
            _ => unreachable!(),
        }
    }
}
