//! Strict source7 replay of a previous feature binding when all numerical
//! coordinates happen to agree (installed ignore-EOS, actual Length completion).
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1;
use ferrum_interfaces::execution_cost::{
    ActualWaveGraphState, HostContentDomainV1, HostCostPolicyV2, PlainTextPolicyCapabilityV2,
    PlainTextSamplingRouteV2,
};

#[test]
fn prospective_completion_same_axes_previous_checkpoint_binding_is_rejected() {
    let mut h = block_header();
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1;
    let h = StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap();
    let limits = CostProfileLoadLimits::default();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut old_binding_bytes = bytes.clone();
    let mut c = StructuredServiceCollectorV7::new(h.clone(), limits.clone()).unwrap();
    let mut closing = paired(1);
    for block in 1..=4 {
        let first = (block - 1) * 8 + 1;
        let opened = c.open_block(first * 2_000 - 1, (first - 1) * 3).unwrap();
        append(&mut bytes, &opened);
        append(&mut old_binding_bytes, &opened);
        for ticket in first..first + 8 {
            let id = format!("length-only-{ticket}");
            let (prepared, offered, _, future) = old::prepared_batch_algorithms_with_host_policy(
                &[&id],
                3,
                2,
                None,
                ActualWaveGraphState::Disabled,
                None,
                64,
                &[3],
                Some(&domain()),
                &["fixture.profile10"],
                HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextInstalledV2(
                        PlainTextPolicyCapabilityV2 {
                            sampling: PlainTextSamplingRouteV2::FullLogits,
                            model_eos: false,
                            user_stop: false,
                        },
                    )),
                    categorical_signature: [4; 32],
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
            );
            let actual =
                prepared::project_service_actual_with_domain(&prepared, &offered, &domain())
                    .unwrap()
                    .with_settled_terminal_causes(&[(0, ferrum_types::FinishReason::Length)])
                    .unwrap();
            assert_eq!(
                actual.regression_axes(),
                future.unwrap().input().regression_axes(),
                "forced Length has identical actual and prospective numerical coordinates"
            );
            let stages = old::stages(&old::header(), &prepared, ticket, 1_000);
            assert_eq!(
                stages.rows[0].terminal.as_ref().unwrap().finish_reason,
                ferrum_types::FinishReason::Length
            );
            let record = StructuredServiceRecordV7::Completed {
                wave: StructuredServiceWaveV7::from_diagnostic(
                    ticket,
                    ticket * 2_000,
                    ticket * 3,
                    serde_json::to_value(stages).unwrap(),
                    None,
                )
                .unwrap(),
            };
            c.push(&record).unwrap();
            append(&mut bytes, &record);
            append(&mut old_binding_bytes, &record);
        }
        closing = paired((first + 7) * 2_000 + 1_101);
        let record = c.close_block(closing).unwrap();
        append(&mut bytes, &record);
        let mut previous = record.clone();
        if block == 2 {
            let legacy = c.previous_completion_fit_bindings_for_test();
            assert_eq!(legacy.len(), 1);
            let StructuredServiceRecordV7::BlockClose { freezes, .. } = &mut previous else {
                unreachable!()
            };
            assert_eq!(freezes.len(), 1);
            assert_eq!(freezes[0].failure, None);
            assert_eq!(freezes[0].owner_attempt_id, legacy[0].0);
            assert!(freezes[0].nonnegative_fit_certificate.is_some());
            assert_ne!(freezes[0].parameters_sha256, Some(legacy[0].1));
            freezes[0].parameters_sha256 = Some(legacy[0].1);
        }
        append(&mut old_binding_bytes, &previous);
    }
    assert_eq!(c.qualified_children(), 1);
    let (checkpoint, _) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &checkpoint);
    append(&mut old_binding_bytes, &checkpoint);
    let current = replay_structured_source_v7(&bytes, &limits).unwrap();
    assert_eq!(
        current
            .activate_same_process_memory(paired(70_000), &limits)
            .unwrap()
            .children
            .len(),
        1
    );
    let error = replay_structured_source_v7(&old_binding_bytes, &limits)
        .err()
        .expect("previous feature binding cannot publish");
    assert!(
        format!("{error:?}").contains("source7 original block/freeze/certificate differs"),
        "reject at original Fit binding, before later prefix/checkpoint differences: {error:?}"
    );
}
