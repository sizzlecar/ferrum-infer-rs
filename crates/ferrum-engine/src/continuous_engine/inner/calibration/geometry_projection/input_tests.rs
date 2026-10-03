use super::tests::{assert_unsubmitted_clean, fixture, limits, requests};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredWaveRoleV2;

#[tokio::test]
async fn initial_prefill_projection_respects_row_ceiling_without_pruning_legal_width() {
    for (whole, row_ceiling, rows, expected_final) in [
        (2, Some(1), 1, Some(false)),
        (2, Some(1), 2, Some(false)),
        (1, Some(1), 2, None),
        (2, None, 1, Some(true)),
    ] {
        let (mut session, executor) = fixture(2).await;
        let mut probes = requests(&session, 2);
        for probe in &mut probes {
            probe.request.prompt = "test ok".into();
            probe.request.sampling_params.max_tokens = 1;
        }
        let mut capacity = limits();
        capacity.prefill_chunk = NonZeroU32::new(whole).unwrap();
        capacity.prefill_row_ceiling = row_ceiling.and_then(NonZeroU32::new);
        let report = session
            .project_geometry_inputs(
                probes,
                &[GeometryInputTarget::InitialPrefill { rows }],
                capacity,
                &[],
            )
            .await
            .unwrap();
        let outcome = &report.outcomes[0];
        if let Some(expected_final) = expected_final {
            assert_eq!(outcome.unknown, None);
            assert_eq!(outcome.branches.len(), 1);
            let input = outcome.branches[0].query.input();
            assert_eq!(input.owner().rows as usize, rows);
            assert!(input.physical_host_rows().iter().all(|row| {
                row.initial_prefill
                    && row.final_prefill == expected_final
                    && row.no_generated_history
            }));
        } else {
            assert_eq!(
                outcome.unknown,
                Some(GeometryProjectionUnknown::Unreachable)
            );
            assert!(outcome.branches.is_empty());
        }
        assert_unsubmitted_clean(&session, &executor);
        session.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn initial_prefill_uses_real_per_row_chunks_without_requiring_decode_capacity() {
    let (mut session, executor) = fixture(2).await;
    let mut probes = requests(&session, 2);
    for probe in &mut probes {
        probe.request.prompt = "test ok v7".into();
        probe.request.sampling_params.max_tokens = 1;
    }
    let report = session
        .project_geometry_inputs(
            probes,
            &[GeometryInputTarget::InitialPrefill { rows: 2 }],
            limits(),
            &[],
        )
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 2);
    assert_eq!(report.projection_attempts, 1);
    let outcome = &report.outcomes[0];
    assert_eq!(outcome.unknown, None);
    assert_eq!(outcome.prefix_condition, None);
    assert_eq!(outcome.branches.len(), 1);
    let branch = &outcome.branches[0];
    assert_eq!(branch.host_branch, None);
    let input = branch.query.input();
    assert_eq!(input.owner().rows, 2);
    assert_eq!(input.owner().role, StructuredWaveRoleV2::Prefill);
    assert_eq!(
        input.numerical_family_key(),
        Err(StructuredUnknownV2::UnsupportedScope)
    );
    assert!(input
        .physical_host_rows()
        .iter()
        .all(|row| row.initial_prefill && !row.final_prefill && row.no_generated_history));
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn initial_prefill_widths_share_one_real_admitted_root_group() {
    let (mut session, executor) = fixture(3).await;
    let probes = requests(&session, 3);
    let targets: Vec<_> = (1..=3)
        .map(|rows| GeometryInputTarget::InitialPrefill { rows })
        .collect();
    let report = session
        .project_geometry_inputs(probes, &targets, limits(), &[])
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 3);
    assert_eq!(report.projection_attempts, 3);
    for (index, outcome) in report.outcomes.iter().enumerate() {
        assert_eq!(outcome.unknown, None);
        assert_eq!(
            outcome.branches[0].query.input().owner().rows as usize,
            index + 1
        );
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn continuation_prefill_reaches_real_middle_and_final_prompt_spans_before_sampling() {
    let (mut session, executor) = fixture(2).await;
    let mut probes = requests(&session, 2);
    for probe in &mut probes {
        probe.request.prompt = "test ok v7".into();
        probe.request.sampling_params.max_tokens = 1;
    }
    let targets = [
        GeometryInputTarget::InitialPrefill { rows: 2 },
        GeometryInputTarget::PrefillSpan { rows: 2, offset: 1 },
        GeometryInputTarget::PrefillSpan { rows: 2, offset: 2 },
    ];
    let report = session
        .project_geometry_inputs(probes, &targets, limits(), &[])
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 2);
    // Every observed target executes its own query. Its checked joint
    // successor carries the same original roots into the next prompt span.
    assert_eq!(report.projection_attempts, 1 + 1 + 1);
    assert_eq!(report.outcomes.len(), 3);
    for (index, outcome) in report.outcomes.iter().enumerate() {
        assert_eq!(outcome.unknown, None);
        assert_eq!(outcome.prefix_condition, None);
        assert_eq!(outcome.branches.len(), 1);
        let branch = &outcome.branches[0];
        assert_eq!(branch.host_branch, None);
        assert_eq!(
            branch.query.input().owner().role,
            StructuredWaveRoleV2::Prefill
        );
        assert!(branch.query.input().physical_host_rows().iter().all(|row| {
            row.initial_prefill == (index == 0)
                && row.final_prefill == (index == 2)
                && row.no_generated_history
        }));
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn continuation_prefill_rejects_zero_unaligned_and_out_of_prompt_offsets() {
    let (mut session, executor) = fixture(1).await;
    let mut probes = requests(&session, 1);
    probes[0].request.prompt = "test ok v7".into();
    probes[0].request.sampling_params.max_tokens = 1;
    let targets = [0, 1, 3].map(|offset| GeometryInputTarget::PrefillSpan { rows: 1, offset });
    let report = session
        .project_geometry_inputs(probes, &targets, limits(), &[])
        .await
        .unwrap();
    assert_eq!(report.projection_attempts, 0);
    for outcome in report.outcomes {
        assert_eq!(
            outcome.unknown,
            Some(GeometryProjectionUnknown::Unreachable)
        );
        assert!(outcome.branches.is_empty());
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn continuation_prefill_budget_cannot_jump_to_a_later_prompt_state() {
    let (mut session, executor) = fixture(2).await;
    let mut probes = requests(&session, 2);
    for probe in &mut probes {
        probe.request.prompt = "test ok v7".into();
        probe.request.sampling_params.max_tokens = 1;
    }
    let mut limit = limits();
    limit.maximum_projections = 1;
    let report = session
        .project_geometry_inputs(
            probes,
            &[GeometryInputTarget::PrefillSpan { rows: 2, offset: 1 }],
            limit,
            &[],
        )
        .await
        .unwrap();
    assert_eq!(report.projection_attempts, 1);
    assert_eq!(
        report.outcomes[0].unknown,
        Some(GeometryProjectionUnknown::BudgetExhausted)
    );
    assert!(report.outcomes[0].branches.is_empty());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}
