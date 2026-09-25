use super::*;
use std::sync::Arc;

fn wave(
    pending: &[bool],
    forced: &[bool],
    product: CostProductOutput,
) -> (
    CanonicalWaveCostShape,
    Arc<UnsettledStructuredWaveEvidenceV1>,
) {
    assert_eq!(pending.len(), forced.len());
    let op = command(pending.len() as u32, "forecast.kernel");
    let rows: Vec<_> = pending
        .iter()
        .zip(forced)
        .map(|(&pending, &forced)| {
            let mut value = row(false);
            value
                .host_features
                .as_mut()
                .unwrap()
                .state
                .pending_decoded_utf8 = pending;
            if let CostRowOutput::Decode {
                requires_full_logits,
                ..
            } = &mut value.output
            {
                *requires_full_logits = forced || pending;
            }
            value
        })
        .collect();
    let wave = finish(builder(
        &[op.canonical_command(0, 0, provider()).unwrap()],
        &rows,
        product,
        CoreReadbackRoute::HostSynchronized,
        0,
        true,
    ));
    (wave.exact, Arc::new(wave.structured.unwrap()))
}

#[test]
fn host_forecast_nonempty_set_is_bound_to_its_original_selected_recipe() {
    let (exact, recipe) = wave(
        &[true, false, true],
        &[false; 3],
        CostProductOutput::FullLogits,
    );
    let set = HostPendingSetV2::new(
        &exact,
        &recipe,
        &[0, 1, 2],
        HostPendingConstraintV2::NonEmptySubset,
    )
    .unwrap();
    assert_eq!(set.eligible_positions(), [0, 1, 2]);
    assert_eq!(set.constraint(), HostPendingConstraintV2::NonEmptySubset);
    let forecast = HostContentForecastV2::Unresolved(set);
    assert!(forecast.clone().validate(&exact, &recipe).is_ok());
    let equal_replacement = recipe.as_ref().clone();
    assert_eq!(
        forecast.validate(&exact, &equal_replacement),
        Err(StatisticalEvidenceUnknown::ExactBindingMismatch)
    );
    let mut changed = exact;
    changed.rows[0] = ActualRowWork::Decode { kv_tokens: 65 };
    assert_eq!(
        forecast.validate(&changed, &recipe),
        Err(StatisticalEvidenceUnknown::ExactBindingMismatch)
    );
}

#[test]
fn host_forecast_fixed_full_peer_covers_empty_subset_without_changing_fixed_pending() {
    let (exact, recipe) = wave(
        &[false, true, false],
        &[false, true, false],
        CostProductOutput::FullLogits,
    );
    let set = HostPendingSetV2::new(&exact, &recipe, &[0, 2], HostPendingConstraintV2::AnySubset)
        .unwrap();
    assert_eq!(set.eligible_positions(), [0, 2]);
    assert!(recipe.physical_host_rows()[1].pending_decoded_utf8);
    assert!(HostPendingSetV2::new(
        &exact,
        &recipe,
        &[0, 2],
        HostPendingConstraintV2::NonEmptySubset
    )
    .is_err());
    assert!(
        HostPendingSetV2::new(&exact, &recipe, &[2, 0], HostPendingConstraintV2::AnySubset)
            .is_err()
    );
    assert!(
        HostPendingSetV2::new(&exact, &recipe, &[0, 0], HostPendingConstraintV2::AnySubset)
            .is_err()
    );
    assert!(
        HostPendingSetV2::new(&exact, &recipe, &[3], HostPendingConstraintV2::AnySubset).is_err()
    );
}

#[test]
fn host_forecast_greedy_condition_is_not_observed_exact_content() {
    let (exact, recipe) = wave(&[false; 2], &[false; 2], CostProductOutput::GreedyToken);
    let set =
        HostPendingSetV2::new(&exact, &recipe, &[], HostPendingConstraintV2::AnySubset).unwrap();
    let forecast = HostContentForecastV2::Unresolved(set);
    assert!(forecast.validate(&exact, &recipe).is_ok());
    assert!(matches!(forecast, HostContentForecastV2::Unresolved(_)));
    assert!(
        HostPendingSetV2::new(&exact, &recipe, &[0], HostPendingConstraintV2::AnySubset).is_err()
    );
    assert!(HostPendingSetV2::new(
        &exact,
        &recipe,
        &[],
        HostPendingConstraintV2::NonEmptySubset
    )
    .is_err());
    assert!(HostContentForecastV2::Exact
        .validate(&exact, &recipe)
        .is_ok());
    let (full, recipe) = wave(&[true], &[false], CostProductOutput::FullLogits);
    assert!(
        HostPendingSetV2::new(&full, &recipe, &[0], HostPendingConstraintV2::AnySubset).is_err()
    );
    assert!(HostPendingSetV2::new(
        &full,
        &recipe,
        &[0],
        HostPendingConstraintV2::NonEmptySubset
    )
    .is_ok());
}

#[test]
fn host_forecast_never_marks_unstarted_or_prefill_rows_as_unknown_decode() {
    let op = command(1, "forecast.kernel");
    for prefill in [false, true] {
        let mut first = row(false);
        let host = first.host_features.as_mut().unwrap();
        host.state.generated_tokens_before = 0;
        host.state.sampling_history_tokens = 0;
        if prefill {
            first.work = ActualRowWork::Prefill {
                offset: 0,
                count: 4,
                total_prompt_tokens: 4,
            };
            first.output = CostRowOutput::Prefill { final_logits: true };
        } else {
            first.output = CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 0,
                repetition_penalty_bits: 1f32.to_bits(),
            };
        }
        let wave = builder(
            &[op.canonical_command(0, 0, provider()).unwrap()],
            &[first],
            CostProductOutput::FullLogits,
            CoreReadbackRoute::HostSynchronized,
            0,
            true,
        )
        .finish_with_structure(
            if prefill {
                ActualWaveKind::Prefill
            } else {
                ActualWaveKind::Decode
            },
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
        let recipe = Arc::new(wave.structured.unwrap());
        assert!(HostContentForecastV2::Exact
            .validate(&wave.exact, &recipe)
            .is_ok());
        assert!(HostPendingSetV2::new(
            &wave.exact,
            &recipe,
            &[0],
            HostPendingConstraintV2::NonEmptySubset
        )
        .is_err());
    }
}
