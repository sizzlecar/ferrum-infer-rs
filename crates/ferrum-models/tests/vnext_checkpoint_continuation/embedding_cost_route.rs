//! Native F32 embedding uses the existing complete program and real graph lifecycle.
use super::*;
use ferrum_types::SloStructuredCostCapture;

fn build(kind: AttentionKind, capture: SloStructuredCostCapture, replay: bool) -> Fixture {
    let definition = Family::new(kind);
    let states = definition.states();
    let profile = definition.profile_id();
    let family = TypedFamilyRegistration::new(definition)
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
        .unwrap();
    let (runtime, registry, materializers, catalog) =
        ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
            0,
            id(format!(
                "device.cuda.embedding-f32.{kind:?}.{capture:?}.{replay}"
            )),
            ferrum_types::AttentionExecutionPolicy::Portable,
            None,
            capture,
        )
        .unwrap()
        .into_parts();
    let materializer =
        ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(&family)
            .unwrap();
    Fixture::from_prepared_family_with_composition(
        kind,
        family,
        states,
        if replay {
            FixtureExecutionMode::Replay
        } else {
            FixtureExecutionMode::Eager
        },
        None,
        if replay { 1 } else { 8 },
        BTreeMap::new(),
        (runtime, registry, materializers, materializer, catalog),
    )
}

#[test]
fn selected_cuda_native_f32_embedding_future_matches_real_output_and_capture_off() {
    for kind in [
        AttentionKind::GatedDelta,
        AttentionKind::GatedDeltaHadamardF32,
    ] {
        assert_eq!(kind.activation_type(), ElementType::F32);
        let enabled = build(kind, SloStructuredCostCapture::HostSettledV1, false);
        let disabled = build(kind, SloStructuredCostCapture::Disabled, false);
        for rows in [&[1][..], &[3][..], &[2, 3][..]] {
            let actual =
                super::super::full_cost_route::run_with_embedding_statistics(&enabled, rows, true);
            let expected = super::super::full_cost_route::run_with_embedding_statistics(
                &disabled, rows, false,
            );
            assert_eq!(
                actual, expected,
                "capture must preserve real complete embedding output"
            );
            let values = actual
                .iter()
                .flat_map(|row| row.chunks_exact(4))
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values.iter().any(|value| *value != 0.0));
        }
    }
}

#[test]
fn selected_cuda_native_f32_embedding_current_work_survives_real_graph_replay() {
    for kind in [
        AttentionKind::GatedDelta,
        AttentionKind::GatedDeltaHadamardF32,
    ] {
        let enabled = build(kind, SloStructuredCostCapture::HostSettledV1, true);
        let disabled = build(kind, SloStructuredCostCapture::Disabled, true);
        let tokens: Arc<[u32]> = Arc::from([3, 4, 5, 6]);
        let observed = enabled.admit("embedding-on", Arc::clone(&tokens));
        let control = disabled.admit("embedding-off", Arc::clone(&tokens));
        for turn in 0..4 {
            let actual = if turn < 2 {
                enabled.execute(&observed, Arc::clone(&tokens), turn..turn + 1)
            } else {
                // The shared helper now checks both real embedding and attention
                // logical commands against current future evidence; no eager fallback.
                enabled.execute_replayed_with_selected(
                    &observed,
                    Arc::clone(&tokens),
                    turn..turn + 1,
                )
            };
            let expected = if turn < 2 {
                disabled.execute(&control, Arc::clone(&tokens), turn..turn + 1)
            } else {
                disabled.execute_replayed(&control, Arc::clone(&tokens), turn..turn + 1)
            };
            actual.assert_state_nonzero();
            actual.assert_same(
                &expected,
                "new token IDs must flow through the real captured embedding",
            );
        }
        observed.try_complete().unwrap();
        control.try_complete().unwrap();
    }
}
