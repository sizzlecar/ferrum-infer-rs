//! Actual RN materialization and full causal API+kernel evidence. Library work
//! is never credited as a private CUDA kernel or as a model-quality result.
use super::*;
use ferrum_types::SloStructuredCostCapture;
use std::sync::atomic::{AtomicU64, Ordering};

fn build(capture: SloStructuredCostCapture, mode: FixtureExecutionMode, rows: u64) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let kind = AttentionKind::Causal;
    let definition = Family::new(kind);
    let states = definition.states();
    let family = TypedFamilyRegistration::new(Rounded(definition))
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
        .unwrap();
    let (runtime, registry, materializers, catalog) =
        ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
            0,
            id(format!(
                "device.cuda.rn-library-causal.{}",
                NEXT.fetch_add(1, Ordering::Relaxed)
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
    assert_eq!(
        materializer.materializer_id().as_str(),
        GGUF_F16_PROJECTION_MATERIALIZER_ID
    );
    Fixture::from_prepared_family_with_composition(
        kind,
        family,
        states,
        mode,
        None,
        rows,
        BTreeMap::new(),
        (runtime, registry, materializers, materializer, catalog),
    )
}

#[test]
fn selected_cuda_rn_causal_library_actual_future_preserves_packed_and_page_boundary_output() {
    let disabled = build(
        SloStructuredCostCapture::Disabled,
        FixtureExecutionMode::Eager,
        9,
    );
    let enabled = build(
        SloStructuredCostCapture::HostSettledV1,
        FixtureExecutionMode::Eager,
        9,
    );
    for rows in [&[1][..], &[3][..], &[2, 3][..], &[4, 4][..], &[129, 2][..]] {
        let expected = full_cost_route::run_with_output(&disabled, rows, false, "node.attention").0;
        // This helper probes the attention node: actual binding+compute and
        // the current future route must match all selected work and seals.
        let actual = full_cost_route::run_with_gdn_statistics(&enabled, rows, true);
        assert_eq!(
            actual, expected,
            "passive API capture changed causal output"
        );
        let values = actual
            .iter()
            .flat_map(|row| row.chunks_exact(4))
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(!values.is_empty() && values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
    }
}

#[test]
fn selected_cuda_rn_causal_library_resident_current_context_matches_independent_eager() {
    let eager = build(
        SloStructuredCostCapture::Disabled,
        FixtureExecutionMode::Eager,
        1,
    );
    let replay = build(
        SloStructuredCostCapture::HostSettledV1,
        FixtureExecutionMode::Replay,
        1,
    );
    let tokens: Arc<[u32]> = Arc::from([3, 7, 13, 17]);
    let eager_owner = eager.admit("causal-library-eager", Arc::clone(&tokens));
    let graph_owner = replay.admit("causal-library-graph", Arc::clone(&tokens));
    let mut previous: Option<Observation> = None;
    for index in 0..4 {
        let expected = eager
            .execute_checked_output_node(
                &eager_owner,
                Arc::clone(&tokens),
                index..index + 1,
                false,
                false,
                false,
                "node.attention",
            )
            .unwrap();
        let actual = replay
            .execute_checked_output_node(
                &graph_owner,
                Arc::clone(&tokens),
                index..index + 1,
                index >= 2,
                false,
                index >= 2,
                "node.attention",
            )
            .unwrap();
        // Direct calls must find this node inside a resident segment and pass
        // ReplayedOnly, plus current actual/future selected-evidence equality.
        actual.assert_same(&expected, "RN causal library current-token output/KV");
        actual.assert_state_nonzero();
        if let Some(previous) = &previous {
            actual.assert_state_changed(previous, "causal current context progresses");
        }
        previous = Some(actual);
    }
    eager_owner.try_complete().unwrap();
    graph_owner.try_complete().unwrap();
}
