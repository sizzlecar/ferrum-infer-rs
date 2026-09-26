//! Real RN materialization, complete FFN API work, and resident replay.
use super::*;
use ferrum_types::SloStructuredCostCapture;
use std::sync::atomic::{AtomicU64, Ordering};

fn build(capture: SloStructuredCostCapture, mode: FixtureExecutionMode, rows: u64) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let kind = AttentionKind::GatedDeltaHadamardF16;
    let definition = ffn_family::Q8FfnFamily {
        base: Family::new(kind),
        policy: ffn_family::FfnPolicy::Strict,
    };
    let states = definition.base.states();
    let family = TypedFamilyRegistration::new(Rounded(definition))
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
        .unwrap();
    let (runtime, registry, materializers, catalog) =
        ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
            0,
            id(format!(
                "device.cuda.rn-library-ffn.{}",
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

fn assert_output(bytes: &[Vec<u8>]) {
    let values = bytes
        .iter()
        .flat_map(|row| row.chunks_exact(2))
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect::<Vec<_>>();
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v != 0.0));
}

#[test]
fn selected_cuda_rn_ffn_library_calls_match_actual_future_and_disabled_output() {
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
    for rows in [&[1][..], &[3][..], &[2, 3][..], &[4, 4][..], &[9][..]] {
        // The existing helper compares the whole actual/future command and
        // algorithm tables plus sealed parameters for the selected FFN node.
        let expected = full_cost_route::run_with_output(&disabled, rows, false, "node.ffn").0;
        let actual = full_cost_route::run_with_ffn_statistics(&enabled, rows, true);
        assert_eq!(
            actual, expected,
            "capture must not change the RN numerical path"
        );
        assert_output(&actual);
    }
}

#[test]
fn selected_cuda_rn_ffn_library_resident_replay_uses_current_tokens_and_its_capture_handle() {
    for rows in [1usize, 4] {
        let eager = build(
            SloStructuredCostCapture::Disabled,
            FixtureExecutionMode::Eager,
            rows as u64,
        );
        let replay = build(
            SloStructuredCostCapture::HostSettledV1,
            FixtureExecutionMode::Replay,
            rows as u64,
        );
        let tokens: Arc<[u32]> = (0..4 * rows).map(|i| 1 + ((i * 7) % 29) as u32).collect();
        let eager_owner = eager.admit("library-eager", Arc::clone(&tokens));
        let graph_owner = replay.admit("library-graph", Arc::clone(&tokens));
        let mut previous: Option<Observation> = None;
        for index in 0..4 {
            let range = index * rows..(index + 1) * rows;
            let expected = eager
                .execute_checked_output_node(
                    &eager_owner,
                    Arc::clone(&tokens),
                    range.clone(),
                    false,
                    false,
                    false,
                    "node.ffn",
                )
                .unwrap();
            let actual = replay
                .execute_checked_output_node(
                    &graph_owner,
                    Arc::clone(&tokens),
                    range,
                    index >= 2,
                    false,
                    index >= 2,
                    "node.ffn",
                )
                .unwrap();
            // ReplayedOnly is mandatory after actual warm/capture. The shared
            // runner checks this node is resident, not an eager boundary; the
            // selected adapter compares current actual/future algorithm work.
            actual.assert_same(&expected, "RN library current-input graph result/state");
            actual.assert_state_nonzero();
            if let Some(previous) = &previous {
                actual.assert_different_output(previous);
            }
            previous = Some(actual);
        }
        eager_owner.try_complete().unwrap();
        graph_owner.try_complete().unwrap();
    }
}
