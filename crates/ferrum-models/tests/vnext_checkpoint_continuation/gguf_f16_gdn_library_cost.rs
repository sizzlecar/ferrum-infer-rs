//! Real RN GDN library work, current/future agreement, and changed-state replay.
use super::*;
use ferrum_types::SloStructuredCostCapture;
use std::sync::atomic::{AtomicU64, Ordering};

fn build(capture: SloStructuredCostCapture, mode: FixtureExecutionMode, rows: u64) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let kind = AttentionKind::GatedDelta;
    let definition = Family::new(kind);
    let states = definition.states();
    let family = TypedFamilyRegistration::new(Rounded(definition))
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
        .unwrap();
    let (runtime, registry, materializers, catalog) =
        ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
            0,
            id(format!(
                "device.cuda.rn-library-gdn.{}",
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

fn nondegenerate(bytes: &[u8]) {
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v != 0.0));
}

#[test]
fn selected_cuda_rn_gdn_library_calls_match_actual_future_and_disabled_output() {
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
        let expected = full_cost_route::run_with_output(&disabled, rows, false, "node.attention").0;
        let actual = full_cost_route::run_with_gdn_statistics(&enabled, rows, true);
        assert_eq!(
            actual, expected,
            "observation must preserve RN GDN arithmetic"
        );
        for row in &actual {
            nondegenerate(row);
        }
    }
}

#[test]
fn selected_cuda_rn_gdn_library_resident_replay_uses_current_tokens_and_owner_state() {
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
        let tokens_a: Arc<[u32]> = (0..4 * rows)
            .map(|i| 3 + ((i + i / rows) % 8) as u32)
            .collect();
        let tokens_b: Arc<[u32]> = (0..4 * rows)
            .map(|i| 11 + ((i + i / rows) % 8) as u32)
            .collect();
        let execute = |fixture: &Fixture,
                       owner: &Arc<SequenceSession<Runtime>>,
                       tokens: &Arc<[u32]>,
                       index,
                       resident| {
            let observed = fixture
                .execute_checked_output_node(
                    owner,
                    Arc::clone(tokens),
                    index * rows..(index + 1) * rows,
                    resident,
                    false,
                    resident,
                    "node.attention",
                )
                .unwrap();
            observed.assert_state_nonzero();
            nondegenerate(&observed.values["output"]);
            observed
        };
        let baseline = |name, tokens: &Arc<[u32]>| {
            let owner = eager.admit(name, Arc::clone(tokens));
            let values = (0..4)
                .map(|i| execute(&eager, &owner, tokens, i, false))
                .collect::<Vec<_>>();
            owner.try_complete().unwrap();
            values
        };
        let expected_a = baseline("rn-gdn-eager-a", &tokens_a);
        let expected_b = baseline("rn-gdn-eager-b", &tokens_b);
        expected_a[0].assert_different_output(&expected_b[0]);
        expected_a[3].assert_state_changed(&expected_a[2], "independent GDN eager frontier");
        let owner_a = replay.admit("rn-gdn-graph-a", Arc::clone(&tokens_a));
        for i in 0..2 {
            execute(&replay, &owner_a, &tokens_a, i, false)
                .assert_same(&expected_a[i], "actual warm/capture");
        }
        // Mandatory ReplayedOnly, actual resident node, and complete current
        // actual/future selected API table; no eager fallback is accepted.
        execute(&replay, &owner_a, &tokens_a, 2, true)
            .assert_same(&expected_a[2], "changed GDN input");
        let owner_b = replay.admit("rn-gdn-graph-b", Arc::clone(&tokens_b));
        for i in 0..3 {
            execute(&replay, &owner_b, &tokens_b, i, true)
                .assert_same(&expected_b[i], "rebound GDN owner");
        }
        execute(&replay, &owner_a, &tokens_a, 3, true)
            .assert_same(&expected_a[3], "parked GDN state after B");
        execute(&replay, &owner_b, &tokens_b, 3, true)
            .assert_same(&expected_b[3], "returned GDN owner B");
        owner_a.try_complete().unwrap();
        owner_b.try_complete().unwrap();
    }
}
