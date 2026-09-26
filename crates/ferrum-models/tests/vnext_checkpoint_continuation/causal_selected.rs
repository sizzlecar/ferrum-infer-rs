//! Real attention provider evidence, using the existing complete fixture and
//! resident-program driver. No simulated command substitutes for a GPU launch.
use super::*;
use ferrum_types::{AttentionExecutionPolicy, SloStructuredCostCapture};

fn fixture(
    capture: SloStructuredCostCapture,
    policy: AttentionExecutionPolicy,
    mode: FixtureExecutionMode,
) -> Fixture {
    let kind = AttentionKind::Causal;
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
                "device.cuda.causal-selected.{policy:?}.{capture:?}"
            )),
            policy,
            None,
            capture,
        )
        .unwrap()
        .into_parts();
    let materializer =
        ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection(&family)
            .unwrap();
    let participants = if matches!(mode, FixtureExecutionMode::Replay) {
        1
    } else {
        8
    };
    Fixture::from_prepared_family_with_composition(
        kind,
        family,
        states,
        mode,
        None,
        participants,
        BTreeMap::new(),
        (runtime, registry, materializers, materializer, catalog),
    )
}
#[test]
fn selected_cuda_causal_current_future_work_preserves_real_scalar_packed_and_page_boundary_outputs()
{
    let disabled = fixture(
        SloStructuredCostCapture::Disabled,
        AttentionExecutionPolicy::Portable,
        FixtureExecutionMode::Eager,
    );
    let enabled = fixture(
        SloStructuredCostCapture::HostSettledV1,
        AttentionExecutionPolicy::Portable,
        FixtureExecutionMode::Eager,
    );
    for rows in [&[1][..], &[4][..], &[2, 3][..], &[129, 2][..]] {
        // The existing attention-node probe validates the full binding+compute
        // selected tables against the real current future query.
        let baseline =
            super::super::full_cost_route::run_with_gdn_statistics(&disabled, rows, false);
        let observed = super::super::full_cost_route::run_with_gdn_statistics(&enabled, rows, true);
        assert_eq!(
            baseline, observed,
            "passive causal capture changes real output"
        );
        let values = observed
            .iter()
            .flat_map(|row| row.chunks_exact(4))
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
    }
}
#[test]
fn selected_cuda_causal_current_evidence_survives_real_capture_and_direct_replay() {
    // This model fixture has context capacity 160 and cannot reach native V2's
    // 513-token boundary. The pinned-native kernel test independently exercises
    // batched V2 and changed-length graph replay; this test proves Portable only.
    for policy in [AttentionExecutionPolicy::Portable] {
        let enabled = fixture(
            SloStructuredCostCapture::HostSettledV1,
            policy,
            FixtureExecutionMode::Replay,
        );
        let disabled = fixture(
            SloStructuredCostCapture::Disabled,
            policy,
            FixtureExecutionMode::Replay,
        );
        let tokens: Arc<[u32]> = Arc::from([3, 4, 5, 6]);
        let observed = enabled.admit("causal-capture-on", Arc::clone(&tokens));
        let baseline = disabled.admit("causal-capture-off", Arc::clone(&tokens));
        for turn in 0..4 {
            let actual = if turn < 2 {
                enabled.execute(&observed, Arc::clone(&tokens), turn..turn + 1)
            } else {
                enabled.execute_replayed_with_selected(
                    &observed,
                    Arc::clone(&tokens),
                    turn..turn + 1,
                )
            };
            let expected = if turn < 2 {
                disabled.execute(&baseline, Arc::clone(&tokens), turn..turn + 1)
            } else {
                disabled.execute_replayed(&baseline, Arc::clone(&tokens), turn..turn + 1)
            };
            actual.assert_state_nonzero();
            actual.assert_same(
                &expected,
                "current context evidence must preserve actual KV state and output",
            );
        }
        observed.try_complete().unwrap();
        baseline.try_complete().unwrap();
    }
}
