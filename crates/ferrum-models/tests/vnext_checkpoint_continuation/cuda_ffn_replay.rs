//! Native FFN resident recipes with real capture, fresh owner bindings and
//! current-input output/state oracles. No evidence comes from a synthetic cache.
use super::*;
use ferrum_types::SloStructuredCostCapture;

fn build(policy: q8_ffn_family::FfnPolicy, rows: usize, replay: bool) -> Fixture {
    let kind = AttentionKind::GatedDeltaHadamardF16;
    let definition = q8_ffn_family::Q8FfnFamily {
        base: Family::new(kind),
        policy,
    };
    let states = definition.base.states();
    let profile = definition.base.profile_id();
    let family = TypedFamilyRegistration::new(definition)
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
        .unwrap();
    let capture = if replay {
        SloStructuredCostCapture::HostSettledV1
    } else {
        SloStructuredCostCapture::Disabled
    };
    let (runtime, registry, materializers, catalog) =
        ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create_with_observation(
            0,
            id(format!("device.cuda.ffn-recipe.{policy:?}.{rows}.{replay}")),
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
        rows as u64,
        BTreeMap::new(),
        (runtime, registry, materializers, materializer, catalog),
    )
}

#[test]
fn selected_cuda_native_ffn_resident_recipe_preserves_changed_inputs_and_rebound_owners() {
    use q8_ffn_family::FfnPolicy;
    for (policy, rows) in [
        (FfnPolicy::Strict, 1),
        (FfnPolicy::Strict, 4),
        (FfnPolicy::Q8, 4),
        (FfnPolicy::Q8InputSum, 4),
        (FfnPolicy::StreamMmq, 8),
        (FfnPolicy::Residual2M2To8, 3),
        (FfnPolicy::Residual2M2To8, 8),
        (FfnPolicy::Residual2M2To8, 9),
    ] {
        let eager = build(policy, rows, false);
        let replay = build(policy, rows, true);
        let tokens_a: Arc<[u32]> = (0..4 * rows).map(|i| 1 + ((i * 7) % 29) as u32).collect();
        let tokens_b: Arc<[u32]> = (0..4 * rows)
            .map(|i| 1 + ((i * 11 + 3) % 29) as u32)
            .collect();
        let eager_a = eager.admit("recipe-eager-a", Arc::clone(&tokens_a));
        let graph_a = replay.admit("recipe-graph-a", Arc::clone(&tokens_a));
        let eager_b = eager.admit("recipe-eager-b", Arc::clone(&tokens_b));
        let graph_b = replay.admit("recipe-graph-b", Arc::clone(&tokens_b));
        let mut previous: Option<Observation> = None;
        // Warm and capture with A, then rebind B while A stays live, and return
        // to A after B overwrote invocation scratch. All later work must use a
        // resident segment, and its complete table must match the future route.
        for (owner_b, index, resident) in [
            (false, 0, false),
            (false, 1, false),
            (false, 2, true),
            (true, 0, true),
            (true, 1, true),
            (false, 3, true),
        ] {
            let (tokens, eager_owner, graph_owner) = if owner_b {
                (&tokens_b, &eager_b, &graph_b)
            } else {
                (&tokens_a, &eager_a, &graph_a)
            };
            let range = index * rows..(index + 1) * rows;
            let expected = eager
                .execute_checked_output_node(
                    eager_owner,
                    Arc::clone(tokens),
                    range.clone(),
                    false,
                    false,
                    false,
                    "node.ffn",
                )
                .unwrap();
            let actual = replay
                .execute_checked_output_node(
                    graph_owner,
                    Arc::clone(tokens),
                    range,
                    resident,
                    false,
                    resident,
                    "node.ffn",
                )
                .unwrap();
            actual.assert_same(&expected, "native FFN current-input resident output/state");
            actual.assert_state_nonzero();
            if !owner_b {
                if let Some(previous) = &previous {
                    actual.assert_different_output(previous);
                }
                previous = Some(actual);
            }
        }
        eager_a.try_complete().unwrap();
        graph_a.try_complete().unwrap();
        eager_b.try_complete().unwrap();
        graph_b.try_complete().unwrap();
    }
}
