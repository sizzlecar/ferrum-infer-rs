//! Changed-input replay of the actual RN-F16 materialized providers.
//! Reuses the fixture's strict ReplayedOnly submission, never selected cost.
use super::*;

fn execute(
    fixture: &Fixture,
    session: &Arc<SequenceSession<Runtime>>,
    tokens: &Arc<[u32]>,
    range: Range<usize>,
    replay: bool,
    node: &str,
) -> Observation {
    let count = range.len();
    let observed = fixture
        .execute_checked_output_node(
            session,
            Arc::clone(tokens),
            range,
            replay,
            false,
            false,
            node,
        )
        .unwrap();
    observed.assert_state_nonzero();
    let bytes = &observed.values["output"];
    assert_eq!(
        bytes.len() as u64,
        count as u64 * HIDDEN * fixture.output_type.size_bytes()
    );
    let values = match fixture.output_type {
        ElementType::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect::<Vec<_>>(),
        ElementType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>(),
        other => panic!("unexpected real output type: {other:?}"),
    };
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    assert!(values.iter().any(|x| *x != 0.0));
    observed
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayExpectation {
    Resident,
    CausalVarlenEagerBoundary,
}

fn assert_declared_eager_boundary(fixture: &Fixture, node: &str) {
    let node_index = fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .position(|n| n.id().as_str() == node)
        .unwrap() as u32;
    let catalog = fixture.lane.reusable_execution_catalog().unwrap();
    assert!(
        !catalog.programs().is_empty(),
        "real adaptive execution must publish its topology"
    );
    for program in catalog.programs() {
        assert!(
            program.eager_boundary_node_indices().contains(&node_index),
            "causal varlen must retain its declared EagerBoundary: {program:?}"
        );
        assert!(
            !program
                .segments()
                .iter()
                .any(|s| s.start_node_index() <= node_index && node_index < s.end_node_index()),
            "causal varlen must not be credited as a resident segment"
        );
    }
}

fn compare_changed_inputs(
    eager: &Fixture,
    replay: &Fixture,
    rows: usize,
    node: &str,
    expectation: ReplayExpectation,
) {
    let require_resident = expectation == ReplayExpectation::Resident;

    // Same width/math for each oracle, distinct actual tokens in every window
    // and between owners. No cross-width equality or strict-vs-rounded claim.
    let tokens_a: Arc<[u32]> = (0..4 * rows)
        .map(|i| 3 + ((i + i / rows) % 8) as u32)
        .collect();
    let tokens_b: Arc<[u32]> = (0..4 * rows)
        .map(|i| 11 + ((i + i / rows) % 8) as u32)
        .collect();
    let window = |index| index * rows..(index + 1) * rows;
    let baseline = |name, tokens: &Arc<[u32]>| {
        let owner = eager.admit(name, Arc::clone(tokens));
        let values = (0..4)
            .map(|index| execute(eager, &owner, tokens, window(index), false, node))
            .collect::<Vec<_>>();
        owner.try_complete().unwrap();
        values
    };
    let expected_a = baseline("rn-f16-eager-a", &tokens_a);
    let expected_b = baseline("rn-f16-eager-b", &tokens_b);
    expected_a[0].assert_different_output(&expected_b[0]);
    expected_a[3].assert_state_changed(&expected_a[2], "independent eager frontier");

    let owner_a = replay.admit("rn-f16-replay-a", Arc::clone(&tokens_a));
    // Warm libraries and the configured graph path with real execution. For
    // Resident, the existing runner then requires a ready program, an actual
    // segment for this node, and ReplayedOnly on every subsequent call. The
    // negative varlen case separately proves its declared eager boundary.
    for index in 0..2 {
        expected_a[index].assert_same(
            &execute(replay, &owner_a, &tokens_a, window(index), false, node),
            "RN-F16 actual warm/capture",
        );
    }
    if !require_resident {
        assert_declared_eager_boundary(replay, node);
    }
    expected_a[2].assert_same(
        &execute(
            replay,
            &owner_a,
            &tokens_a,
            window(2),
            require_resident,
            node,
        ),
        "RN-F16 changed-token output/state",
    );
    let owner_b = replay.admit("rn-f16-replay-b", Arc::clone(&tokens_b));
    for index in 0..3 {
        expected_b[index].assert_same(
            &execute(
                replay,
                &owner_b,
                &tokens_b,
                window(index),
                require_resident,
                node,
            ),
            "RN-F16 rebound owner output/state",
        );
    }
    // B overwrote shared invocation scratch while A stayed live. Returning to
    // A must use A's own current recurrent/KV state and current token inputs.
    expected_a[3].assert_same(
        &execute(
            replay,
            &owner_a,
            &tokens_a,
            window(3),
            require_resident,
            node,
        ),
        "RN-F16 parked owner state",
    );
    expected_b[3].assert_same(
        &execute(
            replay,
            &owner_b,
            &tokens_b,
            window(3),
            require_resident,
            node,
        ),
        "RN-F16 second owner state",
    );
    if !require_resident {
        assert_declared_eager_boundary(replay, node);
    }
    owner_a.try_complete().unwrap();
    owner_b.try_complete().unwrap();
}

#[test]
fn gguf_rn_f16_cuda_gdn_causal_changed_inputs_replay_matches_independent_eager() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        // The existing F16-KV varlen path (M > 1) is intentionally exact-shape
        // eager. Only the original partition-stable causal M1 path can prove
        // changed-context resident replay. GDN supports both tested widths.
        let rows_to_replay: &[usize] = if kind == AttentionKind::Causal {
            &[1]
        } else {
            &[1, 4]
        };
        for &rows in rows_to_replay {
            eprintln!("RN-F16 resident scope: kind={kind:?}, rows={rows}, node=node.attention");

            let eager = fixture(Family::new(kind), kind, "node.attention");
            let replay = fixture_with_mode(
                Family::new(kind),
                kind,
                "node.attention",
                FixtureExecutionMode::Replay,
                rows as u64,
            );
            compare_changed_inputs(
                &eager,
                &replay,
                rows,
                "node.attention",
                ReplayExpectation::Resident,
            );
        }
    }
}

#[test]
fn gguf_rn_f16_cuda_ffn_cublas_changed_inputs_replay_matches_independent_eager() {
    let kind = AttentionKind::GatedDeltaHadamardF16;
    let definition = || ffn_family::Q8FfnFamily {
        base: Family::new(kind),
        policy: ffn_family::FfnPolicy::Strict,
    };
    for rows in [1, 4] {
        let eager = fixture(definition(), kind, "node.ffn");
        let replay = fixture_with_mode(
            definition(),
            kind,
            "node.ffn",
            FixtureExecutionMode::Replay,
            rows as u64,
        );
        eprintln!("RN-F16 resident scope: kind={kind:?}, rows={rows}, node=node.ffn");
        compare_changed_inputs(
            &eager,
            &replay,
            rows,
            "node.ffn",
            ReplayExpectation::Resident,
        );
    }
}

#[test]
fn gguf_rn_f16_cuda_causal_varlen_keeps_eager_boundary_and_complete_state() {
    let kind = AttentionKind::Causal;
    let rows = 4;
    eprintln!("RN-F16 eager-boundary scope: kind={kind:?}, rows={rows}, node=node.attention");
    let eager = fixture(Family::new(kind), kind, "node.attention");
    let configured = fixture_with_mode(
        Family::new(kind),
        kind,
        "node.attention",
        FixtureExecutionMode::Replay,
        rows as u64,
    );
    // This is an explicit negative graph-qualification case. Real adaptive
    // execution must retain the causal node as EagerBoundary, while every
    // output and complete effective K/V element matches independent eager
    // execution across changed inputs, two owners and parked-owner resumption.
    compare_changed_inputs(
        &eager,
        &configured,
        rows,
        "node.attention",
        ReplayExpectation::CausalVarlenEagerBoundary,
    );
}
