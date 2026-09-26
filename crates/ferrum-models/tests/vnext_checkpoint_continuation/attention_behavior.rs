//! Packed provider execution compared with independent single-request waves.
//! These tests replace source-text assumptions about prepare module placement.
use super::*;

fn fixture(kind: AttentionKind) -> Fixture {
    let definition = Family::new(kind);
    let states = definition.states();
    let profile = definition.profile_id();
    let family = TypedFamilyRegistration::new(definition)
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
        .unwrap();
    // Keep causal F16's existing no-checkpoint declaration: this verifies real
    // output/KV, without claiming support for public checkpoint serialization.
    Fixture::from_prepared_family(kind, family, states, false, None, 1, BTreeMap::new())
}

fn compare_packed_and_single(kind: AttentionKind, counts: &[usize], expected: ExpectedRoute) {
    let fixture = fixture(kind);
    let node = fixture
        .compilation
        .executable()
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .position(|node| node.id().as_str() == "node.attention")
        .unwrap();
    let tokens = counts
        .iter()
        .enumerate()
        .map(|(participant, &count)| {
            (0..count)
                .map(|index| ((index * 7 + participant * 3 + 1) % 32) as u32)
                .collect::<Arc<[u32]>>()
        })
        .collect::<Vec<_>>();
    // execute observes actual runtime command attribution, requires Packed for
    // multi-participant rows, waits for GPU completion and reads output + state.
    let packed = execute(&fixture, &tokens, node, true, expected, true);
    assert_eq!(packed.len(), tokens.len());
    for (index, tokens) in tokens.iter().enumerate() {
        let serial = execute(
            &fixture,
            std::slice::from_ref(tokens),
            node,
            true,
            expected,
            true,
        );
        let [serial] = serial.as_slice() else {
            panic!("one independent request")
        };
        packed[index].assert_state_nonzero();
        packed[index].assert_same(serial, "packed and independent request output/state");
    }
    // Distinct inputs and a reversed physical population expose accidental
    // participant-coordinate or QKVZBA slot reuse instead of constant output.
    packed[0].assert_different_output(&packed[1]);
    let reversed_tokens = tokens.iter().rev().cloned().collect::<Vec<_>>();
    let reversed = execute(&fixture, &reversed_tokens, node, true, expected, true);
    for (original, reversed) in packed.iter().zip(reversed.iter().rev()) {
        original.assert_same(reversed, "reordered packed request output/state");
    }
}

#[test]
fn cuda_recurrent_packed_projection_and_qkvzba_match_independent_requests() {
    for counts in [&[2, 3][..], &[4, 4][..]] {
        compare_packed_and_single(AttentionKind::GatedDelta, counts, ExpectedRoute::GatedDelta);
    }
}

#[test]
fn cuda_causal_packed_projection_residual_and_kv_match_independent_requests() {
    for counts in [&[2, 3][..], &[129, 2][..]] {
        compare_packed_and_single(
            AttentionKind::Causal,
            counts,
            ExpectedRoute::Causal { int8: false },
        );
    }
}
