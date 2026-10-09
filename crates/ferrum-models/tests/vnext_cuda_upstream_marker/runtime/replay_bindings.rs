//! Exercise the binding-only encoder through real resident programs. These
//! profiles always retain MarkerV2 dependencies; hybrid small-row G32 would
//! bypass the MMQ replay hook and is covered separately by `hybrid`.
use super::*;

fn compare_waves(kind: AttentionKind, participants: u32, width: usize, prefill: bool) {
    let definition = || {
        if prefill {
            Family::prefill(kind, 2048)
        } else {
            Family::new(kind)
        }
    };
    let eager = Fixture::for_family(kind, false, participants, definition());
    let replay = Fixture::for_family(kind, true, participants, definition());
    let tokens: Vec<Arc<[u32]>> = (0..participants)
        .map(|participant| {
            (0..4 * width)
                .map(|position| ((position * 7 + participant as usize * 3 + 1) % 32) as u32)
                .collect()
        })
        .collect();
    let sessions = [&eager, &replay].map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| fixture.admit(&format!("binding-replay-{i}"), t.clone()))
            .collect::<Vec<_>>()
    });
    for wave in 0..4 {
        let range = wave * width..(wave + 1) * width;
        let expected =
            eager.execute_participants(&sessions[0], &tokens, range.clone(), Path::Eager);
        // The shared helper resolves the actual resident catalog, requires its
        // attention/FFN retained dependencies and exact seal, then submits with
        // ReplayedOnly. Different source positions change the real state/KV
        // bindings; neither eager fallback nor stale binding payload can pass.
        let path = if wave < 2 { Path::Warm } else { Path::Replay };
        let actual = replay.execute_participants(&sessions[1], &tokens, range.clone(), path);
        expected.assert_same(&actual);
        for participant in 0..participants {
            assert_eq!(
                actual.values[&(participant, "output".into())].len(),
                width * HIDDEN as usize * 4
            );
            if kind == AttentionKind::Causal {
                assert_eq!(
                    actual.values[&(participant, "state.kv".into())].len(),
                    range.end * 1024
                );
            }
        }
        actual.dump(kind, participants, range, path);
    }
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires the qualified MMQ prefill CUDA artifact and real GDN state"]
fn mmq_binding_only_prefill_replay_preserves_gdn_and_ffn_state() {
    // Large-row MMQ must remain a genuine resident path, including changing
    // source ranges. This does not force multi-token FP16 causal replay.
    for width in [33, 54] {
        compare_waves(AttentionKind::GatedDelta, 1, width, true);
    }
}

#[test]
#[ignore = "requires qualified MarkerV2 CUDA providers and real FP16 KV state"]
fn marker_binding_only_decode_replay_preserves_causal_kv_and_ffn_state() {
    // M8 attention uses MMVQ and M9 uses MMQ; FFN uses the actual role policy.
    // Both banks carry the same lifetime/authority obligation on every wave.
    for participants in [8, 9] {
        compare_waves(AttentionKind::Causal, participants, 1, false);
    }
}
