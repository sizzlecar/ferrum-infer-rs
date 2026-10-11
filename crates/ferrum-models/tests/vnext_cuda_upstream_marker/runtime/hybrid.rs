//! New hybrid provider, against the actual AtN G32 provider at small local M,
//! then same-policy state continuation across the MMQ threshold. No timing gate.
use super::*;

fn small(kind: AttentionKind) {
    for participants in [1_u32, 3, 4, 7, 8, 9, 16, 32] {
        let atn = Fixture::for_family(kind, false, participants, Family::g32_baseline(kind));
        let eager =
            Fixture::for_family(kind, false, participants, Family::hybrid(kind, MAX_TOKENS));
        let graph = Fixture::for_family(kind, true, participants, Family::hybrid(kind, MAX_TOKENS));
        let tokens: Vec<Arc<[u32]>> = (0..participants)
            .map(|p| (0..8).map(|t| ((t * 7 + p * 3 + 1) % 32) as u32).collect())
            .collect();
        let sessions = [&atn, &eager, &graph].map(|fixture| {
            tokens
                .iter()
                .enumerate()
                .map(|(i, t)| fixture.admit(&format!("hybrid-small-{i}"), t.clone()))
                .collect::<Vec<_>>()
        });
        // Every local projection has <=32 rows, including state initialization.
        // Thus this is a genuine old-G32 comparison, not differing MMQ history.
        for position in 0..8 {
            let range = position..position + 1;
            let expected =
                atn.execute_participants(&sessions[0], &tokens, range.clone(), Path::Eager);
            let actual =
                eager.execute_participants(&sessions[1], &tokens, range.clone(), Path::Eager);
            let replay = graph.execute_participants(
                &sessions[2],
                &tokens,
                range.clone(),
                if position < 2 {
                    Path::Warm
                } else {
                    Path::Replay
                },
            );
            expected.assert_same(&actual);
            actual.assert_same(&replay);
            println!(
                "{}",
                serde_json::json!({"kind":"hybrid_atn_same_input_full_state","attention":format!("{kind:?}"),"actual_local_rows":participants,"range":[position,position+1],"atn_eager_hybrid_eager_replay_bytes_equal":true})
            );
        }
        for group in sessions {
            for session in group {
                session.try_complete().unwrap();
            }
        }
    }
}
fn large(kind: AttentionKind) {
    for rows in [33_usize, 155, 2048] {
        let eager = Fixture::for_family(kind, false, 1, Family::hybrid(kind, 2052));
        let graph = Fixture::for_family(kind, true, 1, Family::hybrid(kind, 2052));
        let tokens: Arc<[u32]> = (0..rows + 4).map(|i| ((i * 7 + 1) % 32) as u32).collect();
        let a = eager.admit("hybrid-large-eager", tokens.clone());
        let b = graph.admit("hybrid-large-graph", tokens.clone());
        for (range, path) in
            std::iter::once((0..rows, Path::Warm)).chain((rows..rows + 4).map(|i| {
                (
                    i..i + 1,
                    if i < rows + 2 {
                        Path::Warm
                    } else {
                        Path::Replay
                    },
                )
            }))
        {
            let expected = eager.execute_participants(
                std::slice::from_ref(&a),
                std::slice::from_ref(&tokens),
                range.clone(),
                Path::Eager,
            );
            let actual = graph.execute_participants(
                std::slice::from_ref(&b),
                std::slice::from_ref(&tokens),
                range.clone(),
                path,
            );
            expected.assert_same(&actual);
            assert_eq!(
                actual.values[&(0, "output".into())].len(),
                range.len() * HIDDEN as usize * 4
            );
            if kind == AttentionKind::Causal {
                assert_eq!(
                    actual.values[&(0, "state.kv".into())].len(),
                    range.end * 1024
                );
            }
            println!(
                "{}",
                serde_json::json!({"kind":"hybrid_mmq_to_g32_continuation","attention":format!("{kind:?}"),"prefill_rows":rows,"range":[range.start,range.end],"path":format!("{path:?}"),"all_state_equal":true})
            );
        }
        a.try_complete().unwrap();
        b.try_complete().unwrap();
    }
}
#[test]
#[ignore = "requires both qualified G32 and locked MMQ MarkerV2 CUDA providers"]
fn hybrid_gdn_small_rows_match_atn_g32_and_replay_full_state() {
    small(AttentionKind::GatedDelta)
}
#[test]
#[ignore = "requires qualified G32/MMQ and real FP16 causal KV providers"]
fn hybrid_causal_small_rows_match_atn_g32_and_replay_valid_kv() {
    small(AttentionKind::Causal)
}
#[test]
#[ignore = "requires qualified MMQ prefill and G32 decode CUDA providers"]
fn hybrid_gdn_mmq_prefill_then_g32_replay_preserves_full_state() {
    large(AttentionKind::GatedDelta)
}
#[test]
#[ignore = "requires qualified MMQ prefill and real FP16 causal KV providers"]
fn hybrid_causal_mmq_prefill_then_g32_replay_preserves_valid_kv() {
    large(AttentionKind::Causal)
}
