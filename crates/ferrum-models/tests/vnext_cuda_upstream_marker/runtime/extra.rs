//! Real six-format providers: independent eager/replay state histories, then
//! shared Plan flags across two CUDA streams. No primitive timing substitute.
use super::*;

fn verify(kind: AttentionKind) {
    // Qualified algorithm switches plus strict extra-format fallbacks on
    // partial M and M>32. Width refers to the actual one-token participants.
    for participants in [1_u32, 3, 4, 8, 9, 16, 32, 33] {
        let eager = Fixture::for_family(kind, false, participants, Family::extra(kind));
        let replay = Fixture::for_family(kind, true, participants, Family::extra(kind));
        let tokens: Vec<Arc<[u32]>> = (0..participants)
            .map(|p| (0..44).map(|i| ((i * 7 + p * 3 + 1) % 32) as u32).collect())
            .collect();
        let sessions = [&eager, &replay].map(|fixture| {
            tokens
                .iter()
                .enumerate()
                .map(|(p, t)| fixture.admit(&format!("extra-provider-{p}"), t.clone()))
                .collect::<Vec<_>>()
        });
        // Two-token causal waves retain their true EagerBoundary. GDN uses
        // the real adaptive topology; no cross-width equality is claimed.
        let boundary = if kind == AttentionKind::Causal {
            Path::EagerBoundary
        } else {
            Path::Warm
        };
        for (range, path) in (0..32)
            .step_by(4)
            .map(|i| (i..i + 4, if i == 0 { Path::Warm } else { boundary }))
            .chain((32..36).map(|i| (i..i + 1, if i < 34 { Path::Warm } else { Path::Replay })))
            .chain([(36..38, Path::Warm), (38..40, boundary)])
        {
            let expected =
                eager.execute_participants(&sessions[0], &tokens, range.clone(), Path::Eager);
            let actual = replay.execute_participants(&sessions[1], &tokens, range.clone(), path);
            expected.assert_same(&actual);
            for participant in 0..participants {
                assert_eq!(
                    actual.values[&(participant, "output".into())].len(),
                    range.len() * HIDDEN as usize * 4
                );
                if kind == AttentionKind::Causal {
                    assert_eq!(
                        actual.values[&(participant, "state.kv".into())].len(),
                        range.end * 1024
                    );
                }
            }
            println!(
                "{}",
                serde_json::json!({"kind":"extra_provider_same_policy_full_state",
                "attention":format!("{kind:?}"),"participants":participants,
                "tokens_per_participant":range.len(),"wave_rows":participants as usize*range.len(),
                "source_start":range.start,"source_end":range.end,"path":format!("{path:?}"),
                "all_output_state_bytes_equal":true,"cross_width_bitwise_claim":false})
            );
            actual.dump(kind, participants, range, path);
        }
        for group in sessions {
            for session in group {
                session.try_complete().unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires actual six-format locked native artifacts and exclusive CUDA"]
fn extra_upstream_gdn_and_swiglu_preserve_full_state_across_eager_and_replay() {
    verify(AttentionKind::GatedDelta)
}
#[test]
#[ignore = "requires actual six-format providers and real FP16 causal KV"]
fn extra_upstream_causal_and_swiglu_preserve_valid_kv_across_eager_and_replay() {
    verify(AttentionKind::Causal)
}
#[test]
#[ignore = "requires actual extra/base operator flags retained by one Plan on two CUDA streams"]
fn extra_upstream_shared_plan_two_streams_preserve_state_and_retained_dependencies() {
    super::two_streams::verify_extra(AttentionKind::GatedDelta);
    super::two_streams::verify_extra(AttentionKind::Causal);
}
