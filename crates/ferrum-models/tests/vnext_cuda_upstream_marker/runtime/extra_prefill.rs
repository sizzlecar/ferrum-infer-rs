//! Nonzero six-format large prefill, decode continuation and true replay.
//! Primitive accuracy is separately qualified with the actual native pack/F64
//! oracle. Here the independent X control applies only to unchanged small M.
use super::*;

fn compare_small_x(kind: AttentionKind) {
    for participants in [1_u32, 8, 32] {
        let old = Fixture::for_family(kind, false, participants, Family::extra(kind));
        let new = Fixture::for_family(kind, true, participants, Family::extra_prefill(kind, 2052));
        let tokens: Vec<Arc<[u32]>> = (0..participants)
            .map(|p| (0..4).map(|i| ((i * 7 + p * 3 + 1) % 32) as u32).collect())
            .collect();
        let sessions = [&old, &new].map(|fixture| {
            tokens
                .iter()
                .enumerate()
                .map(|(p, t)| fixture.admit(&format!("extra-prefill-small-{p}"), t.clone()))
                .collect::<Vec<_>>()
        });
        for position in 0..4 {
            let range = position..position + 1;
            let path = if position < 2 {
                Path::Warm
            } else {
                Path::Replay
            };
            let expected =
                old.execute_participants(&sessions[0], &tokens, range.clone(), Path::Eager);
            let actual = new.execute_participants(&sessions[1], &tokens, range.clone(), path);
            expected.assert_same(&actual);
            println!(
                "{}",
                serde_json::json!({"kind":"extra_prefill_independent_x_small_control",
                    "attention":format!("{kind:?}"),"actual_rows":participants,
                    "range":[range.start,range.end],"path":format!("{path:?}"),
                    "nonzero_six_format_weights":true,"all_output_state_bytes_equal":true})
            );
        }
        for group in sessions {
            for session in group {
                session.try_complete().unwrap();
            }
        }
    }
}

fn verify(kind: AttentionKind) {
    compare_small_x(kind);
    for rows in [33_usize, 54, 2048] {
        let eager = Fixture::for_family(kind, false, 1, Family::extra_prefill(kind, 2052));
        let replay = Fixture::for_family(kind, true, 1, Family::extra_prefill(kind, 2052));
        let tokens: Arc<[u32]> = (0..rows + 4).map(|i| ((i * 7 + 1) % 32) as u32).collect();
        let sessions =
            [&eager, &replay].map(|fixture| fixture.admit("extra-large-prefill", tokens.clone()));
        // A causal multi-token prefix may be an eager boundary. Only the
        // following single-token topology is required to become ReplayOnly.
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
                std::slice::from_ref(&sessions[0]),
                std::slice::from_ref(&tokens),
                range.clone(),
                Path::Eager,
            );
            let actual = replay.execute_participants(
                std::slice::from_ref(&sessions[1]),
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
            let values: Vec<_> = actual
                .values
                .iter()
                .map(|((participant, name), bytes)| {
                    serde_json::json!({"participant":participant,"value":name,
                    "bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))})
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({"kind":"extra_large_prefill_provider_observation",
                    "attention":format!("{kind:?}"),"prefill_rows":rows,
                    "actual_rows":range.len(),"range":[range.start,range.end],
                    "path":format!("{path:?}"),"all_output_state_bytes_equal":true,
                    "primitive_oracle_claim":false,"values":values})
            );
        }
        for session in sessions {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires the new extra MMQ prefill v2 artifact and actual CUDA providers"]
fn extra_large_prefill_gdn_ffn_preserves_x_small_outputs_and_full_replay_state() {
    verify(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires extra MMQ prefill v2 and real FP16 causal KV providers"]
fn extra_large_prefill_causal_ffn_preserves_x_small_outputs_and_full_replay_kv() {
    verify(AttentionKind::Causal);
}

#[test]
#[ignore = "requires one real Plan and extra prefill flags on two CUDA streams"]
fn extra_large_prefill_shared_plan_two_streams_preserve_state_and_replay() {
    super::two_streams::verify_extra_prefill(AttentionKind::GatedDelta);
    super::two_streams::verify_extra_prefill(AttentionKind::Causal);
}
