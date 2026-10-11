//! Complete provider state/replay checks for newly selected partial batches.
//! The Z control is used only where its arithmetic is unchanged. New widths
//! compare eager/replay of this policy; actual-pack/F64 is a separate gate.
use super::*;

fn decode(kind: AttentionKind, participants: u32, unchanged: bool) {
    let old = Fixture::for_family(
        kind,
        false,
        participants,
        if unchanged {
            Family::extra_prefill(kind, 2052)
        } else {
            Family::extra_all_rows(kind, 2052)
        },
    );
    let new = Fixture::for_family(kind, true, participants, Family::extra_all_rows(kind, 2052));
    let tokens: Vec<Arc<[u32]>> = (0..participants)
        .map(|p| (0..4).map(|i| ((i * 7 + p * 3 + 1) % 32) as u32).collect())
        .collect();
    let sessions = [&old, &new].map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, t)| fixture.admit(&format!("extra-all-rows-{p}"), t.clone()))
            .collect::<Vec<_>>()
    });
    for position in 0..4 {
        let range = position..position + 1;
        let path = if position < 2 {
            Path::Warm
        } else {
            Path::Replay
        };
        let expected = old.execute_participants(&sessions[0], &tokens, range.clone(), Path::Eager);
        let actual = new.execute_participants(&sessions[1], &tokens, range.clone(), path);
        expected.assert_same(&actual);
        actual.dump(kind, participants, range.clone(), path);
        println!(
            "{}",
            serde_json::json!({"kind":"extra_all_rows_decode_control",
                "attention":format!("{kind:?}"),"actual_rows":participants,
                "range":[range.start,range.end],"path":format!("{path:?}"),
                "reference":if unchanged {"frozen_z_policy"} else {"same_policy_eager"},
                "nonzero_six_format_weights":true,"all_output_state_bytes_equal":true,
                "primitive_oracle_claim":false})
        );
    }
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

fn prefill_control(kind: AttentionKind, rows: usize) {
    let old = Fixture::for_family(kind, false, 1, Family::extra_prefill(kind, 2052));
    let new = Fixture::for_family(kind, true, 1, Family::extra_all_rows(kind, 2052));
    let tokens: Arc<[u32]> = (0..rows + 4).map(|i| ((i * 7 + 1) % 32) as u32).collect();
    let sessions = [&old, &new].map(|f| f.admit("extra-all-rows-prefill", tokens.clone()));
    // Causal multi-token prefill may be an eager boundary. Stable single-token
    // continuation must really reach ReplayOnly, preserving the existing rule.
    for (range, path) in std::iter::once((0..rows, Path::Warm)).chain((rows..rows + 4).map(|i| {
        (
            i..i + 1,
            if i < rows + 2 {
                Path::Warm
            } else {
                Path::Replay
            },
        )
    })) {
        let execute = |fixture: &Fixture, session, path| {
            fixture.execute_participants(
                std::slice::from_ref(session),
                std::slice::from_ref(&tokens),
                range.clone(),
                path,
            )
        };
        let expected = execute(&old, &sessions[0], Path::Eager);
        let actual = execute(&new, &sessions[1], path);
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
                serde_json::json!({"participant":participant,"value":name,"bytes":bytes.len(),
                "sha256":format!("{:x}",Sha256::digest(bytes))})
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({"kind":"extra_all_rows_unchanged_prefill_control",
                "attention":format!("{kind:?}"),"prefill_rows":rows,
                "actual_rows":range.len(),"range":[range.start,range.end],
                "path":format!("{path:?}"),"reference":"frozen_z_policy",
                "all_output_state_bytes_equal":true,"values":values})
        );
    }
    for session in sessions {
        session.try_complete().unwrap();
    }
}

fn verify(kind: AttentionKind) {
    for rows in [2, 3, 5, 7, 9, 15, 17, 31] {
        decode(kind, rows, false);
    }
    for rows in [1, 4, 8, 16, 32] {
        decode(kind, rows, true);
    }
    for rows in [33, 2048] {
        prefill_control(kind, rows);
    }
}

#[test]
#[ignore = "requires actual all-rows MMQ providers and admitted GDN/FFN state"]
fn extra_all_rows_gdn_ffn_preserves_z_controls_and_partial_batch_replay_state() {
    verify(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires actual all-rows MMQ providers and real FP16 causal KV"]
fn extra_all_rows_causal_ffn_preserves_z_controls_and_partial_batch_replay_kv() {
    verify(AttentionKind::Causal);
}

#[test]
#[ignore = "requires one actual Plan and two independent CUDA execution lanes"]
fn extra_all_rows_shared_plan_two_streams_preserve_state_and_replay() {
    super::two_streams::verify_extra_all_rows(AttentionKind::GatedDelta);
    super::two_streams::verify_extra_all_rows(AttentionKind::Causal);
}
