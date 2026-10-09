//! Real provider admission and full residual/state readback for the explicit
//! prefill profile. This tests same-policy execution, not G32 bitwise equivalence.
use super::*;

fn verify(kind: AttentionKind) {
    for rows in [33_usize, 155, 2048] {
        // The state capacity includes the subsequent decode probes. Maximum
        // scheduled tokens remains2048; no fake physical storage or lease.
        let eager = Fixture::for_family(kind, false, 1, Family::prefill(kind, 2052));
        let adaptive = Fixture::for_family(kind, true, 1, Family::prefill(kind, 2052));
        let tokens: Arc<[u32]> = (0..rows + 4).map(|i| ((i * 7 + 1) % 32) as u32).collect();
        let expected_session = eager.admit("prefill-eager", tokens.clone());
        let actual_session = adaptive.admit("prefill-adaptive", tokens.clone());
        for (range, path) in
            std::iter::once((0..rows, Path::Warm)).chain((rows..rows + 4).map(|position| {
                (
                    position..position + 1,
                    if position < rows + 2 {
                        Path::Warm
                    } else {
                        Path::Replay
                    },
                )
            }))
        {
            let expected = eager.execute_participants(
                std::slice::from_ref(&expected_session),
                std::slice::from_ref(&tokens),
                range.clone(),
                Path::Eager,
            );
            let actual = adaptive.execute_participants(
                std::slice::from_ref(&actual_session),
                std::slice::from_ref(&tokens),
                range.clone(),
                path,
            );
            expected.assert_same(&actual);
            assert_eq!(
                actual.values[&(0, "output".to_owned())].len(),
                range.len() * HIDDEN as usize * 4
            );
            if kind == AttentionKind::Causal {
                assert_eq!(
                    actual.values[&(0, "state.kv".to_owned())].len(),
                    range.end * 1024
                );
            }
            let records:Vec<_>=actual.values.iter().map(|((participant,value),bytes)| serde_json::json!({"participant":participant,"value":value,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))})).collect();
            println!(
                "{}",
                serde_json::json!({"kind":"marker_prefill_provider_observation","attention":format!("{kind:?}"),"local_prefill_rows":rows,"range":[range.start,range.end],"path":format!("{path:?}"),"full_bytes_equal":true,"values":records})
            );
        }
        expected_session.try_complete().unwrap();
        actual_session.try_complete().unwrap();
    }
}

#[test]
#[ignore = "requires CUDA and the source-built explicit MMQ prefill artifact"]
fn upstream_prefill_gdn_and_swiglu_admit_large_rows_and_preserve_residual_state() {
    verify(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires CUDA, explicit MMQ prefill artifact and real FP16 KV provider"]
fn upstream_prefill_causal_and_swiglu_admit_large_rows_and_preserve_residual_kv() {
    verify(AttentionKind::Causal);
}
