use super::*;
use ferrum_types::InvocationPreparationStrategy;
use std::cell::RefCell;

#[derive(Default)]
struct PreparationSink(RefCell<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for PreparationSink {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.borrow_mut().push(stats);
    }
}

fn compare(kind: AttentionKind, candidate: InvocationPreparationStrategy) {
    let fixtures = [
        Fixture::for_attention(kind, true, 2),
        Fixture::for_attention(kind, true, 2),
    ];
    let tokens: Vec<Arc<[u32]>> = (0..2)
        .map(|participant| {
            (0..17)
                .map(|position| ((position * 7 + participant * 3 + 1) % 32) as u32)
                .collect()
        })
        .collect();
    let sessions = fixtures.each_ref().map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(index, t)| {
                fixture.admit_with_ceiling(
                    &format!("identity-projection-{index}"),
                    Arc::from(&t[..1]),
                    17,
                )
            })
            .collect::<Vec<_>>()
    });
    let sinks = [PreparationSink::default(), PreparationSink::default()];
    for position in 0..17 {
        for (fixture, group) in fixtures.iter().zip(&sessions) {
            for (session, token) in group.iter().zip(&tokens) {
                fixture.extend(session, Arc::from(&token[..=position]));
            }
        }
        let path = if position < 2 {
            Path::Warm
        } else {
            Path::Replay
        };
        let outputs = [InvocationPreparationStrategy::Full, candidate]
            .into_iter()
            .enumerate()
            .map(|(index, strategy)| {
                let fixture = &fixtures[index];
                fixture.execute_participants_with_preparation(
                    &fixture.lane,
                    &fixture.reaper,
                    &sessions[index],
                    &tokens,
                    position..position + 1,
                    path,
                    strategy,
                    &sinks[index],
                )
            })
            .collect::<Vec<_>>();
        outputs[0].assert_same(&outputs[1]);
    }
    let full_records = sinks[0].0.borrow();
    assert!(full_records
        .iter()
        .all(|r| r.projected_identities == 0 && r.parts_materialized == 0));
    let records = sinks[1].0.borrow();
    assert_eq!(records.len(), 17);
    assert!(records.iter().all(|r| r.projected_identities > 0));
    assert!(
        records.iter().all(|r| r.parts_materialized == 0),
        "successful Off dispatch must preserve unobserved compiled projections"
    );
    assert!(full_records
        .iter()
        .all(|r| r.agreement_builds == 0 && r.agreement_reuses == 0 && r.agreement_fallbacks == 0));
    if candidate == InvocationPreparationStrategy::WaveAgreement {
        assert!(
            records
                .iter()
                .all(|r| r.agreement_builds > 0 && r.agreement_reuses > 0),
            "each fresh wave must actually publish and reuse participant agreement"
        );
    } else {
        assert!(records.iter().all(|r| r.agreement_builds == 0
            && r.agreement_reuses == 0
            && r.agreement_fallbacks == 0));
    }
    println!(
        "{}",
        serde_json::json!({
            "kind":"identity_projection_actual_cuda", "attention":format!("{kind:?}"),
            "preparation_strategy":candidate.as_runtime_value(),
            "waves":records.len(), "participants":2,
            "projected_identities":records.iter().map(|r| r.projected_identities).sum::<u64>(),
            "parts_materialized_through_dispatch_return":records.iter().map(|r| r.parts_materialized).sum::<u64>(),
            "agreement_builds":records.iter().map(|r| r.agreement_builds).sum::<u64>(),
            "agreement_reuses":records.iter().map(|r| r.agreement_reuses).sum::<u64>(),
            "agreement_fallbacks":records.iter().map(|r| r.agreement_fallbacks).sum::<u64>(),
            "full_output_and_state_equal":true
        })
    );
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA, actual GDN/FFN providers and resident replay"]
fn identity_projection_gdn_matches_full_across_committed_frontiers() {
    compare(
        AttentionKind::GatedDelta,
        InvocationPreparationStrategy::IdentityProjection,
    );
}

#[test]
#[ignore = "requires exclusive CUDA, actual FP16 KV providers and resident replay"]
fn identity_projection_causal_matches_full_across_kv_block_boundary() {
    compare(
        AttentionKind::Causal,
        InvocationPreparationStrategy::IdentityProjection,
    );
}

#[test]
#[ignore = "requires exclusive CUDA, actual GDN/FFN providers and resident replay"]
fn wave_agreement_gdn_matches_full_across_committed_frontiers() {
    compare(
        AttentionKind::GatedDelta,
        InvocationPreparationStrategy::WaveAgreement,
    );
}

#[test]
#[ignore = "requires exclusive CUDA, actual FP16 KV providers and resident replay"]
fn wave_agreement_causal_matches_full_across_kv_block_boundary() {
    compare(
        AttentionKind::Causal,
        InvocationPreparationStrategy::WaveAgreement,
    );
}
