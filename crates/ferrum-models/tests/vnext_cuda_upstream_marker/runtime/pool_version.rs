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

fn compare(kind: AttentionKind) {
    let fixtures = [
        Fixture::for_attention(kind, true, 2),
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
                fixture.admit_with_ceiling(&format!("pool-version-{index}"), Arc::from(&t[..1]), 17)
            })
            .collect::<Vec<_>>()
    });
    let sinks = std::array::from_fn::<_, 3, _>(|_| PreparationSink::default());
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
        let outputs = [
            InvocationPreparationStrategy::Full,
            InvocationPreparationStrategy::IdentityProjection,
            InvocationPreparationStrategy::PoolVersion,
        ]
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
        outputs[0].assert_same(&outputs[2]);
    }
    let full_records = sinks[0].0.borrow();
    assert!(full_records.iter().all(|r| r.projected_identities == 0
        && r.parts_materialized == 0
        && r.pool_version_hits == 0
        && r.pool_version_proofs == 0
        && r.pool_version_fallbacks == 0));
    let ip_records = sinks[1].0.borrow();
    assert!(ip_records.iter().all(|r| r.pool_version_hits == 0
        && r.pool_version_proofs == 0
        && r.pool_version_fallbacks == 0));
    assert!(ip_records.iter().all(|r| r.parts_materialized == 0));
    let records = sinks[2].0.borrow();
    assert_eq!(records.len(), 17);
    assert!(records.iter().all(|r| r.projected_identities > 0));
    assert!(
        records.iter().all(|r| r.parts_materialized == 0),
        "successful Off dispatch must preserve unobserved compiled projections"
    );
    assert!(
        records
            .iter()
            .skip(2)
            .all(|r| r.pool_version_hits > 0 && r.pool_version_proofs > 0),
        "forced resident replay must exercise the versioned retained pool proof"
    );
    println!(
        "{}",
        serde_json::json!({
            "kind":"pool_version_actual_cuda", "attention":format!("{kind:?}"),
            "waves":records.len(), "participants":2,
            "projected_identities":records.iter().map(|r| r.projected_identities).sum::<u64>(),
            "parts_materialized_through_dispatch_return":records.iter().map(|r| r.parts_materialized).sum::<u64>(),
            "pool_version_proofs":records.iter().map(|r| r.pool_version_proofs).sum::<u64>(),
            "pool_version_fallbacks":records.iter().map(|r| r.pool_version_fallbacks).sum::<u64>(),
            "per_wave_pool_version_proofs":records.iter().map(|r| r.pool_version_proofs).collect::<Vec<_>>(),
            "per_wave_pool_version_fallbacks":records.iter().map(|r| r.pool_version_fallbacks).collect::<Vec<_>>(),
            "pool_version_hits":records.iter().map(|r| r.pool_version_hits).sum::<u64>(),
            "per_wave_pool_version_hits":records.iter().map(|r| r.pool_version_hits).collect::<Vec<_>>(),
            "full_and_ip_output_and_state_equal":true,
            "kv_boundary_scope":"logical KV block boundary, not physical pool extent growth"
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
fn pool_version_gdn_matches_full_and_ip_across_committed_frontiers() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA, actual FP16 KV providers and resident replay"]
fn pool_version_causal_matches_full_and_ip_across_kv_block_boundary() {
    compare(AttentionKind::Causal);
}
