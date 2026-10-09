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
                fixture.admit_with_ceiling(
                    &format!("borrowed-dependency-{index}"),
                    Arc::from(&t[..1]),
                    17,
                )
            })
            .collect::<Vec<_>>()
    });
    let sinks = [
        PreparationSink::default(),
        PreparationSink::default(),
        PreparationSink::default(),
    ];
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
            InvocationPreparationStrategy::BorrowedDependency,
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
    let ip_records = sinks[1].0.borrow();
    let records = sinks[2].0.borrow();
    for arm in [&*full_records, &*ip_records, &*records] {
        assert_eq!(arm.len(), tokens[0].len());
        assert!(arm.iter().all(|r| r.parts_materialized == 0));
    }
    assert!(full_records
        .iter()
        .all(|r| r.projected_identities == 0 && r.borrowed_dependency_comparisons == 0));
    assert!(ip_records
        .iter()
        .all(|r| r.projected_identities > 0 && r.borrowed_dependency_comparisons == 0));
    assert!(records.iter().all(|r| r.projected_identities > 0));
    assert!(
        records
            .iter()
            .skip(2)
            .all(|r| r.borrowed_dependency_comparisons > 0),
        "every forced replay must exercise later-participant borrowed dependency comparison"
    );
    println!(
        "{}",
        serde_json::json!({
            "kind":"borrowed_dependency_actual_cuda", "attention":format!("{kind:?}"),
            "waves":records.len(), "participants":2,
            "projected_identities":records.iter().map(|r| r.projected_identities).sum::<u64>(),
            "parts_materialized_through_dispatch_return":records.iter().map(|r| r.parts_materialized).sum::<u64>(),
            "borrowed_dependency_comparisons":records.iter().map(|r| r.borrowed_dependency_comparisons).sum::<u64>(),
            "per_wave_borrowed_dependency_comparisons":records.iter().map(|r| r.borrowed_dependency_comparisons).collect::<Vec<_>>(),
            "per_wave_projected_identities":records.iter().map(|r| r.projected_identities).collect::<Vec<_>>(),
            "full_borrowed_dependency_comparisons":full_records.iter().map(|r| r.borrowed_dependency_comparisons).sum::<u64>(),
            "ip_borrowed_dependency_comparisons":ip_records.iter().map(|r| r.borrowed_dependency_comparisons).sum::<u64>(),
            "full_and_ip_output_and_state_equal":true,
            "comparison_scope":"later participants in successfully issued fresh dependency authorities, through dispatch return; not GPU completions",
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
fn borrowed_dependency_gdn_matches_full_and_ip_across_committed_frontiers() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA, actual FP16 KV providers and resident replay"]
fn borrowed_dependency_causal_matches_full_and_ip_across_kv_block_boundary() {
    compare(AttentionKind::Causal);
}
