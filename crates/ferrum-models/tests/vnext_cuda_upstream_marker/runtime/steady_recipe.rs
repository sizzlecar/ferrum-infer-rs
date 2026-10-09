//! Actual upstream all-rows arithmetic, independent histories and full state.
use super::*;
use ferrum_types::InvocationPreparationStrategy;
use std::cell::RefCell;

#[derive(Default)]
struct PreparationSink {
    identity: RefCell<Vec<InvocationPreparationStats>>,
    recipe: RefCell<Vec<SteadyRecipePreparationStats>>,
}
impl InvocationPreparationSink for PreparationSink {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.identity.borrow_mut().push(stats);
    }

    fn record_steady_recipe(&self, stats: SteadyRecipePreparationStats) {
        self.recipe.borrow_mut().push(stats);
    }
}

fn compare(kind: AttentionKind, participants: u32) {
    let fixtures = std::array::from_fn::<_, 3, _>(|_| {
        Fixture::for_family(kind, true, participants, Family::extra_all_rows(kind, 2052))
    });
    let strategies = [
        InvocationPreparationStrategy::Full,
        InvocationPreparationStrategy::IdentityProjection,
        InvocationPreparationStrategy::SteadyRecipe,
    ];
    let tokens: Vec<Arc<[u32]>> = (0..participants)
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
            .map(|(index, tokens)| {
                fixture.admit_with_ceiling(
                    &format!("steady-recipe-{index}"),
                    Arc::from(&tokens[..1]),
                    17,
                )
            })
            .collect::<Vec<_>>()
    });
    let sinks: [PreparationSink; 3] = std::array::from_fn(|_| PreparationSink::default());
    let mut entry_owners: [BTreeMap<
        DeviceReusableExecutionProgramId,
        DeviceReusableExecutionEntryIdentity,
    >; 3] = std::array::from_fn(|_| BTreeMap::new());
    let mut entry_continuity_checks = 0_u64;
    for position in 0..17 {
        for (fixture, group) in fixtures.iter().zip(&sessions) {
            for (session, tokens) in group.iter().zip(&tokens) {
                fixture.extend(session, Arc::from(&tokens[..=position]));
            }
        }
        let path = if position < 2 {
            Path::Warm
        } else {
            Path::Replay
        };
        let recipe_record_start = sinks[2].recipe.borrow().len();
        let observations: Vec<_> = strategies
            .iter()
            .enumerate()
            .map(|(index, &strategy)| {
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
            .collect();
        observations[0].assert_same(&observations[1]);
        observations[0].assert_same(&observations[2]);
        observations[2].dump(kind, participants, position..position + 1, path);
        let records = sinks[2].recipe.borrow();
        let fresh = &records[recipe_record_start..];
        assert!(fresh.iter().all(|r| r.invalid_nodes == 0));
        if position >= 2 {
            assert!(
                !fresh.is_empty(),
                "resident dispatch must report actual recipe attempts"
            );
            let prepared = fresh.iter().map(|r| r.prepared_nodes).sum::<u64>();
            if kind == AttentionKind::GatedDelta {
                // The first resident binding attempt may only declare a cold
                // candidate. It cannot be used until this wave is terminal.
                if position > 2 {
                    assert!(prepared > 0,
                        "actual upstream GDN must use fresh checked recipe patches after terminal cold preparation");
                }
            } else {
                assert_eq!(prepared, 0, "causal provider remains the original fallback");
                assert!(fresh.iter().any(|r| r.fallback_nodes > 0));
            }
            for (index, fixture) in fixtures.iter().enumerate() {
                let catalog = fixture.lane.reusable_execution_catalog().unwrap();
                for program in catalog
                    .programs()
                    .iter()
                    .filter(|program| program.is_determinism_ready())
                {
                    let token = fixture
                        .lane
                        .reusable_execution_entry_identity(program.program_id())
                        .unwrap()
                        .expect("complete CUDA program has an exact resident entry token");
                    if let Some(previous) =
                        entry_owners[index].insert(program.program_id().clone(), token.clone())
                    {
                        assert!(
                            previous.same_entry(&token),
                            "unchanged ReplayOnly entry must retain its exact owner"
                        );
                        entry_continuity_checks += 1;
                    }
                }
            }
        }
    }
    assert!(entry_continuity_checks > 0);
    for (index, sink) in sinks.iter().enumerate() {
        let records = sink.identity.borrow();
        assert_eq!(records.len(), 17);
        assert!(records.iter().all(|r| r.parts_materialized == 0));
        if index == 0 {
            assert!(records.iter().all(|r| r.projected_identities == 0));
        } else {
            assert!(records.iter().all(|r| r.projected_identities > 0));
        }
        if index < 2 {
            assert!(sink.recipe.borrow().iter().all(|r| r.considered_nodes == 0
                && r.prepared_nodes == 0
                && r.fallback_nodes == 0
                && r.invalid_nodes == 0));
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "kind":"steady_recipe_actual_cuda", "attention":format!("{kind:?}"),
            "profile":"upstream_extra_all_rows", "histories":3,
            "waves_per_history":17, "participants":participants, "forced_replay_waves_per_history":15,
            "recipe_required_replay_waves":if kind == AttentionKind::GatedDelta {14} else {0},
            "full_output_and_state_equal":true,
            "prepared_nodes":sinks[2].recipe.borrow().iter().map(|r|r.prepared_nodes).sum::<u64>(),
            "fallback_nodes":sinks[2].recipe.borrow().iter().map(|r|r.fallback_nodes).sum::<u64>(),
            "invalid_nodes":sinks[2].recipe.borrow().iter().map(|r|r.invalid_nodes).sum::<u64>(),
            "entry_continuity_checks":entry_continuity_checks,
            "eviction_recapture_claim":false,
            "prepared_count_scope":"fresh checked resources plus successful patch encoding; terminal success independently checked by fixture",
            "physical_extent_growth_claim":false
        })
    );
    for group in sessions {
        for session in group {
            session.try_complete().unwrap();
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA, actual upstream GDN and resident replay"]
fn steady_recipe_gdn_matches_full_and_ip_across_committed_frontiers() {
    compare(AttentionKind::GatedDelta, 32);
}

#[test]
#[ignore = "requires exclusive CUDA, actual causal fallback and resident replay"]
fn steady_recipe_causal_fallback_matches_full_and_ip_across_kv_block_boundary() {
    compare(AttentionKind::Causal, 2);
}
