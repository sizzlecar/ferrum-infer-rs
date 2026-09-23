use super::*;
use ferrum_interfaces::execution_cost::CanonicalWaveCostShape;
use std::cell::RefCell;

struct OrderedReplay(RefCell<Vec<Vec<CandidateWork>>>);
impl PlanningShapeResolver for OrderedReplay {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        let mut contexts: Vec<_> = query
            .snapshot
            .requests
            .iter()
            .map(|row| row.context_tokens)
            .collect();
        let mut generated: Vec<_> = query
            .snapshot
            .requests
            .iter()
            .map(|row| row.timing.committed_tokens)
            .collect();
        for wave in query.prior_waves {
            poll()?;
            for work in &wave.work {
                let i = query
                    .snapshot
                    .requests
                    .iter()
                    .position(|row| row.key == work.key)
                    .unwrap();
                assert_eq!(work.action, WaveAction::Decode);
                contexts[i] += 1;
                generated[i] += 1;
            }
        }
        for row in query.rows {
            let i = query
                .snapshot
                .requests
                .iter()
                .position(|old| old.key == row.request.key)
                .unwrap();
            assert_eq!(
                contexts[i], row.request.context_tokens,
                "ordered prior waves must describe the same projected frontier"
            );
            assert_eq!(generated[i], row.request.timing.committed_tokens);
        }
        self.0.borrow_mut().push(
            query
                .prior_waves
                .iter()
                .flat_map(|wave| wave.work.clone())
                .collect(),
        );
        TestResolver.resolve(query, poll)
    }
}

#[test]
fn candidate_and_final_replay_receive_complete_ordered_parent_waves() {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(3);
    request.timing.budgets.itl_ns = n64(10);
    request.timing.budgets.tpot_ns = n64(10);
    let mut snapshot = snapshot(vec![request]);
    snapshot.observed_at_ns = 95;
    snapshot.scope.horizon_end_ns = 115;
    let resolver = OrderedReplay(RefCell::new(Vec::new()));
    let (selected, witness, _) = feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &resolver,
        &mut Clock(95),
    ));
    assert_eq!(witness.predicted_output_tokens, 2);
    let seen = resolver.0.borrow();
    assert!(seen.iter().any(Vec::is_empty));
    assert!(seen.iter().any(|prior| prior == &selected.candidate.work));
    assert_eq!(
        seen.last().unwrap(),
        &selected.candidate.work,
        "the last lookup replays the complete final witness, not a fresh state"
    );
}

enum OrderBehavior {
    Reverse,
    ReplaceAction,
    ReplaceIncarnation,
    Duplicate,
    SwallowBudget,
}
struct Order(OrderBehavior);
impl PlanningShapeResolver for Order {
    fn order_work(
        &self,
        _: &SchedulerSnapshot,
        work: &mut [CandidateWork],
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason> {
        match self.0 {
            OrderBehavior::Reverse => work.reverse(),
            OrderBehavior::ReplaceAction => {
                work[0].action = WaveAction::Prefill {
                    offset: 0,
                    count: n32(1),
                }
            }
            OrderBehavior::ReplaceIncarnation => work[0].key.incarnation += 1,
            OrderBehavior::Duplicate => work[0] = work[1].clone(),
            OrderBehavior::SwallowBudget => {
                let _ = poll();
                let _ = poll();
            }
        }
        Ok(())
    }
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        TestResolver.resolve(query, poll)
    }
}

#[test]
fn physical_order_is_a_complete_work_permutation_and_cannot_swallow_budget() {
    let snapshot = snapshot(vec![decode(1), decode(2)]);
    let original: Vec<_> = snapshot
        .requests
        .iter()
        .map(|row| CandidateWork {
            key: row.key.clone(),
            action: WaveAction::Decode,
        })
        .collect();
    let mut reversed = original.clone();
    shape::order_work(
        &snapshot,
        &mut reversed,
        &Order(OrderBehavior::Reverse),
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(reversed, original.iter().rev().cloned().collect::<Vec<_>>());
    for behavior in [
        OrderBehavior::ReplaceAction,
        OrderBehavior::ReplaceIncarnation,
        OrderBehavior::Duplicate,
    ] {
        assert_eq!(
            shape::order_work(
                &snapshot,
                &mut original.clone(),
                &Order(behavior),
                &mut || Ok(())
            ),
            Err(PlanningUnknownReason::InvalidShapeEvidence)
        );
    }
    let mut polls = 0;
    assert_eq!(
        shape::order_work(
            &snapshot,
            &mut original.clone(),
            &Order(OrderBehavior::SwallowBudget),
            &mut || {
                polls += 1;
                if polls == 2 {
                    Err(PlanningUnknownReason::ComputeBudgetExhausted)
                } else {
                    Ok(())
                }
            }
        ),
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    );
}

#[test]
fn physical_order_and_resolution_share_the_total_invocation_bound() {
    let settings = BoundedPlannerSettings {
        search: SloPlannerConfig {
            candidate_limit: nz(1),
            beam_width: nz(1),
            lookahead_waves: nz(1),
            ..Default::default()
        },
    };
    let resolver = Order(OrderBehavior::Reverse);
    let session = shape::ResolutionSession::new(&resolver, &settings);
    let snapshot = snapshot(vec![decode(1)]);
    let mut work = vec![CandidateWork {
        key: snapshot.requests[0].key.clone(),
        action: WaveAction::Decode,
    }];
    let limit = 16 + 2 * settings.search.lookahead_waves.get();
    for _ in 0..limit {
        session
            .order_work(&snapshot, &mut work, &mut || Ok(()))
            .unwrap();
    }
    assert_eq!(
        session.order_work(&snapshot, &mut work, &mut || Ok(())),
        Err(PlanningUnknownReason::SearchIncomplete)
    );
}
