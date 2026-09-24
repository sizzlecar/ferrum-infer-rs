//! Production Construct and final replay under a shared 2ms virtual clock.
//! No backend performance or empirical cost qualification is claimed.
use super::*;
use std::cell::{Cell, RefCell};

struct Context {
    now: Cell<u64>,
    begins: Cell<usize>,
    seen: RefCell<Vec<(usize, u32)>>,
    change_final_route: bool,
    callback_ns: u64,
}
impl Context {
    fn new() -> Self {
        Self {
            now: Cell::new(300_000),
            begins: Cell::new(0),
            seen: RefCell::new(Vec::new()),
            change_final_route: false,
            callback_ns: 180_000,
        }
    }
}
struct State<'a> {
    context: &'a Context,
    snapshot: &'a SchedulerSnapshot,
    epoch: usize,
    expected_offset: u32,
}
impl PlanningExecutionContext for Context {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        poll()?;
        let epoch = self.begins.get() + 1;
        self.begins.set(epoch);
        self.now.set(self.now.get() + 20_000);
        poll()?;
        Ok(Arc::new(State {
            context: self,
            snapshot,
            epoch,
            expected_offset: 0,
        }))
    }
}
impl<'epoch> PlanningExecutionState<'epoch> for State<'epoch> {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        let WaveAction::Prefill { offset, count } = input.work[0].action else {
            panic!("prefill fixture")
        };
        assert_eq!(offset, self.expected_offset, "siblings must project from the same immutable parent; children use their actual successor");
        self.context
            .seen
            .borrow_mut()
            .push((self.epoch, count.get()));
        self.context
            .now
            .set(self.context.now.get() + self.context.callback_ns);
        let mut canonical = TestResolver
            .resolve(
                &PlanningShapeQuery {
                    snapshot: self.snapshot,
                    prior_waves: &[],
                    kind: input.kind,
                    rows: input.rows,
                    recurrent_state_bytes: input.recurrent_state_bytes,
                },
                poll,
            )?
            .unwrap();
        if self.epoch > 1 && self.context.change_final_route {
            canonical.provider_signature[0] ^= 1;
        }
        Ok(Some(ProjectedExecution {
            ordered_work: input.work.to_vec(),
            canonical_domain: PlanningShapeDomain::Exact(canonical),
            statistical_evidence: None,
            successor: Arc::new(State {
                context: self.context,
                snapshot: self.snapshot,
                epoch: self.epoch,
                expected_offset: offset + count.get(),
            }),
        }))
    }
}
struct Clock<'a>(&'a Cell<u64>);
impl PlanningClock for Clock<'_> {
    fn now_ns(&mut self) -> u64 {
        self.0.get()
    }
    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        Some(PlanningBudgetWindow {
            started_at_ns: 0,
            deadline_ns: 2_000_000,
        })
    }
}
fn workload(chunks: &[u32]) -> SchedulerSnapshot {
    let mut p = prefill(1);
    p.timing.budgets.ttft_ns = n64(50_000_000);
    if let RequestPhaseView::Prefill(progress) = &mut p.phase {
        progress.total_prompt_tokens = n32(128);
        progress.executable_until = 128;
        progress.reference = Arc::new(PrefillReferenceWork {
            evaluation: Default::default(),
            version: 1,
            points: (0..=128)
                .map(|prompt_tokens| ReferenceWorkPoint {
                    prompt_tokens,
                    cumulative_work_ns: u64::from(prompt_tokens) * 100_000,
                })
                .collect(),
        });
    }
    let mut s = snapshot(vec![p]);
    s.observed_at_ns = 300_000;
    s.scope.horizon_end_ns = 50_000_000;
    s.capabilities.native_mixed = false;
    s.capabilities.prefill_alignment = n32(1);
    s.capabilities.prefill_chunk_sizes = chunks.iter().copied().map(n32).collect();
    s.capabilities.max_prefill_tokens_per_wave = n64(128);
    s
}
fn predicted_ns(shape: &WaveExecutionShape) -> Option<u64> {
    match shape.prefill_chunks.as_slice() {
        [chunk] => match chunk.count.get() {
            1 => Some(1_000_000),
            2 => Some(1_200_000),
            4 => Some(1_600_000),
            8 => Some(2_000_000),
            16 => Some(3_000_000),
            32 => Some(4_000_000),
            64 => Some(5_000_000),
            128 => Some(100_000_000), // Largest legal chunk violates TTFT.
            _ => None,
        },
        _ => None,
    }
}

#[test]
fn candidate_ranking_reaches_intermediate_progress_before_tiny_construct_chain() {
    let s = workload(&[1, 2, 4, 8, 16, 32, 64, 128]);
    let context = Context::new();
    let (selected, witness, search) = feasible(planner(3).propose_with_execution(
        &s,
        &Model(predicted_ns),
        &context,
        &mut Clock(&context.now),
    ));
    assert!(
        matches!(selected.candidate.work[0].action, WaveAction::Prefill { count, .. } if count.get() == 64)
    );
    assert_eq!(witness.predicted_output_tokens, 1);
    assert!(selected.final_replay_first_wave.is_some());
    assert!(context
        .seen
        .borrow()
        .iter()
        .any(|&(epoch, count)| epoch > 1 && count == 64));
    assert!(context.now.get() < 1_600_000);
    assert_eq!(search.phase, PlanningSearchPhase::Finalization);
}

#[test]
fn candidate_ranking_uses_actual_whole_wave_cost_not_a_fixed_chunk_preference() {
    let s = workload(&[128, 64, 32, 16, 8, 4, 2, 1]);
    let mut context = Context::new();
    // This case leaves genuine CPU room after the first complete incumbent.
    // The original 180us callback is retained in the reserve-boundary test below.
    context.callback_ns = 30_000;
    // Here a single large wave is efficient and meets the original TTFT.
    // The goal hint still starts at 64; the real cost selects its challenger.
    let fast_large = Model(|shape: &WaveExecutionShape| {
        if shape
            .prefill_chunks
            .iter()
            .any(|chunk| chunk.count.get() == 128)
        {
            Some(7_000_000)
        } else {
            predicted_ns(shape)
        }
    });
    let (selected, witness, _) = feasible(planner(3).propose_with_execution(
        &s,
        &fast_large,
        &context,
        &mut Clock(&context.now),
    ));
    assert!(
        matches!(selected.candidate.work[0].action, WaveAction::Prefill { count, .. } if count.get() == 128)
    );
    assert_eq!(witness.waves, 1);
    assert!(context
        .seen
        .borrow()
        .iter()
        .any(|&(epoch, count)| epoch > 1 && count == 128));
    assert!(context.now.get() < 1_600_000);
}

#[test]
fn candidate_ranking_keeps_complete_tail_when_improvement_exceeds_replay_reserve() {
    let s = workload(&[128, 64, 32, 16, 8, 4, 2, 1]);
    let context = Context::new(); // Original 180us physical projection callback.
    let fast_large = Model(|shape: &WaveExecutionShape| {
        if shape
            .prefill_chunks
            .iter()
            .any(|chunk| chunk.count.get() == 128)
        {
            Some(7_000_000)
        } else {
            predicted_ns(shape)
        }
    });
    let (first, witness, search) = feasible(planner(3).propose_with_execution(
        &s,
        &fast_large,
        &context,
        &mut Clock(&context.now),
    ));
    assert!(matches!(first.candidate.work[0].action,
        WaveAction::Prefill { count, .. } if count.get() == 64));
    // The goal scale is recomputed from the remaining work and wave slots:
    // 128/3 -> 64, then 64/2 -> 32, then the final 32. The complete tail,
    // including all three physical projections, must still fit fresh replay.
    assert_eq!(witness.waves, 3);
    let replay_chunks: Vec<_> = context
        .seen
        .borrow()
        .iter()
        .filter_map(|&(epoch, count)| (epoch > 1).then_some(count))
        .collect();
    assert_eq!(replay_chunks, [64, 32, 32]);
    assert_eq!(witness.predicted_output_tokens, 1);
    assert!(search.search_soft_stops > 0);
    assert!(search.replay_reserve_stops > 0);
    assert!(context.begins.get() > 1);
    assert!(first.final_replay_first_wave.is_some());
    assert!(context.now.get() < 1_600_000);
}

#[test]
fn candidate_ranking_never_promotes_unknown_cost_or_changed_final_route() {
    let s = workload(&[1, 2, 4, 8, 16, 32, 64, 128]);
    let context = Context::new();
    let missing = Model(|shape: &WaveExecutionShape| {
        if shape
            .prefill_chunks
            .iter()
            .any(|chunk| chunk.count.get() >= 64)
        {
            None
        } else {
            predicted_ns(shape)
        }
    });
    assert!(matches!(
        planner(3).propose_with_execution(&s, &missing, &context, &mut Clock(&context.now),),
        PlanningDecision::Unknown { .. }
    ));
    let mut changed = Context::new();
    changed.change_final_route = true;
    assert!(matches!(
        planner(3).propose_with_execution(
            &s,
            &Model(predicted_ns),
            &changed,
            &mut Clock(&changed.now),
        ),
        PlanningDecision::Unknown { .. }
    ));
    assert!(
        changed.begins.get() > 1,
        "rejection follows independent final begin"
    );
}

#[test]
fn candidate_ranking_opening_and_scales_keep_due_owner_and_declared_widths() {
    let mut due = prefill(1);
    due.timing.slo_failed = true;
    due.recovery_service = RecoveryServiceDebt::new(nz(1));
    due.recovery_service.bypass();
    let s = snapshot(vec![due, decode(2)]);
    let scope = PlanningObligationSet::capture(&s, 100).unwrap();
    let mut cursor = candidates::FrontierCursor::with_remaining_waves(
        &s,
        &s.requests,
        100,
        16,
        Some(&scope),
        Some(nz(3)),
        &mut || Ok(()),
    )
    .unwrap();
    let mut attempts = 0;
    let mut returned = false;
    while let Some(work) = cursor
        .next(&s, &s.requests, &mut attempts, 128, &mut || Ok(()))
        .unwrap()
    {
        returned = true;
        assert!(work.iter().any(|row| row.key == s.requests[0].key));
        let prefills = work
            .iter()
            .filter(|row| matches!(row.action, WaveAction::Prefill { .. }))
            .count();
        assert!(s.capabilities.prefill_batch_sizes.contains(&nz(prefills)));
    }
    assert!(returned);
    assert_eq!(
        s.requests[0].recovery_service.eligible_bypasses(),
        1,
        "pure proposals do not settle debt"
    );

    let mut two_only = snapshot(vec![prefill(1), prefill(2)]);
    two_only.capabilities.prefill_batch_sizes = vec![nz(2)];
    let mut cursor = candidates::FrontierCursor::with_remaining_waves(
        &two_only,
        &two_only.requests,
        100,
        16,
        None,
        Some(nz(3)),
        &mut || Ok(()),
    )
    .unwrap();
    let mut attempts = 0;
    let first = cursor
        .next(
            &two_only,
            &two_only.requests,
            &mut attempts,
            128,
            &mut || Ok(()),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        first.len(),
        2,
        "opening cannot invent an unsupported singleton bucket"
    );
}

#[test]
fn candidate_ranking_unknown_challenger_keeps_the_known_child() {
    let s = workload(&[1, 2, 4, 8, 16, 32, 64, 128]);
    let mut context = Context::new();
    // Leave time to really query the Unknown sibling after constructing a tail.
    // The original costly callback/soft-stop case has its own regression above.
    context.callback_ns = 30_000;
    let model = Model(|shape: &WaveExecutionShape| {
        if shape
            .prefill_chunks
            .iter()
            .any(|chunk| chunk.count.get() == 128)
        {
            None
        } else {
            predicted_ns(shape)
        }
    });
    let (selected, _, search) =
        feasible(planner(3).propose_with_execution(&s, &model, &context, &mut Clock(&context.now)));
    assert!(
        matches!(selected.candidate.work[0].action, WaveAction::Prefill { count, .. } if count.get() == 64)
    );
    assert!(search.cost_unknown_candidates > 0);
    assert!(context.begins.get() > 1);
}

#[test]
fn candidate_ranking_probe_cannot_extend_the_original_transaction_budget() {
    let s = workload(&[1, 2, 4, 8, 16, 32, 64, 128]);
    let mut context = Context::new();
    context.callback_ns = 600_000;
    assert!(matches!(
        planner(3).propose_with_execution(
            &s,
            &Model(predicted_ns),
            &context,
            &mut Clock(&context.now),
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    assert!(
        context.now.get() >= 1_600_000,
        "a fresh replay attempt cannot extend the original planner deadline"
    );
}

#[test]
fn candidate_ranking_constructs_known_tail_before_cheaper_uncertified_challenger() {
    let mut s = workload(&[2, 4]);
    s.scope.horizon_end_ns = 10_000_000;
    s.requests[0].timing.budgets.ttft_ns = n64(10_000_000);
    s.requests[0].timing.maximum_output_tokens = n32(1);
    if let RequestPhaseView::Prefill(progress) = &mut s.requests[0].phase {
        progress.total_prompt_tokens = n32(6);
        progress.executable_until = 6;
        progress.reference = Arc::new(PrefillReferenceWork {
            evaluation: Default::default(),
            version: 1,
            points: (0..=6)
                .map(|prompt_tokens| ReferenceWorkPoint {
                    prompt_tokens,
                    cumulative_work_ns: u64::from(prompt_tokens) * 100_000,
                })
                .collect(),
        });
    }
    let mut context = Context::new();
    context.callback_ns = 10_000;
    let queries = RefCell::new(Vec::new());
    let model = Model(|shape: &WaveExecutionShape| {
        let [chunk] = shape.prefill_chunks.as_slice() else {
            panic!("one actual prefill owner")
        };
        let key = (chunk.offset, chunk.count.get());
        queries.borrow_mut().push(key);
        match key {
            (0, 4) => Some(4_000_000),
            (4, 2) => Some(1_000_000), // A complete, known 5ms common plan.
            (0, 2) | (2, 2) => Some(1_000_000),
            (2, 4) => None, // Cheap root prefix has no known complete tail in H=2.
            _ => panic!("unexpected pure frontier {key:?}"),
        }
    });
    let mut p = planner(2);
    p.settings.search.beam_width = nz(1);
    p.settings.search.candidate_limit = nz(2);
    let before = s.clone();
    let (first, witness, search) =
        feasible(p.propose_with_execution(&s, &model, &context, &mut Clock(&context.now)));
    assert!(matches!(first.candidate.work[0].action,
        WaveAction::Prefill { offset: 0, count } if count.get() == 4));
    assert_eq!(witness.waves, 2);
    assert_eq!(witness.predicted_output_tokens, 1);
    let queries = queries.borrow();
    let first_completed_tail = queries.iter().position(|q| *q == (4, 2)).unwrap();
    let cheaper_root = queries.iter().position(|q| *q == (0, 2)).unwrap();
    assert!(first_completed_tail < cheaper_root);
    assert!(
        queries.contains(&(2, 4)),
        "the cheaper sibling's tail really is queried"
    );
    assert!(search.cost_unknown_candidates > 0);
    assert!(search.expanded_candidates <= 2 * 1 * 2);
    assert!(search.enumeration_attempts <= 8 * 2 * 1 * 2);
    assert!(
        context.begins.get() > 1,
        "the complete incumbent is freshly replayed"
    );
    for count in [4, 2] {
        assert!(context
            .seen
            .borrow()
            .iter()
            .any(|&(epoch, n)| epoch > 1 && n == count));
    }
    assert!(context.now.get() < 1_600_000);
    assert!(first.final_replay_first_wave.is_some());
    assert_eq!(s, before);
}
