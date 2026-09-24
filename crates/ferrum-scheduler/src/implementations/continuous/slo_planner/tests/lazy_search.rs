//! Controlled clocks and physical-route callbacks exercise search scheduling;
//! these are not measurements of backend cost coverage or hardware latency.
use super::*;
use ferrum_interfaces::execution_cost::{ActualWaveKind, CanonicalWaveCostShape};
use std::cell::{Cell, RefCell};

struct WindowClock<'a>(&'a Cell<u64>);
impl PlanningClock for WindowClock<'_> {
    fn now_ns(&mut self) -> u64 {
        self.0.get()
    }
    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        Some(PlanningBudgetWindow {
            started_at_ns: 0,
            deadline_ns: 1_000,
        })
    }
}

struct ExpensiveSibling<'a> {
    now: &'a Cell<u64>,
    complete_width: usize,
    widths: RefCell<Vec<usize>>,
}
impl PlanningShapeResolver for ExpensiveSibling<'_> {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        self.widths.borrow_mut().push(query.rows.len());
        if query.rows.len() != self.complete_width {
            self.now.set(1_200);
        }
        TestResolver.resolve(query, poll)
    }
}

#[test]
fn complete_three_wave_witness_precedes_expensive_optional_siblings_and_is_replayed() {
    let mut s = snapshot((1..=8).map(decode).collect());
    s.capabilities.decode_batch_sizes = (1..=8).map(nz).collect();
    for row in &mut s.requests {
        row.timing.maximum_output_tokens = n32(4);
        row.timing.budgets.itl_ns = n64(100_000);
        row.timing.budgets.tpot_ns = n64(100_000);
    }
    // All owners need three real simulated commits to discharge this horizon.
    s.scope.horizon_end_ns = 200_000;
    let before = s.clone();
    let now = Cell::new(100);
    let resolver = ExpensiveSibling {
        now: &now,
        complete_width: 8,
        widths: RefCell::new(Vec::new()),
    };
    struct CompletingModel<'a> {
        now: &'a Cell<u64>,
        queries: RefCell<Vec<(Vec<u32>, u64)>>,
    }
    impl PlanningCostModel for CompletingModel<'_> {
        fn model_version(&self) -> u64 {
            7
        }
        fn predict(
            &self,
            _: &ExecutionFingerprint,
            shape: &WaveExecutionShape,
            at_ns: u64,
        ) -> Option<PlanningCost> {
            self.queries
                .borrow_mut()
                .push((shape.decode_kv_tokens.clone(), at_ns));
            // The third full wave uses each original context plus two. Only
            // this complete common prefix crosses the optional-search cutoff.
            if shape.decode_kv_tokens == (1..=8).map(|id| id * 10 + 2).collect::<Vec<_>>() {
                self.now.set(650);
            }
            Some(PlanningCost {
                typical_ns: 5,
                planning_ns: 5,
                model_version: 7,
                valid_for_ns: u64::MAX,
            })
        }
    }
    let model = CompletingModel {
        now: &now,
        queries: RefCell::new(Vec::new()),
    };
    let (first, witness, search) =
        feasible(planner(3).propose(&s, &model, &resolver, &mut WindowClock(&now)));
    assert_eq!(first.candidate.work.len(), 8);
    assert_eq!(witness.waves, 3);
    assert_eq!(witness.predicted_output_tokens, 24);
    assert_eq!(search.generated_candidates, 3);
    assert_eq!(search.expanded_candidates, 3);
    assert_eq!(search.search_soft_stops, 1);
    // Final replay is independently anchored after the actual search delay.
    // Each complete physical frontier must be predicted at its new start time.
    for step in 0..3 {
        let expected = (1..=8).map(|id| id * 10 + step).collect::<Vec<_>>();
        assert!(model
            .queries
            .borrow()
            .contains(&(expected, 650 + u64::from(step) * 5)));
    }
    assert!(resolver.widths.borrow().iter().all(|width| *width == 8));
    assert_eq!(
        s, before,
        "search cannot change real ingress, work or output limits"
    );
}

struct MissingFullBatch;
impl PlanningShapeResolver for MissingFullBatch {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        if query.rows.len() == 2 {
            return Ok(None);
        }
        TestResolver.resolve(query, poll)
    }
}

#[test]
fn unknown_full_batch_does_not_spend_successful_candidate_limit_or_hide_singletons() {
    let s = snapshot(vec![decode(1), decode(2)]);
    let mut p = planner(2);
    p.settings.search.candidate_limit = nz(1);
    p.settings.search.beam_width = nz(1);
    let (first, witness, search) = feasible(p.propose(
        &s,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &MissingFullBatch,
        &mut Clock(100),
    ));
    assert_eq!(first.candidate.work.len(), 1);
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(witness.waves, 2);
    assert!(search.shape_unknown_candidates > 0);
    assert!(search.expanded_candidates <= 2);
    assert!(search.enumeration_attempts <= 16);
}

struct FirstPrefillUnavailable;
impl PlanningShapeResolver for FirstPrefillUnavailable {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        if query.prior_waves.is_empty() && query.kind == ActualWaveKind::Prefill {
            return Ok(None);
        }
        TestResolver.resolve(query, poll)
    }
}

#[test]
fn dead_end_decode_prefix_does_not_lock_out_later_mixed_three_wave_witness() {
    let mut target = prefill(2);
    target.timing.maximum_output_tokens = n32(2);
    target.timing.budgets.ttft_ns = n64(120);
    target.timing.budgets.itl_ns = n64(10);
    target.timing.budgets.tpot_ns = n64(10);
    let mut s = snapshot(vec![decode(1), target]);
    s.capabilities.decode_batch_sizes = vec![nz(1)];
    s.capabilities.prefill_chunk_sizes = vec![n32(4)];
    let before = s.clone();
    let mut p = planner(3);
    p.settings.search.candidate_limit = nz(3);
    p.settings.search.beam_width = nz(1);
    let (first, witness, search) = feasible(p.propose(
        &s,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &FirstPrefillUnavailable,
        &mut Clock(100),
    ));
    // Decode -> partial -> final reaches H without the new decoder's service.
    // Mixed(partial) -> final -> decode must still be explored at B=1.
    assert!(first
        .candidate
        .work
        .iter()
        .any(|row| row.action == WaveAction::Decode));
    assert!(first
        .candidate
        .work
        .iter()
        .any(|row| matches!(row.action, WaveAction::Prefill { count, .. } if count.get() == 4)));
    assert_eq!(witness.waves, 3);
    assert_eq!(witness.predicted_output_tokens, 3);
    assert!(search.expanded_candidates >= 6);
    assert!(search.expanded_candidates <= 9);
    assert_eq!(s, before);
}

#[test]
fn later_budget_failure_keeps_already_resolved_and_evaluated_candidate_statistics() {
    let mut s = snapshot(vec![decode(1), decode(2)]);
    for row in &mut s.requests {
        row.timing.budgets.itl_ns = n64(100_000);
        row.timing.budgets.tpot_ns = n64(100_000);
    }
    let now = Cell::new(100);
    let resolver = ExpensiveSibling {
        now: &now,
        complete_width: 2,
        widths: RefCell::new(Vec::new()),
    };
    let result = planner(1).propose(
        &s,
        &Model(|_: &WaveExecutionShape| None),
        &resolver,
        &mut WindowClock(&now),
    );
    let PlanningDecision::Unknown { reason, search } = result else {
        panic!("{result:?}")
    };
    assert_eq!(reason, PlanningUnknownReason::ComputeBudgetExhausted);
    assert_eq!(search.generated_candidates, 1);
    assert_eq!(search.expanded_candidates, 1);
    assert_eq!(search.cost_unknown_candidates, 1);
    assert!(search.enumeration_attempts > 0);
    assert_eq!(
        search.search_soft_stops, 0,
        "a Known route alone is not a witness"
    );
}
