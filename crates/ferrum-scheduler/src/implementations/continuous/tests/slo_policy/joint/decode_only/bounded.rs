//! Bounded control consumes supported future work, not complete-request proof.
use super::*;

mod decision_diagnostics;

fn bounded_fixture(execution: CostExecution, blocker: Blocker) -> DecodeOnlyFixture {
    let mut fixture = decode_only_fixture(true, true, execution, blocker);
    fixture.round.config.strategy = ferrum_types::SloStrategy::Bounded;
    // Include the common continuation after the decoder event so postponing
    // prefill must also demonstrate progress within the compared horizon.
    // Applying the chosen first wave still leaves the real prompt unfinished.
    fixture.round.config.planning.max_lookahead_rounds = 8;
    fixture
}

fn cost_calls(source: &MixedCost) -> usize {
    source.scalar_calls.load(Ordering::Relaxed) + source.calls.lock().len()
}

#[test]
fn bounded_actual_selection_accepts_text_history_independently_of_generated_count() {
    for events in [0, 1, 3] {
        let (scheduler, mut round, _, decoder, now) =
            fixture(CostExecution::SequentialPrefillThenDecode, false, false);
        round.config.strategy = ferrum_types::SloStrategy::Bounded;
        round.config.planning.max_elapsed_us = 1_000_000;
        let mut clocks = round.request_clocks().unwrap().to_vec();
        let clock = &mut clocks.iter_mut().find(|(id, _)| *id == decoder).unwrap().1;
        assert_eq!(clock.generated_tokens, 2);
        clock.nonempty_text_events = events;
        clock.first_text = (events > 0).then_some(Duration::from_millis(5));
        clock.last_text = match events {
            0 => None,
            1 => clock.first_text,
            _ => Some(Duration::from_millis(10)),
        };
        clock.max_text_interval = (events > 1).then_some(Duration::from_millis(3));
        let original = *clock;
        assert!(round.set_request_clocks(clocks));
        let batch = next(&scheduler, &mut round, now);
        assert!(!batch.is_empty());
        assert_eq!(round.statistics.bounded.comparisons, 1);
        assert_eq!(round.statistics.bounded.evaluated, 1);
        assert_eq!(round.statistics.bounded.unknown, 0);
        let reasons = serde_json::to_value(round.statistics.joint_unsupported).unwrap();
        assert_eq!(reasons["historical_text_coalescing"], 0);
        assert_eq!(reasons["invalid_clock"], 0);
        assert_eq!(
            round
                .request_clocks()
                .unwrap()
                .iter()
                .find(|(id, _)| *id == decoder)
                .unwrap()
                .1,
            original
        );
        assert!(round.query_count > 0);
    }
}

#[test]
fn bounded_nonstreaming_clock_remains_unknown_without_inventing_visible_history() {
    let mut fixture = bounded_fixture(CostExecution::SingleInvocation, Blocker::None);
    let mut clocks = fixture.round.request_clocks().unwrap().to_vec();
    let clock = &mut clocks
        .iter_mut()
        .find(|(id, _)| Some(id) == fixture.decoder.as_ref())
        .unwrap()
        .1;
    clock.streaming = false;
    clock.nonempty_text_events = 0;
    clock.first_text = None;
    clock.last_text = None;
    assert!(fixture.round.set_request_clocks(clocks));
    let (batch, round) = observed_selection(&fixture.scheduler, fixture.round);
    assert_eq!(batch.size(), 2);
    assert_eq!(round.statistics.bounded.evaluated, 0);
    assert_eq!(round.statistics.bounded.applied, 0);
    assert_eq!(round.query_count, 0);
    assert_eq!(
        serde_json::to_value(round.statistics.joint_unsupported).unwrap()["non_streaming_clock"],
        1
    );
}

#[test]
fn bounded_supported_action_does_not_require_stopping_or_completion_forecasts() {
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        for blocker in [Blocker::StoppingUnknown, Blocker::CompletionUnknown] {
            let fixture = bounded_fixture(execution, blocker);
            let prefill = fixture.prefill.as_ref().unwrap();
            let decoder = fixture.decoder.as_ref().unwrap();
            let before = fixture.scheduler.prefill_queue.read()[0]
                .logical_work_frontier
                .clone();
            let (batch, round) = observed_selection(&fixture.scheduler, fixture.round);

            assert_eq!(batch.size(), 1, "{execution:?} {blocker:?}");
            assert_eq!(tokens(&batch, decoder), 1);
            assert!(batch
                .requests
                .iter()
                .all(|item| &item.request.id != prefill));
            assert_eq!(round.selected.len(), 1);
            assert_eq!(round.selected[0].kind, AttemptWorkKind::Decode);
            assert!(round.selection_is_exact(batch.size()));
            assert_eq!(batch.resource_requirements.gpu_memory, 16);
            assert_eq!(fixture.scheduler.active_count(), 2);
            assert_eq!(fixture.scheduler.waiting_count(), 0);
            let queue = fixture.scheduler.prefill_queue.read();
            assert_eq!(queue[0].prefill_chunk_offset, 0);
            assert_eq!(queue[0].logical_work_frontier, before);
            drop(queue);

            let counts = &round.statistics.bounded;
            assert_eq!(counts.comparisons, 1);
            assert_eq!(counts.selected, 1);
            assert_eq!(counts.applied, 1);
            assert_eq!(counts.unknown, 0);
            assert_eq!(counts.retained, 0);
            assert_eq!(round.statistics.joint_interventions, 1);
            assert_eq!(round.statistics.joint_decode_only_interventions, 1);
            let calls = cost_calls(&fixture.source);
            assert!(calls > 0);
            assert_eq!(
                round.query_count, calls,
                "no terminal-forecast prerequisite"
            );
            let health = fixture.scheduler.slo_snapshot().unwrap();
            assert_eq!(health.selections.bounded.applied, 1);
            assert_eq!(health.selections.joint_interventions, 1);
            serde_json::to_value(health).unwrap();
            assert_eq!(cost_calls(&fixture.source), calls, "health is read-only");
        }
    }
}

#[test]
fn bounded_default_fallback_preserves_admitted_work_on_unknown_cost_or_capacity() {
    for blocker in [Blocker::NoCapacity, Blocker::CostUnknown] {
        let fixture = bounded_fixture(CostExecution::SingleInvocation, blocker);
        assert_eq!(
            fixture.round.config.unknown_cost,
            UnknownCostPolicy::StaticFallback
        );
        assert_eq!(
            fixture.round.config.unknown_stopping,
            UnknownCostPolicy::StaticFallback
        );
        fixture.round.config.validate().unwrap();
        let (batch, round) = observed_selection(&fixture.scheduler, fixture.round);
        assert_eq!(batch.size(), 2, "{blocker:?}");
        assert_eq!(tokens(&batch, fixture.prefill.as_ref().unwrap()), 4);
        assert_eq!(tokens(&batch, fixture.decoder.as_ref().unwrap()), 1);
        assert!(round.selection_is_exact(batch.size()));
        assert_eq!(fixture.scheduler.active_count(), 2);
        assert_eq!(fixture.scheduler.waiting_count(), 0);
        assert_eq!(round.statistics.bounded.comparisons, 1);
        assert_eq!(round.statistics.bounded.selected, 0);
        assert_eq!(round.statistics.bounded.applied, 0);
        assert_eq!(round.statistics.bounded.unknown, 1);
        assert_eq!(round.statistics.joint_interventions, 0);
        assert_eq!(round.statistics.joint_decode_only_interventions, 0);
        assert_eq!(
            fixture.scheduler.prefill_queue.read()[0].prefill_chunk_offset,
            0
        );
        assert_eq!(round.query_count, cost_calls(&fixture.source));
        if blocker == Blocker::NoCapacity {
            assert_eq!(cost_calls(&fixture.source), 0);
        } else {
            assert!(cost_calls(&fixture.source) > 0);
        }
    }
}

#[test]
fn bounded_fact_budget_failure_keeps_static_work_without_querying_cost() {
    let mut fixture = bounded_fixture(CostExecution::SingleInvocation, Blocker::None);
    // Facts were captured normally. A later, smaller planning allowance cannot
    // authorize more retained evidence or invalidate already admitted work.
    // This exercises the product fallback, not a particular allocation layout.
    fixture.round.config.planning.max_request_fact_bytes = 1;
    assert!(fixture.round.config.validate().is_ok());
    let before = fixture.scheduler.prefill_queue.read()[0]
        .logical_work_frontier
        .clone();
    let (batch, round) = observed_selection(&fixture.scheduler, fixture.round);
    assert_eq!(batch.size(), 2);
    assert_eq!(tokens(&batch, fixture.prefill.as_ref().unwrap()), 4);
    assert_eq!(tokens(&batch, fixture.decoder.as_ref().unwrap()), 1);
    assert!(round.selection_is_exact(batch.size()));
    assert_eq!(fixture.scheduler.active_count(), 2);
    assert_eq!(fixture.scheduler.waiting_count(), 0);
    assert_eq!(round.statistics.bounded.comparisons, 1);
    assert_eq!(round.statistics.bounded.evaluated, 0);
    assert_eq!(round.statistics.bounded.selected, 0);
    assert_eq!(round.statistics.bounded.applied, 0);
    assert_eq!(round.statistics.bounded.unknown, 1);
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_eq!(
        serde_json::to_value(round.statistics.joint_unsupported).unwrap()["participant_budget"],
        1
    );
    assert_eq!(round.query_count, 0);
    assert_eq!(cost_calls(&fixture.source), 0);
    let queue = fixture.scheduler.prefill_queue.read();
    assert_eq!(queue[0].prefill_chunk_offset, 0);
    // Static fallback schedules the ordinary prefill wave, but does not
    // manufacture a compute commit or advance its progress generation.
    let mut expected = before;
    expected.mark_scheduled(tokens(&batch, fixture.prefill.as_ref().unwrap()));
    assert_eq!(queue[0].logical_work_frontier, expected);
}

#[test]
fn bounded_default_fallback_allows_a_resource_probe_without_granting_admission() {
    let mut fixture = decode_only_fixture(
        false,
        false,
        CostExecution::SingleInvocation,
        Blocker::CostUnknown,
    );
    fixture.round.config.strategy = ferrum_types::SloStrategy::Bounded;
    assert_eq!(
        fixture.round.config.unknown_cost,
        UnknownCostPolicy::StaticFallback
    );
    assert_eq!(
        fixture.round.config.unknown_stopping,
        UnknownCostPolicy::StaticFallback
    );
    fixture.round.config.validate().unwrap();
    fixture.round.set_capacity(None);
    let id = waiting(&fixture.scheduler, 4);
    {
        let queue = fixture.scheduler.waiting_queue.read();
        let request = queue.iter().next().unwrap();
        assert!(matches!(
            fixture.scheduler.admission_time_decision(
                request,
                &mut fixture.round,
                Instant::now()
            ),
            time_policy::TimeDecision::Allow(tokens) if tokens > 0
        ));
    }
    let mut scan = fixture.scheduler.begin_token_admission(1, None).unwrap();
    let candidate = scan
        .next(&mut |_| {}, Some(&mut fixture.round))
        .unwrap()
        .expect("bounded timing leaves the resource probe to its original owner");
    assert_eq!(candidate.request().id, id);
    assert_eq!(fixture.scheduler.waiting_count(), 1);
    assert_eq!(fixture.scheduler.active_count(), 0);
    // No executor outcome is manufactured. Withdrawal preserves the waiter.
    drop(candidate);
    let receipt = scan.finish().unwrap();
    assert_eq!(receipt.admitted(), 0);
    assert_eq!(receipt.waiting_after(), 1);
    assert_eq!(fixture.scheduler.active_count(), 0);
    assert_eq!(fixture.round.query_count, 0);
    assert_eq!(cost_calls(&fixture.source), 0);
    assert_eq!(fixture.round.statistics.bounded.applied, 0);
    assert_eq!(fixture.round.statistics.joint_interventions, 0);
    assert!(fixture
        .scheduler
        .admission_failed_requests
        .read()
        .is_empty());
}

#[test]
fn bounded_decode_only_cannot_redefer_without_actual_prefill_progress() {
    let fixture = bounded_fixture(CostExecution::SingleInvocation, Blocker::StoppingUnknown);
    let prefill = fixture.prefill.unwrap();
    let decoder = fixture.decoder.unwrap();
    let original_generation = fixture.scheduler.prefill_queue.read()[0]
        .logical_work_frontier
        .progress_generation();
    let (first, first_round) = observed_selection(&fixture.scheduler, fixture.round);
    assert_eq!(first.size(), 1);
    assert_eq!(tokens(&first, &decoder), 1);
    assert_eq!(first_round.statistics.bounded.applied, 1);
    assert_eq!(first_round.statistics.joint_decode_only_interventions, 1);

    let mut previous = first_round;
    for _ in 0..2 {
        let (batch, round) = observed_selection(&fixture.scheduler, recapture_round(&previous));
        assert!(
            tokens(&batch, &prefill) > 0,
            "unchanged generation owes prefill progress"
        );
        assert_eq!(tokens(&batch, &decoder), 1);
        assert_eq!(round.statistics.joint_decode_only_interventions, 0);
        assert_eq!(
            fixture.scheduler.prefill_queue.read()[0].prefill_chunk_offset,
            0
        );
        assert_eq!(
            fixture.scheduler.prefill_queue.read()[0]
                .logical_work_frontier
                .progress_generation(),
            original_generation
        );
        previous = round;
    }

    assert!(!fixture
        .scheduler
        .mark_prefill_chunk_processed(&prefill, 4, 1));
    let committed = fixture.scheduler.prefill_queue.read()[0]
        .logical_work_frontier
        .clone();
    assert_ne!(committed.progress_generation(), original_generation);
    let (after_progress, round) =
        observed_selection(&fixture.scheduler, recapture_round(&previous));
    assert_eq!(after_progress.size(), 1);
    assert_eq!(tokens(&after_progress, &decoder), 1);
    assert_eq!(round.statistics.bounded.applied, 1);
    assert_eq!(round.statistics.joint_decode_only_interventions, 1);
    let queue = fixture.scheduler.prefill_queue.read();
    assert_eq!(queue[0].prefill_chunk_offset, 1);
    assert_eq!(queue[0].logical_work_frontier, committed);
    assert!(queue[0].logical_work_frontier.can_restore_at(1));
}
