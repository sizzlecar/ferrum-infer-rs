use super::*;

fn evidence(round: &TimePlanningRound) -> serde_json::Value {
    serde_json::to_value(round.diagnostics()).unwrap()["bounded_decision"].clone()
}

#[test]
fn bounded_decision_diagnostics_report_real_choice_and_read_without_cost_queries() {
    let fixture = bounded_fixture(CostExecution::SequentialPrefillThenDecode, Blocker::None);
    assert!(evidence(&fixture.round).is_null());
    let (batch, round) = observed_selection(&fixture.scheduler, fixture.round);
    assert_eq!(batch.size(), 1);
    assert_eq!(tokens(&batch, fixture.decoder.as_ref().unwrap()), 1);
    let calls = cost_calls(&fixture.source);
    assert_eq!(round.query_count, calls);
    assert_eq!(round.statistics.bounded.provider_calls, calls as u64);
    let observed = evidence(&round);
    let selected = observed["selected_index"].as_u64().unwrap() as usize;
    assert_eq!(observed["outcome"]["status"], "applied");
    assert_eq!(observed["snapshot_revision"], fixture.source.revision());
    assert_eq!(
        observed["lookahead_rounds"],
        round.config.planning.max_lookahead_rounds
    );
    assert_eq!(
        observed["max_catchup_rounds"],
        round.config.planning.max_catchup_rounds
    );
    assert_eq!(
        observed["candidate_actions"][selected]["kind"],
        "decode_only"
    );
    assert_eq!(
        observed["candidate_actions"].as_array().unwrap().len(),
        observed["alternatives"].as_array().unwrap().len()
    );
    assert_eq!(
        observed["observed_ranks"],
        round.config.planning.max_scenarios
    );
    assert!(
        observed["static_evidence"]["supported_mass"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        observed["selected_score"].as_f64().unwrap() < observed["static_score"].as_f64().unwrap()
    );
    assert_eq!(observed["alternatives"][selected]["unsupported_mass"], 0.0);
    assert_eq!(round.statistics.bounded.applied, 1);
    assert_eq!(
        cost_calls(&fixture.source),
        calls,
        "diagnostics only borrow already computed evidence"
    );
    assert!(
        evidence(&recapture_round(&round)).is_null(),
        "a fresh round has no inherited decision"
    );
}

#[test]
fn bounded_decision_diagnostics_preserve_selection_when_real_apply_is_invalidated() {
    let fixture = bounded_fixture(CostExecution::SingleInvocation, Blocker::None);
    // As in the phase-commit atomicity fixture, freeze the real static wave,
    // compare it, then race a real progress commit before consumer application.
    let mut batch = select(&fixture.scheduler, None).batch.unwrap();
    let mut round = fixture.round;
    {
        let prefill = fixture.scheduler.prefill_queue.read();
        let decode = fixture.scheduler.decode_queue.read();
        for scheduled in &batch.requests {
            let request = prefill
                .iter()
                .chain(decode.requests.values())
                .find(|request| request.inner.request.id == scheduled.request.id)
                .unwrap();
            round.selected_work(fixture.scheduler.cost_work(
                request,
                scheduled.tokens_to_process.unwrap(),
                request.phase == RequestPhase::Decoding,
                round.prefill_shape_policy,
            ));
        }
    }
    let selected = fixture
        .scheduler
        .compare_bounded_first_wave(&hint(), &batch.requests, &mut round, Instant::now())
        .unwrap()
        .expect("the complete supported comparison selects decode-only");
    let before = evidence(&round);
    let selected_index = before["selected_index"].as_u64().unwrap() as usize;
    assert_eq!(before["outcome"]["status"], "computed");
    assert_eq!(
        before["candidate_actions"][selected_index]["kind"],
        "decode_only"
    );
    let calls = cost_calls(&fixture.source);
    assert!(!fixture.scheduler.mark_prefill_chunk_processed(
        fixture.prefill.as_ref().unwrap(),
        4,
        1
    ));
    fixture
        .scheduler
        .apply_bounded_first_wave(&mut batch.requests, &mut round, selected);
    let after = evidence(&round);
    assert_eq!(after["outcome"]["status"], "invalidated");
    assert_eq!(after["selected_index"], before["selected_index"]);
    assert_eq!(after["alternatives"], before["alternatives"]);
    assert_eq!(batch.size(), 2, "a stale edit cannot remove the prefill");
    assert_eq!(round.statistics.bounded.selected, 1);
    assert_eq!(round.statistics.bounded.invalidated, 1);
    assert_eq!(round.statistics.bounded.applied, 0);
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_eq!(cost_calls(&fixture.source), calls);
}

#[test]
fn bounded_decision_diagnostics_clear_on_same_round_unknown_and_empty_selection() {
    for empty in [false, true] {
        let fixture = bounded_fixture(CostExecution::SingleInvocation, Blocker::None);
        let (_, mut round) = observed_selection(&fixture.scheduler, fixture.round);
        assert_eq!(evidence(&round)["outcome"]["status"], "applied");
        let calls = cost_calls(&fixture.source);
        if empty {
            assert!(fixture
                .scheduler
                .cancel_request(fixture.prefill.unwrap())
                .unwrap());
            assert!(fixture
                .scheduler
                .cancel_request(fixture.decoder.unwrap())
                .unwrap());
        } else {
            // A newly missing owner proof makes the next comparison unknown;
            // it must not attach the prior successful choice to this batch.
            round.set_capacity(None);
        }
        let result = fixture
            .scheduler
            .create_iteration_batch_with_time(
                hint(),
                WaitingAdmissionMode::Prepared {
                    initial_priority: None,
                    wake: None,
                    observer: &mut |_| {},
                },
                Instant::now(),
                Some(&mut round),
            )
            .unwrap();
        assert_eq!(result.is_none(), empty);
        assert!(evidence(&round).is_null());
        assert_eq!(
            round.statistics.bounded.applied, 1,
            "aggregate counters remain cumulative"
        );
        if !empty {
            assert_eq!(round.statistics.bounded.unknown, 1);
        }
        assert_eq!(cost_calls(&fixture.source), calls);
    }
}
