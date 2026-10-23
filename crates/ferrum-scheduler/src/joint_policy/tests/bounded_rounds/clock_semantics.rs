use super::*;

#[test]
fn bounded_generated_history_without_text_does_not_define_tpot_from_one_opportunity() {
    let mut request = decode(64);
    request.output_tokens = 2;
    request.central_clock = RequestClock::default();
    request.risk_clock = request.central_clock;
    let requests = [request];
    let policy = config();
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let mut accumulator = BoundedAccumulator::new(&requests, 0, &policy).unwrap();
    let summary = SloPolicy::evaluate_scenarios_with_requests(
        input(
            &snapshot,
            &policy,
            &requests,
            &adapter,
            &[],
            EvaluationHorizon::BoundedRounds(1),
        ),
        |_, _| Ok(()),
        |rank, comparison| {
            let result = comparison.static_result.complete().unwrap();
            assert_eq!(result.requests[0].output_tokens, 3);
            assert_eq!(result.requests[0].central_clock.first_output, None);
            let service = result.bounded_timing.as_ref().unwrap().requests[0]
                .central
                .projection;
            assert_eq!(service.first_output, service.last_output);
            assert!(service.first_output.is_some());
            assert_eq!(service.interval_count, 0);
            accumulator.observe(rank, comparison).unwrap();
        },
    )
    .unwrap();
    let evidence = accumulator.finish(summary).unwrap().static_evidence;
    assert_eq!(evidence.service_prefix_tpot_undefined_requests, 1.0);
    assert_eq!(evidence.observed_prefix_tpot_undefined_requests, 1.0);
    assert_eq!(evidence.future_visible_timing_unknown_requests, 1.0);
    assert_eq!(
        evidence.score.tpot_time_debt, 0.0,
        "no observed first text anchors TPOT debt"
    );
}

#[test]
fn bounded_silent_generated_opportunity_cannot_clear_observed_text_gap() {
    let mut request = decode(64);
    request.output_tokens = 2;
    request.age = Duration::from_millis(24);
    request.central_clock.first_output = Some(Duration::ZERO);
    request.central_clock.last_output = Some(Duration::ZERO);
    request.risk_clock = request.central_clock;
    let requests = [request];
    let mut policy = config();
    policy.itl_ms = 25.0;
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let mut accumulator = BoundedAccumulator::new(&requests, 0, &policy).unwrap();
    let summary = SloPolicy::evaluate_scenarios_with_requests(
        input(
            &snapshot,
            &policy,
            &requests,
            &adapter,
            &[],
            EvaluationHorizon::BoundedRounds(2),
        ),
        |_, _| Ok(()),
        |rank, comparison| {
            let result = comparison.static_result.complete().unwrap();
            assert_eq!(result.requests[0].output_tokens, 4);
            assert_eq!(result.requests[0].central_clock, requests[0].central_clock);
            assert_eq!(result.requests[0].risk_clock, requests[0].risk_clock);
            let service = result.bounded_timing.as_ref().unwrap().requests[0]
                .central
                .projection;
            if rank.index == 0 {
                assert_eq!(service.last_output, Some(Duration::from_millis(26)));
                assert_eq!(service.violating_interval_count, 0);
                // A lawful silent token at t25 followed by text at t26 would
                // have one 26ms visible interval, not the service gaps 25+1.
                let actual_visible_gap =
                    Duration::from_millis(26) - requests[0].central_clock.last_output.unwrap();
                assert!(actual_visible_gap > Duration::from_millis(25));
            }
            accumulator.observe(rank, comparison).unwrap();
        },
    )
    .unwrap();
    let decision = accumulator.finish(summary).unwrap();
    assert_eq!(decision.historical_visible_intervals, 0);
    assert_eq!(
        decision
            .static_evidence
            .future_visible_timing_unknown_requests,
        1.0
    );
    assert!(decision.static_evidence.unclosed_visible_itl_exposure > 0.0);
    assert_eq!(decision.selected, None);
    assert!(
        snapshot.call_count() == 0,
        "conditional costs use cost_scenarios, not predict"
    );
}

#[test]
fn bounded_text_history_accepts_zero_fewer_or_more_events_than_generated_tokens() {
    for text in [
        RequestClock::default(),
        RequestClock {
            first_output: Some(Duration::ZERO),
            last_output: Some(Duration::ZERO),
            ..Default::default()
        },
        RequestClock {
            first_output: Some(Duration::ZERO),
            last_output: Some(Duration::from_millis(4)),
            interval_count: 2,
            maximum_output_interval: Duration::from_millis(2),
            ..Default::default()
        },
    ] {
        let mut request = decode(64);
        request.output_tokens = 2;
        request.central_clock = text;
        request.risk_clock = text;
        let requests = [request];
        let policy = config();
        let snapshot = Snapshot::new();
        let adapter = Adapter::new();
        let mut accumulator = BoundedAccumulator::new(&requests, 0, &policy).unwrap();
        let summary = SloPolicy::evaluate_scenarios_with_requests(
            input(
                &snapshot,
                &policy,
                &requests,
                &adapter,
                &[],
                EvaluationHorizon::BoundedRounds(2),
            ),
            |_, _| Ok(()),
            |rank, comparison| {
                let result = comparison.static_result.complete().unwrap();
                assert_eq!(result.requests[0].output_tokens, 4);
                assert_eq!(result.requests[0].central_clock, text);
                assert_eq!(result.requests[0].risk_clock, text);
                if text.first_output.is_none() {
                    assert_eq!(result.requests[0].central_clock.first_output, None);
                    assert!(result.bounded_timing.as_ref().unwrap().requests[0]
                        .central
                        .projection
                        .first_output
                        .is_some());
                }
                accumulator.observe(rank, comparison).unwrap();
            },
        )
        .unwrap();
        let decision = accumulator.finish(summary).unwrap();
        assert_eq!(decision.historical_visible_intervals, text.interval_count);
        assert_eq!(
            decision
                .static_evidence
                .future_visible_timing_unknown_requests,
            1.0
        );
    }

    // One generated token can have multiple text spans. Its token denominator
    // remains zero even though those observed spans provide timing anchors.
    // A second request keeps the real bounded traversal active; the first
    // request's cap prevents it from acquiring invented future opportunities.
    let mut one_token = decode(1);
    one_token.central_clock = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(Duration::from_millis(4)),
        interval_count: 2,
        maximum_output_interval: Duration::from_millis(2),
        ..Default::default()
    };
    one_token.risk_clock = one_token.central_clock;
    let requests = [one_token, decode(64)];
    let policy = config();
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let mut accumulator = BoundedAccumulator::new(&requests, 0, &policy).unwrap();
    let summary = SloPolicy::evaluate_scenarios_with_requests(
        input(
            &snapshot,
            &policy,
            &requests,
            &adapter,
            &[],
            EvaluationHorizon::BoundedRounds(1),
        ),
        |_, _| Ok(()),
        |rank, comparison| {
            let result = comparison.static_result.complete().unwrap();
            assert_eq!(result.requests[0].output_tokens, 1);
            assert_eq!(result.requests[0].central_clock, requests[0].central_clock);
            accumulator.observe(rank, comparison).unwrap();
        },
    )
    .unwrap();
    let evidence = accumulator.finish(summary).unwrap().static_evidence;
    assert_eq!(evidence.service_prefix_tpot_undefined_requests, 1.0);
    assert_eq!(evidence.observed_prefix_tpot_undefined_requests, 2.0);
}

#[test]
fn bounded_service_opportunities_match_old_single_token_projection_only_as_proxy() {
    let requests = [decode(64)];
    let policy = config();
    let legacy_snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let legacy = SloPolicy::evaluate(input(
        &legacy_snapshot,
        &policy,
        &requests,
        &adapter,
        &[],
        EvaluationHorizon::StaticRounds(2),
    ))
    .unwrap();
    let snapshot = Snapshot::new();
    let bounded = SloPolicy::evaluate(input(
        &snapshot,
        &policy,
        &requests,
        &adapter,
        &[],
        EvaluationHorizon::BoundedRounds(2),
    ))
    .unwrap();
    let legacy = legacy.static_result.complete().unwrap();
    let bounded = bounded.static_result.complete().unwrap();
    let clocks = bounded.bounded_timing.as_ref().unwrap().requests[0];
    assert_eq!(clocks.central.projection, legacy.requests[0].central_clock);
    assert_eq!(clocks.risk.projection, legacy.requests[0].risk_clock);
    assert_eq!(bounded.requests[0].central_clock, requests[0].central_clock);
    assert_eq!(bounded.requests[0].risk_clock, requests[0].risk_clock);
    assert_eq!(
        bounded.requests[0].progress(),
        legacy.requests[0].progress()
    );
    assert_eq!(bounded.queries, 2);
    assert_eq!(bounded.queries, legacy.queries);
    assert_eq!(snapshot.call_count(), legacy_snapshot.call_count());
}

#[test]
fn bounded_timing_charges_all_retained_actions_and_restores_history_on_error() {
    let requests = [decode(64)];
    let bytes = BoundedTiming::retained_bytes_bound(1, 2).unwrap();
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let mut policy = config();
    policy.planning.max_request_fact_bytes = bytes - 1;
    assert!(matches!(
        SloPolicy::evaluate(input(
            &snapshot,
            &policy,
            &requests,
            &adapter,
            &[Action::Ordinary],
            EvaluationHorizon::BoundedRounds(2),
        )),
        Err(Unsupported::ParticipantBudget)
    ));
    assert_eq!(snapshot.call_count(), 0);
    policy.planning.max_request_fact_bytes = bytes;
    let result = SloPolicy::evaluate(input(
        &snapshot,
        &policy,
        &requests,
        &adapter,
        &[Action::Ordinary],
        EvaluationHorizon::BoundedRounds(2),
    ))
    .unwrap();
    assert!(result
        .static_result
        .complete()
        .unwrap()
        .bounded_timing
        .is_some());
    assert!(result.alternatives[0]
        .as_ref()
        .unwrap()
        .bounded_timing
        .is_some());
    assert_eq!(snapshot.call_count(), 4);
    assert_eq!(BoundedTiming::retained_bytes_bound(usize::MAX, 2), None);

    let mut request = requests[0].clone();
    request.age = Duration::MAX;
    let observed = request.central_clock;
    let mut timing = BoundedTiming::new(std::slice::from_ref(&request)).unwrap();
    let error = timing.apply_member(
        0,
        &mut request,
        CostWork {
            kind: AttemptWorkKind::Decode,
            requested: ExecutionRange::new(1, 1).unwrap(),
            context_tokens: 2,
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
        },
        Duration::from_nanos(1),
        Duration::from_nanos(1),
        Duration::from_millis(25),
        ClockCoupling::Equal,
    );
    assert_eq!(error, Err(Unsupported::Arithmetic));
    assert_eq!(
        request.output_tokens, 2,
        "the original progress-before-clock-error order remains"
    );
    assert_eq!(request.central_clock, observed);
    assert_eq!(request.risk_clock, observed);
}

#[test]
fn initial_reference_ordinal_without_text_has_no_fabricated_clock_anchor() {
    let mut request = decode(64);
    request.output_tokens = 3;
    request.central_clock = RequestClock::default();
    request.risk_clock = RequestClock::default();
    assert!(request.valid_bounded());
    let initial = request.clone();
    let mut timing = BoundedTiming::new(std::slice::from_ref(&request)).unwrap();
    timing
        .bind_reference(std::slice::from_ref(&request), &[request.progress()])
        .unwrap();
    for elapsed in [1, 2] {
        let context = request.prompt_tokens + request.output_tokens;
        timing
            .apply_member(
                0,
                &mut request,
                CostWork {
                    kind: AttemptWorkKind::Decode,
                    requested: ExecutionRange::new(context - 1, 1).unwrap(),
                    context_tokens: context,
                    prefix_restore: AttemptPrefixRestore::NoneObserved,
                    internally_reused_tokens: None,
                },
                Duration::from_millis(elapsed),
                Duration::from_millis(elapsed),
                Duration::from_millis(100),
                ClockCoupling::Equal,
            )
            .unwrap();
        timing.finish_wave(Duration::from_millis(elapsed));
    }
    let reference = timing.requests[0].reference.unwrap();
    assert!(reference.output_initially_satisfied);
    assert_eq!(reference.progress_reached_at, Some(Duration::ZERO));
    assert_eq!(reference.progress_at_reach, Some(initial.progress()));
    assert_eq!(
        reference.central.unwrap().projection,
        RequestClock::default()
    );
    assert_eq!(reference.risk.unwrap().projection, RequestClock::default());
    assert_eq!(
        reference.following_central_output_at,
        Some(Duration::from_millis(6))
    );
    assert_eq!(
        reference.following_risk_output_at,
        Some(Duration::from_millis(6))
    );
    assert_eq!(timing.reference_reached_at, Some(Duration::from_millis(1)));
    assert_eq!(
        timing.requests[0].central.projection.last_output,
        Some(Duration::from_millis(7))
    );
    assert_eq!(request.output_tokens, 5);
    assert_eq!(request.central_clock, initial.central_clock);
    assert_eq!(request.risk_clock, initial.risk_clock);
}
