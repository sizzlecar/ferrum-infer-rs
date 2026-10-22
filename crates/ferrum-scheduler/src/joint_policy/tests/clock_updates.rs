use super::*;

/// Original independent update, retained only as a failure/mutation oracle.
fn independent(
    request: &mut SimulatedRequest,
    work: CostWork,
    central: Duration,
    risk: Duration,
    itl_limit: Duration,
) -> Result<(), Unsupported> {
    let emitted = match work.kind {
        AttemptWorkKind::Prefill => {
            request.prefill_tokens = work.requested.end();
            request.prefill_tokens == request.prompt_tokens
        }
        AttemptWorkKind::Decode => true,
    };
    if emitted {
        request.output_tokens = request
            .output_tokens
            .checked_add(1)
            .ok_or(Unsupported::Arithmetic)?;
        for (clock, elapsed) in [
            (&mut request.central_clock, central),
            (&mut request.risk_clock, risk),
        ] {
            let at = request
                .age
                .checked_add(elapsed)
                .ok_or(Unsupported::Arithmetic)?;
            if let Some(previous) = clock.last_output {
                let interval = at.checked_sub(previous).ok_or(Unsupported::InvalidState)?;
                clock.maximum_output_interval = clock.maximum_output_interval.max(interval);
                clock.interval_count = clock
                    .interval_count
                    .checked_add(1)
                    .ok_or(Unsupported::Arithmetic)?;
                if interval > itl_limit {
                    clock.violating_interval_count = clock
                        .violating_interval_count
                        .checked_add(1)
                        .ok_or(Unsupported::Arithmetic)?;
                }
            }
            clock.first_output.get_or_insert(at);
            clock.last_output = Some(at);
            if request.output_tokens == request.scenario_end_tokens {
                clock.completed = Some(at);
            }
        }
    }
    Ok(())
}

fn same_mutation(actual: &SimulatedRequest, reference: &SimulatedRequest) {
    assert_eq!(actual.progress(), reference.progress());
    assert_eq!(actual.central_clock, reference.central_clock);
    assert_eq!(actual.risk_clock, reference.risk_clock);
}

#[test]
fn equal_clocks_keep_partial_prefill_first_output_itl_and_completion() {
    let mut actual = request();
    actual.prompt_tokens = 4;
    actual.prefill_tokens = 0;
    actual.output_tokens = 0;
    actual.output_cap = 3;
    actual.scenario_end_tokens = 3;
    actual.age = Duration::from_millis(2);
    actual.central_clock = RequestClock::default();
    actual.risk_clock = RequestClock::default();
    let mut reference = actual.clone();
    let mut mode = ClockCoupling::for_requests(std::slice::from_ref(&actual));
    assert_eq!(mode, ClockCoupling::Equal);
    for (kind, start, count, elapsed_ms) in [
        (AttemptWorkKind::Prefill, 0, 2, 2),
        (AttemptWorkKind::Prefill, 2, 2, 5),
        (AttemptWorkKind::Decode, 4, 1, 8),
        (AttemptWorkKind::Decode, 5, 1, 12),
    ] {
        let work = CostWork {
            kind,
            requested: ExecutionRange::new(start, count).unwrap(),
            context_tokens: if kind == AttemptWorkKind::Prefill {
                4
            } else {
                start + 1
            },
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
        };
        let elapsed = Duration::from_millis(elapsed_ms);
        validate_member(
            std::slice::from_ref(&actual),
            GroupMember { request: 0, work },
        )
        .unwrap();
        mode.observe_elapsed(elapsed, elapsed);
        let got = apply_member(
            &mut actual,
            work,
            elapsed,
            elapsed,
            Duration::from_millis(3),
            mode,
        );
        let expected = independent(
            &mut reference,
            work,
            elapsed,
            elapsed,
            Duration::from_millis(3),
        );
        assert_eq!(got, expected);
        assert_eq!(got, Ok(()));
        same_mutation(&actual, &reference);
        assert_eq!(mode, ClockCoupling::Equal);
    }
    assert_eq!(
        actual.central_clock.first_output,
        Some(Duration::from_millis(7))
    );
    assert_eq!(
        actual.central_clock.completed,
        Some(Duration::from_millis(14))
    );
    assert_eq!(actual.central_clock.interval_count, 2);
    // The exactly-3ms interval is allowed; only the subsequent 4ms is bad.
    assert_eq!(actual.central_clock.violating_interval_count, 1);
}

struct DivergingCosts {
    at: Instant,
}
impl CostSnapshot for DivergingCosts {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        let context = query.work[0].context_tokens;
        CostPrediction::Known {
            central_service_time: Duration::from_millis(5),
            upper_service_time: Duration::from_millis(if context == 2 { 7 } else { 3 }),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ExactObservedWork,
            samples: 2,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(10),
        }
    }
}

#[test]
fn clocks_that_diverged_do_not_rejoin_when_elapsed_totals_match() {
    let mut decoder = request();
    decoder.output_cap = 3;
    decoder.scenario_end_tokens = 3;
    let costs = DivergingCosts { at: Instant::now() };
    let mut policy = config();
    policy.itl_ms = 6.0;
    let comparison = SloPolicy::evaluate(EvaluationInput {
        snapshot: &costs,
        now: costs.at,
        config: &policy,
        execution: CostExecution::SingleInvocation,
        requests: &[decoder],
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &Adapter,
        static_action: &Action::Full,
        alternatives: &[],
        horizon: EvaluationHorizon::ScenarioCompletion,
    })
    .unwrap();
    let result = comparison.static_result.complete().unwrap();
    assert_eq!(result.central_elapsed, Duration::from_millis(10));
    assert_eq!(result.risk_elapsed, Duration::from_millis(10));
    let request = &result.requests[0];
    assert_eq!(
        request.central_clock.maximum_output_interval,
        Duration::from_millis(5)
    );
    assert_eq!(
        request.risk_clock.maximum_output_interval,
        Duration::from_millis(7)
    );
    assert_eq!(request.central_clock.violating_interval_count, 0);
    assert_eq!(request.risk_clock.violating_interval_count, 1);
    assert_eq!(
        request.central_clock.completed,
        request.risk_clock.completed
    );
}

#[test]
fn different_initial_histories_never_acquire_equal_clock_authority() {
    let mut actual = request();
    actual.age = Duration::from_millis(2);
    actual.risk_clock.last_output = Some(Duration::from_millis(1));
    let mut reference = actual.clone();
    let mut mode = ClockCoupling::for_requests(std::slice::from_ref(&actual));
    assert_eq!(mode, ClockCoupling::Independent);
    mode.observe_elapsed(Duration::from_millis(5), Duration::from_millis(5));
    assert_eq!(mode, ClockCoupling::Independent);
    let work = decode(&actual);
    assert_eq!(
        apply_member(
            &mut actual,
            work,
            Duration::from_millis(5),
            Duration::from_millis(5),
            Duration::from_millis(6),
            mode
        ),
        independent(
            &mut reference,
            work,
            Duration::from_millis(5),
            Duration::from_millis(5),
            Duration::from_millis(6)
        ),
    );
    same_mutation(&actual, &reference);
    assert_eq!(actual.central_clock.violating_interval_count, 1);
    assert_eq!(actual.risk_clock.violating_interval_count, 0);
}

#[test]
fn clock_failures_preserve_original_error_and_partial_mutation_order() {
    // Direct helper boundary cases intentionally reach checked-arithmetic guards
    // that normal validated ingress may already rule out; no feasible trajectory
    // or product event history is claimed for these malformed histories.
    for case in 0..6 {
        let mut actual = request();
        let (central, risk, expected) = match case {
            0 => {
                actual.output_tokens = usize::MAX;
                (Duration::ZERO, Duration::ZERO, Unsupported::Arithmetic)
            }
            1 => {
                actual.age = Duration::MAX;
                (
                    Duration::from_nanos(1),
                    Duration::from_nanos(1),
                    Unsupported::Arithmetic,
                )
            }
            2 => {
                actual.central_clock.last_output = Some(Duration::from_millis(10));
                actual.risk_clock = actual.central_clock;
                (
                    Duration::from_millis(5),
                    Duration::from_millis(5),
                    Unsupported::InvalidState,
                )
            }
            3 => {
                actual.central_clock.interval_count = usize::MAX;
                actual.risk_clock = actual.central_clock;
                (
                    Duration::from_millis(5),
                    Duration::from_millis(5),
                    Unsupported::Arithmetic,
                )
            }
            4 => {
                actual.central_clock.violating_interval_count = usize::MAX;
                actual.risk_clock = actual.central_clock;
                (
                    Duration::from_millis(5),
                    Duration::from_millis(5),
                    Unsupported::Arithmetic,
                )
            }
            _ => {
                actual.age = Duration::from_nanos(1);
                (
                    Duration::from_nanos(1),
                    Duration::MAX,
                    Unsupported::Arithmetic,
                )
            }
        };
        let mut reference = actual.clone();
        let mut mode = ClockCoupling::for_requests(std::slice::from_ref(&actual));
        mode.observe_elapsed(central, risk);
        // The arithmetic case can exceed a valid work projection. apply_member
        // has always received projection validation separately; this oracle
        // concerns its own update/error order only.
        let work = CostWork {
            kind: AttemptWorkKind::Decode,
            requested: ExecutionRange::new(1, 1).unwrap(),
            context_tokens: 2,
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
        };
        let got = apply_member(&mut actual, work, central, risk, Duration::ZERO, mode);
        let old = independent(&mut reference, work, central, risk, Duration::ZERO);
        assert_eq!(got, Err(expected), "case {case}");
        assert_eq!(got, old, "case {case}");
        same_mutation(&actual, &reference);
        if case == 5 {
            assert_eq!(mode, ClockCoupling::Independent);
            assert_eq!(
                actual.central_clock.last_output,
                Some(Duration::from_nanos(2))
            );
            assert_eq!(actual.risk_clock.last_output, Some(Duration::ZERO));
        }
    }
}
