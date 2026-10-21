use super::*;
use std::sync::Arc;

fn clock() -> Instant {
    static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn rules(name: &str) -> Arc<EffectiveStoppingRules> {
    Arc::new(
        EffectiveStoppingRules::new(name.as_bytes().to_vec(), StoppingMechanism::Observed, 64)
            .unwrap(),
    )
}
fn window(capacity: usize) -> StoppingWindow {
    StoppingWindow::new(StoppingConfig {
        window_requests: capacity,
        min_risk_set: 1,
        max_rule_bytes: 64,
        ..StoppingConfig::default()
    })
    .unwrap()
}
fn fact(
    admission: u64,
    generated: usize,
    terminal: Option<RequestTerminalReason>,
) -> RequestStoppingObservation {
    RequestStoppingObservation {
        admitted_at: clock(),
        lifecycle: RequestLifecycleId {
            producer: 1,
            admission,
        },
        revision: generated as u64 + 1,
        observed_at: clock(),
        rules: rules("ordinary-eos"),
        cap: 100,
        generated_tokens: generated,
        verified_survival_through: if matches!(
            terminal,
            Some(RequestTerminalReason::Eos | RequestTerminalReason::Stop)
        ) {
            generated - 1
        } else {
            generated
        },
        terminal,
        completion: None,
    }
}
fn forecast(window: &StoppingWindow, now: Instant, cap: usize, horizon: usize) -> StoppingForecast {
    window.snapshot().forecast(&StoppingQuery {
        rules: &rules("ordinary-eos"),
        verified_survival_through: 0,
        cap,
        horizon,
        now,
    })
}
fn conserved(prediction: &StoppingForecast) {
    assert!(
        (prediction.resolved_mass() + prediction.known_survival_mass + prediction.unresolved_mass
            - 1.0)
            .abs()
            < 1e-12,
        "{prediction:?}"
    );
    assert!(prediction.stops.iter().all(|mass| mass.mass >= 0.0));
}

#[test]
fn retains_actual_8_and_25_endpoints_and_mass() {
    let now = clock();
    let mut window = window(4);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    window.observe(&fact(2, 25, Some(RequestTerminalReason::Stop)), now);
    let prediction = forecast(&window, now, 100, 100);
    assert_eq!(
        prediction.stops,
        vec![
            StoppingMass {
                generated_tokens: 8,
                mass: 0.5
            },
            StoppingMass {
                generated_tokens: 25,
                mass: 0.5
            }
        ]
    );
    assert_eq!(prediction.unresolved_mass, 0.0);
    conserved(&prediction);
}

#[test]
fn missing_terminal_is_only_verified_prefix_then_unknown_not_active() {
    let now = clock();
    let mut window = window(4);
    window.observe(&fact(1, 8, None), now);
    let prefix = forecast(&window, now, 100, 8);
    assert_eq!(prefix.known_survival_mass, 1.0);
    let beyond = forecast(&window, now, 100, 100);
    assert_eq!(beyond.unresolved_mass, 1.0);
    assert!(beyond.stops.is_empty());
    conserved(&beyond);
}

#[test]
fn complete_later_fact_repairs_missing_middle_and_initial_notifications() {
    let now = clock();
    let mut window = window(4);
    window.observe(&fact(1, 2, None), now);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    window.observe(&fact(2, 25, Some(RequestTerminalReason::Eos)), now);
    assert_eq!(forecast(&window, now, 100, 100).stops.len(), 2);
    assert_eq!(window.stats.conflicts, 0);
}

#[test]
fn admission_order_evicts_long_request_before_recent_short_completion() {
    let now = clock();
    let mut window = window(2);
    window.observe(&fact(1, 2, None), now);
    window.observe(&fact(2, 8, Some(RequestTerminalReason::Eos)), now);
    window.observe(&fact(3, 25, Some(RequestTerminalReason::Eos)), now);
    window.observe(&fact(1, 90, Some(RequestTerminalReason::Eos)), now);
    assert_eq!(window.retained_requests(), 2);
    assert_eq!(window.stats.evicted, 1);
    assert_eq!(window.stats.late_retired, 1);
    assert_eq!(forecast(&window, now, 100, 100).stops.len(), 2);
}

#[test]
fn reordering_admissions_does_not_turn_late_old_record_into_new_member() {
    let now = clock();
    let mut window = window(2);
    window.observe(&fact(3, 25, Some(RequestTerminalReason::Eos)), now);
    window.observe(&fact(2, 8, Some(RequestTerminalReason::Eos)), now);
    window.observe(&fact(1, 1, None), now);
    window.observe(&fact(1, 90, Some(RequestTerminalReason::Eos)), now);
    assert_eq!(
        window
            .records
            .iter()
            .map(|record| record.lifecycle.admission)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(window.stats.late_retired, 2);
}

#[test]
fn conflicts_never_restore_a_favorable_terminal() {
    let now = clock();
    let mut window = window(4);
    let natural = fact(1, 8, Some(RequestTerminalReason::Eos));
    window.observe(&natural, now);
    let mut contradiction = natural.clone();
    contradiction.terminal = Some(RequestTerminalReason::Stop);
    window.observe(&contradiction, now);
    let mut later = natural;
    later.revision += 1;
    window.observe(&later, now);
    let prediction = forecast(&window, now, 100, 100);
    assert_eq!(prediction.unresolved_mass, 1.0);
    assert!(prediction.stops.is_empty());
    assert_eq!(window.stats.conflicts, 1);
}

#[test]
fn stale_updates_are_ignored_and_duplicate_terminal_is_idempotent() {
    let now = clock();
    let mut window = window(4);
    let natural = fact(1, 8, Some(RequestTerminalReason::Eos));
    window.observe(&natural, now);
    window.observe(&natural, now);
    let mut retry = natural.clone();
    retry.revision += 1;
    retry.observed_at += Duration::from_secs(61);
    window.observe(&retry, retry.observed_at);
    window.observe(&fact(1, 2, None), now);
    assert_eq!(window.stats.stale_updates, 1);
    assert_eq!(window.stats.conflicts, 0);
    assert_eq!(forecast(&window, now, 100, 100).stops[0].mass, 1.0);
    assert_eq!(
        forecast(&window, retry.observed_at, 100, 100).unresolved_mass,
        1.0
    );
}

#[test]
fn first_support_gap_preserves_partial_stop_mass_and_does_not_move_unknown_to_cap() {
    let now = clock();
    let mut window = window(4);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    window.observe(&fact(2, 8, None), now);
    let prediction = forecast(&window, now, 100, 100);
    assert_eq!(
        prediction.stops,
        vec![StoppingMass {
            generated_tokens: 8,
            mass: 0.5
        }]
    );
    assert_eq!(prediction.unresolved_mass, 0.5);
    conserved(&prediction);
    assert_eq!(prediction.exceedance_interval(0.0), Some((0.0, 0.5)));
}

#[test]
fn cap_precedes_statistical_support_gate_but_not_an_earlier_gap() {
    let now = clock();
    let window = window(4);
    let immediate = forecast(&window, now, 1, 1);
    assert_eq!(
        immediate.stops,
        vec![StoppingMass {
            generated_tokens: 1,
            mass: 1.0
        }]
    );
    assert_eq!(immediate.tpot_undefined_mass, 1.0);
    let later = forecast(&window, now, 2, 2);
    assert!(later.stops.is_empty());
    assert_eq!(later.unresolved_mass, 1.0);
}

#[test]
fn length_cancellation_and_error_are_not_natural_stop_events() {
    let now = clock();
    let mut window = window(4);
    for (index, reason) in [
        RequestTerminalReason::Length,
        RequestTerminalReason::Cancelled,
        RequestTerminalReason::Error,
    ]
    .into_iter()
    .enumerate()
    {
        let mut observation = fact(index as u64 + 1, 8, Some(reason));
        if reason == RequestTerminalReason::Length {
            observation.cap = 8;
        }
        window.observe(&observation, now);
    }
    let prediction = forecast(&window, now, 100, 100);
    assert!(prediction.stops.is_empty());
    assert_eq!(prediction.unresolved_mass, 1.0);
}

#[test]
fn constrained_and_ordinary_full_identities_never_share_support() {
    let now = clock();
    let mut window = window(4);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    let snapshot = window.snapshot();
    for identity in [
        "user-stop-A",
        "grammar-start-after-delimiter",
        "response-boundary",
    ] {
        let prediction = snapshot.forecast(&StoppingQuery {
            rules: &rules(identity),
            verified_survival_through: 0,
            cap: 100,
            horizon: 100,
            now,
        });
        assert_eq!(prediction.unresolved_mass, 1.0);
    }
}

#[test]
fn confirmed_cap_only_is_deterministic_without_claiming_request_success() {
    let now = clock();
    let window = window(4);
    let rules =
        EffectiveStoppingRules::new(b"no-terminators".to_vec(), StoppingMechanism::CapOnly, 64)
            .unwrap();
    let prediction = window.snapshot().forecast(&StoppingQuery {
        rules: &rules,
        verified_survival_through: 0,
        cap: 25,
        horizon: 25,
        now,
    });
    assert_eq!(
        prediction.stops,
        vec![StoppingMass {
            generated_tokens: 25,
            mass: 1.0
        }]
    );
    conserved(&prediction);
}

#[test]
fn ttl_is_per_record_and_snapshot_is_immutable() {
    let now = clock();
    let mut window = window(4);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    let snapshot = window.snapshot();
    window.observe(&fact(2, 25, Some(RequestTerminalReason::Eos)), now);
    let query = |now| StoppingQuery {
        rules: snapshot.records[0].fact.as_ref().unwrap().rules.as_ref(),
        verified_survival_through: 0,
        cap: 100,
        horizon: 100,
        now,
    };
    assert_eq!(snapshot.forecast(&query(now)).stops.len(), 1);
    assert_eq!(
        snapshot
            .forecast(&query(now + Duration::from_secs(61)))
            .unresolved_mass,
        1.0
    );
}

#[test]
fn domain_capacity_identity_bytes_and_forecast_work_are_bounded() {
    let now = clock();
    let mut config = StoppingConfig {
        min_risk_set: 1,
        max_domains: 1,
        max_forecast_positions: 2,
        ..StoppingConfig::default()
    };
    let mut window = StoppingWindow::new(config).unwrap();
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    let mut other = fact(2, 25, Some(RequestTerminalReason::Eos));
    other.rules = rules("different");
    window.observe(&other, now);
    assert_eq!(window.stats.domain_limit, 1);
    assert_eq!(forecast(&window, now, 100, 100).unresolved_mass, 1.0);
    config.max_rule_bytes = 2;
    let mut tiny = StoppingWindow::new(config).unwrap();
    tiny.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    assert_eq!(tiny.stats.invalid, 1);
    assert_eq!(tiny.retained_requests(), 0);
}

#[test]
fn producer_identity_isolates_request_id_reuse_and_lifecycle_domains() {
    let now = clock();
    let mut window = window(4);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    let mut wrong_producer = fact(2, 25, Some(RequestTerminalReason::Eos));
    wrong_producer.lifecycle.producer = 2;
    window.observe(&wrong_producer, now);
    assert_eq!(window.stats.invalid, 1);
    assert_eq!(
        forecast(&window, now, 100, 100).stops[0].generated_tokens,
        8
    );
}

#[test]
fn delayed_delivery_does_not_refresh_producer_fact_and_future_facts_are_rejected() {
    let now = clock();
    let mut window = window(4);
    let observation = fact(1, 8, Some(RequestTerminalReason::Eos));
    window.observe(&observation, now + Duration::from_secs(61));
    assert_eq!(
        forecast(&window, now + Duration::from_secs(61), 100, 100).unresolved_mass,
        1.0
    );
    let mut future = fact(2, 25, Some(RequestTerminalReason::Eos));
    future.observed_at = now + Duration::from_secs(1);
    window.observe(&future, now);
    assert_eq!(window.stats.invalid, 1);
}

#[test]
fn domain_limit_misses_share_the_admission_window_and_allow_old_domains_to_age_out() {
    let now = clock();
    let mut window = StoppingWindow::new(StoppingConfig {
        window_requests: 2,
        max_domains: 1,
        min_risk_set: 1,
        ..Default::default()
    })
    .unwrap();
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    let mut second_domain = fact(2, 25, Some(RequestTerminalReason::Eos));
    second_domain.rules = rules("new-policy");
    for admission in 2..=4 {
        second_domain.lifecycle.admission = admission;
        window.observe(&second_domain, now);
    }
    assert_eq!(window.retained_requests(), 2);
    assert_eq!(window.stats.domain_limit, 1);
    assert_eq!(window.stats.evicted, 2);
    assert_eq!(
        window
            .records
            .iter()
            .filter(|record| record.fact.is_some())
            .count(),
        2
    );
    let forecast = window.snapshot().forecast(&StoppingQuery {
        rules: &second_domain.rules,
        verified_survival_through: 0,
        cap: 100,
        horizon: 100,
        now,
    });
    assert_eq!(
        forecast.stops,
        vec![StoppingMass {
            generated_tokens: 25,
            mass: 1.0
        }]
    );
    assert_eq!(forecast.unresolved_mass, 0.0);
    second_domain.lifecycle.admission = 2;
    window.observe(&second_domain, now);
    assert_eq!(window.stats.late_retired, 1);
}

#[test]
fn matured_completion_cohort_retains_failures_and_unknowns_and_excludes_recent_successes() {
    let start = clock();
    let now = start + Duration::from_secs(40);
    let mut window = window(8);
    for (admission, terminal, outcome) in [
        (
            1,
            Some(RequestTerminalReason::Eos),
            Some(RequestCompletionOutcome::Succeeded),
        ),
        (
            2,
            Some(RequestTerminalReason::Error),
            Some(RequestCompletionOutcome::Failed),
        ),
        (
            3,
            Some(RequestTerminalReason::Cancelled),
            Some(RequestCompletionOutcome::Failed),
        ),
        (4, None, None),
    ] {
        let mut observation = fact(admission, 8, terminal);
        observation.completion = outcome;
        window.observe(&observation, now);
    }
    for (admission, terminal, outcome) in [
        (
            5,
            Some(RequestTerminalReason::Eos),
            Some(RequestCompletionOutcome::Succeeded),
        ),
        (6, None, None),
    ] {
        let mut recent = fact(admission, 8, terminal);
        recent.admitted_at = now - Duration::from_secs(2);
        recent.observed_at = now;
        recent.completion = outcome;
        window.observe(&recent, now);
    }
    let key = rules("ordinary-eos");
    let query = CompletionQuery {
        rules: &key,
        cap: 100,
        maturity: Duration::from_secs(30),
        now,
    };
    let forecast = window.snapshot().completion_forecast(&query);
    assert_eq!(forecast.samples, 4);
    assert_eq!(forecast.succeeded_mass, 0.25);
    assert_eq!(forecast.failed_mass, 0.5);
    assert_eq!(forecast.unresolved_mass, 0.25);
    assert_eq!(
        forecast.approximation,
        Some(CompletionApproximation::EmpiricalMaturedCohort)
    );
    assert_eq!(
        window
            .snapshot()
            .completion_forecast(&CompletionQuery { cap: 99, ..query }),
        CompletionForecast::unknown()
    );
    let stale = window.snapshot().completion_forecast(&CompletionQuery {
        now: now + Duration::from_secs(100),
        ..query
    });
    assert_eq!(stale.samples, 6);
    assert_eq!(stale.unresolved_mass, 1.0);
}

#[test]
fn product_completion_enrichment_preserves_natural_stop_freshness_and_conflicts_stay_unknown() {
    let start = clock();
    let mut window = window(2);
    let mut endpoint = fact(1, 8, Some(RequestTerminalReason::Eos));
    window.observe(&endpoint, start);
    endpoint.revision += 1;
    endpoint.observed_at = start + Duration::from_secs(35);
    endpoint.completion = Some(RequestCompletionOutcome::Succeeded);
    window.observe(&endpoint, endpoint.observed_at);
    let key = rules("ordinary-eos");
    let query = CompletionQuery {
        rules: &key,
        cap: 100,
        maturity: Duration::from_secs(30),
        now: start + Duration::from_secs(61),
    };
    assert_eq!(
        window.snapshot().completion_forecast(&query).succeeded_mass,
        1.0
    );
    assert_eq!(
        forecast(&window, query.now, 100, 100).unresolved_mass,
        1.0,
        "late product delivery cannot refresh the earlier natural endpoint"
    );
    endpoint.revision += 1;
    endpoint.completion = Some(RequestCompletionOutcome::Failed);
    endpoint.observed_at = query.now;
    window.observe(&endpoint, query.now);
    assert_eq!(
        window
            .snapshot()
            .completion_forecast(&query)
            .unresolved_mass,
        1.0
    );
    endpoint.revision += 1;
    endpoint.completion = Some(RequestCompletionOutcome::Succeeded);
    window.observe(&endpoint, query.now);
    assert_eq!(
        window
            .snapshot()
            .completion_forecast(&query)
            .unresolved_mass,
        1.0
    );
    assert_eq!(window.stats.conflicts, 1);
}

#[test]
fn admission_clock_cannot_be_renewed_by_recompute_or_completion() {
    let now = clock();
    let mut window = window(2);
    let first = fact(1, 2, None);
    window.observe(&first, now);
    let mut renewal = fact(1, 8, Some(RequestTerminalReason::Eos));
    renewal.admitted_at = now + Duration::from_secs(1);
    renewal.observed_at = renewal.admitted_at;
    renewal.completion = Some(RequestCompletionOutcome::Succeeded);
    window.observe(&renewal, renewal.observed_at);
    assert_eq!(window.stats.conflicts, 1);
    let key = rules("ordinary-eos");
    let query = CompletionQuery {
        rules: &key,
        cap: 100,
        maturity: Duration::from_secs(30),
        now: now + Duration::from_secs(40),
    };
    assert_eq!(
        window
            .snapshot()
            .completion_forecast(&query)
            .unresolved_mass,
        1.0
    );
}

#[test]
fn unseen_serials_are_unknown_and_large_gaps_do_not_retain_old_completions() {
    let now = clock();
    let mut window = window(2);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    assert!(window.advance_admission_watermark(RequestLifecycleId {
        producer: 1,
        admission: u64::MAX
    }));
    assert_eq!(window.retained_requests(), 0);
    assert_eq!(window.stats().unobserved_lifetimes, 2);
    let mut observed = fact(u64::MAX, 25, Some(RequestTerminalReason::Eos));
    observed.completion = Some(RequestCompletionOutcome::Succeeded);
    window.observe(&observed, now);
    assert_eq!(window.stats().unobserved_lifetimes, 1);
    window.observe(&fact(1, 8, Some(RequestTerminalReason::Eos)), now);
    assert_eq!(window.stats.late_retired, 1);
    let stopping = forecast(&window, now, 100, 100);
    assert_eq!(
        stopping.stops,
        vec![StoppingMass {
            generated_tokens: 25,
            mass: 0.5
        }]
    );
    assert_eq!(stopping.unresolved_mass, 0.5);
    conserved(&stopping);
    let key = rules("ordinary-eos");
    let completion = window.snapshot().completion_forecast(&CompletionQuery {
        rules: &key,
        cap: 100,
        maturity: Duration::from_secs(30),
        now: now + Duration::from_secs(40),
    });
    assert_eq!(completion.samples, 2);
    assert_eq!(completion.succeeded_mass, 0.5);
    assert_eq!(completion.unresolved_mass, 0.5);
}

#[test]
fn recent_missing_and_domain_rejected_admissions_enter_only_when_mature() {
    let start = clock();
    let now = start + Duration::from_secs(40);
    let mut window = StoppingWindow::new(StoppingConfig {
        window_requests: 8,
        max_domains: 1,
        min_risk_set: 1,
        ..Default::default()
    })
    .unwrap();
    let mut completed = fact(1, 8, Some(RequestTerminalReason::Eos));
    completed.completion = Some(RequestCompletionOutcome::Succeeded);
    window.observe(&completed, now);
    // Arrival time is independent of serial order and notification order.
    window.record_admission(
        RequestLifecycleId {
            producer: 1,
            admission: 3,
        },
        now,
    );
    window.record_admission(
        RequestLifecycleId {
            producer: 1,
            admission: 2,
        },
        start,
    );
    let mut rejected = fact(4, 2, None);
    rejected.admitted_at = now;
    rejected.observed_at = now;
    rejected.rules = rules("rejected-domain");
    window.observe(&rejected, now);
    assert_eq!(window.stats().unobserved_lifetimes, 2);
    assert_eq!(window.stats().domain_limit, 1);
    let query = CompletionQuery {
        rules: &completed.rules,
        cap: 100,
        maturity: Duration::from_secs(30),
        now,
    };
    let current = window.snapshot().completion_forecast(&query);
    assert_eq!(current.samples, 2);
    assert_eq!(current.succeeded_mass, 0.5);
    assert_eq!(
        current.unresolved_mass, 0.5,
        "only the actually old missing serial belongs to this cohort"
    );
    let later = window.snapshot().completion_forecast(&CompletionQuery {
        now: now + Duration::from_secs(30),
        ..query
    });
    assert_eq!(later.samples, 4);
    assert_eq!(
        later.unresolved_mass, 1.0,
        "newly mature missing members and expired success stay unknown"
    );
}

#[test]
fn stale_hazard_members_cannot_dilute_whole_lifetime_unknown_mass() {
    let start = clock();
    let now = start + Duration::from_secs(61);
    let mut window = window(10);
    for admission in 1..=8 {
        window.observe(&fact(admission, 8, Some(RequestTerminalReason::Eos)), now);
    }
    let mut fresh = fact(9, 8, Some(RequestTerminalReason::Eos));
    fresh.observed_at = now;
    window.observe(&fresh, now);
    window.record_admission(
        RequestLifecycleId {
            producer: 1,
            admission: 10,
        },
        now,
    );
    let result = forecast(&window, now, 100, 100);
    assert_eq!(
        result.stops,
        vec![StoppingMass {
            generated_tokens: 8,
            mass: 0.5
        }]
    );
    assert_eq!(result.unresolved_mass, 0.5);
    conserved(&result);
}
