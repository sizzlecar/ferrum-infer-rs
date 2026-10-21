//! Differential checks against the original per-position risk-set calculation.
//! Fixtures enter through the real observation/retirement/conflict boundaries.

use super::*;
use std::{hint::black_box, sync::Arc};

fn rules(name: &str, mechanism: StoppingMechanism) -> Arc<EffectiveStoppingRules> {
    Arc::new(EffectiveStoppingRules::new(name.as_bytes().to_vec(), mechanism, 64).unwrap())
}

fn observation(
    admission: u64,
    generated: usize,
    terminal: Option<RequestTerminalReason>,
    at: Instant,
) -> RequestStoppingObservation {
    RequestStoppingObservation {
        lifecycle: RequestLifecycleId {
            producer: 1,
            admission,
        },
        revision: 1,
        admitted_at: at - Duration::from_secs(1),
        observed_at: at,
        rules: rules("observed", StoppingMechanism::Observed),
        cap: if terminal == Some(RequestTerminalReason::Length) {
            generated
        } else {
            generated.max(64)
        },
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

fn mixed_window(config: StoppingConfig, now: Instant) -> StoppingWindow {
    let mut window = StoppingWindow::new(config).unwrap();
    for (admission, generated, terminal, age) in [
        (1, 1, Some(RequestTerminalReason::Eos), 59),
        (2, 4, Some(RequestTerminalReason::Stop), 60),
        (3, 8, Some(RequestTerminalReason::Length), 1),
        (4, 6, None, 0),
        (5, 5, Some(RequestTerminalReason::Error), 0),
        (6, 7, Some(RequestTerminalReason::Cancelled), 61),
        (7, 12, Some(RequestTerminalReason::Eos), 30),
    ] {
        window.observe(
            &observation(
                admission,
                generated,
                terminal,
                now - Duration::from_secs(age),
            ),
            now,
        );
    }
    // A contradiction leaves only the common verified prefix, never its old
    // natural endpoint. A later favorable update cannot heal the conflict.
    let original = observation(8, 10, Some(RequestTerminalReason::Eos), now);
    window.observe(&original, now);
    let mut conflicting = observation(8, 8, Some(RequestTerminalReason::Stop), now);
    conflicting.revision = 2;
    window.observe(&conflicting, now);
    let mut later = original.clone();
    later.revision = 3;
    window.observe(&later, now);
    assert_eq!(window.stats().conflicts, 1);

    // Completion enrichment is fresh, while the hazard's original timestamp
    // remains exactly on its TTL boundary.
    let terminal = observation(9, 16, Some(RequestTerminalReason::Eos), now - config.ttl);
    window.observe(&terminal, now);
    let mut enriched = terminal.clone();
    enriched.revision = 2;
    enriched.observed_at = now;
    enriched.completion = Some(RequestCompletionOutcome::Succeeded);
    window.observe(&enriched, now);

    let mut other = observation(10, 4, Some(RequestTerminalReason::Eos), now);
    other.rules = rules("other", StoppingMechanism::Observed);
    window.observe(&other, now);
    let mut unsupported = observation(11, 8, Some(RequestTerminalReason::Eos), now);
    unsupported.rules = rules("third-domain", StoppingMechanism::Observed);
    window.observe(&unsupported, now);
    assert_eq!(window.stats().domain_limit, 1);
    // An unseen serial and the domain-limit miss both remain unknown members.
    window.advance_admission_watermark(RequestLifecycleId {
        producer: 1,
        admission: 12,
    });
    assert_eq!(window.stats().unobserved_lifetimes, 1);
    assert_eq!(window.stats().invalid, 0);
    window
}

#[test]
fn forecast_scan_matches_position_oracle_at_real_window_boundaries() {
    let now = Instant::now() + Duration::from_secs(120);
    for min_risk_set in [1, 4] {
        for max_forecast_positions in [1, 3, 64] {
            let window = mixed_window(
                StoppingConfig {
                    window_requests: 16,
                    max_domains: 2,
                    min_risk_set,
                    max_forecast_positions,
                    ..Default::default()
                },
                now,
            );
            let snapshot = window.snapshot();
            for at in [
                now - Duration::from_nanos(1),
                now,
                now + Duration::from_nanos(1),
                now + Duration::from_secs(60),
                now + Duration::from_secs(60) + Duration::from_nanos(1),
            ] {
                for key in [
                    rules("observed", StoppingMechanism::Observed),
                    rules("other", StoppingMechanism::Observed),
                    rules("observed", StoppingMechanism::CapOnly),
                ] {
                    for cap in [1, 8, 64] {
                        for survival in [0, 1, 3, 7, 63, cap - 1, cap] {
                            for horizon in [0, survival, survival + 1, cap, usize::MAX] {
                                let query = StoppingQuery {
                                    rules: &key,
                                    verified_survival_through: survival,
                                    cap,
                                    horizon,
                                    now: at,
                                };
                                assert_eq!(
                                    snapshot.forecast(&query),
                                    position_oracle(&snapshot, &query),
                                    "risk={min_risk_set} work={max_forecast_positions} cap={cap} survival={survival} horizon={horizon} at={at:?} rules={key:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn forecast_scan_preserves_first_gap_cap_priority_and_partial_horizon() {
    let now = Instant::now();
    let mut window = StoppingWindow::new(StoppingConfig {
        min_risk_set: 2,
        max_forecast_positions: 3,
        ..Default::default()
    })
    .unwrap();
    window.observe(
        &observation(1, 2, Some(RequestTerminalReason::Eos), now),
        now,
    );
    window.observe(&observation(2, 2, None, now), now);
    let snapshot = window.snapshot();
    let key = rules("observed", StoppingMechanism::Observed);
    let mut query = StoppingQuery {
        rules: &key,
        verified_survival_through: 0,
        cap: 4,
        horizon: 4,
        now,
    };
    let prediction = snapshot.forecast(&query);
    assert_eq!(prediction, position_oracle(&snapshot, &query));
    assert_eq!(
        prediction.stops,
        vec![StoppingMass {
            generated_tokens: 2,
            mass: 0.5
        }]
    );
    assert_eq!(prediction.unresolved_mass, 0.5);
    assert_eq!(prediction.known_survival_mass, 0.0);
    query.horizon = 2;
    let partial = snapshot.forecast(&query);
    assert_eq!(partial, position_oracle(&snapshot, &query));
    assert_eq!(partial.known_survival_mass, 0.5);
    assert_eq!(partial.unresolved_mass, 0.0);

    let mut full = StoppingWindow::new(StoppingConfig {
        min_risk_set: 1,
        max_forecast_positions: 3,
        ..Default::default()
    })
    .unwrap();
    full.observe(
        &observation(1, 4, Some(RequestTerminalReason::Length), now),
        now,
    );
    let snapshot = full.snapshot();
    query.horizon = 4;
    assert_eq!(snapshot.forecast(&query).resolved_mass(), 1.0);
    assert_eq!(
        snapshot.forecast(&query),
        position_oracle(&snapshot, &query)
    );
    // With one less statistical position, the gap precedes the cap. It cannot
    // be silently converted into a deterministic cap endpoint.
    full.config.max_forecast_positions = 2;
    let snapshot = full.snapshot();
    assert_eq!(snapshot.forecast(&query).unresolved_mass, 1.0);
    assert_eq!(
        snapshot.forecast(&query),
        position_oracle(&snapshot, &query)
    );
}

#[test]
fn forecast_scan_bounds_scratch_at_extreme_query_frontiers() {
    let now = Instant::now();
    let mut window = StoppingWindow::new(StoppingConfig {
        min_risk_set: 1,
        max_forecast_positions: 2,
        ..Default::default()
    })
    .unwrap();
    window.observe(
        &observation(1, usize::MAX, Some(RequestTerminalReason::Length), now),
        now,
    );
    let snapshot = window.snapshot();
    let key = rules("observed", StoppingMechanism::Observed);
    for survival in [
        0,
        usize::MAX - 3,
        usize::MAX - 2,
        usize::MAX - 1,
        usize::MAX,
    ] {
        let query = StoppingQuery {
            rules: &key,
            verified_survival_through: survival,
            cap: usize::MAX,
            horizon: usize::MAX,
            now,
        };
        assert_eq!(
            snapshot.forecast(&query),
            position_oracle(&snapshot, &query)
        );
    }
    assert!(StoppingConfig {
        max_forecast_positions: usize::MAX,
        ..Default::default()
    }
    .validate()
    .is_err());
}

#[test]
#[ignore = "CPU-only timing comparison; report elapsed values, never a performance pass threshold"]
fn forecast_scan_cpu_timing() {
    let now = Instant::now();
    let key = rules("observed", StoppingMechanism::Observed);
    for supported in [false, true] {
        let mut window = StoppingWindow::new(StoppingConfig::default()).unwrap();
        for admission in 1..=window.config.window_requests as u64 {
            let fact = if supported {
                observation(admission, 64, Some(RequestTerminalReason::Length), now)
            } else {
                observation(admission, 0, None, now)
            };
            window.observe(&fact, now);
        }
        let snapshot = window.snapshot();
        let query = StoppingQuery {
            rules: &key,
            verified_survival_through: 0,
            cap: 64,
            horizon: 64,
            now,
        };
        assert_eq!(
            snapshot.forecast(&query),
            position_oracle(&snapshot, &query)
        );
        let iterations = 500;
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(position_oracle(black_box(&snapshot), black_box(&query)));
        }
        let original = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(snapshot.forecast(black_box(&query)));
        }
        let swept = start.elapsed();
        eprintln!(
            "forecast_scan supported={supported} records={} cap=64 iterations={iterations} original={original:?} swept={swept:?}",
            snapshot.records.len()
        );
    }
}

/// Original implementation retained only as a differential oracle. It tests
/// record membership separately at each position; it does not use event counts
/// or the continuous-interval transformation under test.
fn position_oracle(table: &StoppingTable, query: &StoppingQuery<'_>) -> StoppingForecast {
    if query.cap == 0
        || query.verified_survival_through >= query.cap
        || query.horizon <= query.verified_survival_through
        || query.rules.identity().len() > table.config.max_rule_bytes
    {
        return StoppingForecast::unknown();
    }
    let horizon = query.horizon.min(query.cap);
    if query.rules.mechanism() == StoppingMechanism::CapOnly
        || query.verified_survival_through.checked_add(1) == Some(query.cap)
    {
        return if horizon == query.cap {
            StoppingForecast {
                stops: vec![StoppingMass {
                    generated_tokens: query.cap,
                    mass: 1.0,
                }],
                known_survival_mass: 0.0,
                unresolved_mass: 0.0,
                tpot_undefined_mass: if query.cap < 2 { 1.0 } else { 0.0 },
            }
        } else {
            StoppingForecast {
                stops: Vec::new(),
                known_survival_mass: 1.0,
                unresolved_mass: 0.0,
                tpot_undefined_mass: 0.0,
            }
        };
    }
    let unknown_members = table.unobserved_admission_times.len()
        + table
            .records
            .iter()
            .filter(|record| record.fact.is_none())
            .count();
    let known_members = table
        .records
        .iter()
        .filter(|record| {
            record.fact.as_ref().is_some_and(|fact| {
                fact.rules.as_ref() == query.rules
                    && query
                        .now
                        .checked_duration_since(record.observed_at)
                        .is_some_and(|age| age <= table.config.ttl)
                    && !(matches!(
                        fact.terminal,
                        Some(RequestTerminalReason::Eos | RequestTerminalReason::Stop)
                    ) && fact.generated_tokens <= query.verified_survival_through)
            })
        })
        .count();
    let unknown_mass = if unknown_members == 0 {
        0.0
    } else {
        unknown_members as f64 / (known_members + unknown_members) as f64
    };
    let mut forecast = StoppingForecast {
        stops: Vec::new(),
        known_survival_mass: 1.0 - unknown_mass,
        unresolved_mass: unknown_mass,
        tpot_undefined_mass: 0.0,
    };
    for (position, generated) in ((query.verified_survival_through + 1)..=horizon).enumerate() {
        if generated == query.cap {
            push_stop(&mut forecast, generated, 1.0);
            break;
        }
        if position >= table.config.max_forecast_positions {
            forecast.unresolved_mass += forecast.known_survival_mass;
            forecast.known_survival_mass = 0.0;
            break;
        }
        let mut risk = 0usize;
        let mut ended = 0usize;
        for record in &table.records {
            let Some(fact) = &record.fact else { continue };
            if fact.rules.as_ref() != query.rules
                || query
                    .now
                    .checked_duration_since(record.observed_at)
                    .is_none_or(|age| age > table.config.ttl)
            {
                continue;
            }
            let natural_end = !record.conflicted
                && matches!(
                    fact.terminal,
                    Some(RequestTerminalReason::Eos | RequestTerminalReason::Stop)
                );
            if natural_end && fact.generated_tokens == generated {
                risk += 1;
                ended += 1;
            } else if fact.verified_survival_through >= generated {
                risk += 1;
            }
        }
        if risk < table.config.min_risk_set {
            forecast.unresolved_mass += forecast.known_survival_mass;
            forecast.known_survival_mass = 0.0;
            break;
        }
        if ended > 0 {
            push_stop(&mut forecast, generated, ended as f64 / risk as f64);
        }
        if forecast.known_survival_mass <= 0.0 {
            break;
        }
    }
    forecast
}
