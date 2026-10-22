use super::*;
use std::time::Duration;

fn digest(value: u8) -> Digest {
    [value; 32]
}
fn spec() -> Spec {
    Spec {
        absolute: AbsoluteLimits {
            quantile: Fraction {
                numerator: 95,
                denominator: 100,
            },
            ttft_ms: 1000.0,
            tpot_ms: 1000.0,
            visible_itl_ms: 1000.0,
            request_deadline: Duration::from_secs(30),
        },
        tolerances: Tolerances {
            feasible_min_throughput_ratio: 0.95,
            feasible_max_latency_ratio: 1.05,
            rescue_min_throughput_ratio: 0.93,
        },
        kind: CandidateKind::Controller,
        bootstrap: Bootstrap {
            prng: Prng::SplitMix64V1,
            seed: 17,
            replicates: 333,
            max_rng_draws: 100_000,
            confidence: Fraction {
                numerator: 95,
                denominator: 100,
            },
            episode_sensitivity: false,
        },
        budget: Budget {
            max_runs: 20,
            max_requests: 1000,
            max_events: 10_000,
            max_replicates: 1000,
            max_sample_visits: 1_000_000,
            max_scratch_bytes: 1_000_000,
        },
    }
}
fn fixture(scales: &[f64]) -> (Workload, Vec<RunPair>) {
    let keys: Vec<_> = (0..2)
        .map(|i| RequestKey {
            index: i,
            episode: i as u32,
            source: digest(i as u8 + 1),
            prompt: digest(i as u8 + 3),
            scheduled_ms: i as u64 * 5,
            output_cap: 8,
        })
        .collect();
    let workload = Workload {
        cell: digest(10),
        inputs: digest(11),
        repeat_ids: (0..scales.len() as u64).collect(),
        requests: keys,
    };
    let pairs = scales
        .iter()
        .enumerate()
        .map(|(i, scale)| {
            let requests = workload
                .requests
                .iter()
                .map(|key| Request {
                    key: key.clone(),
                    outcome: Outcome::Success,
                    usage: Some(Usage {
                        prompt: 4,
                        completion: 2,
                        total: 6,
                    }),
                    finish: Some(Finish::Stop),
                    done: true,
                    terminal_ms: Some(key.scheduled_ms as f64 + 12.0 * scale + 100.0),
                    ttft_from_scheduled_ms: Some(10.0 * scale),
                    visible_span_ms: Some(2.0 * scale),
                    visible_text_events: Some(2),
                    visible_gaps_ms: vec![2.0 * scale],
                })
                .collect();
            let baseline = Run {
                repeat_id: i as u64,
                cell: workload.cell,
                inputs: workload.inputs,
                server_incarnation: digest((i * 2 + 40) as u8),
                client_exit: Some(0),
                requests,
            };
            let mut candidate = baseline.clone();
            candidate.server_incarnation = digest((i * 2 + 41) as u8);
            RunPair {
                baseline,
                candidate,
            }
        })
        .collect();
    (workload, pairs)
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
        "{a} != {b}"
    );
}

#[test]
fn primary_preserves_pairing_under_shared_restart_variability_and_seed_replays() {
    let (workload, mut pairs) = fixture(&[1.0, 7.0, 17.0]);
    for pair in &mut pairs {
        for r in &mut pair.candidate.requests {
            r.ttft_from_scheduled_ms = r.ttft_from_scheduled_ms.map(|v| v * 1.02);
            r.visible_span_ms = r.visible_span_ms.map(|v| v * 1.02);
            r.visible_gaps_ms.iter_mut().for_each(|v| *v *= 1.02);
        }
    }
    let first = analyze(&workload, &pairs, spec()).unwrap();
    let second = analyze(&workload, &pairs, spec()).unwrap();
    let a = first.primary.unwrap().ratios;
    assert_eq!(a, second.primary.unwrap().ratios);
    for metric in [a.ttft, a.tpot, a.visible_itl] {
        near(metric.estimate, 1.02);
        near(metric.lower, 1.02);
        near(metric.upper, 1.02);
    }
    assert_eq!(first.candidate_absolute, Gate::Pass);
    assert_eq!(first.development_gate, Gate::Pass);
    assert_eq!(first.method_status, "DRAFT_METHOD_NOT_FORMAL_ACCEPTANCE");
}

#[test]
fn client_nearest_rank_and_whole_event_vectors_are_not_mean_episode_percentiles() {
    assert_eq!(
        rank(
            &[1.0, 2.0, 3.0, 4.0],
            Fraction {
                numerator: 95,
                denominator: 100
            }
        ),
        Ok(4.0)
    );
    let (_, mut pairs) = fixture(&[1.0, 1.0]);
    let run = &mut pairs[0].baseline;
    run.requests[0].visible_span_ms = Some(1.0);
    run.requests[0].visible_gaps_ms = vec![1.0];
    run.requests[1].usage = Some(Usage {
        prompt: 4,
        completion: 4,
        total: 8,
    });
    run.requests[1].visible_span_ms = Some(60.0);
    run.requests[1].visible_text_events = Some(4);
    run.requests[1].visible_gaps_ms = vec![10.0, 20.0, 30.0];
    let mut config = spec();
    config.absolute.quantile = Fraction {
        numerator: 1,
        denominator: 2,
    };
    let combined = episode_quantiles(run, &[0, 1], config, 20).unwrap();
    assert_eq!(combined[2], 10.0); // pooled [1,10,20,30], not (1+20)/2
    let repeated = episode_quantiles(run, &[1, 1], config, 20).unwrap();
    assert_eq!(repeated[2], 20.0); // [10,20,30] copied as one episode twice
}

#[test]
fn overlapping_episodes_keep_original_run_throughput_and_sensitivity_cannot_replace_primary() {
    let (workload, mut pairs) = fixture(&[1.0, 2.0, 3.0]);
    for pair in &mut pairs {
        for run in [&mut pair.baseline, &mut pair.candidate] {
            run.requests[0].terminal_ms = Some(100.0);
            run.requests[1].terminal_ms = Some(200.0);
        }
    }
    let primary_only = analyze(&workload, &pairs, spec()).unwrap();
    assert_eq!(
        primary_only.pairs[0].candidate.finite_drain_seconds,
        Some(0.2)
    );
    assert_eq!(primary_only.pairs[0].candidate.throughput_tps, Some(20.0));
    let mut config = spec();
    config.bootstrap.episode_sensitivity = true;
    let with_sensitivity = analyze(&workload, &pairs, config).unwrap();
    assert_eq!(
        primary_only.primary.unwrap().ratios,
        with_sensitivity.primary.unwrap().ratios
    );
    assert_eq!(
        primary_only.development_gate,
        with_sensitivity.development_gate
    );
    assert!(with_sensitivity.episode_sensitivity.unwrap().is_ok());
}

#[test]
fn observed_rejection_retains_all_arrivals_and_absolute_fail_despite_unavailable_inference() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    let failed = &mut pairs[0].candidate.requests[1];
    failed.outcome = Outcome::Failed(FailureKind::Rejected);
    failed.usage = None;
    failed.finish = None;
    failed.done = false;
    failed.visible_gaps_ms.clear();
    failed.visible_text_events = Some(0);
    failed.visible_span_ms = None;
    failed.ttft_from_scheduled_ms = None;
    failed.terminal_ms = Some(200.0);
    let report = analyze(&workload, &pairs, spec()).unwrap();
    let run = &report.pairs[0].candidate;
    assert_eq!((run.arrivals, run.successes, run.failures), (2, 1, 1));
    assert_eq!(run.successful_usage_tokens, Some(2));
    assert_eq!(run.throughput_tps, Some(10.0)); // rejected arrival remains terminal denominator
    assert_eq!(report.candidate_absolute, Gate::Fail);
    assert!(matches!(report.primary, Err(Issue::FailedRequest)));
    assert_eq!(report.development_gate, Gate::Fail);
}

#[test]
fn one_absolute_run_violation_cannot_be_hidden_by_a_passing_relative_interval() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0, 1.0]);
    let mut config = spec();
    config.absolute.ttft_ms = 55.0;
    config.tolerances.feasible_max_latency_ratio = 2.0;
    for pair in &mut pairs {
        for r in &mut pair.baseline.requests {
            r.ttft_from_scheduled_ms = Some(50.0);
        }
    }
    for r in &mut pairs[0].candidate.requests {
        r.ttft_from_scheduled_ms = Some(60.0);
    }
    let report = analyze(&workload, &pairs, config).unwrap();
    assert_eq!(report.primary.unwrap().tolerance_gate, Gate::Pass);
    assert_eq!(report.pairs[0].candidate.absolute, Gate::Fail);
    assert_eq!(report.development_gate, Gate::Fail);
}

#[test]
fn mixed_static_feasibility_does_not_pick_the_more_permissive_rescue_tolerance() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    for r in &mut pairs[0].baseline.requests {
        r.ttft_from_scheduled_ms = Some(60.0);
    }
    let mut config = spec();
    config.absolute.ttft_ms = 55.0;
    let report = analyze(&workload, &pairs, config).unwrap();
    assert_eq!(report.pairs[0].baseline.absolute, Gate::Fail);
    assert_eq!(report.pairs[1].baseline.absolute, Gate::Pass);
    assert!(matches!(report.primary, Err(Issue::MixedStaticFeasibility)));
    assert_eq!(
        report.development_gate,
        Gate::Unresolved(Issue::MixedStaticFeasibility)
    );
}

#[test]
fn missing_terminal_and_short_output_undefined_are_never_filled_with_zero() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    pairs[0].candidate.requests[1].outcome = Outcome::Missing;
    pairs[0].candidate.requests[1].terminal_ms = None;
    pairs[0].candidate.client_exit = None;
    let missing = analyze(&workload, &pairs, spec()).unwrap();
    assert_eq!(missing.pairs[0].candidate.arrivals, 2);
    assert_eq!(missing.pairs[0].candidate.missing, 1);
    assert!(matches!(missing.development_gate, Gate::Unresolved(_)));
    let (workload, mut short) = fixture(&[1.0, 1.0]);
    for pair in &mut short {
        for run in [&mut pair.baseline, &mut pair.candidate] {
            for r in &mut run.requests {
                r.usage = Some(Usage {
                    prompt: 4,
                    completion: 1,
                    total: 5,
                });
                r.visible_span_ms = Some(0.0);
                r.visible_text_events = Some(1);
                r.visible_gaps_ms.clear();
            }
        }
    }
    let result = analyze(&workload, &short, spec()).unwrap();
    assert_eq!(result.pairs[0].candidate.tpot_undefined, 2);
    assert!(result.pairs[0].candidate.tpot.is_none());
    assert!(matches!(result.primary, Err(Issue::UndefinedMetric)));
}

#[test]
fn known_ttft_failure_is_not_downgraded_when_another_metric_is_missing() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    let mut config = spec();
    config.absolute.ttft_ms = 5.0;
    pairs[0].candidate.requests[0].visible_span_ms = None;
    let report = analyze(&workload, &pairs, config).unwrap();
    assert_eq!(report.pairs[0].candidate.target_quantiles[0], Some(10.0));
    assert_eq!(report.candidate_absolute, Gate::Fail);
    assert_eq!(report.development_gate, Gate::Fail);
}

#[test]
fn failed_stream_long_stall_remains_in_visible_itl_without_usage_or_first_timestamp() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    let failed = &mut pairs[0].candidate.requests[1];
    failed.outcome = Outcome::Failed(FailureKind::Transport);
    failed.usage = None;
    failed.finish = None;
    failed.done = false;
    failed.ttft_from_scheduled_ms = None;
    failed.visible_span_ms = Some(1000.0);
    failed.visible_text_events = Some(2);
    failed.visible_gaps_ms = vec![1000.0];
    failed.terminal_ms = Some(2000.0);
    let report = analyze(&workload, &pairs, spec()).unwrap();
    let run = &report.pairs[0].candidate;
    assert_eq!((run.arrivals, run.successes, run.failures), (2, 1, 1));
    assert_eq!(run.visible_intervals, 2);
    assert_eq!(run.itl_unknown_requests, 0);
    assert_eq!(run.visible_itl.unwrap().p95, 1000.0); // [2,1000], not survivor-only2
    assert_eq!(run.target_quantiles[2], Some(1000.0));
    assert_eq!(run.tpot_unknown, 0); // TPOT population remains successful requests
    assert!(matches!(report.primary, Err(Issue::FailedRequest)));
    assert_eq!(report.development_gate, Gate::Fail);
}

#[test]
fn mismatched_pairs_zero_denominators_and_real_resource_exhaustion_are_explicit() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    pairs[0].candidate.requests[0].key.prompt = digest(99);
    assert!(matches!(
        analyze(&workload, &pairs, spec()),
        Err(Issue::WorkloadMismatch)
    ));
    let (workload, pairs) = fixture(&[1.0, 1.0]);
    let mut config = spec();
    config.budget.max_scratch_bytes = 0;
    assert!(matches!(
        analyze(&workload, &pairs, config),
        Err(Issue::Budget)
    ));
    config = spec();
    config.bootstrap.max_rng_draws = 1;
    let report = analyze(&workload, &pairs, config).unwrap();
    assert!(matches!(report.primary, Err(Issue::Budget)));
    assert_eq!(report.development_gate, Gate::Unresolved(Issue::Budget));
    assert_eq!(log_ratio(1.0, 0.0), Err(Issue::NonpositiveRatio));
}

#[test]
fn invalid_usage_counts_unknown_tpot_once_and_keeps_observed_ttft() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    pairs[0].candidate.requests[0].usage = Some(Usage {
        prompt: 4,
        completion: 2,
        total: 99,
    });
    let report = analyze(&workload, &pairs, spec()).unwrap();
    let run = &report.pairs[0].candidate;
    assert_eq!(run.tpot_unknown, 1);
    assert_eq!(run.tpot_undefined, 0);
    assert_eq!(run.protocol_violations, 1);
    assert_eq!(run.target_quantiles[0], Some(10.0));
    assert_eq!(report.development_gate, Gate::Fail);
}

#[test]
fn one_cluster_cannot_support_inference_but_its_observed_failure_still_fails() {
    let (workload, mut pairs) = fixture(&[1.0]);
    let failed = &mut pairs[0].candidate.requests[1];
    failed.outcome = Outcome::Failed(FailureKind::Rejected);
    failed.usage = None;
    failed.finish = None;
    failed.done = false;
    failed.visible_text_events = Some(0);
    failed.visible_gaps_ms.clear();
    failed.visible_span_ms = None;
    failed.ttft_from_scheduled_ms = None;
    let report = analyze(&workload, &pairs, spec()).unwrap();
    assert_eq!(report.pairs.len(), 1);
    assert_eq!(report.pairs[0].candidate.failures, 1);
    assert_eq!(report.candidate_absolute, Gate::Fail);
    assert!(matches!(
        report.primary,
        Err(Issue::InsufficientRunClusters)
    ));
    assert_eq!(report.development_gate, Gate::Fail);
}

#[test]
fn partial_ttft_can_prove_failure_but_cannot_prove_pass_or_an_exact_quantile() {
    let mut config = spec();
    config.absolute.ttft_ms = 50.0;
    for (observed, expected) in [
        (100.0, Gate::Fail),
        (50.0, Gate::Unresolved(Issue::UndefinedMetric)),
    ] {
        let (workload, mut pairs) = fixture(&[1.0, 1.0]);
        pairs[0].candidate.requests[0].ttft_from_scheduled_ms = Some(observed);
        pairs[0].candidate.requests[1].ttft_from_scheduled_ms = None;
        let report = analyze(&workload, &pairs, config).unwrap();
        let run = &report.pairs[0].candidate;
        assert_eq!(run.ttft.unwrap().p95, observed); // descriptive known sample only
        assert_eq!(run.target_quantiles[0], None);
        assert_eq!(run.latency_gates[0], expected);
        assert_eq!(run.absolute, expected);
        assert!(report.primary.is_err());
        assert_eq!(report.development_gate == Gate::Fail, observed > 50.0);
    }
}

#[test]
fn missing_outcome_is_not_excluded_from_the_possible_latency_population() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    pairs[0].candidate.requests[0].ttft_from_scheduled_ms = Some(100.0);
    let missing = &mut pairs[0].candidate.requests[1];
    missing.outcome = Outcome::Missing;
    missing.usage = None;
    missing.finish = None;
    missing.done = false;
    missing.terminal_ms = None;
    missing.ttft_from_scheduled_ms = None;
    missing.visible_span_ms = None;
    missing.visible_text_events = None;
    missing.visible_gaps_ms.clear();
    let mut config = spec();
    config.absolute.ttft_ms = 50.0;
    config.absolute.quantile = Fraction {
        numerator: 1,
        denominator: 2,
    };
    let uncertain = analyze(&workload, &pairs, config).unwrap();
    let run = &uncertain.pairs[0].candidate;
    // The missing request may be successful with TTFT0: rank1 of [0,100]
    // passes. The observed survivor's percentile cannot establish failure.
    assert_eq!((run.arrivals, run.successes, run.missing), (2, 1, 1));
    assert_eq!(run.target_quantiles[0], None);
    assert!(matches!(run.latency_gates[0], Gate::Unresolved(_)));
    assert!(matches!(run.absolute, Gate::Unresolved(_)));
    assert_eq!(run.throughput_tps, None); // no fabricated terminal/drain time
    config.absolute.quantile = spec().absolute.quantile;
    let proved = analyze(&workload, &pairs, config).unwrap();
    // P95 is rank2 if the missing request succeeds, or rank1 otherwise;
    // both must include the observed100. No synthetic zero is materialized.
    assert_eq!(proved.pairs[0].candidate.latency_gates[0], Gate::Fail);
    assert_eq!(proved.pairs[0].candidate.target_quantiles[0], None);
    assert_eq!(proved.development_gate, Gate::Fail);
}

#[test]
fn partial_tpot_uses_known_short_output_eligibility_without_imputing_timing() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    let long = &mut pairs[0].candidate.requests[0];
    long.visible_span_ms = Some(100.0);
    long.visible_gaps_ms = vec![100.0];
    let mut config = spec();
    config.absolute.tpot_ms = 50.0;
    pairs[0].candidate.requests[1].visible_span_ms = None;
    let proved = analyze(&workload, &pairs, config).unwrap();
    assert_eq!(proved.pairs[0].candidate.tpot_unknown, 1);
    assert_eq!(proved.pairs[0].candidate.tpot.unwrap().p95, 100.0);
    assert_eq!(proved.pairs[0].candidate.target_quantiles[1], None);
    assert_eq!(proved.pairs[0].candidate.latency_gates[1], Gate::Fail);
    config.absolute.quantile = Fraction {
        numerator: 1,
        denominator: 2,
    };
    let uncertain = analyze(&workload, &pairs, config).unwrap();
    assert!(matches!(
        uncertain.pairs[0].candidate.latency_gates[1],
        Gate::Unresolved(_)
    ));
    // A valid one-token outcome is excluded, rather than added as a zero TPOT.
    let short = &mut pairs[0].candidate.requests[1];
    short.usage = Some(Usage {
        prompt: 4,
        completion: 1,
        total: 5,
    });
    short.visible_text_events = Some(1);
    short.visible_gaps_ms.clear();
    let exact = analyze(&workload, &pairs, config).unwrap();
    assert_eq!(exact.pairs[0].candidate.tpot_undefined, 1);
    assert_eq!(exact.pairs[0].candidate.tpot_unknown, 0);
    assert_eq!(exact.pairs[0].candidate.target_quantiles[1], Some(100.0));
    assert_eq!(exact.pairs[0].candidate.latency_gates[1], Gate::Fail);
}

#[test]
fn partial_visible_gaps_keep_stalls_but_unknown_or_contradictory_counts_cannot_prove_a_quantile() {
    for (events, expected) in [
        (Some(3), Gate::Fail),
        (None, Gate::Unresolved(Issue::UndefinedMetric)),
        (Some(1), Gate::Unresolved(Issue::UndefinedMetric)),
    ] {
        let (workload, mut pairs) = fixture(&[1.0, 1.0]);
        let partial = &mut pairs[0].candidate.requests[1];
        partial.visible_text_events = events;
        partial.visible_gaps_ms = vec![1000.0, f64::NAN];
        partial.visible_span_ms = None;
        partial.terminal_ms = Some(2000.0);
        let mut config = spec();
        config.absolute.visible_itl_ms = 100.0;
        let report = analyze(&workload, &pairs, config).unwrap();
        let run = &report.pairs[0].candidate;
        assert_eq!(run.visible_intervals, 2); // observed [2,1000], NaN is not a gap
        assert_eq!(run.visible_itl.unwrap().p95, 1000.0);
        assert_eq!(run.itl_unknown_requests, 1);
        assert_eq!(run.target_quantiles[2], None);
        assert_eq!(run.latency_gates[2], expected);
        assert_eq!(run.absolute, expected);
        assert!(report.primary.is_err());
    }
}

#[test]
fn complete_visible_event_count_never_completes_a_missing_gap_vector() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    let partial = &mut pairs[0].candidate.requests[1];
    partial.usage = Some(Usage {
        prompt: 4,
        completion: 3,
        total: 7,
    });
    partial.visible_text_events = Some(3);
    partial.visible_gaps_ms = vec![1000.0]; // one of two independently counted gaps missing
    partial.visible_span_ms = None;
    partial.terminal_ms = Some(2000.0);
    let mut config = spec();
    config.absolute.visible_itl_ms = 100.0;
    config.absolute.quantile = Fraction {
        numerator: 1,
        denominator: 2,
    };
    let uncertain = analyze(&workload, &pairs, config).unwrap();
    assert_eq!(uncertain.pairs[0].candidate.visible_intervals, 2);
    assert_eq!(uncertain.pairs[0].candidate.target_quantiles[2], None);
    // One exceedance among three possible intervals cannot prove median failure.
    assert!(matches!(
        uncertain.pairs[0].candidate.latency_gates[2],
        Gate::Unresolved(_)
    ));
    let complete = &mut pairs[0].candidate.requests[1];
    complete.visible_gaps_ms.push(1000.0);
    complete.visible_span_ms = Some(2000.0);
    complete.terminal_ms = Some(3000.0);
    let exact = analyze(&workload, &pairs, config).unwrap();
    let run = &exact.pairs[0].candidate;
    assert_eq!(run.itl_unknown_requests, 0);
    assert_eq!(run.visible_intervals, 3);
    assert_eq!(run.target_quantiles[2], Some(1000.0));
    assert_eq!(run.latency_gates[2], Gate::Fail);
}

#[test]
fn failed_stream_partial_vector_preserves_each_independently_valid_gap() {
    let (workload, mut pairs) = fixture(&[1.0, 1.0]);
    let failed = &mut pairs[0].candidate.requests[1];
    failed.outcome = Outcome::Failed(FailureKind::Transport);
    failed.usage = None;
    failed.finish = None;
    failed.done = false;
    failed.ttft_from_scheduled_ms = None;
    failed.visible_text_events = None;
    failed.visible_gaps_ms = vec![1000.0, f64::NAN, 0.0];
    failed.visible_span_ms = Some(f64::NAN);
    failed.terminal_ms = Some(2000.0);
    let report = analyze(&workload, &pairs, spec()).unwrap();
    let run = &report.pairs[0].candidate;
    assert_eq!(run.visible_intervals, 3); // [2,1000,0], including coalesced zero
    assert_eq!(run.visible_itl.unwrap().p95, 1000.0);
    assert_eq!(run.itl_unknown_requests, 1);
    assert_eq!(run.target_quantiles[2], None);
    assert!(matches!(run.latency_gates[2], Gate::Unresolved(_)));
    assert_eq!((run.arrivals, run.successes, run.failures), (2, 1, 1));
    assert_eq!(run.throughput_tps, Some(1.0)); // success usage2 / full2s drain
    assert_eq!(report.development_gate, Gate::Fail);
}

#[test]
fn scalar_ties_keep_nearest_rank_and_percentile_interval_endpoints() {
    let mut scalar = [4.0, 2.0, 0.0, 2.0];
    scalar.sort_unstable_by(f64::total_cmp);
    let p = percentiles(&scalar).unwrap();
    assert_eq!(p.p50, 2.0);
    assert_eq!(p.p95, 4.0);
    assert_eq!(p.p99, 4.0);
    let mut logs = [4.0_f64.ln(), 2.0_f64.ln(), 0.0, 2.0_f64.ln()];
    let ci = interval(&mut logs, 2.0_f64.ln(), spec().bootstrap.confidence).unwrap();
    near(ci.estimate, 2.0);
    near(ci.lower, 1.0);
    near(ci.upper, 4.0);
}
