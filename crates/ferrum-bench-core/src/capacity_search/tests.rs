use super::*;
#[path = "tests/server_queue.rs"]
mod server_queue;
#[path = "tests/session.rs"]
mod session;
use crate::{
    dataset::{ShareGptSample, ShareGptSelection},
    slo::{
        AdmissionEvidence, RawOutputCount, RawOutputCountSource, RequestOutcome,
        RequestSloEvidence, SloEvaluationConfig, SloStatus, TpotBoundary, VisibleTextEvidence,
    },
    slo_comparison::{
        artifact::SidecarArrival, FixedServerCapacity, FrozenServerIdentity,
        SharedExecutionIdentity,
    },
    BenchmarkPhase, BenchmarkRequestCorrelation, BenchmarkRequestRecord,
    BenchmarkRequestTimingEvidence, ItlEvidenceSource, QualityIssueCounts, RequestItlEvidence,
};
use ferrum_types::SloAttainmentTargets;
use sha2::{Digest, Sha256};

fn contract() -> CapacityContract {
    let hash = "a".repeat(64);
    let samples: Vec<_> = [3, 4]
        .into_iter()
        .enumerate()
        .map(|(i, n)| ShareGptSample {
            source_record_index: i as u64,
            original_id: None,
            phase: BenchmarkPhase::Measured,
            request_index: i as u32,
            prompt_sha256: format!("{:064x}", i + 1),
            assistant_sha256: hash.clone(),
            input_tokens: (i + 4) as u32,
            reference_output_tokens: n,
            requested_output_tokens: n,
        })
        .collect();
    let selection_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&samples).unwrap())
    );
    CapacityContract {
        schema_version: 1,
        frozen_unix_ns: 1,
        identity: CapacityIdentity {
            shared: SharedExecutionIdentity {
                hardware_fingerprint_sha256: hash.clone(),
                hardware_label: "test device".into(),
                model_content_sha256: hash.clone(),
                weight_precision: "test".into(),
                kv_precision: "fp16".into(),
                tokenizer_sha256: hash.clone(),
                chat_template_sha256: hash.clone(),
                client_binary_sha256: hash.clone(),
                client_slo_config_sha256: hash.clone(),
            },
            server: FrozenServerIdentity {
                implementation: "ferrum".into(),
                backend: "test".into(),
                request_model_alias: "test".into(),
                binary_sha256: hash.clone(),
                effective_configuration_sha256: hash.clone(),
                numerical_policy: "test".into(),
                intentional_differences: vec![],
            },
            capacity: FixedServerCapacity {
                slots: 32,
                context_tokens_per_request: 2048,
                batch_tokens: 2048,
            },
            ordered_workload_sha256: selection_sha256.clone(),
            dataset_source_sha256: hash,
        },
        rates_rps: vec![10.0, 20.0, 30.0, 40.0, 50.0],
        coarse_indices: vec![0, 2, 4],
        aa_indices: vec![0],
        repetitions: CapacityRepetitions {
            aa_pairs: 1,
            coarse: 1,
            neighborhood: 1,
            confirmation: 2,
        },
        window: CapacityWindow {
            send_seconds: 1.0,
            observe_from_seconds: 0.0,
            maximum_drain_seconds: 1.0,
            maximum_request_start_lag_ms: 5.0,
            maximum_client_dispatch_backlog: 2,
            maximum_queue_sample_gap_seconds: 0.25,
            maximum_unfinished_requests_slope_per_second: 0.0,
            maximum_oldest_age_slope_ms_per_second: 0.0,
        },
        queue_observation_source: QueueObservationSource::ServerQueue,
        sampling: crate::env::HttpRequestSampling {
            temperature: 0.0,
            top_k: None,
            top_p: Some(1.0),
            repetition_penalty: Some(1.0),
            seed: Some(42),
        },
        enable_thinking: Some(false),
        http_connection_mode: "fresh".into(),
        slo: SloEvaluationConfig {
            ttft_ms: 100.0,
            tpot_ms: 20.0,
            visible_itl_ms: 20.0,
            attainment: SloAttainmentTargets {
                min_accepted_joint_attainment: 0.99,
                min_offered_joint_attainment: Some(0.99),
                max_reject_rate: 0.0,
                max_error_rate: 0.0,
                ..Default::default()
            },
            tpot_boundary: TpotBoundary::LastVisibleOutput,
        },
        seed: 42,
        maximum_requests_per_run: 1024,
        maximum_planned_runs: 100,
        maximum_queue_samples_per_run: 1000,
        workload: ShareGptSelection {
            repeat_index: 0,
            rng_seed: 42,
            selection_sha256,
            samples,
        },
    }
}

fn request(tokens: u32) -> RequestSloEvidence {
    let gaps = vec![1.0; tokens as usize - 1];
    RequestSloEvidence {
        outcome: RequestOutcome::Completed,
        admission: AdmissionEvidence::Accepted,
        first_visible_ms: Some(1.0),
        last_visible_ms: Some(tokens as f64),
        terminal_ms: Some(tokens as f64 + 1.0),
        usage_output_tokens: Some(tokens),
        raw_output_count: RawOutputCount {
            count: tokens,
            source: RawOutputCountSource::Usage,
        },
        visible_text: Some(VisibleTextEvidence {
            output_events: tokens,
            gaps_ms: gaps,
            transport_coalesced_output_chunks: 0,
        }),
        strict_token_evidence: RequestItlEvidence::sse(true, tokens, Some(tokens), tokens - 1, 0),
    }
}

fn evidence(
    search: &CapacitySearch,
    planned: &PlannedCapacityRun,
    ordinal: u64,
) -> CapacityRunEvidence {
    CapacityRunEvidence {
        session: None,
        contract_sha256: planned.contract_sha256.clone(),
        key: planned.key.clone(),
        identity: search.contract().identity.clone(),
        run_started_unix_ns: (ordinal + 1) * 10_000_000_000,
        run_ended_unix_ns: (ordinal + 1) * 10_000_000_000 + 2_000_000_000,
        independent_run_id: format!("server-{ordinal}"),
        benchmark_run_id: format!("run-{ordinal}"),
        warmup: CapacityWarmupEvidence {
            acquisition: Default::default(),
            expected: 0,
            completed: 0,
            errored: 0,
            quality: QualityIssueCounts::default(),
        },
        send_window_seconds: 1.0,
        measured_duration_seconds: 1.25,
        arrivals: planned
            .scheduled_arrival_ms
            .iter()
            .enumerate()
            .map(|(index, &t)| SidecarArrival {
                scheduled_arrival_ms: Some(t),
                dispatched_ms: Some(t),
                request_started_ms: Some(t),
                client_dispatch_backlog: Some(
                    planned
                        .scheduled_arrival_ms
                        .partition_point(|s| *s <= t)
                        .saturating_sub(index) as u64,
                ),
            })
            .collect(),
        requests: planned
            .output_token_budgets
            .iter()
            .map(|&n| request(n))
            .collect(),
        request_records: planned
            .workload_sample_indices
            .iter()
            .enumerate()
            .map(|(index, &sample_index)| {
                let sample = &search.contract().workload.samples[sample_index];
                let req = request(sample.requested_output_tokens);
                CapacityRequestRecord {
                    workload_sample_index: sample_index,
                    dispatched_prompt_sha256: sample.prompt_sha256.clone(),
                    input_tokens: sample.input_tokens,
                    server_input_tokens: Some(sample.input_tokens + 2),
                    record: BenchmarkRequestRecord {
                        correlation: BenchmarkRequestCorrelation::new(
                            format!("run-{ordinal}"),
                            planned.cell_id.clone(),
                            planned.key.repetition,
                            BenchmarkPhase::Measured,
                            index as u32,
                        )
                        .unwrap(),
                        server_request_id: Some(format!("request-{ordinal}-{index}")),
                        timing: Some(BenchmarkRequestTimingEvidence {
                            success: true,
                            reported_ttft_ms: req.first_visible_ms.unwrap(),
                            reported_e2e_ms: req.terminal_ms.unwrap(),
                            event_source: ItlEvidenceSource::SseDeltaEvents,
                            observed_first_output: Some(true),
                            raw_event_gaps_ms: req.visible_text.unwrap().gaps_ms,
                        }),
                    },
                }
            })
            .collect(),
        quality: vec![QualityIssueCounts::default(); planned.output_token_budgets.len()],
        queue_capture_complete: true,
        server_queue_attempts: Vec::new(),
        queue_observation_source: QueueObservationSource::ServerQueue,
        queue: (0..=5)
            .map(|i| ServiceQueueSample {
                at_seconds: i as f64 * 0.25,
                waiting_requests: 0,
                active_requests: 0,
                oldest_request_age_ms: 0.0,
            })
            .collect(),
    }
}

fn next(search: &CapacitySearch) -> PlannedCapacityRun {
    let SearchProgress::Awaiting { runs, .. } = search.progress() else {
        panic!("expected pending phase")
    };
    search.planned_run(&runs[0]).unwrap()
}

#[test]
fn explicit_contract_rejects_absent_or_invalid_rates_windows_and_repetitions() {
    let good = contract();
    assert!(good.validate().is_ok());
    let mut bad = good.clone();
    bad.rates_rps[1] = bad.rates_rps[0];
    assert!(bad.validate().is_err());
    let mut bad = good.clone();
    bad.rates_rps[1] = f64::NAN;
    assert!(bad.validate().is_err());
    let mut bad = good.clone();
    bad.repetitions.confirmation = 0;
    assert!(bad.validate().is_err());
    let mut bad = good.clone();
    bad.window.observe_from_seconds = bad.window.send_seconds;
    assert!(bad.validate().is_err());
    let mut bad = good.clone();
    bad.coarse_indices.pop();
    assert!(bad.validate().is_err());
    let mut bad = good.clone();
    bad.maximum_planned_runs = 1;
    assert!(bad.validate().is_err());
    let mut wire = serde_json::to_value(&good).unwrap();
    wire.as_object_mut().unwrap().remove("window");
    assert!(serde_json::from_value::<CapacityContract>(wire).is_err());
}

#[test]
fn aa_reuses_offered_schedule_then_independent_phases_get_distinct_arrivals() {
    let mut search = CapacitySearch::new(contract()).unwrap();
    let first = next(&search);
    assert_eq!(first.key.phase, CapacityPhase::Aa);
    let first_evidence = evidence(&search, &first, 0);
    assert_eq!(
        search.record(first_evidence).unwrap().status,
        SloStatus::Pass
    );
    let second = next(&search);
    assert_eq!(first.scheduled_arrival_ms, second.scheduled_arrival_ms);
    assert_ne!(first.key.replica, second.key.replica);
    let second_evidence = evidence(&search, &second, 1);
    search.record(second_evidence).unwrap();
    let coarse = next(&search);
    assert_eq!(coarse.key.phase, CapacityPhase::Coarse);
    assert_ne!(coarse.scheduled_arrival_ms, first.scheduled_arrival_ms);
    assert_eq!(search.report().aa_noise.len(), 1);
}

#[test]
fn full_grid_keeps_nonmonotonic_feasible_islands_and_independent_failures() {
    let mut search = CapacitySearch::new(contract()).unwrap();
    let mut ordinal = 0;
    while search.progress() != SearchProgress::Complete {
        let planned = next(&search);
        let mut raw = evidence(&search, &planned, ordinal);
        // Both coarse endpoints of index 3 fail. The fine point nevertheless
        // passes and must be measured; an early failure cannot stop the sweep.
        let fail = planned.key.phase != CapacityPhase::Aa
            && (planned.key.rate_index == 2 || planned.key.rate_index == 4);
        if fail {
            for row in &mut raw.requests {
                row.first_visible_ms = Some(200.0);
                row.last_visible_ms = Some(203.0);
                row.terminal_ms = Some(204.0);
            }
        }
        search.record(raw).unwrap();
        ordinal += 1;
    }
    let report = search.report();
    assert!(report.complete);
    assert!(report.unmeasured_rate_indices.is_empty());
    assert_eq!(report.maximum_confirmed_tested_rate_rps, Some(40.0));
    assert!(report.independently_confirmed_rate_indices.contains(&3));
    assert!(report
        .runs
        .iter()
        .any(|r| r.key.phase == CapacityPhase::Confirmation && r.key.rate_index == 4));
    assert!(report
        .runs
        .iter()
        .filter(|r| r.key.rate_index == 2 && r.key.phase != CapacityPhase::Aa)
        .all(|r| r.status == SloStatus::Fail));
}

#[test]
fn missing_or_forged_evidence_never_becomes_capacity_success() {
    let search = CapacitySearch::new(contract()).unwrap();
    let planned = next(&search);
    let raw = evidence(&search, &planned, 0);
    assert_eq!(
        evaluate_run(search.contract(), &planned, &raw)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut bad = raw.clone();
    bad.arrivals.pop();
    assert_ne!(
        evaluate_run(search.contract(), &planned, &bad)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut bad = raw.clone();
    bad.queue.clear();
    assert_eq!(
        evaluate_run(search.contract(), &planned, &bad)
            .unwrap()
            .status,
        SloStatus::Unknown
    );
    let mut bad = raw.clone();
    bad.quality.pop();
    assert_ne!(
        evaluate_run(search.contract(), &planned, &bad)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut bad = raw.clone();
    bad.requests[0].outcome = RequestOutcome::Pending;
    bad.requests[0].terminal_ms = None;
    assert_eq!(
        evaluate_run(search.contract(), &planned, &bad)
            .unwrap()
            .status,
        SloStatus::Fail
    );
    let mut bad = raw.clone();
    bad.requests[0] = request(2);
    assert_eq!(
        evaluate_run(search.contract(), &planned, &bad)
            .unwrap()
            .status,
        SloStatus::Fail
    );
    let mut bad = raw.clone();
    bad.quality[0].missing_done = 1;
    assert_eq!(
        evaluate_run(search.contract(), &planned, &bad)
            .unwrap()
            .status,
        SloStatus::Fail
    );
    let mut bad = raw.clone();
    bad.requests[0].usage_output_tokens = None;
    bad.requests[0].raw_output_count.source = RawOutputCountSource::Unknown;
    assert_ne!(
        evaluate_run(search.contract(), &planned, &bad)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut forged = planned.clone();
    forged.output_token_budgets[0] = 2;
    let mut bad = raw.clone();
    bad.requests[0] = request(2);
    assert!(evaluate_run(search.contract(), &forged, &bad).is_err());
    let mut bad = raw.clone();
    bad.identity.server.binary_sha256 = "b".repeat(64);
    assert!(evaluate_run(search.contract(), &planned, &bad).is_err());
}

#[test]
fn client_saturation_and_service_accumulation_are_separate_failures() {
    let search = CapacitySearch::new(contract()).unwrap();
    let planned = next(&search);
    let raw = evidence(&search, &planned, 0);
    let mut delayed = raw.clone();
    delayed.arrivals[0].request_started_ms =
        delayed.arrivals[0].request_started_ms.map(|t| t + 10.0);
    let report = evaluate_run(search.contract(), &planned, &delayed).unwrap();
    assert_eq!(report.status, SloStatus::Fail);
    assert!(!report.arrival_schedule_delivered);
    let mut growing = raw.clone();
    for (index, s) in growing.queue.iter_mut().take(5).enumerate() {
        s.waiting_requests = index as u64;
        s.oldest_request_age_ms = index as f64 * 10.0;
    }
    let report = evaluate_run(search.contract(), &planned, &growing).unwrap();
    assert_eq!(report.status, SloStatus::Fail);
    assert!(report.arrival_schedule_delivered);
    assert!(report.unfinished_requests_slope_per_second.unwrap() > 0.0);
    let mut undrained = raw.clone();
    undrained.queue.last_mut().unwrap().active_requests = 1;
    assert_eq!(
        evaluate_run(search.contract(), &planned, &undrained)
            .unwrap()
            .status,
        SloStatus::Fail
    );
    let mut hidden_backlog = raw.clone();
    hidden_backlog.arrivals[0].client_dispatch_backlog = Some(0);
    assert_ne!(
        evaluate_run(search.contract(), &planned, &hidden_backlog)
            .unwrap()
            .status,
        SloStatus::Pass
    );
}

#[test]
fn repetition_order_lifecycle_and_acquisition_bound_are_enforced() {
    let mut search = CapacitySearch::new(contract()).unwrap();
    let first = next(&search);
    let raw = evidence(&search, &first, 0);
    let mut future = raw.clone();
    future.key.phase = CapacityPhase::Coarse;
    assert!(search.record(future).is_err());
    search.record(raw.clone()).unwrap();
    assert!(search.record(raw.clone()).is_err());
    let second = next(&search);
    let mut reused = evidence(&search, &second, 1);
    reused.independent_run_id = raw.independent_run_id;
    assert!(search.record(reused).is_err());
    let mut overlapping = evidence(&search, &second, 1);
    overlapping.run_started_unix_ns = raw.run_started_unix_ns;
    assert!(search.record(overlapping).is_err());
    let mut bounded = contract();
    bounded.maximum_requests_per_run = 1;
    let bounded = CapacitySearch::new(bounded).unwrap();
    let SearchProgress::Awaiting { runs, .. } = bounded.progress() else {
        unreachable!()
    };
    assert!(bounded.planned_run(&runs[0]).is_err());
}

#[test]
fn incomplete_aa_cannot_unlock_a_capacity_claim() {
    let mut search = CapacitySearch::new(contract()).unwrap();
    let mut ordinal = 0;
    while search.progress() != SearchProgress::Complete {
        let planned = next(&search);
        let mut raw = evidence(&search, &planned, ordinal);
        if ordinal == 0 {
            raw.queue.clear();
        }
        search.record(raw).unwrap();
        ordinal += 1;
    }
    let report = search.report();
    assert!(report.complete);
    assert!(report.aa_noise.is_empty());
    assert_eq!(report.maximum_confirmed_tested_rate_rps, None);
}

#[test]
fn explicit_acquisition_limit_counts_only_the_runs_that_will_be_requested() {
    let mut config = contract();
    // 2 A/A members + 3 coarse + 2 fine + 5 rates * 2 confirmations.
    config.maximum_planned_runs = 17;
    assert!(CapacitySearch::new(config.clone()).is_ok());
    config.maximum_planned_runs = 16;
    assert!(CapacitySearch::new(config).is_err());
    let mut config = contract();
    config.window.send_seconds = f64::MAX;
    assert!(config.validate().is_err());
}

#[test]
fn a_measurement_window_cannot_exceed_the_recorded_server_lifecycle() {
    let search = CapacitySearch::new(contract()).unwrap();
    let planned = next(&search);
    let mut raw = evidence(&search, &planned, 0);
    raw.run_ended_unix_ns = raw.run_started_unix_ns + 1_000_000;
    assert_ne!(
        evaluate_run(search.contract(), &planned, &raw)
            .unwrap()
            .status,
        SloStatus::Pass
    );
}

#[test]
fn frozen_prompts_request_correlation_and_warmup_cannot_be_substituted() {
    let search = CapacitySearch::new(contract()).unwrap();
    let planned = next(&search);
    let raw = evidence(&search, &planned, 0);
    let mut wrong_prompt = raw.clone();
    wrong_prompt.request_records[0].dispatched_prompt_sha256 = "b".repeat(64);
    assert_ne!(
        evaluate_run(search.contract(), &planned, &wrong_prompt)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut shorter = raw.clone();
    shorter.request_records[0].input_tokens = 1;
    assert_ne!(
        evaluate_run(search.contract(), &planned, &shorter)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut reordered = raw.clone();
    reordered.request_records.swap(0, 1);
    assert_ne!(
        evaluate_run(search.contract(), &planned, &reordered)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut wrong_run = raw.clone();
    wrong_run.request_records[0].record.correlation.cell_id = "another-cell".into();
    assert_ne!(
        evaluate_run(search.contract(), &planned, &wrong_run)
            .unwrap()
            .status,
        SloStatus::Pass
    );
    let mut failed_warmup = raw.clone();
    failed_warmup.warmup.errored = 1;
    assert_eq!(
        evaluate_run(search.contract(), &planned, &failed_warmup)
            .unwrap()
            .status,
        SloStatus::Fail
    );
    let mut changed_contract = contract();
    changed_contract.workload.samples[0].input_tokens = 1;
    assert!(CapacitySearch::new(changed_contract).is_err());
}

#[test]
fn low_last_visible_tpot_cannot_contradict_the_original_visible_gaps() {
    let mut config = contract();
    config.slo.visible_itl_ms = 50.0;
    config.slo.tpot_ms = 15.0;
    let search = CapacitySearch::new(config).unwrap();
    let planned = next(&search);
    let mut raw = evidence(&search, &planned, 0);
    for (request, record) in raw.requests.iter_mut().zip(&mut raw.request_records) {
        let gaps = vec![40.0; request.usage_output_tokens.unwrap() as usize - 1];
        request.last_visible_ms = request.first_visible_ms;
        request.terminal_ms = Some(150.0);
        request.visible_text.as_mut().unwrap().gaps_ms = gaps.clone();
        let timing = record.record.timing.as_mut().unwrap();
        timing.reported_e2e_ms = 150.0;
        timing.raw_event_gaps_ms = gaps;
    }
    // The general SLO evaluator alone accepts these internally inconsistent
    // timestamps; capacity must also verify the original event span.
    assert_eq!(
        crate::slo::evaluate_slo(
            &search.contract().slo,
            &raw.requests,
            raw.measured_duration_seconds
        )
        .unwrap()
        .latency_and_outcome_status,
        SloStatus::Pass
    );
    let report = evaluate_run(search.contract(), &planned, &raw).unwrap();
    assert_eq!(report.status, SloStatus::Unknown);
    assert!(!report.workload_completed);
}

#[test]
fn client_queue_requires_real_lifecycle_and_complete_capture() {
    let mut config = contract();
    config.queue_observation_source = QueueObservationSource::ClientScheduledLifecycle;
    let search = CapacitySearch::new(config).unwrap();
    let planned = next(&search);
    let mut raw = evidence(&search, &planned, 0);
    raw.queue_observation_source = QueueObservationSource::ClientScheduledLifecycle;
    raw.queue.push(ServiceQueueSample {
        at_seconds: (planned.scheduled_arrival_ms[0] + 0.5) / 1000.0,
        waiting_requests: 0,
        active_requests: 0,
        oldest_request_age_ms: 0.0,
    });
    raw.queue
        .sort_by(|a, b| a.at_seconds.total_cmp(&b.at_seconds));
    let report = evaluate_run(search.contract(), &planned, &raw).unwrap();
    assert_ne!(report.status, SloStatus::Pass);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.contains("client lifecycle queue")));

    let search = CapacitySearch::new(contract()).unwrap();
    let planned = next(&search);
    let mut raw = evidence(&search, &planned, 0);
    raw.queue_capture_complete = false;
    let report = evaluate_run(search.contract(), &planned, &raw).unwrap();
    assert_ne!(report.status, SloStatus::Pass);
    assert!(!report.evidence_complete);
}
