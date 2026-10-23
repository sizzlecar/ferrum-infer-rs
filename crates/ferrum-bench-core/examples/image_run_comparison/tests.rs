use super::*;
use ferrum_bench_core::images::{
    sha256, ImageDataset, ImageFailureKind, ImageMetrics, ImageRequestMode, ImageRequestPhase,
    ImageRequestRecord, ImageResponseLimits, ImageSample, ValidatedImage,
};
use input::*;

fn fixture(repetitions: usize) -> (Workload, BenchInput) {
    let dataset = ImageDataset {
        schema_version: 1,
        mode: ImageRequestMode::Generate,
        samples: vec![ImageSample {
            id: "one".into(),
            prompt: "fixture".into(),
            seed: 7,
            width: 1,
            height: 1,
            num_inference_steps: Some(2),
        }],
    };
    let arm = || Arm {
        concurrency: 1,
        model: "fixture".into(),
        server_config_sha256: sha256(b"frozen"),
    };
    let workload = Workload {
        id: "chosen-workload".into(),
        dataset_sha256: dataset.fingerprint().unwrap(),
        dataset_samples: 1,
        repetitions,
        warmup_repetitions: 1,
        timeout_seconds: 60,
        latency_slo_p99_ms: 100.0,
        capture_enabled: false,
        planned_pairs: 3,
        baseline: arm(),
        candidate: arm(),
        pairs: vec![],
    };
    let mut requests = Vec::new();
    let mut rounds = Vec::new();
    for (phase, count) in [
        (ImageRequestPhase::Warmup, 1),
        (ImageRequestPhase::Measured, repetitions),
    ] {
        for repetition in 0..count {
            let record = ImageRequestRecord {
                sample_index: 0,
                sample_id: "one".into(),
                phase,
                repetition,
                dispatch_index: 0,
                http_status: Some(200),
                e2e_ms: 10.0,
                validation_ms: 1.0,
                response_bytes: 100,
                image: Some(ValidatedImage {
                    width: 1,
                    height: 1,
                    png_bytes: 70,
                    rgba8_bytes: 4,
                    png_sha256: sha256(b"png"),
                    rgba8_sha256: sha256(b"rgba"),
                }),
                failure: None,
                error: None,
            };
            let metrics = ImageMetrics::from_records(
                2.0,
                std::slice::from_ref(&record),
                (phase == ImageRequestPhase::Measured).then_some(100.0),
            );
            requests.push(record);
            rounds.push(Repetition {
                phase,
                repetition,
                metrics,
            });
        }
    }
    let measured: Vec<_> = requests
        .iter()
        .filter(|r| r.phase == ImageRequestPhase::Measured)
        .cloned()
        .collect();
    let report = BenchInput {
        schema_version: 1,
        client_version: "fixture-client".into(),
        started_unix_ms: 1,
        dataset_sha256: dataset.fingerprint().unwrap(),
        dataset,
        configuration: Configuration {
            endpoints: vec![Endpoint {
                model: "fixture".into(),
                server_config_declaration: "frozen".into(),
            }],
            concurrency: 1,
            repetitions,
            warmup_repetitions: 1,
            timeout_seconds: 60,
            latency_slo_p99_ms: Some(100.0),
            limits: ImageResponseLimits::default(),
            capture: None,
        },
        endpoints: vec![EndpointReport {
            endpoint_index: 0,
            measured: ImageMetrics::from_records(2.0 * repetitions as f64, &measured, Some(100.0)),
            repetitions: rounds,
            requests,
        }],
        reference_image: None,
        capture: None,
    };
    (workload, report)
}
fn source(path: PathBuf) -> RunSource {
    RunSource {
        report: path,
        endpoint_index: 0,
        fresh_instance_evidence: vec!["external-run.log".into()],
    }
}

#[test]
fn fieller_matches_constant_denominator_student_interval() {
    let result = fieller::analyze(&[(10.0, 19.0), (10.0, 20.0), (10.0, 21.0)]);
    assert_eq!(result.status, fieller::Status::AtLeastParity);
    let statistics = result.statistics.unwrap();
    let interval = result.interval.unwrap();
    let half = statistics.t_critical / (10.0 * 3.0_f64.sqrt());
    assert!((interval.lower - (2.0 - half)).abs() < 1e-12);
    assert!((interval.upper - (2.0 + half)).abs() < 1e-12);
    assert_eq!(statistics.independent_pairs, 3);
}

#[test]
fn pairing_covariance_and_common_units_affect_only_the_correct_quantities() {
    let pairs = [(10.0, 20.0), (11.0, 22.2), (12.0, 23.8), (13.0, 26.1)];
    let a = fieller::analyze(&pairs);
    let b = fieller::analyze(&[(10.0, 26.1), (11.0, 23.8), (12.0, 22.2), (13.0, 20.0)]);
    assert!(
        (a.statistics.as_ref().unwrap().ratio_of_means
            - b.statistics.as_ref().unwrap().ratio_of_means)
            .abs()
            < 1e-12
    );
    let a = a.interval.unwrap();
    let b = b.interval.unwrap();
    assert!(b.upper - b.lower > a.upper - a.lower);
    for scale in [1e-150, 1e150] {
        let scaled: Vec<_> = pairs.iter().map(|&(x, y)| (x * scale, y * scale)).collect();
        let c = fieller::analyze(&scaled).interval.unwrap();
        assert!((a.lower - c.lower).abs() < 1e-11);
        assert!((a.upper - c.upper).abs() < 1e-11);
    }
}

#[test]
fn unbounded_invalid_small_samples_and_below_target_are_distinct() {
    let unbounded = fieller::analyze(&[(0.001, 1.0), (0.001, 1.0), (10.0, 1.0)]);
    assert_eq!(unbounded.status, fieller::Status::Insufficient);
    assert!(unbounded.interval.is_none());
    assert!(unbounded.statistics.is_some());
    for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(fieller::analyze(&[(1.0, 1.0), (1.0, 1.0), (bad, 1.0)])
            .interval
            .is_none());
    }
    assert!(fieller::analyze(&[(1.0, 1.0); 2]).interval.is_none());
    assert!(fieller::analyze(&[(1.0, 1.0); 31]).interval.is_none());
    assert_eq!(
        fieller::analyze(&[(10.0, 8.0), (10.0, 8.01), (10.0, 7.99)]).status,
        fieller::Status::BelowTarget
    );
    assert_eq!(
        fieller::analyze(&[(10.0, 9.0), (10.0, 10.0), (10.0, 11.0)]).status,
        fieller::Status::Insufficient
    );
}

#[test]
fn typed_reader_excludes_warmup_and_checks_complete_records_and_summary() {
    let (w, mut r) = fixture(2);
    r.endpoints[0].requests[0].e2e_ms = 999.0;
    r.endpoints[0].repetitions[0].metrics =
        ImageMetrics::from_records(2.0, &r.endpoints[0].requests[..1], None);
    let o = validate_report(r, 0, &w, &w.baseline).unwrap();
    assert!(o.eligible);
    assert_eq!(o.measured.attempted, 2);
    assert_eq!(o.measured.e2e_p99_ms, Some(10.0));
    let (w, mut r) = fixture(2);
    r.endpoints[0].requests[2] = r.endpoints[0].requests[1].clone();
    assert!(validate_report(r, 0, &w, &w.baseline).is_err());
    let (w, mut r) = fixture(2);
    r.endpoints[0].measured.images_per_second = Some(123.0);
    assert!(validate_report(r, 0, &w, &w.baseline).is_err());
}

#[test]
fn failures_remain_observations_and_block_qualification() {
    let (w, mut r) = fixture(2);
    let ep = &mut r.endpoints[0];
    ep.requests[1].image = None;
    ep.requests[1].failure = Some(ImageFailureKind::Timeout);
    ep.requests[1].http_status = None;
    ep.requests[1].error = Some("timeout".into());
    ep.repetitions[1].metrics = ImageMetrics::from_records(2.0, &ep.requests[1..2], Some(100.0));
    ep.measured = ImageMetrics::from_records(4.0, &ep.requests[1..], Some(100.0));
    let o = validate_report(r, 0, &w, &w.baseline).unwrap();
    assert!(!o.eligible);
    assert_eq!(o.measured.errors, 1);
    assert_eq!(o.measured.images_per_second, Some(0.25));
}

#[test]
fn edit_identity_is_bound_to_the_actual_reference_hash() {
    for tampered in [false, true] {
        let (mut w, mut r) = fixture(1);
        let reference = r.endpoints[0].requests[0].image.clone().unwrap();
        r.schema_version = 2;
        r.dataset.schema_version = 2;
        r.dataset.mode = ImageRequestMode::Edit {
            image: ferrum_bench_core::images::ImageReference {
                path: "reference.png".into(),
                encoded_sha256: Some(reference.png_sha256.clone()),
            },
        };
        r.dataset_sha256 = r.dataset.fingerprint().unwrap();
        w.dataset_sha256 = r.dataset_sha256.clone();
        r.reference_image = Some(reference);
        if tampered {
            r.reference_image.as_mut().unwrap().png_sha256 = sha256(b"different reference");
        }
        assert_eq!(validate_report(r, 0, &w, &w.baseline).is_err(), tampered);
    }
}

#[test]
fn duplicating_an_original_report_never_creates_independent_runs() {
    let directory = tempfile::tempdir().unwrap();
    let (mut w, r) = fixture(1);
    let path = directory.path().join("original.json");
    std::fs::write(&path, serde_json::to_vec(&r).unwrap()).unwrap();
    for index in 0..w.planned_pairs {
        w.pairs.push(Pair {
            id: format!("claimed-independent-{index}"),
            order: if index % 2 == 0 {
                Order::BaselineFirst
            } else {
                Order::CandidateFirst
            },
            baseline: source(path.clone()),
            candidate: source(path.clone()),
        });
    }
    let plan = Plan {
        schema_version: 1,
        workloads: vec![w],
    };
    plan.validate().unwrap();
    let report = analyze(plan, "digest".into(), directory.path(), 1024 * 1024);
    assert_eq!(
        report.workloads[0].analysis.status,
        fieller::Status::Insufficient
    );
    assert_eq!(report.workloads[0].complete_pairs, 0);
    let reused = &report.workloads[0].pairs[0].candidate;
    assert!(reused.problem.as_ref().unwrap().contains("reused"));
    assert!(reused.observation.is_some());
}

#[test]
fn missing_pair_is_not_a_complete_case_analysis_and_all_paths_are_retained() {
    let directory = tempfile::tempdir().unwrap();
    let (mut w, _) = fixture(1);
    w.pairs.push(Pair {
        id: "declared-pair".into(),
        order: Order::BaselineFirst,
        baseline: source("missing-baseline.json".into()),
        candidate: source("missing-candidate.json".into()),
    });
    let plan = Plan {
        schema_version: 1,
        workloads: vec![w],
    };
    plan.validate().unwrap();
    let report = analyze(plan, "plan-digest".into(), directory.path(), 1024);
    let w = &report.workloads[0];
    assert_eq!(w.analysis.status, fieller::Status::Insufficient);
    assert_eq!(w.complete_pairs, 0);
    assert_eq!(w.pairs.len(), 1);
    assert!(w.pairs[0].baseline.problem.is_some());
    assert!(w.pairs[0].candidate.problem.is_some());
}

#[test]
fn original_reports_are_independent_units_even_with_more_in_process_repetitions() {
    for repetitions in [1, 5] {
        let directory = tempfile::tempdir().unwrap();
        let (mut w, _) = fixture(repetitions);
        for pair in 0..w.planned_pairs {
            let mut sources = Vec::new();
            for arm in 0..2 {
                let (_, mut r) = fixture(repetitions);
                r.started_unix_ms = (pair * 2 + arm + 1) as u128;
                // Vary measured whole-round time, independently of warmup.
                let elapsed = if arm == 0 {
                    2.0
                } else {
                    1.0 + pair as f64 * 0.01
                };
                let e = &mut r.endpoints[0];
                for round in e
                    .repetitions
                    .iter_mut()
                    .filter(|r| r.phase == ImageRequestPhase::Measured)
                {
                    let records: Vec<_> = e
                        .requests
                        .iter()
                        .filter(|r| r.phase == round.phase && r.repetition == round.repetition)
                        .cloned()
                        .collect();
                    round.metrics = ImageMetrics::from_records(elapsed, &records, Some(100.0));
                }
                let records: Vec<_> = e
                    .requests
                    .iter()
                    .filter(|r| r.phase == ImageRequestPhase::Measured)
                    .cloned()
                    .collect();
                e.measured =
                    ImageMetrics::from_records(elapsed * repetitions as f64, &records, Some(100.0));
                let path = directory.path().join(format!("pair{pair}-arm{arm}.json"));
                std::fs::write(&path, serde_json::to_vec(&r).unwrap()).unwrap();
                sources.push(source(path));
            }
            w.pairs.push(Pair {
                id: format!("pair{pair}"),
                order: if pair % 2 == 0 {
                    Order::BaselineFirst
                } else {
                    Order::CandidateFirst
                },
                baseline: sources.remove(0),
                candidate: sources.remove(0),
            });
        }
        let plan = Plan {
            schema_version: 1,
            workloads: vec![w],
        };
        plan.validate().unwrap();
        let report = analyze(plan, "digest".into(), directory.path(), 1024 * 1024);
        let a = &report.workloads[0].analysis;
        assert_eq!(a.statistics.as_ref().unwrap().independent_pairs, 3);
        assert_eq!(a.status, fieller::Status::AtLeastParity);
    }
}
