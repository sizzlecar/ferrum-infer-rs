use super::*;
use ferrum_bench_core::{BenchmarkPhase, BenchmarkRequestCorrelation, BenchmarkRequestRecord};

fn workload() -> Workload {
    Workload {
        input_tokens: 32,
        output_tokens: 8,
        measured_requests: 2,
        warmup_requests: 1,
        repeats: 3,
        seed: 7,
        max_model_len: 128,
        concurrency: 2,
        max_num_batched_tokens: Some(64),
    }
}
fn limits() -> Limits {
    Limits {
        ttft_max_relative_increase: 0.1,
        tpot_max_relative_increase: 0.1,
    }
}
fn report(run_id: &str, reverse: bool) -> BenchReport {
    let mut report = fixture_report([1.0; 3]);
    report.concurrency = Some(2);
    report.benchmark_run_id = Some(run_id.into());
    report.cell_id = Some("concurrent".into());
    report.request_records = Some(
        (0..3)
            .map(|repeat| {
                let mut records: Vec<_> = (0..2)
                    .map(|request| BenchmarkRequestRecord {
                        correlation: BenchmarkRequestCorrelation::new(
                            run_id.into(),
                            "concurrent".into(),
                            repeat,
                            BenchmarkPhase::Measured,
                            request,
                        )
                        .unwrap(),
                        server_request_id: None,
                    })
                    .collect();
                let lengths =
                    &mut report.server_input_tokens_per_request.as_mut().unwrap()[repeat as usize];
                *lengths = vec![Some(35), Some(37)];
                if reverse {
                    records.reverse();
                    lengths.reverse();
                }
                records
            })
            .collect(),
    );
    report
}

#[test]
fn concurrency_compares_same_requests_even_when_they_complete_in_a_different_order() {
    let baseline = report("baseline", false);
    let candidate = report("candidate", true);
    assert_ne!(
        baseline.server_input_tokens_per_request,
        candidate.server_input_tokens_per_request
    );
    assert_eq!(
        compare(&baseline, &baseline, &candidate, &workload(), &limits())
            .unwrap()
            .status,
        ComparisonStatus::Passed
    );
}

#[test]
fn concurrent_comparison_rejects_missing_duplicate_wrong_repeat_and_wrong_input_bindings() {
    let baseline = report("baseline", false);
    let changes: Vec<Box<dyn Fn(&mut BenchReport)>> = vec![
        Box::new(|r| r.request_records = None),
        Box::new(|r| {
            r.request_records
                .as_mut()
                .unwrap()
                .pop()
                .map(|_| ())
                .unwrap()
        }),
        Box::new(|r| {
            r.request_records.as_mut().unwrap()[0]
                .pop()
                .map(|_| ())
                .unwrap()
        }),
        Box::new(|r| {
            r.request_records.as_mut().unwrap()[0][0]
                .correlation
                .request_index = 0
        }),
        Box::new(|r| {
            r.request_records.as_mut().unwrap()[0][0]
                .correlation
                .request_index = 2
        }),
        Box::new(|r| {
            r.request_records.as_mut().unwrap()[0][0]
                .correlation
                .repeat_index = 1
        }),
        Box::new(|r| {
            r.request_records.as_mut().unwrap()[0][0].correlation.phase = BenchmarkPhase::Warmup
        }),
        Box::new(|r| {
            r.request_records.as_mut().unwrap()[0][0]
                .correlation
                .benchmark_run_id = "different-run".into()
        }),
        Box::new(|r| {
            r.request_records.as_mut().unwrap()[0][0]
                .correlation
                .cell_id = "different-cell".into()
        }),
        Box::new(|r| r.server_input_tokens_per_request.as_mut().unwrap()[0].reverse()),
        Box::new(|r| r.concurrency = Some(1)),
    ];
    for (index, change) in changes.into_iter().enumerate() {
        let mut candidate = report("candidate", true);
        change(&mut candidate);
        assert!(
            compare(&baseline, &baseline, &candidate, &workload(), &limits()).is_err(),
            "accepted mutation {index}"
        );
    }
}

#[test]
fn concurrency_and_batch_contract_rejects_impossible_workloads() {
    let mut w = workload();
    w.concurrency = 0;
    assert!(w.validate().is_err());
    w.concurrency = w.measured_requests + 1;
    assert!(w.validate().is_err());
    w.concurrency = 2;
    w.max_num_batched_tokens = Some(0);
    assert!(w.validate().is_err());
}
