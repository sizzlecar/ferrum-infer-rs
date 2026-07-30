mod vnext_event_contract;

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use vnext_event_contract::*;

const SAMPLE_COUNT: usize = 30;
const WARMUP_COUNT: usize = 5;
const JOURNAL_REPETITIONS: usize = 160;
const PROVIDER_REPETITIONS: usize = 250_000;
const BENCHMARK_PREFIX_EVENTS: usize = 2;

#[derive(Default)]
struct CountingExecutionEventSink {
    records: AtomicU64,
}

impl ExecutionEventSink for CountingExecutionEventSink {
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::All
    }

    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        true
    }

    fn record(&self, permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        black_box(permit.event().kind());
        self.records.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

struct JournalFixture {
    journal: Vec<ExecutionEvent>,
    topology: TrustedExecutionTopology,
    active: TrustedActiveSequenceBinding,
    completed: TrustedCompletedSequenceBinding,
    submissions: Vec<SubmittedOperationReceipt>,
    completions: Vec<OperationCompletionReceipt>,
}

fn journal_fixture() -> JournalFixture {
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("g01b-overhead", &operation_registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    let resolved = resolved_model_plan(&plan, "g01b-overhead", &operation_registry);
    let ProvisionedRuntimePool {
        resources,
        runtime,
        evidence: _,
        journal: _,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "g01b-overhead");
    let SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    } = execute_sequence(
        &resources,
        &runtime,
        &resolved,
        &operation_registry,
        "run.g01b-overhead",
        "request.g01b-overhead",
        4,
    );
    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 4);
    JournalFixture {
        journal,
        topology,
        active,
        completed,
        submissions,
        completions,
    }
}

fn baseline_elapsed(fixture: &JournalFixture) -> u128 {
    let started = Instant::now();
    for _ in 0..JOURNAL_REPETITIONS {
        let mut cursor = ExecutionEventCursor::new(
            fixture.active.run_id().clone(),
            fixture.active.request_id().clone(),
        );
        for event in fixture.journal.iter().take(BENCHMARK_PREFIX_EVENTS) {
            let event = black_box(event.clone());
            cursor
                .observe_against(
                    &event,
                    &event_context(
                        &event,
                        &fixture.topology,
                        &fixture.active,
                        &fixture.completed,
                        &fixture.submissions,
                        &fixture.completions,
                    ),
                )
                .unwrap();
        }
        black_box(cursor.last_sequence());
    }
    started.elapsed().as_nanos()
}

fn disabled_elapsed(fixture: &JournalFixture) -> u128 {
    let sink = DisabledExecutionEventSink;
    let started = Instant::now();
    for _ in 0..JOURNAL_REPETITIONS {
        let mut emitter = ExecutionEventEmitter::new(
            &sink,
            fixture.active.run_id().clone(),
            fixture.active.request_id().clone(),
        );
        for event in fixture.journal.iter().take(BENCHMARK_PREFIX_EVENTS) {
            emitter
                .emit(
                    event.clone(),
                    &event_context(
                        event,
                        &fixture.topology,
                        &fixture.active,
                        &fixture.completed,
                        &fixture.submissions,
                        &fixture.completions,
                    ),
                )
                .unwrap();
        }
        black_box(emitter.cursor().last_sequence());
    }
    started.elapsed().as_nanos()
}

fn basic_elapsed(fixture: &JournalFixture) -> u128 {
    let sink = CountingExecutionEventSink::default();
    let started = Instant::now();
    for _ in 0..JOURNAL_REPETITIONS {
        let mut emitter = ExecutionEventEmitter::new(
            &sink,
            fixture.active.run_id().clone(),
            fixture.active.request_id().clone(),
        );
        for event in fixture.journal.iter().take(BENCHMARK_PREFIX_EVENTS) {
            emitter
                .emit(
                    event.clone(),
                    &event_context(
                        event,
                        &fixture.topology,
                        &fixture.active,
                        &fixture.completed,
                        &fixture.submissions,
                        &fixture.completions,
                    ),
                )
                .unwrap();
        }
        black_box(emitter.cursor().last_sequence());
    }
    let expected = u64::try_from(JOURNAL_REPETITIONS * BENCHMARK_PREFIX_EVENTS).unwrap();
    assert_eq!(sink.records.load(Ordering::Relaxed), expected);
    started.elapsed().as_nanos()
}

fn provider_fingerprint(provider: &dyn OperationResourceEstimator) -> usize {
    let descriptor = black_box(provider).descriptor();
    black_box(
        descriptor.provider_id().as_str().len()
            + descriptor.operation_id().as_str().len()
            + descriptor.operation_fingerprint().len()
            + descriptor.resource_estimator_id().len()
            + descriptor
                .resource_estimator_implementation_fingerprint()
                .len()
            + descriptor.provider_implementation_fingerprint().len(),
    )
}

fn direct_provider_elapsed(provider: &TestExecutionProvider) -> u128 {
    let started = Instant::now();
    let mut value = 0_usize;
    for _ in 0..PROVIDER_REPETITIONS {
        value ^= provider_fingerprint(black_box(provider));
    }
    black_box(value);
    started.elapsed().as_nanos()
}

fn dynamic_provider_elapsed(provider: &dyn OperationResourceEstimator) -> u128 {
    let started = Instant::now();
    let mut value = 0_usize;
    for _ in 0..PROVIDER_REPETITIONS {
        value ^= provider_fingerprint(black_box(provider));
    }
    black_box(value);
    started.elapsed().as_nanos()
}

fn paired_average(first: u128, second: u128) -> f64 {
    (first as f64 + second as f64) / 2.0
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn overhead_ratio(measured: &[f64], baseline: &[f64]) -> Vec<f64> {
    measured
        .iter()
        .zip(baseline)
        .map(|(measured, baseline)| measured / baseline - 1.0)
        .collect()
}

#[test]
fn g01b_preselected_provider_and_event_sink_overhead_are_bounded() {
    let fixture = journal_fixture();
    assert!(fixture.journal.len() > 32);
    let full_cursor = observe_journal(
        &fixture.journal,
        &fixture.topology,
        &fixture.active,
        &fixture.completed,
        &fixture.submissions,
        &fixture.completions,
    )
    .unwrap();
    assert!(full_cursor.is_terminal());
    assert_eq!(
        usize::try_from(full_cursor.last_sequence()).unwrap(),
        fixture.journal.len(),
    );
    let provider = TestExecutionProvider::new(&catalog());
    let provider_dyn: &dyn OperationResourceEstimator = &provider;

    for _ in 0..WARMUP_COUNT {
        black_box(baseline_elapsed(&fixture));
        black_box(disabled_elapsed(&fixture));
        black_box(basic_elapsed(&fixture));
        black_box(direct_provider_elapsed(&provider));
        black_box(dynamic_provider_elapsed(provider_dyn));
    }

    let mut cursor_baseline_ns = Vec::with_capacity(SAMPLE_COUNT);
    let mut disabled_ns = Vec::with_capacity(SAMPLE_COUNT);
    let mut basic_ns = Vec::with_capacity(SAMPLE_COUNT);
    let mut provider_direct_ns = Vec::with_capacity(SAMPLE_COUNT);
    let mut provider_dynamic_ns = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let baseline_first = baseline_elapsed(&fixture);
        let disabled_first = disabled_elapsed(&fixture);
        let basic_first = basic_elapsed(&fixture);
        let provider_direct_first = direct_provider_elapsed(&provider);
        let provider_dynamic_first = dynamic_provider_elapsed(provider_dyn);

        let provider_dynamic_second = dynamic_provider_elapsed(provider_dyn);
        let provider_direct_second = direct_provider_elapsed(&provider);
        let basic_second = basic_elapsed(&fixture);
        let disabled_second = disabled_elapsed(&fixture);
        let baseline_second = baseline_elapsed(&fixture);

        cursor_baseline_ns.push(paired_average(baseline_first, baseline_second));
        disabled_ns.push(paired_average(disabled_first, disabled_second));
        basic_ns.push(paired_average(basic_first, basic_second));
        provider_direct_ns.push(paired_average(
            provider_direct_first,
            provider_direct_second,
        ));
        provider_dynamic_ns.push(paired_average(
            provider_dynamic_first,
            provider_dynamic_second,
        ));
    }

    let disabled_ratios = overhead_ratio(&disabled_ns, &cursor_baseline_ns);
    let basic_ratios = overhead_ratio(&basic_ns, &cursor_baseline_ns);
    let provider_ratios = overhead_ratio(&provider_dynamic_ns, &provider_direct_ns);
    let disabled_median = median(disabled_ratios.clone());
    let basic_median = median(basic_ratios.clone());
    let provider_median = median(provider_ratios.clone());
    let provider_delta_per_call_ns = median(
        provider_dynamic_ns
            .iter()
            .zip(&provider_direct_ns)
            .map(|(dynamic, direct)| (dynamic - direct) / PROVIDER_REPETITIONS as f64)
            .collect(),
    );

    let report = serde_json::json!({
        "schema_version": 1,
        "sample_count": SAMPLE_COUNT,
        "warmup_count": WARMUP_COUNT,
        "journal_events": fixture.journal.len(),
        "benchmark_prefix_events": BENCHMARK_PREFIX_EVENTS,
        "journal_repetitions_per_sample": JOURNAL_REPETITIONS,
        "provider_calls_per_sample": PROVIDER_REPETITIONS,
        "cursor_baseline_ns": cursor_baseline_ns,
        "disabled_sink_ns": disabled_ns,
        "basic_sink_ns": basic_ns,
        "provider_direct_ns": provider_direct_ns,
        "provider_dynamic_ns": provider_dynamic_ns,
        "disabled_sink_paired_overhead": disabled_ratios,
        "basic_sink_paired_overhead": basic_ratios,
        "provider_paired_overhead": provider_ratios,
        "disabled_sink_median_overhead": disabled_median,
        "basic_sink_median_overhead": basic_median,
        "provider_median_overhead": provider_median,
        "provider_median_delta_per_call_ns": provider_delta_per_call_ns,
        "thresholds": {
            "disabled_sink_median_overhead": 0.01,
            "basic_sink_median_overhead": 0.02,
            "provider_median_overhead_or_delta_ns": {
                "ratio": 0.01,
                "absolute_ns": 2.0
            }
        }
    });
    println!("G01B OVERHEAD JSON: {report}");

    assert!(
        disabled_median <= 0.01,
        "disabled event sink median overhead {disabled_median:.6} exceeds 0.01"
    );
    assert!(
        basic_median <= 0.02,
        "basic event sink median overhead {basic_median:.6} exceeds 0.02"
    );
    assert!(
        provider_median <= 0.01 || provider_delta_per_call_ns <= 2.0,
        "provider dispatch median overhead {provider_median:.6} and delta \
         {provider_delta_per_call_ns:.6}ns exceed the accepted bounds"
    );
    println!("G01B PROVIDER DISPATCH PASS: 30/30");
    println!("G01B DISABLED EVENT SINK PASS: 30/30");
    println!("G01B BASIC EVENT SINK PASS: 30/30");
}
