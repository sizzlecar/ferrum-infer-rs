//! Exercise the real observer/sink boundary without simulating device execution.
use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Default)]
struct PreparationCapture(Mutex<Vec<InvocationPreparationStats>>);

impl InvocationPreparationSink for PreparationCapture {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.lock().push(stats);
    }
}

fn observed_attempt(
    stats: InvocationPreparationStats,
    outcome: Outcome,
    elapsed: Option<Duration>,
) -> AttemptSummary {
    let timing = VNextWaveTimingMetrics::default();
    let preparation = PreparationCapture::default();
    let observer = AttemptObservation::new(&timing, &preparation).unwrap();
    observer.record_preparation(stats);
    if let Some(elapsed) = elapsed {
        observer.record(SubmissionWaveDispatchStage::ProviderNodeEncode, elapsed);
    }
    observer.finish(outcome)
}

fn observations(snapshot: &serde_json::Value, table: &str) -> u64 {
    snapshot[table]
        .as_object()
        .unwrap()
        .values()
        .flat_map(|row| row.as_object().unwrap().values())
        .map(|bucket| bucket["observations"].as_u64().unwrap())
        .sum()
}

#[test]
fn segment_outcomes_attribute_once_to_the_selected_phase_and_reset_after_startup() {
    let metrics = VNextExecutorMetrics::default();
    let cases = [
        (
            VNextExecutionWaveKind::Prefill,
            "no_resident_program",
            InvocationPreparationStats {
                segment_misses: 1,
                segment_no_resident_program: 1,
                ..Default::default()
            },
        ),
        (
            VNextExecutionWaveKind::Decode,
            "hit",
            InvocationPreparationStats {
                segment_hits: 1,
                ..Default::default()
            },
        ),
        (
            VNextExecutionWaveKind::Mixed,
            "no_cached_recipe",
            InvocationPreparationStats {
                segment_misses: 1,
                segment_no_cached_recipe: 1,
                ..Default::default()
            },
        ),
    ];
    for (kind, _, stats) in cases {
        let phase = metrics.wave_timing_for(kind);
        let timing = VNextWaveTimingSink {
            aggregate: &metrics.wave_timing,
            phase,
        };
        let preparation = PreparationCapture::default();
        let observer = AttemptObservation::new(&timing, &preparation).unwrap();
        observer.record_preparation(stats);
        observer.record(
            SubmissionWaveDispatchStage::ProviderNodeEncode,
            Duration::from_nanos(17),
        );
        let mut wave = WaveObservation::default();
        wave.record_attempt(
            observer.finish(Outcome::Success),
            &metrics.wave_timing.segment_outcomes,
            &phase.segment_outcomes,
        );
        wave.finish(
            Outcome::Success,
            Duration::from_nanos(29),
            &metrics.wave_timing.segment_outcomes,
            &phase.segment_outcomes,
        );
    }
    for (kind, disposition, _) in cases {
        let phase = metrics.wave_timing_for(kind);
        let snapshot = phase.segment_outcomes.snapshot();
        assert_eq!(observations(&snapshot, "provider_attempts"), 1);
        assert_eq!(observations(&snapshot, "whole_host"), 1);
        assert_eq!(
            snapshot["provider_attempts"][disposition]["success"]["elapsed"]["total_ns"],
            17
        );
        assert_eq!(
            snapshot["whole_host"][disposition]["success"]["elapsed"]["total_ns"],
            29
        );
        assert_eq!(phase.provider_node_encode.snapshot()["samples"], 1);
    }
    let aggregate = metrics.wave_timing.segment_outcomes.snapshot();
    assert_eq!(observations(&aggregate, "provider_attempts"), 3);
    assert_eq!(observations(&aggregate, "whole_host"), 3);
    metrics.reset_after_startup();
    for timing in [
        &metrics.wave_timing,
        &metrics.prefill_wave_timing,
        &metrics.decode_wave_timing,
        &metrics.mixed_wave_timing,
    ] {
        assert_eq!(
            timing.segment_outcomes.snapshot(),
            OutcomeMetrics::default().snapshot()
        );
    }
}

#[test]
fn segment_outcomes_forward_existing_callbacks_without_counting_child_time_twice() {
    let timing = VNextWaveTimingMetrics::default();
    let preparation = PreparationCapture::default();
    let observer = AttemptObservation::new(&timing, &preparation).unwrap();
    let stats = InvocationPreparationStats {
        segment_hits: 1,
        segment_encoded_nodes: 3,
        ..Default::default()
    };
    observer.record_preparation(stats);
    observer.record(
        SubmissionWaveDispatchStage::ProviderNodeEncode,
        Duration::from_nanos(37),
    );
    observer.record(
        SubmissionWaveDispatchStage::NodeInvocationConstruct,
        Duration::from_nanos(11),
    );
    observer.record_device_submission(
        DeviceSubmissionStage::EnqueueCommands,
        Duration::from_nanos(7),
    );
    let mut reusable = DeviceReusableExecutionObservation::default();
    reusable.observe_candidate_segment();
    reusable.observe_replayed_segment(3);
    observer.record_reusable_execution(reusable);
    let summary = observer.finish(Outcome::Success);
    assert_eq!(summary.disposition, Disposition::Hit);
    assert_eq!(summary.provider_elapsed, Some(Duration::from_nanos(37)));
    assert_eq!(*preparation.0.lock(), vec![stats]);
    assert_eq!(timing.provider_node_encode.snapshot()["total_ns"], 37);
    assert_eq!(timing.node_invocation_construct.snapshot()["total_ns"], 11);
    assert_eq!(
        timing.device_submit_enqueue_commands.snapshot()["total_ns"],
        7
    );
    assert_eq!(timing.reusable_execution.snapshot()["replayed_commands"], 3);
}

#[test]
fn segment_outcomes_incomplete_declarations_is_a_flag_not_another_elapsed_bucket() {
    let aggregate = OutcomeMetrics::default();
    let phase = OutcomeMetrics::default();
    let mut wave = WaveObservation::default();
    wave.record_attempt(
        observed_attempt(
            InvocationPreparationStats {
                segment_misses: 1,
                segment_no_cached_recipe: 1,
                segment_incomplete_declarations: 1,
                ..Default::default()
            },
            Outcome::Success,
            Some(Duration::from_nanos(19)),
        ),
        &aggregate,
        &phase,
    );
    wave.finish(
        Outcome::Success,
        Duration::from_nanos(31),
        &aggregate,
        &phase,
    );
    let snapshot = aggregate.snapshot();
    assert_eq!(observations(&snapshot, "provider_attempts"), 1);
    assert_eq!(observations(&snapshot, "whole_host"), 1);
    assert_eq!(
        snapshot["provider_attempts"]["no_cached_recipe"]["success"]["incomplete_declarations"],
        1
    );
    assert_eq!(
        snapshot["whole_host"]["no_cached_recipe"]["success"]["elapsed"]["total_ns"],
        31
    );
    let ambiguous = observed_attempt(
        InvocationPreparationStats {
            segment_no_cached_recipe: 1,
            segment_unsupported_encoder: 1,
            ..Default::default()
        },
        Outcome::Error,
        None,
    );
    assert_eq!(ambiguous.disposition, Disposition::Other);
}

#[test]
fn segment_outcomes_retry_keeps_each_attempt_and_does_not_relabel_whole_host_by_last_cause() {
    let aggregate = OutcomeMetrics::default();
    let phase = OutcomeMetrics::default();
    let mut wave = WaveObservation::default();
    wave.record_attempt(
        observed_attempt(
            InvocationPreparationStats {
                segment_no_cached_recipe: 1,
                ..Default::default()
            },
            Outcome::Error,
            Some(Duration::from_nanos(13)),
        ),
        &aggregate,
        &phase,
    );
    wave.mark_retry();
    wave.record_attempt(
        observed_attempt(
            InvocationPreparationStats {
                segment_hits: 1,
                ..Default::default()
            },
            Outcome::Success,
            Some(Duration::from_nanos(23)),
        ),
        &aggregate,
        &phase,
    );
    wave.finish(
        Outcome::Success,
        Duration::from_nanos(53),
        &aggregate,
        &phase,
    );
    let snapshot = aggregate.snapshot();
    assert_eq!(observations(&snapshot, "provider_attempts"), 2);
    assert_eq!(observations(&snapshot, "whole_host"), 1);
    assert_eq!(
        snapshot["provider_attempts"]["no_cached_recipe"]["error"]["elapsed"]["total_ns"],
        13
    );
    assert_eq!(
        snapshot["provider_attempts"]["hit"]["success"]["elapsed"]["total_ns"],
        23
    );
    assert_eq!(
        snapshot["whole_host"]["retried"]["success"]["elapsed"]["total_ns"],
        53
    );
    assert_eq!(snapshot["whole_host"]["hit"]["success"]["observations"], 0);
}

#[test]
fn segment_outcomes_retry_failure_before_next_dispatch_is_still_retried() {
    let aggregate = OutcomeMetrics::default();
    let phase = OutcomeMetrics::default();
    let mut wave = WaveObservation::default();
    wave.record_attempt(
        observed_attempt(
            InvocationPreparationStats {
                segment_no_resident_program: 1,
                ..Default::default()
            },
            Outcome::Error,
            Some(Duration::from_nanos(5)),
        ),
        &aggregate,
        &phase,
    );
    wave.mark_retry();
    // Rebuilding retry inputs failed; no second core dispatch was entered.
    wave.finish(Outcome::Error, Duration::from_nanos(41), &aggregate, &phase);
    let snapshot = aggregate.snapshot();
    assert_eq!(observations(&snapshot, "provider_attempts"), 1);
    assert_eq!(
        snapshot["whole_host"]["retried"]["error"]["elapsed"]["total_ns"],
        41
    );
    assert_eq!(
        snapshot["whole_host"]["no_resident_program"]["error"]["observations"],
        0
    );
}

#[test]
fn segment_outcomes_ordinary_error_records_without_inventing_provider_elapsed() {
    let timing = VNextWaveTimingMetrics::default();
    let preparation = PreparationCapture::default();
    let observer = AttemptObservation::new(&timing, &preparation).unwrap();
    let summary = observer.finish(Outcome::Error);
    assert_eq!(summary.disposition, Disposition::BeforeDecision);
    assert_eq!(summary.provider_elapsed, None);
    let aggregate = OutcomeMetrics::default();
    let phase = OutcomeMetrics::default();
    let mut wave = WaveObservation::default();
    wave.record_attempt(summary, &aggregate, &phase);
    wave.finish(Outcome::Error, Duration::from_nanos(9), &aggregate, &phase);
    let snapshot = aggregate.snapshot();
    assert_eq!(
        snapshot["provider_attempts"]["before_decision"]["error"]["observations"],
        1
    );
    assert_eq!(
        snapshot["provider_attempts"]["before_decision"]["error"]["elapsed"]["samples"],
        0
    );
    assert_eq!(
        snapshot["whole_host"]["before_decision"]["error"]["elapsed"]["total_ns"],
        9
    );
}

#[test]
fn segment_outcomes_unwind_does_not_flush_an_unfinished_attempt_or_wave() {
    let timing = VNextWaveTimingMetrics::default();
    let preparation = PreparationCapture::default();
    let aggregate = OutcomeMetrics::default();
    let phase = OutcomeMetrics::default();
    let existing_parent = AtomicDurationMetrics::default();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _parent = existing_parent.start();
        let mut wave = WaveObservation::default();
        wave.record_attempt(
            observed_attempt(
                InvocationPreparationStats {
                    segment_no_cached_recipe: 1,
                    ..Default::default()
                },
                Outcome::Error,
                Some(Duration::from_nanos(5)),
            ),
            &aggregate,
            &phase,
        );
        wave.mark_retry();
        let observer = AttemptObservation::new(&timing, &preparation).unwrap();
        observer.record_preparation(InvocationPreparationStats {
            segment_hits: 1,
            ..Default::default()
        });
        panic!("test unwinding before dispatch returns");
    }))
    .is_err());
    // Existing parent RAII retains its previous unwind rule; the new explicit
    // outcome boundary must not turn the unwind into a successful observation.
    assert_eq!(existing_parent.snapshot()["samples"], 1);
    let snapshot = aggregate.snapshot();
    assert_eq!(observations(&snapshot, "provider_attempts"), 1);
    assert_eq!(
        snapshot["provider_attempts"]["no_cached_recipe"]["error"]["elapsed"]["total_ns"],
        5
    );
    assert_eq!(observations(&snapshot, "whole_host"), 0);
    assert_eq!(phase.snapshot(), snapshot);
    assert_eq!(preparation.0.lock().len(), 1);
}

#[test]
fn segment_outcomes_finished_parent_reuses_exact_elapsed_and_records_only_once() {
    let parent = AtomicDurationMetrics::default();
    let elapsed = parent.start().finish();
    let aggregate = OutcomeMetrics::default();
    let phase = OutcomeMetrics::default();
    WaveObservation::default().finish(Outcome::Error, elapsed, &aggregate, &phase);
    assert_eq!(parent.snapshot()["samples"], 1);
    let bucket = &aggregate.snapshot()["whole_host"]["before_decision"]["error"]["elapsed"];
    assert_eq!(bucket["samples"], 1);
    assert_eq!(bucket["total_ns"], parent.snapshot()["total_ns"]);
    assert_eq!(bucket["total_ns"], elapsed.as_nanos() as u64);
}

struct OffTiming;

impl DeviceSubmissionTimingSink for OffTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        panic!("Off must not call the device timing observer");
    }
}

impl SubmissionWaveDispatchTimingSink for OffTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        panic!("Off must not call the host timing observer");
    }
}

#[test]
fn segment_outcomes_off_constructs_no_observer_and_preserves_original_preparation_sink() {
    let preparation = PreparationCapture::default();
    assert!(AttemptObservation::new(&OffTiming, &preparation).is_none());
    let parent = AtomicDurationMetrics::default();
    assert!(parent.start_if(OffTiming::ENABLED).is_none());
    let stats = InvocationPreparationStats {
        segment_hits: 1,
        ..Default::default()
    };
    preparation.record_preparation(stats);
    assert_eq!(*preparation.0.lock(), vec![stats]);
    assert_eq!(parent.snapshot()["samples"], 0);
}
