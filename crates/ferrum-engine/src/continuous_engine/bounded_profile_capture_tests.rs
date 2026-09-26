use super::*;
use ferrum_interfaces::vnext::ExecutionFrameCaptureSummary;
use std::num::NonZeroU32;

fn bounded_capture_sink(
    label: &str,
) -> (
    VNextProfileExecutionEventSink,
    SchedulerTraceJournal,
    PathBuf,
) {
    let path = resource_trace_temp_path(label);
    let journal = create_scheduler_trace_sink(Some(&path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Kernel;
    config.runtime.profile_max_frames_per_request = NonZeroU32::new(2);
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Serve, &config);
    (sink, journal, path)
}

#[test]
fn bounded_profile_deferred_events_disclose_partial_scope_and_keep_identity() {
    let (sink, journal, path) = bounded_capture_sink("bounded-profile-deferred");
    let (_, request_id, accepted) = vnext_profile_test_event();
    sink.enqueue_events(vec![accepted, vnext_profile_test_operation_event()])
        .unwrap();
    journal.close().unwrap();
    let rows = std::fs::read_to_string(&path).unwrap();
    let records: Vec<FerrumProfileEvent> = rows
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].phase, "vnext.request_accepted");
    assert_eq!(records[1].phase, "vnext.operation_submitted");
    for record in &records {
        record.validate().unwrap();
        assert_eq!(record.request_id, request_id.to_string());
        assert_eq!(record.attributes["profile_max_frames_per_request"], 2);
        assert_eq!(
            record.attributes["capture_completeness"],
            "bounded_prefix_not_complete_execution"
        );
        assert_eq!(
            record.attributes["device_capture_scope"],
            "waves_with_at_least_one_in_prefix_participant"
        );
    }
    assert_eq!(records[1].shape["frame_id"], 1);
    assert_eq!(
        records[1].attributes["execution_identity"]["resource_pool_id"],
        7
    );
    assert_eq!(
        sink.device_timing_mode(),
        ferrum_interfaces::vnext::DeviceTimingMode::Kernel
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn bounded_profile_summary_distinguishes_omission_and_unobserved_completion() {
    let (sink, journal, path) = bounded_capture_sink("bounded-profile-summary");
    let (run_id, request_id, _) = vnext_profile_test_event();
    for (completed, captured, succeeded, terminal, pending) in [
        (2, 2, true, true, false),
        (5, 2, true, true, false),
        (2, 2, false, false, false),
        (1, 1, false, false, true),
    ] {
        sink.record_frame_capture_summary(&ExecutionFrameCaptureSummary {
            run_id: run_id.clone(),
            request_id: request_id.clone(),
            limit: NonZeroU32::new(2).unwrap(),
            completed_frames: completed,
            captured_completed_frames: captured,
            request_succeeded: succeeded,
            journal_terminal_observed: terminal,
            pending_submission: pending,
        })
        .unwrap();
    }
    journal.close().unwrap();
    let rows = std::fs::read_to_string(&path).unwrap();
    let records: Vec<FerrumProfileEvent> = rows
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 4);
    assert_eq!(
        records[0].attributes["full_execution_frames_captured"],
        true
    );
    assert_eq!(records[1].shape["omitted_completed_frames"], 3);
    for record in &records[1..] {
        record.validate().unwrap();
        assert_eq!(record.attributes["full_execution_frames_captured"], false);
        assert_eq!(
            record.attributes["prefill_coverage"],
            "requires_chunk_and_first_token_join"
        );
    }
    assert_eq!(records[2].attributes["journal_terminal_observed"], false);
    assert_eq!(records[3].attributes["pending_submission"], true);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn bounded_profile_rejects_inconsistent_accounting_before_writing() {
    let (sink, journal, path) = bounded_capture_sink("bounded-profile-invalid-accounting");
    let (run_id, request_id, _) = vnext_profile_test_event();
    let mut summary = ExecutionFrameCaptureSummary {
        run_id,
        request_id,
        limit: NonZeroU32::new(2).unwrap(),
        completed_frames: 1,
        captured_completed_frames: 2,
        request_succeeded: true,
        journal_terminal_observed: true,
        pending_submission: false,
    };
    assert!(sink.record_frame_capture_summary(&summary).is_err());
    summary.completed_frames = 3;
    summary.captured_completed_frames = 3;
    assert!(sink.record_frame_capture_summary(&summary).is_err());
    summary.captured_completed_frames = 2;
    summary.limit = NonZeroU32::new(3).unwrap();
    assert!(sink.record_frame_capture_summary(&summary).is_err());
    journal.close().unwrap();
    assert!(std::fs::read_to_string(&path).unwrap().is_empty());
    std::fs::remove_file(path).unwrap();
}
