use super::*;
use ferrum_types::observability_profile::FerrumProfileEvent;

fn path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ferrum-prefix-resource-{label}-{}.jsonl",
        RequestId::new()
    ))
}

#[test]
fn prefix_resource_recorder_preserves_typed_unknown_and_declares_loss() {
    let path = path("unknown");
    let journal = SchedulerTraceJournal::create(&path).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Resource;
    let recorder = Recorder::for_detail(&config, ProfileEntrypoint::Serve, Some(&journal)).unwrap();
    let request = RequestId::new();
    recorder.record(Event::Decision {
        route: Route::CacheCapture,
        owner: Owner::new(&request, 17, 23),
        peer: None,
        boundary_tokens: 127,
        snapshot_generation: 8,
        inference_epoch: 4,
        maintenance_epoch: 6,
        decision: Decision::Unknown(PlanningUnknownReason::ComputeBudgetExhausted),
    });
    recorder.finish().unwrap();
    journal.close().unwrap();
    let rows: Vec<FerrumProfileEvent> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    for row in &rows {
        row.validate().unwrap();
        assert_eq!(row.phase, "slo.prefix_resource");
        assert_eq!(row.duration_us, None);
    }
    let event = &rows[0].attributes["prefix"];
    assert_eq!(rows[0].request_id, request.to_string());
    assert_eq!(rows[0].entrypoint, ProfileEntrypoint::Serve);
    assert_eq!(event["owner"]["incarnation"], 17);
    assert_eq!(event["owner"]["work_generation"], 23);
    assert_eq!(event["decision"]["reason"], "ComputeBudgetExhausted");
    assert_eq!(event["inference_epoch"], 4);
    assert_eq!(event["maintenance_epoch"], 6);
    assert_eq!(rows[1].attributes["prefix"]["offered"], 1);
    assert_eq!(rows[1].attributes["prefix"]["accepted"], 1);
    assert_eq!(rows[1].attributes["prefix"]["dropped"], 0);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn prefix_resource_recorder_disabled_and_closed_paths_are_explicit() {
    let path = path("closed");
    let journal = SchedulerTraceJournal::create(&path).unwrap();
    let mut config = EngineConfig::default();
    for detail in [
        ObservabilityProfileDetail::Off,
        ObservabilityProfileDetail::Basic,
        ObservabilityProfileDetail::Host,
    ] {
        config.runtime.profile_detail = detail;
        assert!(Recorder::for_detail(&config, ProfileEntrypoint::Run, Some(&journal)).is_none());
    }
    config.runtime.profile_detail = ObservabilityProfileDetail::Resource;
    assert!(Recorder::for_detail(&config, ProfileEntrypoint::Run, None).is_none());
    let recorder = Recorder::for_detail(&config, ProfileEntrypoint::Run, Some(&journal)).unwrap();
    journal.close().unwrap();
    recorder.record(Event::Decision {
        route: Route::ReadyRestore,
        owner: Owner::new(&RequestId::new(), 1, 0),
        peer: None,
        boundary_tokens: 1,
        snapshot_generation: 1,
        inference_epoch: 2,
        maintenance_epoch: 3,
        decision: Decision::PreferDirect,
    });
    assert_eq!(recorder.counters.offered.load(Ordering::Acquire), 1);
    assert_eq!(recorder.counters.accepted.load(Ordering::Acquire), 0);
    assert_eq!(recorder.counters.dropped.load(Ordering::Acquire), 1);
    assert!(recorder.finish().is_err());
    assert!(std::fs::read_to_string(&path).unwrap().is_empty());
    std::fs::remove_file(path).unwrap();
}
