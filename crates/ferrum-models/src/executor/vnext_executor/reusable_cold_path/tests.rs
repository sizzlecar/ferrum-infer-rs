//! Real ledger/sink contracts. CPU lanes supply owned identities; these tests
//! do not simulate CUDA capture, dispatch completion, or measured performance.
use super::*;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use serde_json::{json, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};

struct IdentityFixture {
    composition: CpuVNextComposition,
    lane: Arc<ExecutionLane<CpuDeviceRuntime>>,
}

impl IdentityFixture {
    fn new() -> Self {
        let composition = CpuVNextComposition::create(
            DeviceId::new("device.cold-path-observation.cpu").unwrap(),
            1024 * 1024,
        )
        .unwrap();
        let lane = ExecutionLane::create(Arc::clone(composition.runtime())).unwrap();
        Self { composition, lane }
    }

    fn program(&self, slot: u64, layout: char, topology: u8) -> DeviceReusableExecutionProgramId {
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("execution.cold-path-observation").unwrap(),
            ReusableExecutionCapacity::new(8, 8, 65).unwrap(),
        )
        .unwrap();
        DeviceReusableExecutionProgramId::new(
            serde_json::from_value(json!("a".repeat(64))).unwrap(),
            self.composition
                .runtime()
                .descriptor()
                .runtime_implementation_fingerprint
                .clone(),
            self.lane.id(),
            bucket.bucket_id().clone(),
            layout.to_string().repeat(64),
            "c".repeat(64),
            slot,
            8,
            8,
            65,
        )
        .unwrap()
        .with_topology_fingerprint(
            DeviceReusableExecutionTopologyFingerprint::from_sha256([topology; 32]),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Host(SubmissionWaveDispatchStage, Duration),
    Device(DeviceSubmissionStage, Duration),
    Submission(DeviceReusableExecutionObservation),
    Preparation(
        Option<DeviceReusableExecutionProgramId>,
        bool,
        DeviceReusableExecutionObservation,
    ),
    Gaps(
        DeviceReusableExecutionProgramId,
        Vec<DeviceReusableExecutionProgramGap>,
    ),
}

#[derive(Default)]
struct CaptureSink(Mutex<Vec<Event>>);

impl DeviceSubmissionTimingSink for CaptureSink {
    const ENABLED: bool = true;

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration) {
        self.0.lock().push(Event::Device(stage, elapsed));
    }

    fn record_reusable_execution(&self, observation: DeviceReusableExecutionObservation) {
        self.0.lock().push(Event::Submission(observation));
    }

    fn record_reusable_preparation(
        &self,
        program_id: Option<&DeviceReusableExecutionProgramId>,
        capture_allowed: bool,
        observation: DeviceReusableExecutionObservation,
    ) {
        self.0.lock().push(Event::Preparation(
            program_id.cloned(),
            capture_allowed,
            observation,
        ));
    }

    fn record_reusable_program_gaps(
        &self,
        program_id: &DeviceReusableExecutionProgramId,
        gaps: &[DeviceReusableExecutionProgramGap],
    ) {
        self.0
            .lock()
            .push(Event::Gaps(program_id.clone(), gaps.to_vec()));
    }
}

impl SubmissionWaveDispatchTimingSink for CaptureSink {
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration) {
        self.0.lock().push(Event::Host(stage, elapsed));
    }
}

fn no_resident() -> InvocationPreparationStats {
    InvocationPreparationStats {
        segment_misses: 1,
        segment_no_resident_program: 1,
        ..Default::default()
    }
}

fn records(snapshot: &Value) -> &[Value] {
    snapshot["records"].as_array().unwrap()
}

#[test]
fn reusable_cold_path_full_identity_and_callback_association_do_not_collapse_same_shape() {
    let fixture = IdentityFixture::new();
    let base = fixture.program(1, 'b', 1);
    let other_slot = fixture.program(2, 'b', 1);
    let other_layout = fixture.program(1, 'd', 1);
    let other_topology = fixture.program(1, 'b', 2);
    let sink = CaptureSink::default();
    let ledger = Ledger::default();
    for id in [&base, &other_slot, &other_layout, &other_topology] {
        assert_eq!(id.bucket_id(), base.bucket_id());
        let mut wave = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
        let attempt = AttemptObservation::new(
            &sink,
            Some(id.clone()),
            9,
            Some(8),
            Some(VNextReusableExecutionCatalogMissReason::EpochMismatch),
        );
        attempt.record_reusable_preparation(Some(id), false, Default::default());
        attempt.record_reusable_program_gaps(&base, &[]);
        wave.record_attempt(attempt, no_resident(), Outcome::Submitted, None);
        wave.finish(None, Outcome::Submitted);
    }
    let snapshot = ledger.snapshot();
    for (row, id) in
        records(&snapshot)
            .iter()
            .zip([&base, &other_slot, &other_layout, &other_topology])
    {
        let attempt = &row["attempts"][0];
        assert_eq!(attempt["program_id"], serde_json::to_value(id).unwrap());
        assert_eq!(attempt["lane_epoch"], 9);
        assert_eq!(attempt["catalog_epoch"], 8);
        assert_eq!(attempt["catalog_miss"], "epoch_mismatch");
        assert_eq!(attempt["backend_preparation"]["identity"], "exact");
        assert_eq!(
            attempt["program_gaps"]["identity"],
            if id == &base { "exact" } else { "mismatch" }
        );
    }
    assert_eq!(records(&snapshot).len(), 4);
}

#[test]
fn reusable_cold_path_warm_capture_and_hit_keep_actual_gap_boundaries() {
    let fixture = IdentityFixture::new();
    let id = fixture.program(1, 'b', 1);
    let sink = CaptureSink::default();
    let ledger = Ledger::default();
    let mut warm = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
    let attempt = AttemptObservation::new(
        &sink,
        Some(id.clone()),
        3,
        Some(3),
        Some(VNextReusableExecutionCatalogMissReason::ProgramAbsent),
    );
    let mut observed = DeviceReusableExecutionObservation::default();
    observed.observe_warmup_required_segment();
    attempt.record_reusable_preparation(Some(&id), true, observed);
    attempt.record_reusable_program_gaps(
        &id,
        &[
            DeviceReusableExecutionProgramGap::new(
                2,
                DeviceReusableExecutionProgramGapReason::WarmupRequired,
            ),
            DeviceReusableExecutionProgramGap::new(
                3,
                DeviceReusableExecutionProgramGapReason::WarmupRequired,
            ),
        ],
    );
    warm.record_attempt(
        attempt,
        no_resident(),
        Outcome::Submitted,
        Some(Duration::from_nanos(17)),
    );
    warm.finish(Some(Duration::from_nanos(31)), Outcome::Submitted);

    let mut capture = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
    let attempt = AttemptObservation::new(
        &sink,
        Some(id.clone()),
        3,
        Some(3),
        Some(VNextReusableExecutionCatalogMissReason::ProgramAbsent),
    );
    let mut observed = DeviceReusableExecutionObservation::default();
    observed.observe_captured_segment();
    attempt.record_reusable_preparation(Some(&id), true, observed);
    attempt.record_reusable_program_gaps(&id, &[]);
    capture.record_attempt(attempt, no_resident(), Outcome::Submitted, None);
    capture.finish(None, Outcome::Submitted);

    let mut hit = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
    let attempt = AttemptObservation::new(&sink, Some(id.clone()), 3, Some(3), None);
    let mut observed = DeviceReusableExecutionObservation::default();
    observed.observe_replayed_segment(2);
    attempt.record_reusable_execution(observed);
    hit.record_attempt(
        attempt,
        InvocationPreparationStats {
            segment_hits: 1,
            ..Default::default()
        },
        Outcome::Submitted,
        None,
    );
    hit.finish(None, Outcome::Submitted);

    let snapshot = ledger.snapshot();
    let rows = records(&snapshot);
    let warmup_index = snapshot["gap_reason_order"]
        .as_array()
        .unwrap()
        .iter()
        .position(|reason| reason == "warmup_required")
        .unwrap();
    assert_eq!(
        rows[0]["attempts"][0]["program_gaps"]["node_counts_by_reason"][warmup_index],
        2
    );
    assert_eq!(
        rows[0]["attempts"][0]["backend_preparation"]["segments"]["warmup_required_segments"],
        1
    );
    assert_eq!(rows[0]["attempts"][0]["disposition"], "no_resident_program");
    assert_eq!(rows[1]["attempts"][0]["disposition"], "no_resident_program");
    assert!(
        rows[1]["attempts"][0]["program_gaps"]["node_counts_by_reason"]
            .as_array()
            .unwrap()
            .iter()
            .all(|count| count == 0)
    );
    assert_eq!(rows[2]["attempts"][0]["disposition"], "hit");
    assert!(rows[2]["attempts"][0]["program_gaps"].is_null());
    assert!(rows[2]["attempts"][0]["backend_preparation"].is_null());
    for row in rows {
        assert_eq!(
            row["attempts"][0]["program_id"],
            serde_json::to_value(&id).unwrap()
        );
    }
}

#[test]
fn reusable_cold_path_retry_and_phase_keep_nested_intervals_on_their_own_attempts() {
    let fixture = IdentityFixture::new();
    let id = fixture.program(1, 'b', 1);
    let metrics = VNextExecutorMetrics::default();
    let ledger = Ledger::default();
    let timing = VNextWaveTimingSink {
        aggregate: &metrics.wave_timing,
        phase: &metrics.decode_wave_timing,
    };
    let mut wave = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
    let first = AttemptObservation::new(&timing, Some(id.clone()), 3, Some(3), None);
    first.record(
        SubmissionWaveDispatchStage::ProviderNodeEncode,
        Duration::from_nanos(7),
    );
    first.record_device_submission(
        DeviceSubmissionStage::EnqueueCommands,
        Duration::from_nanos(5),
    );
    wave.record_attempt(
        first,
        InvocationPreparationStats {
            segment_misses: 1,
            segment_no_cached_recipe: 1,
            segment_incomplete_declarations: 1,
            ..Default::default()
        },
        Outcome::DefinitelyNotSubmitted,
        Some(Duration::from_nanos(13)),
    );
    let retry = AttemptObservation::new(&timing, Some(id.clone()), 3, Some(3), None);
    retry.record(
        SubmissionWaveDispatchStage::ProviderNodeEncode,
        Duration::from_nanos(11),
    );
    wave.record_attempt(
        retry,
        no_resident(),
        Outcome::Submitted,
        Some(Duration::from_nanos(17)),
    );
    wave.finish(Some(Duration::from_nanos(43)), Outcome::Submitted);

    let mut prefill = ledger.begin(true, VNextExecutionWaveKind::Prefill).unwrap();
    let prefill_timing = VNextWaveTimingSink {
        aggregate: &metrics.wave_timing,
        phase: &metrics.prefill_wave_timing,
    };
    let attempt = AttemptObservation::new(
        &prefill_timing,
        None,
        4,
        None,
        Some(VNextReusableExecutionCatalogMissReason::ProgramIdentityUnavailable),
    );
    attempt.record_reusable_preparation(None, false, Default::default());
    prefill.record_attempt(attempt, Default::default(), Outcome::ContractError, None);
    prefill.finish(Some(Duration::from_nanos(19)), Outcome::ContractError);
    let snapshot = ledger.snapshot();
    let rows = records(&snapshot);
    assert_eq!(rows[0]["phase"], "decode");
    assert_eq!(rows[0]["host_encode_submit_ns"], 43);
    assert_eq!(rows[0]["attempts"].as_array().unwrap().len(), 2);
    assert_eq!(
        rows[0]["attempts"][0]["outcome"],
        "definitely_not_submitted"
    );
    assert_eq!(rows[0]["attempts"][0]["incomplete_declarations"], true);
    assert_eq!(rows[0]["attempts"][0]["provider_encode_submit_ns"], 13);
    assert_eq!(rows[0]["attempts"][0]["provider_node_encode_ns"], 7);
    assert_eq!(rows[0]["attempts"][0]["enqueue_commands_ns"], 5);
    assert_eq!(rows[0]["attempts"][1]["ordinal"], 1);
    assert_eq!(rows[0]["attempts"][1]["provider_encode_submit_ns"], 17);
    assert_eq!(rows[0]["attempts"][1]["provider_node_encode_ns"], 11);
    assert!(rows[0]["attempts"][1]["enqueue_commands_ns"].is_null());
    assert_eq!(rows[1]["ordinal"], 1);
    assert_eq!(rows[1]["phase"], "prefill");
    assert_eq!(rows[1]["attempts"][0]["ordinal"], 0);
    assert_eq!(rows[1]["attempts"][0]["disposition"], "before_decision");
    assert!(rows[1]["attempts"][0]["program_id"].is_null());
    assert!(rows[1]["attempts"][0]["provider_encode_submit_ns"].is_null());
    assert_eq!(
        rows[1]["attempts"][0]["backend_preparation"]["identity"],
        "both_unavailable"
    );
    assert_eq!(
        metrics.wave_timing.provider_node_encode.snapshot()["total_ns"],
        18
    );
    assert_eq!(
        metrics.decode_wave_timing.provider_node_encode.snapshot()["samples"],
        2
    );
    assert_eq!(
        metrics.prefill_wave_timing.provider_node_encode.snapshot()["samples"],
        0
    );
}

#[test]
fn reusable_cold_path_callbacks_forward_unchanged_and_repeats_remain_visible() {
    let fixture = IdentityFixture::new();
    let id = fixture.program(1, 'b', 1);
    let sink = CaptureSink::default();
    let direct = CaptureSink::default();
    let ledger = Ledger::default();
    let mut wave = ledger.begin(true, VNextExecutionWaveKind::Mixed).unwrap();
    let observer = AttemptObservation::new(&sink, Some(id.clone()), 1, Some(1), None);
    fn callbacks(
        sink: &impl SubmissionWaveDispatchTimingSink,
        id: &DeviceReusableExecutionProgramId,
    ) {
        sink.record(
            SubmissionWaveDispatchStage::ProviderNodeEncode,
            Duration::from_nanos(7),
        );
        sink.record(
            SubmissionWaveDispatchStage::ProviderNodeEncode,
            Duration::from_nanos(19),
        );
        sink.record(
            SubmissionWaveDispatchStage::SegmentBackingWindowIntersections,
            Duration::from_nanos(3),
        );
        sink.record_device_submission(DeviceSubmissionStage::EnqueueCommands, Duration::ZERO);
        sink.record_device_submission(
            DeviceSubmissionStage::EnqueueCommands,
            Duration::from_nanos(5),
        );
        let mut warm = DeviceReusableExecutionObservation::default();
        warm.observe_warmup_required_segment();
        sink.record_reusable_preparation(Some(id), false, warm);
        sink.record_reusable_preparation(None, true, Default::default());
        sink.record_reusable_program_gaps(id, &[]);
        sink.record_reusable_program_gaps(
            id,
            &[DeviceReusableExecutionProgramGap::new(
                1,
                DeviceReusableExecutionProgramGapReason::CaptureRejected,
            )],
        );
        sink.record_reusable_execution(warm);
        sink.record_reusable_execution(Default::default());
    }
    callbacks(&direct, &id);
    callbacks(&observer, &id);
    wave.record_attempt(observer, no_resident(), Outcome::ProviderError, None);
    wave.finish(None, Outcome::ProviderError);
    assert_eq!(*sink.0.lock(), *direct.0.lock());
    let snapshot = ledger.snapshot();
    let attempt = &snapshot["records"][0]["attempts"][0];
    assert_eq!(attempt["repeated_callback"], true);
    assert_eq!(attempt["provider_node_encode_ns"], 7);
    assert_eq!(attempt["enqueue_commands_ns"], 0);
    assert_eq!(attempt["backend_preparation"]["identity"], "exact");
    assert_eq!(
        attempt["backend_preparation"]["capture_allowed_at_entry"],
        false
    );
    assert!(attempt["program_gaps"]["node_counts_by_reason"]
        .as_array()
        .unwrap()
        .iter()
        .all(|count| count == 0));
    assert_eq!(
        attempt["backend_submission_observation"]["warmup_required_segments"],
        1
    );
}

#[test]
fn reusable_cold_path_capacity_drop_and_reset_never_invent_completed_history() {
    let ledger = Ledger::with_capacity(2);
    let sink = CaptureSink::default();
    let mut wave = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
    for _ in 0..MAX_DEFINITELY_NOT_SUBMITTED_RETRIES + 2 {
        let observer = AttemptObservation::new(&sink, None, 1, None, None);
        wave.record_attempt(
            observer,
            Default::default(),
            Outcome::InputUploadError,
            None,
        );
    }
    wave.finish(None, Outcome::InputUploadError);
    let unfinalized = ledger.begin(true, VNextExecutionWaveKind::Mixed).unwrap();
    drop(unfinalized);
    assert!(ledger
        .begin(true, VNextExecutionWaveKind::Prefill)
        .is_none());
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot["waves_started"], 3);
    assert_eq!(snapshot["overflow_waves"], 1);
    assert_eq!(snapshot["unfinalized_waves"], 1);
    assert_eq!(records(&snapshot).len(), 1);
    assert_eq!(records(&snapshot)[0]["attempt_overflow"], 1);
    assert_eq!(
        records(&snapshot)[0]["attempts"].as_array().unwrap().len(),
        MAX_DEFINITELY_NOT_SUBMITTED_RETRIES as usize + 1
    );
    assert!(records(&snapshot)[0]["host_encode_submit_ns"].is_null());
    ledger.reset();
    assert_eq!(ledger.snapshot(), Ledger::with_capacity(2).snapshot());

    let metrics = AtomicDurationMetrics::default();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _original_timer = metrics.start();
        let mut wave = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
        wave.record_attempt(
            AttemptObservation::new(&sink, None, 1, None, None),
            Default::default(),
            Outcome::DefinitelyNotSubmitted,
            None,
        );
        panic!("fixture unwinds before wave finalization");
    }))
    .is_err());
    assert_eq!(metrics.snapshot()["samples"], 1);
    assert_eq!(ledger.snapshot()["unfinalized_waves"], 1);
    assert!(records(&ledger.snapshot()).is_empty());
}

#[test]
fn reusable_cold_path_disabled_observation_and_enabled_records_retain_no_lane_owner() {
    let fixture = IdentityFixture::new();
    let id = fixture.program(1, 'b', 1);
    let lane = Arc::downgrade(&fixture.lane);
    let runtime = Arc::downgrade(fixture.composition.runtime());
    let lane_owners = Arc::strong_count(&fixture.lane);
    let runtime_owners = Arc::strong_count(fixture.composition.runtime());
    let ledger = Ledger::default();
    let metrics = AtomicDurationMetrics::default();
    assert!(ledger
        .begin(false, VNextExecutionWaveKind::Decode)
        .is_none());
    assert!(metrics.start_if(false).is_none());
    assert!(!<AttemptObservation<'_, DisabledDeviceSubmissionTimingSink> as DeviceSubmissionTimingSink>::ENABLED);
    assert_eq!(ledger.snapshot()["waves_started"], 0);
    assert_eq!(metrics.snapshot()["samples"], 0);

    let sink = CaptureSink::default();
    let mut wave = ledger.begin(true, VNextExecutionWaveKind::Decode).unwrap();
    let observer = AttemptObservation::new(&sink, Some(id), 2, None, None);
    observer.record_reusable_preparation(None, false, Default::default());
    wave.record_attempt(observer, no_resident(), Outcome::Submitted, None);
    let elapsed = metrics.start().finish().unwrap();
    wave.finish(Some(elapsed), Outcome::Submitted);
    assert_eq!(metrics.snapshot()["samples"], 1);
    assert_eq!(
        ledger.snapshot()["records"][0]["host_encode_submit_ns"],
        metrics.snapshot()["total_ns"]
    );
    assert_eq!(
        ledger.snapshot()["records"][0]["attempts"][0]["backend_preparation"]["identity"],
        "mismatch"
    );
    assert_eq!(Arc::strong_count(&fixture.lane), lane_owners);
    assert_eq!(
        Arc::strong_count(fixture.composition.runtime()),
        runtime_owners
    );
    drop(fixture);
    assert!(lane.upgrade().is_none());
    assert!(runtime.upgrade().is_none());
    assert_eq!(records(&ledger.snapshot()).len(), 1);
}
