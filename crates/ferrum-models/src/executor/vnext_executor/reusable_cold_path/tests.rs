//! Real ledger/sink contracts. CPU lanes supply owned identities; these tests
//! do not simulate CUDA capture, dispatch completion, or measured performance.
use super::*;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use serde_json::{json, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};

const SELECTED_STAGES: [(SubmissionWaveDispatchStage, &str); 6] = [
    (
        SubmissionWaveDispatchStage::ContractValidateAndReserve,
        "contract_validate_reserve",
    ),
    (
        SubmissionWaveDispatchStage::SegmentBindingPrepareAndEncode,
        "segment_binding_prepare_encode",
    ),
    (
        SubmissionWaveDispatchStage::SegmentFreshAuthorityAndWindows,
        "segment_fresh_authority_and_windows",
    ),
    (
        SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata,
        "segment_backing_permit_and_metadata",
    ),
    (
        SubmissionWaveDispatchStage::SegmentNodeDependenciesAndRegions,
        "segment_node_dependencies_and_regions",
    ),
    (
        SubmissionWaveDispatchStage::SegmentBackendEncodeAndValidate,
        "segment_backend_encode_and_validate",
    ),
];

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
    Enqueue(DeviceEnqueueStage, Duration, Option<Duration>),
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

    fn record_device_enqueue(
        &self,
        stage: DeviceEnqueueStage,
        wall: Duration,
        thread_cpu: Option<Duration>,
    ) {
        self.0.lock().push(Event::Enqueue(stage, wall, thread_cpu));
    }

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
fn reusable_cold_path_decode_preparation_boundaries_close_once_across_dispatch_retries() {
    let ledger = Ledger::default();
    let sink = CaptureSink::default();
    let start = Instant::now();
    let mut wave = ledger.begin_decode(true, 8, start).unwrap();
    let preparation = wave.preparation().unwrap();
    preparation
        .resource_started
        .set(start + Duration::from_nanos(30))
        .unwrap();
    preparation
        .resource_ready
        .set((start + Duration::from_nanos(80), 7))
        .unwrap();
    preparation
        .closed
        .set((start + Duration::from_nanos(100), true))
        .unwrap();
    preparation.stages[PreparationStage::CanonicalParticipants as usize].record(Duration::ZERO);
    preparation.stages[PreparationStage::OperationLockWait as usize]
        .record(Duration::from_nanos(10));
    preparation.stages[PreparationStage::TokensAndExtension as usize]
        .record(Duration::from_nanos(15));
    for nanos in [11, 13] {
        preparation.stages[PreparationStage::StepAdmission as usize]
            .record(Duration::from_nanos(nanos));
    }
    // A second close cannot replace the actual entry-to-dispatch boundary.
    wave.close_preparation(false);
    for outcome in [Outcome::DefinitelyNotSubmitted, Outcome::Submitted] {
        wave.record_attempt(
            AttemptObservation::new(&sink, None, 1, None, None),
            Default::default(),
            outcome,
            None,
        );
    }
    wave.finish(Some(Duration::from_nanos(200)), Outcome::Submitted);
    let snapshot = ledger.snapshot();
    let row = &records(&snapshot)[0];
    let preparation = &row["preparation"];
    assert_eq!(preparation["input_width"], 8);
    assert_eq!(preparation["prepared_width"], 7);
    assert_eq!(preparation["reached_dispatch"], true);
    assert_eq!(preparation["executor_to_dispatch_ns"], 100);
    assert_eq!(
        preparation["executor_prefix_ns"].as_u64().unwrap()
            + preparation["resource_prepare_ns"].as_u64().unwrap()
            + preparation["dispatch_tail_ns"].as_u64().unwrap(),
        100
    );
    assert!(preparation["executor_returned_before_dispatch_ns"].is_null());
    assert_eq!(
        preparation["canonical_participants"],
        json!({"samples":1,"total_ns":0,"overflowed":false})
    );
    assert_eq!(
        preparation["step_admission"],
        json!({"samples":2,"total_ns":24,"overflowed":false})
    );
    assert!(preparation["step_request_prepare"].is_null());
    assert_eq!(row["attempts"].as_array().unwrap().len(), 2);
    assert_eq!(row["host_encode_submit_ns"], 200);
}

#[test]
fn reusable_cold_path_decode_preparation_early_exit_missing_and_disabled_are_distinct() {
    let ledger = Ledger::with_capacity(3);
    let start = Instant::now();
    assert!(ledger.begin_decode(false, 8, start).is_none());
    assert_eq!(ledger.snapshot()["waves_started"], 0);
    for outcome in [Outcome::PreparationDeferred, Outcome::PreparationError] {
        let wave = ledger.begin_decode(true, 8, start).unwrap();
        let preparation = wave.preparation().unwrap();
        preparation
            .resource_started
            .set(start + Duration::from_nanos(3))
            .unwrap();
        preparation
            .closed
            .set((start + Duration::from_nanos(7), false))
            .unwrap();
        preparation.stages[PreparationStage::ResourcePrepareAttempt as usize]
            .record(Duration::ZERO);
        wave.finish(None, outcome);
    }
    ledger
        .begin(true, VNextExecutionWaveKind::Prefill)
        .unwrap()
        .finish(None, Outcome::Submitted);
    assert!(ledger.begin_decode(true, 8, start).is_none());
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot["waves_started"], 4);
    assert_eq!(snapshot["overflow_waves"], 1);
    for row in &records(&snapshot)[..2] {
        let preparation = &row["preparation"];
        assert_eq!(preparation["reached_dispatch"], false);
        assert_eq!(preparation["executor_returned_before_dispatch_ns"], 7);
        for missing in [
            "prepared_width",
            "executor_to_dispatch_ns",
            "resource_prepare_ns",
            "dispatch_tail_ns",
            "tokens_and_extension",
        ] {
            assert!(preparation[missing].is_null(), "{missing}");
        }
        assert_eq!(preparation["resource_prepare_attempt"]["samples"], 1);
        assert_eq!(preparation["resource_prepare_attempt"]["total_ns"], 0);
        assert!(row["host_encode_submit_ns"].is_null());
        assert!(row["attempts"].as_array().unwrap().is_empty());
    }
    assert!(records(&snapshot)[2]["preparation"].is_null());
}

#[test]
fn reusable_cold_path_decode_preparation_reuses_metrics_and_preserves_unwind() {
    let metrics = AtomicDurationMetrics::default();
    let preparation = DecodePreparation::new(Instant::now(), 1);
    for _ in 0..2 {
        let _timing = PreparationTimer::existing(
            Some(metrics.start()),
            Some(&preparation),
            PreparationStage::StepAdmission,
        );
    }
    let record = serde_json::to_value(preparation.into_record()).unwrap();
    assert_eq!(
        record["step_admission"]["samples"],
        metrics.snapshot()["samples"]
    );
    assert_eq!(
        record["step_admission"]["total_ns"],
        metrics.snapshot()["total_ns"]
    );
    assert_eq!(metrics.snapshot()["samples"], 2);
    let disabled = PreparationTimer::start(None, PreparationStage::CanonicalParticipants);
    assert!(disabled.started.is_none());
    drop(disabled);

    let ledger = Ledger::default();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let wave = ledger.begin_decode(true, 1, Instant::now()).unwrap();
        let _timing = PreparationTimer::existing(
            Some(metrics.start()),
            wave.preparation(),
            PreparationStage::StepAdmission,
        );
        panic!("preparation unwinds before dispatch");
    }))
    .is_err());
    assert_eq!(metrics.snapshot()["samples"], 3);
    assert_eq!(ledger.snapshot()["unfinalized_waves"], 1);
    assert!(records(&ledger.snapshot()).is_empty());
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
    first.record(
        SubmissionWaveDispatchStage::ContractValidateAndReserve,
        Duration::from_nanos(2),
    );
    first.record(
        SubmissionWaveDispatchStage::ContractValidateAndReserve,
        Duration::from_nanos(3),
    );
    first.record(
        SubmissionWaveDispatchStage::SegmentFreshAuthorityAndWindows,
        Duration::ZERO,
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
    retry.record(
        SubmissionWaveDispatchStage::ContractValidateAndReserve,
        Duration::from_nanos(7),
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
    assert_eq!(
        rows[0]["attempts"][0]["contract_validate_reserve"],
        json!({"samples": 2, "total_ns": 5, "overflowed": false})
    );
    assert_eq!(
        rows[0]["attempts"][0]["segment_fresh_authority_and_windows"],
        json!({"samples": 1, "total_ns": 0, "overflowed": false})
    );
    assert_eq!(rows[0]["attempts"][0]["repeated_callback"], false);
    assert_eq!(rows[0]["attempts"][1]["ordinal"], 1);
    assert_eq!(rows[0]["attempts"][1]["provider_encode_submit_ns"], 17);
    assert_eq!(rows[0]["attempts"][1]["provider_node_encode_ns"], 11);
    assert!(rows[0]["attempts"][1]["enqueue_commands_ns"].is_null());
    assert_eq!(
        rows[0]["attempts"][1]["contract_validate_reserve"],
        json!({"samples": 1, "total_ns": 7, "overflowed": false})
    );
    assert!(rows[0]["attempts"][1]["segment_fresh_authority_and_windows"].is_null());
    assert_eq!(rows[0]["attempts"][1]["repeated_callback"], false);
    assert_eq!(rows[1]["ordinal"], 1);
    assert_eq!(rows[1]["phase"], "prefill");
    assert_eq!(rows[1]["attempts"][0]["ordinal"], 0);
    assert_eq!(rows[1]["attempts"][0]["disposition"], "before_decision");
    assert!(rows[1]["attempts"][0]["program_id"].is_null());
    assert!(rows[1]["attempts"][0]["provider_encode_submit_ns"].is_null());
    for (_, field) in SELECTED_STAGES {
        assert!(rows[1]["attempts"][0][field].is_null(), "{field}");
    }
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
    assert_eq!(
        metrics.wave_timing.contract_validate_reserve.snapshot()["total_ns"],
        12
    );
    assert_eq!(
        metrics
            .decode_wave_timing
            .contract_validate_reserve
            .snapshot()["samples"],
        3
    );
    assert_eq!(
        metrics
            .prefill_wave_timing
            .contract_validate_reserve
            .snapshot()["samples"],
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
        for (index, (stage, _)) in SELECTED_STAGES.into_iter().enumerate() {
            sink.record(stage, Duration::ZERO);
            sink.record(stage, Duration::from_nanos(index as u64 + 1));
        }
        sink.record_device_submission(DeviceSubmissionStage::EnqueueCommands, Duration::ZERO);
        sink.record_device_submission(
            DeviceSubmissionStage::EnqueueCommands,
            Duration::from_nanos(5),
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::GraphLaunch,
            Duration::from_nanos(7),
            Some(Duration::from_nanos(3)),
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::GraphLaunch,
            Duration::from_nanos(11),
            None,
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::EagerCommand,
            Duration::from_nanos(13),
            None,
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::DeferredSnapshotReadback,
            Duration::from_nanos(17),
            None,
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
    for (index, (_, field)) in SELECTED_STAGES.into_iter().enumerate() {
        assert_eq!(
            attempt[field],
            json!({"samples": 2, "total_ns": index as u64 + 1, "overflowed": false}),
            "{field}"
        );
    }
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
fn reusable_cold_path_stage_totals_saturate_without_losing_overflow_evidence() {
    assert!(StageDurationAccumulator::default().into_record().is_none());

    let exact_limit = StageDurationAccumulator::default();
    exact_limit.record(Duration::from_nanos(u64::MAX));
    assert_eq!(
        serde_json::to_value(exact_limit.into_record()).unwrap(),
        json!({"samples": 1, "total_ns": u64::MAX, "overflowed": false})
    );

    let total_overflow = StageDurationAccumulator::default();
    total_overflow.record(Duration::from_nanos(u64::MAX));
    total_overflow.record(Duration::from_nanos(1));
    total_overflow.record(Duration::ZERO);
    assert_eq!(
        serde_json::to_value(total_overflow.into_record()).unwrap(),
        json!({"samples": 3, "total_ns": u64::MAX, "overflowed": true})
    );

    let duration_overflow = StageDurationAccumulator::default();
    duration_overflow.record(Duration::from_nanos(u64::MAX) + Duration::from_nanos(1));
    assert_eq!(
        serde_json::to_value(duration_overflow.into_record()).unwrap(),
        json!({"samples": 1, "total_ns": u64::MAX, "overflowed": true})
    );

    let samples_overflow = StageDurationAccumulator::default();
    samples_overflow.samples.store(u64::MAX, Ordering::Relaxed);
    samples_overflow.record(Duration::ZERO);
    assert_eq!(
        serde_json::to_value(samples_overflow.into_record()).unwrap(),
        json!({"samples": u64::MAX, "total_ns": 0, "overflowed": true})
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
    let disabled = invocation_preparation::DisabledTiming;
    let observer = AttemptObservation::new(&disabled, None, 1, None, None);
    for (stage, _) in SELECTED_STAGES {
        observer.record(stage, Duration::from_nanos(1));
    }
    let disabled_record =
        serde_json::to_value(observer.finish(0, Default::default(), Outcome::Submitted, None))
            .unwrap();
    for (_, field) in SELECTED_STAGES {
        assert!(disabled_record[field].is_null(), "{field}");
    }
    assert_eq!(disabled_record["repeated_callback"], false);

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
    for (_, field) in SELECTED_STAGES {
        assert!(
            ledger.snapshot()["records"][0]["attempts"][0][field].is_null(),
            "{field}"
        );
    }
}
