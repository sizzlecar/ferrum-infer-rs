//! Original typed deferral -> raw FIFO -> private V2 retirement -> source replay.
//! Virtual clocks test attribution, not measured backend costs.
use super::*;
use ferrum_interfaces::model_executor::{
    ExecutorExecutionCapacityDeferral, ExecutorExecutionCapacityStage, ExecutorExecutionDeferral,
};
use ferrum_interfaces::vnext::{
    CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
    DeviceCapacityPressure, DeviceCapacityPressureScope,
};
use ferrum_interfaces::ExecutorAdmissionEpochs;

pub(super) fn capacity() -> ExecutorExecutionDeferral {
    let observed =
        CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 7).unwrap();
    ExecutorExecutionCapacityDeferral::from_backing_pressure(
        ExecutorAdmissionEpochs::new(NonZeroU64::new(19).unwrap(), 3, 5),
        CapacityWaitCondition::from_observation(19, vec![observed]).unwrap(),
        DeviceCapacityPressure::new(
            DeviceCapacityPressureScope::PlanBudget,
            "original-capacity".into(),
            1,
            1,
            1,
            1,
            1,
        )
        .unwrap()
        .into(),
        ExecutorExecutionCapacityStage::StepAdmission,
    )
    .unwrap()
    .into()
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    GenericDeferred,
    Lost,
    Cancelled,
    Unknown,
}

#[path = "no_submission/private_capture.rs"]
mod private_capture;

fn defer(f: &Automatic, fault: Fault) -> Arc<CostCalibrationCapture> {
    defer_original(f, fault, true)
}
fn defer_original(f: &Automatic, fault: Fault, live: bool) -> Arc<CostCalibrationCapture> {
    let at = f.clock.now_ns().unwrap() + 100;
    f.clock.set(at);
    let w = wave("fixture.no-submission");
    let row = &w.actual.rows[0];
    let ticket = live.then(|| {
        f.runtime
            .reserve_live_ticket(Some(at))
            .expect("original offer")
    });
    let mut call = EngineCostCall::begin(
        &f.runtime.ids,
        f.clock.clone(),
        f.runtime.sink.clone(),
        EngineCostCallSpec {
            identity: identity(),
            participants: vec![CostObservationParticipant {
                request_id: row.request_id.clone(),
                owner_incarnation: row.owner_incarnation,
                work_generation: row.work_generation,
                input_index: row.input_index,
                output_policy_signature: Some([6; 32]),
                host_features: Some(w.host),
            }],
            prepare_started_at_ns: Some(at),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 8,
                max_retained_rows: 128,
            },
        },
    )
    .unwrap()
    .with_live_ticket(ticket);
    // This uses the actual product hook, not a manually enabled recorder.
    assert_eq!(
        call.recorder.route_diagnostic().is_some(),
        tracing::enabled!(
            target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
            tracing::Level::DEBUG
        )
    );
    let capture = Arc::new(CostCalibrationCapture::default());
    call.attach_calibration_capture(Arc::clone(&capture));
    {
        let mut context = call.context().unwrap();
        f.clock.set(at + 3);
        if matches!(fault, Fault::GenericDeferred) {
            context.finish_call(ObservedCallOutcome::Deferred);
        } else {
            context.finish_capacity_deferred(&capacity());
        }
        // The executor decision, context return and original seal are distinct
        // real clock reads. A later worker cannot rewrite any of them.
        f.clock.set(at + 5);
    }
    match fault {
        Fault::Lost => call.recorder.note_lost(1),
        Fault::Cancelled => call.host_cancelled(&row.request_id),
        Fault::Unknown => call.reject(CostCallRejection::ActualEvidenceUnknown),
        _ => {}
    }
    f.clock.set(at + 8);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    f.clock.set(at + 20);
    f.runtime.consume_samples();
    capture
}

fn v2(offers: usize) -> Automatic {
    Automatic::with_population(
        offers,
        SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2,
    )
}

#[tokio::test]
async fn no_submission_v2_original_quota_publishes_and_profile13_replays() {
    for debug in [false, true] {
        tracing::subscriber::with_default(super::diagnostics::RuntimeDebug(debug), || {
            original_quota_publishes_and_replays();
        });
    }
}

fn original_quota_publishes_and_replays() {
    let f = v2(9);
    let capture = defer(&f, Fault::None);
    let CostCalibrationStatus::Complete(legacy) = capture.status() else {
        panic!("original FIFO acknowledgement")
    };
    assert!(matches!(
        legacy.as_ref(),
        CostCalibrationResult::Rejected(CostCallRejection::NoPhysicalWave)
    ));
    assert!(
        capture.host_stages().is_none(),
        "no invented physical wave or host settlement"
    );
    let audit = f.live().audit();
    assert_eq!(
        (
            audit.population.issued,
            audit.population.retired,
            audit.population.no_submission
        ),
        (1, 1, 1)
    );
    assert!(!audit.population.failed);
    assert!(audit.population.first_ticket_failure.is_none());
    assert_eq!(f.runtime.sink.stats().raw_no_submission, 1);
    assert_eq!(f.runtime.sink.stats().raw_resolution_failed, 0);
    f.phase(8, 0); // remaining original discovery offers
    for _ in 0..3 {
        defer(&f, Fault::None);
        f.phase(8, 0); // unchanged eight actual numerical members per phase
    }
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 1, "{audit:#?}");
    let json = serde_json::to_value(&audit).unwrap();
    for phase in json["automatic"]["last_published_population"]["phase_populations"]
        .as_array()
        .unwrap()
    {
        assert_eq!(phase["issued"], 9);
        assert_eq!(phase["retired"], 9);
        assert_eq!(phase["eligible_route"], 8);
        assert_eq!(phase["no_submission"], 1);
        assert!(
            phase["first_ticket_failure"].is_null(),
            "legacy Composite is not a failed ticket"
        );
    }
    let source = audit
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap()
        .source
        .path;
    let bytes = fs::read(&source).unwrap();
    let lines: Vec<serde_json::Value> = bytes
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    let attempts: Vec<_> = lines
        .iter()
        .filter(|r| r["kind"] == "not_submitted")
        .collect();
    assert_eq!(attempts.len(), 3);
    for record in &attempts {
        let a = &record["attempt"];
        assert!(a["returned_at_ns"].as_u64().unwrap() < a["call_returned_at_ns"].as_u64().unwrap());
        assert!(
            a["call_returned_at_ns"].as_u64().unwrap() < a["finalized_at_ns"].as_u64().unwrap()
        );
    }
    let profile = f.directory.join("no-submission-profile13.json");
    let limits = file::CostProfileLoadLimits::default();
    file::export_structured_profile_v13(
        &source,
        sha2::Sha256::digest(&bytes).into(),
        &profile,
        0,
        &limits,
    )
    .unwrap();
    let fingerprint = f.runtime.snapshot().unwrap().fingerprint().clone();
    let closing = lines.last().unwrap()["closing"]["wall_unix_ns"]
        .as_u64()
        .unwrap();
    let imported = file::load_structured_profile_v13(
        &profile,
        &fingerprint,
        &limits,
        file::ProfileLoadClock {
            wall_unix_ns: Some(closing + 10),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 1000,
        },
    )
    .unwrap();
    assert_eq!(imported.offered_attempts, 27);
    assert_eq!(
        imported.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [8; 3]
    );
    // The new evidence remains Unknown to each old population declaration.
    let header: file::StructuredServiceHeaderV6 = serde_json::from_value(lines[0].clone()).unwrap();
    let open: file::StructuredServiceRecordV6 = serde_json::from_value(lines[1].clone()).unwrap();
    let no_submit: file::StructuredServiceRecordV6 =
        serde_json::from_value(attempts[0].clone()).unwrap();
    for policy in [
        SloCalibrationRoutePopulationV1::AllAttempts,
        SloCalibrationRoutePopulationV1::WarmOrGraphDisabledV1,
    ] {
        let mut declaration = header.declaration.clone();
        declaration.route_population = policy;
        let old = file::StructuredServiceHeaderV6::new(
            header.capture_identity,
            header.generation,
            header.fingerprint.clone(),
            header.producer.clone(),
            header.opening,
            declaration,
            header.maximum_file_bytes,
        )
        .unwrap();
        assert_ne!(old.protocol, header.protocol);
        let mut collector = file::StructuredServiceCollectorV6::new(old, limits.clone()).unwrap();
        collector.push(&open).unwrap();
        assert!(collector.push(&no_submit).is_err());
    }
    // Every wire identity/clock/order mutation still faces original validation.
    let file::StructuredServiceRecordV6::NotSubmitted { attempt } = no_submit else {
        unreachable!()
    };
    for fault in 0..5 {
        let mut changed = serde_json::to_value(&attempt).unwrap();
        match fault {
            0 => changed["participants"][0]["work_generation"] = 0.into(),
            1 => changed["participants"][0]["input_index"] = 99.into(),
            2 => changed["issued_at_ns"] = 0.into(),
            3 => changed["call_returned_at_ns"] = 0.into(),
            _ => changed["finalized_at_ns"] = 0.into(),
        }
        let changed: file::StructuredServiceNoSubmissionV6 =
            serde_json::from_value(changed).unwrap();
        assert!(changed
            .validate_settlement(&header.fingerprint, header.opening.monotonic_ns)
            .is_err());
    }
    f.runtime.consume_samples();
    let previous = f.runtime.snapshot().unwrap();
    let feedback_before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    defer(&f, Fault::None);
    let feedback_after = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        feedback_after.no_submission_observations,
        feedback_before.no_submission_observations + 1
    );
    assert_eq!(
        feedback_after.uncomparable_observations,
        feedback_before.uncomparable_observations
    );
    assert!(feedback_after.revoked.is_none());
    assert!(previous.current());
    let w = wave("fixture.outside-route");
    let query = StructuredQueryV2::exact(
        StructuredInputV2::from_actual(&w.prepared.exact, &w.prepared.selected, &w.prepared.recipe)
            .unwrap(),
    );
    previous
        .audit_structured_query_v2(&query, f.clock.now_ns().unwrap())
        .unwrap();
}

#[tokio::test]
async fn no_submission_v2_cannot_replace_numeric_minimum_or_old_failure_population() {
    for debug in [false, true] {
        tracing::subscriber::with_default(super::diagnostics::RuntimeDebug(debug), || {
            preserves_numeric_minimum_and_failures();
        });
    }
}

fn preserves_numeric_minimum_and_failures() {
    let f = v2(8);
    defer(&f, Fault::None);
    f.phase(8, 0);
    defer(&f, Fault::None);
    f.phase(7, 0); // complete first Fit fails its unchanged eight-member minimum
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert!(audit.failed_generations > 0);
    let failed = serde_json::to_value(&audit).unwrap();
    let fit = &failed["automatic"]["history"][0]["phase_populations"][0];
    assert_eq!(fit["issued"], 8);
    assert_eq!(fit["eligible_route"], 7);
    assert_eq!(fit["no_submission"], 1);
    f.runtime.consume_samples();
    assert_eq!(
        f.live().audit().population.phase,
        3,
        "an eight-offer phase cannot replace nine discovery offers"
    );
    for policy in [
        SloCalibrationRoutePopulationV1::AllAttempts,
        SloCalibrationRoutePopulationV1::WarmOrGraphDisabledV1,
    ] {
        let f = Automatic::with_population(9, policy);
        defer(&f, Fault::None);
        assert_eq!(f.runtime.sink.stats().raw_no_submission, 0);
        assert!(f.live().audit().failed_generations > 0);
    }
    for fault in [
        Fault::GenericDeferred,
        Fault::Lost,
        Fault::Cancelled,
        Fault::Unknown,
    ] {
        let f = v2(9);
        defer(&f, fault);
        assert_eq!(f.runtime.sink.stats().raw_no_submission, 0);
        assert_eq!(f.runtime.sink.stats().raw_resolution_failed, 1);
        assert!(f.live().audit().failed_generations > 0);
        assert_eq!(f.live().audit().qualified_publications, 0);
    }
}

#[tokio::test]
async fn no_submission_shadow_consumes_original_quota_and_new_generation_requires_all_phases() {
    let f = v2(9);
    defer(&f, Fault::None);
    f.phase(8, 0);
    let next_owner = "fixture.no-submission.new-owner";
    defer(&f, Fault::None);
    for _ in 0..8 {
        record_route(&f.runtime, &f.clock, wave(next_owner), false, false).unwrap();
        f.runtime.consume_samples();
    }
    let failed = f.live().audit();
    assert_eq!(failed.failed_generations, 1);
    assert_eq!(failed.qualified_publications, 0);
    assert_eq!(failed.population.no_submission, 1);
    assert_eq!(
        (failed.population.issued, failed.population.retired),
        (9, 9)
    );
    f.runtime.consume_samples();
    let audit = f.live().audit();
    assert_eq!(
        (audit.population.generation, audit.population.phase),
        (2, 0)
    );
    assert_eq!(audit.population.issued, 0);
    let json = serde_json::to_value(audit).unwrap();
    let original = &json["automatic"]["discovery_origin"]["population"];
    assert_eq!(original["issued"], 9);
    assert_eq!(original["eligible_route"], 8);
    assert_eq!(original["no_submission"], 1);
    assert_eq!(original["failed"], false);
    for phase in 0..3 {
        defer(&f, Fault::None);
        for _ in 0..8 {
            record_route(&f.runtime, &f.clock, wave(next_owner), false, false).unwrap();
            f.runtime.consume_samples();
        }
        assert_eq!(
            f.live().audit().qualified_publications,
            u64::from(phase == 2)
        );
    }
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.offered_samples, 27);
    assert_eq!(receipt.recorded_samples, 24);
}
