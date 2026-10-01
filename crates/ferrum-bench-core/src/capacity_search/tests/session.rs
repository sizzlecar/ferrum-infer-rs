use super::*;

fn with_warmup() -> CapacityContract {
    let mut value = contract();
    let mut sample = value.workload.samples[0].clone();
    sample.phase = BenchmarkPhase::Warmup;
    value.workload.samples.insert(0, sample);
    let digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&value.workload.samples).unwrap())
    );
    value.identity.ordered_workload_sha256 = digest.clone();
    value.workload.selection_sha256 = digest;
    value
}
fn block(
    planned: &PlannedCapacityRun,
    ordinal: u32,
    previous: Option<String>,
) -> CapacitySessionBlock {
    CapacitySessionBlock {
        session: CapacityServerSession {
            session_id: "held-child-session".into(),
            process_id: 123,
            process_birth: CapacityProcessBirth::LinuxBootTicks {
                boot_id: "boot".into(),
                start_ticks: 1,
            },
            server_binary_sha256: "a".repeat(64),
            effective_configuration_sha256: "a".repeat(64),
            process_configuration_sha256: "b".repeat(64),
            endpoint: "http://localhost:1".into(),
            engine_instance: None,
            started_unix_ns: 2,
        },
        key: planned.key.clone(),
        block_ordinal: ordinal,
        checked_live_unix_ns: u64::from(ordinal) * 10_000_000_000 - 1,
        previous_evidence_sha256: previous,
    }
}
fn first(search: &CapacitySearch) -> CapacityRunEvidence {
    let planned = next(search);
    let mut value = evidence(search, &planned, 0);
    value.session = Some(block(&planned, 1, None));
    value.warmup.expected = 1;
    value.warmup.completed = 1;
    value
}
fn reused(search: &CapacitySearch, previous: &CapacityRunEvidence) -> CapacityRunEvidence {
    let planned = next(search);
    let digest = capacity_evidence_sha256(previous).unwrap();
    let mut value = evidence(search, &planned, 1);
    let authorization = search
        .authorize_session_block(&block(&planned, 2, Some(digest.clone())), Some(&digest))
        .unwrap();
    assert!(!authorization.executes_warmup());
    value.session = Some(authorization.block().clone());
    value.warmup.acquisition = authorization.acquisition().clone();
    value
}

#[test]
fn capacity_session_preserves_valid_performance_fail_and_replays_original_warmup() {
    let mut contract = with_warmup();
    // The original fixture observes TTFT=1ms; equality meets the SLO.
    contract.slo.ttft_ms = 0.5;
    let mut search = CapacitySearch::new(contract.clone()).unwrap();
    let one = first(&search);
    let assessment = search.record(one.clone()).unwrap();
    assert_eq!(assessment.status, SloStatus::Fail);
    assert_eq!(
        assessment.acquisition_disposition,
        CapacityAcquisitionDisposition::Complete
    );
    let two = reused(&search, &one);
    assert_eq!(two.warmup.expected, 0);
    assert_eq!(two.warmup.completed, 0);
    assert!(evaluate_run(&contract, &next(&search), &two).is_err());
    assert_eq!(search.record(two.clone()).unwrap().status, SloStatus::Fail);
    let mut replay = CapacitySearch::new(contract).unwrap();
    replay.record(one).unwrap();
    replay.record(two).unwrap();
    assert_eq!(replay.report().runs.len(), 2);
}

#[test]
fn capacity_session_rejects_forged_origin_changed_process_and_fake_warmup_counts() {
    let mut search = CapacitySearch::new(with_warmup()).unwrap();
    let one = first(&search);
    search.record(one.clone()).unwrap();
    let two = reused(&search, &one);
    let receipt = two.session.as_ref().unwrap();
    let digest = capacity_evidence_sha256(&one).unwrap();
    assert!(search
        .authorize_session_block(receipt, Some(&"0".repeat(64)))
        .is_err());
    for mutation in 0..5 {
        let mut changed = receipt.clone();
        match mutation {
            0 => changed.session.process_id += 1,
            1 => {
                changed.session.process_birth = CapacityProcessBirth::LinuxBootTicks {
                    boot_id: "boot".into(),
                    start_ticks: 2,
                }
            }
            2 => changed.session.process_configuration_sha256 = "c".repeat(64),
            3 => changed.session.engine_instance = Some("new-engine".into()),
            _ => changed.previous_evidence_sha256 = None,
        }
        assert!(search
            .authorize_session_block(&changed, Some(&digest))
            .is_err());
    }
    let mut fake = two.clone();
    fake.warmup.expected = 1;
    fake.warmup.completed = 1;
    assert!(search.record(fake).is_err());
    search.record(two).unwrap();
}

#[test]
fn capacity_session_never_reuses_failed_warmup_or_undrained_server() {
    for undrained in [false, true] {
        let mut search = CapacitySearch::new(with_warmup()).unwrap();
        let mut one = first(&search);
        if undrained {
            one.queue.last_mut().unwrap().active_requests = 1;
        } else {
            one.warmup.completed = 0;
            one.warmup.errored = 1;
        }
        assert_ne!(
            search.record(one.clone()).unwrap().acquisition_disposition,
            CapacityAcquisitionDisposition::Complete
        );
        let digest = capacity_evidence_sha256(&one).unwrap();
        assert!(search
            .authorize_session_block(
                &block(&next(&search), 2, Some(digest.clone())),
                Some(&digest)
            )
            .is_err());
    }
}

#[test]
fn capacity_session_missing_rows_and_invalid_raw_timing_are_acquisition_failures() {
    let search = CapacitySearch::new(with_warmup()).unwrap();
    let planned = next(&search);
    let mut missing = first(&search);
    missing.arrivals.pop();
    assert_eq!(
        evaluate_run(search.contract(), &planned, &missing)
            .unwrap()
            .acquisition_disposition,
        CapacityAcquisitionDisposition::IncompleteEvidence
    );
    let mut malformed = first(&search);
    malformed.queue[0].at_seconds = f64::NAN;
    assert_eq!(
        evaluate_run(search.contract(), &planned, &malformed)
            .unwrap()
            .acquisition_disposition,
        CapacityAcquisitionDisposition::ProtocolFailure
    );
}

#[test]
fn capacity_session_confirmation_cannot_reuse_discovery_process() {
    let contract = with_warmup();
    let search = CapacitySearch::new(contract.clone()).unwrap();
    let one = first(&search);
    let assessment = evaluate_run(&contract, &next(&search), &one).unwrap();
    let mut history = crate::capacity_search::session::SessionHistory::default();
    history.record(&contract, &one, &assessment).unwrap();
    let digest = capacity_evidence_sha256(&one).unwrap();
    let mut next_block = block(&next(&search), 2, Some(digest.clone()));
    next_block.key.phase = CapacityPhase::Confirmation;
    assert!(history
        .authorize(&contract, &next_block, Some(&digest))
        .is_err());
    next_block.session.session_id = "fresh-confirmation".into();
    next_block.session.process_id += 1;
    next_block.block_ordinal = 1;
    next_block.previous_evidence_sha256 = None;
    assert!(history
        .authorize(&contract, &next_block, None)
        .unwrap()
        .executes_warmup());
}

#[test]
fn capacity_session_binds_ferrum_original_engine_epoch() {
    let mut contract = with_warmup();
    contract.queue_observation_source = QueueObservationSource::ServerAdmissionV1;
    let mut search = CapacitySearch::new(contract).unwrap();
    let mut raw = first(&search);
    raw.queue_observation_source = QueueObservationSource::ServerAdmissionV1;
    raw.queue.clear();
    raw.server_queue_attempts.push(ServerQueueAttempt {
        request_started_seconds: 1.25,
        response_completed_seconds: 1.251,
        observation: Ok(ferrum_types::ExecutorQueueObservation {
            schema_version: 1,
            engine_instance: "actual-engine".into(),
            observed_at_ns: 1,
            waiting_requests: 0,
            active_prefill_sequences: 0,
            active_decode_sequences: 0,
            preempted_requests: 0,
            oldest_waiting_ingress_age_ns: None,
            oldest_unfinished_ingress_age_ns: None,
        }),
    });
    assert!(
        search.record(raw.clone()).is_err(),
        "Ferrum session requires the actual epoch"
    );
    raw.session.as_mut().unwrap().session.engine_instance = Some("other-engine".into());
    assert!(
        search.record(raw.clone()).is_err(),
        "checkpoint and raw observation must agree"
    );
    raw.session.as_mut().unwrap().session.engine_instance = Some("actual-engine".into());
    assert_eq!(
        search.record(raw).unwrap().status,
        SloStatus::Unknown,
        "an exact epoch does not manufacture missing queue-window coverage"
    );
}
