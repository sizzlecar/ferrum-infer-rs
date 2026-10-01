use super::*;

fn server_case() -> (CapacitySearch, PlannedCapacityRun, CapacityRunEvidence) {
    let mut config = contract();
    config.queue_observation_source = QueueObservationSource::ServerAdmissionV1;
    let search = CapacitySearch::new(config).unwrap();
    let planned = next(&search);
    let mut raw = evidence(&search, &planned, 0);
    raw.queue_observation_source = QueueObservationSource::ServerAdmissionV1;
    raw.queue.clear();
    raw.server_queue_attempts = (0..=10)
        .map(|index| ServerQueueAttempt {
            request_started_seconds: if index == 0 {
                -0.01
            } else {
                index as f64 * 0.125
            },
            response_completed_seconds: if index == 0 {
                0.0
            } else {
                index as f64 * 0.125 + 0.001
            },
            observation: Ok(ferrum_types::ExecutorQueueObservation {
                schema_version: 1,
                engine_instance: "instance".into(),
                observed_at_ns: 7_000_000_000 + index * 125_000_000,
                waiting_requests: 0,
                active_prefill_sequences: 0,
                active_decode_sequences: 0,
                preempted_requests: 0,
                oldest_waiting_ingress_age_ns: None,
                oldest_unfinished_ingress_age_ns: None,
            }),
        })
        .collect();
    (search, planned, raw)
}

#[test]
fn capacity_server_queue_original_clock_and_brackets_cover_send_and_drain() {
    let (search, planned, raw) = server_case();
    let report = evaluate_run(search.contract(), &planned, &raw).unwrap();
    assert_eq!(report.status, SloStatus::Pass, "{:?}", report.issues);
    assert_eq!(report.unfinished_requests_slope_per_second, Some(0.0));
    let mut restarted = raw.clone();
    restarted.server_queue_attempts[5]
        .observation
        .as_mut()
        .unwrap()
        .engine_instance = "new-instance".into();
    assert_eq!(
        evaluate_run(search.contract(), &planned, &restarted)
            .unwrap()
            .status,
        SloStatus::Unknown
    );
    let mut stale = raw;
    stale.server_queue_attempts[5]
        .observation
        .as_mut()
        .unwrap()
        .observed_at_ns = 1;
    assert_eq!(
        evaluate_run(search.contract(), &planned, &stale)
            .unwrap()
            .status,
        SloStatus::Unknown
    );
}

#[test]
fn capacity_server_queue_missing_age_and_coverage_cannot_be_replaced_by_client_backlog() {
    let (search, planned, raw) = server_case();
    let mut unavailable = raw.clone();
    for attempt in &mut unavailable.server_queue_attempts {
        attempt.observation = Err(ServerQueueFailure::Unavailable);
    }
    let report = evaluate_run(search.contract(), &planned, &unavailable).unwrap();
    assert_eq!(report.status, SloStatus::Unknown);
    assert_eq!(report.oldest_age_slope_ms_per_second, None);
    let mut gap = raw.clone();
    gap.server_queue_attempts[4].observation = Err(ServerQueueFailure::Timeout);
    gap.server_queue_attempts[5].observation = Err(ServerQueueFailure::Runtime("busy".into()));
    assert_eq!(
        evaluate_run(search.contract(), &planned, &gap)
            .unwrap()
            .status,
        SloStatus::Unknown
    );
    let mut missing_age = raw;
    missing_age.server_queue_attempts[3]
        .observation
        .as_mut()
        .unwrap()
        .waiting_requests = 1;
    assert_eq!(
        evaluate_run(search.contract(), &planned, &missing_age)
            .unwrap()
            .status,
        SloStatus::Unknown
    );
}

#[test]
fn capacity_server_queue_growth_and_preempted_work_cannot_disappear_at_drain() {
    let (search, planned, mut raw) = server_case();
    for (index, attempt) in raw.server_queue_attempts.iter_mut().enumerate().skip(1) {
        let q = attempt.observation.as_mut().unwrap();
        q.preempted_requests = index as u32;
        q.oldest_unfinished_ingress_age_ns = Some(index as u64 * 125_000_000);
    }
    let report = evaluate_run(search.contract(), &planned, &raw).unwrap();
    assert_eq!(report.status, SloStatus::Fail);
    assert!(report.unfinished_requests_slope_per_second.unwrap() > 0.0);
    assert!(report.oldest_age_slope_ms_per_second.unwrap() > 0.0);
}
