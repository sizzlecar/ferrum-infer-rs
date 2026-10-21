use super::*;

fn fact(now: Instant) -> RequestStoppingObservation {
    RequestStoppingObservation {
        admitted_at: now,
        lifecycle: RequestLifecycleId {
            producer: 1,
            admission: 1,
        },
        revision: 2,
        observed_at: now,
        rules: Arc::new(
            EffectiveStoppingRules::new(b"eos".to_vec(), StoppingMechanism::Observed, 64).unwrap(),
        ),
        cap: 20,
        generated_tokens: 8,
        verified_survival_through: 7,
        terminal: Some(RequestTerminalReason::Eos),
        completion: None,
    }
}

fn paused_collector(
    capacity: usize,
    bytes: usize,
) -> (Observer, crossbeam_channel::Receiver<Envelope>) {
    let (wake, _) = crossbeam_channel::bounded(1);
    let (sender, receiver) = crossbeam_channel::bounded(capacity);
    let shared = Arc::new(Shared {
        config: StoppingCollectorConfig {
            queue_capacity: capacity,
            max_queue_bytes: bytes,
            ..Default::default()
        },
        counters: Counters::default(),
        closed: AtomicBool::new(false),
        publishing: AtomicUsize::new(0),
        records: AtomicUsize::new(0),
        bytes: AtomicUsize::new(0),
        producer: AtomicU64::new(0),
        highest_admission: AtomicU64::new(0),
        admissions: (0..StoppingConfig::default().window_requests)
            .map(|_| ArcSwapOption::empty())
            .collect(),
        wake,
    });
    (Observer { shared, sender }, receiver)
}

#[test]
fn saturation_and_byte_caps_release_all_reservations_without_waiting() {
    let observation = fact(Instant::now());
    let (observer, receiver) = paused_collector(1, 4096);
    assert_eq!(
        observer.try_record(&observation),
        RequestObservationDelivery::Recorded
    );
    assert_eq!(
        observer.try_record(&observation),
        RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull)
    );
    assert_eq!(observer.shared.stats().queued_records, 1);
    drop(receiver.recv().unwrap());
    assert_eq!(observer.shared.stats().queued_records, 0);
    assert_eq!(observer.shared.stats().queued_bytes, 0);
    let (observer, _receiver) = paused_collector(2, std::mem::size_of::<Envelope>() + 2);
    assert_eq!(
        observer.try_record(&observation),
        RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull)
    );
    assert_eq!(observer.shared.stats().dropped_bytes, 1);
    assert_eq!(observer.shared.stats().queued_records, 0);
    assert_eq!(observer.shared.stats().queued_bytes, 0);
}

#[test]
fn disconnected_or_closed_collection_does_not_retain_payload_credit() {
    let observation = fact(Instant::now());
    let (observer, receiver) = paused_collector(1, 4096);
    drop(receiver);
    assert_eq!(
        observer.try_record(&observation),
        RequestObservationDelivery::Dropped(RequestObservationDrop::Closed)
    );
    assert_eq!(observer.shared.stats().queued_bytes, 0);
    observer.shared.stop();
    assert_eq!(
        observer.try_record(&observation),
        RequestObservationDelivery::Dropped(RequestObservationDrop::Closed)
    );
    assert!(observer.shared.drained());
}

#[test]
fn full_queue_preserves_last_unseen_admission_watermark_without_allocating_a_placeholder() {
    let now = Instant::now();
    let (observer, receiver) = paused_collector(1, 4096);
    let first = fact(now);
    assert_eq!(
        observer.try_record(&first),
        RequestObservationDelivery::Recorded
    );
    let mut lost = first.clone();
    lost.lifecycle.admission = 2;
    assert_eq!(
        observer.try_record(&lost),
        RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull)
    );
    let mut window = StoppingWindow::new(StoppingConfig {
        min_risk_set: 1,
        ..Default::default()
    })
    .unwrap();
    window.observe(&receiver.recv().unwrap().observation, now);
    observer.shared.advance_window(&mut window);
    assert_eq!(window.retained_requests(), 1);
    assert_eq!(window.stats().unobserved_lifetimes, 1);
    assert_eq!(observer.shared.stats().queued_records, 0);
}

#[test]
fn admission_timestamp_precedes_watermark_and_survives_a_dropped_initial_fact() {
    let now = Instant::now();
    let (observer, receiver) = paused_collector(1, 4096);
    let mut completed = fact(now - Duration::from_secs(40));
    completed.completion = Some(RequestCompletionOutcome::Succeeded);
    assert_eq!(
        observer.try_record(&completed),
        RequestObservationDelivery::Recorded
    );
    let recent = RequestLifecycleId {
        producer: 1,
        admission: 2,
    };
    observer.record_admission(recent, now);
    let mut lost = fact(now);
    lost.lifecycle = recent;
    assert_eq!(
        observer.try_record(&lost),
        RequestObservationDelivery::Dropped(RequestObservationDrop::QueueFull)
    );
    let mut window = StoppingWindow::new(StoppingConfig {
        min_risk_set: 1,
        ..Default::default()
    })
    .unwrap();
    window.observe(&receiver.recv().unwrap().observation, now);
    observer.shared.advance_window(&mut window);
    let query = CompletionQuery {
        rules: &completed.rules,
        cap: completed.cap,
        maturity: Duration::from_secs(30),
        now,
    };
    let current = window.snapshot().completion_forecast(&query);
    assert_eq!(current.samples, 1);
    assert_eq!(
        current.succeeded_mass, 1.0,
        "young in-flight/lost facts do not contaminate the mature cohort"
    );
    let later = window.snapshot().completion_forecast(&CompletionQuery {
        now: now + Duration::from_secs(30),
        ..query
    });
    assert_eq!(later.samples, 2);
    assert_eq!(later.unresolved_mass, 1.0);

    // A delayed old producer never overwrites a more recent ring incarnation.
    let newer = RequestLifecycleId {
        producer: 1,
        admission: 2 + observer.shared.admissions.len() as u64,
    };
    observer.record_admission(newer, now);
    observer.record_admission(recent, now - Duration::from_secs(50));
    let slot = observer.shared.admissions[2].load();
    assert_eq!(slot.as_ref().unwrap().lifecycle, newer);
    assert_eq!(slot.as_ref().unwrap().admitted_at, now);
}

#[tokio::test]
async fn projection_only_admission_is_published_as_unknown_even_when_shutdown_has_no_queue_work() {
    let owner = StoppingEstimatorOwner::spawn(StoppingCollectorConfig {
        stopping: StoppingConfig {
            min_risk_set: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let observer = owner.observer();
    let now = Instant::now();
    let mut first = fact(now - Duration::from_secs(40));
    first.completion = Some(RequestCompletionOutcome::Succeeded);
    assert_eq!(
        observer.try_record(&first),
        RequestObservationDelivery::Recorded
    );
    observer.record_admission(
        RequestLifecycleId {
            producer: 1,
            admission: 2,
        },
        now - Duration::from_secs(40),
    );
    observer.record_drop(RequestObservationDrop::ProjectionLimit);
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.window.unobserved_lifetimes, 1);
    assert_eq!(exit.collector.dropped_projection, 1);
    let completion = owner
        .source()
        .snapshot()
        .completion_forecast(&CompletionQuery {
            rules: &first.rules,
            cap: first.cap,
            maturity: Duration::from_secs(30),
            now,
        });
    assert_eq!(completion.samples, 2);
    assert_eq!(completion.succeeded_mass, 0.5);
    assert_eq!(completion.unresolved_mass, 0.5);
}

#[tokio::test]
async fn accepted_facts_drain_before_join_and_publish_original_timestamp() {
    let owner = StoppingEstimatorOwner::spawn(StoppingCollectorConfig {
        stopping: StoppingConfig {
            min_risk_set: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let now = Instant::now();
    let mut observation = fact(now);
    observation.observed_at = now - Duration::from_secs(61);
    observation.admitted_at = observation.observed_at;
    assert_eq!(
        owner.observer().try_record(&observation),
        RequestObservationDelivery::Recorded
    );
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.collector.accepted, 1);
    assert_eq!(exit.collector.processed, 1);
    assert_eq!(exit.collector.queued_records, 0);
    assert_eq!(exit.collector.queued_bytes, 0);
    assert!(exit.collector.max_feedback_delay_ns >= 61_000_000_000);
    let snapshot = owner.source().snapshot();
    let forecast = snapshot.forecast(&StoppingQuery {
        rules: &observation.rules,
        verified_survival_through: 0,
        cap: 20,
        horizon: 20,
        now: Instant::now(),
    });
    assert_eq!(
        forecast.unresolved_mass, 1.0,
        "delivery cannot refresh expired producer facts"
    );
    assert_eq!(
        owner.observer().try_record(&observation),
        RequestObservationDelivery::Dropped(RequestObservationDrop::Closed)
    );
}

#[tokio::test]
async fn cancelling_shutdown_observation_does_not_cancel_original_worker() {
    let owner = StoppingEstimatorOwner::spawn(StoppingCollectorConfig {
        stopping: StoppingConfig {
            min_risk_set: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let observation = fact(Instant::now());
    assert_eq!(
        owner.observer().try_record(&observation),
        RequestObservationDelivery::Recorded
    );
    let mut first_wait = Box::pin(owner.shutdown());
    // Poll once to request shutdown, then drop only the public observation.
    let _ = futures::poll!(&mut first_wait);
    drop(first_wait);
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.collector.processed, 1);
    assert_eq!(exit.collector.worker_failures, 0);
    let snapshot = owner.source().snapshot();
    let forecast = snapshot.forecast(&StoppingQuery {
        rules: &observation.rules,
        verified_survival_through: 0,
        cap: 20,
        horizon: 20,
        now: Instant::now(),
    });
    assert_eq!(
        forecast.stops,
        vec![StoppingMass {
            generated_tokens: 8,
            mass: 1.0
        }]
    );
}

#[test]
fn invalid_allocation_extents_and_absent_runtime_return_errors() {
    assert!(matches!(
        StoppingEstimatorOwner::spawn(StoppingCollectorConfig {
            queue_capacity: usize::MAX,
            ..Default::default()
        }),
        Err(EstimatorError::Configuration(_))
    ));
    assert!(matches!(
        StoppingEstimatorOwner::spawn(StoppingCollectorConfig::default()),
        Err(EstimatorError::RuntimeUnavailable)
    ));
}
