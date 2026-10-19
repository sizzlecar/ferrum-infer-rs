use super::*;
use ferrum_execution::{execution_attempt::*, model_executor::PrefillChunk, token_execution::*};
use ferrum_scheduler::cost_prediction::*;
use ferrum_types::RequestId;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Condvar, Mutex,
};

#[derive(Default)]
pub(crate) struct Gate {
    released: Mutex<bool>,
    wake: Condvar,
    entered: AtomicUsize,
    notification: tokio::sync::Notify,
}

impl Gate {
    pub(crate) fn enter(&self) {
        self.entered.fetch_add(1, Ordering::Release);
        self.notification.notify_one();
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.wake.wait(released).unwrap();
        }
    }
    async fn wait_entered(&self) {
        loop {
            let notified = self.notification.notified();
            if self.entered.load(Ordering::Acquire) > 0 {
                return;
            }
            notified.await;
        }
    }
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_all();
    }
}

struct Release(Arc<Gate>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn config() -> ReferenceEstimatorConfig {
    ReferenceEstimatorConfig {
        min_samples: 1,
        ..Default::default()
    }
}

fn sample(milliseconds: u64, position: usize) -> ExecutionAttemptSummary {
    let completed_at = Instant::now();
    ExecutionAttemptSummary {
        started_at: completed_at
            .checked_sub(Duration::from_millis(milliseconds))
            .unwrap(),
        completed_at,
        participants: vec![ExecutionAttemptParticipant {
            request_id: RequestId::new(),
            kind: AttemptWorkKind::Decode,
            requested: ExecutionRange::new(position, 1).unwrap(),
            context_tokens: position + 1,
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
            outcome: AttemptParticipantOutcome::Completed {
                work: ExecutedTokenWork::Decode { position },
                frontier_committed: true,
            },
            product_failed: false,
        }],
        eligibility: ExecutionAttemptEligibility::Completed,
    }
}

fn work(sample: &ExecutionAttemptSummary) -> Vec<CostWork> {
    sample
        .participants
        .iter()
        .map(|member| CostWork {
            kind: member.kind,
            requested: member.requested,
            context_tokens: member.context_tokens,
            prefix_restore: member.prefix_restore,
            internally_reused_tokens: member.internally_reused_tokens,
        })
        .collect()
}

async fn revision(source: &dyn CostSnapshotSource, target: u64) -> Arc<dyn CostSnapshot> {
    let mut changed = source.subscribe();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let snapshot = source.load();
            if snapshot.revision() >= target {
                return snapshot;
            }
            changed.changed().await.unwrap();
        }
    })
    .await
    .unwrap()
}

fn predict(
    snapshot: &dyn CostSnapshot,
    sample: &ExecutionAttemptSummary,
    now: Instant,
) -> CostPrediction {
    snapshot.predict(
        &CostQuery {
            work: &work(sample),
            work_is_exact: true,
        },
        now,
    )
}

#[tokio::test]
async fn reference_queue_credit_covers_active_payload_and_nonblocking_rejection() {
    for byte_limited in [false, true] {
        let gate = Arc::new(Gate::default());
        let release = Release(Arc::clone(&gate));
        let bytes = ExecutionAttemptSummary::projected_bytes(1).unwrap();
        let owner = ReferenceEstimator::spawn_inner(
            ReferenceEstimatorConfig {
                queue_capacity: if byte_limited { 2 } else { 1 },
                max_queue_bytes: if byte_limited { bytes } else { bytes * 2 },
                max_summary_bytes: bytes,
                ..config()
            },
            worker::Hooks {
                gate: Some(Arc::clone(&gate)),
                ..Default::default()
            },
        )
        .unwrap();
        let observer = owner.observer();
        let value = sample(2, 4);
        assert_eq!(
            observer.try_record(&value),
            ExecutionAttemptDelivery::Recorded
        );
        tokio::time::timeout(Duration::from_secs(5), gate.wait_entered())
            .await
            .unwrap();
        assert_eq!(owner.stats().queued_records, 1);
        assert_eq!(owner.stats().queued_bytes, bytes);
        assert_eq!(
            observer.try_record(&value),
            ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::QueueFull)
        );
        assert_eq!(
            owner.stats().queued_records,
            1,
            "failed byte reservation rolls back its record"
        );
        assert_eq!(owner.stats().queued_bytes, bytes);
        let mut oversized = value.clone();
        oversized
            .participants
            .push(sample(1, 4).participants.remove(0));
        assert_eq!(
            observer.try_record(&oversized),
            ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::ProjectionLimit)
        );
        assert_eq!(owner.stats().queued_bytes, bytes);
        drop(release);
        let exit = owner.shutdown().await.unwrap();
        assert_eq!(exit.stats.accepted, 1);
        assert_eq!(exit.stats.trained_samples, 1);
        assert_eq!(exit.stats.queued_records, 0);
        assert_eq!(exit.stats.queued_bytes, 0);
        assert_eq!(exit.stats.dropped_bytes, u64::from(byte_limited));
        assert_eq!(exit.stats.dropped_full, u64::from(!byte_limited));
        assert_eq!(
            observer.try_record(&value),
            ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::Closed)
        );
    }
}

#[tokio::test]
async fn reference_shutdown_cancel_and_owner_drop_join_the_same_blocked_worker() {
    for drop_owner in [false, true] {
        let gate = Arc::new(Gate::default());
        let release = Release(Arc::clone(&gate));
        let owner = ReferenceEstimator::spawn_inner(
            config(),
            worker::Hooks {
                gate: Some(Arc::clone(&gate)),
                ..Default::default()
            },
        )
        .unwrap();
        let observer = owner.observer();
        let mut completion = owner.completion.clone();
        assert_eq!(
            observer.try_record(&sample(1, 3)),
            ExecutionAttemptDelivery::Recorded
        );
        tokio::time::timeout(Duration::from_secs(5), gate.wait_entered())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), owner.shutdown())
                .await
                .is_err()
        );
        assert!(
            completion.borrow().is_none(),
            "a stopped request is not a joined worker"
        );
        assert_eq!(
            observer.try_record(&sample(1, 3)),
            ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::Closed)
        );
        if drop_owner {
            drop(owner);
            drop(release);
            let exit = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if let Some(result) = completion.borrow_and_update().clone() {
                        break result.unwrap();
                    }
                    completion.changed().await.unwrap();
                }
            })
            .await
            .unwrap();
            assert_eq!(exit.stats.trained_samples, 1);
            assert_eq!(exit.stats.queued_bytes, 0);
        } else {
            drop(release);
            let exit = owner.shutdown().await.unwrap();
            assert_eq!(exit, owner.shutdown().await.unwrap());
            assert_eq!(exit.stats.trained_samples, 1);
            assert_eq!(exit.stats.queued_bytes, 0);
        }
    }
}

#[tokio::test]
async fn reference_publishes_immutable_bounded_windows_and_expiring_estimates() {
    let owner = ReferenceEstimator::spawn(ReferenceEstimatorConfig {
        min_samples: 2,
        window_samples: 2,
        ..config()
    })
    .unwrap();
    let observer = owner.observer();
    let source = owner.source();
    let changes = source.subscribe();
    let first = sample(4, 3);
    assert_eq!(
        observer.try_record(&first),
        ExecutionAttemptDelivery::Recorded
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while owner.stats().queued_records != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let cold = source.load();
    assert_eq!(owner.stats().trained_samples, 1);
    assert_eq!(owner.stats().published_snapshots, 0);
    assert_eq!(cold.revision(), 0);
    assert!(!changes.has_changed().unwrap());
    assert_eq!(
        predict(cold.as_ref(), &first, Instant::now()),
        CostPrediction::Unknown(CostUnknownReason::NoCoverage)
    );
    assert_eq!(
        observer.try_record(&sample(8, 3)),
        ExecutionAttemptDelivery::Recorded
    );
    let original = revision(source.as_ref(), 1).await;
    let CostPrediction::Known {
        upper_service_time,
        samples,
        expires_at,
        ..
    } = predict(original.as_ref(), &first, Instant::now())
    else {
        panic!("trained domain missing");
    };
    assert_eq!(upper_service_time, Duration::from_millis(10));
    assert_eq!(samples, 2);
    for (target, duration) in [(2, 2), (3, 3)] {
        assert_eq!(
            observer.try_record(&sample(duration, 3)),
            ExecutionAttemptDelivery::Recorded
        );
        revision(source.as_ref(), target).await;
    }
    let latest = source.load();
    let CostPrediction::Known {
        upper_service_time,
        samples,
        ..
    } = predict(latest.as_ref(), &first, Instant::now())
    else {
        panic!("window domain missing");
    };
    assert_eq!(upper_service_time, Duration::from_micros(3750));
    assert_eq!(samples, 2);
    assert_eq!(original.revision(), 1);
    assert!(
        matches!(predict(original.as_ref(), &first, Instant::now()), CostPrediction::Known { upper_service_time, .. } if upper_service_time == Duration::from_millis(10))
    );
    assert_eq!(
        predict(original.as_ref(), &first, expires_at),
        CostPrediction::Unknown(CostUnknownReason::Stale)
    );
    assert_eq!(
        latest.predict(
            &CostQuery {
                work: &work(&first),
                work_is_exact: false
            },
            Instant::now()
        ),
        CostPrediction::Unknown(CostUnknownReason::InexactWork)
    );
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.stats.trained_samples, 4);
    assert_eq!(exit.revision, 3);
    assert_eq!(exit.stats.queued_bytes, 0);
}

#[tokio::test]
async fn reference_failure_partial_and_unknown_never_train_successful_peer_wall() {
    let owner = ReferenceEstimator::spawn(config()).unwrap();
    let observer = owner.observer();
    for eligibility in [
        ExecutionAttemptEligibility::CompletedWithNotSubmitted,
        ExecutionAttemptEligibility::NotSubmitted,
        ExecutionAttemptEligibility::ExecutorError,
        ExecutionAttemptEligibility::InvalidOutcome,
        ExecutionAttemptEligibility::Interrupted,
    ] {
        let mut value = sample(1, 3);
        value.eligibility = eligibility;
        assert_eq!(
            observer.try_record(&value),
            ExecutionAttemptDelivery::Recorded
        );
    }
    let mut bad_values = Vec::new();
    let mut zero = sample(1, 3);
    zero.started_at = zero.completed_at;
    bad_values.push(zero);
    let mut backwards = sample(1, 3);
    backwards.started_at = backwards.completed_at + Duration::from_secs(1);
    bad_values.push(backwards);
    let mut unknown = sample(1, 3);
    unknown.participants[0].outcome =
        AttemptParticipantOutcome::Interrupted(ExecutionProgress::Unknown {
            last_known: LastKnownExecutionStage::Submitted,
            known_executed_range: None,
        });
    bad_values.push(unknown);
    let mut uncommitted = sample(1, 3);
    uncommitted.participants[0].outcome = AttemptParticipantOutcome::Completed {
        work: ExecutedTokenWork::Decode { position: 3 },
        frontier_committed: false,
    };
    bad_values.push(uncommitted);
    let mut duplicate = sample(1, 3);
    duplicate
        .participants
        .push(duplicate.participants[0].clone());
    bad_values.push(duplicate);
    let mut partial = sample(1, 3);
    partial.participants[0].kind = AttemptWorkKind::Prefill;
    partial.participants[0].requested = ExecutionRange::new(0, 4).unwrap();
    partial.participants[0].outcome = AttemptParticipantOutcome::Completed {
        work: ExecutedTokenWork::Prefill {
            chunk: PrefillChunk::new(0, 2, 4).unwrap(),
            capacity_probe_count: 0,
        },
        frontier_committed: true,
    };
    bad_values.push(partial);
    for value in &bad_values {
        assert_eq!(
            observer.try_record(value),
            ExecutionAttemptDelivery::Recorded
        );
    }
    let mut completed = sample(2, 3);
    completed.participants[0].product_failed = true;
    assert_eq!(
        observer.try_record(&completed),
        ExecutionAttemptDelivery::Recorded
    );
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.stats.excluded_attempts, 5);
    assert_eq!(exit.stats.invalid_attempts, bad_values.len() as u64);
    assert_eq!(
        exit.stats.trained_samples, 1,
        "post-execution sampling failure preserves actual service work"
    );
    assert!(matches!(
        predict(owner.source().load().as_ref(), &completed, Instant::now()),
        CostPrediction::Known { .. }
    ));
}

#[tokio::test]
async fn reference_prefix_and_instance_domains_remain_separate_and_bounded() {
    let owner = ReferenceEstimator::spawn(ReferenceEstimatorConfig {
        max_domains: 1,
        ..config()
    })
    .unwrap();
    let independent = ReferenceEstimator::spawn(config()).unwrap();
    let cold = sample(2, 4);
    assert_eq!(
        owner.observer().try_record(&cold),
        ExecutionAttemptDelivery::Recorded
    );
    revision(owner.source().as_ref(), 1).await;
    let mut restored = sample(3, 4);
    restored.participants[0].prefix_restore = AttemptPrefixRestore::Restored { tokens: 2 };
    assert_eq!(
        owner.observer().try_record(&restored),
        ExecutionAttemptDelivery::Recorded
    );
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.stats.domain_limit, 1);
    assert_eq!(
        predict(owner.source().load().as_ref(), &restored, Instant::now()),
        CostPrediction::Unknown(CostUnknownReason::NoCoverage)
    );
    assert_eq!(
        predict(independent.source().load().as_ref(), &cold, Instant::now()),
        CostPrediction::Unknown(CostUnknownReason::NoCoverage)
    );
    assert_eq!(
        independent.observer().try_record(&restored),
        ExecutionAttemptDelivery::Recorded
    );
    revision(independent.source().as_ref(), 1).await;
    assert!(matches!(
        predict(
            independent.source().load().as_ref(),
            &restored,
            Instant::now()
        ),
        CostPrediction::Known { .. }
    ));
    let mut internally_known = restored.clone();
    internally_known.participants[0].internally_reused_tokens = Some(0);
    assert_eq!(
        predict(
            independent.source().load().as_ref(),
            &internally_known,
            Instant::now()
        ),
        CostPrediction::Unknown(CostUnknownReason::NoCoverage)
    );
    let exit = independent.shutdown().await.unwrap();
    assert_eq!(exit.stats.trained_samples, 1);
    assert_eq!(exit.stats.queued_bytes, 0);
}

#[tokio::test]
async fn reference_worker_panic_keeps_last_snapshot_until_its_original_expiry() {
    let panic_next = Arc::new(AtomicBool::new(false));
    let owner = ReferenceEstimator::spawn_inner(
        config(),
        worker::Hooks {
            panic_next: Some(Arc::clone(&panic_next)),
            ..Default::default()
        },
    )
    .unwrap();
    let source = owner.source();
    let original = sample(2, 3);
    assert_eq!(
        owner.observer().try_record(&original),
        ExecutionAttemptDelivery::Recorded
    );
    let snapshot = revision(source.as_ref(), 1).await;
    let CostPrediction::Known { expires_at, .. } =
        predict(snapshot.as_ref(), &original, Instant::now())
    else {
        panic!("valid snapshot missing");
    };
    panic_next.store(true, Ordering::Release);
    assert_eq!(
        owner.observer().try_record(&sample(9, 3)),
        ExecutionAttemptDelivery::Recorded
    );
    assert_eq!(
        owner.shutdown().await.unwrap_err(),
        EstimatorError::WorkerPanicked
    );
    assert_eq!(source.load().revision(), 1);
    assert_eq!(owner.stats().worker_failures, 1);
    assert_eq!(owner.stats().queued_records, 0);
    assert_eq!(owner.stats().queued_bytes, 0);
    assert!(matches!(
        predict(source.load().as_ref(), &original, Instant::now()),
        CostPrediction::Known { .. }
    ));
    assert_eq!(
        predict(source.load().as_ref(), &original, expires_at),
        CostPrediction::Unknown(CostUnknownReason::Stale)
    );
    let mut changed = source.subscribe();
    changed.borrow_and_update();
    assert!(changed.changed().await.is_err());
}

#[tokio::test]
async fn reference_invalid_configuration_and_training_overflow_cannot_publish() {
    for invalid in [
        ReferenceEstimatorConfig {
            queue_capacity: 0,
            ..config()
        },
        ReferenceEstimatorConfig {
            queue_capacity: usize::MAX,
            ..config()
        },
        ReferenceEstimatorConfig {
            queue_capacity: isize::MAX as usize / std::mem::size_of::<queue::Envelope>() + 1,
            ..config()
        },
        ReferenceEstimatorConfig {
            max_participants: usize::MAX,
            ..config()
        },
        ReferenceEstimatorConfig {
            max_summary_bytes: 1,
            ..config()
        },
    ] {
        assert!(
            matches!(
                ReferenceEstimator::spawn(invalid),
                Err(EstimatorError::Configuration(_))
            ),
            "{invalid:?}"
        );
    }
    let largest_ttl = ReferenceEstimatorConfig {
        snapshot_ttl_ms: u64::MAX,
        ..config()
    };
    assert_eq!(
        validate(&largest_ttl).is_ok(),
        Instant::now()
            .checked_add(Duration::from_millis(u64::MAX))
            .is_some(),
        "TTL validation follows this platform's actual monotonic clock range"
    );
    let owner = ReferenceEstimator::spawn(ReferenceEstimatorConfig {
        safety_margin: f64::MAX,
        ..config()
    })
    .unwrap();
    assert_eq!(
        owner.observer().try_record(&sample(1000, 3)),
        ExecutionAttemptDelivery::Recorded
    );
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.stats.training_failures, 1);
    assert_eq!(exit.revision, 0);
}
