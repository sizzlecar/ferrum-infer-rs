//! Original engine source8 -> exact same-boot file replay -> feedback handoff.
//! The clock domain is an explicit CPU fixture, not a claim about this host OS.
use super::*;
use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;
use std::future::{poll_fn, Future};

/// The restart proof requires a complete original source. Drive its real FIFO
/// consumer after the producer yields, so the nonblocking sink's intentional
/// contention loss cannot turn this persistence test into a scheduling lottery.
/// Each Pending action keeps its original checkpoint waiter and queue cutoff.
async fn drive_manual_checkpoint<T>(
    runtime: &EngineCostRuntime,
    action: impl Future<Output = T>,
) -> T {
    assert_eq!(runtime.test_activity().0, 0);
    let mut action = std::pin::pin!(action);
    poll_fn(|cx| {
        let result = action.as_mut().poll(cx);
        if result.is_pending() {
            // Completing the real oneshot wakes this task for its next poll.
            // Do not spin, retry collection, or replace the pending action.
            runtime.drain_calibration_fixture();
        }
        result
    })
    .await
}

struct SameBootClock {
    now: AtomicU64,
    domain: CostMonotonicDomainV1,
}
impl CostObservationClock for SameBootClock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.now.fetch_add(1, Ordering::Relaxed))
    }
    fn monotonic_domain(&self) -> Option<&CostMonotonicDomainV1> {
        Some(&self.domain)
    }
}

#[tokio::test]
async fn restart_feedback_handoff_real_source8_replay_preserves_original_model_and_margin() {
    let directory = Directory::new();
    let clock = Arc::new(SameBootClock {
        now: AtomicU64::new(100),
        domain: CostMonotonicDomainV1::new_macos_continuous([4; 16]).unwrap(),
    });
    let (mut session, executor) = automatic_session_with_storage_and_worker(
        2,
        clock.clone(),
        ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: directory.0.clone(),
            maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(64 * 1024 * 1024).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
        },
        ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {},
        FixtureCostWorker::Manual,
    )
    .await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    session.begin_startup_owner_series(1, deadline).unwrap();
    let declared = full_population(&session);
    drive_manual_checkpoint(
        &runtime,
        session.begin_prepared_owner_source(declared, CostProfileLoadLimits::default()),
    )
    .await
    .unwrap();
    let observer = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .journal_observer()
        .unwrap();
    // Each cohort's first prefix owner freezes an original FIFO cut inside
    // install_prefix_request. That nested waiter needs the same real consumer.
    drive_manual_checkpoint(&runtime, collect_source(&mut session, [16, 8, 8], deadline)).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 96);
    assert!(
        drive_manual_checkpoint(&runtime, session.activate_prepared_owner_source())
            .await
            .unwrap()
            > 0
    );
    let original = runtime.snapshot().unwrap();
    let original_children = runtime.startup_series_children_for_test().unwrap();
    assert!(!original_children.is_empty());
    let fingerprint = original_children[0].fingerprint().clone();
    let workload_domain = original_children[0].workload_domain().cloned().unwrap();
    drive_manual_checkpoint(&runtime, session.retire_startup_owner_source())
        .await
        .unwrap();
    let status = observer.status();
    let PreparedSourceJournalStatus::Completed(archive) = status.as_ref() else {
        panic!("{status:?}");
    };
    assert_eq!(archive.checkpoints.len(), 1);
    let limits = CostProfileLoadLimits::default();
    let destination = directory.0.join("profile15.json");
    let now = file::StructuredServiceClockV7 {
        monotonic_ns: clock.now_ns().unwrap(),
        wall_unix_ns: 1,
    };
    let exported = file::export_structured_profile_v15_same_boot(
        &archive.path,
        archive.sha256,
        archive.checkpoints[0].source_bytes,
        &destination,
        &clock.domain,
        now,
        &limits,
    )
    .unwrap();
    assert_eq!(exported.source_sha256, archive.checkpoints[0].source_sha256);
    let loaded = file::load_structured_profile_v15_same_boot(
        &destination,
        &fingerprint,
        &limits,
        &clock.domain,
        now,
    )
    .unwrap();
    assert_eq!(loaded.children.len(), original_children.len());
    assert!(
        loaded.children.iter().all(|new| {
            let old = original_children
                .iter()
                .find(|old| old.domain_signature() == new.domain_signature())
                .unwrap();
            new.provenance().source_sha256 == old.provenance().source_sha256
                && new.provenance().file_sha256 != old.provenance().file_sha256
        }),
        "metadata changes at File storage, original source does not"
    );
    original.check_restart_feedback_handoff_for_test(
        loaded.children,
        &workload_domain,
        &clock.domain,
        now.monotonic_ns,
        &directory.0.join("feedback.json"),
    );
    original.check_automatic_reuse_replay_for_test(
        &archive.path,
        archive.sha256,
        (
            archive.checkpoints[0].source_bytes,
            archive.checkpoints[0].source_sha256,
        ),
        &fingerprint,
        &workload_domain,
        clock.as_ref(),
        &directory.0.join("automatic-cache"),
    );
    session.finish_startup_owner_series().unwrap();
    session.shutdown().await.unwrap();
}
