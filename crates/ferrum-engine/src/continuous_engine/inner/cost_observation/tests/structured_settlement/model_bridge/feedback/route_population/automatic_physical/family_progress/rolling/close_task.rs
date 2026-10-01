//! Real FIFO/feedback consumption while the unique collector is held by a latch.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    trainer::TrainingWorkerOwner, worker::CostTrainingWorker,
};

fn attach_worker(f: &Families) -> CostTrainingWorker {
    let state = TrainingWorkerOwner(f.runtime.training.clone());
    let worker =
        CostTrainingWorker::spawn_with_progress(move || state.consume_progress(), None).unwrap();
    f.runtime
        .sink
        .attach_worker(worker.notification_thread())
        .unwrap();
    f.live().attach_worker(worker.notification_thread());
    worker
}

async fn drain_cut(f: &Families) {
    let cut = f.runtime.request_checkpoint().unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(3), cut.wait())
        .await
        .unwrap()
        .unwrap();
}

fn queue(f: &Families) {
    let stages = fixture::record_cohort_route(&f.runtime, &f.clock, Families::wave(A)).unwrap();
    assert!(stages
        .structured_evidence
        .as_ref()
        .is_some_and(Result::is_ok));
}

#[tokio::test]
async fn rolling_owned_close_keeps_original_feedback_fifo_moving_and_stop_waits() {
    let mut f = fixture(4);
    for _ in 0..4 {
        block(&mut f, A);
    }
    assert!(f.known(A));
    open_original_block(&f);
    for _ in 1..OFFERS {
        f.record(A);
    }
    let gate = f.live().hold_automatic_close_for_test();
    let worker = Arc::new(attach_worker(&f));
    queue(&f);
    gate.wait_started();
    let pending = rolling!(&f);
    assert!(pending.pending_close_enrollment.is_some());
    let before = f.runtime.audit_snapshot();
    let compared = before.structured_feedback.unwrap().compared;
    let enrolled = f.live().audit().population;
    assert_eq!(enrolled.issued, OFFERS);
    assert_eq!(enrolled.retired, OFFERS);
    for _ in 0..2 {
        assert!(f.runtime.reserve_live_ticket(f.clock.now_ns()).is_none());
        let stages = fixture::record_unticketed_cohort_with_hook(
            &f.runtime,
            &f.clock,
            Families::wave(A),
            |_| {},
        )
        .unwrap();
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_some_and(Result::is_ok));
        drain_cut(&f).await;
    }
    let after = f.runtime.audit_snapshot();
    assert_eq!(after.sink.raw_resolved, before.sink.raw_resolved + 2);
    assert_eq!(after.structured_feedback.unwrap().compared, compared + 2);
    assert_eq!(after.sink.raw_lost, 0);
    assert_eq!(after.sink.raw_resolution_failed, 0);
    let still = rolling!(&f);
    assert_eq!(still.original_block, pending.original_block);
    assert_eq!(still.resource.spent, pending.resource.spent);
    assert_eq!(
        still.resource.source_reservations,
        pending.resource.source_reservations
    );
    assert_eq!(
        still.pending_close_enrollment,
        pending.pending_close_enrollment
    );
    assert_eq!(f.live().audit().population.issued, OFFERS);
    f.runtime.training.request_export_finalization();
    let stopping = worker.clone();
    let stop = tokio::spawn(async move { stopping.shutdown().await });
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !worker.shutdown_started() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        !stop.is_finished(),
        "stop must join the owned close before sealing feedback"
    );
    gate.release();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(3), stop)
            .await
            .unwrap()
            .unwrap()
    );
    let done = rolling!(&f);
    assert!(done.pending_close_enrollment.is_none());
    assert_eq!(done.resource.source_reservations, 0);
    assert!(done.resource.original_blocks_credited >= pending.resource.original_blocks_credited);
    assert!(
        !f.runtime
            .audit_snapshot()
            .structured_feedback
            .unwrap()
            .worker_failed
    );
}

#[tokio::test]
async fn rolling_owned_close_expired_result_is_discarded_and_reservation_settles() {
    let f = fixture(2);
    let gate = f.live().hold_automatic_close_for_test();
    let worker = attach_worker(&f);
    for _ in 0..OFFERS {
        queue(&f);
    }
    gate.wait_started();
    let pending = rolling!(&f);
    let expiry = pending
        .pending_close_enrollment
        .as_ref()
        .unwrap()
        .deadline_ns;
    f.clock.set(expiry + 1);
    worker.wake();
    drain_cut(&f).await;
    assert!(f.live().automatic_computation_pending());
    f.runtime.training.request_export_finalization();
    gate.release();
    assert!(worker.shutdown().await);
    let done = rolling!(&f);
    assert!(done.pending_close_enrollment.is_none());
    assert_eq!(done.resource.source_reservations, 0);
    assert_eq!(f.live().audit().qualified_publications, 0);
    assert!(f.live().audit().failed_generations >= 1);
}

struct RetirementDirectory(PathBuf);
impl Drop for RetirementDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn rolling_owned_retirement_backfills_original_archive_while_feedback_and_stop_progress() {
    use sha2::{Digest, Sha256};
    let directory = RetirementDirectory(
        std::env::temp_dir().join(format!("ferrum-owned-retirement-{}", uuid::Uuid::new_v4())),
    );
    let mut f = Families::configured_with_settings(
        SloAutomaticCalibrationSettingsV1 {
            population_schedule:
                ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2,
            discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
            phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
            maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
            diagnostics: SloAutomaticCalibrationDiagnosticsV1::Directory {
                directory: directory.0.clone(),
                maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
                maximum_total_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
                maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
            },
            ..Default::default()
        },
        true,
    );
    for _ in 0..4 {
        block(&mut f, A);
    }
    assert!(f.known(A));
    open_original_block(&f);
    f.record(A);
    f.record(A);
    let missing = f.runtime.reserve_live_ticket(f.clock.now_ns()).unwrap();
    for _ in 3..OFFERS {
        f.record(A);
    }
    let gate = f.live().hold_automatic_close_for_test();
    let worker = Arc::new(attach_worker(&f));
    drop(missing);
    gate.wait_started();
    let pending = rolling!(&f);
    assert!(pending.pending_retirement_generation.is_some());
    let before = f.runtime.audit_snapshot();
    let compared = before.structured_feedback.unwrap().compared;
    let frozen_publications = f.live().audit().qualified_publications;
    for _ in 0..2 {
        assert!(f.runtime.reserve_live_ticket(f.clock.now_ns()).is_none());
        fixture::record_unticketed_cohort_with_hook(
            &f.runtime,
            &f.clock,
            Families::wave(A),
            |_| {},
        )
        .unwrap();
        drain_cut(&f).await;
    }
    let after = f.runtime.audit_snapshot();
    assert_eq!(after.sink.raw_resolved, before.sink.raw_resolved + 2);
    assert_eq!(after.structured_feedback.unwrap().compared, compared + 2);
    assert_eq!(after.sink.raw_lost, 0);
    let still = rolling!(&f);
    assert_eq!(still.resource.spent, pending.resource.spent);
    assert_eq!(
        still.resource.source_reservations,
        pending.resource.source_reservations
    );
    assert_eq!(
        still.resource.original_blocks_credited,
        pending.resource.original_blocks_credited
    );
    assert_eq!(
        still.pending_retirement_generation,
        pending.pending_retirement_generation
    );
    assert_eq!(still.original_block, pending.original_block);
    f.runtime.training.request_export_finalization();
    let stopping = worker.clone();
    let stop = tokio::spawn(async move { stopping.shutdown().await });
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !worker.shutdown_started() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        !stop.is_finished(),
        "stop must wait for the unique archive transaction"
    );
    gate.release();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(3), stop)
            .await
            .unwrap()
            .unwrap()
    );
    let done = rolling!(&f);
    assert!(done.pending_retirement_generation.is_none());
    assert_eq!(done.resource.source_reservations, 0);
    assert_eq!(
        done.resource.original_blocks_credited,
        pending.resource.original_blocks_credited
    );
    assert_eq!(f.live().audit().qualified_publications, frozen_publications);
    let failed = f.live().audit().last_failed_source.unwrap();
    assert!(
        failed.footer_complete && !failed.retained_unpublished,
        "{failed:?}"
    );
    let bytes = fs::read(failed.path.unwrap()).unwrap();
    assert_eq!(failed.bytes, bytes.len() as u64);
    assert_eq!(
        failed.sha256.unwrap(),
        format!("{:x}", Sha256::digest(&bytes))
    );
    let rows: Vec<serde_json::Value> = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    let failed_ticket = rows.iter().find(|row| row["kind"] == "failed").unwrap()["ticket"]
        .as_u64()
        .unwrap();
    let later: Vec<_> = rows
        .iter()
        .filter(|row| row["kind"] == "completed")
        .map(|row| row["wave"]["ticket"].as_u64().unwrap())
        .filter(|ticket| *ticket > failed_ticket)
        .collect();
    assert_eq!(
        later,
        ((failed_ticket + 1)..(failed_ticket + (OFFERS - 2) as u64)).collect::<Vec<_>>(),
        "all later original rows, and no unticketed row, remain in the archive"
    );
    assert_eq!(rows.last().unwrap()["kind"], "footer");
}

#[tokio::test]
async fn rolling_owned_retirement_expired_source_finishes_once_without_new_credit() {
    let mut f = fixture(2);
    f.record(A);
    for _ in 1..OFFERS {
        queue(&f);
    }
    let deadline = rolling!(&f).active_enrollments[0].deadline_ns;
    // All records were really issued/settled before expiry, but remain in the
    // original FIFO. Later consumption must retain them as failed evidence.
    f.clock.set(deadline + 1);
    let gate = f.live().hold_automatic_close_for_test();
    let worker = Arc::new(attach_worker(&f));
    worker.wake();
    gate.wait_started();
    let pending = rolling!(&f);
    assert_eq!(pending.pending_retirement_generation, Some(1));
    let before = f.runtime.audit_snapshot().sink.raw_resolved;
    for _ in 0..2 {
        fixture::record_unticketed_cohort_with_hook(
            &f.runtime,
            &f.clock,
            Families::wave(A),
            |_| {},
        )
        .unwrap();
        drain_cut(&f).await;
    }
    assert_eq!(f.runtime.audit_snapshot().sink.raw_resolved, before + 2);
    assert_eq!(rolling!(&f).resource.spent, pending.resource.spent);
    f.runtime.training.request_export_finalization();
    gate.release();
    assert!(worker.shutdown().await);
    let done = rolling!(&f);
    assert_eq!(done.resource.source_reservations, 0);
    assert!(done.pending_retirement_generation.is_none());
    assert_eq!(done.resource.original_blocks_credited, 0);
    assert_eq!(f.live().audit().qualified_publications, 0);
    assert_eq!(f.live().audit().failed_generations, 1);
}
