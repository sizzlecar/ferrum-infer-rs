use super::*;

pub(super) struct Directory(pub(super) std::path::PathBuf);
impl Directory {
    pub(super) fn new() -> Self {
        Self(std::env::temp_dir().join(format!("ferrum-video-store-test-{}", Uuid::new_v4())))
    }
    fn registry(&self, maximum: usize) -> Registry {
        Registry::new(maximum, &self.0, VideoJobRetention::default(), 1).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
pub(super) fn pending(id: Uuid) -> Snapshot {
    Snapshot {
        id,
        status: Status::InProgress,
        created_at: chrono::Utc::now().timestamp(),
        finished_at: None,
        seed: 42,
        cancel_requested: false,
        delete_requested: false,
        error: None,
        encoded_bytes: None,
    }
}
fn install_pending(registry: &Registry, id: Uuid, slot: Arc<OwnedSemaphorePermit>) {
    let snapshot = pending(id);
    registry.store.write(&snapshot).unwrap();
    registry.state().entries.insert(
        id,
        Entry {
            snapshot,
            cancel: None,
            outcome: None,
            _slot: Some(slot),
        },
    );
}

#[tokio::test]
async fn delete_and_shutdown_keep_pending_waiter_and_slot_until_real_terminal() {
    let directory = Directory::new();
    let registry = Arc::new(directory.registry(1));
    let slot = registry.reserve().unwrap();
    let id = Uuid::new_v4();
    let host = super::super::tests::host();
    let owner = host
        .allocate_zeroed(ferrum_runtime::HostMemoryClass::Output, 128)
        .unwrap();
    // Exercise the HTTP record/terminal boundary with an actual blocked waiter
    // and admitted payload. Engine/codec cancellation itself is qualified by
    // their tests; this fixture never constructs or returns a synthetic video.
    install_pending(&registry, id, slot.clone());
    let (release, wait) = tokio::sync::oneshot::channel::<()>();
    let worker = registry.clone();
    registry.state().tasks.push(tokio::spawn(async move {
        let _slot = slot;
        let _owner = owner;
        wait.await.unwrap();
        worker.complete(id, Outcome::Finished(Err(VideoJobError::Cancelled)));
    }));
    assert_eq!(
        registry.delete(id).unwrap().unwrap().status,
        Status::InProgress
    );
    assert!(registry.reserve().is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 128);
    let tasks = registry.close();
    assert!(registry.reserve().is_err());
    assert_eq!(tasks.len(), 1);
    assert!(!tasks[0].is_finished());
    release.send(()).unwrap();
    for task in tasks {
        task.await.unwrap();
    }
    assert!(registry.get(id).is_none());
    assert_eq!(registry.slots.available_permits(), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn ingress_reservations_and_readers_share_atomic_shutdown_boundary() {
    let directory = Directory::new();
    let registry = directory.registry(1);
    let first = registry.reserve().unwrap();
    assert!(registry.reserve().is_err());
    drop(first);
    assert!(registry.reserve().is_ok());
    let readers = Arc::new(Semaphore::new(1));
    let reader = registry.admit_reader(&readers).unwrap();
    assert!(registry.admit_reader(&readers).is_err());
    assert!(registry.close().is_empty());
    drop(reader);
    assert_eq!(readers.available_permits(), 1);
    assert!(registry.admit_reader(&readers).is_err());
}

#[test]
fn terminal_jobs_release_active_admission_and_recover_after_shutdown() {
    let directory = Directory::new();
    let registry = directory.registry(1);
    let mut ids = Vec::new();
    for _ in 0..3 {
        let id = Uuid::new_v4();
        install_pending(&registry, id, registry.reserve().unwrap());
        registry.complete(id, Outcome::Finished(Err(VideoJobError::Cancelled)));
        assert_eq!(registry.get(id).unwrap().status, Status::Cancelled);
        assert_eq!(registry.slots.available_permits(), 1);
        ids.push(id);
        assert_eq!(registry.retained_jobs(), ids.len());
        assert_eq!(registry.available_slots(), 1);
    }
    registry.close();
    registry.clear_retired();
    drop(registry);
    let registry = directory.registry(1);
    for id in ids {
        assert_eq!(registry.get(id).unwrap().status, Status::Cancelled);
    }
    assert!(registry.reserve().is_ok());
}

#[test]
fn capacity_and_ttl_evict_only_safe_terminal_records_from_completion_time() {
    let directory = Directory::new();
    let retention = VideoJobRetention {
        maximum_jobs: 1,
        ttl: std::time::Duration::from_secs(30),
        ..Default::default()
    };
    let registry = Registry::new(2, &directory.0, retention, 1).unwrap();
    let long_job = Uuid::new_v4();
    install_pending(&registry, long_job, registry.reserve().unwrap());
    registry
        .state()
        .entries
        .get_mut(&long_job)
        .unwrap()
        .snapshot
        .created_at = 0;
    registry.complete(long_job, Outcome::Finished(Err(VideoJobError::Cancelled)));
    let finished = registry.get(long_job).unwrap().finished_at.unwrap();
    assert!(registry.get(long_job).is_some());
    let active = Uuid::new_v4();
    install_pending(&registry, active, registry.reserve().unwrap());
    registry.prune(&mut registry.state(), finished + 31);
    assert!(registry.get(long_job).is_none());
    assert!(registry.get(active).is_some());
    let next = Uuid::new_v4();
    install_pending(&registry, next, registry.reserve().unwrap());
    registry.complete(next, Outcome::Finished(Err(VideoJobError::Cancelled)));
    registry.complete(active, Outcome::Finished(Err(VideoJobError::Cancelled)));
    assert_eq!(registry.len(), 1);
}

#[test]
fn panic_owner_stays_charged_through_delete_and_ttl() {
    let directory = Directory::new();
    let registry = directory.registry(1);
    let id = Uuid::new_v4();
    install_pending(&registry, id, registry.reserve().unwrap());
    let host = super::super::tests::host();
    let allocation = host
        .allocate_zeroed(ferrum_runtime::HostMemoryClass::Output, 128)
        .unwrap();
    registry.complete(
        id,
        Outcome::Panicked {
            _payload: Box::new(allocation),
        },
    );
    registry.delete(id).unwrap();
    registry.prune(&mut registry.state(), i64::MAX);
    assert!(registry.get(id).is_some());
    assert!(registry.reserve().is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 128);
    registry.close();
    registry.clear_retired();
    assert!(registry.drain_error().is_some());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 128);
}

// A signature-only fixture exercises storage publication/reader semantics,
// never codec or video-generation validity.
pub(super) const STORED_BYTES: &[u8] = b"\0\0\0\x18ftypisomstorage-fixture";
fn stored(registry: &Registry, id: Uuid) -> Artifact {
    registry
        .store
        .copy(id, STORED_BYTES.len() as u64, STORED_BYTES)
        .unwrap()
}

#[test]
fn completed_artifact_survives_restart_and_delete_keeps_download_charged() {
    use std::io::Read;
    let directory = Directory::new();
    let registry = directory.registry(1);
    let id = Uuid::new_v4();
    install_pending(&registry, id, registry.reserve().unwrap());
    let artifact = stored(&registry, id);
    registry.record_completion(
        id,
        Outcome::Stored(artifact),
        None,
        Some((42, STORED_BYTES.len() as u64)),
    );
    assert_eq!(registry.slots.available_permits(), 1);
    registry.close();
    registry.clear_retired();
    drop(registry);
    let registry = directory.registry(1);
    assert_eq!(registry.get(id).unwrap().status, Status::Completed);
    let mut reader = registry.artifact(id).unwrap().unwrap().open().unwrap();
    registry.delete(id).unwrap();
    assert!(registry.get(id).is_none());
    assert_eq!(registry.retained_bytes(), STORED_BYTES.len() as u64);
    let mut data = Vec::new();
    reader.read_to_end(&mut data).unwrap();
    assert_eq!(data, STORED_BYTES);
    drop(reader);
    assert_eq!(registry.retained_bytes(), 0);
}

#[test]
fn publication_or_delete_metadata_failure_is_not_reported_as_success() {
    let directory = Directory::new();
    let registry = directory.registry(1);
    let id = Uuid::new_v4();
    install_pending(&registry, id, registry.reserve().unwrap());
    let artifact = stored(&registry, id);
    registry.store.fail_next_record_write();
    registry.record_completion(
        id,
        Outcome::Stored(artifact),
        None,
        Some((42, STORED_BYTES.len() as u64)),
    );
    assert_eq!(registry.get(id).unwrap().status, Status::Failed);
    assert!(registry.artifact(id).unwrap().is_err());
    assert!(registry.is_closing());
    assert_eq!(registry.retained_bytes(), STORED_BYTES.len() as u64);
    registry.store.fail_next_record_write();
    assert!(registry.delete(id).is_err());
    assert!(registry.state().entries.contains_key(&id));
}

#[test]
fn concurrent_completed_outputs_reserve_bytes_and_evict_before_copying() {
    let directory = Directory::new();
    let registry = Arc::new(
        Registry::new(
            2,
            &directory.0,
            VideoJobRetention {
                maximum_bytes: 100,
                ..Default::default()
            },
            1,
        )
        .unwrap(),
    );
    let old = Uuid::new_v4();
    install_pending(&registry, old, registry.reserve().unwrap());
    let mut old_bytes = STORED_BYTES.to_vec();
    old_bytes.resize(60, 0);
    let old_artifact = registry.store.copy(old, 60, old_bytes.as_slice()).unwrap();
    registry.record_completion(old, Outcome::Stored(old_artifact), None, Some((42, 60)));
    let ids = [Uuid::new_v4(), Uuid::new_v4()];
    for id in ids {
        install_pending(&registry, id, registry.reserve().unwrap());
    }
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let threads: Vec<_> = ids
        .into_iter()
        .map(|id| {
            let registry = registry.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                registry.reserve_output(id, 30).unwrap().unwrap()
            })
        })
        .collect();
    let reservations: Vec<_> = threads
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    assert_eq!(registry.retained_bytes(), 60);
    assert!(registry.get(old).is_none());
    for (id, reservation) in ids.into_iter().zip(reservations) {
        let mut bytes = STORED_BYTES.to_vec();
        bytes.resize(30, 0);
        let artifact = registry
            .store
            .copy_reserved(id, reservation, bytes.as_slice())
            .unwrap();
        registry.record_completion(id, Outcome::Stored(artifact), None, Some((42, 30)));
        assert_eq!(registry.get(id).unwrap().status, Status::Completed);
    }
    assert_eq!(registry.retained_bytes(), 60);
}

#[tokio::test]
#[ignore = "requires explicitly installed ffmpeg with libx264 and AAC on PATH"]
async fn real_mp4_completion_releases_encoder_budget_and_survives_restart() {
    use ferrum_output::{
        mp4::{
            H264AacFragmentedV1, H264SpeedPreset, Mp4Encoder, Mp4EncoderConfig, Mp4Spool,
            Mp4SpoolLimits, VideoAudioSpec,
        },
        OutputBudget,
    };
    use std::{io::Read, time::Duration};
    let ffmpeg = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|path| path.join("ffmpeg"))
        .find(|path| path.is_file())
        .expect("ffmpeg on PATH");
    let host = super::super::tests::host();
    let spec = VideoAudioSpec {
        width: 32,
        height: 24,
        frames: 4,
        fps_numerator: 24,
        fps_denominator: 1,
        audio_sample_rate: 32000,
        audio_sample_frames: 5400,
    };
    let rgb = host
        .allocate_zeroed(
            ferrum_runtime::HostMemoryClass::Output,
            spec.rgb_bytes().unwrap(),
        )
        .unwrap()
        .freeze();
    let pcm = host
        .allocate_zeroed(
            ferrum_runtime::HostMemoryClass::Output,
            spec.pcm_bytes().unwrap(),
        )
        .unwrap()
        .freeze();
    let temporary = Directory::new();
    let spool = Mp4Spool::create(
        &temporary.0,
        Mp4SpoolLimits {
            maximum_bytes: 2 << 20,
            maximum_files: 3,
        },
    )
    .unwrap();
    let output = OutputBudget::new(ferrum_types::OutputConfig {
        max_sessions: 1,
        ..Default::default()
    })
    .unwrap();
    let encoder = Mp4Encoder::new(
        Mp4EncoderConfig {
            ffmpeg,
            threads: 1,
            timeout: Duration::from_secs(10),
            maximum_encoded_bytes: 1 << 20,
            io_buffer_bytes: 4096,
            maximum_stderr_bytes: 4096,
            preset: H264AacFragmentedV1 {
                h264_speed: H264SpeedPreset::Veryfast,
                crf: 20,
                audio_bitrate: 128000,
            },
        },
        spool.clone(),
    )
    .unwrap();
    let artifact = encoder
        .try_admit(output.try_open_session().unwrap(), spec)
        .unwrap()
        .unwrap()
        .start(rgb, pcm)
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(output.usage().sessions, 1);
    assert!(spool.usage().reserved_bytes > 0);
    let mut expected = Vec::new();
    artifact.open().unwrap().read_to_end(&mut expected).unwrap();
    let directory = Directory::new();
    let registry = Arc::new(directory.registry(1));
    let correlation = ferrum_types::RequestCorrelation::new(None);
    let id = correlation.request_id;
    install_pending(&registry, id, registry.reserve().unwrap());
    let completing = registry.clone();
    tokio::task::spawn_blocking(move || {
        completing.complete(
            id,
            Outcome::Finished(Ok(VideoGenerationResult {
                correlation,
                seed: 42,
                artifact,
            })),
        );
    })
    .await
    .unwrap();
    assert_eq!(registry.get(id).unwrap().status, Status::Completed);
    assert_eq!(output.usage().sessions, 0);
    assert_eq!(spool.usage().reserved_bytes, 0);
    assert_eq!(spool.usage().reserved_files, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(registry.slots.available_permits(), 1);
    assert_eq!(registry.retained_bytes(), expected.len() as u64);
    registry.close();
    registry.clear_retired();
    drop(registry);
    let registry = directory.registry(1);
    let mut reader = registry.artifact(id).unwrap().unwrap().open().unwrap();
    registry.delete(id).unwrap();
    assert_eq!(registry.retained_bytes(), expected.len() as u64);
    let mut downloaded = Vec::new();
    reader.read_to_end(&mut downloaded).unwrap();
    assert_eq!(downloaded, expected);
    drop(reader);
    assert_eq!(registry.retained_bytes(), 0);
}
