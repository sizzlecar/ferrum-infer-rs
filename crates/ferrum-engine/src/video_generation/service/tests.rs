use super::*;
use crate::iterative_service::{
    observations::{LifecycleEvent, RequestOutcome},
    BatchError, BatchPreparation, BatchWorker, LifecycleObserver, LifecycleObserverHandle,
};
use crate::video_generation::{
    VideoFrameRate, VideoGenerationParameters, VideoReference, VideoReferenceLimits,
};
use ferrum_output::{
    mp4::{H264AacFragmentedV1, H264SpeedPreset, Mp4EncoderConfig, Mp4Spool, Mp4SpoolLimits},
    OutputBudget,
};
use ferrum_runtime::{HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use ferrum_types::{IterativeBatchPolicy, OutputConfig};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
};

fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
fn spec() -> VideoAudioSpec {
    VideoAudioSpec {
        width: 2,
        height: 2,
        frames: 1,
        fps_numerator: 24,
        fps_denominator: 1,
        audio_sample_rate: 32000,
        audio_sample_frames: 7,
    }
}
fn request(host: &HostMemoryAccount, seed: u64) -> VideoGenerationRequest {
    VideoGenerationRequest {
        parameters: VideoGenerationParameters {
            prompt: "reference".into(),
            width: 2,
            height: 2,
            num_frames: 1,
            frame_rate: VideoFrameRate {
                numerator: 24,
                denominator: 1,
            },
            seed,
        },
        references: vec![VideoReference::Image(
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
                .unwrap()
                .freeze(),
        )],
    }
}
fn config() -> VideoServiceConfig {
    VideoServiceConfig::new(
        ServiceConfig {
            batching: IterativeBatchPolicy::Configured,
            maximum_pending: 1,
            maximum_pending_bytes: 8192,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
        VideoRequestLimits {
            maximum_prompt_bytes: 128,
            references: VideoReferenceLimits {
                minimum_visual: 1,
                maximum_images: 1,
                maximum_videos: 0,
                maximum_audio: 0,
                maximum_total: 1,
            },
            maximum_encoded_reference_bytes: 128,
        },
    )
    .unwrap()
}
fn output_fixture() -> (tempfile::TempDir, Mp4Spool, OutputBudget) {
    let directory = tempfile::tempdir().unwrap();
    let spool = Mp4Spool::create(
        directory.path().join("spool"),
        Mp4SpoolLimits {
            maximum_bytes: 1 << 20,
            maximum_files: 12,
        },
    )
    .unwrap();
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 4,
        ..Default::default()
    })
    .unwrap();
    (directory, spool, budget)
}
fn output(spool: &Mp4Spool, budget: &OutputBudget) -> VideoOutputPlan {
    VideoOutputPlan::new(
        spec(),
        Mp4EncoderConfig {
            ffmpeg: PathBuf::from("/never-start-a-codec-in-this-protocol-test"),
            threads: 1,
            timeout: Duration::from_secs(1),
            maximum_encoded_bytes: 4096,
            io_buffer_bytes: 64,
            maximum_stderr_bytes: 64,
            preset: H264AacFragmentedV1 {
                h264_speed: H264SpeedPreset::Veryfast,
                crf: 20,
                audio_bitrate: 128000,
            },
        },
        spool.clone(),
        budget,
    )
    .unwrap()
}
struct Completed(mpsc::Sender<()>);
impl LifecycleObserver for Completed {
    fn record_at(&self, _: Instant, event: LifecycleEvent) -> Option<u64> {
        if matches!(
            event,
            LifecycleEvent::Finished {
                outcome: RequestOutcome::Completed,
                ..
            }
        ) {
            self.0.send(()).unwrap();
        }
        None
    }
}
struct WorkerFixture {
    host: HostMemoryAccount,
    entered: Option<mpsc::Sender<()>>,
    release: Option<mpsc::Receiver<()>>,
    completed: Option<Arc<Completed>>,
    visits: Arc<Mutex<Vec<u64>>>,
    shutdown: Arc<AtomicUsize>,
    failed_owner: Option<Arc<AtomicUsize>>,
}
impl BatchWorker for WorkerFixture {
    type Request = VideoGenerationRequest;
    type Output = RawVideoOutput;
    fn lifecycle_observer(&self) -> Option<LifecycleObserverHandle> {
        self.completed.clone().map(|v| v as _)
    }
    fn execute(
        &mut self,
        _: BatchPreparation,
        requests: Vec<Self::Request>,
    ) -> Result<Vec<Self::Output>, BatchError> {
        self.visits
            .lock()
            .unwrap()
            .extend(requests.iter().map(|r| r.parameters.seed));
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
            self.release
                .take()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
        if self.completed.is_some() {
            // Only a fenced byte-owner fixture for a cancelled handoff. It is
            // never encoded or returned as a generated/synthetic video.
            Ok(requests
                .iter()
                .map(|_| {
                    RawVideoOutput::new(
                        spec(),
                        self.host
                            .allocate_zeroed(HostMemoryClass::Output, spec().rgb_bytes().unwrap())
                            .unwrap()
                            .freeze(),
                        self.host
                            .allocate_zeroed(HostMemoryClass::Output, spec().pcm_bytes().unwrap())
                            .unwrap()
                            .freeze(),
                        &self.host,
                    )
                    .unwrap()
                })
                .collect())
        } else {
            Err(BatchError::Rejected(
                "stop after actual admitted worker invocation".into(),
            ))
        }
    }
    fn compatible(&self, _: &Self::Request, _: &Self::Request) -> bool {
        true
    }
    fn maximum_batch_size(&self, _: &Self::Request, configured: usize) -> usize {
        configured
    }
    fn shutdown(&mut self) -> Result<(), core::GeneratorError> {
        self.shutdown.fetch_add(1, Ordering::SeqCst);
        if let Some(owner) = self.failed_owner.take() {
            Err(Box::new(OriginalFailure(owner)))
        } else {
            Ok(())
        }
    }
}
#[derive(Debug)]
struct OriginalFailure(Arc<AtomicUsize>);
impl fmt::Display for OriginalFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("original shutdown failure")
    }
}
impl std::error::Error for OriginalFailure {}
impl Drop for OriginalFailure {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn worker(host: &HostMemoryAccount) -> WorkerFixture {
    WorkerFixture {
        host: host.clone(),
        entered: None,
        release: None,
        completed: None,
        visits: Arc::new(Mutex::new(Vec::new())),
        shutdown: Arc::new(AtomicUsize::new(0)),
        failed_owner: None,
    }
}

#[tokio::test]
async fn video_queue_cancellation_refunds_original_ingress_and_output_admission() {
    let host = host();
    let (_directory, spool, budget) = output_fixture();
    let (entered, boundary) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let mut worker = worker(&host);
    worker.entered = Some(entered);
    worker.release = Some(released);
    let visits = worker.visits.clone();
    let service = VideoService::start_worker(Box::new(worker), config(), host.clone()).unwrap();
    let active = service
        .try_submit(request(&host, 1), output(&spool, &budget))
        .unwrap();
    boundary.recv_timeout(Duration::from_secs(10)).unwrap();
    let queued = service
        .try_submit(request(&host, 2), output(&spool, &budget))
        .unwrap();
    let cancel = queued.cancellation_handle();
    let waiter = tokio::spawn(queued.wait());
    cancel.cancel();
    let replacement = service
        .try_submit(request(&host, 3), output(&spool, &budget))
        .unwrap();
    assert!(waiter.await.unwrap().unwrap_err().is_cancelled());
    release.send(()).unwrap();
    assert!(matches!(
        active.wait().await,
        Err(VideoJobError::Generation(_))
    ));
    assert!(matches!(
        replacement.wait().await,
        Err(VideoJobError::Generation(_))
    ));
    assert_eq!(*visits.lock().unwrap(), vec![1, 3]);
    assert!(!service.shutdown().await.unwrap().has_failure());
    drop(service);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(spool.usage().reserved_bytes, 0);
}

#[tokio::test]
async fn cancellation_after_raw_publication_prevents_codec_start_and_releases_raw_owners() {
    let host = host();
    let (_directory, spool, budget) = output_fixture();
    let (finished, published) = mpsc::channel();
    let mut worker = worker(&host);
    worker.completed = Some(Arc::new(Completed(finished)));
    let service = VideoService::start_worker(Box::new(worker), config(), host.clone()).unwrap();
    let job = service
        .try_submit(request(&host, 1), output(&spool, &budget))
        .unwrap();
    published.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(
        host.usage().unwrap().allocated_bytes()
            >= spec().rgb_bytes().unwrap() + spec().pcm_bytes().unwrap()
    );
    job.cancellation_handle().cancel();
    assert!(matches!(job.wait().await, Err(VideoJobError::Cancelled)));
    assert!(!service.shutdown().await.unwrap().has_failure());
    drop(service);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(spool.usage().reserved_files, 0);
}

#[tokio::test]
async fn finite_video_run_uses_original_worker_and_retains_original_failure() {
    let host = host();
    let (_directory, spool, budget) = output_fixture();
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut worker = worker(&host);
    worker.failed_owner = Some(dropped.clone());
    let visits = worker.visits.clone();
    let shutdown = worker.shutdown.clone();
    let run = VideoService::run_with_worker(
        Box::new(worker),
        config(),
        host.clone(),
        request(&host, 11),
        output(&spool, &budget),
    )
    .await;
    assert!(matches!(run.result, Err(VideoJobError::Generation(_))));
    assert_eq!(*visits.lock().unwrap(), vec![11]);
    assert_eq!(shutdown.load(Ordering::SeqCst), 1);
    assert!(run.shutdown.has_failure());
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    let (owner, original) = run.shutdown.into_parts();
    assert!(
        matches!(original.as_ref(), Some(WorkerFailure::Generation(error)) if error.downcast_ref::<OriginalFailure>().is_some())
    );
    drop((original, owner));
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(budget.usage().sessions, 0);
    assert_eq!(spool.usage().reserved_bytes, 0);
}

#[test]
fn raw_result_rejects_foreign_host_wrong_class_and_same_extent_wrong_geometry() {
    let host = host();
    let other = self::host();
    let rgb = host
        .allocate_zeroed(HostMemoryClass::Output, spec().rgb_bytes().unwrap())
        .unwrap()
        .freeze();
    let pcm = host
        .allocate_zeroed(HostMemoryClass::Output, spec().pcm_bytes().unwrap())
        .unwrap()
        .freeze();
    assert!(RawVideoOutput::new(spec(), rgb.clone(), pcm.clone(), &other).is_err());
    let input_class = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, rgb.len())
        .unwrap()
        .freeze();
    assert!(RawVideoOutput::new(spec(), input_class, pcm.clone(), &host).is_err());
    let raw = RawVideoOutput::new(spec(), rgb, pcm, &host).unwrap();
    let mut mismatched = spec();
    mismatched.fps_numerator = 25;
    assert!(raw.validate(&host, mismatched).is_err());
    drop(raw);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
