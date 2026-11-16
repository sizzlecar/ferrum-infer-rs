use super::*;
use crate::OutputBudget;
use ferrum_types::OutputConfig;
use std::{
    io::Write,
    process::{Command, Stdio},
    sync::atomic::AtomicUsize,
};

fn spec() -> VideoAudioSpec {
    VideoAudioSpec {
        width: 32,
        height: 24,
        frames: 12,
        fps_numerator: 24,
        fps_denominator: 1,
        audio_sample_rate: 32000,
        // Independent model lengths: the audio tail is longer than 12/24s.
        audio_sample_frames: 16560,
    }
}
fn config(ffmpeg: PathBuf) -> Mp4EncoderConfig {
    Mp4EncoderConfig {
        ffmpeg,
        threads: 1,
        timeout: Duration::from_secs(30),
        maximum_encoded_bytes: 1 << 20,
        io_buffer_bytes: 8192,
        maximum_stderr_bytes: 8192,
        preset: H264AacFragmentedV1 {
            h264_speed: H264SpeedPreset::Veryfast,
            crf: 20,
            audio_bitrate: 128000,
        },
    }
}
fn budget() -> OutputBudget {
    OutputBudget::new(OutputConfig {
        max_sessions: 1,
        ..Default::default()
    })
    .unwrap()
}
fn spool(directory: &tempfile::TempDir, bytes: u64) -> Mp4Spool {
    Mp4Spool::create(
        directory.path().join("spool"),
        Mp4SpoolLimits {
            maximum_bytes: bytes,
            maximum_files: 3,
        },
    )
    .unwrap()
}
fn bytes(spec: VideoAudioSpec) -> (Vec<u8>, Vec<u8>) {
    let rgb = (0..spec.rgb_bytes().unwrap())
        .map(|n| (n.wrapping_mul(73) ^ (n >> 5)) as u8)
        .collect();
    let pcm = (0..spec.audio_sample_frames * 2)
        .flat_map(|n| ((n as i32 % 127 - 63) as f32 / 128.0).to_le_bytes())
        .collect();
    (rgb, pcm)
}

#[test]
fn admission_reserves_real_staging_and_buffers_then_refunds_without_starting() {
    let directory = tempfile::tempdir().unwrap();
    let config = config(PathBuf::from("/explicit/ffmpeg"));
    let spec = spec();
    let total = spec.rgb_bytes().unwrap() as u64
        + spec.pcm_bytes().unwrap() as u64
        + config.maximum_encoded_bytes;
    let store = spool(&directory, total);
    let encoder = Mp4Encoder::new(config.clone(), store.clone()).unwrap();
    let budget = budget();
    let session = budget.try_open_session().unwrap();
    let permit = encoder.try_admit(session.clone(), spec).unwrap().unwrap();
    assert_eq!(store.usage().reserved_bytes, total);
    assert_eq!(store.usage().reserved_files, 3);
    assert_eq!(budget.usage().encoded_bytes, config.buffer_bytes().unwrap());
    struct Counter(AtomicUsize);
    impl std::task::Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let io_counter = Arc::new(Counter(AtomicUsize::new(0)));
    let spool_counter = Arc::new(Counter(AtomicUsize::new(0)));
    let _io_wake = session.subscribe_capacity(std::task::Waker::from(io_counter.clone()));
    let _spool_wake = store.subscribe_capacity(std::task::Waker::from(spool_counter.clone()));
    assert!(encoder.try_admit(session.clone(), spec).unwrap().is_none());
    assert_eq!(
        io_counter.0.load(Ordering::SeqCst),
        0,
        "unchanged full spool must not self-wake through an I/O refund"
    );
    assert_eq!(spool_counter.0.load(Ordering::SeqCst), 0);
    assert_eq!(budget.usage().encoded_bytes, config.buffer_bytes().unwrap());
    assert!(permit
        .start(vec![0u8; 1], vec![0u8; spec.pcm_bytes().unwrap()])
        .is_err());
    assert_eq!(store.usage(), Mp4SpoolUsage::default());
    assert!(
        spool_counter.0.load(Ordering::SeqCst) > 0,
        "real disk retirement wakes the subscribed spool waiter"
    );
    assert_eq!(budget.usage().encoded_bytes, 0);
    drop(session);
    assert_eq!(budget.usage().sessions, 0);
}

#[test]
fn media_geometry_and_existing_unowned_spool_are_rejected() {
    let mut invalid = spec();
    invalid.width = 31;
    assert!(invalid.rgb_bytes().is_err());
    invalid = spec();
    invalid.frames = u64::MAX;
    assert!(invalid.rgb_bytes().is_err());
    invalid = spec();
    invalid.audio_sample_rate = 44100;
    assert!(invalid.pcm_bytes().is_err());
    let mut overflowing = config(PathBuf::from("/explicit/ffmpeg"));
    overflowing.io_buffer_bytes = u32::MAX as usize;
    assert!(overflowing.validate().is_err());
    let directory = tempfile::tempdir().unwrap();
    assert!(Mp4Spool::create(
        directory.path(),
        Mp4SpoolLimits {
            maximum_bytes: 100,
            maximum_files: 3
        }
    )
    .is_err());
}

fn tool(name: &str) -> PathBuf {
    // These tests are explicitly opted into; this lookup is test configuration,
    // never a hidden product fallback for its required absolute ffmpeg path.
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let path = directory.join(name);
        if path.is_file() {
            return std::fs::canonicalize(path).unwrap();
        }
    }
    panic!("explicit codec test requires {name} on PATH");
}
fn assert_refunded(store: &Mp4Spool, budget: &OutputBudget) {
    assert_eq!(store.usage(), Mp4SpoolUsage::default());
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}

#[tokio::test]
#[ignore = "requires ffmpeg/libx264/AAC and ffprobe"]
async fn real_ffmpeg_mp4_decodes_and_last_reader_retains_artifact_capacity() {
    for speed in [H264SpeedPreset::Veryfast, H264SpeedPreset::Ultrafast] {
        assert_fragmented_audio_video_timing(speed).await;
    }
}

async fn assert_fragmented_audio_video_timing(speed: H264SpeedPreset) {
    let directory = tempfile::tempdir().unwrap();
    let store = spool(&directory, 4 << 20);
    let mut configuration = config(tool("ffmpeg"));
    configuration.preset.h264_speed = speed;
    let encoder = Mp4Encoder::new(configuration, store.clone()).unwrap();
    let budget = budget();
    let (rgb, pcm) = bytes(spec());
    let artifact = encoder
        .try_admit(budget.try_open_session().unwrap(), spec())
        .unwrap()
        .unwrap()
        .start(rgb, pcm)
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert!(artifact.receipt().process.success && artifact.receipt().process.reaped);
    assert_eq!(
        artifact.receipt().container_profile,
        "h264-aac-fragmented-v1"
    );
    assert_eq!(
        store.usage().reserved_bytes,
        artifact.receipt().encoded_bytes
    );
    assert_eq!(store.usage().reserved_files, 1);
    assert_eq!(budget.usage().encoded_bytes, 0);
    let mut reader = artifact.open().unwrap();
    drop(artifact);
    assert_eq!(budget.usage().sessions, 1);
    assert!(budget.try_open_session().is_err());
    // A diagnostic copy is outside the product spool; its original reader
    // remains live and charged throughout both independent media validators.
    let copy = directory.path().join("validation.mp4");
    let mut file = std::fs::File::create(&copy).unwrap();
    std::io::copy(&mut reader, &mut file).unwrap();
    file.flush().unwrap();
    let probe = Command::new(tool("ffprobe"))
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_frames",
            "-show_streams",
            "-of",
            "json",
        ])
        .arg(&copy)
        .output()
        .unwrap();
    assert!(
        probe.status.success(),
        "{}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
    let streams = value["streams"].as_array().unwrap();
    let video = streams.iter().find(|s| s["codec_type"] == "video").unwrap();
    assert_eq!(video["codec_name"], "h264");
    assert_eq!(video["width"], 32);
    assert_eq!(video["height"], 24);
    assert_eq!(video["nb_read_frames"], "12");
    assert_eq!(video["avg_frame_rate"], "24/1");
    // Inspect decoded presentation timestamps, not just the advertised rate.
    // AAC priming must not stretch the first video frame or delay video start;
    // this applies both with and without H.264 frame reordering.
    let (time_numerator, time_denominator) = video["time_base"]
        .as_str()
        .unwrap()
        .split_once('/')
        .unwrap();
    let time_numerator: i64 = time_numerator.parse().unwrap();
    let time_denominator: i64 = time_denominator.parse().unwrap();
    let frames = value["frames"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|frame| frame["media_type"] == "video")
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), spec().frames as usize);
    for (index, frame) in frames.iter().enumerate() {
        let timestamp = frame["best_effort_timestamp"].as_i64().unwrap();
        assert_eq!(
            timestamp * time_numerator * i64::from(spec().fps_numerator),
            index as i64 * i64::from(spec().fps_denominator) * time_denominator,
            "presentation timestamp differs for {speed:?} frame {index}"
        );
    }
    let audio = streams.iter().find(|s| s["codec_type"] == "audio").unwrap();
    assert_eq!(audio["codec_name"], "aac");
    assert_eq!(audio["sample_rate"], "32000");
    assert_eq!(audio["channels"], 2);
    let duration = audio["duration"].as_str().unwrap().parse::<f64>().unwrap();
    assert!(duration >= spec().audio_sample_frames as f64 / spec().audio_sample_rate as f64);
    assert!(
        audio["nb_read_frames"]
            .as_str()
            .unwrap()
            .parse::<usize>()
            .unwrap()
            > 0
    );
    let decoded = Command::new(tool("ffmpeg"))
        .args(["-nostdin", "-v", "error", "-xerror", "-i"])
        .arg(&copy)
        .args(["-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    drop(reader);
    assert_refunded(&store, &budget);
}

#[tokio::test]
#[ignore = "requires ffmpeg/libx264/AAC"]
async fn real_ffmpeg_encoded_limit_kills_reaps_and_refunds() {
    let directory = tempfile::tempdir().unwrap();
    let store = spool(&directory, 4 << 20);
    let mut config = config(tool("ffmpeg"));
    config.maximum_encoded_bytes = 32;
    let encoder = Mp4Encoder::new(config, store.clone()).unwrap();
    let budget = budget();
    let (rgb, pcm) = bytes(spec());
    let error = encoder
        .try_admit(budget.try_open_session().unwrap(), spec())
        .unwrap()
        .unwrap()
        .start(rgb, pcm)
        .unwrap()
        .wait()
        .await
        .unwrap_err();
    assert_eq!(error.kind, Mp4ErrorKind::EncodedLimit);
    assert!(error.process.as_ref().unwrap().reaped);
    assert!(!error.has_unreaped_process());
    drop(error);
    assert_refunded(&store, &budget);
}

struct OriginalInput {
    bytes: Vec<u8>,
    dropped: Arc<AtomicUsize>,
    control: Arc<Mutex<Option<Mp4CancellationHandle>>>,
}
impl AsRef<[u8]> for OriginalInput {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl Drop for OriginalInput {
    fn drop(&mut self) {
        if let Some(control) = self.control.lock().unwrap().as_ref() {
            if control.process_id().is_some() {
                assert!(
                    control.process_reaped(),
                    "source owner released before child reaping"
                );
            }
        }
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
async fn await_started(control: &Mp4CancellationHandle) {
    tokio::time::timeout(Duration::from_secs(20), async {
        while control.process_id().is_none() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("real ffmpeg must start");
}
fn slow_spec() -> VideoAudioSpec {
    VideoAudioSpec {
        width: 256,
        height: 256,
        frames: 600,
        ..spec()
    }
}

#[tokio::test]
#[ignore = "requires ffmpeg/libx264/AAC"]
async fn real_ffmpeg_cancel_and_drop_retain_original_sources_until_reaping() {
    for abandon_waiter in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let store = spool(&directory, 256 << 20);
        let mut config = config(tool("ffmpeg"));
        config.preset.h264_speed = H264SpeedPreset::Veryslow;
        let encoder = Mp4Encoder::new(config, store.clone()).unwrap();
        let budget = budget();
        let dropped = Arc::new(AtomicUsize::new(0));
        let shared = Arc::new(Mutex::new(None));
        let (rgb, pcm) = bytes(slow_spec());
        let wrap = |bytes| OriginalInput {
            bytes,
            dropped: dropped.clone(),
            control: shared.clone(),
        };
        let job = encoder
            .try_admit(budget.try_open_session().unwrap(), slow_spec())
            .unwrap()
            .unwrap()
            .start(wrap(rgb), wrap(pcm))
            .unwrap();
        let control = job.cancellation_handle();
        *shared.lock().unwrap() = Some(control.clone());
        await_started(&control).await;
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        assert!(!control.process_reaped());
        if abandon_waiter {
            drop(job);
        } else {
            control.cancel();
            let error = job.wait().await.unwrap_err();
            assert_eq!(error.kind, Mp4ErrorKind::Cancelled);
            assert!(error.process.as_ref().unwrap().kill_requested);
            assert!(error.process.as_ref().unwrap().reaped);
            drop(error);
        }
        tokio::time::timeout(Duration::from_secs(20), async {
            while dropped.load(Ordering::SeqCst) != 2
                || store.usage().reserved_files != 0
                || budget.usage().sessions != 0
                || budget.usage().encoded_bytes != 0
            {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        assert!(control.process_reaped());
        assert_refunded(&store, &budget);
    }
}

#[tokio::test]
#[ignore = "requires ffmpeg/libx264/AAC"]
async fn real_ffmpeg_deadline_reaps_the_started_process() {
    let directory = tempfile::tempdir().unwrap();
    let store = spool(&directory, 256 << 20);
    let mut config = config(tool("ffmpeg"));
    config.timeout = Duration::from_millis(100);
    config.preset.h264_speed = H264SpeedPreset::Veryslow;
    let encoder = Mp4Encoder::new(config, store.clone()).unwrap();
    let budget = budget();
    // Many tiny frames keep raw staging small while the real encoder has
    // sustained work. Assert that timeout reaches an actually spawned child.
    let spec = VideoAudioSpec {
        width: 16,
        height: 16,
        frames: 5000,
        ..spec()
    };
    let (rgb, pcm) = bytes(spec);
    let job = encoder
        .try_admit(budget.try_open_session().unwrap(), spec)
        .unwrap()
        .unwrap()
        .start(rgb, pcm)
        .unwrap();
    let control = job.cancellation_handle();
    await_started(&control).await;
    let error = job.wait().await.unwrap_err();
    assert_eq!(error.kind, Mp4ErrorKind::Deadline);
    let process = error.process.as_ref().unwrap();
    assert!(process.kill_requested && process.reaped);
    drop(error);
    assert_refunded(&store, &budget);
}
