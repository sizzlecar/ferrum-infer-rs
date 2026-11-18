use super::*;
use std::{io::Write, process::Command, sync::atomic::Ordering, time::Instant};

mod decoder_capacity;

fn tool(name: &str) -> PathBuf {
    // Tool discovery selects the test host's installed codec, never a machine
    // identity or a hidden product path. Production always requires paths.
    let output = Command::new("which").arg(name).output().unwrap();
    assert!(output.status.success(), "requires installed {name}");
    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
}
fn config(directory: &std::path::Path) -> VideoInputCodecConfig {
    VideoInputCodecConfig {
        ffmpeg: tool("ffmpeg"),
        ffprobe: tool("ffprobe"),
        staging_directory: directory.join("staging"),
        threads: 1,
        timeout: Duration::from_secs(10),
        maximum_duration: Duration::from_secs(2),
        maximum_encoded_bytes: 1 << 20,
        maximum_metadata_bytes: 64 << 10,
        maximum_stderr_bytes: 8192,
        maximum_source_frames: 24,
        image: ImageInputLimits::new(1 << 20, 1 << 20, 256, 256, 256 * 256).unwrap(),
    }
}
fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: 0,
                transfer_staging_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn fixture(
    config: &VideoInputCodecConfig,
    directory: &std::path::Path,
    host: &HostMemoryAccount,
) -> ManagedHostBytes {
    fixture_sized(config, directory, host, 32, 32)
}
fn fixture_sized(
    config: &VideoInputCodecConfig,
    directory: &std::path::Path,
    host: &HostMemoryAccount,
    width: u8,
    height: u8,
) -> ManagedHostBytes {
    let raw = directory.join("frames.rgba");
    let video = directory.join("source.mkv");
    let mut file = std::fs::File::create(&raw).unwrap();
    for frame in 0..6u8 {
        for y in 0..height {
            for x in 0..width {
                file.write_all(&[frame * 20, x.wrapping_mul(7), y.wrapping_mul(7), 255])
                    .unwrap();
            }
        }
    }
    drop(file);
    let status = Command::new(&config.ffmpeg)
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            "12",
            "-i",
        ])
        .arg(&raw)
        .args([
            "-frames:v",
            "6",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgra",
            "-threads",
            "1",
        ])
        .arg(&video)
        .status()
        .unwrap();
    assert!(status.success());
    let bytes = std::fs::read(video).unwrap();
    let mut admitted = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    admitted.as_mut().copy_from_slice(&bytes);
    admitted.freeze()
}

#[test]
#[ignore = "requires installed ffmpeg/ffprobe with lossless FFV1"]
fn real_codec_rejects_playlists_that_read_another_local_media_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = config(directory.path());
    config.staging_directory = directory.path().to_owned();
    let host = account(1 << 20);
    drop(fixture(&config, directory.path(), &host));
    let playlist = b"ffconcat version 1.0\nfile 'source.mkv'\n";
    let path = directory.path().join("playlist.ffconcat");
    std::fs::write(&path, playlist).unwrap();
    // Establish that this fixture really opens the neighboring video without
    // the product's container boundary, rather than failing as malformed data.
    let unrestricted = Command::new(&config.ffprobe)
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
        ])
        .arg(&path)
        .output()
        .unwrap();
    assert!(unrestricted.status.success());
    let metadata: serde_json::Value = serde_json::from_slice(&unrestricted.stdout).unwrap();
    assert_eq!(metadata["streams"][0]["width"], 32);
    assert_eq!(metadata["streams"][0]["height"], 32);
    assert_eq!(metadata["streams"][0]["nb_read_frames"], "6");
    let retained_files = || {
        let mut paths = std::fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        paths.sort();
        paths
    };
    let before = retained_files();
    let mut encoded = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, playlist.len())
        .unwrap();
    encoded.as_mut().copy_from_slice(playlist);
    let codec = VideoInputCodec::new(config).unwrap();
    assert!(codec
        .inspect(encoded.freeze(), &AtomicBool::new(false))
        .is_err());
    assert_eq!(retained_files(), before);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires installed ffprobe"]
fn real_codec_rejects_reference_playlists_without_opening_network_sources() {
    use std::net::TcpListener;
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let playlist = format!(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\nhttp://{}/segment.ts\n#EXT-X-ENDLIST\n",
        listener.local_addr().unwrap()
    );
    let directory = tempfile::tempdir().unwrap();
    let mut config = config(directory.path());
    config.timeout = Duration::from_secs(1);
    let host = account(1 << 20);
    let mut encoded = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, playlist.len())
        .unwrap();
    encoded.as_mut().copy_from_slice(playlist.as_bytes());
    let encoded = encoded.freeze();
    let baseline = host.usage().unwrap().committed_bytes();
    let codec = VideoInputCodec::new(config.clone()).unwrap();
    assert!(codec
        .inspect(encoded.clone(), &AtomicBool::new(false))
        .is_err());
    assert!(
        matches!(listener.accept(), Err(e) if e.kind() == std::io::ErrorKind::WouldBlock),
        "an uploaded video container must not initiate an external media fetch"
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), baseline);
    assert_eq!(
        std::fs::read_dir(&config.staging_directory)
            .unwrap()
            .count(),
        0
    );
    drop(encoded);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires installed ffmpeg/ffprobe with lossless FFV1"]
fn real_codec_probes_and_streams_selected_lossless_frames_with_original_host_ownership() {
    let directory = tempfile::tempdir().unwrap();
    let config = config(directory.path());
    let host = account(1 << 20);
    let encoded = fixture(&config, directory.path(), &host);
    let baseline = host.usage().unwrap().committed_bytes();
    let codec = VideoInputCodec::new(config.clone()).unwrap();
    let cancelled = AtomicBool::new(false);
    let plan = codec.inspect(encoded.clone(), &cancelled).unwrap();
    let g = plan.geometry();
    assert_eq!(
        (
            g.width,
            g.height,
            g.frames,
            g.frame_rate_numerator,
            g.frame_rate_denominator
        ),
        (32, 32, 6, 12, 1)
    );
    assert!(plan.audio().is_none());
    let output = plan
        .with_frames(6, &host, &cancelled, |reader| {
            Ok([reader.read(0)?, reader.read(5)?])
        })
        .unwrap();
    for (frame, image) in [0u8, 5].into_iter().zip(&output) {
        assert!(host.owns(image.bytes()));
        for (i, pixel) in image.bytes().bytes().chunks_exact(4).enumerate() {
            assert_eq!(
                pixel,
                [frame * 20, (i % 32) as u8 * 7, (i / 32) as u8 * 7, 255]
            );
        }
    }
    assert_eq!(
        std::fs::read_dir(&config.staging_directory)
            .unwrap()
            .count(),
        0
    );
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), baseline);
    // A real admission refusal remains recognizable after the child is killed
    // and reaped; the wrapper must not replace it with a generic codec error.
    let remaining = host.limits().class_bytes(HostMemoryClass::InputPreprocess)
        - host
            .usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::InputPreprocess);
    let pressure = host
        .reserve(HostMemoryClass::InputPreprocess, remaining)
        .unwrap();
    let error = plan
        .with_frames(6, &host, &cancelled, |reader| reader.read(0))
        .unwrap_err();
    assert!(
        error.downcast_ref::<HostMemoryError>().is_some(),
        "{error:?}"
    );
    assert_eq!(
        std::fs::read_dir(&config.staging_directory)
            .unwrap()
            .count(),
        0
    );
    drop(pressure);
    let retried = plan
        .with_frames(1, &host, &cancelled, |reader| reader.read(0))
        .unwrap();
    drop((retried, plan, encoded));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires installed ffmpeg/ffprobe with FFV1 and PCM"]
fn real_soundtrack_preserves_pcm_channels_resamples_mono_and_retains_full_backing() {
    for (rate, channels) in [(32000u32, 2usize), (48000, 1)] {
        let directory = tempfile::tempdir().unwrap();
        let config = config(directory.path());
        let host = account(2 << 20);
        drop(fixture(&config, directory.path(), &host));
        let pcm = directory.path().join("audio.f32");
        let mut file = std::fs::File::create(&pcm).unwrap();
        for i in 0..rate / 10 {
            for c in 0..channels {
                let x = ((i % 127) as f32 - 63.) / 128. * if c == 0 { 1. } else { -0.5 };
                file.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        drop(file);
        let path = directory.path().join("with_audio.mkv");
        let status = Command::new(&config.ffmpeg)
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(directory.path().join("source.mkv"))
            .args([
                "-f",
                "f32le",
                "-ar",
                &rate.to_string(),
                "-ac",
                &channels.to_string(),
                "-i",
            ])
            .arg(&pcm)
            .args([
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_f32le",
            ])
            .arg(&path)
            .status()
            .unwrap();
        assert!(status.success());
        let raw = std::fs::read(path).unwrap();
        let mut encoded = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, raw.len())
            .unwrap();
        encoded.as_mut().copy_from_slice(&raw);
        let codec = VideoInputCodec::new(config).unwrap();
        let plan = codec
            .inspect(encoded.freeze(), &AtomicBool::new(false))
            .unwrap();
        let baseline = host.usage().unwrap().committed_bytes();
        let audio = plan
            .prepare_audio(32000, 6400, &host, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(audio.tensor().dimensions(), [2, 1, 3200]);
        assert!(audio.bytes().backing_bytes() >= 6400 * 8);
        let values = audio
            .bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        for i in 0..3200 {
            assert!(values[i].is_finite());
            if channels == 2 {
                let expected = ((i % 127) as f32 - 63.) / 128.;
                assert_eq!(values[i], expected);
                assert_eq!(values[3200 + i], expected * -0.5);
            } else {
                assert_eq!(values[i], values[3200 + i]);
            }
        }
        drop(audio);
        assert_eq!(host.usage().unwrap().committed_bytes(), baseline);
        let trimmed = plan
            .prepare_audio(32000, 1600, &host, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(trimmed.tensor().dimensions(), [2, 1, 1600]);
        drop((trimmed, plan));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
#[ignore = "requires installed ffmpeg/ffprobe"]
fn real_codec_deadline_and_cancel_reap_before_staging_or_source_credit_is_released() {
    let directory = tempfile::tempdir().unwrap();
    let config = config(directory.path());
    let host = account(1 << 20);
    let encoded = fixture(&config, directory.path(), &host);
    for cancel in [false, true] {
        let mut config = config.clone();
        if !cancel {
            config.timeout = Duration::from_millis(150);
        }
        let codec = VideoInputCodec::new(config.clone()).unwrap();
        let cancelled = AtomicBool::new(false);
        let started = Instant::now();
        let before = host.usage().unwrap().committed_bytes();
        let error = std::thread::scope(|scope| {
            if cancel {
                scope.spawn(|| {
                    std::thread::sleep(Duration::from_millis(150));
                    cancelled.store(true, Ordering::Release);
                });
            }
            process::run(
                &codec,
                &config.ffmpeg,
                &encoded,
                ["-vf", "loop=loop=-1:size=6:start=0", "-f", "null", "-"]
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                true,
                &cancelled,
                |stdout| process::read_capped(stdout, 16),
            )
            .unwrap_err()
        });
        assert!(
            matches!(
                error.downcast_ref::<VideoInputInterruption>(),
                Some(VideoInputInterruption::Cancelled)
            ) == cancel,
            "{error:?}"
        );
        assert!(
            matches!(
                error.downcast_ref::<VideoInputInterruption>(),
                Some(VideoInputInterruption::Deadline)
            ) != cancel,
            "{error:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        assert_eq!(
            std::fs::read_dir(&config.staging_directory)
                .unwrap()
                .count(),
            0
        );
    }
    drop(encoded);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
