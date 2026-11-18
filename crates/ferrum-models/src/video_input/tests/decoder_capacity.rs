use super::*;

struct ObservedProcess {
    result: Result<()>,
    stdout: Vec<u8>,
    read_to_end: bool,
}

fn observe(codec: &VideoInputCodec, encoded: &ManagedHostBytes, decode: bool) -> ObservedProcess {
    let config = codec.config();
    let (program, arguments, maximum_stdout) = if decode {
        (
            &config.ffmpeg,
            vec![
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-frames:v",
                "6",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgba",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            6 * 64 * 64 * 4,
        )
    } else {
        (
            &config.ffprobe,
            vec![
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                // Keep each field independently readable if ffprobe exits
                // mid-report after the decoder refuses a frame.
                "default=noprint_wrappers=1",
            ],
            config.maximum_metadata_bytes,
        )
    };
    let mut stdout = Vec::new();
    let mut read_to_end = false;
    // Use the product's actual child owner, input options and wait path, but
    // bypass inspect()'s geometry rejection. The same reader accepts the full
    // oversized fixture under the larger budget, so it cannot manufacture the
    // decoder rejection by truncating output or cancelling the child.
    let result = process::run(
        codec,
        program,
        encoded,
        arguments.into_iter().map(Into::into).collect(),
        decode,
        &AtomicBool::new(false),
        |reader| {
            stdout = process::read_capped(reader, maximum_stdout)?;
            read_to_end = true;
            Ok(())
        },
    );
    ObservedProcess {
        result,
        stdout,
        read_to_end,
    }
}

fn assert_probe(observed: &ObservedProcess, width: u32, height: u32) {
    assert!(observed.read_to_end);
    assert!(observed.result.is_ok(), "{:?}", observed.result);
    let metadata: std::collections::BTreeMap<_, _> = std::str::from_utf8(&observed.stdout)
        .unwrap()
        .lines()
        .map(|line| line.split_once('=').unwrap())
        .collect();
    assert_eq!(metadata["width"].parse::<u32>().unwrap(), width);
    assert_eq!(metadata["height"].parse::<u32>().unwrap(), height);
    assert_eq!(metadata["nb_read_frames"], "6");
}

fn assert_pixels(bytes: &[u8], width: usize, height: usize) {
    assert_eq!(bytes.len(), 6 * width * height * 4);
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        let frame = index / (width * height);
        let x = index % width;
        let y = index / width % height;
        assert_eq!(
            pixel,
            [(frame * 20) as u8, (x * 7) as u8, (y * 7) as u8, 255]
        );
    }
}

fn assert_released(codec: &VideoInputCodec, host: &HostMemoryAccount, baseline: usize) {
    assert_eq!(host.usage().unwrap().committed_bytes(), baseline);
    assert_eq!(
        std::fs::read_dir(&codec.config().staging_directory)
            .unwrap()
            .count(),
        0
    );
}

#[test]
#[ignore = "requires installed ffmpeg/ffprobe with lossless FFV1"]
fn real_codec_rejects_oversized_frames_in_probe_and_decode_before_geometry_validation() {
    let directory = tempfile::tempdir().unwrap();
    let recovery_directory = tempfile::tempdir().unwrap();
    let host = account(1 << 20);
    let mut wide = config(directory.path());
    wide.threads = 2;
    wide.image = ImageInputLimits::new(1 << 20, 64 * 64, 64, 64, 1 << 20).unwrap();
    let oversized = fixture_sized(&wide, directory.path(), &host, 64, 64);
    let recovery = fixture(&wide, recovery_directory.path(), &host);
    let baseline = host.usage().unwrap().committed_bytes();
    let mut narrow = wide.clone();
    narrow.image = ImageInputLimits::new(1 << 20, 33 * 32, 33, 32, 1 << 20).unwrap();
    let wide = VideoInputCodec::new(wide).unwrap();
    let narrow = VideoInputCodec::new(narrow).unwrap();

    assert_probe(&observe(&wide, &oversized, false), 64, 64);
    assert_released(&wide, &host, baseline);
    let decoded = observe(&wide, &oversized, true);
    assert!(decoded.read_to_end);
    assert!(decoded.result.is_ok(), "{:?}", decoded.result);
    assert_pixels(&decoded.stdout, 64, 64);
    assert_released(&wide, &host, baseline);

    let probed = observe(&narrow, &oversized, false);
    assert!(probed.read_to_end, "{:?}", probed.result);
    let error = probed.result.as_ref().unwrap_err();
    assert!(error.downcast_ref::<VideoInputInterruption>().is_none());
    for count in std::str::from_utf8(&probed.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("nb_read_frames="))
    {
        assert!(
            matches!(count, "0" | "N/A"),
            "probe decoded {count} oversized frames"
        );
    }
    assert_released(&narrow, &host, baseline);

    let decoded = observe(&narrow, &oversized, true);
    assert!(decoded.read_to_end, "{:?}", decoded.result);
    let error = decoded.result.as_ref().unwrap_err();
    assert!(error.downcast_ref::<VideoInputInterruption>().is_none());
    assert!(
        decoded.stdout.is_empty(),
        "oversized RGBA reached the reader"
    );
    assert_released(&narrow, &host, baseline);

    // A rejected source must leave the same codec owner usable for a later
    // admitted source, through the public inspection/frame-reading flow too.
    let plan = narrow
        .inspect(recovery.clone(), &AtomicBool::new(false))
        .unwrap();
    let pixels = plan
        .with_frames(6, &host, &AtomicBool::new(false), |reader| {
            let mut pixels = Vec::new();
            for index in 0..6 {
                pixels.extend_from_slice(reader.read(index)?.bytes().bytes());
            }
            Ok(pixels)
        })
        .unwrap();
    assert_pixels(&pixels, 32, 32);
    drop(plan);
    assert_released(&narrow, &host, baseline);
    drop((oversized, recovery));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires installed ffmpeg/ffprobe with lossless FFV1"]
fn real_codec_preserves_unaligned_frames_at_the_visible_pixel_limit() {
    let directory = tempfile::tempdir().unwrap();
    let host = account(1 << 20);
    let mut config = config(directory.path());
    config.image = ImageInputLimits::new(1 << 20, 1056, 33, 32, 1 << 20).unwrap();
    let encoded = fixture_sized(&config, directory.path(), &host, 33, 32);
    let baseline = host.usage().unwrap().committed_bytes();
    let codec = VideoInputCodec::new(config.clone()).unwrap();
    let plan = codec
        .inspect(encoded.clone(), &AtomicBool::new(false))
        .unwrap();
    assert_eq!((plan.geometry().width, plan.geometry().height), (33, 32));
    let pixels = plan
        .with_frames(6, &host, &AtomicBool::new(false), |reader| {
            let mut pixels = Vec::new();
            for index in 0..6 {
                pixels.extend_from_slice(reader.read(index)?.bytes().bytes());
            }
            Ok(pixels)
        })
        .unwrap();
    assert_pixels(&pixels, 33, 32);
    drop(plan);
    assert_released(&codec, &host, baseline);

    // Decoder row-padding allowance must not widen the public visible-pixel
    // limit. The same complete video is one visible pixel over this budget.
    config.image = ImageInputLimits::new(1 << 20, 1055, 33, 32, 1 << 20).unwrap();
    let codec = VideoInputCodec::new(config).unwrap();
    assert_probe(&observe(&codec, &encoded, false), 33, 32);
    assert!(codec
        .inspect(encoded.clone(), &AtomicBool::new(false))
        .is_err());
    assert_released(&codec, &host, baseline);
    drop(encoded);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
