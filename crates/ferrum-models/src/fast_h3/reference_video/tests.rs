use super::*;

fn limits() -> FastH3ReferenceVideoLimits {
    FastH3ReferenceVideoLimits {
        image: ImageInputLimits::new(1 << 20, 1 << 20, 4096, 4096, 16 << 20).unwrap(),
        maximum_source_frames: 8192,
        maximum_frames: 362,
        maximum_output_bytes: 2 << 30,
    }
}
fn geometry(frames: usize, n: u32, d: u32) -> FastH3DecodedVideoGeometry {
    FastH3DecodedVideoGeometry {
        width: 32,
        height: 32,
        frames,
        frame_rate_numerator: n,
        frame_rate_denominator: d,
    }
}
fn canvas() -> FastH3ReferenceVideoCanvas {
    FastH3ReferenceVideoCanvas {
        short_edge: 32,
        maximum_pixels: 1024,
    }
}

#[test]
fn reference_timeline_matches_released_resampling_and_keeps_qwen_before_vae_trim() {
    for (n, d, expected) in [
        (12, 1, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        (24, 1, vec![0, 1, 2, 3, 4]),
        (30, 1, vec![0, 1, 3, 4]),
        (48, 1, vec![0, 2, 4]),
        (30000, 1001, vec![0, 1, 3, 4]),
    ] {
        let p = FastH3ReferenceVideoPlan::new(geometry(5, n, d), 124, canvas(), limits()).unwrap();
        assert_eq!(p.source_frame_indices(), expected);
    }
    let p = FastH3ReferenceVideoPlan::new(geometry(38, 24, 1), 124, canvas(), limits()).unwrap();
    assert_eq!(p.qwen_frame_indices(), [0, 12, 24, 36]);
    assert_eq!(p.block_timestamps(), [0.25, 1.25]);
    let p = FastH3ReferenceVideoPlan::new(geometry(25, 24, 1), 124, canvas(), limits()).unwrap();
    assert_eq!(p.qwen_frame_indices(), [0, 12, 24]);
    assert_eq!(p.block_timestamps(), [0.25, 1.]);
    let p = FastH3ReferenceVideoPlan::new(geometry(100, 24, 1), 13, canvas(), limits()).unwrap();
    assert_eq!(p.source_frame_indices().len(), 13);
    assert_eq!(p.block_timestamps(), [0.25]);
    assert_eq!(
        FastH3ReferenceVideoPlan::dimensions(16, 9, Default::default()).unwrap(),
        (1344, 768)
    );
    assert_eq!(
        FastH3ReferenceVideoPlan::dimensions(4, 1, Default::default()).unwrap(),
        (2016, 512)
    );
    assert!(FastH3ReferenceVideoPlan::new(geometry(5, 24, 0), 124, canvas(), limits()).is_err());
}

#[test]
fn streaming_frames_share_canvas_repeat_sources_and_refund_failed_preparation() {
    let p = FastH3ReferenceVideoPlan::new(geometry(3, 12, 1), 5, canvas(), limits()).unwrap();
    let account = |bytes| {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: bytes,
                    output_bytes: 0,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    };
    let host = account(1 << 20);
    let make = |index: usize| -> Result<ManagedRgba8Image> {
        let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, 32 * 32 * 4)?;
        for pixel in bytes.as_mut().chunks_exact_mut(4) {
            pixel.copy_from_slice(&[index as u8, 100, 200, 0]);
        }
        Ok(ManagedRgba8Image::from_rgba8(
            &host,
            bytes.freeze(),
            32,
            32,
            limits().image,
        )?)
    };
    let mut calls = vec![];
    let output = p
        .prepare_with_frames(&host, |i| {
            calls.push(i);
            make(i)
        })
        .unwrap();
    assert_eq!(calls, [0, 1, 2]);
    for (c, channel) in output.bytes().bytes().chunks_exact(5 * 1024).enumerate() {
        for (t, frame) in channel.chunks_exact(1024).enumerate() {
            assert!(frame
                .iter()
                .all(|&v| v == [p.source_frames[t] as u8, 100, 200][c]));
        }
    }
    drop(output);
    assert!(p
        .prepare_with_frames(&host, |i| if i == 1 {
            anyhow::bail!("truncated codec frame")
        } else {
            make(i)
        })
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let small = account(1024);
    let mut touched = false;
    assert!(p
        .prepare_with_frames(&small, |_| {
            touched = true;
            make(0)
        })
        .is_err());
    assert!(!touched);
    assert_eq!(small.usage().unwrap().committed_bytes(), 0);
    assert_eq!(
        p.input_peak_bytes(&host).unwrap(),
        host.backing_bytes(5 * 3 * 1024).unwrap() + host.backing_bytes(4 * 1024).unwrap()
    );
}
