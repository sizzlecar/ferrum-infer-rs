use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

mod benchmark;

fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: bytes,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn config() -> FastH3VideoVaeConfig {
    crate::fast_h3::video_vae::tests::config()
}
fn limits() -> FastH3VideoTemporalLimits {
    FastH3VideoTemporalLimits {
        maximum_output_frames: 1000,
        maximum_buffer_bytes: 16384,
    }
}
fn raw(host: &HostMemoryAccount, value: f32) -> FastH3VideoRawClip {
    let shape = [2, 3, 28, 2, 3];
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::Output, 2 * shape.iter().product::<usize>())
        .unwrap();
    for (bc, plane) in bytes.as_mut().chunks_exact_mut(28 * 2 * 3 * 2).enumerate() {
        let bits = bf16::from_f32(value * (1u32 << bc) as f32)
            .to_bits()
            .to_le_bytes();
        for word in plane.chunks_exact_mut(2) {
            word.copy_from_slice(&bits);
        }
    }
    FastH3VideoRawClip {
        shape,
        bytes: bytes.freeze(),
    }
}
#[test]
fn temporal_bf16_weights_products_and_sum_have_independent_rounding_witness() {
    // Direct dyadic result from separately rounded 1 - BF16(j/5), products,
    // then sum; j=2 differs from a single final BF16 round of linear blending.
    let expected = [1f32, 1.203125, 1.40625, 1.6015625, 1.796875];
    for (i, e) in expected.into_iter().enumerate() {
        assert_eq!(
            blend(0x3f80, 0x4000, i, 5).unwrap(),
            bf16::from_f32(e).to_bits()
        );
    }
    assert_ne!(
        blend(0x3f80, 0x4000, 2, 5).unwrap(),
        bf16::from_f32(1.4).to_bits()
    );
    assert_eq!(blend(0x7f7f, 0x7f7f, 2, 5).unwrap(), 0x7f7f);
}
#[test]
fn temporal_transaction_preserves_old_overlap_on_drop_capacity_failure_and_stale_input() {
    let host = host(32768);
    let a = raw(&host, 1.);
    let b = raw(&host, 2.);
    let baseline = host.usage().unwrap().allocated_bytes();
    let mut state = FastH3VideoTemporalAssembler::new(&config(), 11, 2, 2, 3, limits()).unwrap();
    assert_eq!(
        (state.plan().chunk_count(), state.plan().output_frames()),
        (2, 35)
    );
    assert!(state.prepare_finish(&host).is_err());
    let pending = state.prepare_chunk(0, &a, &host).unwrap();
    assert_eq!(pending.output().unwrap().ncthw_shape(), [2, 3, 17, 2, 3]);
    drop(pending);
    assert_eq!(host.usage().unwrap().allocated_bytes(), baseline);
    assert_eq!(state.next_chunk(), 0);
    let first = state.prepare_chunk(0, &a, &host).unwrap().commit().unwrap();
    assert_eq!((state.next_chunk(), state.published_frames()), (1, 17));
    let old = state.overlap.as_ref().unwrap().bytes().to_vec();
    assert!(state.prepare_chunk(0, &b, &host).is_err());
    let insufficient = super::tests::host(2 * 3 * 17 * 2 * 3 * 2); // publish admitted; next overlap refused
    assert!(state.prepare_chunk(1, &b, &insufficient).is_err());
    assert_eq!(insufficient.usage().unwrap().allocated_bytes(), 0);
    assert_eq!((state.next_chunk(), state.published_frames()), (1, 17));
    assert_eq!(state.overlap.as_ref().unwrap().bytes(), old);
    drop(state.prepare_chunk(1, &b, &host).unwrap());
    assert_eq!(state.overlap.as_ref().unwrap().bytes(), old);
    let second = state.prepare_chunk(1, &b, &host).unwrap().commit().unwrap();
    let blend_expected = [1f32, 1.203125, 1.40625, 1.6015625, 1.796875];
    for (bc, channel) in second
        .bytes()
        .bytes()
        .chunks_exact(17 * 2 * 3 * 2)
        .enumerate()
    {
        for (t, frame) in channel.chunks_exact(2 * 3 * 2).enumerate() {
            let expected =
                bf16::from_f32((if t < 5 { blend_expected[t] } else { 2. }) * (1u32 << bc) as f32)
                    .to_bits()
                    .to_le_bytes();
            for word in frame.chunks_exact(2) {
                assert_eq!(word, expected);
            }
        }
    }
    assert_eq!(second.output_frame_start(), 17);
    assert!(state.prepare_finish(&super::tests::host(71)).is_err());
    assert!(!state.is_finished());
    drop(state.prepare_finish(&host).unwrap());
    assert!(!state.is_finished());
    assert_eq!(state.published_frames(), 34);
    let tail = state.prepare_finish(&host).unwrap().commit().unwrap();
    assert_eq!(
        (tail.output_frame_start(), tail.ncthw_shape()),
        (34, [2, 3, 1, 2, 3])
    );
    for (bc, plane) in tail.bytes().bytes().chunks_exact(2 * 3 * 2).enumerate() {
        for word in plane.chunks_exact(2) {
            assert_eq!(
                word,
                bf16::from_f32(2. * (1u32 << bc) as f32)
                    .to_bits()
                    .to_le_bytes()
            );
        }
    }
    assert!(state.is_finished());
    assert_eq!(state.published_frames(), 35);
    assert!(state.overlap.is_none());
    assert!(state.prepare_finish(&host).is_err());
    drop((first, second, tail, a, b, state));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
#[test]
fn short_sequence_zero_tail_and_nonfinite_rejection_are_transactional() {
    let h = host(32768);
    let a = raw(&h, 1.);
    let mut short = FastH3VideoTemporalAssembler::new(&config(), 1, 2, 2, 3, limits()).unwrap();
    let chunk = short.prepare_chunk(0, &a, &h).unwrap().commit().unwrap();
    assert_eq!(chunk.ncthw_shape()[2], 1);
    assert!(short.prepare_finish(&h).unwrap().commit().is_none());
    assert!(short.is_finished());
    let shape = [2, 3, 28, 2, 3];
    let mut bytes = h
        .allocate_zeroed(HostMemoryClass::Output, 2 * shape.iter().product::<usize>())
        .unwrap();
    for word in bytes.as_mut().chunks_exact_mut(2) {
        word.copy_from_slice(&0x7f7fu16.to_le_bytes());
    }
    let huge = FastH3VideoRawClip {
        shape,
        bytes: bytes.freeze(),
    };
    let mut state = FastH3VideoTemporalAssembler::new(&config(), 11, 2, 2, 3, limits()).unwrap();
    drop(state.prepare_chunk(0, &huge, &h).unwrap().commit());
    let mut invalid = h
        .allocate_zeroed(HostMemoryClass::Output, huge.bytes().len())
        .unwrap();
    invalid.as_mut().copy_from_slice(huge.bytes().bytes());
    invalid.as_mut()[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
    let invalid = FastH3VideoRawClip {
        shape,
        bytes: invalid.freeze(),
    };
    let before = h.usage().unwrap().allocated_bytes();
    assert!(state.prepare_chunk(1, &invalid, &h).is_err());
    assert_eq!((state.next_chunk(), state.published_frames()), (1, 17));
    assert_eq!(h.usage().unwrap().allocated_bytes(), before);
    drop((a, huge, invalid, chunk, state, short));
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn rgb_conversion_preserves_batch_time_and_interleaves_original_channel_statistics() {
    let h = host(256);
    let mut input = h
        .allocate_zeroed(HostMemoryClass::Output, 2 * 2 * 3 * 2)
        .unwrap();
    for (b, samples) in [[0f32, 1.], [-1., 100.]].into_iter().enumerate() {
        for c in 0..3 {
            for (t, x) in samples.into_iter().enumerate() {
                let i = 2 * ((b * 3 + c) * 2 + t);
                input.as_mut()[i..i + 2]
                    .copy_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
            }
        }
    }
    let chunk = FastH3VideoFrameChunk {
        start: 17,
        shape: [2, 3, 2, 1, 1],
        bytes: input.freeze(),
    };
    assert!(chunk.managed_rgb8(&h, 11).is_err());
    let rgb = chunk.managed_rgb8(&h, 12).unwrap();
    assert_eq!(rgb.output_frame_start(), 17);
    assert_eq!(rgb.nthwc_shape(), [2, 2, 1, 1, 3]);
    assert_eq!(
        rgb.bytes().bytes(),
        [123, 116, 103, 182, 173, 160, 65, 59, 46, 255, 255, 255]
    );
    drop(chunk);
    assert_eq!(h.usage().unwrap().allocated_bytes(), 12);
    drop(rgb);
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn rgb_conversion_rejects_nonfinite_words_and_returns_the_new_output_grant() {
    let h = host(64);
    for word in [0x7f80u16, 0xff80, 0x7f81, 0xff81, 0x7fc0, 0xffff] {
        for channel in 0..3 {
            let mut input = h.allocate_zeroed(HostMemoryClass::Output, 6).unwrap();
            input.as_mut()[2 * channel..2 * channel + 2].copy_from_slice(&word.to_le_bytes());
            let chunk = FastH3VideoFrameChunk {
                start: 0,
                shape: [1, 3, 1, 1, 1],
                bytes: input.freeze(),
            };
            assert!(chunk.managed_rgb8(&h, 3).is_err());
            assert_eq!(h.usage().unwrap().allocated_bytes(), 6);
            drop(chunk);
            assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
        }
    }
}
