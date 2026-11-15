use super::*;
fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn tensor(
    host: &HostMemoryAccount,
    batch: u64,
    values: &[f32],
) -> (ResolvedTensorSpec, ManagedHostBytes) {
    (
        ResolvedTensorSpec::new(
            vec![batch, 4, 2, 4],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        stored_values(host, values),
    )
}
fn stored_values(host: &HostMemoryAccount, values: &[f32]) -> ManagedHostBytes {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::Output, values.len() * 4)
        .unwrap();
    for (slot, value) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes.freeze()
}

fn id() -> ProgramValueId {
    ProgramValueId::new("value.test.keyframe").unwrap()
}
#[test]
fn keyframe_posterior_applies_clamp_f16_boundary_and_original_channel_statistics() {
    let config = super::super::tests::config();
    let means: [f32; 8] = [0.3333, -0.127, 1., 0., 0.75, -1.25, 0.0003, 0.];
    let logvar: [f32; 8] = [0., 0., -40., 25., 0., 0., -30., 20.];
    let noise: [f32; 8] = [0.125, -0.0625, 0.5, 0.75, -0.25, 0.125, -0.5, 0.25];
    let moments = means
        .into_iter()
        .chain(logvar)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut output = noise
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    sample_and_normalize(&config, 4, &moments, &mut output).unwrap();
    let mut changed_by_f16 = false;
    for (i, actual) in output.chunks_exact(4).map(read).enumerate() {
        // Independent F64 exp; the declared boundary rounds the sample through
        // IEEE half before subtracting the channel mean and dividing by std.
        let sample = f64::from(means[i])
            + (f64::from(logvar[i]).clamp(-30., 20.) * 0.5).exp() * f64::from(noise[i]);
        let rounded = f16::from_f64(sample).to_f32();
        let expected =
            (rounded - config.latent_mean()[i / 4] as f32) / config.latent_std()[i / 4] as f32;
        assert_eq!(actual, expected);
        changed_by_f16 |= rounded != sample as f32;
    }
    assert!(changed_by_f16);
}
#[test]
fn keyframe_posterior_reseeds_each_reference_and_admits_only_finite_bounded_storage() {
    let config = super::super::tests::config();
    let host = account(1 << 16);
    let first = (0..32)
        .map(|i| if i < 16 { i as f32 / 32. } else { 0. })
        .collect::<Vec<_>>();
    let single = tensor(&host, 1, &first);
    let batch = tensor(
        &host,
        2,
        &first.iter().chain(&first).copied().collect::<Vec<_>>(),
    );
    let a = sample_fast_h3_keyframe_posterior(&config, &single.0, &single.1, id(), &host).unwrap();
    let b = sample_fast_h3_keyframe_posterior(&config, &batch.0, &batch.1, id(), &host).unwrap();
    assert_eq!(a.tensor().dimensions(), [1, 2, 1, 2, 4]);
    assert_eq!(&b.bytes().bytes()[..64], a.bytes().bytes());
    assert_eq!(&b.bytes().bytes()[64..], a.bytes().bytes());
    let tiny = account(64 + SCRATCH_BYTES - 1);
    assert!(sample_fast_h3_keyframe_posterior(&config, &single.0, &single.1, id(), &tiny).is_err());
    assert_eq!(tiny.usage().unwrap().committed_bytes(), 0);
    let mut invalid = first;
    invalid[0] = f32::NAN;
    let bad = tensor(&host, 1, &invalid);
    let before = host.usage().unwrap().committed_bytes();
    assert!(sample_fast_h3_keyframe_posterior(&config, &bad.0, &bad.1, id(), &host).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    drop((single, batch, a, b, bad));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn video_posterior_preserves_ncthw_channel_statistics_across_all_frames() {
    let config = super::super::tests::config();
    let host = account(1 << 16);
    let per_channel = 3 * 2 * 3;
    let per_reference = 2 * per_channel;
    let mut values = vec![];
    for batch in 0..2 {
        values.extend((0..per_reference).map(|i| 0.5 + (i + batch * 7) as f32 / 64.));
        // Clamped variance is far below a half-precision unit at these means;
        // the exact rounded mean independently reveals channel/time indexing.
        values.extend(std::iter::repeat_n(-40f32, per_reference));
    }
    let bytes = stored_values(&host, &values);
    let shape = ResolvedTensorSpec::new(
        vec![2, 4, 3, 2, 3],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let output = sample_fast_h3_video_posterior(&config, &shape, &bytes, id(), &host).unwrap();
    assert_eq!(output.tensor().dimensions(), [2, 2, 3, 2, 3]);
    for (i, actual) in output.bytes().bytes().chunks_exact(4).map(read).enumerate() {
        let batch = i / per_reference;
        let local = i % per_reference;
        let channel = local / per_channel;
        let mean = values[batch * per_reference * 2 + local];
        assert_eq!(
            actual,
            (f16::from_f32(mean).to_f32() - config.latent_mean()[channel] as f32)
                / config.latent_std()[channel] as f32
        );
    }
    drop((output, bytes));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
