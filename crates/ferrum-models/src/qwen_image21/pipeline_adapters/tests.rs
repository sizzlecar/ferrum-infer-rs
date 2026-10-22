use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
use serde_json::json;

fn account(input: usize, output: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            input + output,
            HostMemoryClassLimits {
                input_preprocess_bytes: input,
                output_bytes: output,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn id(value: &str) -> ProgramValueId {
    ProgramValueId::new(value).unwrap()
}

fn backing(host: &HostMemoryAccount, class: HostMemoryClass, bits: &[u16]) -> ManagedHostBytes {
    let mut buffer = host.allocate_zeroed(class, bits.len() * 2).unwrap();
    for (bytes, value) in buffer.as_mut().chunks_exact_mut(2).zip(bits) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    buffer.freeze()
}

fn input(
    host: &HostMemoryAccount,
    shape: QwenImage21LatentShape,
    layout: QwenImage21LatentLayout,
    domain: QwenImage21LatentDomain,
    bits: &[u16],
) -> QwenImage21ManagedLatents {
    let tensor = ManagedHostTensor::new(
        id("value.source"),
        shape.tensor(layout).unwrap(),
        backing(host, HostMemoryClass::InputPreprocess, bits),
    )
    .unwrap();
    QwenImage21ManagedLatents::new(host, tensor, shape, layout, domain).unwrap()
}

fn bits(bytes: &ManagedHostBytes) -> Vec<u16> {
    bytes
        .bytes()
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}

fn statistics(mean: f64, std: f64) -> QwenImage21LatentStatistics {
    QwenImage21LatentStatistics::from_vae_config(&json!({
        "latents_mean": vec![mean; 64], "latents_std": vec![std; 64],
    }))
    .unwrap()
}

#[test]
fn statistics_are_fixed_64_channels_and_validate_after_both_casts() {
    let statistics = statistics(1.0 + 1.0 / 256.0, 127.0 / 128.0);
    assert_eq!(statistics.mean()[0].to_bits(), 0x3f80); // BF16 ties to even one.
    assert_eq!(statistics.std()[63].to_bits(), 0x3f7e);
    for std in [0.0, -1.0, 1e-50, 3.4e38, 1e39] {
        assert!(QwenImage21LatentStatistics::from_vae_config(&json!({
            "latents_mean": vec![0.; 64], "latents_std": vec![std; 64],
        }))
        .is_err());
    }
    for count in [0, 63, 65] {
        assert!(QwenImage21LatentStatistics::from_vae_config(&json!({
            "latents_mean": vec![0.; count], "latents_std": vec![1.; 64],
        }))
        .is_err());
    }
    assert!(QwenImage21LatentStatistics::from_vae_config(&json!({})).is_err());
    let mut bad = json!({"latents_mean": vec![0.; 64], "latents_std": vec![1.; 64]});
    bad["latents_mean"][63] = json!("not numeric");
    assert!(QwenImage21LatentStatistics::from_vae_config(&bad).is_err());
}

#[test]
fn checked_shape_rejects_zero_overflow_and_equal_numel_wrong_rank() {
    for dims in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (usize::MAX, 1, 1),
        (1, usize::MAX, 2),
        (1, 1, isize::MAX as usize / 64),
    ] {
        assert!(QwenImage21LatentShape::new(dims.0, dims.1, dims.2).is_err());
    }
    let shape = QwenImage21LatentShape::new(2, 2, 3).unwrap();
    assert_eq!(shape.nchw_dimensions(), [2, 64, 2, 3]);
    assert_eq!(shape.packed_dimensions(), [12, 64]);
    let host = account(shape.byte_len(), 0);
    let data = backing(&host, HostMemoryClass::InputPreprocess, &vec![0; 768]);
    // The file's logical [B, HW, C] receipt is not a valid DiT rank-two port.
    let wrong = ResolvedTensorSpec::new(
        vec![2, 6, 64],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let tensor = ManagedHostTensor::new(id("value.source"), wrong, data).unwrap();
    assert!(QwenImage21ManagedLatents::new(
        &host,
        tensor,
        shape,
        QwenImage21LatentLayout::Packed,
        QwenImage21LatentDomain::Normalized
    )
    .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn repack_preserves_distinct_channels_positions_batches_and_source_owner() {
    let shape = QwenImage21LatentShape::new(2, 2, 3).unwrap();
    let host = account(3 * shape.byte_len(), 0);
    // Each NCHW element has a distinct finite bit pattern, not a rounded index.
    let nchw: Vec<_> = (0..768).map(|i| 0x3f00 + i as u16).collect();
    let source = input(
        &host,
        shape,
        QwenImage21LatentLayout::Nchw,
        QwenImage21LatentDomain::Normalized,
        &nchw,
    );
    let packed = source
        .repack(
            &host.clone(),
            QwenImage21LatentLayout::Packed,
            id("value.packed"),
        )
        .unwrap();
    let actual = bits(packed.bytes());
    assert_eq!(&actual[..4], &[0x3f00, 0x3f06, 0x3f0c, 0x3f12]);
    assert_eq!(actual[64], 0x3f01);
    assert_eq!(actual[6 * 64], 0x4080); // Sample one, channel zero, position zero.
    assert_eq!(actual[6 * 64 + 5 * 64 + 63], 0x41ff);
    for batch in 0..2 {
        for position in 0..6 {
            for channel in 0..64 {
                assert_eq!(
                    actual[(batch * 6 + position) * 64 + channel],
                    0x3f00 + ((batch * 64 + channel) * 6 + position) as u16
                );
            }
        }
    }
    let restored = packed
        .repack(&host, QwenImage21LatentLayout::Nchw, id("value.restored"))
        .unwrap();
    assert_eq!(bits(restored.bytes()), nchw);
    assert_eq!(restored.domain(), QwenImage21LatentDomain::Normalized);
    assert_eq!(bits(source.bytes()), nchw);
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        3 * shape.byte_len()
    );
    let retained = restored.bytes().clone();
    let restored = restored.into_input_tensor().unwrap();
    assert_eq!(restored.value_id(), &id("value.restored"));
    assert!(retained.same_region(restored.bytes()));
    drop((restored, retained, packed, source));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn admission_counts_old_and_new_backings_and_rejects_foreign_parent() {
    let shape = QwenImage21LatentShape::new(1, 1, 1).unwrap();
    let host = account(shape.byte_len() * 2 - 1, 0);
    let source = input(
        &host,
        shape,
        QwenImage21LatentLayout::Packed,
        QwenImage21LatentDomain::Normalized,
        &[0x3f80; 64],
    );
    let foreign = account(shape.byte_len() * 2, 0);
    assert!(source
        .repack(&foreign, QwenImage21LatentLayout::Nchw, id("value.next"))
        .is_err());
    assert!(source
        .denormalize_for_vae(&foreign, &statistics(0., 1.))
        .is_err());
    assert!(source
        .repack(&host, QwenImage21LatentLayout::Packed, id("value.next"))
        .is_err());
    assert!(source
        .denormalize_for_vae(&host, &statistics(0., 1.))
        .is_err());
    assert_eq!(bits(source.bytes()), vec![0x3f80; 64]);
    assert_eq!(host.usage().unwrap().allocated_bytes(), shape.byte_len());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn output_source_cannot_be_relabelled_as_input_without_new_admission() {
    let shape = QwenImage21LatentShape::new(1, 1, 1).unwrap();
    let host = account(shape.byte_len(), shape.byte_len());
    let data = backing(&host, HostMemoryClass::Output, &[0x3f80; 64]);
    let source = QwenImage21ManagedLatents::from_parts(
        &host,
        id("value.output"),
        shape.tensor(QwenImage21LatentLayout::Packed).unwrap(),
        data,
        shape,
        QwenImage21LatentLayout::Packed,
        QwenImage21LatentDomain::Normalized,
        HostMemoryClass::Output,
    )
    .unwrap();
    assert!(source.clone().into_input_tensor().is_err());
    let next = source
        .repack(&host, QwenImage21LatentLayout::Packed, id("value.next"))
        .unwrap();
    assert_eq!(next.bytes().class(), HostMemoryClass::InputPreprocess);
    assert!(!next.bytes().same_region(source.bytes()));
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        2 * shape.byte_len()
    );
    assert_eq!(source.value_id(), &id("value.output"));
    drop(source);
    assert_eq!(host.usage().unwrap().allocated_bytes(), shape.byte_len());
    drop(next);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn denormalization_has_two_bf16_boundaries_and_per_channel_statistics() {
    let shape = QwenImage21LatentShape::new(2, 2, 3).unwrap();
    let host = account(shape.byte_len() * 2, 0);
    let mut values = vec![0x3f80; 768];
    for sample_position in 0..12 {
        values[sample_position * 64] = 0x3f81;
    }
    let mut means: Vec<f64> = (0..64).map(f64::from).collect();
    let mut stds = vec![2.; 64];
    means[0] = -1.;
    stds[0] = 127. / 128.;
    let stats = QwenImage21LatentStatistics::from_vae_config(&json!({
        "latents_mean": means, "latents_std": stds,
    }))
    .unwrap();
    let source = input(
        &host,
        shape,
        QwenImage21LatentLayout::Packed,
        QwenImage21LatentDomain::Normalized,
        &values,
    );
    let output = source.denormalize_for_vae(&host, &stats).unwrap();
    let actual = bits(output.bytes());
    for batch in 0..2 {
        assert_eq!(&actual[batch * 384..batch * 384 + 6], &[0; 6]);
        assert_eq!(&actual[batch * 384 + 6..batch * 384 + 12], &[0x4040; 6]); // 1*2+1=3.
        assert_eq!(actual[batch * 384 + 383], 0x4282); // 1*2+63=65.
    }
    // Removing the inner B would produce -2^-14 instead of exact +0.
    assert_eq!(
        bf16::from_f32((129. / 128.) * (127. / 128.) - 1.).to_bits(),
        0xb880
    );
    assert_eq!(output.domain(), QwenImage21LatentDomain::Denormalized);
    assert_eq!(output.layout(), QwenImage21LatentLayout::Nchw);
    assert_eq!(
        output.value_id(),
        &id(QwenImage21VaeDecoderFamily::INPUT_ID)
    );
    assert!(output.denormalize_for_vae(&host, &stats).is_err());
    assert_eq!(bits(source.bytes()), values);
}

#[test]
fn late_overflow_drops_private_result_without_changing_source_or_budget() {
    let shape = QwenImage21LatentShape::new(1, 1, 2).unwrap();
    let host = account(shape.byte_len() * 2, 0);
    for (mean, std) in [(0., 2.), (bf16::MAX.to_f32() as f64, 1.)] {
        let mut values = vec![0; 128];
        values[127] = bf16::MAX.to_bits();
        let source = input(
            &host,
            shape,
            QwenImage21LatentLayout::Nchw,
            QwenImage21LatentDomain::Normalized,
            &values,
        );
        assert!(source
            .denormalize_for_vae(&host, &statistics(mean, std))
            .is_err());
        assert_eq!(bits(source.bytes()), values);
        assert_eq!(host.usage().unwrap().allocated_bytes(), shape.byte_len());
        assert_eq!(host.usage().unwrap().committed_bytes(), shape.byte_len());
        drop(source);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn source_validation_rejects_nonfinite_foreign_owner_and_wrong_class() {
    let shape = QwenImage21LatentShape::new(1, 1, 1).unwrap();
    let host = account(shape.byte_len(), shape.byte_len());
    let foreign = account(shape.byte_len(), 0);
    for nonfinite in [0x7f80, 0xff80, 0x7fc1] {
        let mut values = [0; 64];
        values[63] = nonfinite;
        let data = backing(&host, HostMemoryClass::InputPreprocess, &values);
        let tensor = ManagedHostTensor::new(
            id("value.source"),
            shape.tensor(QwenImage21LatentLayout::Packed).unwrap(),
            data,
        )
        .unwrap();
        assert!(QwenImage21ManagedLatents::new(
            &host,
            tensor,
            shape,
            QwenImage21LatentLayout::Packed,
            QwenImage21LatentDomain::Normalized
        )
        .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let source = input(
        &host,
        shape,
        QwenImage21LatentLayout::Packed,
        QwenImage21LatentDomain::Normalized,
        &[0; 64],
    );
    assert!(QwenImage21ManagedLatents::new(
        &foreign,
        source.clone().into_input_tensor().unwrap(),
        shape,
        QwenImage21LatentLayout::Packed,
        QwenImage21LatentDomain::Normalized
    )
    .is_err());
    assert!(QwenImage21ManagedLatents::from_parts(
        &host,
        source.value_id.clone(),
        source.tensor.clone(),
        source.bytes.clone(),
        shape,
        source.layout,
        source.domain,
        HostMemoryClass::Output
    )
    .is_err());
}

fn rgba_spec(batch: u64, height: u64, width: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![batch, 4, height, width],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

#[test]
fn rgba_stream_is_interleaved_batched_and_keeps_bf16_pixel_boundaries() {
    let host = account(0, 32);
    // Per frame: R[2], G[2], B[2], A[2]. Alpha is processed just like RGB.
    let values = [
        -1., 1., 0., 0.5, -0.5, 2., -2., 1., 1., -1., 0.5, 0., 2., -0.5, 1., -2.,
    ];
    let data = backing(
        &host,
        HostMemoryClass::Output,
        &values.map(|x| bf16::from_f32(x).to_bits()),
    );
    let output = QwenImage21RgbaOutput::new(&host, &rgba_spec(2, 1, 2), &data).unwrap();
    assert_eq!(output.dimensions(), [2, 4, 1, 2]);
    assert_eq!(output.frame_bytes(), 8);
    let mut first = Vec::new();
    output
        .write_frame_rgba8(0, &mut [0; 3], &mut first)
        .unwrap();
    assert_eq!(first, [0, 128, 64, 0, 255, 191, 255, 255]);
    let mut second = Vec::new();
    output
        .write_frame_rgba8(1, &mut [0; 1], &mut second)
        .unwrap();
    assert_eq!(second, [255, 191, 255, 255, 0, 128, 64, 0]);
    // x=5/512: early BF16 shift rounds 0.5048828125 to 0.50390625.
    assert_eq!(rgba8_finite(bf16::from_bits(0x3c20)), 128);
    assert_eq!(
        (((5_f32 / 512.) * 0.5 + 0.5) * 255.).round_ties_even() as u8,
        129
    );
    assert_eq!(rgba8_finite(bf16::from_bits(0x8000)), 128);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 32);
}

#[test]
fn rgba_bad_numeric_shape_parent_class_and_writer_bounds_fail_before_writes() {
    let host = account(8, 8);
    let foreign = account(0, 8);
    let data = backing(&host, HostMemoryClass::Output, &[0; 4]);
    let spec = rgba_spec(1, 1, 1);
    assert!(QwenImage21RgbaOutput::new(&foreign, &spec, &data).is_err());
    assert!(QwenImage21RgbaOutput::new(&host, &rgba_spec(1, 1, 2), &data).is_err());
    let wrong_rank = ResolvedTensorSpec::new(
        vec![1, 4],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(QwenImage21RgbaOutput::new(&host, &wrong_rank, &data).is_err());
    let input = backing(&host, HostMemoryClass::InputPreprocess, &[0; 4]);
    assert!(QwenImage21RgbaOutput::new(&host, &spec, &input).is_err());
    let output = QwenImage21RgbaOutput::new(&host, &spec, &data).unwrap();
    let mut writer = vec![99];
    assert!(output
        .write_frame_rgba8(1, &mut [0; 4], &mut writer)
        .is_err());
    assert!(output.write_frame_rgba8(0, &mut [], &mut writer).is_err());
    assert_eq!(writer, [99]);
    drop(data);
    let bad = backing(&host, HostMemoryClass::Output, &[0, 0, 0, 0x7fc1]);
    assert!(QwenImage21RgbaOutput::new(&host, &spec, &bad).is_err());
}

#[test]
fn writer_io_failure_is_reported_without_claiming_sink_rollback() {
    struct FailsAfterOne {
        bytes: Vec<u8>,
    }
    impl Write for FailsAfterOne {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.bytes.is_empty() {
                self.bytes.push(bytes[0]);
                Ok(1)
            } else {
                Err(std::io::Error::other("closed sink"))
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let host = account(0, 8);
    let data = backing(&host, HostMemoryClass::Output, &[0; 4]);
    let output = QwenImage21RgbaOutput::new(&host, &rgba_spec(1, 1, 1), &data).unwrap();
    let mut writer = FailsAfterOne { bytes: Vec::new() };
    assert!(output
        .write_frame_rgba8(0, &mut [0; 4], &mut writer)
        .is_err());
    assert_eq!(writer.bytes, [128]);
    assert_eq!(bits(&data), [0; 4]);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 8);
}
