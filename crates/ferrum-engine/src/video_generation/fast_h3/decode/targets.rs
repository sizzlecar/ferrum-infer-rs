use anyhow::{ensure, Context, Result};
use ferrum_models::fast_h3::{FastH3DenoisingFamily, FastH3DenoisingRequest, FastH3PackedLayout};
use ferrum_runtime::*;
use std::ops::Range;

pub(crate) const VIDEO_TARGET_ID: &str = "value.fast_h3.decode.target.video";
pub(crate) const AUDIO_TARGET_ID: &str = "value.fast_h3.decode.target.audio";

/// Only a completed request may cross the solver/decoder boundary. This makes
/// new InputPreprocess copies; neither frozen references nor Output aliases
/// are supplied to the decoder.
pub(crate) fn managed_decoder_targets(
    request: &FastH3DenoisingRequest,
    layout: &FastH3PackedLayout,
    host: &HostMemoryAccount,
    video: &ResolvedTensorSpec,
    audio: &ResolvedTensorSpec,
) -> Result<[ManagedHostTensor; 2]> {
    ensure!(
        request.is_complete(),
        "decoder requires a completed PDD continuation"
    );
    copy_targets(
        [request.video(), request.audio()],
        [layout.target_video_rows(), layout.target_audio_rows()],
        [video, audio],
        host,
    )
}
fn copy_targets(
    sources: [&ManagedHostTensor; 2],
    rows: [Range<usize>; 2],
    specs: [&ResolvedTensorSpec; 2],
    host: &HostMemoryAccount,
) -> Result<[ManagedHostTensor; 2]> {
    let mut ranges = [0..0, 0..0];
    for i in 0..2 {
        let source = sources[i];
        let expected_id = [
            FastH3DenoisingFamily::VIDEO_INPUT_ID,
            FastH3DenoisingFamily::AUDIO_INPUT_ID,
        ][i];
        let target = specs[i];
        ensure!(
            target.dimensions().len() == 2
                && target.element_type() == ElementType::F32
                && target.layout() == &ResolvedTensorLayout::Contiguous
                && rows[i].start < rows[i].end
                && target.dimensions()[0] == (rows[i].end - rows[i].start) as u64,
            "decoder target rows or numerical contract differ"
        );
        ensure!(
            source.value_id().as_str() == expected_id
                && source.tensor().dimensions() == [rows[i].end as u64, target.dimensions()[1]]
                && source.tensor().element_type() == ElementType::F32
                && source.tensor().layout() == &ResolvedTensorLayout::Contiguous
                && source.bytes().class() == HostMemoryClass::InputPreprocess
                && host.owns(source.bytes())
                && source.bytes().len() as u64 == source.tensor().minimum_storage_bytes()?,
            "completed modality identity, full shape or Host owner differs"
        );
        let stride = usize::try_from(target.dimensions()[1])?
            .checked_mul(4)
            .context("target stride overflow")?;
        ranges[i] = rows[i]
            .start
            .checked_mul(stride)
            .context("target offset overflow")?
            ..rows[i]
                .end
                .checked_mul(stride)
                .context("target extent overflow")?;
    }
    // One admission covers both copies, so a short request scope publishes
    // neither stream and cannot consume another request's reservation.
    let grants = host.reserve_many(&[
        (HostMemoryClass::InputPreprocess, ranges[0].len()),
        (HostMemoryClass::InputPreprocess, ranges[1].len()),
    ])?;
    let mut tensors = Vec::with_capacity(2);
    for (i, grant) in grants.into_iter().enumerate() {
        let mut bytes = grant.allocate_zeroed()?;
        bytes
            .as_mut()
            .copy_from_slice(&sources[i].bytes().bytes()[ranges[i].clone()]);
        tensors.push(ManagedHostTensor::new(
            ProgramValueId::new([VIDEO_TARGET_ID, AUDIO_TARGET_ID][i])?,
            specs[i].clone(),
            bytes.freeze(),
        )?);
    }
    Ok(tensors
        .try_into()
        .map_err(|_| anyhow::anyhow!("target pair missing"))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn host(capacity: usize) -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                capacity,
                HostMemoryClassLimits {
                    input_preprocess_bytes: capacity,
                    output_bytes: capacity,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    }
    fn input(host: &HostMemoryAccount, name: &str, values: &[f32]) -> ManagedHostTensor {
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
            .unwrap();
        for (out, n) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
            out.copy_from_slice(&n.to_le_bytes());
        }
        ManagedHostTensor::new(
            ProgramValueId::new(name).unwrap(),
            ResolvedTensorSpec::new(
                vec![values.len() as u64, 1],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            bytes.freeze(),
        )
        .unwrap()
    }
    #[test]
    fn decoder_target_copy_excludes_reference_and_preserves_original_owner() {
        let h = host(64);
        let v = input(&h, FastH3DenoisingFamily::VIDEO_INPUT_ID, &[99., 1., 2.]);
        let a = input(
            &h,
            FastH3DenoisingFamily::AUDIO_INPUT_ID,
            &[88., 77., 3., 4.],
        );
        let spec = ResolvedTensorSpec::new(
            vec![2, 1],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let result = copy_targets([&v, &a], [1..3, 2..4], [&spec, &spec], &h).unwrap();
        for (tensor, expected) in result.iter().zip([[1.0f32, 2.], [3., 4.]]) {
            assert_eq!(
                tensor.bytes().bytes(),
                expected
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>()
            );
            assert_eq!(tensor.bytes().class(), HostMemoryClass::InputPreprocess);
            assert!(h.owns(tensor.bytes()));
        }
        assert_eq!(
            f32::from_le_bytes(v.bytes().bytes()[..4].try_into().unwrap()),
            99.
        );
        let before = h.usage().unwrap().committed_bytes();
        assert!(copy_targets([&a, &v], [1..3, 2..4], [&spec, &spec], &h).is_err());
        assert_eq!(h.usage().unwrap().committed_bytes(), before);
        assert!(copy_targets([&v, &a], [1..3, 2..4], [&spec, &spec], &host(64)).is_err());
        drop((result, v, a));
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
    #[test]
    fn decoder_target_pair_admission_rolls_back_when_joint_capacity_is_short() {
        let h = host(43); // source 28B plus target pair 16B cannot fit.
        let v = input(&h, FastH3DenoisingFamily::VIDEO_INPUT_ID, &[99., 1., 2.]);
        let a = input(
            &h,
            FastH3DenoisingFamily::AUDIO_INPUT_ID,
            &[88., 77., 3., 4.],
        );
        let spec = ResolvedTensorSpec::new(
            vec![2, 1],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        assert!(copy_targets([&v, &a], [1..3, 2..4], [&spec, &spec], &h).is_err());
        assert_eq!(h.usage().unwrap().committed_bytes(), 28);
    }
}
