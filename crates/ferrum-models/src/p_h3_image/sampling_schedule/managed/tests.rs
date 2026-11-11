use super::*;
use crate::p_h3_image::PH3ImageSamplingLimits;
use ferrum_runtime::{
    HostMemoryClassLimits, HostMemoryError, HostMemoryLimits, HostMemoryPressureScope,
    ProgramValueId,
};
use serde_json::json;

fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn schedule(sigmas: &[f32]) -> PH3ImageSamplingSchedule {
    let index = json!({"_class_name":"PH3ImageModularPipeline", "_blocks_class_name":"PH3ImageBlocks",
        "scheduler":["diffusers","MiniMaxH3Scheduler",{}],
        "audio_scheduler":["diffusers","MiniMaxH3Scheduler",{}], "sample_sigmas":sigmas});
    let config =
        json!({"_class_name":"MiniMaxH3Scheduler", "shift":12.0, "shift_provided_sigmas":false});
    PH3ImageSamplingSchedule::parse(
        &serde_json::to_vec(&index).unwrap(),
        &serde_json::to_vec(&config).unwrap(),
        &serde_json::to_vec(&config).unwrap(),
        PH3ImageSamplingLimits {
            max_metadata_bytes: 4096,
            max_evaluations: 16,
        },
    )
    .unwrap()
}
fn spec(elements: usize, precision: PH3ImageVelocityPrecision) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![1, elements as u64],
        precision.element_type(),
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
struct VelocityFixture {
    spec: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
}
impl VelocityFixture {
    fn view(&self) -> PH3ImageSamplingVelocity<'_> {
        PH3ImageSamplingVelocity::new(&self.spec, &self.bytes)
    }
}
fn tensor_bytes(
    host: &HostMemoryAccount,
    class: HostMemoryClass,
    precision: PH3ImageVelocityPrecision,
    values: &[f32],
) -> ManagedHostBytes {
    let mut bytes = host
        .allocate_zeroed(class, values.len() * precision.bytes())
        .unwrap();
    for (out, &value) in bytes
        .as_mut()
        .chunks_exact_mut(precision.bytes())
        .zip(values)
    {
        match precision {
            PH3ImageVelocityPrecision::F32 => out.copy_from_slice(&value.to_le_bytes()),
            PH3ImageVelocityPrecision::Bf16 => {
                out.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
            }
        }
    }
    bytes.freeze()
}
fn sample(host: &HostMemoryAccount, name: &str, values: &[f32]) -> ManagedHostTensor {
    let bytes = tensor_bytes(
        host,
        HostMemoryClass::InputPreprocess,
        PH3ImageVelocityPrecision::F32,
        values,
    );
    ManagedHostTensor::new(
        ProgramValueId::new(name).unwrap(),
        spec(values.len(), PH3ImageVelocityPrecision::F32),
        bytes,
    )
    .unwrap()
}
fn velocity(
    host: &HostMemoryAccount,
    values: &[f32],
    precision: PH3ImageVelocityPrecision,
) -> VelocityFixture {
    let bytes = tensor_bytes(host, HostMemoryClass::Output, precision, values);
    VelocityFixture {
        spec: spec(values.len(), precision),
        bytes,
    }
}
fn values(tensor: &ManagedHostTensor) -> Vec<f32> {
    tensor
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(decode_f32)
        .collect()
}
fn sampling(
    host: &HostMemoryAccount,
    schedule: &PH3ImageSamplingSchedule,
    image: usize,
    audio: usize,
    precision: PH3ImageVelocityPrecision,
) -> PH3ImageManagedSampling {
    PH3ImageManagedSampling::new(
        schedule,
        host.clone(),
        spec(image, PH3ImageVelocityPrecision::F32),
        spec(audio, PH3ImageVelocityPrecision::F32),
        precision,
    )
    .unwrap()
}

#[test]
fn both_velocity_precisions_follow_independent_f32_trajectory_without_sample_truncation() {
    for precision in [
        PH3ImageVelocityPrecision::F32,
        PH3ImageVelocityPrecision::Bf16,
    ] {
        let host = account(4096);
        let sigmas = [
            1.0_f32, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0,
        ];
        let schedule = schedule(&sigmas);
        let mut solver = sampling(&host, &schedule, 3, 2, precision);
        let mut image = sample(&host, "value.image", &[1.0002441, -4.0, 0.125]);
        let mut audio = sample(&host, "value.audio", &[2.0004883, -3.0]);
        let mut expected_image = values(&image);
        let mut expected_audio = values(&audio);
        for (index, interval) in sigmas.windows(2).enumerate() {
            let vi = velocity(&host, &[0.125, -0.5, 0.25], precision);
            let va = velocity(&host, &[0.5, -0.25], precision);
            let pending = solver
                .prepare_advance(&image, vi.view(), &audio, va.view())
                .unwrap();
            assert_eq!(pending.step().index(), index);
            let (step, next_image, next_audio) = pending.commit();
            assert_eq!(next_image.value_id(), image.value_id());
            assert_eq!(next_audio.value_id(), audio.value_id());
            assert_eq!(step.timestep(), 1.0_f32 - interval[0]);
            for (sample, predicted) in [
                (&mut expected_image[..], &[0.125, -0.5, 0.25][..]),
                (&mut expected_audio[..], &[0.5, -0.25][..]),
            ] {
                for (x, &v) in sample.iter_mut().zip(predicted) {
                    let time = 1.0_f32 - interval[0];
                    let x0 = *x + (1.0_f32 - time) * v;
                    let ratio = interval[1] / interval[0];
                    *x = ratio * *x + (1.0_f32 - ratio) * x0;
                }
            }
            assert_eq!(values(&next_image), expected_image);
            assert_eq!(values(&next_audio), expected_audio);
            image = next_image;
            audio = next_audio;
        }
        assert!(solver.is_complete());
        assert_ne!(
            values(&image)[0].to_bits() & 0xffff,
            0,
            "F32 sample must not be truncated to BF16"
        );
        drop((solver, image, audio));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn second_stream_capacity_failure_and_late_numeric_failure_return_every_private_claim() {
    for capacity in [23, 24] {
        let host = account(capacity);
        let mut solver = sampling(
            &host,
            &schedule(&[1.0, 0.0]),
            1,
            1,
            PH3ImageVelocityPrecision::F32,
        );
        let image = sample(&host, "value.image", &[1.0]);
        let audio = sample(&host, "value.audio", &[2.0]);
        let vi = velocity(&host, &[3.0], PH3ImageVelocityPrecision::F32);
        let va = velocity(
            &host,
            &[if capacity == 23 { 4.0 } else { f32::NAN }],
            PH3ImageVelocityPrecision::F32,
        );
        let before = host.usage().unwrap().committed_bytes();
        let error = match solver.prepare_advance(&image, vi.view(), &audio, va.view()) {
            Err(error) => error,
            Ok(_) => panic!("expected capacity or numerical rejection"),
        };
        if capacity == 23 {
            assert!(matches!(
                error.downcast_ref::<HostMemoryError>(),
                Some(HostMemoryError::Capacity {
                    scope: HostMemoryPressureScope::Total,
                    requested: 4,
                    committed: 20,
                    limit: 23
                })
            ));
        } else {
            assert!(error.to_string().contains("nonfinite"));
        }
        assert_eq!(solver.step_index(), 0);
        assert_eq!(values(&image), [1.0]);
        assert_eq!(values(&audio), [2.0]);
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        drop((solver, image, audio, vi, va));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn dropping_prepared_peer_preserves_mixed_progress_and_allows_retry() {
    let host = account(256);
    let schedule = schedule(&[1.0, 0.5, 0.0]);
    let mut a = sampling(&host, &schedule, 1, 1, PH3ImageVelocityPrecision::F32);
    let mut b = sampling(&host, &schedule, 1, 1, PH3ImageVelocityPrecision::F32);
    let image = sample(&host, "value.image", &[1.0]);
    let audio = sample(&host, "value.audio", &[2.0]);
    let vi = velocity(&host, &[3.0], PH3ImageVelocityPrecision::F32);
    let va = velocity(&host, &[4.0], PH3ImageVelocityPrecision::F32);
    let (_, ai, aa) = a
        .prepare_advance(&image, vi.view(), &audio, va.view())
        .unwrap()
        .commit();
    let before = host.usage().unwrap().committed_bytes();
    let first = a.prepare_advance(&ai, vi.view(), &aa, va.view()).unwrap();
    let second = b
        .prepare_advance(&image, vi.view(), &audio, va.view())
        .unwrap();
    assert_eq!(first.step().index(), 1);
    assert_eq!(second.step().index(), 0);
    drop((first, second));
    assert_eq!((a.step_index(), b.step_index()), (1, 0));
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    let first = a.prepare_advance(&ai, vi.view(), &aa, va.view()).unwrap();
    let second = b
        .prepare_advance(&image, vi.view(), &audio, va.view())
        .unwrap();
    let (_, final_image, final_audio) = first.commit();
    let (_, bi, ba) = second.commit();
    assert!(a.is_complete());
    assert_eq!(b.step_index(), 1);
    assert_eq!(values(&bi), values(&ai));
    assert_eq!(values(&ba), values(&aa));
    drop((
        a,
        b,
        image,
        audio,
        vi,
        va,
        ai,
        aa,
        bi,
        ba,
        final_image,
        final_audio,
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn foreign_owners_wrong_class_and_velocity_dtype_are_rejected_without_progress() {
    let host = account(256);
    let foreign = account(256);
    let mut solver = sampling(
        &host,
        &schedule(&[1.0, 0.0]),
        1,
        1,
        PH3ImageVelocityPrecision::F32,
    );
    let image = sample(&host, "value.image", &[1.0]);
    let audio = sample(&host, "value.audio", &[2.0]);
    let vi = velocity(&host, &[3.0], PH3ImageVelocityPrecision::F32);
    for bad in [
        velocity(&foreign, &[4.0], PH3ImageVelocityPrecision::F32),
        VelocityFixture {
            spec: spec(1, PH3ImageVelocityPrecision::F32),
            bytes: tensor_bytes(
                &host,
                HostMemoryClass::InputPreprocess,
                PH3ImageVelocityPrecision::F32,
                &[4.0],
            ),
        },
        velocity(&host, &[4.0], PH3ImageVelocityPrecision::Bf16),
    ] {
        let before = host.usage().unwrap().committed_bytes();
        assert!(solver
            .prepare_advance(&image, vi.view(), &audio, bad.view())
            .is_err());
        assert_eq!(solver.step_index(), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
    }
    drop((solver, image, audio, vi));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}
