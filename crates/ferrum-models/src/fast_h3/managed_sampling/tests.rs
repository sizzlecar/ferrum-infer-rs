use super::*;
use crate::fast_h3::{
    FastH3Checkpoint, FastH3MetadataLimits, FastH3Reference, FastH3VideoGeometry,
};
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryError, HostMemoryLimits, ProgramValueId};

fn account(capacity: usize) -> HostMemoryAccount {
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

fn plan() -> Arc<FastH3PddPlan> {
    let checkpoint = FastH3Checkpoint::parse(
        include_bytes!("../fixtures/inference.json"),
        include_bytes!("../fixtures/transformer.json"),
        include_bytes!("../fixtures/video_scheduler.json"),
        include_bytes!("../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 256,
        },
    )
    .unwrap();
    Arc::new(FastH3PddPlan::new(&checkpoint).unwrap())
}

fn layout() -> FastH3PackedLayout {
    FastH3PackedLayout::ref2va(
        2,
        &[
            FastH3Reference::Image {
                latent_height: 2,
                latent_width: 2,
            },
            FastH3Reference::Audio {
                latents_per_channel: 1,
            },
        ],
        &FastH3VideoGeometry::new(32, 32, 5).unwrap(),
        100,
    )
    .unwrap()
}

fn bytes(
    host: &HostMemoryAccount,
    class: HostMemoryClass,
    dtype: ElementType,
    values: &[f32],
) -> ManagedHostBytes {
    let stride = if dtype == ElementType::F32 { 4 } else { 2 };
    let mut buffer = host.allocate_zeroed(class, values.len() * stride).unwrap();
    for (destination, value) in buffer.as_mut().chunks_exact_mut(stride).zip(values) {
        if stride == 4 {
            destination.copy_from_slice(&value.to_le_bytes());
        } else {
            destination.copy_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
        }
    }
    buffer.freeze()
}

fn noise(
    host: &HostMemoryAccount,
    name: &str,
    spec: &ResolvedTensorSpec,
    frozen_rows: usize,
) -> ManagedHostTensor {
    let count = usize::try_from(spec.minimum_storage_bytes().unwrap()).unwrap() / 4;
    let frozen = frozen_rows * spec.dimensions()[1] as usize;
    let values: Vec<_> = (0..count)
        .map(|index| {
            if index < frozen {
                if index % 2 == 0 {
                    -0.0
                } else {
                    3.25
                }
            } else {
                [1.0002441, -4.0004883, 0.125, 10.75][index % 4]
            }
        })
        .collect();
    ManagedHostTensor::new(
        ProgramValueId::new(name).unwrap(),
        spec.clone(),
        bytes(
            host,
            HostMemoryClass::InputPreprocess,
            ElementType::F32,
            &values,
        ),
    )
    .unwrap()
}

fn continuation(host: &HostMemoryAccount, name: &str) -> FastH3ManagedContinuation {
    let plan = plan();
    let layout = layout();
    let requirements = FastH3SamplingHostRequirements::new(&plan, &layout).unwrap();
    let video = noise(
        host,
        "latent.video",
        requirements.video_tensor(),
        layout.target_video_rows().start,
    );
    let audio = noise(
        host,
        "latent.audio",
        requirements.audio_tensor(),
        layout.target_audio_rows().start,
    );
    FastH3ManagedContinuation::initialize(
        RequestIdentity::new(name).unwrap(),
        plan,
        &layout,
        host.clone(),
        &video,
        &audio,
    )
    .unwrap()
}

struct Prediction {
    request: RequestIdentity,
    step: usize,
    video: ResolvedTensorSpec,
    audio: ResolvedTensorSpec,
    video_bytes: ManagedHostBytes,
    audio_bytes: ManagedHostBytes,
}

impl Prediction {
    fn view(&self) -> FastH3PredictionView<'_> {
        FastH3PredictionView::new(
            &self.request,
            self.step,
            FastH3VelocityInput::new(&self.video, &self.video_bytes),
            FastH3VelocityInput::new(&self.audio, &self.audio_bytes),
        )
    }
}

fn prediction(
    host: &HostMemoryAccount,
    continuation: &FastH3ManagedContinuation,
    dtype: ElementType,
    bad_audio: bool,
) -> Prediction {
    let build = |spec: &StreamSpec, value: f32| {
        let tensor = ResolvedTensorSpec::new(
            spec.tensor.dimensions().to_vec(),
            dtype,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let mut values = vec![value; spec.bytes / 4];
        // The unused model predictions for frozen references must never be
        // integrated or interpreted as target values.
        values[..spec.target_bytes.start / 4].fill(f32::NAN);
        (tensor, bytes(host, HostMemoryClass::Output, dtype, &values))
    };
    let (video, video_bytes) = build(&continuation.requirements.video, 0.1234567);
    let (audio, audio_bytes) = build(
        &continuation.requirements.audio,
        if bad_audio { f32::NAN } else { -0.3456789 },
    );
    Prediction {
        request: continuation.request.clone(),
        step: continuation.step_index,
        video,
        audio,
        video_bytes,
        audio_bytes,
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

// A separate wider-arithmetic oracle rounds each specified operation to f32.
// Fixtures here avoid overflow/subnormal double-rounding corner cases; device
// FTZ and GPU kernel parity remain separate qualification work.
fn reference_update(sample: f32, velocity: f32, step: &FastH3ModalityStep) -> f32 {
    let r = |value: f64| f64::from(value as f32);
    let sigma = f64::from(step.sigma());
    let time = r(1.0 - sigma);
    let contribution = r(r(1.0 - time) * f64::from(velocity));
    let clean = r(f64::from(sample) + contribution);
    let ratio = r(f64::from(step.next_sigma()) / sigma);
    r(r(ratio * f64::from(sample)) + r(r(1.0 - ratio) * clean)) as f32
}

#[test]
fn complete_joint_trajectory_matches_rounded_reference_and_preserves_frozen_prefixes() {
    for dtype in [ElementType::F32, ElementType::Bf16] {
        let host = account(100_000);
        let mut state = continuation(&host, "request.a");
        let mut expected_video = values(state.video());
        let mut expected_audio = values(state.audio());
        let video_frozen = state.requirements.video.target_bytes.start / 4;
        let audio_frozen = state.requirements.audio.target_bytes.start / 4;
        let frozen_video = state.video.bytes().bytes()[..video_frozen * 4].to_vec();
        let frozen_audio = state.audio.bytes().bytes()[..audio_frozen * 4].to_vec();
        let initial = [1.0002441_f32, -4.0004883, 0.125, 10.75][video_frozen % 4];
        assert_eq!(
            expected_video[video_frozen].to_bits(),
            (initial * 0.999_f32).to_bits()
        );
        for index in 0..state.plan.steps().len() {
            let output = prediction(&host, &state, dtype, false);
            let rounded = |v| {
                if dtype == ElementType::Bf16 {
                    bf16::from_f32(v).to_f32()
                } else {
                    v
                }
            };
            let step = state.current_step().unwrap();
            for value in &mut expected_video[video_frozen..] {
                *value = reference_update(*value, rounded(0.1234567), &step.video);
            }
            for value in &mut expected_audio[audio_frozen..] {
                *value = reference_update(*value, rounded(-0.3456789), &step.audio);
            }
            let prepared = state.prepare_advance(output.view()).unwrap();
            assert_eq!(prepared.step_index(), index);
            assert_eq!(prepared.step().heads, index * 4..(index + 1) * 4);
            prepared.commit();
            assert_eq!(
                values(state.video())
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected_video
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                values(state.audio())
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected_audio
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                &state.video.bytes().bytes()[..video_frozen * 4],
                &frozen_video
            );
            assert_eq!(
                &state.audio.bytes().bytes()[..audio_frozen * 4],
                &frozen_audio
            );
        }
        assert!(state.is_complete());
        let output = prediction(&host, &state, dtype, false);
        assert!(state.prepare_advance(output.view()).is_err());
        drop((output, state));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn allocation_failure_late_numeric_failure_and_drop_preserve_both_streams() {
    let requirement = FastH3SamplingHostRequirements::new(&plan(), &layout()).unwrap();
    let bytes_per_pair = requirement.successor_bytes();
    for failure in ["capacity", "numeric", "drop"] {
        let limit = bytes_per_pair * 3 - usize::from(failure == "capacity");
        let host = account(limit);
        let mut state = continuation(&host, "request.a");
        let output = prediction(&host, &state, ElementType::F32, failure == "numeric");
        let original_video = state.video.bytes().bytes().to_vec();
        let original_audio = state.audio.bytes().bytes().to_vec();
        let before = host.usage().unwrap().committed_bytes();
        match state.prepare_advance(output.view()) {
            Ok(prepared) => {
                assert_eq!(failure, "drop");
                drop(prepared);
            }
            Err(error) => {
                assert_ne!(failure, "drop");
                if failure == "capacity" {
                    assert!(error.downcast_ref::<HostMemoryError>().is_some());
                } else {
                    assert!(error.to_string().contains("nonfinite"));
                }
            }
        }
        assert_eq!(state.step_index(), 0);
        assert_eq!(state.video.bytes().bytes(), original_video);
        assert_eq!(state.audio.bytes().bytes(), original_audio);
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
    }
}

#[test]
fn stale_or_foreign_prediction_cannot_advance_a_request() {
    let host = account(100_000);
    let mut state = continuation(&host, "request.a");
    let mut output = prediction(&host, &state, ElementType::F32, false);
    output.request = RequestIdentity::new("request.b").unwrap();
    assert!(state.prepare_advance(output.view()).is_err());
    output.request = state.request.clone();
    state.prepare_advance(output.view()).unwrap().commit();
    let before = host.usage().unwrap().committed_bytes();
    assert!(state.prepare_advance(output.view()).is_err());
    assert_eq!(state.step_index(), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    let foreign = account(100_000);
    let output = prediction(&foreign, &state, ElementType::F32, false);
    assert!(state.prepare_advance(output.view()).is_err());
    assert_eq!(state.step_index(), 1);
}

#[test]
fn peer_prepare_failure_rolls_back_independent_mixed_progress_requests() {
    let host = account(100_000);
    let mut first = continuation(&host, "request.a");
    let mut second = continuation(&host, "request.b");
    let first_output = prediction(&host, &first, ElementType::F32, false);
    first.prepare_advance(first_output.view()).unwrap().commit();
    drop(first_output);
    let first_output = prediction(&host, &first, ElementType::F32, false);
    let second_output = prediction(&host, &second, ElementType::F32, true);
    let before = host.usage().unwrap().committed_bytes();
    let first_prepared = first.prepare_advance(first_output.view()).unwrap();
    assert!(second.prepare_advance(second_output.view()).is_err());
    drop(first_prepared);
    assert_eq!((first.step_index(), second.step_index()), (1, 0));
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    drop(second_output);
    let second_output = prediction(&host, &second, ElementType::F32, false);
    let first_prepared = first.prepare_advance(first_output.view()).unwrap();
    let second_prepared = second.prepare_advance(second_output.view()).unwrap();
    assert_eq!(first_prepared.step().heads, 4..8);
    assert_eq!(second_prepared.step().heads, 0..4);
    first_prepared.commit();
    second_prepared.commit();
    assert_eq!((first.step_index(), second.step_index()), (2, 1));
}
