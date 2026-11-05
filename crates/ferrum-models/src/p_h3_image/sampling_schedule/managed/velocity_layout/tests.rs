use super::*;
use crate::p_h3_image::PH3ImageSamplingLimits;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits, ProgramValueId};
mod managed_component;
fn account(limit: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            limit,
            HostMemoryClassLimits {
                input_preprocess_bytes: limit,
                output_bytes: limit,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn schedule() -> PH3ImageSamplingSchedule {
    let model = serde_json::json!({"_class_name":"PH3ImageModularPipeline","_blocks_class_name":"PH3ImageBlocks","scheduler":["diffusers","MiniMaxH3Scheduler",{}],"audio_scheduler":["diffusers","MiniMaxH3Scheduler",{}],"sample_sigmas":[1.,0.75,0.5,0.]});
    let config = serde_json::json!({"_class_name":"MiniMaxH3Scheduler","shift":1.,"shift_provided_sigmas":false});
    PH3ImageSamplingSchedule::parse(
        &serde_json::to_vec(&model).unwrap(),
        &serde_json::to_vec(&config).unwrap(),
        &serde_json::to_vec(&config).unwrap(),
        PH3ImageSamplingLimits {
            max_metadata_bytes: 4096,
            max_evaluations: 8,
        },
    )
    .unwrap()
}
fn id(s: &str) -> RequestIdentity {
    RequestIdentity::new(s).unwrap()
}
fn tensor(
    host: &HostMemoryAccount,
    name: &str,
    shape: Vec<u64>,
    values: &[f32],
) -> ManagedHostTensor {
    let mut a = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (to, v) in a.as_mut().chunks_exact_mut(4).zip(values) {
        to.copy_from_slice(&v.to_le_bytes());
    }
    ManagedHostTensor::new(
        ProgramValueId::new(name).unwrap(),
        ResolvedTensorSpec::new(shape, ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap(),
        a.freeze(),
    )
    .unwrap()
}
struct State {
    request: PH3ImageVelocityRequest,
    solver: PH3ImageManagedSampling,
    image: ManagedHostTensor,
    audio: ManagedHostTensor,
}
fn state(
    host: &HostMemoryAccount,
    name: &str,
    h: usize,
    w: usize,
    precision: PH3ImageVelocityPrecision,
    five: bool,
) -> State {
    let request =
        PH3ImageVelocityRequest::new(id(name), PH3ImageReferenceImage::new(h, w).unwrap());
    let image = tensor(
        host,
        "value.image",
        if five {
            vec![1, 2, 1, h as u64, w as u64]
        } else {
            vec![1, 2, h as u64, w as u64]
        },
        &(0..2 * h * w)
            .map(|i| 1.0002441 + i as f32 / 64.)
            .collect::<Vec<_>>(),
    );
    let audio = tensor(
        host,
        "value.audio",
        vec![1, 2, 2, 2],
        &(0..8)
            .map(|i| 2.0004883 + i as f32 / 64.)
            .collect::<Vec<_>>(),
    );
    let solver = PH3ImageManagedSampling::new_for_request(
        &schedule(),
        host.clone(),
        image.tensor().clone(),
        audio.tensor().clone(),
        precision,
        request.request.clone(),
    )
    .unwrap();
    State {
        request,
        solver,
        image,
        audio,
    }
}
struct Output {
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
}
fn output(
    host: &HostMemoryAccount,
    shape: Vec<u64>,
    values: &[f32],
    precision: PH3ImageVelocityPrecision,
) -> Output {
    let mut a = host
        .allocate_zeroed(HostMemoryClass::Output, values.len() * precision.bytes())
        .unwrap();
    for (to, &v) in a.as_mut().chunks_exact_mut(precision.bytes()).zip(values) {
        match precision {
            PH3ImageVelocityPrecision::F32 => to.copy_from_slice(&v.to_le_bytes()),
            PH3ImageVelocityPrecision::Bf16 => {
                to.copy_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes())
            }
        }
    }
    Output {
        tensor: ResolvedTensorSpec::new(
            shape,
            precision.element_type(),
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes: a.freeze(),
    }
}
fn code(r: &PH3ImageVelocityRequest) -> usize {
    if r.request.as_str().ends_with('a') {
        1
    } else {
        5
    }
}
fn physical(
    host: &HostMemoryAccount,
    requests: &[PH3ImageVelocityRequest],
    precision: PH3ImageVelocityPrecision,
    bad_audio: bool,
) -> (Output, Output) {
    let mut image = Vec::new();
    let mut audio = Vec::new();
    for request in requests {
        let h = request.target.latent_height();
        let w = request.target.latent_width();
        for py in 0..h / 2 {
            for px in 0..w / 2 {
                for channel in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            image.push(
                                (code(request) * 17
                                    + channel * 11
                                    + (py * 2 + dy) * 3
                                    + px * 2
                                    + dx) as f32
                                    / 64.,
                            );
                        }
                    }
                }
            }
        }
        for row in 0..4 {
            for channel in 0..2 {
                audio.push((code(request) * 13 + channel * 7 + row) as f32 / 64.);
            }
        }
    }
    if bad_audio {
        *audio.last_mut().unwrap() = f32::NAN;
    }
    (
        output(host, vec![(image.len() / 8) as u64, 8], &image, precision),
        output(host, vec![(audio.len() / 2) as u64, 2], &audio, precision),
    )
}
fn values(t: &ManagedHostTensor) -> Vec<f32> {
    t.bytes().bytes().chunks_exact(4).map(decode_f32).collect()
}
fn advance(
    states: &mut [State],
    layout: &PH3ImageVelocityBatchLayout<'_>,
    image: &Output,
    audio: &Output,
) -> Result<()> {
    let pending = states
        .iter_mut()
        .map(|state| {
            state.solver.prepare_advance(
                &state.image,
                layout.image_storage(&state.request.request, &image.tensor, &image.bytes)?,
                &state.audio,
                layout.audio_storage(&state.request.request, &audio.tensor, &audio.bytes)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let successors = pending
        .into_iter()
        .map(PreparedPH3ImageSamplingAdvance::commit)
        .collect::<Vec<_>>();
    for (state, (_, image, audio)) in states.iter_mut().zip(successors) {
        state.image = image;
        state.audio = audio;
    }
    Ok(())
}
fn oracle(s: &State, precision: PH3ImageVelocityPrecision) -> (Vec<f32>, Vec<f32>) {
    let step = s.solver.current_step().unwrap();
    let h = s.request.target.latent_height();
    let w = s.request.target.latent_width();
    let round = |v| match precision {
        PH3ImageVelocityPrecision::F32 => v,
        PH3ImageVelocityPrecision::Bf16 => bf16::from_f32(v).to_f32(),
    };
    let mut image = values(&s.image);
    for channel in 0..2 {
        for y in 0..h {
            for x in 0..w {
                let index = (channel * h + y) * w + x;
                let velocity =
                    round((code(&s.request) * 17 + channel * 11 + y * 3 + x) as f32 / 64.);
                let old = image[index];
                let denoised = old + (1. - step.timestep()) * velocity;
                let ratio = step.next_sigma() / step.sigma();
                image[index] = ratio * old + (1. - ratio) * denoised;
            }
        }
    }
    let mut audio = values(&s.audio);
    for channel in 0..2 {
        for row in 0..4 {
            let index = channel * 4 + row;
            let velocity = round((code(&s.request) * 13 + channel * 7 + row) as f32 / 64.);
            let old = audio[index];
            let denoised = old + (1. - step.timestep()) * velocity;
            let ratio = step.next_sigma() / step.sigma();
            audio[index] = ratio * old + (1. - ratio) * denoised;
        }
    }
    (image, audio)
}
#[test]
fn image_patches_and_audio_rows_follow_mixed_b2_b1_progress_and_request_reordering() {
    for precision in [
        PH3ImageVelocityPrecision::F32,
        PH3ImageVelocityPrecision::Bf16,
    ] {
        let host = account(1 << 20);
        let t = crate::p_h3_image::test_fixture::tiny_transformer(16, 2);
        let mut states = [
            state(&host, "request.a", 2, 4, precision, true),
            state(&host, "request.b", 4, 2, precision, false),
        ];
        let a = [states[0].request.clone()];
        let layout = PH3ImageVelocityBatchLayout::new(&t, &a, 4096).unwrap();
        let (vi, va) = physical(&host, &a, precision, false);
        let expected = oracle(&states[0], precision);
        advance(&mut states[..1], &layout, &vi, &va).unwrap();
        assert_eq!(values(&states[0].image), expected.0);
        assert_eq!(values(&states[0].audio), expected.1);
        assert_eq!(states[0].solver.step_index(), 1);
        assert_eq!(states[1].solver.step_index(), 0);
        drop((vi, va));
        for reverse in [false, true] {
            let requests = if reverse {
                vec![states[1].request.clone(), states[0].request.clone()]
            } else {
                vec![states[0].request.clone(), states[1].request.clone()]
            };
            let layout = PH3ImageVelocityBatchLayout::new(&t, &requests, 4096).unwrap();
            let (vi, va) = physical(&host, &requests, precision, false);
            let expected = states
                .iter()
                .map(|s| oracle(s, precision))
                .collect::<Vec<_>>();
            let before = host.usage().unwrap().committed_bytes();
            let view = layout
                .image_storage(&states[0].request.request, &vi.tensor, &vi.bytes)
                .unwrap();
            assert!(std::ptr::eq(view.backing(), &vi.bytes));
            assert_eq!(view.physical_tensor().dimensions(), [4, 8]);
            assert_eq!(host.usage().unwrap().committed_bytes(), before);
            advance(&mut states, &layout, &vi, &va).unwrap();
            for (s, e) in states.iter().zip(expected) {
                assert_eq!(values(&s.image), e.0);
                assert_eq!(values(&s.audio), e.1);
            }
        }
        assert!(states[0].solver.is_complete());
        assert!(!states[1].solver.is_complete());
        let b = [states[1].request.clone()];
        let layout = PH3ImageVelocityBatchLayout::new(&t, &b, 4096).unwrap();
        let (vi, va) = physical(&host, &b, precision, false);
        let expected = oracle(&states[1], precision);
        advance(&mut states[1..], &layout, &vi, &va).unwrap();
        assert_eq!(values(&states[1].image), expected.0);
        assert_eq!(values(&states[1].audio), expected.1);
        drop((vi, va, states));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
#[test]
fn late_peer_failure_keeps_every_stream_and_cursor_and_refunds_all_successors() {
    let host = account(1 << 20);
    let precision = PH3ImageVelocityPrecision::F32;
    let t = crate::p_h3_image::test_fixture::tiny_transformer(16, 2);
    let mut states = [
        state(&host, "request.a", 2, 4, precision, true),
        state(&host, "request.b", 4, 2, precision, false),
    ];
    let requests = states.iter().map(|s| s.request.clone()).collect::<Vec<_>>();
    let layout = PH3ImageVelocityBatchLayout::new(&t, &requests, 4096).unwrap();
    let (vi, va) = physical(&host, &requests, precision, true);
    let before = host.usage().unwrap().committed_bytes();
    let old = states
        .iter()
        .map(|s| (values(&s.image), values(&s.audio)))
        .collect::<Vec<_>>();
    assert!(advance(&mut states, &layout, &vi, &va).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    for (s, old) in states.iter().zip(old) {
        assert_eq!(s.solver.step_index(), 0);
        assert_eq!(values(&s.image), old.0);
        assert_eq!(values(&s.audio), old.1);
    }
    drop(va);
    let (unused, va) = physical(&host, &requests, precision, false);
    drop(unused);
    let used = host.usage().unwrap().committed_bytes();
    let free = states[0].image.bytes().len()
        + states[0].audio.bytes().len()
        + states[1].image.bytes().len();
    let padding = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, (1 << 20) - used - free)
        .unwrap();
    let before = host.usage().unwrap().committed_bytes();
    assert!(advance(&mut states, &layout, &vi, &va).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    assert!(states.iter().all(|s| s.solver.step_index() == 0));
    drop((padding, vi, va, states));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn mismatched_identity_geometry_dtype_and_layout_capacity_are_rejected() {
    let host = account(1 << 20);
    let precision = PH3ImageVelocityPrecision::F32;
    let t = crate::p_h3_image::test_fixture::tiny_transformer(16, 2);
    let mut s = state(&host, "request.a", 2, 4, precision, false);
    let requests = [
        s.request.clone(),
        PH3ImageVelocityRequest::new(id("request.b"), PH3ImageReferenceImage::new(2, 4).unwrap()),
    ];
    let layout = PH3ImageVelocityBatchLayout::new(&t, &requests, 4096).unwrap();
    let (vi, va) = physical(&host, &requests, precision, false);
    let wrong = layout
        .image_storage(&id("request.b"), &vi.tensor, &vi.bytes)
        .unwrap();
    let audio = layout
        .audio_storage(&s.request.request, &va.tensor, &va.bytes)
        .unwrap();
    let before = host.usage().unwrap().committed_bytes();
    assert!(s
        .solver
        .prepare_advance(&s.image, wrong, &s.audio, audio)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    assert!(PH3ImageVelocityBatchLayout::new(&t, &requests, 1).is_err());
    let duplicate = [s.request.clone(), s.request.clone()];
    assert!(PH3ImageVelocityBatchLayout::new(&t, &duplicate, 4096).is_err());
    assert!(layout
        .image_storage(&id("request.absent"), &vi.tensor, &vi.bytes)
        .is_err());
    let malformed = ResolvedTensorSpec::new(
        vec![2, 16],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(layout
        .image_storage(&s.request.request, &malformed, &vi.bytes)
        .is_err());
    let (wrong_dtype, _) = physical(&host, &requests, PH3ImageVelocityPrecision::Bf16, false);
    let view = layout
        .image_storage(&s.request.request, &wrong_dtype.tensor, &wrong_dtype.bytes)
        .unwrap();
    assert!(s
        .solver
        .prepare_advance(&s.image, view, &s.audio, audio)
        .is_err());
    let foreign = account(4096);
    let (foreign_image, _) = physical(&foreign, &requests, precision, false);
    let view = layout
        .image_storage(
            &s.request.request,
            &foreign_image.tensor,
            &foreign_image.bytes,
        )
        .unwrap();
    assert!(s
        .solver
        .prepare_advance(&s.image, view, &s.audio, audio)
        .is_err());
    assert_eq!(s.solver.step_index(), 0);
}
