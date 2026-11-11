//! Real Stored CUDA decoder/controller integration. There is no executable
//! Reference W8A8-F32 provider; the CPU quantization oracle is not registered
//! as a backend. All decoder math below uses the original raw W8 fixture.
use super::*;
use crate::p_h3_image_vae::{
    decoder::tests::{
        architecture, archive,
        cuda::{Execution, GPU},
        limits, originals,
    },
    decoder::{PH3ImageVaeDecoderExecutionConfig, PH3ImageVaeDecoderFamily as Family},
    decoder_block::PH3ImageVaeDecoderBlockExecutionPreset,
    tiling::PH3ImageVaeSpatialPolicy,
};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig, TensorSerialActor,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use safetensors::Dtype;
use std::collections::BTreeMap;

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn config(batch: u64) -> PH3ImageVaeDecoderExecutionConfig {
    PH3ImageVaeDecoderExecutionConfig::new(
        architecture(2),
        batch,
        2,
        2,
        PH3ImageVaeDecoderBlockExecutionPreset::F32A8FullAttention,
    )
    .unwrap()
}
fn plan() -> PH3ImageVaeTilePlan {
    PH3ImageVaeTilePlan::new(
        6,
        6,
        2,
        PH3ImageVaeSpatialPolicy::Tiled {
            tile_pixels: 4,
            minimum_overlap_pixels: 2,
            max_tiles: 4,
        },
    )
    .unwrap()
}
fn statistics_source(
    c: &PH3ImageVaeDecoderExecutionConfig,
) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let mut parameters = originals(c);
    for (name, values) in [
        (
            "latents_mean",
            (0..24)
                .map(|i| i as f32 / 64. - 0.125 + 0.000123)
                .collect::<Vec<_>>(),
        ),
        (
            "latents_std",
            (0..24)
                .map(|i| 0.75 + i as f32 / 64. + 0.000123)
                .collect::<Vec<_>>(),
        ),
    ] {
        parameters.insert(name.into(), (Dtype::F32, vec![24], bytes(values)));
    }
    archive(&parameters)
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 48 << 20,
                transfer_staging_bytes: 32 << 20,
                weight_backing_bytes: 32 << 20,
            },
        )
        .unwrap(),
    )
}
fn normalized(batch: usize) -> Vec<f32> {
    (0..batch * 24 * 9)
        .map(|i| ((i * 7 + i / (24 * 9) * 13) % 59) as f32 / 32. - 0.75 + 0.000123)
        .collect()
}
fn tensor(host: &HostMemoryAccount, values: &[f32], batch: usize) -> ManagedHostTensor {
    let data = bytes(values.iter().copied());
    let mut backing = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
        .unwrap();
    backing.as_mut().copy_from_slice(&data);
    ManagedHostTensor::new(
        id("value.fixture.normalized_sampler"),
        ResolvedTensorSpec::new(
            vec![batch as u64, 24, 3, 3],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        backing.freeze(),
    )
    .unwrap()
}
fn initialize(
    f: &Execution,
    c: &PH3ImageVaeDecoderExecutionConfig,
    source: &ComfyCanonicalWeightSource,
    host: &HostMemoryAccount,
) -> TensorComponentSession<CudaDeviceRuntime> {
    let family = Family::new(limits()).unwrap();
    let options = family.compile_options(c).unwrap();
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &id(Family::CUDA_PROFILE_ID),
        )
        .unwrap();
    let d = f.runtime.descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        &f.registry,
        &f.catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &f.materializers,
        &options,
    )
    .unwrap();
    let checkpoint = PH3ImageVaeDecoderCheckpoint::new(c, source, host.clone(), limits()).unwrap();
    compiled
        .initialize(
            f.runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&checkpoint),
            ComponentInitializationConfig {
                request_id: id("request.tiled.init"),
                run_id: id("run.tiled.init"),
                transaction_id: id("transaction.tiled.init"),
                static_policy: StaticInitializationPolicy::new(32 << 20, 1024).unwrap(),
                executor: TensorExecutorConfig {
                    device_timing_mode: DeviceTimingMode::Completion,
                    ..TensorExecutorConfig::default()
                },
            },
        )
        .unwrap_or_else(|e| panic!("tiled Stored init: {e:?}"))
}
fn output_pressure(result: std::result::Result<bool, PH3ImageVaeDecodeError<CudaDeviceRuntime>>) {
    let Err(PH3ImageVaeDecodeError::Preparation(error)) = result else {
        panic!("expected Host postprocessing rejection, received {result:?}");
    };
    assert!(
        error.chain().any(|e| matches!(
            e.downcast_ref::<HostMemoryError>(),
            Some(HostMemoryError::ScopeCapacity {
                class: HostMemoryClass::Output,
                ..
            })
        )),
        "expected real scoped Output admission pressure: {error:#}"
    );
}
fn run_controller(
    f: &Execution,
    c: &PH3ImageVaeDecoderExecutionConfig,
    source: &ComfyCanonicalWeightSource,
    values: &[f32],
    retry: bool,
) -> Vec<f32> {
    let parent = host();
    let mut session = initialize(f, c, source, &parent);
    let initial = serde_json::to_value(session.initialization_receipt()).unwrap();
    let mut actor = TensorSerialActor::new(f.runtime.clone()).unwrap();
    let batch = c.batch() as usize;
    let layout = PH3ImageVaeImageLayout::new(c.architecture().clone(), batch, 2, 2).unwrap();
    let physical = |logical| parent.backing_bytes(logical).unwrap();
    let input_budget = physical(values.len() * 4)
        + physical(batch * 4 * 24 * 4)
        + 2 * physical(
            layout
                .frequency_spec()
                .unwrap()
                .minimum_storage_bytes()
                .unwrap() as usize,
        );
    let inputs = parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, input_budget)])
        .unwrap();
    let input_host = inputs.account();
    let readback_bytes = layout
        .patch_output_spec()
        .unwrap()
        .minimum_storage_bytes()
        .unwrap() as usize;
    let readbacks = parent
        .reserve_scope(&[(HostMemoryClass::Output, physical(readback_bytes))])
        .unwrap();
    session
        .set_output_host_account(readbacks.account())
        .unwrap();
    let canvas = physical(batch * 3 * 6 * 6 * 4);
    let pixel_logical = batch * 3 * 4 * 4 * 4;
    let strip_logical = batch * 3 * 2 * 6 * 4;
    let tail_logical = batch * 3 * 4 * 2 * 4;
    let output_budget = canvas
        + 2 * physical(pixel_logical)
        + 2 * physical(strip_logical)
        + 2 * physical(tail_logical);
    let outputs = parent
        .reserve_scope(&[(HostMemoryClass::Output, output_budget)])
        .unwrap();
    let output_host = outputs.account();
    let hold = |logical| {
        output_host
            .allocate_zeroed(HostMemoryClass::Output, logical)
            .unwrap()
            .freeze()
    };
    let mut first_blocker = retry.then(|| hold(pixel_logical));
    let mut rest_blockers = if retry {
        vec![
            hold(pixel_logical),
            hold(strip_logical),
            hold(strip_logical),
            hold(tail_logical),
            hold(tail_logical),
        ]
    } else {
        vec![]
    };
    let normalized = tensor(&input_host, values, batch);
    let original = normalized.bytes().bytes().to_vec();
    let checkpoint =
        PH3ImageVaeDecoderCheckpoint::new(c, source, parent.clone(), limits()).unwrap();
    // Equal numeric budgets from an unrelated parent cannot authorize this
    // session, sampler or canvas. Failure precedes decoder execution.
    let foreign = host();
    let before = parent.usage().unwrap().committed_bytes();
    assert!(PH3ImageVaeTileDecoder::bind(
        &mut session,
        &checkpoint,
        limits(),
        plan(),
        &normalized,
        id("request.tiled.foreign"),
        input_host.clone(),
        foreign
    )
    .is_err());
    assert_eq!(parent.usage().unwrap().committed_bytes(), before);
    let mut decoder = PH3ImageVaeTileDecoder::bind(
        &mut session,
        &checkpoint,
        limits(),
        plan(),
        &normalized,
        id("request.tiled"),
        input_host.clone(),
        output_host.clone(),
    )
    .unwrap();
    let frequency_owners = (
        decoder.state.cosine.bytes().bytes().as_ptr(),
        decoder.state.sine.bytes().bytes().as_ptr(),
    );
    if retry {
        output_pressure(decoder.advance(&mut actor, id("run.tiled.first.gpu")));
        assert_eq!(decoder.completed_tiles(), 0);
        let patch_pointer = match decoder.state.pending.as_ref() {
            Some(Pending::Patches(p)) => p[0].bytes().bytes().as_ptr(),
            _ => panic!("actual completed patch owner must survive Host failure"),
        };
        assert!(
            decoder.session.take_execution_profile().is_some(),
            "the first failure follows actual GPU completion"
        );
        // Holding exactly the postprocessing credit forces another admission
        // failure. Its retry must preserve the same completed readback owner.
        output_pressure(decoder.advance(&mut actor, id("run.tiled.retry.patches")));
        assert!(decoder.session.take_execution_profile().is_none());
        assert_eq!(
            match decoder.state.pending.as_ref() {
                Some(Pending::Patches(p)) => p[0].bytes().bytes().as_ptr(),
                _ => panic!("pending patch was lost"),
            },
            patch_pointer
        );
        drop(first_blocker.take());
        output_pressure(decoder.advance(&mut actor, id("run.tiled.retry.unpatchify")));
        assert_eq!(decoder.completed_tiles(), 0);
        assert!(decoder.session.take_execution_profile().is_none());
        let pixel_pointer = match decoder.state.pending.as_ref() {
            Some(Pending::Pixels(_, p)) => p.bytes().as_ptr(),
            _ => panic!("completed pixels must survive stitch admission failure"),
        };
        output_pressure(decoder.advance(&mut actor, id("run.tiled.retry.pixels")));
        assert!(decoder.session.take_execution_profile().is_none());
        assert_eq!(
            match decoder.state.pending.as_ref() {
                Some(Pending::Pixels(_, p)) => p.bytes().as_ptr(),
                _ => panic!("pending pixels were lost"),
            },
            pixel_pointer
        );
        rest_blockers.clear();
        assert!(!decoder
            .advance(&mut actor, id("run.tiled.retry.commit"))
            .unwrap());
        assert_eq!(decoder.completed_tiles(), 1);
        assert!(
            decoder.session.take_execution_profile().is_none(),
            "Host retries cannot repeat GPU execution"
        );
    }
    while decoder.completed_tiles() < 4 {
        let index = decoder.completed_tiles();
        let done = decoder
            .advance(&mut actor, id(format!("run.tiled.tile{index}")))
            .unwrap();
        assert_eq!(done, index == 3);
        assert_eq!(decoder.completed_tiles(), index + 1);
        assert!(decoder.session.take_execution_profile().is_some());
        assert_eq!(
            (
                decoder.state.cosine.bytes().bytes().as_ptr(),
                decoder.state.sine.bytes().bytes().as_ptr()
            ),
            frequency_owners,
            "each tile retains the same charged immutable frequencies"
        );
    }
    let (shape, canvas) = decoder.finish().unwrap();
    assert_eq!(shape.dimensions(), [batch as u64, 3, 6, 6]);
    assert!(output_host.owns(&canvas));
    assert_eq!(normalized.bytes().bytes(), original);
    assert_eq!(
        serde_json::to_value(session.initialization_receipt()).unwrap(),
        initial,
        "all tiles reuse the installed static weights"
    );
    drop(actor);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("tile session close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    inputs.close().unwrap();
    readbacks.close().unwrap();
    outputs.close().unwrap();
    assert!(parent.owns(&canvas));
    let result = canvas
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(result
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    drop(canvas);
    drop(normalized);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    result
}

fn read(source: &ComfyCanonicalWeightSource, name: &str) -> Vec<f32> {
    source
        .archive()
        .tensor(name)
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn blend(a: f32, b: f32, i: usize) -> f32 {
    let ratio = i as f32 / 2.;
    let lhs = a * (1. - ratio);
    let rhs = b * ratio;
    lhs + rhs
}
/// Independent literal 2x2 tile oracle: normalized NCHW crop/denormalize,
/// real full Stored GPU decoder, target-only unpatchification, then recursive
/// vertical-before-horizontal F32 stitching and one final RGB normalization.
fn expected(
    f: &Execution,
    c: &PH3ImageVaeDecoderExecutionConfig,
    source: &ComfyCanonicalWeightSource,
    normalized: &[f32],
) -> Vec<f32> {
    let batch = c.batch() as usize;
    let mean = read(source, "latents_mean");
    let std = read(source, "latents_std");
    let account = host();
    let layout = PH3ImageVaeImageLayout::new(c.architecture().clone(), batch, 2, 2).unwrap();
    let (cosine, sine) = layout
        .managed_rotary(&account, id(Family::COSINE_ID), id(Family::SINE_ID))
        .unwrap();
    let read_tensor = |t: &ManagedHostTensor| {
        t.bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let mut tiles = Vec::new();
    for tile in 0..4 {
        let mut rows = Vec::new();
        for b in 0..batch {
            for y in 0..2 {
                for x in 0..2 {
                    for channel in 0..24 {
                        let index = ((b * 24 + channel) * 3 + tile / 2 + y) * 3 + tile % 2 + x;
                        let scaled = normalized[index] * std[channel];
                        rows.push(scaled + mean[channel]);
                    }
                }
            }
        }
        let outputs = f.run(
            c,
            source,
            &BTreeMap::from([
                (id(Family::INPUT_ID), rows),
                (id(Family::COSINE_ID), read_tensor(&cosine)),
                (id(Family::SINE_ID), read_tensor(&sine)),
            ]),
        );
        let mut pixels = vec![0.; batch * 3 * 16];
        for b in 0..batch {
            for channel in 0..3 {
                for y in 0..4 {
                    for x in 0..4 {
                        let patch = (y / 2) * 2 + x / 2;
                        let column = (channel * 2 + y % 2) * 2 + x % 2;
                        pixels[((b * 3 + channel) * 4 + y) * 4 + x] =
                            outputs[(b * 9 + patch) * 12 + column];
                    }
                }
            }
        }
        tiles.push(pixels);
    }
    let mut canvas = vec![0.; batch * 3 * 36];
    for plane in 0..batch * 3 {
        // First tile-row's horizontal blend, then its complete blended strip.
        for y in 0..4 {
            for x in 0..2 {
                let previous = tiles[0][plane * 16 + y * 4 + x + 2];
                let at = plane * 16 + y * 4 + x;
                tiles[1][at] = blend(previous, tiles[1][at], x);
            }
        }
        let strip = |y: usize, x: usize, tiles: &Vec<Vec<f32>>| {
            if x < 2 {
                tiles[0][plane * 16 + (y + 2) * 4 + x]
            } else {
                tiles[1][plane * 16 + (y + 2) * 4 + x - 2]
            }
        };
        for tile in [2, 3] {
            for y in 0..2 {
                for x in 0..4 {
                    let previous = strip(y, x + (tile % 2) * 2, &tiles);
                    let at = plane * 16 + y * 4 + x;
                    tiles[tile][at] = blend(previous, tiles[tile][at], y);
                }
            }
        }
        for y in 0..4 {
            for x in 0..2 {
                let previous = tiles[2][plane * 16 + y * 4 + x + 2];
                let at = plane * 16 + y * 4 + x;
                tiles[3][at] = blend(previous, tiles[3][at], x);
            }
        }
        for tile in 0..4 {
            let kept_y = if tile < 2 { 2 } else { 4 };
            let kept_x = if tile % 2 == 0 { 2 } else { 4 };
            for y in 0..kept_y {
                for x in 0..kept_x {
                    canvas[plane * 36 + (tile / 2 * 2 + y) * 6 + tile % 2 * 2 + x] =
                        tiles[tile][plane * 16 + y * 4 + x];
                }
            }
        }
        let channel = plane % 3;
        for p in 0..36 {
            let scaled = canvas[plane * 36 + p] * [0.229, 0.224, 0.225][channel];
            canvas[plane * 36 + p] = (scaled + [0.485, 0.456, 0.406][channel]).clamp(0., 1.);
        }
    }
    canvas
}

#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn stored_tiled_decoder_four_tiles_preserves_parent_scopes_and_retries_host_failures_without_reexecuting_gpu(
) {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let f = Execution::new();
    let c = config(2);
    let (_dir, source) = statistics_source(&c);
    let normalized = normalized(2);
    let actual = run_controller(&f, &c, &source, &normalized, true);
    let wanted = expected(&f, &c, &source, &normalized);
    assert_eq!(actual,wanted,"controller normalization, target placement and recursive seams must match independent literal composition");
    let per = 24 * 9;
    for sample in 0..2 {
        let one = config(1);
        let independently = run_controller(
            &f,
            &one,
            &source,
            &normalized[sample * per..(sample + 1) * per],
            false,
        );
        for (i, (&a, &b)) in actual[sample * 3 * 36..(sample + 1) * 3 * 36]
            .iter()
            .zip(&independently)
            .enumerate()
        {
            let bound = 1e-4 * (1. + b.abs());
            assert!(
                (a - b).abs() <= bound,
                "B2/B1 sample{sample} pixel{i}: {a} vs {b}"
            );
        }
    }
}
