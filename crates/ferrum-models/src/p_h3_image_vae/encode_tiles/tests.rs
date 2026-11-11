//! Complete actual Stored Reference encoder executions, not fabricated moment
//! outputs. Independent scalar pixel/blend/posterior boundaries and B1 runs
//! qualify this consumer; full CNN F64 boundary tests live with encoder core.
use super::*;
use crate::p_h3_image_vae::{
    encoder::{
        tests::{architecture, archive, limits, originals},
        PH3ImageVaeEncoderConfig,
    },
    tiling::PH3ImageVaeSpatialPolicy,
    PH3ImageVaeDecoderConfig,
};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use safetensors::Dtype;

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn config(batch: u64) -> PH3ImageVaeEncoderConfig {
    PH3ImageVaeEncoderConfig::new(architecture(true), batch, 32, 32).unwrap()
}
fn plan() -> PH3ImageVaeTilePlan {
    PH3ImageVaeTilePlan::new(
        48,
        48,
        16,
        PH3ImageVaeSpatialPolicy::Tiled {
            tile_pixels: 32,
            minimum_overlap_pixels: 16,
            max_tiles: 4,
        },
    )
    .unwrap()
}
fn statistics_config() -> PH3ImageVaeDecoderConfig {
    PH3ImageVaeDecoderConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "_class_name":"AutoencoderKLPH3Image","latent_channels":24,"out_channels":3,
            "spatial_downsample_factors":[2,2,2,2,1,1],"decoder_num_layers":2,
            "decoder_num_attention_heads":2,"decoder_attention_head_dim":16,
            "decoder_num_register_tokens":4,"decoder_ffn_mult":4,"decoder_rope_dim_ratio":0.75,
            "decoder_rope_theta":100,"decoder_norm_eps":0.00001
        }))
        .unwrap(),
    )
    .unwrap()
}
fn bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn fixture_source(c: &PH3ImageVaeEncoderConfig) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let mut parameters = originals(c);
    for (key, values) in [
        (
            "latents_mean",
            (0..24)
                .map(|i| i as f32 / 64. - 0.125 + 0.000123)
                .collect::<Vec<_>>(),
        ),
        (
            "latents_std",
            (0..24).map(|i| 0.75 + i as f32 / 64. + 0.000123).collect(),
        ),
    ] {
        parameters.insert(key.into(), (Dtype::F32, vec![24], bytes(values)));
    }
    let (dir, archive) = archive(&parameters);
    (dir, ComfyCanonicalWeightSource::new(archive).unwrap())
}
fn new_parent() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
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
        64,
    )
    .unwrap()
}
fn tensor(
    host: &HostMemoryAccount,
    shape: &[usize],
    values: &[f32],
    name: &str,
) -> ManagedHostTensor {
    let mut output = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    output
        .as_mut()
        .copy_from_slice(&bytes(values.iter().copied()));
    ManagedHostTensor::new(id(name), f32_spec(shape).unwrap(), output.freeze()).unwrap()
}
fn floats(bytes: &ManagedHostBytes) -> Vec<f32> {
    bytes
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn pixels(batch: usize) -> Vec<f32> {
    (0..batch * 3 * 48 * 48)
        .map(|i| {
            if i % (3 * 48 * 48) == 0 {
                2_f32.powi(-26)
            } else {
                ((i * 7 + i / (3 * 48 * 48) * 13) % 67) as f32 / 66.
            }
        })
        .collect()
}
fn noise(batch: usize) -> Vec<f32> {
    (0..batch * 24 * 9)
        .map(|i| ((i * 7 + i / (24 * 9) * 19) % 31) as f32 / 16. - 0.875 + 0.000123)
        .collect()
}
fn initialize(
    backend: &ReferenceVNextComposition,
    c: &PH3ImageVaeEncoderConfig,
    source: &SafetensorsArchive,
    host: &HostMemoryAccount,
) -> TensorComponentSession<ReferenceDeviceRuntime> {
    let prepared = TypedFamilyRegistration::new(PH3ImageVaeEncoderFamily::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &id(PH3ImageVaeEncoderFamily::REFERENCE_PROFILE_ID),
        )
        .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = materializers
        .augment_catalog(backend.catalog().clone())
        .unwrap();
    let d = backend.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        backend.registry(),
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &materializers,
        &c.compile_options().unwrap(),
    )
    .unwrap();
    let checkpoint = PH3ImageVaeEncoderCheckpoint::new(c, source, host.clone(), limits()).unwrap();
    compiled
        .initialize(
            backend.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&checkpoint),
            ComponentInitializationConfig {
                request_id: id("request.encoder_tiles.init"),
                run_id: id("run.encoder_tiles.init"),
                transaction_id: id("transaction.encoder_tiles.init"),
                static_policy: StaticInitializationPolicy::new(16 << 20, 512).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| panic!("complete Stored encoder init: {e:?}"))
}
fn preparation_pressure<T: std::fmt::Debug>(result: Result<T>, class: HostMemoryClass) {
    let error = result.expect_err("real Host pressure must reject allocation");
    assert!(error.chain().any(|e|matches!(e.downcast_ref::<HostMemoryError>(),Some(HostMemoryError::ScopeCapacity {class:c,..}) if *c==class)),"expected real scoped {class:?} capacity failure: {error:#}");
}
fn advance_pressure(
    result: std::result::Result<bool, PH3ImageVaeEncodeError<ReferenceDeviceRuntime>>,
) {
    let Err(PH3ImageVaeEncodeError::Preparation(error)) = result else {
        panic!("expected post-device Host rejection: {result:?}");
    };
    preparation_pressure::<()>(Err(error), HostMemoryClass::Output);
}

fn run_controller(
    backend: &ReferenceVNextComposition,
    c: &PH3ImageVaeEncoderConfig,
    source: &ComfyCanonicalWeightSource,
    pixel_values: &[f32],
    noise_values: &[f32],
    domain: PH3ImageVaePixelDomain,
    pressure: bool,
) -> Vec<f32> {
    let parent = new_parent();
    let mut session = initialize(backend, c, source.archive(), &parent);
    let initial = serde_json::to_value(session.initialization_receipt()).unwrap();
    let mut actor = TensorSerialActor::new(backend.runtime().clone()).unwrap();
    let batch = c.batch() as usize;
    let physical = |n| parent.backing_bytes(n).unwrap();
    let tile_input = batch * 3 * 32 * 32 * 4;
    // Exact one-tile preprocessing credit: a normalized 48x48 full copy cannot
    // fit alongside immutable source/noise in this Input scope.
    let inputs = parent
        .reserve_scope(&[(
            HostMemoryClass::InputPreprocess,
            physical(pixel_values.len() * 4)
                + physical(noise_values.len() * 4)
                + physical(tile_input),
        )])
        .unwrap();
    let input_host = inputs.account();
    let tile_moments = batch * 48 * 2 * 2 * 4;
    let full_moments = batch * 48 * 3 * 3 * 4;
    let readbacks = parent
        .reserve_scope(&[(HostMemoryClass::Output, 3 * physical(tile_moments))])
        .unwrap();
    session
        .set_output_host_account(readbacks.account())
        .unwrap();
    let outputs = parent
        .reserve_scope(&[(
            HostMemoryClass::Output,
            physical(full_moments) + physical(tile_moments),
        )])
        .unwrap();
    let output_host = outputs.account();
    let image = tensor(
        &input_host,
        &[batch, 3, 48, 48],
        pixel_values,
        "value.fixture.edit_image",
    );
    let noise = tensor(
        &input_host,
        &[batch, 24, 3, 3],
        noise_values,
        "value.fixture.provided_posterior_noise",
    );
    let original = image.bytes().bytes().to_vec();
    let original_noise = noise.bytes().bytes().to_vec();
    let statistics_config = statistics_config();
    let checkpoint =
        PH3ImageVaeEncoderCheckpoint::new(c, source.archive(), parent.clone(), limits()).unwrap();
    let before = backend.runtime().snapshot().commands;
    // Numeric budgets alone cannot authorize another parent or another mmap
    // archive, even when the second archive contains identical source bytes.
    let foreign = new_parent();
    assert!(PH3ImageVaeTileEncoder::bind(
        &mut session,
        &checkpoint,
        limits(),
        plan(),
        &image,
        domain,
        &noise,
        PH3ImageVaeLatentStatistics::new(&statistics_config, source).unwrap(),
        id("request.encoder_tiles.foreign"),
        input_host.clone(),
        foreign
    )
    .is_err());
    assert_eq!(backend.runtime().snapshot().commands, before);
    let (_other_dir, other_source) = fixture_source(c);
    let committed = parent.usage().unwrap().committed_bytes();
    assert!(PH3ImageVaeTileEncoder::bind(
        &mut session,
        &checkpoint,
        limits(),
        plan(),
        &image,
        domain,
        &noise,
        PH3ImageVaeLatentStatistics::new(&statistics_config, &other_source).unwrap(),
        id("request.encoder_tiles.other_archive"),
        input_host.clone(),
        output_host.clone()
    )
    .is_err());
    assert_eq!(parent.usage().unwrap().committed_bytes(), committed);
    assert_eq!(backend.runtime().snapshot().commands, before);
    let statistics = PH3ImageVaeLatentStatistics::new(&statistics_config, source).unwrap();
    let mut encoder = PH3ImageVaeTileEncoder::bind(
        &mut session,
        &checkpoint,
        limits(),
        plan(),
        &image,
        domain,
        &noise,
        statistics,
        id("request.encoder_tiles"),
        input_host.clone(),
        output_host.clone(),
    )
    .unwrap();
    assert!(encoder.finish(id("value.incomplete")).is_err());
    let tile = encoder.tile_input(plan().tile(0).unwrap()).unwrap();
    let crop = floats(tile.bytes());
    let expected_crop = literal_crop(pixel_values, batch, 0, 0, domain);
    assert_eq!(
        crop, expected_crop,
        "direct tile normalization changed the separate F32 source boundary"
    );
    drop(tile);
    let mut blocker = pressure.then(|| {
        output_host
            .allocate_zeroed(HostMemoryClass::Output, tile_moments)
            .unwrap()
            .freeze()
    });
    if pressure {
        advance_pressure(encoder.advance(&mut actor, id("run.encoder_tiles.first")));
        assert_eq!(encoder.completed_tiles(), 0);
        assert_eq!(encoder.executed_tiles(), 1);
        // Reference uses command evidence; timing is Off for this fixture.
        // The facade must preserve the absence of device timing evidence.
        assert!(encoder.take_execution_profile().is_none());
        assert!(
            !actor.is_failed(),
            "Host postprocessing cannot poison a successfully fenced actor"
        );
        let pointer = encoder.pending.as_ref().unwrap()[0]
            .bytes()
            .bytes()
            .as_ptr();
        let commands = backend.runtime().snapshot().commands;
        assert!(
            commands > before,
            "first Host failure must follow real full encoder execution"
        );
        advance_pressure(encoder.advance(&mut actor, id("run.encoder_tiles.retry")));
        assert_eq!(encoder.executed_tiles(), 1);
        assert!(encoder.take_execution_profile().is_none());
        assert_eq!(
            encoder.pending.as_ref().unwrap()[0]
                .bytes()
                .bytes()
                .as_ptr(),
            pointer
        );
        assert_eq!(backend.runtime().snapshot().commands, commands);
        drop(blocker.take());
        assert!(!encoder
            .advance(&mut actor, id("run.encoder_tiles.commit_pending"))
            .unwrap());
        assert_eq!(backend.runtime().snapshot().commands, commands);
        assert_eq!(encoder.completed_tiles(), 1);
        assert_eq!(encoder.executed_tiles(), 1);
    }
    while encoder.completed_tiles() < 4 {
        let tile = encoder.completed_tiles();
        let commands = backend.runtime().snapshot().commands;
        let done = encoder
            .advance(&mut actor, id(format!("run.encoder_tiles.tile{tile}")))
            .unwrap();
        assert_eq!(done, tile == 3);
        assert_eq!(encoder.completed_tiles(), tile + 1);
        assert_eq!(encoder.executed_tiles(), tile + 1);
        assert!(encoder.take_execution_profile().is_none());
        assert!(backend.runtime().snapshot().commands > commands);
    }
    let commands = backend.runtime().snapshot().commands;
    if pressure {
        // Consume the preprocessing credit only after every device tile has
        // finished. finish must retain the whole assembled moments on failure.
        let blocker = input_host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, tile_input)
            .unwrap();
        preparation_pressure(
            encoder.finish(id("value.reference_latents")),
            HostMemoryClass::InputPreprocess,
        );
        assert!(encoder.canvas.is_none());
        let pointer = encoder.moments.as_ref().unwrap().1.bytes().as_ptr();
        let content = encoder.moments.as_ref().unwrap().1.bytes().to_vec();
        preparation_pressure(
            encoder.finish(id("value.reference_latents")),
            HostMemoryClass::InputPreprocess,
        );
        assert_eq!(
            encoder.moments.as_ref().unwrap().1.bytes().as_ptr(),
            pointer
        );
        assert_eq!(encoder.moments.as_ref().unwrap().1.bytes(), content);
        assert_eq!(backend.runtime().snapshot().commands, commands);
        drop(blocker);
    }
    let result = encoder.finish(id("value.reference_latents")).unwrap();
    assert_eq!(result.tensor(), &f32_spec(&[batch, 24, 3, 3]).unwrap());
    assert!(encoder.finish(id("value.reference_latents.again")).is_err());
    assert_eq!(backend.runtime().snapshot().commands, commands);
    assert_eq!(image.bytes().bytes(), original);
    assert_eq!(noise.bytes().bytes(), original_noise);
    drop(encoder);
    assert_eq!(
        serde_json::to_value(session.initialization_receipt()).unwrap(),
        initial
    );
    drop(actor);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("tile encoder close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    inputs.close().unwrap();
    readbacks.close().unwrap();
    outputs.close().unwrap();
    assert!(parent.owns(result.bytes()));
    let values = floats(result.bytes());
    assert!(values.iter().all(|v| v.is_finite()));
    drop((result, image, noise));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(backend.runtime().snapshot().live_allocations, 0);
    values
}

/// Literal crop+normalization independent of production normalization helpers.
fn literal_crop(
    pixels: &[f32],
    batch: usize,
    oy: usize,
    ox: usize,
    domain: PH3ImageVaePixelDomain,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(batch * 3 * 32 * 32);
    for b in 0..batch {
        for c in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    let pixel = pixels[((b * 3 + c) * 48 + oy + y) * 48 + ox + x];
                    let signed = if domain == PH3ImageVaePixelDomain::UnitRgb {
                        let scaled = pixel * 2.;
                        scaled - 1.
                    } else {
                        pixel
                    };
                    let shifted = signed + 1.;
                    let unit = shifted * 0.5;
                    let centered = unit - [0.485_f32, 0.456, 0.406][c];
                    output.push(centered / [0.229_f32, 0.224, 0.225][c]);
                }
            }
        }
    }
    output
}

/// Independent single-request complete Stored runs return original raw moments.
/// The seam oracle below consumes these raw neighbours, never blended tails.
fn independent_b1(
    backend: &ReferenceVNextComposition,
    source: &ComfyCanonicalWeightSource,
    pixels: &[f32],
    noise: &[f32],
) -> Vec<f32> {
    let parent = new_parent();
    let c = config(1);
    let mut session = initialize(backend, &c, source.archive(), &parent);
    let mut actor = TensorSerialActor::new(backend.runtime().clone()).unwrap();
    let mut raw = Vec::new();
    for (i, (y, x)) in [(0, 0), (0, 16), (16, 0), (16, 16)].into_iter().enumerate() {
        let crop = literal_crop(pixels, 1, y, x, PH3ImageVaePixelDomain::UnitRgb);
        let input = tensor(
            &parent,
            &[1, 3, 32, 32],
            &crop,
            PH3ImageVaeEncoderFamily::INPUT_ID,
        );
        let output = session
            .execute_on_actor(
                &mut actor,
                id(format!("run.encoder_oracle.tile{i}")),
                id("request.encoder_oracle"),
                vec![input],
            )
            .unwrap();
        assert_eq!(output.len(), 1);
        raw.push(floats(output[0].bytes()));
    }
    let mut moments = vec![0.; 48 * 9];
    for (i, (oy, ox)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
        let mut work = raw[i].clone();
        for c in 0..48 {
            // overlap=one latent row/column. At ratio0 the upper/left raw
            // neighbour dominates; corner distinction survives this fixture.
            if oy > 0 {
                for x in 0..2 {
                    work[c * 4 + x] = raw[i - 2][c * 4 + 2 + x] * 1. + work[c * 4 + x] * 0.;
                }
            }
            if ox > 0 {
                for y in 0..2 {
                    work[c * 4 + y * 2] =
                        raw[i - 1][c * 4 + y * 2 + 1] * 1. + work[c * 4 + y * 2] * 0.;
                }
            }
            let kept_h = if oy == 0 { 1 } else { 2 };
            let kept_w = if ox == 0 { 1 } else { 2 };
            for y in 0..kept_h {
                for x in 0..kept_w {
                    moments[(c * 3 + oy + y) * 3 + ox + x] = work[c * 4 + y * 2 + x];
                }
            }
        }
    }
    let mean: Vec<_> = source
        .archive()
        .tensor("latents_mean")
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let std: Vec<_> = source
        .archive()
        .tensor("latents_std")
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut result = vec![0.; 24 * 9];
    for c in 0..24 {
        for p in 0..9 {
            let logvar = moments[(c + 24) * 9 + p].clamp(-30., 20.);
            let deviation = libm::expf(0.5_f32 * logvar);
            let sample = deviation.mul_add(noise[c * 9 + p], moments[c * 9 + p]);
            let centered = sample - mean[c];
            result[c * 9 + p] = centered / std[c];
        }
    }
    drop(actor);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("oracle close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    result
}

#[test]
fn actual_full_stored_reference_encoder_tiles_match_independent_b1_and_retry_host_pressure() {
    let backend = ReferenceVNextComposition::create(id("device.reference.encoder_tiles")).unwrap();
    let c = config(2);
    let (_dir, source) = fixture_source(&c);
    let pixels = pixels(2);
    let noise = noise(2);
    let actual = run_controller(
        &backend,
        &c,
        &source,
        &pixels,
        &noise,
        PH3ImageVaePixelDomain::UnitRgb,
        true,
    );
    let mut expected = Vec::new();
    for b in 0..2 {
        expected.extend(independent_b1(
            &backend,
            &source,
            &pixels[b * 3 * 48 * 48..(b + 1) * 3 * 48 * 48],
            &noise[b * 24 * 9..(b + 1) * 24 * 9],
        ));
    }
    assert_eq!(
        actual, expected,
        "B2 moment blending/posterior differs from independent complete B1 trajectories"
    );
    // Same original image through explicit SignedRgb retains the wrapper's
    // required F32 round trip and request identity; source bytes stay fixed.
    let signed: Vec<_> = pixels[..3 * 48 * 48]
        .iter()
        .map(|&p| {
            let scaled = p * 2.;
            scaled - 1.
        })
        .collect();
    let single = run_controller(
        &backend,
        &config(1),
        &source,
        &signed,
        &noise[..24 * 9],
        PH3ImageVaePixelDomain::SignedRgb,
        false,
    );
    assert_eq!(single, actual[..24 * 9]);
    assert_eq!(backend.runtime().snapshot().live_allocations, 0);
}
