//! Real Metal denoising from already prepared, admitted tiny conditioning.
//! This exercises the engine's D2D request handoff and VAE, not PNG/vision
//! preprocessing (whose public edit path intentionally prepares a 1024² input).
use super::*;
use crate::image_generation::{prefix_denoising::PrefixDenoise, prepare::PreparedPrefixDecode};
use ferrum_kernels::backend::metal::vnext_runtime::MetalDeviceRuntime;
use ferrum_models::qwen_image21::{ExpectedWeights, QwenImage21TransformerPhase};
use half::bf16;

type Pipeline = QwenImage21Pipeline<MetalDeviceRuntime>;
type F = QwenImage21TransformerFamily;

fn nonzero_archive(
    directory: &std::path::Path,
    schema: &WeightSchema,
    transformer: bool,
) -> Arc<SafetensorsArchive> {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    std::fs::create_dir(directory).unwrap();
    let tensors: Vec<_> = schema
        .components
        .iter()
        .map(|weight| {
            let [name] = weight.external_names.as_slice() else {
                panic!("fixture requires a single physical source")
            };
            let shape: Vec<_> = weight.dimensions.iter().map(|&n| n as usize).collect();
            let bytes: Vec<_> = (0..shape.iter().product())
                .flat_map(|i| {
                    let value = if transformer {
                        dit_weight(name, &shape, i)
                    } else if name.ends_with("gamma") || name.ends_with("norm.weight") {
                        1.0
                    } else if name.ends_with("bias") || name.ends_with("beta") {
                        0.0
                    } else {
                        ((i * 7 + i / 11) % 13) as f32 / 32.0 - 0.1875
                    };
                    bf16::from_f32(value).to_bits().to_le_bytes()
                })
                .collect();
            (name.clone(), shape, bytes)
        })
        .collect();
    let views = tensors.iter().map(|(name, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, &None, &directory.join("model.safetensors")).unwrap();
    Arc::new(SafetensorsArchive::open(directory).unwrap())
}

fn dit_weight(name: &str, shape: &[usize], i: usize) -> f32 {
    if shape.len() == 1 {
        return 1.0; // RMS scales, including both Q/K norms.
    }
    let indicator = |yes| if yes { 1.0 } else { 0.0 };
    let (row, col) = (i / shape[1], i % shape[1]);
    match name {
        "img_in.weight" => indicator(row == col),
        "proj_out.weight" => indicator(row % 8 == col),
        "txt_in.in_layer.weight" => indicator(row == col),
        "txt_in.out_layer.weight" => indicator(row == col),
        "time_text_embed.timestep_embedder.linear_1.weight" => {
            8.0 * indicator(row == 0 && col == 0)
        }
        "time_text_embed.timestep_embedder.linear_2.weight" => indicator(row == 0 && col == 0),
        // Nonzero reference gates at t=0; no random cancellation in the fixture.
        "modulation.1.weight" => {
            if col == 0 {
                [0.0, 0.125, 0.0, 0.25][row / 8]
            } else {
                0.0
            }
        }
        "norm_out.linear.weight" => 0.0,
        _ if name.ends_with("attn.to_q.weight") || name.ends_with("attn.to_k.weight") => 0.0,
        _ if name.ends_with("attn.to_v.weight") => 0.5 * indicator(row == col),
        _ if name.ends_with("attn.to_out.0.weight") => indicator(col == (row + 1) % 8),
        _ => 0.0,
    }
}

fn input(
    pipeline: &Pipeline,
    config: &QwenImage21TransformerConfig,
    name: &str,
    offset: usize,
) -> ManagedHostTensor {
    let spec = config
        .input_tensor_specs()
        .unwrap()
        .remove(&id(name).unwrap())
        .unwrap();
    let tensor = ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap();
    let mut bytes = pipeline
        .host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            tensor.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for (i, pair) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
        let value = ((i * 3 + i / 8 + offset) % 17) as f32 * 0.125 - 1.0;
        pair.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    ManagedHostTensor::new(id(name).unwrap(), tensor, bytes.freeze()).unwrap()
}

fn complete<T>(result: Result<T, ImageGenerationFailure<MetalDeviceRuntime>>) -> T {
    result.unwrap_or_else(|error| panic!("actual Metal prefix execution: {error}"))
}

struct ResultImage {
    latent: ManagedHostTensor,
    rgba: TensorExecutionOutput,
}

fn execute(
    pipeline: &mut Pipeline,
    policy: ImageEditPrefixPolicy,
    reference_offset: usize,
    job: u64,
) -> ResultImage {
    let full = QwenImage21TransformerConfig::new(
        pipeline.assets.transformer.clone(),
        QwenImage21T2iLayout::new(4, 2, 2, vec![4]).unwrap(),
    )
    .unwrap()
    .with_reference_image(1, 1, 2, 2)
    .unwrap();
    let cached = policy == ImageEditPrefixPolicy::RequestCache;
    let config = full
        .clone()
        .with_execution_phase(if cached {
            QwenImage21TransformerPhase::PrefixPrefill
        } else {
            QwenImage21TransformerPhase::Full
        })
        .unwrap();
    let denoiser = pipeline.prepare_denoiser(&config, 1).unwrap();
    let decode = cached.then(|| {
        let config = full
            .clone()
            .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
            .unwrap();
        let stage = pipeline.prepare_denoiser(&config, 1).unwrap();
        assert!(
            stage
                .component
                .executable()
                .execution_plan()
                .payload()
                .memory()
                .minimum_request_bytes()
                > 0
        );
        assert_eq!(
            config.prefix_cache_ports().unwrap().len(),
            2 * pipeline.assets.transformer.layers() as usize
        );
        PreparedPrefixDecode { config, stage }
    });
    let decoder = pipeline.assets.decoder(2, 2).unwrap();
    let family = pipeline
        .prepare_family(
            ImageStage::Decoder,
            ImageStagePlanRole::WholeStage,
            1,
            QwenImage21VaeDecoderFamily::new().unwrap(),
            &decoder,
            &pipeline.profiles.decoder,
        )
        .unwrap();
    let vae = PreparedStage::compile(
        ImageStage::Decoder,
        1,
        family,
        &decoder.compile_options().unwrap(),
        pipeline,
    )
    .unwrap();
    let mut prepared = vec![&denoiser, &vae];
    if let Some(decode) = &decode {
        prepared.push(&decode.stage);
    }
    pipeline
        .validate_capacity_with_prefix(&prepared, decode.as_ref().map(|d| (&denoiser, &d.stage)))
        .unwrap();
    drop(prepared);

    let mut sample = input(pipeline, &full, F::LATENTS_ID, 0);
    let text = input(pipeline, &full, F::TEXT_ID, 3);
    let reference = input(pipeline, &full, F::REFERENCE_LATENTS_ID, reference_offset);
    let frequencies = config.managed_rotary_frequencies(&pipeline.host).unwrap();
    let mut solver = ManagedBf16Euler::new(
        pipeline.assets.schedule.solver().clone(),
        pipeline.host.clone(),
        sample.tensor().clone(),
        pipeline.profiles.euler,
    )
    .unwrap();
    let expected_steps = pipeline.assets.schedule.evaluation_count();
    assert!(
        expected_steps >= 2,
        "fixture must execute prefill and decode"
    );
    let mut stages = Vec::new();
    let mut steps = Vec::new();
    if let Some(decode) = decode {
        sample = complete(pipeline.run_prefix_denoising(
            PrefixDenoise {
                prefill: denoiser,
                decode,
                config,
                sample,
                text,
                reference,
                frequencies,
            },
            &mut solver,
            job,
            &mut stages,
            &mut steps,
        ));
    } else {
        let mut active = complete(pipeline.start_stage(denoiser, job, &mut stages));
        while let Some(step) = solver.current_step() {
            let time = full
                .managed_time_features(&pipeline.host, step.timestep().as_f32())
                .unwrap();
            let (next, outputs) = complete(pipeline.execute_stage(
                active,
                job,
                vec![
                    sample.clone(),
                    text.clone(),
                    time,
                    frequencies.clone(),
                    reference.clone(),
                ],
            ));
            active = next;
            let velocity = only_output(outputs, F::VELOCITY_ID).unwrap();
            sample = solver
                .advance(&sample, velocity.tensor(), velocity.bytes())
                .unwrap()
                .1;
            steps.push(ImageDenoiseStepReceipt {
                index: step.index(),
                sigma: step.sigma(),
                next_sigma: step.next_sigma(),
                scheduler_timestep: step.timestep().as_f32(),
            });
        }
        complete(pipeline.finish_stage(active, &mut stages));
    }
    assert_eq!(steps.len(), expected_steps);
    for (role, count) in if cached {
        vec![
            (ImageStagePlanRole::PrefixPrefill, 1),
            (ImageStagePlanRole::PrefixDecode, expected_steps - 1),
        ]
    } else {
        vec![(ImageStagePlanRole::WholeStage, expected_steps)]
    } {
        assert_eq!(
            stages
                .iter()
                .filter(|s| s.execution.plan.stage == ImageStage::Denoiser
                    && s.execution.plan.role == role)
                .map(|s| s.execution.completed_executions)
                .sum::<usize>(),
            count
        );
    }
    if cached {
        let decoder = stages
            .iter()
            .find(|s| {
                s.execution.plan.role == ImageStagePlanRole::PrefixDecode
                    && s.execution.completed_executions > 0
            })
            .unwrap();
        assert!(
            decoder.execution.weights_reused,
            "decode must attach the full prefill installation"
        );
    }
    let latents = QwenImage21ManagedLatents::new(
        &pipeline.host,
        sample.clone(),
        QwenImage21LatentShape::new(1, 2, 2).unwrap(),
        QwenImage21LatentLayout::Packed,
        QwenImage21LatentDomain::Normalized,
    )
    .unwrap();
    let vae_input = latents
        .denormalize_for_vae(&pipeline.host, &pipeline.assets.statistics)
        .unwrap()
        .into_input_tensor()
        .unwrap();
    drop(latents);
    let rgba =
        complete(pipeline.decode_batch(vae, &decoder, vae_input, 1, job, &mut stages)).remove(0);
    assert_eq!(rgba.tensor().dimensions(), [1, 4, 32, 32]);
    assert!(stages.iter().all(|s| matches!(
        s.retirement,
        ImageStageRetirement::Closed | ImageStageRetirement::SharedWeightsClosed { .. }
    )));
    let native = pipeline.runtime.native_memory_snapshot().unwrap();
    assert!(native.observed_peak_allocated_bytes > 0);
    assert!(native.observed_peak_allocated_bytes >= native.allocated_bytes);
    eprintln!(
        "Metal {policy:?} job={job}, steps={}, native={native:?}, stage retirements={:?}",
        steps.len(),
        stages
            .iter()
            .map(|s| (
                s.execution.plan.role,
                s.execution.completed_executions,
                &s.retirement
            ))
            .collect::<Vec<_>>()
    );
    ResultImage {
        latent: sample,
        rgba,
    }
}

fn compare(actual: &ManagedHostBytes, expected: &ManagedHostBytes) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual
        .bytes()
        .chunks_exact(2)
        .zip(expected.bytes().chunks_exact(2))
    {
        let a = bf16::from_bits(u16::from_le_bytes(a.try_into().unwrap())).to_f32();
        let b = bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32();
        assert!(a.is_finite() && b.is_finite());
    }
    // This tiny fixture has zero Q/K scores and fits one key tile. Like the
    // existing Reference prefix regression, require exact BF16 output, not a
    // widened whole-model tolerance or a trained-image quality claim.
    assert_eq!(actual.bytes(), expected.bytes());
}

#[test]
fn metal_request_cache_matches_recompute_and_reinitializes_different_references() {
    let _guard = METAL_FACTORY_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let (mut assets, directory) = assets();
    let dit = ExpectedWeights::from_config(&assets.transformer, StructureLimits::default())
        .unwrap()
        .weight_schema()
        .unwrap();
    assets.sources.transformer = nonzero_archive(&directory.path().join("prefix-dit"), &dit, true);
    let vae = QwenImage21VaeDecoderFamily::new()
        .unwrap()
        .weights(&assets.vae)
        .unwrap();
    assets.sources.vae = nonzero_archive(
        &directory.path().join("prefix-vae"),
        vae.stored_schema().unwrap(),
        false,
    );
    let mut config = metal_config();
    config.pipeline.edit_prefix_policy = ImageEditPrefixPolicy::RequestCache;
    let mut pipeline = config.create_pipeline(assets).unwrap();
    let host = pipeline.host_account().clone();
    let runtime = Arc::downgrade(&pipeline.runtime);
    let ceiling = pipeline.config.device_capacity_bytes;
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), ceiling).unwrap();
    let a = execute(&mut pipeline, ImageEditPrefixPolicy::Recompute, 1, 1);
    let b = execute(&mut pipeline, ImageEditPrefixPolicy::Recompute, 7, 2);
    assert_ne!(
        a.latent.bytes().bytes(),
        b.latent.bytes().bytes(),
        "fixture must observe reference content"
    );
    for (offset, expected, job) in [(1, &a, 3), (7, &b, 4), (1, &a, 5)] {
        let cached = execute(
            &mut pipeline,
            ImageEditPrefixPolicy::RequestCache,
            offset,
            job,
        );
        compare(cached.latent.bytes(), expected.latent.bytes());
        compare(cached.rgba.bytes(), expected.rgba.bytes());
    }
    pipeline.shutdown().unwrap();
    drop(pipeline);
    assert!(runtime.upgrade().is_none());
    assert!(host.usage().unwrap().committed_bytes() > 0);
    drop((a, b));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let all_capacity = capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, ceiling)
        .unwrap();
    assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
    drop(all_capacity);
}
