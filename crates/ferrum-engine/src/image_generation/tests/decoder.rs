use super::*;
use crate::image_generation::decoder::decoder_inputs;
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
use half::bf16;

fn host(output_bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1024 * 1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64 * 1024,
                output_bytes,
                transfer_staging_bytes: 64 * 1024,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn input(host: &HostMemoryAccount) -> ManagedHostTensor {
    let shape = QwenImage21LatentShape::new(2, 2, 2).unwrap();
    let mut backing = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, shape.byte_len())
        .unwrap();
    let midpoint = backing.as_mut().len() / 2;
    for (index, bytes) in backing.as_mut().chunks_exact_mut(2).enumerate() {
        bytes.copy_from_slice(
            &(if index * 2 < midpoint {
                0x3f80_u16
            } else {
                0xbf80
            })
            .to_le_bytes(),
        );
    }
    ManagedHostTensor::new(
        id(QwenImage21VaeDecoderFamily::INPUT_ID).unwrap(),
        shape.tensor(QwenImage21LatentLayout::Nchw).unwrap(),
        backing.freeze(),
    )
    .unwrap()
}

#[test]
fn decoder_views_preserve_sample_order_exact_geometry_and_one_parent_claim() {
    let host = host(65536);
    let batch = input(&host);
    let bytes = batch.bytes().len();
    let original = batch.bytes().clone();
    let expected = fixture::decoder().input_tensor().unwrap();
    let views = decoder_inputs(&host, batch.clone(), &expected, 2).unwrap();
    assert_eq!(views.len(), 2);
    for (index, view) in views.iter().enumerate() {
        assert_eq!(view.tensor(), &expected);
        assert!(view.bytes().same_region(
            &original
                .slice(index * bytes / 2..(index + 1) * bytes / 2)
                .unwrap()
        ));
        assert_eq!(
            &view.bytes().bytes()[..2],
            &(if index == 0 { 0x3f80_u16 } else { 0xbf80 }).to_le_bytes()
        );
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), bytes);
    let foreign = self::host(65536);
    assert!(decoder_inputs(&foreign, batch.clone(), &expected, 2).is_err());
    assert!(decoder_inputs(&host, batch.clone(), &expected, 1).is_err());
    let wrong = ResolvedTensorSpec::new(
        vec![1, 64, 1, 4],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(decoder_inputs(&host, batch.clone(), &wrong, 2).is_err());
    let full = decoder_inputs(&host, batch.clone(), batch.tensor(), 2).unwrap();
    assert_eq!(full.len(), 1);
    assert!(full[0].bytes().same_region(batch.bytes()));
    drop((full, batch, original));
    assert_eq!(host.usage().unwrap().committed_bytes(), bytes);
    drop(views);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

/// Actual complete VAE graph and real BF16 weights. Residual branches are zero,
/// shortcuts carry nonzero positive/negative samples through all up blocks,
/// and the final channel biases distinguish RGBA. This is Reference evidence,
/// not a cuDNN batch-geometry equivalence claim.
fn decoder_source(
    path: &std::path::Path,
    config: &QwenImage21VaeDecoderConfig,
) -> Arc<SafetensorsArchive> {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    std::fs::create_dir(path).unwrap();
    let payloads: Vec<_> = config
        .expected_weights()
        .unwrap()
        .into_iter()
        .map(|w| {
            let shape = w.dimensions();
            let name = w.external_name();
            let mut values = vec![0u16; w.bytes() as usize / 2];
            if name.ends_with(".gamma") {
                values.fill(0x3f80);
            }
            if name == "post_quant_conv.weight" || name.ends_with("conv_shortcut.weight") {
                for channel in 0..shape[0].min(shape[1]) as usize {
                    values[channel * shape[1] as usize + channel] = 0x3f80;
                }
            }
            if name == "decoder.conv_in.weight" {
                // Real stored OIHW, with the center of its 3x3 spatial kernel.
                let area = values.len() / (shape[0] * shape[1]) as usize;
                for channel in 0..shape[0].min(shape[1]) as usize {
                    values[(channel * shape[1] as usize + channel) * area + area / 2] = 0x3f80;
                }
            }
            if name == "decoder.conv_out.weight" {
                let area = values.len() / (shape[0] * shape[1]) as usize;
                for channel in 0..4 {
                    values[channel * shape[1] as usize * area + area / 2] = 0x3f80;
                }
            }
            if name == "decoder.conv_out.bias" {
                values.copy_from_slice(&[0xc000, 0xbf00, 0x3f00, 0x4000]);
            }
            (
                name.to_owned(),
                shape.iter().map(|&n| n as usize).collect::<Vec<_>>(),
                values
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    serialize_to_file(
        payloads.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            )
        }),
        &None,
        &path.join("model.safetensors"),
    )
    .unwrap();
    Arc::new(SafetensorsArchive::open(path).unwrap())
}

fn stage(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    batch: usize,
) -> (
    PreparedStage<ReferenceDeviceRuntime>,
    QwenImage21VaeDecoderConfig,
) {
    let decoder = pipeline.assets.decoder_batch(batch, 2, 2).unwrap();
    let family = TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&decoder).unwrap(),
            &pipeline.profiles.decoder,
        )
        .unwrap();
    let stage = PreparedStage::compile(
        ImageStage::Decoder,
        batch,
        family,
        &decoder.compile_options().unwrap(),
        pipeline,
    )
    .unwrap();
    (stage, decoder)
}

#[test]
fn real_reference_decoder_microbatch_reuses_initialization_and_matches_full_batch() {
    let (mut pipeline, directory) = super::adapter::pipeline(2);
    pipeline.host = host(65536);
    pipeline.config.static_initialization = StaticInitializationPolicy::new(65536, 64).unwrap();
    let config = pipeline.assets.decoder_batch(1, 2, 2).unwrap();
    pipeline.assets.sources.vae = decoder_source(&directory.path().join("numeric-vae"), &config);
    let weights = config.expected_weights().unwrap();
    let weight_bytes: u64 = weights.iter().map(|w| w.bytes()).sum();
    let source = input(&pipeline.host);
    let mut results = Vec::new();
    for (job, policy, batch, executions) in [
        (0, ImageDecoderBatchPolicy::SingleImage, 1, 2),
        (1, ImageDecoderBatchPolicy::FullBatch, 2, 1),
    ] {
        pipeline.config.decoder_batch_policy = policy;
        let (stage, decoder) = stage(&pipeline, batch);
        let mut receipts = Vec::new();
        let images = pipeline
            .decode_batch(stage, &decoder, source.clone(), 2, job, &mut receipts)
            .unwrap_or_else(|_| panic!("real Reference decoder must execute and close"));
        assert_eq!(images.len(), 2);
        assert_eq!(receipts.len(), 1);
        let receipt = &receipts[0].execution;
        assert_eq!(receipt.plan.batch_size, batch);
        assert_eq!(receipt.completed_executions, executions);
        let ComponentInitializationReceipt::Initialized { receipt } = &receipt.initialization
        else {
            panic!("real decoder has stored weights")
        };
        assert_eq!(receipt.uploaded_component_count(), weights.len());
        assert_eq!(receipt.uploaded_bytes(), weight_bytes); // Once, not per execution.
        assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
        for output in &images {
            assert_eq!(output.tensor().dimensions(), [1, 4, 32, 32]);
            QwenImage21RgbaOutput::new(&pipeline.host, output.tensor(), output.bytes()).unwrap();
        }
        assert_ne!(images[0].bytes().bytes(), images[1].bytes().bytes());
        results.push(images);
    }
    // Declared tolerance: zero BF16 bit differences for this Reference fixture.
    // CUDA algorithms can differ with B; this does not qualify their parity.
    for index in 0..2 {
        assert_eq!(
            results[0][index].bytes().bytes(),
            results[1][index].bytes().bytes()
        );
    }
    let micro_alias = results[0][1].bytes().clone();
    let full_alias = results[1][1].bytes().clone();
    let host = pipeline.host.clone();
    drop((source, results, pipeline));
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        micro_alias.len() + 2 * full_alias.len()
    );
    assert!(micro_alias
        .bytes()
        .chunks_exact(2)
        .all(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite()));
    drop(full_alias);
    assert_eq!(host.usage().unwrap().committed_bytes(), micro_alias.len());
    drop(micro_alias);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn later_decoder_output_quota_failure_retains_the_active_session_for_close() {
    let (mut pipeline, _directory) = super::adapter::pipeline(2);
    pipeline.host = host(4 * 32 * 32 * 2); // Exactly one returned image.
    pipeline.config.static_initialization = StaticInitializationPolicy::new(65536, 64).unwrap();
    let (stage, decoder) = stage(&pipeline, 1);
    let mut receipts = Vec::new();
    let failure = pipeline
        .decode_batch(stage, &decoder, input(&pipeline.host), 2, 0, &mut receipts)
        .err()
        .unwrap();
    let ImageGenerationFailure::Execution { source, active } = failure else {
        panic!("second output admission must keep the real execution/session error")
    };
    assert_eq!(active.receipt.completed_executions, 1);
    assert!(receipts.is_empty());
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    drop(source);
    active
        .close()
        .unwrap_or_else(|_| panic!("quiescent quota rejection must permit explicit close"));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cached_decoder_plan_reinitializes_without_retaining_device_or_host_payloads() {
    let (mut pipeline, directory) = super::adapter::pipeline(2);
    pipeline.host = host(65536);
    pipeline.config.static_initialization = StaticInitializationPolicy::new(65536, 64).unwrap();
    let config = pipeline.assets.decoder_batch(1, 2, 2).unwrap();
    pipeline.assets.sources.vae =
        decoder_source(&directory.path().join("cached-numeric-vae"), &config);
    let source = input(&pipeline.host);
    let mut previous = None;
    let mut previous_plan = None;
    for job in 0..2 {
        let (prepared, decoder) = stage(&pipeline, 1);
        assert_eq!(prepared.plan.plan_cache_hit, job == 1);
        if let Some(ref previous) = previous_plan {
            assert!(Arc::ptr_eq(previous, &prepared.component));
        }
        previous_plan = Some(prepared.component.clone());
        let mut receipts = Vec::new();
        let images = pipeline
            .decode_batch(prepared, &decoder, source.clone(), 2, job, &mut receipts)
            .unwrap_or_else(|_| {
                panic!("cached component must initialize a new independent session")
            });
        assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
        assert_eq!(receipts[0].execution.completed_executions, 2);
        assert!(!receipts[0].execution.session_reused);
        assert!(receipts[0].close.is_some());
        assert!(matches!(
            receipts[0].retirement,
            ImageStageRetirement::Closed
        ));
        let ComponentInitializationReceipt::Initialized { receipt } =
            &receipts[0].execution.initialization
        else {
            panic!("weights must be reinitialized")
        };
        assert!(receipt.uploaded_bytes() > 0);
        let bytes = images
            .iter()
            .map(|image| image.bytes().bytes().to_vec())
            .collect::<Vec<_>>();
        if let Some(ref previous) = previous {
            assert_eq!(&bytes, previous);
        }
        previous = Some(bytes);
        drop(images);
    }
    drop(source);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

#[test]
fn resident_real_decoder_reuses_one_initialization_and_shutdown_preserves_host_images() {
    let (mut pipeline, directory) = super::adapter::pipeline(2);
    pipeline.host = host(65536);
    pipeline.config.residency = ImageResidencyPolicy::DenoiserAndDecoder;
    pipeline.config.static_initialization = StaticInitializationPolicy::new(65536, 64).unwrap();
    let config = pipeline.assets.decoder_batch(1, 2, 2).unwrap();
    pipeline.assets.sources.vae =
        decoder_source(&directory.path().join("resident-numeric-vae"), &config);
    let source = input(&pipeline.host);
    let mut outputs = Vec::new();
    let mut uploaded = 0;
    for job in 0..2 {
        let (prepared, decoder) = stage(&pipeline, 1);
        let mut receipts = Vec::new();
        let images = pipeline
            .decode_batch(prepared, &decoder, source.clone(), 2, job, &mut receipts)
            .unwrap_or_else(|_| panic!("resident real decoder"));
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].execution.session_reused, job == 1);
        assert_eq!(receipts[0].execution.completed_executions, 2);
        assert!(receipts[0].close.is_none());
        assert!(matches!(
            receipts[0].retirement,
            ImageStageRetirement::Resident { .. }
        ));
        let snapshot = pipeline.runtime.snapshot();
        if job == 1 {
            assert_eq!(
                snapshot.uploaded_bytes - uploaded,
                source.bytes().len() as u64
            );
        }
        uploaded = snapshot.uploaded_bytes;
        assert!(snapshot.live_allocations > 0);
        assert_ne!(images[0].bytes().bytes(), images[1].bytes().bytes());
        outputs.push(images);
    }
    for sample in 0..2 {
        assert_eq!(
            outputs[0][sample].bytes().bytes(),
            outputs[1][sample].bytes().bytes()
        );
    }
    let receipt = pipeline.shutdown().unwrap();
    assert_eq!(receipt.len(), 1);
    assert!(receipt[0].close.is_some());
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    drop(source);
    assert_eq!(
        pipeline.host.usage().unwrap().committed_bytes(),
        4 * 4 * 32 * 32 * 2
    );
    for batch in &outputs {
        for output in batch {
            QwenImage21RgbaOutput::new(&pipeline.host, output.tensor(), output.bytes()).unwrap();
        }
    }
    drop(outputs);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}
