use super::*;
use crate::qwen3_vl_vision::comfy::tests::{fixture, oracle};
use crate::qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, TensorArchive};
use half::bf16;
use std::sync::Arc;

fn processor(min: u64, max: u64) -> Qwen3VLVisionHfProcessor {
    let raw = serde_json::json!({"processor_class":"Qwen3VLProcessor","image_processor":{"image_processor_type":"Qwen2VLImageProcessor","do_convert_rgb":true,"do_normalize":true,"do_rescale":true,"do_resize":true,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"patch_size":2,"merge_size":2,"temporal_patch_size":2,"rescale_factor":1./255.,"resample":3,"size":{"shortest_edge":min,"longest_edge":max}}});
    Qwen3VLVisionHfProcessor::from_source_bytes(
        &serde_json::to_vec(&raw).unwrap(),
        processor_limits(),
    )
    .unwrap()
}
fn processor_limits() -> Qwen3VLVisionHfProcessorLimits {
    Qwen3VLVisionHfProcessorLimits {
        max_processor_bytes: 16384,
        image: ImageInputLimits::new(1 << 20, 2_000_000, 2048, 2048, 1 << 20).unwrap(),
        max_patch_rows: 500_000,
        max_patch_bytes: 48 << 20,
    }
}
fn rgba(host: &HostMemoryAccount, width: u32, height: u32, constant: bool) -> ManagedRgba8Image {
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            width as usize * height as usize * 4,
        )
        .unwrap();
    for (i, pixel) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
        pixel.copy_from_slice(&if constant {
            [64, 128, 192, 0]
        } else {
            [
                (i % 256) as u8,
                ((i * 13 + 17) % 256) as u8,
                ((i * 7 + 91) % 256) as u8,
                (i % 2) as u8 * 255,
            ]
        });
    }
    ManagedRgba8Image::from_rgba8(
        host,
        bytes.freeze(),
        width,
        height,
        processor_limits().image,
    )
    .unwrap()
}
fn floats(t: &ManagedHostTensor) -> Vec<f32> {
    t.bytes()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn id<T: TryFrom<String>>(name: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(name.into()).unwrap()
}
fn config(p: &Qwen3VLVisionHfProcessor, h: u64, w: u64) -> Qwen3VLVisionHfConfig {
    Qwen3VLVisionHfConfig::new(
        fixture::qwen35_config(h, w).architecture().clone(),
        Qwen3VLVisionGrid::new(1, h, w).unwrap(),
        p,
    )
    .unwrap()
}

#[test]
fn fused_f32_hf_profile_preserves_ports_weights_and_processor_geometry() {
    let p = processor(16, 96);
    let config = config(&p, 4, 6);
    let family = Qwen3VLVisionHfFamily::new(fixture::limits()).unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    let strict = profiles
        .resolve(&id(Qwen3VLVisionHfFamily::CUDA_PROFILE_ID))
        .unwrap();
    let fused = profiles
        .resolve(&id(Qwen3VLVisionHfFamily::CUDA_FUSED_F32_PROFILE_ID))
        .unwrap();
    let strict_graph = family.semantic_program(&config, strict).unwrap();
    let fused_graph = family.semantic_program(&config, fused).unwrap();
    fused.validate_program(&fused_graph).unwrap();
    assert_eq!(strict.boundaries, fused.boundaries);
    assert_eq!(strict_graph.inputs(), fused_graph.inputs());
    assert_eq!(strict_graph.outputs(), fused_graph.outputs());
    assert_eq!(strict_graph.weights(), fused_graph.weights());
    let mut expected = strict_graph.blocks().to_vec();
    let mut found_attention = false;
    for node in expected.iter_mut().flat_map(|block| &mut block.nodes) {
        if node.operation_id.as_str() == STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID {
            found_attention = true;
            node.operation_id = id(STRUCTURED_GQA_ATTENTION_FUSED_F32_OPERATION_ID);
        }
    }
    assert!(found_attention);
    assert_eq!(expected, fused_graph.blocks());
}

#[test]
fn original_processor_policy_is_frozen_and_unsupported_resize_fails_closed() {
    // Exact processor fixture, from HF 7f98acef503cf2bcfba821a78cb1cf1a9928b2df,
    // prompt_upsampler_processor/processor_config.json (Apache-2.0).
    let original = include_bytes!("tests/processor_config.json");
    let p = Qwen3VLVisionHfProcessor::from_source_bytes(original, processor_limits()).unwrap();
    assert_eq!(
        (p.patch_size(), p.temporal_patch_size(), p.merge_size()),
        (16, 2, 2)
    );
    assert_eq!(p.pixel_area_bounds(), (65536, 16777216));
    use sha2::{Digest, Sha256};
    assert_eq!(p.source_sha256(), format!("{:x}", Sha256::digest(original)));
    for field in ["resample", "do_resize", "image_mean", "rescale_factor"] {
        let mut value: serde_json::Value = serde_json::from_slice(original).unwrap();
        value["image_processor"][field] = serde_json::json!(0);
        assert!(Qwen3VLVisionHfProcessor::from_source_bytes(
            &serde_json::to_vec(&value).unwrap(),
            processor_limits()
        )
        .is_err());
    }
    let p = processor(16, 1024);
    let a = fixture::qwen35_config(4, 6).architecture().clone();
    assert!(Qwen3VLVisionGrid::new(2, 4, 6).is_err());
    let wrong = fixture::config(4, 6).architecture().clone();
    assert!(
        Qwen3VLVisionHfConfig::new(wrong, Qwen3VLVisionGrid::new(1, 4, 6).unwrap(), &p).is_err()
    );
    assert!(Qwen3VLVisionHfImagePlan::processor_input(a, 1, 201, &p).is_err());
}

#[test]
fn complete_outer_lanczos_then_inner_cubic_executes_with_exact_peak_and_refunds_failures() {
    let p = processor(64, 192);
    let plan = Qwen3VLVisionHfImagePlan::new(
        fixture::qwen35_config(4, 6).architecture().clone(),
        3,
        2,
        &p,
    )
    .unwrap();
    // sqrt(1_000_000 / 1.5) / 32 = 25.5155..., rounded to 26.
    assert_eq!(plan.outer_dimensions(), (1216, 832));
    assert_eq!(plan.resized_dimensions(), (16, 8));
    assert_eq!(
        plan.preprocessing_profile(),
        QWEN3_VL_HF_IMAGE_PREPROCESS_PROFILE
    );
    let query = fixture::host(64 << 20);
    let original_bytes = query.backing_bytes(3 * 2 * 4).unwrap();
    let peak = plan.input_backing_bytes(&query).unwrap();
    for budget in [peak - 1, peak] {
        let h = fixture::host(original_bytes + budget);
        let original = rgba(&h, 3, 2, true);
        let result = plan.managed_patches(&h, &original);
        if budget < peak {
            assert!(
                result.is_err(),
                "one physical byte below the real simultaneous peak must reject"
            );
            assert_eq!(h.usage().unwrap().committed_bytes(), original_bytes);
        } else {
            let output = result.unwrap();
            for row in floats(&output).chunks_exact(24) {
                for channel in 0..3 {
                    let expected = (f32::from([64u8, 128, 192][channel]) - 127.5) / 127.5;
                    assert!(row[channel * 8..channel * 8 + 8]
                        .iter()
                        .all(|v| v.to_bits() == expected.to_bits()));
                }
            }
            assert_eq!(
                h.usage().unwrap().committed_bytes(),
                original_bytes + plan.retained_input_bytes(&h).unwrap()
            );
            drop(output);
        }
        drop(original);
        assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn hf_patch_rows_preserve_merge_order_repeated_frames_rgb_and_original_owner() {
    let p = processor(16, 1024);
    let plan = Qwen3VLVisionHfImagePlan::processor_input(
        fixture::qwen35_config(4, 6).architecture().clone(),
        12,
        8,
        &p,
    )
    .unwrap();
    let h = fixture::host(1 << 20);
    let image = rgba(&h, 12, 8, false);
    let before = image.bytes().bytes().to_vec();
    let output = plan.managed_patches(&h, &image).unwrap();
    let values = floats(&output);
    let first_patches = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    ];
    for (row, (py, px)) in first_patches.into_iter().enumerate() {
        for channel in 0..3 {
            for t in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        let pixel = ((py * 2 + y) * 12 + px * 2 + x) * 4 + channel;
                        let expected = (f32::from(before[pixel]) - 127.5) / 127.5;
                        assert_eq!(
                            values[row * 24 + (channel * 2 + t) * 4 + y * 2 + x].to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
    }
    assert_eq!(image.bytes().bytes(), before);
    drop(output);
    drop(image);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn hf_positions_retain_f32_products_and_refund_atomic_input_failure() {
    let p = processor(16, 1024);
    let c = config(&p, 4, 6);
    let mut original = fixture::Original::new(&c.base);
    let table = original
        .values
        .get_mut(super::super::Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)
        .unwrap();
    table.fill(0.);
    table[..8].fill(-2.);
    table[8..16].fill(1.);
    let (_dir, archive) = original.archive();
    let table = TensorArchive::tensor(
        &archive,
        super::super::Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME,
    )
    .unwrap();
    let h = fixture::host(1 << 20);
    let positions = c.managed_position_inputs(&h, &table).unwrap();
    let expected = -2f32 * (1f32 - 0.6f32) + 0.6f32;
    assert_eq!(floats(&positions[0])[8].to_bits(), expected.to_bits());
    let comfy = c.base.managed_position_inputs(&h, &table).unwrap();
    assert_ne!(
        floats(&positions[0])[8].to_bits(),
        floats(&comfy[0])[8].to_bits()
    );
    assert_eq!(floats(&positions[1])[4].to_bits(), 1f32.to_bits());
    assert_eq!(floats(&positions[2])[5].to_bits(), libm::sinf(1.).to_bits());
    drop(positions);
    drop(comfy);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
    let denied = fixture::host(c.position_input_backing_bytes(&h).unwrap() - 1);
    assert!(c.managed_position_inputs(&denied, &table).is_err());
    assert_eq!(denied.usage().unwrap().committed_bytes(), 0);
    original
        .values
        .get_mut(super::super::Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)
        .unwrap()[0] = bf16::NAN.to_f32();
    let (_dir, archive) = original.archive();
    let bad = TensorArchive::tensor(
        &archive,
        super::super::Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME,
    )
    .unwrap();
    assert!(c.managed_position_inputs(&h, &bad).is_err());
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}

fn compile(
    c: &Qwen3VLVisionHfConfig,
    r: &ReferenceVNextComposition,
) -> CompiledTensorComponent<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    let f = TypedFamilyRegistration::new(Qwen3VLVisionHfFamily::new(fixture::limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &id(Qwen3VLVisionHfFamily::REFERENCE_PROFILE_ID),
        )
        .unwrap();
    let d = r.runtime().descriptor();
    CompiledTensorComponent::compile(
        f,
        r.registry(),
        r.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        r.weight_materializers(),
        &c.compile_options().unwrap(),
    )
    .unwrap()
}
fn init(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id(format!("run.vision.hf.{name}")),
        request_id: id(format!("request.vision.hf.{name}")),
        transaction_id: id(format!("transaction.vision.hf.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 256).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
#[test]
fn actual_main_only_stored_vision_consumes_hf_inputs_then_rebinds_original_snapshot() {
    let p = processor(16, 1024);
    let c = config(&p, 4, 6);
    let original = fixture::Original::new(&c.base);
    let (_dir, archive) = original.archive();
    let source = Arc::new(ComfyCanonicalWeightSource::new(archive).unwrap());
    let adapter = Qwen3VLVisionHfWeightAdapter::new(&c, source, fixture::limits()).unwrap();
    let h = fixture::host(8 << 20);
    let r = ReferenceVNextComposition::create(id("device.reference.vision.hf.native")).unwrap();
    let compiled = compile(&c, &r);
    assert!(
        adapter
            .maximum_group_backing_bytes(
                compiled.family(),
                compiled.executable().execution_plan(),
                &h
            )
            .unwrap()
            > 0
    );
    let weight_source = adapter.weight_source(&h);
    let (mut session, weights) = compiled
        .initialize_shared_weights(
            r.runtime().clone(),
            h.clone(),
            ComponentWeightSource::Stored(&weight_source),
            init("first"),
        )
        .unwrap();
    for (index, (height, width)) in [(4, 6), (6, 4)].into_iter().enumerate() {
        let c = config(&p, height, width);
        if index != 0 {
            assert!(matches!(
                session
                    .close()
                    .unwrap_or_else(|failure| panic!("close: {:?}", failure.failure())),
                PlanRuntimeCloseOutcome::Closed(_)
            ));
            session = compile(&c, &r)
                .initialize_with_shared_weights(
                    r.runtime().clone(),
                    h.clone(),
                    &weights,
                    init("rebind"),
                )
                .unwrap();
        }
        let image = rgba(&h, width as u32 * 2, height as u32 * 2, false);
        let plan = Qwen3VLVisionHfImagePlan::processor_input(
            c.architecture().clone(),
            image.width(),
            image.height(),
            &p,
        )
        .unwrap();
        let patches = plan.managed_patches(&h, &image).unwrap();
        let pos = c
            .managed_position_inputs(
                &h,
                &TensorArchive::tensor(
                    adapter.source().archive(),
                    super::super::Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME,
                )
                .unwrap(),
            )
            .unwrap();
        let expected = oracle::forward(
            &c.base,
            &floats(&patches),
            &floats(&pos[0]),
            &floats(&pos[1]),
            &floats(&pos[2]),
            &original,
        );
        let mut inputs = vec![patches];
        inputs.extend(pos);
        let outputs = session
            .execute(
                id(format!("run.vision.hf.{index}")),
                id(format!("request.vision.hf.{index}")),
                inputs,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        assert_eq!(output.tensor().dimensions(), [6, 256]);
        let actual = output
            .bytes()
            .bytes()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        for (&actual, &target) in actual
            .iter()
            .zip(&expected[Qwen3VLVisionHfFamily::OUTPUT_ID])
        {
            assert!((actual - target).abs() <= 3e-5, "{actual} vs {target}");
        }
        assert!(h.owns(output.bytes()));
        drop(outputs);
        drop(image);
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|failure| panic!("final close: {:?}", failure.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(matches!(
        weights.close(),
        Ok(SharedComponentWeightsCloseOutcome::Closed(_))
    ));
    assert_eq!(r.runtime().snapshot().live_allocations, 0);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
