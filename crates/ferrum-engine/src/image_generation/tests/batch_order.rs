use super::*;
use crate::image_generation::prepare::BatchOrder;
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
use ferrum_models::qwen_image21::{IMAGE_PROJECTION_INPUT_ID, IMAGE_PROJECTION_OUTPUT_ID};
use half::bf16;

fn request_with(prompt: &str, seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: prompt.into(),
        seed,
        width: 32,
        height: 32,
        n: 1,
    }
}

fn compiled_stages(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    prompts: &[&str],
) -> [PreparedStage<ReferenceDeviceRuntime>; 2] {
    let (prompts, _) = pipeline
        .assets
        .tokenizer
        .prepare_canonical(prompts, 512)
        .unwrap();
    let text =
        Qwen3VLTextConfig::new(pipeline.assets.text.clone(), prompts.layout().clone()).unwrap();
    let dit = QwenImage21TransformerConfig::new(
        pipeline.assets.transformer.clone(),
        QwenImage21T2iLayout::new(
            prompts.feature_sequence_length() as u32,
            2,
            2,
            prompts
                .valid_feature_lengths()
                .into_iter()
                .map(|n| n as u32)
                .collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let text_family = TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&text).unwrap(),
            &pipeline.profiles.text_encoder,
        )
        .unwrap();
    let dit_family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&dit).unwrap(),
            &pipeline.profiles.denoiser,
        )
        .unwrap();
    [
        PreparedStage::compile(
            ImageStage::TextEncoder,
            prompts.batch(),
            text_family,
            &text.compile_options().unwrap(),
            pipeline,
        )
        .unwrap(),
        PreparedStage::compile(
            ImageStage::Denoiser,
            prompts.batch(),
            dit_family,
            &dit.compile_options().unwrap(),
            pipeline,
        )
        .unwrap(),
    ]
}

#[test]
fn canonical_request_multiset_hits_exact_text_and_dit_plans_but_changed_mask_misses() {
    let (pipeline, _directory) = super::adapter::pipeline(3);
    let first = compiled_stages(&pipeline, &["blue cat blue", "cat", "blue cat"]);
    let reordered = compiled_stages(&pipeline, &["blue cat", "blue cat blue", "cat"]);
    for (initial, repeated) in first.iter().zip(&reordered) {
        assert!(!initial.plan.plan_cache_hit);
        assert!(repeated.plan.plan_cache_hit);
        assert!(Arc::ptr_eq(&initial.component, &repeated.component));
        assert_eq!(initial.plan.plan_hash, repeated.plan.plan_hash);
    }
    // The maximum padded sequence and every input extent stay unchanged;
    // changing one valid length must still change both exact mask plans.
    let changed = compiled_stages(&pipeline, &["blue cat blue", "cat", "blue"]);
    for (initial, changed) in first.iter().zip(&changed) {
        assert!(!changed.plan.plan_cache_hit);
        assert_ne!(
            initial.plan.family_fingerprint,
            changed.plan.family_fingerprint
        );
        assert_ne!(initial.plan.plan_hash, changed.plan.plan_hash);
    }
    assert_eq!(pipeline.runtime.snapshot().allocations, 0);
    assert_eq!(pipeline.runtime.snapshot().submissions, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn canonical_request_order_keeps_prompt_seed_pairs_and_restores_real_managed_outputs() {
    // Exercise mapping through a real nonzero Reference projection. This is
    // not a substitute normal provider or a claimed complete image pipeline.
    let (mut pipeline, _directory) = super::residency::numeric_pipeline();
    pipeline.config.residency = ImageResidencyPolicy::Cold;
    let requests = [request_with("blue cat", 42), request_with("cat", 43)];
    let (prompts, indices) = pipeline
        .assets
        .tokenizer
        .prepare_canonical(&[&requests[0].prompt, &requests[1].prompt], 512)
        .unwrap();
    assert_eq!(indices, [1, 0]);
    let order = BatchOrder::new(indices, 2).unwrap();
    let ordered: Vec<_> = order.requests(&requests).unwrap().collect();
    assert_eq!(
        ordered
            .iter()
            .map(|r| r.prompt.as_str())
            .collect::<Vec<_>>(),
        ["cat", "blue cat"]
    );
    assert_eq!(prompts.valid_input_lengths(), [15, 16]);
    let tensor = ResolvedTensorSpec::new(
        vec![2, 64],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut backing = pipeline
        .host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 256)
        .unwrap();
    for (execution_index, request) in ordered.into_iter().enumerate() {
        // Use the same fresh-seed builder as generation. Sorting does not
        // merge independent streams or substitute a batch seed.
        let state = noise_state(&pipeline.host, request.seed).unwrap();
        let seed = u64::from_le_bytes(state.bytes().bytes()[..8].try_into().unwrap());
        assert_eq!(seed, [43, 42][execution_index]);
        assert!(state.bytes().bytes()[8..].iter().all(|&byte| byte == 0));
        let values = [
            seed as f32,
            prompts.valid_input_lengths()[execution_index] as f32,
        ];
        for (channel, value) in values.into_iter().enumerate() {
            let start = (execution_index * 64 + channel) * 2;
            backing.as_mut()[start..start + 2]
                .copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    let input = ManagedHostTensor::new(
        id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
        tensor,
        backing.freeze(),
    )
    .unwrap();
    let stage = super::residency::prepared(&pipeline, 2);
    let mut stages = Vec::new();
    let active = pipeline
        .start_stage(stage, 0, &mut stages)
        .unwrap_or_else(|_| panic!("projection init"));
    let (active, outputs) = active
        .execute(0, vec![input])
        .unwrap_or_else(|_| panic!("projection execute"));
    let output = only_output(outputs, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
    let execution_outputs = vec![output.batch_item(0).unwrap(), output.batch_item(1).unwrap()];
    let original_regions = [
        execution_outputs[1].bytes().clone(),
        execution_outputs[0].bytes().clone(),
    ];
    drop(output);
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("projection close"));
    let restored = order.restore_outputs(execution_outputs).unwrap();
    for (index, output) in restored.iter().enumerate() {
        let actual: Vec<_> = output
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|p| bf16::from_bits(u16::from_le_bytes([p[0], p[1]])).to_f32())
            .collect();
        assert_eq!(&actual[..2], &[[42.0, 16.0], [43.0, 15.0]][index]);
        assert!(actual[2..].iter().all(|&value| value == 0.0));
        assert!(output.bytes().same_region(&original_regions[index]));
    }
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    let host = pipeline.host.clone();
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    let survivor = restored[0].bytes().clone();
    drop((restored, original_regions, pipeline));
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    assert_eq!(
        &survivor.bytes()[..2],
        &bf16::from_f32(42.0).to_bits().to_le_bytes()
    );
    drop(survivor);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn invalid_batch_permutations_fail_before_owners_or_execution() {
    for indices in [vec![], vec![0], vec![0, 0], vec![0, 2]] {
        assert!(BatchOrder::new(indices, 2).is_err());
    }
    assert!(BatchOrder::new(Vec::new(), 0).is_err());
    let order = BatchOrder::new(vec![1, 0], 2).unwrap();
    assert!(order.requests(&[request_with("cat", 42)]).is_err());
    assert!(order.restore_outputs(Vec::new()).is_err());
}
