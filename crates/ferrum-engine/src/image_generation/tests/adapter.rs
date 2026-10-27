//! The actual Qwen adapter, real asset/header preparation and Reference
//! contracts. No pretend normal provider or successful synthetic image path.
use super::{fixture::*, *};
use crate::image_generation::service::{
    ImageBatchError, ImageGenerator, ImageService, ImageServiceConfig, ImageServiceErrorKind,
};
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceDeviceRuntime,
};
use ferrum_models::qwen3_vl_text::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use ferrum_models::qwen_image21::ExpectedWeights;
use serde_json::json;

fn write_archive(
    path: &std::path::Path,
    schema: &WeightSchema,
    embedding: bool,
) -> Arc<SafetensorsArchive> {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    std::fs::create_dir(path).unwrap();
    let mut values: Vec<_> = schema
        .components
        .iter()
        .map(|component| {
            assert_eq!(component.external_names.len(), 1);
            (
                component.external_names[0].clone(),
                component
                    .dimensions
                    .iter()
                    .map(|&n| n as usize)
                    .collect::<Vec<_>>(),
                vec![0u8; component.physical_bytes().unwrap() as usize],
            )
        })
        .collect();
    if embedding {
        values.push((
            Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME.into(),
            vec![32, 8],
            vec![0; 32 * 8 * 2],
        ));
    }
    let views = values.iter().map(|(name, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, &None, &path.join("model.safetensors")).unwrap();
    Arc::new(SafetensorsArchive::open(path).unwrap())
}

pub(super) fn tokenizer() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let vocabulary = [
        "[UNK]",
        "system",
        "Comprehend",
        "and",
        "analyze",
        "the",
        "provided",
        "prompt.",
        "user",
        "assistant",
        "cat",
        "blue",
    ]
    .into_iter()
    .enumerate()
    .map(|(i, s)| (s.to_owned(), i as u32))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.with_post_processor(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
    ));
    tokenizer.add_special_tokens(
        &["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
            .map(|s| tokenizers::AddedToken::from(s, true)),
    );
    let config = json!({"tokenizer_class":"Qwen2Tokenizer","processor_class":"Qwen3VLProcessor","bos_token":null,
        "pad_token":"<|endoftext|>","eos_token":"<|im_end|>","add_prefix_space":false,"split_special_tokens":false});
    let template = "{%- for m in messages -%}{{ '<|im_start|>' + m.role + '\\n' }}{%- for c in m.content -%}{{ c.text }}{%- endfor -%}{{ '<|im_end|>\\n' }}{%- endfor -%}";
    (
        tokenizer.to_string(false).unwrap().into_bytes(),
        serde_json::to_vec(&config).unwrap(),
        template.as_bytes().to_vec(),
    )
}

pub(super) fn pipeline(
    maximum_batch_size: usize,
) -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let directory = tempfile::tempdir().unwrap();
    let text_config = Qwen3VLTextConfig::new(
        text(),
        Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]]).unwrap(),
    )
    .unwrap();
    let text_weights = Qwen3VLTextEncoderFamily::new()
        .unwrap()
        .weights(&text_config)
        .unwrap();
    let dit_weights = ExpectedWeights::from_config(&transformer(), StructureLimits::default())
        .unwrap()
        .weight_schema()
        .unwrap();
    let vae_weights = QwenImage21VaeDecoderFamily::new()
        .unwrap()
        .weights(&decoder())
        .unwrap();
    let sources = QwenImage21Sources {
        text_encoder: write_archive(
            &directory.path().join("text"),
            text_weights.stored_schema().unwrap(),
            true,
        ),
        transformer: write_archive(&directory.path().join("dit"), &dit_weights, false),
        vae: write_archive(
            &directory.path().join("vae"),
            vae_weights.stored_schema().unwrap(),
            false,
        ),
    };
    let vae = json!({"_class_name":"AutoencoderKLQwenImage21","is_residual":true,"patch_size":null,
        "dropout":0.0,"out_channels":4,"attn_scales":[],"z_dim":64,"base_dim":1,
        "dim_mult":[1,2,4,8,8],"num_res_blocks":2,"temperal_downsample":[false,true,true,true],
        "latents_mean":vec![0.;64],"latents_std":vec![1.;64]});
    let scheduler = json!({"_class_name":"FlowMatchEulerDiscreteScheduler","_diffusers_version":"fixture",
        "base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,"max_image_seq_len":8192,
        "max_shift":0.9,"num_train_timesteps":1000,"shift":1.0,"shift_terminal":null,
        "stochastic_sampling":false,"time_shift_type":"exponential","use_beta_sigmas":false,
        "use_dynamic_shifting":false,"use_exponential_sigmas":false,"use_karras_sigmas":false});
    let index = json!({"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"sample_sigmas":[1.,0.5]});
    let (tokens, tokenizer_config, template) = tokenizer();
    let assets = QwenImage21Assets::prepare(
        QwenImage21Metadata {
            text_encoder: &serde_json::to_vec(&text_metadata()).unwrap(),
            transformer: &serde_json::to_vec(&transformer_metadata()).unwrap(),
            vae: &serde_json::to_vec(&vae).unwrap(),
            model_index: &serde_json::to_vec(&index).unwrap(),
            scheduler: &serde_json::to_vec(&scheduler).unwrap(),
            tokenizer: &tokens,
            tokenizer_config: &tokenizer_config,
            chat_template: &template,
        },
        sources,
    )
    .unwrap();
    let composition = composition("device.image.adapter.batch");
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                output_bytes: 4096,
                transfer_staging_bytes: 4096,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let config = ImagePipelineConfig {
        device_capacity_bytes: DEVICE_BYTES,
        maximum_batch_size,
        ..config()
    };
    let pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        assets,
        QwenImage21Profiles {
            text_encoder: NumericalProfileId::new(Qwen3VLTextEncoderFamily::REFERENCE_PROFILE_ID)
                .unwrap(),
            noise: NumericalProfileId::new("test.required.real-philox.v1").unwrap(),
            denoiser: NumericalProfileId::new(QwenImage21TransformerFamily::REFERENCE_PROFILE_ID)
                .unwrap(),
            decoder: NumericalProfileId::new(QwenImage21VaeDecoderFamily::REFERENCE_PROFILE_ID)
                .unwrap(),
            euler: Bf16EulerProfile::CudaScalarBf16StagedRneV1,
        },
        config,
    )
    .unwrap();
    (pipeline, directory)
}

fn two_requests() -> Vec<ImageGenerationRequest> {
    ["cat", "blue cat"]
        .into_iter()
        .enumerate()
        .map(|(i, prompt)| ImageGenerationRequest {
            prompt: prompt.into(),
            width: 32,
            height: 32,
            seed: 42 + i as u64,
            n: 1,
        })
        .collect()
}

#[test]
fn real_qwen_adapter_exposes_configured_capacity_and_delegates_b2_preflight() {
    let (mut pipeline, _sources) = pipeline(2);
    assert_eq!(
        ImageGenerator::capabilities(&pipeline)
            .unwrap()
            .maximum_batch_size,
        2
    );
    let before = pipeline.runtime.snapshot();
    let error = match ImageGenerator::generate_batch(&mut pipeline, two_requests()) {
        Ok(_) => panic!("Reference has no real Philox provider"),
        Err(error) => error,
    };
    let ImageBatchError::Rejected(message) = error else {
        panic!("compile-time missing provider must reject only this batch")
    };
    assert!(message.contains("philox"), "{message}");
    assert!(!pipeline.is_stopped());
    let after = pipeline.runtime.snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(
        pipeline.host_account().usage().unwrap().committed_bytes(),
        0
    );
}

#[tokio::test]
async fn real_plan_capacity_error_retains_stage_source_and_run_serve_diagnostics() {
    let (mut pipeline, _sources) = pipeline(1);
    // Force the real text plan to exceed its policy before the deliberately
    // absent Reference Philox provider is reached. No fabricated error is used.
    let available = 1;
    pipeline.config.device_capacity_bytes = available;
    pipeline.policy = TensorComponentPolicy::new(
        available,
        pipeline
            .runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let runtime = pipeline.runtime.clone();
    let host = pipeline.host.clone();
    let before = runtime.snapshot();
    let request = two_requests().remove(0);
    let error = pipeline
        .generate_batch(vec![request.clone()])
        .err()
        .expect("the real text plan must exceed its capacity");
    let ImageGenerationFailure::Preflight(source) = &error.failure else {
        panic!("capacity failure must remain an owner-free preflight rejection")
    };
    let Some(VNextError::PlanCapacityExceeded {
        required_bytes,
        usable_capacity_bytes,
    }) = source.downcast_ref::<VNextError>()
    else {
        panic!("stage context must retain the original typed capacity error")
    };
    assert!(*required_bytes > *usable_capacity_bytes);
    assert_eq!(*usable_capacity_bytes, available);
    let required = *required_bytes;
    let image_capacity = source
        .downcast_ref::<ImageDeviceCapacityExceeded>()
        .expect("fresh-plan capacity must be an explicit image shrink signal");
    assert!(!image_capacity.report.complete_batch);
    assert_eq!(image_capacity.report.required_bytes, required);
    assert_eq!(image_capacity.report.capacity_bytes, available);
    // ImageProductError::Generation uses this Display for `ferrum run`.
    let run_message = error.to_string();
    for detail in [
        "TextEncoder".to_owned(),
        "WholeStage".to_owned(),
        "batch 1".to_owned(),
        format!("requires {required} bytes"),
        format!("usable capacity {available}"),
    ] {
        assert!(run_message.contains(&detail), "{run_message}");
    }
    assert!(error.completed_stages.is_empty());
    assert!(error.completed_steps.is_empty());
    drop(error);
    assert!(!pipeline.is_stopped());

    let service = ImageService::new(
        pipeline,
        ImageServiceConfig {
            batching: ferrum_types::ImageBatchPolicy::Configured,
            maximum_pending: 2,
            maximum_pending_bytes: 1024 * 1024,
            maximum_prompt_bytes: 4096,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    let served = service
        .try_submit(request)
        .unwrap()
        .wait()
        .await
        .err()
        .expect("serving must expose the same real capacity rejection");
    // The existing HTTP handler forwards this message for Generation errors;
    // its status and routing policy remain unchanged.
    assert_eq!(served.kind(), ImageServiceErrorKind::Generation);
    assert_eq!(served.to_string(), run_message);
    assert!(service.is_accepting());
    assert_eq!(service.snapshot().active_requests, 0);
    assert_eq!(service.snapshot().pending_requests, 0);
    let retired = service.shutdown().await.unwrap();
    assert!(!retired.has_failure());
    drop(retired);
    let after = runtime.snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn real_qwen_service_rejects_excess_capacity_and_preserves_default_one() {
    let (mut single, _single_sources) = pipeline(1);
    assert_eq!(
        ImageGenerator::capabilities(&single)
            .unwrap()
            .maximum_batch_size,
        1
    );
    let error = ImageGenerator::generate_batch(&mut single, two_requests())
        .err()
        .unwrap();
    assert!(matches!(&error, ImageBatchError::Rejected(_)));
    assert!(error.to_string().contains("maximum_batch_size"));
    let service_config = ImageServiceConfig {
        batching: ferrum_types::ImageBatchPolicy::Configured,
        maximum_pending: 2,
        maximum_pending_bytes: 1024 * 1024,
        maximum_prompt_bytes: 4096,
        max_num_seqs: 2,
        request_batch_max_wait_ms: 0,
    };
    assert!(ImageService::new(single, service_config).is_err());
    let (batched, _batch_sources) = pipeline(2);
    let service = ImageService::new(batched, service_config).unwrap();
    assert_eq!(service.capabilities().maximum_batch_size, 2);
    assert_eq!(service.snapshot().max_num_seqs, 2);
    assert!(!service.shutdown().await.unwrap().has_failure());
}

#[tokio::test]
async fn real_qwen_token_preflight_rejects_job_without_stopping_service_or_submitting() {
    let (mut pipeline, _sources) = pipeline(1);
    pipeline.config.maximum_user_tokens = 2;
    let runtime = pipeline.runtime.clone();
    let host = pipeline.host.clone();
    let before = runtime.snapshot();
    let service = ImageService::new(
        pipeline,
        ImageServiceConfig {
            batching: ferrum_types::ImageBatchPolicy::Configured,
            maximum_pending: 2,
            maximum_pending_bytes: 1024 * 1024,
            maximum_prompt_bytes: 4096,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    let mut oversized = two_requests().remove(0);
    oversized.prompt = "cat cat cat cat cat cat cat cat".into();
    // Byte/dimension admission accepts it; the real tokenizer then enforces
    // the independent user-token limit during owner-free preparation.
    service.validate_request(&oversized).unwrap();
    let error = service
        .try_submit(oversized)
        .unwrap()
        .wait()
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ImageServiceErrorKind::Generation);
    assert!(error.to_string().contains("user-token budget"));
    assert!(service.is_accepting());

    // A later short prompt reaches plan compilation, beyond token validation.
    // This CPU fixture intentionally lacks Philox, so it cannot claim a
    // successful image. Its separate preflight rejection still keeps serving.
    let error = service
        .try_submit(two_requests().remove(0))
        .unwrap()
        .wait()
        .await
        .err()
        .unwrap();
    assert_eq!(error.kind(), ImageServiceErrorKind::Generation);
    assert!(error.to_string().contains("philox"));
    assert!(service.is_accepting());
    assert_eq!(runtime.snapshot().submissions, before.submissions);
    assert_eq!(runtime.snapshot().allocations, before.allocations);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let retired = service.shutdown().await.unwrap();
    assert!(!retired.has_failure());
    drop(retired);
}
