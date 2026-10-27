use super::*;
use crate::commands::image_product::directory;

pub(super) fn metadata_fixture(root: &std::path::Path) {
    let write = |name: &str, value: Value| {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    };
    write(
        "model_index.json",
        json!({"_class_name":"QwenImage21Pipeline",
        "processor":["transformers","Qwen3VLProcessor"],
        "text_encoder":["transformers","Qwen3VLForConditionalGeneration"],
        "transformer":["diffusers","QwenImage21Transformer2DModel"],
        "vae":["diffusers","AutoencoderKLQwenImage21"],
        "scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"]}),
    );
    write(
        "text_encoder/config.json",
        json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl"}),
    );
    write(
        "transformer/config.json",
        json!({"_class_name":"QwenImage21Transformer2DModel"}),
    );
    write(
        "vae/config.json",
        json!({"_class_name":"AutoencoderKLQwenImage21"}),
    );
    write(
        "scheduler/scheduler_config.json",
        json!({"_class_name":"FlowMatchEulerDiscreteScheduler"}),
    );
    write(
        "processor/processor_config.json",
        json!({"processor_class":"Qwen3VLProcessor"}),
    );
    write(
        "processor/tokenizer_config.json",
        json!({"processor_class":"Qwen3VLProcessor","tokenizer_class":"Qwen2Tokenizer"}),
    );
    let vocab: tokenizers::models::bpe::Vocab = [("a".to_owned(), 0)].into_iter().collect();
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    fs::write(
        root.join("processor/tokenizer.json"),
        tokenizer.to_string(false).unwrap(),
    )
    .unwrap();
    fs::write(root.join("processor/chat_template.jinja"), "{{ messages }}").unwrap();
}
#[test]
fn explicit_image_precision_routes_both_entrypoints_and_rejects_text_or_config_mixing() {
    use ferrum_engine::image_generation::ImageCudaPrecisionPolicy;
    let root = tempfile::tempdir().unwrap();
    metadata_fixture(root.path());
    let model = root.path().to_str().unwrap();
    assert!(validate(&[
        "ferrum",
        "run",
        model,
        "--prompt",
        "cat",
        "--output",
        "picture.png",
        "--image-precision",
        "fp8_linear_experimental_v1",
        "--backend",
        "cuda"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "serve",
        model,
        "--image-precision",
        "fp8_linear_experimental_v1",
        "--backend",
        "cuda"
    ])
    .is_ok());
    assert!(validate(&["ferrum", "serve", model, "--image-precision", "omni"]).is_err());
    assert!(validate(&["ferrum", "serve", "text-model", "--image-precision", "bf16"]).is_err());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "config.json",
        "--image-precision",
        "bf16"
    ])
    .is_err());
    let args = ImageProductArgs {
        image_config: None,
        image_cudnn_library: None,
        image_precision: Some(ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1),
    };
    let ImageProductSource::LocalDirectory { options, .. } = args
        .source(Some(model), 0.9, None, Some(2), "cuda", None)
        .unwrap()
        .unwrap()
    else {
        panic!("directory");
    };
    assert_eq!(
        options.precision,
        ImageCudaPrecisionPolicy::Fp8LinearExperimentalV1
    );
    assert_eq!(options.preset_name(), "fp8_linear_adaptive_v1");
    assert!(options.host().weight_backing_bytes > 0);
    assert_eq!(
        options.pipeline().residency,
        ferrum_engine::image_generation::ImageResidencyPolicy::Adaptive
    );
    assert_eq!(
        options.pipeline().dynamic_residency,
        ferrum_engine::image_generation::ImageDynamicResidencyPolicy::Adaptive
    );
}

#[test]
fn local_pipeline_routes_both_commands_without_names_or_weight_payloads() {
    let root = tempfile::tempdir().unwrap();
    metadata_fixture(root.path());
    let model = root.path().to_str().unwrap();
    assert_eq!(
        directory::discover(Some(model)).unwrap().unwrap(),
        fs::canonicalize(root.path()).unwrap()
    );
    assert!(validate(&[
        "ferrum",
        "run",
        model,
        "--prompt",
        "cat",
        "--output",
        "picture.png"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "serve",
        model,
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.95"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--model",
        model,
        "--image-cudnn-library",
        "/explicit/libcudnn.so.9",
        "--backend",
        "cuda"
    ])
    .is_ok());
    let args = ImageProductArgs {
        image_config: None,
        image_cudnn_library: None,
        image_precision: None,
    };
    let selected = args
        .source(Some(model), 0.9, None, Some(2), "cuda", Some("1"))
        .unwrap()
        .unwrap();
    let ImageProductSource::LocalDirectory { path, options } = selected else {
        panic!("local route expected");
    };
    assert_eq!(path, fs::canonicalize(root.path()).unwrap());
    assert_eq!(options.ordinal, 1);
    assert_eq!(options.maximum_batch_size, 2);
    assert_eq!(
        options.pipeline().maximum_batch_size,
        options.serving().unwrap().max_num_seqs
    );
    // Identification is metadata-only; executable asset preparation still
    // rejects these deliberately incomplete numerical configs and no weights.
    assert!(
        ferrum_engine::image_generation::QwenImage21Assets::from_directory(
            &path,
            ferrum_models::pipeline_source::PipelineSourceLimits::default()
        )
        .is_err()
    );
}

#[test]
fn incomplete_or_unknown_pipeline_never_falls_back_to_text() {
    let root = tempfile::tempdir().unwrap();
    metadata_fixture(root.path());
    let model = root.path().to_str().unwrap();
    fs::remove_file(root.path().join("transformer/config.json")).unwrap();
    let error = directory::discover(Some(model)).unwrap_err();
    assert!(format!("{error:#}").contains("transformer/config.json"));
    assert!(validate(&["ferrum", "serve", model]).is_err());
    fs::write(
        root.path().join("model_index.json"),
        br#"{"_class_name":"UnknownPipeline"}"#,
    )
    .unwrap();
    assert!(validate(&["ferrum", "run", model, "--prompt", "cat", "--output", "x.png"]).is_err());
    fs::remove_file(root.path().join("model_index.json")).unwrap();
    fs::write(
        root.path().join("config.json"),
        br#"{"model_type":"qwen3"}"#,
    )
    .unwrap();
    assert!(directory::discover(Some(model)).unwrap().is_none());
    assert!(validate(&[
        "ferrum",
        "run",
        model,
        "--prompt",
        "hello",
        "--max-tokens",
        "4"
    ])
    .is_ok());
    assert!(validate(&["ferrum", "serve", model, "--max-model-len", "1024"]).is_ok());
    assert!(validate(&["ferrum", "serve", model, "--image-cudnn-library", "lib.so"]).is_err());
}

#[test]
fn automatic_image_route_rejects_text_controls_and_invalid_product_limits() {
    let root = tempfile::tempdir().unwrap();
    metadata_fixture(root.path());
    let model = root.path().to_str().unwrap();
    for extra in [
        vec!["--max-tokens", "4096"],
        vec!["--backend", "cpu"],
        vec!["--gpu-devices", "0,1"],
        vec!["--max-num-seqs", "0"],
        vec!["--gpu-memory-utilization", "0"],
        vec!["--gpu-memory-utilization", "NaN"],
        vec!["--gpu-memory-utilization", "1.01"],
        vec!["--runtime-memory-budget-bytes", "0"],
        vec!["--width", "0"],
        vec!["--height", "2049"],
    ] {
        let mut arguments = vec![
            "ferrum", "run", model, "--prompt", "cat", "--output", "x.png",
        ];
        arguments.extend(extra);
        assert!(validate(&arguments).is_err(), "{arguments:?}");
    }
    assert!(validate(&["ferrum", "run", model, "--prompt", "cat"]).is_err());
    assert!(validate(&["ferrum", "run", model, "--output", "x.png"]).is_err());
    assert!(validate(&[
        "ferrum",
        "serve",
        model,
        "--image-config",
        "diagnostic.json"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "serve",
        model,
        "--backend",
        "cuda",
        "--runtime-memory-budget-bytes",
        "32212254720"
    ])
    .is_ok());
}

#[test]
fn adaptive_preset_has_one_real_batch_limit_and_finite_host_bounds() {
    let one = DirectoryOptions::from_cli(0.9, None, None, "cuda", None, None).unwrap();
    assert_eq!(directory::PRESET_NAME, "bf16_adaptive_v1");
    assert_eq!(one.serving().unwrap().request_batch_max_wait_ms, 0);
    assert_eq!(one.batching, ferrum_types::ImageBatchPolicy::Auto);
    assert_eq!(
        one.maximum_batch_size,
        ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH
    );
    assert_eq!(
        one.serving().unwrap().maximum_pending_bytes.unwrap().get(),
        one.host().input_preprocess_bytes / 2
    );
    assert_eq!(
        (one.pipeline().maximum_width, one.pipeline().maximum_height),
        (2048, 2048)
    );
    one.host().limits().unwrap();
    #[cfg(feature = "cuda")]
    {
        // Resolving a typed preset must not load a native library.
        let directory = tempfile::tempdir().unwrap();
        let library = directory.path().join("libcudnn.so.9");
        fs::write(&library, b"metadata-only fixture").unwrap();
        let mut selected = one.clone();
        selected.cudnn_library = Some(library);
        assert_eq!(
            selected.cuda().unwrap().attention_policy,
            ferrum_engine::image_generation::ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1
        );
    }
    for backend in ["cuda", "metal"] {
        let options = DirectoryOptions::from_cli(0.9, None, None, backend, None, None).unwrap();
        let pipeline = options.pipeline();
        assert_eq!(
            pipeline.edit_prefix_policy,
            if backend == "cuda" {
                ferrum_engine::image_generation::ImageEditPrefixPolicy::RequestCache
            } else {
                ferrum_engine::image_generation::ImageEditPrefixPolicy::Recompute
            }
        );
        assert_eq!(
            pipeline.residency,
            ferrum_engine::image_generation::ImageResidencyPolicy::Adaptive
        );
        assert_eq!(
            pipeline.dynamic_residency,
            ferrum_engine::image_generation::ImageDynamicResidencyPolicy::Adaptive
        );
    }
    let eight = DirectoryOptions::from_cli(0.9, None, Some(8), "cuda", Some("0"), None).unwrap();
    assert_eq!(eight.serving().unwrap().max_num_seqs, 8);
    assert_eq!(eight.pipeline().maximum_batch_size, 8);
    assert_eq!(eight.serving().unwrap().request_batch_max_wait_ms, 0);
    assert_eq!(
        eight.serving().unwrap().batching,
        ferrum_types::ImageBatchPolicy::Auto
    );
    assert!(eight.maximum_memory_bytes.is_none());
    assert!(DirectoryOptions::from_cli(
        1.0,
        None,
        Some(ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1),
        "auto",
        None,
        None
    )
    .is_err());
}

#[test]
fn run_has_single_request_capacity_and_metal_keeps_configured_batching() {
    for explicit_cap in [None, Some(3)] {
        let cuda = DirectoryOptions::from_cli(0.9, None, explicit_cap, "cuda", None, None).unwrap();
        let run = cuda.single_request();
        assert_eq!(run.pipeline().maximum_batch_size, 1);
        assert_eq!(run.pipeline().maximum_width, 2048);
        assert_eq!(
            run.pipeline().edit_prefix_policy,
            ferrum_engine::image_generation::ImageEditPrefixPolicy::RequestCache
        );
        assert_eq!(run.serving().unwrap().request_batch_max_wait_ms, 0);
        let metal =
            DirectoryOptions::from_cli(0.9, None, explicit_cap, "metal", None, None).unwrap();
        assert_eq!(metal.batching, ferrum_types::ImageBatchPolicy::Configured);
        assert_eq!(metal.pipeline().maximum_width, 1024);
        assert_eq!(
            metal.pipeline().edit_prefix_policy,
            ferrum_engine::image_generation::ImageEditPrefixPolicy::Recompute
        );
        assert_eq!(metal.maximum_batch_size, explicit_cap.unwrap_or(1));
        assert_eq!(
            metal.serving().unwrap().request_batch_max_wait_ms,
            if explicit_cap.is_some() { 200 } else { 0 }
        );
    }
}

#[test]
fn cuda_directory_run_admits_2k_geometry_without_changing_metal_bounds() {
    let root = tempfile::tempdir().unwrap();
    metadata_fixture(root.path());
    let model = root.path().to_str().unwrap();
    for backend in ["cuda", "metal"] {
        let result = validate(&[
            "ferrum",
            "run",
            model,
            "--prompt",
            "cat",
            "--output",
            "x.png",
            "--backend",
            backend,
            "--width",
            "2048",
            "--height",
            "2048",
        ]);
        assert_eq!(result.is_ok(), backend == "cuda");
    }
}

#[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
#[tokio::test]
async fn both_product_wrappers_route_local_pipeline_to_explicit_backend_error() {
    let root = tempfile::tempdir().unwrap();
    metadata_fixture(root.path());
    let model = root.path().to_str().unwrap();
    let cli = TestCli::try_parse_from([
        "ferrum",
        "run",
        model,
        "--prompt",
        "cat",
        "--output",
        "never-written.png",
    ])
    .unwrap();
    let TestCommand::Run(command) = cli.command else {
        panic!("run command expected");
    };
    let result =
        crate::commands::run::execute_cli(command, crate::config::CliConfig::default()).await;
    assert!(matches!(
        result,
        Err(CommandError::Image(ImageProductError::Unsupported(_)))
    ));
    let cli = TestCli::try_parse_from(["ferrum", "serve", model]).unwrap();
    let TestCommand::Serve(command) = cli.command else {
        panic!("serve command expected");
    };
    let result =
        crate::commands::serve::execute_cli(command, crate::config::CliConfig::default(), false)
            .await;
    assert!(matches!(
        result,
        Err(CommandError::Image(ImageProductError::Unsupported(_)))
    ));
}
