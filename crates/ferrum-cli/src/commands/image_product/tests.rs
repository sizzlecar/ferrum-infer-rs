use super::{configuration::*, *};
use clap::{CommandFactory, Parser, Subcommand};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

fn configuration() -> Value {
    json!({
        "model":"local-picture", "execution_profile":"experimental_serial_cold",
        "metadata": {"model_index":"model_index.json", "text_encoder":"text/config.json",
            "transformer":"transformer/config.json", "vae":"vae/config.json",
            "scheduler":"scheduler/config.json", "tokenizer":"tokenizer/tokenizer.json",
            "tokenizer_config":"tokenizer/tokenizer_config.json", "chat_template":"tokenizer/chat_template.jinja"},
        "weights": {"text_encoder":{"kind":"index","directory":"text","file":"weights.index.json"},
            "transformer":{"kind":"index","directory":"transformer","file":"diffusion.index.json"},
            "vae":{"kind":"single","directory":"vae","file":"diffusion.safetensors"}},
        "backend":{"kind":"cuda", "ordinal":0, "usable_memory_bytes":1024,
            "bootstrap_memory_bytes":64,"stream_memory_bytes":64,"maximum_streams":1,
            "maximum_in_flight_per_stream":1,"maximum_commands_per_submission":1024,
            "opaque_memory_profile":"unqualified-test",
            "cudnn":{"library_path":"lib/libcudnn.so.9","opaque_memory_bytes":64,
                "workspace_limit_bytes":64,"opaque_memory_profile":"unqualified-test"},
            "host":{"total_bytes":4096,"input_preprocess_bytes":2048,
                "transfer_staging_bytes":1024,"output_bytes":2048,"weight_backing_bytes":0},
            "pipeline":{"device_capacity_bytes":1024,"maximum_user_tokens":128,
                "maximum_width":512,"maximum_height":512,"maximum_staging_bytes":512,
                "maximum_initialization_commands":32,"maximum_backing_maintenance_passes":8}}
    })
}

#[test]
fn image_configuration_is_explicit_strict_and_resolves_from_its_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    fs::write(&path, serde_json::to_vec(&configuration()).unwrap()).unwrap();
    let config = ImageProductConfig::load(&path).unwrap();
    let base = fs::canonicalize(dir.path()).unwrap();
    assert_eq!(
        config.metadata.transformer,
        base.join("transformer/config.json")
    );
    let ImageBackendConfig::Cuda(cuda) = config.backend;
    assert_eq!(cuda.pipeline.maximum_batch_size, 1);
    assert_eq!(
        cuda.pipeline.resolved().unwrap().decoder_batch_policy,
        ferrum_engine::image_generation::ImageDecoderBatchPolicy::SingleImage
    );
    assert_eq!(cuda.cudnn.library_path, base.join("lib/libcudnn.so.9"));
    assert!(
        matches!(config.weights.vae, WeightSource::Single{directory,file}
        if directory == base.join("vae") && file == "diffusion.safetensors")
    );
    for (key, value) in [
        ("maximum_batch_size", json!(8)),
        ("execution_profile", json!("concurrent")),
    ] {
        let mut invalid = configuration();
        invalid[key] = value;
        fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(ImageProductConfig::load(&path).is_err());
    }
    let mut invalid = configuration();
    invalid["backend"]["kind"] = json!("cpu");
    assert!(serde_json::from_value::<ImageProductConfig>(invalid).is_err());
    let original = serde_json::to_string(&configuration()).unwrap();
    let duplicate = format!("{{\"model\":\"duplicate\",{}", &original[1..]);
    assert!(serde_json::from_str::<ImageProductConfig>(&duplicate).is_err());
    fs::write(&path, b"[]").unwrap();
    assert!(ImageProductConfig::load(&path).is_err());
}

#[test]
fn experimental_batch_profile_connects_service_limit_to_real_pipeline_capacity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image.json");
    let mut wire = configuration();
    wire["serving"] = json!({"maximum_pending":4,"maximum_prompt_bytes":4096,"max_num_seqs":2,
        "request_batch_max_wait_ms":25,"default_width":512,"default_height":512,
        "default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
    let load = |value: &Value| {
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        ImageProductConfig::load(&path)
    };
    assert!(load(&wire).is_err()); // Missing cap remains one, including for serve.
    wire["backend"]["pipeline"]["maximum_batch_size"] = json!(2);
    assert!(load(&wire).is_err()); // Serial is not silently upgraded.
    wire["execution_profile"] = json!("experimental_batched_cold");
    let config = load(&wire).unwrap();
    assert_eq!(
        config.execution_profile.as_str(),
        "experimental_batched_cold"
    );
    let ImageBackendConfig::Cuda(cuda) = &config.backend;
    assert_eq!(cuda.pipeline.resolved().unwrap().maximum_batch_size, 2);
    assert_eq!(config.serving.as_ref().unwrap().max_num_seqs, 2);
    wire["serving"]["max_num_seqs"] = json!(3);
    assert!(load(&wire).is_err());
    wire["serving"]["max_num_seqs"] = json!(2);
    wire["serving"]["maximum_prompt_bytes"] = json!(512 * 1024 + 1);
    assert!(load(&wire).is_err());
    wire["serving"]["maximum_prompt_bytes"] = json!(512 * 1024);
    assert!(load(&wire).is_ok());
    wire["serving"]["maximum_prompt_bytes"] = json!(4096);
    for cap in [0, ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1] {
        wire["backend"]["pipeline"]["maximum_batch_size"] = json!(cap);
        assert!(load(&wire).is_err());
    }
    wire["backend"]["pipeline"]["maximum_batch_size"] =
        json!(ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH);
    assert!(load(&wire).is_ok());
}

#[test]
fn decoder_batch_policy_is_explicit_independent_and_strict() {
    use ferrum_engine::image_generation::ImageDecoderBatchPolicy;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("image.json");
    let mut wire = configuration();
    wire["execution_profile"] = json!("experimental_batched_cold");
    wire["backend"]["pipeline"]["maximum_batch_size"] = json!(8);
    for (name, expected) in [
        ("single_image", ImageDecoderBatchPolicy::SingleImage),
        ("full_batch", ImageDecoderBatchPolicy::FullBatch),
    ] {
        wire["backend"]["pipeline"]["decoder_batch_policy"] = json!(name);
        fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
        let config = ImageProductConfig::load(&path).unwrap();
        let ImageBackendConfig::Cuda(cuda) = config.backend;
        let resolved = cuda.pipeline.resolved().unwrap();
        assert_eq!(resolved.maximum_batch_size, 8);
        assert_eq!(resolved.decoder_batch_policy, expected);
    }
    for invalid in [
        json!(2),
        json!(null),
        json!("automatic"),
        json!({"single_image": {"max": 4}}),
    ] {
        wire["backend"]["pipeline"]["decoder_batch_policy"] = invalid;
        assert!(serde_json::from_value::<ImageProductConfig>(wire.clone()).is_err());
    }
}

#[test]
fn bounded_metadata_and_declared_architecture_fail_before_missing_weight_sources() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.json");
    fs::write(&path, b"{}").unwrap();
    assert_eq!(source::read_bounded(&path, 2).unwrap(), b"{}");
    assert!(source::read_bounded(&path, 1).is_err());
    fs::write(&path, []).unwrap();
    assert!(source::read_bounded(&path, 2).is_err());

    let config_path = dir.path().join("image.json");
    fs::write(&config_path, serde_json::to_vec(&configuration()).unwrap()).unwrap();
    fs::write(
        dir.path().join("model_index.json"),
        br#"{"_class_name":"UnsupportedPipeline"}"#,
    )
    .unwrap();
    let error = ImageProductConfig::load(&config_path)
        .unwrap()
        .prepare_assets()
        .err()
        .unwrap();
    assert!(
        !error.to_string().contains("open "),
        "architecture must fail before opening other assets: {error}"
    );
}

#[test]
fn exact_weight_source_never_falls_back_to_another_filename() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    let dir = tempfile::tempdir().unwrap();
    let data = [0u8, 0];
    serialize_to_file(
        [("w", TensorView::new(Dtype::BF16, vec![1], &data).unwrap())],
        &None,
        &dir.path().join("chosen.safetensors"),
    )
    .unwrap();
    let source = WeightSource::Single {
        directory: dir.path().into(),
        file: "chosen.safetensors".into(),
    };
    assert!(source.open().is_ok());
    for file in [
        "missing.safetensors",
        "../chosen.safetensors",
        "dir/chosen.safetensors",
    ] {
        assert!(WeightSource::Single {
            directory: dir.path().into(),
            file: file.into()
        }
        .open()
        .is_err());
    }
    assert!(WeightSource::Index {
        directory: dir.path().into(),
        file: "missing.index.json".into()
    }
    .open()
    .is_err());
}

#[derive(Parser)]
struct TestCli {
    #[command(subcommand)]
    command: TestCommand,
}
#[derive(Subcommand)]
enum TestCommand {
    Run(crate::commands::run::RunCliCommand),
    Serve(crate::commands::serve::ServeCliCommand),
}
fn validate(arguments: &[&str]) -> anyhow::Result<()> {
    let definition = TestCli::command();
    let matches = definition.clone().try_get_matches_from(arguments)?;
    validate_cli_matches(&definition, &matches)
}

#[test]
fn run_image_controls_reject_explicit_text_defaults_and_keep_text_commands() {
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--prompt",
        "a cat",
        "--output",
        "image.png"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "run",
        "text-model",
        "--prompt",
        "hello",
        "--max-tokens",
        "4096"
    ])
    .is_ok());
    for extra in [
        vec!["--max-tokens", "4096"],
        vec!["--backend", "auto"],
        vec!["text-model"],
        vec!["--temperature", "0.0"],
    ] {
        let mut args = vec![
            "ferrum",
            "run",
            "--image-config",
            "image.json",
            "--prompt",
            "a cat",
            "--output",
            "image.png",
        ];
        args.extend(extra);
        assert!(validate(&args).is_err(), "{args:?}");
    }
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--output",
        "image.png"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "run",
        "--image-config",
        "image.json",
        "--prompt",
        "a cat"
    ])
    .is_err());
    assert!(validate(&["ferrum", "run", "text-model", "--width", "512"]).is_err());
}

#[cfg(not(feature = "cuda"))]
#[test]
fn non_cuda_image_run_rejects_before_file_or_asset_io() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("image.png");
    let error = execute_run(
        &PathBuf::from("missing.json"),
        "cat".into(),
        512,
        512,
        0,
        &output,
    )
    .unwrap_err();
    assert!(matches!(error, ImageProductError::Unsupported(_)));
    assert!(!output.exists());
}

#[test]
fn image_serve_rejects_text_controls_and_requires_bounded_capacity() {
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "image.json",
        "--host",
        "127.0.0.1",
        "--port",
        "8090"
    ])
    .is_ok());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "image.json",
        "--backend",
        "auto"
    ])
    .is_err());
    assert!(validate(&[
        "ferrum",
        "serve",
        "--image-config",
        "image.json",
        "--prefix-rendezvous-max-wait-ms",
        "1"
    ])
    .is_err());
    assert!(validate(&["ferrum", "serve", "--model", "text-model"]).is_ok());
    let mut wire = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,"max_num_seqs":1,
        "request_batch_max_wait_ms":0,"default_width":512,"default_height":512,
        "default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
    serde_json::from_value::<ServingConfig>(wire.clone())
        .unwrap()
        .validate()
        .unwrap();
    wire["max_num_seqs"] = json!(ferrum_runtime::STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1);
    assert!(serde_json::from_value::<ServingConfig>(wire.clone())
        .unwrap()
        .validate()
        .is_err());
    wire["max_num_seqs"] = json!(1);
    wire["maximum_pending"] = json!(0);
    assert!(serde_json::from_value::<ServingConfig>(wire)
        .unwrap()
        .validate()
        .is_err());
}
