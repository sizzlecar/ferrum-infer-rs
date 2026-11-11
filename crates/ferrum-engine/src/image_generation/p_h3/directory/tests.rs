use super::*;
use ferrum_runtime::{
    ElementType, WeightComponentRole, WeightComponentSource, WeightComponentSpec, WeightEncoding,
    WeightId,
};
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use serde_json::json;

const CONFIG_PATHS: [&str; 7] = [
    "text_encoder/config.json",
    "text_connector/config.json",
    "transformer/config.json",
    "vae/config.json",
    "scheduler/scheduler_config.json",
    "audio_scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
];

fn write(root: &Path, relative: &str, bytes: &[u8]) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}
fn model_index() -> Vec<u8> {
    let mut value = json!({
        "_class_name":"PH3ImageModularPipeline", "_blocks_class_name":"PH3ImageBlocks",
        "reference_min_pixels":1, "reference_max_pixels":2,
        "canvas_min_pixels":1, "vision_pixels":1
    });
    for (slot, library, class) in [
        (
            "text_encoder",
            "transformers",
            "Qwen3VLForConditionalGeneration",
        ),
        ("text_connector", "diffusers", "PH3ImageTextConnector"),
        ("tokenizer", "transformers", "Qwen2TokenizerFast"),
        ("processor", "transformers", "Qwen3VLProcessor"),
        ("transformer", "diffusers", "MiniMaxH3Transformer3DModel"),
        ("vae", "diffusers", "AutoencoderKLPH3Image"),
        ("scheduler", "diffusers", "MiniMaxH3Scheduler"),
        ("audio_scheduler", "diffusers", "MiniMaxH3Scheduler"),
    ] {
        value[slot] = json!([library, class, {
            "type_hint":[library, class], "pretrained_model_name_or_path":"offline-fixture",
            "subfolder":slot, "variant":null, "revision":null
        }]);
    }
    serde_json::to_vec(&value).unwrap()
}
fn installation() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    write(root.path(), "modular_model_index.json", &model_index());
    for path in CONFIG_PATHS {
        write(root.path(), path, b"{ \"fixture\": true }\n");
    }
    write(
        root.path(),
        "tokenizer/tokenizer.json",
        b"{ \"vocabulary_fixture\": true }\n",
    );
    root
}
fn missing_archives(root: &Path) -> PH3ImageArchivePaths {
    PH3ImageArchivePaths::new(
        root.join("missing-text.safetensors"),
        root.join("missing-transformer.safetensors"),
        root.join("missing-vae.safetensors"),
    )
}
fn directory_error(root: &Path) -> String {
    format!(
        "{:#}",
        PH3ImageAssets::from_directory(
            root,
            missing_archives(root),
            PipelineSourceLimits::default(),
            PH3ImageAssetLimits::default(),
        )
        .err()
        .expect("fixture must fail before any runtime exists")
    )
}

#[test]
fn reads_original_modular_metadata_and_retains_exact_bytes_without_following_registrations() {
    let root = installation();
    // The declaration's remote identity is data. No directory named after it
    // is needed, and model_index.json cannot replace the modular entrypoint.
    write(root.path(), "model_index.json", b"not the entrypoint");
    let metadata = LocalMetadata::open(
        root.path(),
        PipelineSourceLimits::default(),
        PH3ImageAssetLimits::default(),
    )
    .unwrap();
    let original = fs::read(root.path().join("tokenizer/tokenizer.json")).unwrap();
    assert_eq!(metadata.tokenizer, original);
    assert_eq!(metadata.model_index, model_index());
    let borrowed = metadata.borrowed();
    assert_eq!(borrowed.tokenizer.as_ptr(), metadata.tokenizer.as_ptr());
    fs::write(root.path().join("tokenizer/tokenizer.json"), b"{}").unwrap();
    assert_eq!(borrowed.tokenizer, original);
    fs::remove_file(root.path().join("modular_model_index.json")).unwrap();
    let error = directory_error(root.path());
    assert!(
        error.contains("required H3 metadata modular_model_index.json"),
        "{error}"
    );
    assert!(!error.contains("open H3 text_encoder archive"), "{error}");
}

#[test]
fn missing_empty_and_malformed_metadata_fail_before_any_archive_is_opened() {
    let root = installation();
    fs::remove_file(root.path().join("transformer/config.json")).unwrap();
    let error = directory_error(root.path());
    assert!(
        error.contains("required H3 metadata transformer/config.json"),
        "{error}"
    );
    write(root.path(), "transformer/config.json", b"");
    let error = directory_error(root.path());
    assert!(
        error.contains("transformer/config.json is empty"),
        "{error}"
    );
    write(root.path(), "transformer/config.json", b"{");
    let error = directory_error(root.path());
    assert!(
        error.contains("invalid H3 metadata transformer/config.json"),
        "{error}"
    );
    assert!(!error.contains("open H3 text_encoder archive"), "{error}");
    let mut wrong: serde_json::Value = serde_json::from_slice(&model_index()).unwrap();
    wrong["_class_name"] = json!("UnsupportedPipeline");
    write(
        root.path(),
        "modular_model_index.json",
        &serde_json::to_vec(&wrong).unwrap(),
    );
    assert!(directory_error(root.path())
        .contains("does not declare the supported image component graph"));
}

#[test]
fn exact_file_total_and_combined_tokenizer_caps_bound_reads() {
    let root = installation();
    let files = LocalMetadata::open(
        root.path(),
        PipelineSourceLimits::default(),
        PH3ImageAssetLimits::default(),
    )
    .unwrap();
    let all = files.files();
    let total: usize = all.iter().map(|(_, bytes)| bytes.len()).sum();
    let maximum_config = all
        .iter()
        .filter(|(path, _)| *path != "tokenizer/tokenizer.json")
        .map(|(_, bytes)| bytes.len())
        .max()
        .unwrap();
    let tokenizer = files.tokenizer.len();
    let combined = tokenizer + files.tokenizer_config.len();
    let limits =
        PipelineSourceLimits::new(maximum_config as u64, tokenizer as u64, total as u64).unwrap();
    let mut assets = PH3ImageAssetLimits::default();
    assets.metadata_bytes_per_file = maximum_config;
    assets.tokenizer.max_source_bytes = combined;
    LocalMetadata::open(root.path(), limits, assets).unwrap();

    let short_total =
        PipelineSourceLimits::new(maximum_config as u64, tokenizer as u64, (total - 1) as u64)
            .unwrap();
    let error = LocalMetadata::open(root.path(), short_total, assets)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("tokenizer/tokenizer.json exceeds its remaining read limit"),
        "{error}"
    );
    assets.tokenizer.max_source_bytes -= 1;
    let error = LocalMetadata::open(root.path(), limits, assets)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("tokenizer/tokenizer.json exceeds its remaining read limit"),
        "{error}"
    );
    assets.tokenizer.max_source_bytes = combined;
    assets.metadata_bytes_per_file = model_index().len() - 1;
    let error = LocalMetadata::open(root.path(), limits, assets)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("modular_model_index.json exceeds its remaining read limit"),
        "{error}"
    );
    assets.metadata_bytes_per_file = maximum_config;
    let short_tokenizer =
        PipelineSourceLimits::new(maximum_config as u64, (tokenizer - 1) as u64, total as u64)
            .unwrap();
    assert!(LocalMetadata::open(root.path(), short_tokenizer, assets).is_err());
    assets.tokenizer.max_source_bytes = files.tokenizer_config.len() - 1;
    let error = LocalMetadata::open(root.path(), limits, assets)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("tokenizer/tokenizer_config.json exceeds its remaining read limit"),
        "{error}"
    );
}

fn archive(path: &Path, value: f32) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let bytes = value.to_le_bytes();
    serialize_to_file(
        [(
            "original",
            TensorView::new(Dtype::F32, vec![1], &bytes).unwrap(),
        )],
        &None,
        path,
    )
    .unwrap();
}

#[test]
fn explicit_archive_paths_keep_original_mmaps_and_never_discover_an_alternative() {
    let root = tempfile::tempdir().unwrap();
    let paths = PH3ImageArchivePaths::new(
        root.path().join("chosen-text.safetensors"),
        root.path().join("chosen-dit.safetensors"),
        root.path().join("chosen-codec.safetensors"),
    );
    archive(&paths.text_encoder, 1.25);
    archive(&paths.transformer, -2.5);
    archive(&paths.vae, 3.75);
    write(
        root.path(),
        "model.safetensors",
        b"invalid discovery candidate",
    );
    let sources = paths.open().unwrap();
    let spec = WeightComponentSpec {
        id: WeightId::new("weight.original").unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec!["original".into()],
        dimensions: vec![1],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
        required: true,
    };
    for (source, expected, filename) in [
        (&sources.text_encoder, 1.25_f32, "chosen-text.safetensors"),
        (&sources.transformer, -2.5_f32, "chosen-dit.safetensors"),
        (&sources.vae, 3.75_f32, "chosen-codec.safetensors"),
    ] {
        let raw = source.archive().tensor("original").unwrap();
        let payload = source.component(&spec).unwrap();
        assert_eq!(raw.dtype(), Dtype::F32);
        assert_eq!(raw.source_file(), filename);
        assert_eq!(payload.bytes(), expected.to_le_bytes());
        assert_eq!(payload.bytes().as_ptr(), raw.bytes().as_ptr());
    }
    drop(sources);
    fs::remove_file(&paths.text_encoder).unwrap();
    let error = format!("{:#}", paths.open().err().unwrap());
    assert!(error.contains("chosen-text.safetensors"), "{error}");
    assert!(error.contains("open H3 text_encoder archive"), "{error}");

    let preset = PH3ImageArchivePaths::comfy_layout(root.path());
    assert_eq!(
        preset.text_encoder,
        root.path()
            .join("text_encoders/p-h3-image_text_encoder_w4a8.safetensors")
    );
    assert_eq!(
        preset.transformer,
        root.path()
            .join("diffusion_models/p-h3-image_w4a8.safetensors")
    );
    assert_eq!(
        preset.vae,
        root.path()
            .join("vae/p-h3-image_vae_int8_convrot.safetensors")
    );
}

#[cfg(unix)]
#[test]
fn metadata_root_rejects_external_links_but_explicit_archive_links_are_authorized() {
    use std::os::unix::fs::symlink;
    let root = installation();
    let outside = tempfile::tempdir().unwrap();
    write(outside.path(), "config.json", b"{}");
    let config = root.path().join("transformer/config.json");
    fs::remove_file(&config).unwrap();
    symlink(outside.path().join("config.json"), &config).unwrap();
    let error = directory_error(root.path());
    assert!(
        error.contains("transformer/config.json resolves outside its source root"),
        "{error}"
    );
    fs::remove_file(&config).unwrap();
    write(root.path(), "transformer/original.json", b"{}");
    symlink("original.json", &config).unwrap();
    LocalMetadata::open(
        root.path(),
        PipelineSourceLimits::default(),
        PH3ImageAssetLimits::default(),
    )
    .unwrap();

    // The explicit filename is a safetensors path; its authorized immutable
    // HF-style blob target need not have a safetensors extension.
    let blob = outside.path().join("original-blob");
    archive(&blob, 7.125);
    let selected = root.path().join("explicit.safetensors");
    symlink(&blob, &selected).unwrap();
    let source = open_archive("text_encoder", &selected).unwrap();
    assert_eq!(
        source.archive().tensor("original").unwrap().bytes(),
        7.125_f32.to_le_bytes()
    );
}

#[test]
fn nonregular_metadata_and_explicit_archive_paths_are_rejected_clearly() {
    let root = installation();
    let config = root.path().join("vae/config.json");
    fs::remove_file(&config).unwrap();
    fs::create_dir(&config).unwrap();
    let error = directory_error(root.path());
    assert!(
        error.contains("vae/config.json must be a regular file"),
        "{error}"
    );
    let path = root.path().join("directory.safetensors");
    fs::create_dir(&path).unwrap();
    let error = format!("{:#}", open_archive("vae", &path).err().unwrap());
    assert!(
        error.contains("explicit archive must be a regular file"),
        "{error}"
    );
}
