use std::fs;
use std::path::Path;

use serde_json::{json, Value};
use tempfile::TempDir;
use tokenizers::models::bpe::{Vocab, BPE};

use super::*;

// These small fixtures exercise source preflight, not the model's numerical
// configuration or output. No production model download is needed by tests.
fn model_index() -> Value {
    json!({
        "_class_name": "QwenImage21Pipeline",
        "_diffusers_version": "0.41.0.dev0",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
        "sample_sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
    })
}

struct MetadataFixture {
    directory: TempDir,
}

impl MetadataFixture {
    fn new() -> Self {
        let fixture = Self {
            directory: tempfile::tempdir().unwrap(),
        };
        fixture.write_json("model_index.json", &model_index());
        fixture.write_json(
            "text_encoder/config.json",
            &json!({
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl"
            }),
        );
        // Actual 2.1 transformer fields. Their numerical interpretation belongs
        // to the component program; this stage verifies declaration identity.
        fixture.write_json(
            "transformer/config.json",
            &json!({
                "_class_name": "QwenImage21Transformer2DModel",
                "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56],
                "causal_condition": true, "context_in_dim": 4096, "eps": 0.000001,
                "in_channels": 64, "mlp_ratio": 3, "num_attention_heads": 32,
                "num_layers": 32, "out_channels": 64, "patch_size": 1
            }),
        );
        fixture.write_json(
            "vae/config.json",
            &json!({"_class_name": "AutoencoderKLQwenImage21"}),
        );
        fixture.write_json(
            "scheduler/scheduler_config.json",
            &json!({"_class_name": "FlowMatchEulerDiscreteScheduler"}),
        );
        fixture.write_json(
            "processor/processor_config.json",
            &json!({"processor_class": "Qwen3VLProcessor"}),
        );
        fixture.write_json(
            "processor/tokenizer_config.json",
            &json!({
                "processor_class": "Qwen3VLProcessor", "tokenizer_class": "Qwen2Tokenizer"
            }),
        );
        let vocab: Vocab = [("[UNK]".to_owned(), 0), ("a".to_owned(), 1)]
            .into_iter()
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let tokenizer = tokenizers::Tokenizer::new(model).to_string(false).unwrap();
        fixture.write("processor/tokenizer.json", tokenizer.as_bytes());
        fixture.write("processor/chat_template.jinja", b"{{ messages }}");
        fixture
    }

    fn root(&self) -> &Path {
        self.directory.path()
    }

    fn write(&self, path: &str, bytes: &[u8]) {
        let path = self.root().join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn write_json(&self, path: &str, value: &Value) {
        self.write(path, &serde_json::to_vec(value).unwrap());
    }

    fn open(&self) -> Result<PipelineSourceBundle> {
        PipelineSourceBundle::open(self.root(), PipelineSourceLimits::default())
    }
}

fn lock_for(bundle: &PipelineSourceBundle) -> PipelineMetadataLock {
    PipelineMetadataLock::new(
        "fixture/component-pipeline",
        "0123456789abcdef0123456789abcdef01234567",
        bundle.manifest().files().to_vec(),
    )
    .unwrap()
}

#[test]
fn recognizes_real_pipeline_declarations_without_repo_name_dispatch() {
    let fixture = MetadataFixture::new();
    // A directory at the usual weight path must not be inspected or opened.
    fs::create_dir_all(
        fixture
            .root()
            .join("transformer/diffusion_pytorch_model.safetensors"),
    )
    .unwrap();
    let bundle = fixture.open().unwrap();
    assert_eq!(
        bundle.declaration().architecture(),
        PipelineArchitecture::QwenImage21
    );
    assert_eq!(bundle.declaration().components().len(), 5);
    assert_eq!(
        bundle.declaration().saved_sigmas().unwrap(),
        &[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
    );
    assert_eq!(
        bundle
            .component("transformer")
            .unwrap()
            .declaration()
            .architecture(),
        ComponentArchitecture::QwenImage21Denoiser
    );
    assert_eq!(
        bundle
            .component("vae")
            .unwrap()
            .declaration()
            .architecture(),
        ComponentArchitecture::QwenImage21LatentCodec
    );
    assert!(bundle
        .component("processor")
        .unwrap()
        .auxiliary("processor/tokenizer.json")
        .is_some());
    assert!(!bundle
        .manifest()
        .files()
        .iter()
        .any(|file| file.path().contains("safetensors")));
}

#[test]
fn rejects_unsupported_or_incomplete_declarations_before_component_reads() {
    for (field, value, expected) in [
        (
            "_class_name",
            json!("QwenImagePipeline"),
            "unregistered pipeline architecture",
        ),
        (
            "transformer",
            json!(["diffusers", "QwenImageTransformer2DModel"]),
            "unsupported transformer",
        ),
        (
            "vae",
            json!(["custom_code", "AutoencoderKLQwenImage21"]),
            "unsupported vae",
        ),
        ("processor", Value::Null, "missing required component"),
        (
            "audio_encoder",
            json!(["transformers", "UnregisteredAudioEncoder"]),
            "unknown field",
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut index = model_index();
        index[field] = value;
        fs::write(
            root.path().join("model_index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let error = PipelineSourceBundle::open(root.path(), PipelineSourceLimits::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn missing_and_empty_required_files_are_explicit_preflight_errors() {
    for path in [
        "model_index.json",
        "text_encoder/config.json",
        "transformer/config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
        "processor/processor_config.json",
        "processor/tokenizer_config.json",
        "processor/tokenizer.json",
        "processor/chat_template.jinja",
    ] {
        let fixture = MetadataFixture::new();
        fs::remove_file(fixture.root().join(path)).unwrap();
        let error = fixture.open().unwrap_err().to_string();
        assert!(error.contains(path), "{error}");
        fixture.write(path, b"");
        let error = fixture.open().unwrap_err().to_string();
        assert!(error.contains(path) && error.contains("empty"), "{error}");
    }
}

#[test]
fn malformed_or_mismatched_metadata_does_not_pass_preflight() {
    for (path, bytes) in [
        ("transformer/config.json", b"not json".as_slice()),
        (
            "transformer/config.json",
            br#"{"_class_name":"QwenImageTransformer2DModel"}"#.as_slice(),
        ),
        (
            "text_encoder/config.json",
            br#"{"architectures":["Qwen2_5_VLForConditionalGeneration"],"model_type":"qwen3_vl"}"#
                .as_slice(),
        ),
        ("processor/processor_config.json", b"[]".as_slice()),
        (
            "processor/tokenizer_config.json",
            br#"{"processor_class":"Qwen3VLProcessor","tokenizer_class":"UnsupportedTokenizer"}"#
                .as_slice(),
        ),
        ("processor/tokenizer.json", b"{}".as_slice()),
        ("processor/chat_template.jinja", b" \n\t".as_slice()),
        ("processor/chat_template.jinja", b"\xff".as_slice()),
    ] {
        let fixture = MetadataFixture::new();
        fixture.write(path, bytes);
        let error = fixture.open().unwrap_err().to_string();
        assert!(error.contains(path), "{error}");
    }
}

#[test]
fn optional_generation_config_is_retained_if_present_and_must_be_valid() {
    let fixture = MetadataFixture::new();
    let without = fixture.open().unwrap();
    fixture.write_json(
        "text_encoder/generation_config.json",
        &json!({"eos_token_id": 151645}),
    );
    let with = fixture.open().unwrap();
    assert_ne!(
        without.manifest().fingerprint(),
        with.manifest().fingerprint()
    );
    assert!(with
        .component("text_encoder")
        .unwrap()
        .auxiliary("text_encoder/generation_config.json")
        .is_some());
    fixture.write("text_encoder/generation_config.json", b"");
    assert!(fixture
        .open()
        .unwrap_err()
        .to_string()
        .contains("text_encoder/generation_config.json"));
}

#[test]
fn saved_schedule_validation_does_not_guess_a_model_name_or_step_count() {
    let mut index = model_index();
    for sigmas in [
        json!([]),
        json!([1.0, 0.0]),
        json!([0.5, 0.75]),
        json!([1.1]),
        json!([0.5, 0.5]),
    ] {
        index["sample_sigmas"] = sigmas;
        assert!(PipelineDeclaration::from_json(&serde_json::to_vec(&index).unwrap()).is_err());
    }
    index["sample_sigmas"] = json!([0.9, 0.2]);
    let declaration = PipelineDeclaration::from_json(&serde_json::to_vec(&index).unwrap()).unwrap();
    assert_eq!(declaration.saved_sigmas(), Some([0.9, 0.2].as_slice()));
    index.as_object_mut().unwrap().remove("sample_sigmas");
    assert!(
        PipelineDeclaration::from_json(&serde_json::to_vec(&index).unwrap())
            .unwrap()
            .saved_sigmas()
            .is_none()
    );
}

#[test]
fn source_snapshot_and_hash_are_bound_to_retained_bytes() {
    let fixture = MetadataFixture::new();
    let bundle = fixture.open().unwrap();
    let lock = lock_for(&bundle);
    fixture.write("processor/chat_template.jinja", b"{{ changed_messages }}");
    assert_eq!(
        bundle
            .component("processor")
            .unwrap()
            .auxiliary("processor/chat_template.jinja")
            .unwrap()
            .bytes(),
        b"{{ messages }}"
    );
    bundle.verify_metadata_lock(&lock).unwrap();
    let changed = fixture.open().unwrap();
    assert_ne!(
        bundle.manifest().fingerprint(),
        changed.manifest().fingerprint()
    );
    assert!(changed
        .verify_metadata_lock(&lock)
        .unwrap_err()
        .to_string()
        .contains("processor/chat_template.jinja"));
    let known = MetadataFileDigest::from_bytes("sample.json", b"abc").unwrap();
    assert_eq!(known.bytes(), 3);
    assert_eq!(
        known.sha256(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn source_identity_is_independent_of_local_directory_and_lock_order() {
    let first = MetadataFixture::new().open().unwrap();
    let second = MetadataFixture::new().open().unwrap();
    assert_ne!(first.root(), second.root());
    assert_eq!(first.manifest(), second.manifest());
    let lock = lock_for(&first);
    let bytes = serde_json::to_vec(&lock).unwrap();
    assert_eq!(PipelineMetadataLock::from_json(&bytes).unwrap(), lock);
    let mut files = lock.files().to_vec();
    files.reverse();
    let reversed = PipelineMetadataLock::new(lock.repository_id(), lock.revision(), files).unwrap();
    second.verify_metadata_lock(&reversed).unwrap();
}

#[test]
fn revision_alone_and_partial_or_duplicate_locks_are_not_source_proof() {
    let bundle = MetadataFixture::new().open().unwrap();
    let lock = lock_for(&bundle);
    assert!(
        PipelineMetadataLock::new(lock.repository_id(), "main", lock.files().to_vec()).is_err()
    );
    assert!(PipelineMetadataLock::new(lock.repository_id(), lock.revision(), vec![]).is_err());
    let file = lock.files()[0].clone();
    assert!(PipelineMetadataLock::new(
        lock.repository_id(),
        lock.revision(),
        vec![file.clone(), file.clone()]
    )
    .is_err());
    let partial =
        PipelineMetadataLock::new(lock.repository_id(), lock.revision(), vec![file]).unwrap();
    assert!(bundle.verify_metadata_lock(&partial).is_err());
    for path in [
        "../config.json",
        "a/../config.json",
        "/config.json",
        "a//b",
        "a\\b",
        "C:/config.json",
    ] {
        assert!(MetadataFileDigest::new(path, 1, "a".repeat(64)).is_err());
    }
    assert!(MetadataFileDigest::new("config.json", 1, "not-a-hash").is_err());
    assert!(MetadataFileDigest::new("config.json", 0, "a".repeat(64)).is_err());
}

#[test]
fn read_limits_cover_per_file_and_total_retained_bytes() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("one.json"), b"1234").unwrap();
    fs::write(root.path().join("two.json"), b"12345").unwrap();
    let limits = PipelineSourceLimits::new(4, 5, 9).unwrap();
    let mut reader = files::MetadataReader::new(root.path(), limits).unwrap();
    assert_eq!(reader.read("one.json", false).unwrap().bytes(), b"1234");
    assert!(reader.read("two.json", false).is_err());
    assert_eq!(reader.read("two.json", true).unwrap().bytes(), b"12345");
    assert!(reader.read("one.json", false).is_err());
    assert!(PipelineSourceLimits::new(0, 1, 1).is_err());
    let max_allocation = isize::MAX as u64;
    assert!(
        PipelineSourceLimits::new(max_allocation - 1, max_allocation - 1, max_allocation - 1)
            .is_ok()
    );
    for invalid in [max_allocation, max_allocation + 1, u64::MAX] {
        assert!(PipelineSourceLimits::new(invalid, 1, 1).is_err());
        assert!(PipelineSourceLimits::new(1, invalid, 1).is_err());
        assert!(PipelineSourceLimits::new(1, 1, invalid).is_err());
    }
    assert!(PipelineSourceLimits::new(1, 1, u64::MAX).is_err());
}

#[test]
fn component_instances_with_the_same_role_remain_distinct() {
    let fixture = MetadataFixture::new();
    let mut bundle = fixture.open().unwrap();
    let first = bundle.component("text_encoder").unwrap().clone();
    fixture.write("text_encoder_2/config.json", first.config().bytes());
    let mut reader =
        files::MetadataReader::new(fixture.root(), PipelineSourceLimits::default()).unwrap();
    // Internal collection fixture only: no second model architecture is
    // registered, and this does not claim a second executable pipeline.
    let second = PipelineComponentSources {
        declaration: ComponentDeclaration::registered(
            ComponentRole::ConditionEncoder,
            ComponentArchitecture::Qwen3VlConditionEncoder,
            "text_encoder_2",
            "text_encoder_2/config.json",
        ),
        config: reader.read("text_encoder_2/config.json", false).unwrap(),
        auxiliary: BTreeMap::new(),
    };
    insert_component(&mut bundle.components, second.clone()).unwrap();
    let encoder_ids: Vec<_> = bundle
        .components_with_role(ComponentRole::ConditionEncoder)
        .map(|component| component.declaration().id())
        .collect();
    assert_eq!(encoder_ids, ["text_encoder", "text_encoder_2"]);
    assert_eq!(
        bundle.component("text_encoder").unwrap().config().path(),
        "text_encoder/config.json"
    );
    assert_eq!(
        bundle.component("text_encoder_2").unwrap().config().path(),
        "text_encoder_2/config.json"
    );
    assert!(insert_component(&mut bundle.components, second).is_err());
    assert_eq!(
        bundle
            .components_with_role(ComponentRole::ConditionEncoder)
            .count(),
        2
    );
}

#[cfg(unix)]
#[test]
fn metadata_symlinks_must_stay_within_the_explicit_source_boundary() {
    use std::os::unix::fs::symlink;

    let fixture = MetadataFixture::new();
    let outside = tempfile::tempdir().unwrap();
    let target = fixture.root().join("transformer/config.json");
    fs::copy(&target, outside.path().join("config.json")).unwrap();
    fs::remove_file(&target).unwrap();
    symlink(outside.path().join("config.json"), &target).unwrap();
    assert!(fixture
        .open()
        .unwrap_err()
        .to_string()
        .contains("outside its source root"));

    fs::remove_file(&target).unwrap();
    fs::copy(
        outside.path().join("config.json"),
        fixture.root().join("retained-config.json"),
    )
    .unwrap();
    symlink("../retained-config.json", &target).unwrap();
    fixture.open().unwrap();
}
