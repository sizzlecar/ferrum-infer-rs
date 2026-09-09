use super::*;
use candle_core::quantized::gguf_file::{self, Content, Value};
use std::io::Cursor;

pub(crate) fn write_metadata_fixture(path: &Path, architecture: &str, metadata: &[(&str, &Value)]) {
    let mut values = metadata.to_vec();
    let architecture = Value::String(architecture.into());
    values.push(("general.architecture", &architecture));
    gguf_file::write(&mut std::fs::File::create(path).unwrap(), &values, &[]).unwrap();
}

/// Header-only IQ4_XS artifact. Source composition does not execute its weights.
fn write_iq_fixture(path: &Path, metadata: &[(&str, &Value)]) {
    write_metadata_fixture(path, "qwen35", metadata);
    let mut bytes = std::fs::read(path).unwrap();
    let mut reader = Cursor::new(&bytes);
    Content::read(&mut reader).unwrap();
    let descriptor_start = reader.position() as usize;
    bytes.truncate(descriptor_start);
    bytes[8..16].copy_from_slice(&1_u64.to_le_bytes());
    let name = "projection.weight";
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&256_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&23_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.resize(bytes.len().div_ceil(32) * 32 + 136, 0);
    assert!(Content::read(&mut Cursor::new(&bytes)).is_err());
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn provenance_selects_only_an_unambiguous_huggingface_model_repository() {
    let mut metadata = GgufModelMetadata {
        architecture: "qwen35".into(),
        source_repository_url: None,
        base_model_count: None,
        base_model_repository_urls: Default::default(),
    };
    assert_eq!(declared_metadata_repository(&metadata).unwrap(), None);
    metadata.base_model_count = Some(1);
    metadata
        .base_model_repository_urls
        .insert(0, "https://huggingface.co/Original/Model/".into());
    assert_eq!(
        declared_metadata_repository(&metadata).unwrap().as_deref(),
        Some("Original/Model")
    );
    metadata.base_model_count = Some(2);
    assert!(declared_metadata_repository(&metadata).is_err());
    metadata.source_repository_url = Some("https://huggingface.co/Finetuner/Checkpoint".into());
    assert_eq!(
        declared_metadata_repository(&metadata).unwrap().as_deref(),
        Some("Finetuner/Checkpoint")
    );
    for url in [
        "http://huggingface.co/Original/Model",
        "https://huggingface.co.evil.test/Original/Model",
        "https://huggingface.co@evil.test/Original/Model",
        "https://user@huggingface.co/Original/Model",
        "https://huggingface.co:8443/Original/Model",
        "https://huggingface.co/Original/Model?revision=main",
        "https://huggingface.co/Original/Model#x",
        "https://huggingface.co/Original/Model/tree/main",
        "https://huggingface.co/Original%2fOther/Model",
        "https://huggingface.co/Model",
    ] {
        assert!(huggingface_repository_url(url).is_err(), "{url}");
    }
}

#[tokio::test]
async fn iq_repository_uses_declared_source_for_default_pinned_and_tokenizer_override() {
    let cache = tempfile::tempdir().unwrap();
    let repo = "Quantizer/Unknown-GGUF";
    let revision = "a".repeat(40);
    let snapshot = cache
        .path()
        .join("hub/models--Quantizer--Unknown-GGUF/snapshots")
        .join(&revision);
    let semantic = cache
        .path()
        .join("hub/models--Original--Checkpoint/snapshots")
        .join("b".repeat(40));
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(&semantic).unwrap();
    let gguf = snapshot.join("model.gguf");
    write_iq_fixture(
        &gguf,
        &[
            ("general.base_model.count", &Value::U32(1)),
            (
                "general.base_model.0.repo_url",
                &Value::String("https://huggingface.co/Original/Checkpoint".into()),
            ),
        ],
    );
    std::fs::write(
        semantic.join("config.json"),
        super::super::tests::qwen35_semantic_config(false),
    )
    .unwrap();
    std::fs::write(semantic.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
    std::fs::write(
        semantic.join("tokenizer_config.json"),
        br#"{"chat_template":"fixture-template"}"#,
    )
    .unwrap();
    for request in [repo.to_owned(), format!("{repo}@{revision}")] {
        let resolved = resolve_model_source_with_product_sources(
            &request,
            cache.path(),
            DownloadPolicy::NoDownload,
            None,
            &ProductSourceArgs::default(),
        )
        .await
        .unwrap()
        .into_product_engine_input();
        let sources = resolved.model_sources.unwrap();
        assert_eq!(sources.weights().path(), gguf.canonicalize().unwrap());
        assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
        assert_eq!(sources.tokenizer_root(), semantic.canonicalize().unwrap());
        assert_eq!(
            sources.original_sources().semantic.location,
            "Original/Checkpoint"
        );
        assert_eq!(sources.original_sources().weights.location, repo);
        assert_eq!(
            sources
                .original_sources()
                .weights
                .requested_revision
                .as_deref(),
            request.contains('@').then_some(revision.as_str())
        );
    }
    let tokenizer = cache.path().join("explicit-tokenizer");
    std::fs::create_dir(&tokenizer).unwrap();
    for file in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::rename(semantic.join(file), tokenizer.join(file)).unwrap();
    }
    let resolved = resolve_model_source_with_product_sources(
        repo,
        cache.path(),
        DownloadPolicy::NoDownload,
        None,
        &ProductSourceArgs {
            semantic_source: None,
            tokenizer_source: Some(tokenizer.clone()),
        },
    )
    .await
    .unwrap()
    .into_product_engine_input();
    let sources = resolved.model_sources.unwrap();
    assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
    assert_eq!(sources.tokenizer_root(), tokenizer.canonicalize().unwrap());
    assert!(!snapshot.join("config.json").exists());
}

#[tokio::test]
async fn malformed_gguf_metadata_is_reported_before_legacy_tokenizer_resolution() {
    let root = tempfile::tempdir().unwrap();
    let gguf = root.path().join("broken.gguf");
    std::fs::write(&gguf, b"GGUF").unwrap();
    let error = resolve_model_source(
        gguf.to_str().unwrap(),
        root.path(),
        DownloadPolicy::NoDownload,
        None,
    )
    .await
    .err()
    .expect("malformed metadata must fail")
    .to_string();
    assert!(error.contains("cannot read GGUF metadata"), "{error}");
}
