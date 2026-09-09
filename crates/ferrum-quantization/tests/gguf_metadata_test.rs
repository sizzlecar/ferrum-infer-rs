use candle_core::quantized::gguf_file::{self, Value};
use ferrum_quantization::gguf::GgufModelMetadata;
use std::io::Cursor;

fn read(metadata: &[(&str, &Value)]) -> candle_core::Result<GgufModelMetadata> {
    let mut bytes = Cursor::new(Vec::new());
    gguf_file::write(&mut bytes, metadata, &[]).unwrap();
    bytes.set_position(0);
    GgufModelMetadata::read(&mut bytes)
}

#[test]
fn reads_optional_sparse_provenance_and_skips_tokenizer_arrays() {
    let architecture = Value::String("fixture".into());
    let source = Value::String("https://huggingface.co/Owner/Finetuned".into());
    let parent = Value::String("https://huggingface.co/Owner/Base".into());
    let metadata = read(&[
        ("general.architecture", &architecture),
        ("general.source.repo_url", &source),
        ("general.base_model.count", &Value::U32(2)),
        ("general.base_model.1.repo_url", &parent),
        (
            "tokenizer.ggml.tokens",
            &Value::Array(vec![Value::String("hello".into())]),
        ),
    ])
    .unwrap();
    assert_eq!(metadata.architecture, "fixture");
    assert_eq!(
        metadata.source_repository_url.as_deref(),
        Some("https://huggingface.co/Owner/Finetuned")
    );
    assert_eq!(metadata.base_model_count, Some(2));
    assert_eq!(metadata.base_model_repository_urls.len(), 1);
    assert_eq!(
        metadata.base_model_repository_urls[&1],
        "https://huggingface.co/Owner/Base"
    );
    let optional = read(&[("general.architecture", &architecture)]).unwrap();
    assert!(optional.source_repository_url.is_none());
    assert!(optional.base_model_count.is_none());
    assert!(optional.base_model_repository_urls.is_empty());
}

#[test]
fn rejects_missing_architecture_invalid_provenance_types_and_parent_indices() {
    let architecture = Value::String("fixture".into());
    let parent = Value::String("https://huggingface.co/Owner/Base".into());
    assert!(read(&[]).is_err());
    assert!(read(&[("general.architecture", &Value::String(String::new()))]).is_err());
    for extra in [
        vec![("general.source.repo_url", &Value::U32(1))],
        vec![("general.base_model.0.repo_url", &parent)],
        vec![
            ("general.base_model.count", &Value::U32(0)),
            ("general.base_model.0.repo_url", &parent),
        ],
        vec![
            ("general.base_model.count", &Value::U32(1)),
            ("general.base_model.1.repo_url", &parent),
        ],
        vec![("general.base_model.count", &Value::I32(-1))],
    ] {
        let mut metadata = vec![("general.architecture", &architecture)];
        metadata.extend(extra);
        assert!(read(&metadata).is_err());
    }
}
