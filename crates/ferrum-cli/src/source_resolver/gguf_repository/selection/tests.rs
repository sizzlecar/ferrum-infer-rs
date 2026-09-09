use super::*;
use crate::source_resolver::gguf_repository::tests::write_iq_fixture;
use ferrum_quantization::gguf::Value;

#[tokio::test]
async fn explicit_filename_selects_one_cached_quantization_and_respects_revision_and_roles() {
    let cache = tempfile::tempdir().unwrap();
    let repo = "Quantizer/Checkpoint-GGUF";
    let revision = "a".repeat(40);
    let current = "b".repeat(40);
    let repo_dir = cache.path().join("hub/models--Quantizer--Checkpoint-GGUF");
    let semantic = cache
        .path()
        .join("hub/models--Original--Checkpoint/snapshots")
        .join("c".repeat(40));
    std::fs::create_dir_all(&semantic).unwrap();
    std::fs::write(
        semantic.join("config.json"),
        crate::source_resolver::tests::qwen35_semantic_config(false),
    )
    .unwrap();
    std::fs::write(semantic.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
    std::fs::write(
        semantic.join("tokenizer_config.json"),
        br#"{"chat_template":"fixture-template"}"#,
    )
    .unwrap();
    let filename = "weights/model-IQ4_XS.gguf";
    for sha in [&revision, &current] {
        let snapshot = repo_dir.join("snapshots").join(sha);
        std::fs::create_dir_all(snapshot.join("weights")).unwrap();
        for name in [filename, "other-quantization.gguf"] {
            write_iq_fixture(
                &snapshot.join(name),
                &[
                    ("general.base_model.count", &Value::U32(1)),
                    (
                        "general.base_model.0.repo_url",
                        &Value::String("https://huggingface.co/Original/Checkpoint".into()),
                    ),
                ],
            );
        }
    }
    std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
    std::fs::write(repo_dir.join("refs/main"), &current).unwrap();
    let mut args = ProductSourceArgs {
        gguf_file: Some(filename.into()),
        ..Default::default()
    };
    for (request, selected_revision) in [
        (repo.to_owned(), &current),
        (format!("{repo}@{revision}"), &revision),
    ] {
        let product = resolve_model_source_with_product_sources(
            &request,
            cache.path(),
            DownloadPolicy::NoDownload,
            None,
            &args,
        )
        .await
        .unwrap()
        .into_product_engine_input();
        let expected = repo_dir
            .join("snapshots")
            .join(selected_revision)
            .join(filename)
            .canonicalize()
            .unwrap();
        let sources = product.model_sources.unwrap();
        assert_eq!(sources.weights().path(), expected);
        assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
        assert_eq!(sources.tokenizer_root(), semantic.canonicalize().unwrap());
        assert_eq!(sources.original_sources().weights.location, repo);
        assert_eq!(
            sources.resolved_sources().weights.files[0].relative_path,
            filename
        );
        assert_eq!(
            sources.weight_payload_bytes().unwrap(),
            std::fs::metadata(&expected).unwrap().len()
        );
        assert_eq!(
            sources
                .original_sources()
                .weights
                .requested_revision
                .as_deref(),
            request.contains('@').then_some(revision.as_str())
        );
        assert!(product.source.from_cache);
    }
    let absent_pin = format!("{repo}@{}", "d".repeat(40));
    assert!(resolve_model_source_with_product_sources(
        &absent_pin,
        cache.path(),
        DownloadPolicy::NoDownload,
        None,
        &args
    )
    .await
    .is_err());
    let tokenizer = cache.path().join("tokenizer-override");
    std::fs::create_dir(&tokenizer).unwrap();
    for file in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::rename(semantic.join(file), tokenizer.join(file)).unwrap();
    }
    args.tokenizer_source = Some(tokenizer.clone());
    let product = resolve_model_source_with_product_sources(
        repo,
        cache.path(),
        DownloadPolicy::NoDownload,
        None,
        &args,
    )
    .await
    .unwrap()
    .into_product_engine_input();
    let sources = product.model_sources.unwrap();
    assert_eq!(sources.tokenizer_root(), tokenizer.canonicalize().unwrap());
    assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
}

#[tokio::test]
async fn exact_file_option_rejects_local_paths_aliases_and_escaping_filenames_before_access() {
    let cache = tempfile::tempdir().unwrap();
    let args = ProductSourceArgs {
        gguf_file: Some("model.gguf".into()),
        ..Default::default()
    };
    for model in [
        "qwen3.5:4b",
        "local.gguf",
        "/tmp/model",
        "Owner/Repo/extra",
        "Owner/../Repo",
        "Owner/Repo@main",
    ] {
        assert!(
            resolve_model_source_with_product_sources(
                model,
                cache.path(),
                DownloadPolicy::NoDownload,
                None,
                &args
            )
            .await
            .is_err(),
            "{model}"
        );
    }
    for filename in ["../model.gguf", "/model.gguf", "model.safetensors"] {
        let args = ProductSourceArgs {
            gguf_file: Some(filename.into()),
            ..Default::default()
        };
        assert!(resolve_model_source_with_product_sources(
            "Owner/Repo",
            cache.path(),
            DownloadPolicy::NoDownload,
            None,
            &args
        )
        .await
        .is_err());
    }
    assert!(!cache.path().join("hub").exists());
}
