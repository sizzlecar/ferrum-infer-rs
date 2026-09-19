use super::*;
use crate::source_resolver::{
    gguf_repository::tests::write_iq_fixture, resolve_gguf_alias, resolve_model_alias,
    resolve_model_source, resolve_model_source_with_product_sources, DownloadPolicy,
};
use ferrum_interfaces::vnext::ModelSourceKind;
use std::path::{Path, PathBuf};

fn snapshot(cache: &Path, selector: &str) -> PathBuf {
    let (repo, revision) = selector.split_once('@').unwrap();
    let root = cache
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn metadata(root: &Path) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(
        root.join("config.json"),
        crate::source_resolver::tests::qwen35_semantic_config(false),
    )
    .unwrap();
    tokenizer(root);
}

fn tokenizer(root: &Path) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        br#"{"chat_template":"fixture-template"}"#,
    )
    .unwrap();
}

fn cached_recipe(cache: &Path) -> (PathBuf, PathBuf) {
    let recipe = find("bonsai2:27b").unwrap();
    let weights = snapshot(cache, recipe.requested_model).join(recipe.gguf_file);
    // Source composition needs a valid small GGUF, not executable 27B weights.
    write_iq_fixture(&weights, &[]);
    let semantic = snapshot(cache, recipe.semantic_source);
    metadata(&semantic);
    (weights, semantic)
}

#[test]
fn recipes_are_explicit_and_expose_the_exact_quantized_artifact() {
    let recipe = find("bonsai2:27b").unwrap();
    assert!(std::ptr::eq(recipe, find("BONSAI2:27B-PQ2_0").unwrap()));
    for other in [
        "bonsai",
        "bonsai2:27b-q4_k_m",
        "prism-ml/Ternary-Bonsai-2-27B-gguf",
        recipe.requested_model,
        recipe.gguf_file,
    ] {
        assert!(find(other).is_none(), "{other}");
    }
    assert_eq!(
        resolve_gguf_alias("bonsai2:27b"),
        Some((recipe.requested_model.into(), recipe.gguf_file.into()))
    );
    assert_eq!(resolve_model_alias("bonsai2:27b"), recipe.requested_model);
}

#[tokio::test]
async fn recipe_aliases_resolve_independent_pinned_roles_for_product_and_pull() {
    let cache = tempfile::tempdir().unwrap();
    let (weights, semantic) = cached_recipe(cache.path());
    let recipe = find("bonsai2:27b").unwrap();
    let (weight_repo, weight_revision) = recipe.requested_model.split_once('@').unwrap();
    let (semantic_repo, semantic_revision) = recipe.semantic_source.split_once('@').unwrap();
    for alias in ["bonsai2:27b", "bonsai2:27b-pq2_0"] {
        for product_entrypoint in [false, true] {
            let resolved = if product_entrypoint {
                resolve_model_source_with_product_sources(
                    alias,
                    cache.path(),
                    DownloadPolicy::NoDownload,
                    None,
                    &ProductSourceArgs::default(),
                )
                .await
            } else {
                resolve_model_source(alias, cache.path(), DownloadPolicy::NoDownload, None).await
            }
            .unwrap();
            let product = resolved.into_product_engine_input();
            assert_eq!(product.requested_model, alias);
            assert_eq!(product.source.original, alias);
            assert!(product.source.from_cache);
            assert_eq!(product.public_model_id, weight_repo);
            let sources = product.model_sources.unwrap();
            assert_eq!(sources.weights().path(), weights.canonicalize().unwrap());
            assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
            assert_eq!(sources.tokenizer_root(), semantic.canonicalize().unwrap());
            for role in [
                &sources.original_sources().semantic,
                &sources.original_sources().tokenizer,
            ] {
                assert_eq!(role.kind, ModelSourceKind::Repository);
                assert_eq!(role.location, semantic_repo);
                assert_eq!(role.requested_revision.as_deref(), Some(semantic_revision));
            }
            let weight_source = &sources.original_sources().weights;
            assert_eq!(weight_source.location, weight_repo);
            assert_eq!(
                weight_source.requested_revision.as_deref(),
                Some(weight_revision)
            );
            assert_eq!(
                sources.resolved_sources().weights.resolved_revision,
                weight_revision
            );
            assert_eq!(
                sources.resolved_sources().semantic.resolved_revision,
                semantic_revision
            );
            assert_eq!(
                sources.resolved_sources().tokenizer.resolved_revision,
                semantic_revision
            );
        }
    }
}

#[tokio::test]
async fn explicit_artifact_and_metadata_roles_override_recipe_defaults() {
    let cache = tempfile::tempdir().unwrap();
    let recipe = find("bonsai2:27b").unwrap();
    let filename = "alternative-IQ4_XS.gguf";
    let weights = snapshot(cache.path(), recipe.requested_model).join(filename);
    write_iq_fixture(&weights, &[]);
    let semantic = cache.path().join("explicit-semantic");
    metadata(&semantic);
    let separate_tokenizer = cache.path().join("explicit-tokenizer");
    tokenizer(&separate_tokenizer);
    for tokenizer_source in [None, Some(separate_tokenizer.clone())] {
        let product = resolve_model_source_with_product_sources(
            "bonsai2:27b",
            cache.path(),
            DownloadPolicy::NoDownload,
            None,
            &ProductSourceArgs {
                gguf_file: Some(filename.into()),
                semantic_source: Some(semantic.clone()),
                tokenizer_source: tokenizer_source.clone(),
            },
        )
        .await
        .unwrap()
        .into_product_engine_input();
        let sources = product.model_sources.unwrap();
        assert_eq!(sources.weights().path(), weights.canonicalize().unwrap());
        assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
        assert_eq!(
            sources.tokenizer_root(),
            tokenizer_source
                .unwrap_or_else(|| semantic.clone())
                .canonicalize()
                .unwrap()
        );
        assert_eq!(
            sources.original_sources().semantic.kind,
            ModelSourceKind::LocalDirectory
        );
        assert_eq!(
            sources.original_sources().tokenizer.kind,
            ModelSourceKind::LocalDirectory
        );
    }
}

#[tokio::test]
async fn explicit_tokenizer_does_not_require_the_recipes_tokenizer_files() {
    let cache = tempfile::tempdir().unwrap();
    let (weights, semantic) = cached_recipe(cache.path());
    std::fs::remove_file(semantic.join("tokenizer.json")).unwrap();
    std::fs::remove_file(semantic.join("tokenizer_config.json")).unwrap();
    let separate_tokenizer = cache.path().join("explicit-tokenizer");
    tokenizer(&separate_tokenizer);
    let product = resolve_model_source_with_product_sources(
        "bonsai2:27b",
        cache.path(),
        DownloadPolicy::NoDownload,
        None,
        &ProductSourceArgs {
            tokenizer_source: Some(separate_tokenizer.clone()),
            ..Default::default()
        },
    )
    .await
    .unwrap()
    .into_product_engine_input();
    let sources = product.model_sources.unwrap();
    assert_eq!(sources.weights().path(), weights.canonicalize().unwrap());
    assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
    assert_eq!(
        sources.tokenizer_root(),
        separate_tokenizer.canonicalize().unwrap()
    );
}

#[tokio::test]
async fn recipe_never_accepts_another_cached_weight_or_metadata_revision() {
    for move_weights in [true, false] {
        let cache = tempfile::tempdir().unwrap();
        let (weights, semantic) = cached_recipe(cache.path());
        let original = if move_weights {
            weights.parent().unwrap()
        } else {
            &semantic
        };
        let snapshots = original.parent().unwrap();
        let wrong_revision = "a".repeat(40);
        std::fs::rename(original, snapshots.join(&wrong_revision)).unwrap();
        let repository = snapshots.parent().unwrap();
        std::fs::create_dir_all(repository.join("refs")).unwrap();
        std::fs::write(repository.join("refs/main"), &wrong_revision).unwrap();
        let error = resolve_model_source_with_product_sources(
            "bonsai2:27b",
            cache.path(),
            DownloadPolicy::NoDownload,
            None,
            &ProductSourceArgs::default(),
        )
        .await
        .err()
        .expect("a different revision must not satisfy an immutable recipe");
        assert!(error.to_string().contains("NoDownload"), "{error}");
    }
}
