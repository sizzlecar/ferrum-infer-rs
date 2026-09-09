use super::*;
use crate::source_resolver::{
    resolve_model_source, resolve_model_source_with_product_sources, DownloadPolicy,
    ProductSourceArgs,
};
use serde_json::json;
use std::fs;
use tempfile::{tempdir, TempDir};

const REPO: &str = "fixture/cache-selection";
const CURRENT: &str = "1111111111111111111111111111111111111111";
const OTHER: &str = "2222222222222222222222222222222222222222";

struct CacheFixture {
    directory: TempDir,
    model_dir: PathBuf,
}

impl CacheFixture {
    fn new() -> Self {
        let directory = tempdir().unwrap();
        let model_dir = directory
            .path()
            .join("hub/models--fixture--cache-selection");
        Self {
            directory,
            model_dir,
        }
    }

    fn root(&self) -> &Path {
        self.directory.path()
    }

    fn snapshot(&self, revision: &str) -> SnapshotFixture {
        let path = self.model_dir.join("snapshots").join(revision);
        fs::create_dir_all(&path).unwrap();
        SnapshotFixture { path }
    }

    fn select(&self, revision: &str) {
        fs::create_dir_all(self.model_dir.join("refs")).unwrap();
        fs::write(self.model_dir.join("refs/main"), revision).unwrap();
    }

    fn record_optional_template(&self, revision: &str) {
        let inventory = self.model_dir.join(".ferrum-downloads").join(revision);
        fs::create_dir_all(&inventory).unwrap();
        fs::write(
            inventory.join("sidecars-fixture.json"),
            serde_json::to_vec(&json!({
                "version": 1,
                "files": {"chat_template.jinja": 8}
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

struct SnapshotFixture {
    path: PathBuf,
}

impl SnapshotFixture {
    fn complete_weights(self) -> Self {
        fs::write(self.path.join("model.safetensors"), b"fixture-weights").unwrap();
        self
    }

    fn missing_shard(self) -> Self {
        fs::write(
            self.path.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "weight_map": {"weight": "missing.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        self
    }

    fn metadata(self) -> Self {
        write_semantic(&self.path);
        write_tokenizer(&self.path);
        self
    }
}

fn write_semantic(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::write(
        path.join("config.json"),
        serde_json::to_vec(&json!({"architectures": ["Qwen3ForCausalLM"]})).unwrap(),
    )
    .unwrap();
}

fn write_tokenizer(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::write(
        path.join("tokenizer.json"),
        serde_json::to_vec(&json!({"version": "1.0"})).unwrap(),
    )
    .unwrap();
    fs::write(
        path.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"chat_template": "fixture-template"})).unwrap(),
    )
    .unwrap();
}

#[tokio::test]
async fn a_single_cached_gguf_is_the_shared_product_weight_source() {
    let cache = CacheFixture::new();
    let snapshot = cache.snapshot(CURRENT).metadata();
    let file = snapshot.path.join("model.gguf");
    fs::write(&file, b"nonempty GGUF source fixture").unwrap();
    cache.select(CURRENT);
    assert!(model_cache_ready(&cache.model_dir));
    for request in [REPO.to_owned(), format!("{REPO}@{CURRENT}")] {
        // Product resolution is the common run/serve path and previously
        // required SafeTensors despite list reporting this snapshot as ready.
        let product = resolve_model_source_with_product_sources(
            &request,
            cache.root(),
            DownloadPolicy::NoDownload,
            None,
            &ProductSourceArgs::default(),
        )
        .await
        .unwrap()
        .into_product_engine_input();
        assert_eq!(product.source.local_path, file);
        assert_eq!(product.source.format, ModelFormat::GGUF);
        assert!(product.source.from_cache);
        let sources = product.model_sources.unwrap();
        assert_eq!(sources.weights().path(), file.canonicalize().unwrap());
        assert_eq!(sources.original_sources().weights.location, REPO);
    }
}

#[tokio::test]
async fn cached_legacy_gguf_does_not_require_an_invented_metadata_repository() {
    let cache = CacheFixture::new();
    let snapshot = cache.snapshot(CURRENT);
    let file = snapshot.path.join("model.gguf");
    crate::source_resolver::gguf_repository::tests::write_metadata_fixture(&file, "qwen3", &[]);
    cache.select(CURRENT);
    let product = resolve_model_source_with_product_sources(
        REPO,
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
        &ProductSourceArgs::default(),
    )
    .await
    .unwrap()
    .into_product_engine_input();
    assert_eq!(product.source.local_path, file);
    assert!(product.model_sources.is_none());
}

#[test]
fn ambiguous_and_empty_gguf_caches_are_not_ready() {
    let cache = CacheFixture::new();
    let snapshot = cache.snapshot(CURRENT);
    cache.select(CURRENT);
    let first = snapshot.path.join("a.gguf");
    let second = snapshot.path.join("b.gguf");
    fs::write(&first, []).unwrap();
    assert!(!model_cache_ready(&cache.model_dir));
    fs::write(&first, b"first").unwrap();
    fs::write(&second, b"second").unwrap();
    assert!(!model_cache_ready(&cache.model_dir));
    let error = inspect_cached_model(cache.root(), REPO, CacheRequirements::Product)
        .err()
        .expect("ambiguous selection must be explicit")
        .to_string();
    assert!(
        error.contains("a.gguf") && error.contains("b.gguf"),
        "{error}"
    );
    fs::remove_file(second).unwrap();
    assert!(model_cache_ready(&cache.model_dir));
}

#[cfg(unix)]
#[test]
fn dangling_gguf_link_is_not_a_ready_cache() {
    let cache = CacheFixture::new();
    let snapshot = cache.snapshot(CURRENT);
    std::os::unix::fs::symlink("missing-blob", snapshot.path.join("model.gguf")).unwrap();
    cache.select(CURRENT);
    assert!(!model_cache_ready(&cache.model_dir));
    assert!(matches!(
        inspect_cached_model(cache.root(), REPO, CacheRequirements::Product).unwrap(),
        CachedModel::Missing(_)
    ));
}

#[tokio::test]
async fn local_gguf_directory_rejects_empty_and_ambiguous_weights() {
    let cache = CacheFixture::new();
    let local = cache.root().join("local-model");
    fs::create_dir(&local).unwrap();
    let first = local.join("a.gguf");
    let second = local.join("b.gguf");
    fs::write(&first, []).unwrap();

    let error = resolve_model_source(
        local.to_str().unwrap(),
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
    )
    .await
    .err()
    .expect("empty GGUF files are not usable weights")
    .to_string();
    assert!(error.contains("no supported weights"), "{error}");

    fs::write(&first, b"first").unwrap();
    fs::write(&second, b"second").unwrap();
    let error = resolve_model_source(
        local.to_str().unwrap(),
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
    )
    .await
    .err()
    .expect("multiple quantizations require an explicit file selection")
    .to_string();
    assert!(error.contains("explicit .gguf path"), "{error}");
    assert!(
        error.contains("a.gguf") && error.contains("b.gguf"),
        "{error}"
    );
    assert!(!cache.root().join("hub").exists());
}

#[tokio::test]
async fn explicit_tokenizer_with_gguf_preserves_colocated_semantics() {
    let cache = CacheFixture::new();
    let snapshot = cache.snapshot(CURRENT);
    write_semantic(&snapshot.path);
    let file = snapshot.path.join("model.gguf");
    crate::source_resolver::gguf_repository::tests::write_metadata_fixture(&file, "qwen35", &[]);
    cache.select(CURRENT);
    let tokenizer = cache.root().join("external-tokenizer");
    write_tokenizer(&tokenizer);
    let request = format!("{REPO}@{CURRENT}");
    for (model, revision) in [
        (request.as_str(), Some(CURRENT)),
        (snapshot.path.to_str().unwrap(), None),
    ] {
        let product = resolve_model_source_with_product_sources(
            model,
            cache.root(),
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
        let sources = product.model_sources.unwrap();
        assert_eq!(sources.weights().path(), file.canonicalize().unwrap());
        assert_eq!(
            sources.semantic_root(),
            snapshot.path.canonicalize().unwrap()
        );
        assert_eq!(sources.tokenizer_root(), tokenizer.canonicalize().unwrap());
        assert_eq!(
            sources
                .original_sources()
                .semantic
                .requested_revision
                .as_deref(),
            revision
        );
    }
    assert!(!snapshot.path.join("tokenizer.json").exists());
}

#[test]
fn referenced_incomplete_snapshot_does_not_select_another_revision_or_list_ready() {
    let cache = CacheFixture::new();
    let selected = cache.snapshot(CURRENT).missing_shard().metadata();
    cache.snapshot(OTHER).complete_weights().metadata();
    cache.select(CURRENT);

    let result = inspect_cached_model(cache.root(), REPO, CacheRequirements::Product).unwrap();
    let CachedModel::Missing(Some(reason)) = result else {
        panic!("the referenced snapshot must be repaired before it is selectable");
    };
    assert!(reason.contains("missing.safetensors"), "{reason}");
    assert!(!model_cache_ready(&cache.model_dir));

    fs::write(selected.path.join("missing.safetensors"), b"recovered").unwrap();
    let CachedModel::Ready(source) =
        inspect_cached_model(cache.root(), REPO, CacheRequirements::Product).unwrap()
    else {
        panic!("the recovered referenced snapshot should be selectable");
    };
    assert_eq!(source.local_path, selected.path);
    assert!(model_cache_ready(&cache.model_dir));
}

#[test]
fn unreferenced_cache_can_select_a_complete_fallback() {
    let cache = CacheFixture::new();
    cache.snapshot(CURRENT).missing_shard().metadata();
    let complete = cache.snapshot(OTHER).complete_weights().metadata();

    let CachedModel::Ready(source) =
        inspect_cached_model(cache.root(), REPO, CacheRequirements::Product).unwrap()
    else {
        panic!("without a ref, a complete cached snapshot is usable");
    };
    assert_eq!(source.local_path, complete.path);
    assert!(model_cache_ready(&cache.model_dir));
}

#[tokio::test]
async fn invalid_selected_index_keeps_its_error_instead_of_becoming_a_download_miss() {
    let cache = CacheFixture::new();
    let selected = cache.snapshot(CURRENT).metadata();
    fs::write(
        selected.path.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "weight_map": {"weight": "../outside.safetensors"}
        }))
        .unwrap(),
    )
    .unwrap();
    cache.snapshot(OTHER).complete_weights().metadata();
    cache.select(CURRENT);

    assert!(inspect_cached_model(cache.root(), REPO, CacheRequirements::Product).is_err());
    let error = resolve_model_source(REPO, cache.root(), DownloadPolicy::NoDownload, None)
        .await
        .err()
        .expect("invalid referenced metadata is an error even with another complete snapshot")
        .to_string();
    assert!(error.contains("Invalid shard path"), "{error}");
    assert!(!error.contains("NoDownload"), "{error}");
    assert!(!model_cache_ready(&cache.model_dir));
}

#[tokio::test]
async fn product_no_download_reports_missing_metadata_and_list_is_not_ready() {
    let cache = CacheFixture::new();
    let selected = cache.snapshot(CURRENT).complete_weights();
    write_semantic(&selected.path);
    cache.select(CURRENT);

    let error = resolve_model_source_with_product_sources(
        REPO,
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
        &ProductSourceArgs::default(),
    )
    .await
    .err()
    .expect("offline product resolution must explain the missing tokenizer")
    .to_string();
    assert!(error.contains("tokenizer.json"), "{error}");
    assert!(error.contains("NoDownload"), "{error}");
    assert!(!model_cache_ready(&cache.model_dir));

    write_tokenizer(&selected.path);
    assert!(model_cache_ready(&cache.model_dir));
}

#[tokio::test]
async fn recorded_optional_sidecar_prevents_a_false_ready_product_cache() {
    let cache = CacheFixture::new();
    let selected = cache.snapshot(CURRENT).complete_weights().metadata();
    cache.select(CURRENT);
    cache.record_optional_template(CURRENT);

    let error = resolve_model_source_with_product_sources(
        REPO,
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
        &ProductSourceArgs::default(),
    )
    .await
    .err()
    .expect("an interrupted selected template must not silently change product behavior")
    .to_string();
    assert!(error.contains("chat_template.jinja"), "{error}");
    assert!(error.contains("NoDownload"), "{error}");
    assert!(!model_cache_ready(&cache.model_dir));

    fs::write(selected.path.join("chat_template.jinja"), b"template").unwrap();
    assert!(model_cache_ready(&cache.model_dir));
}

#[tokio::test]
async fn existing_local_directory_without_weights_remains_a_local_error() {
    let cache = CacheFixture::new();
    let local = cache.root().join("local-model");
    write_semantic(&local);

    let error = resolve_model_source(
        local.to_str().unwrap(),
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
    )
    .await
    .err()
    .expect("an explicit local directory must not be reinterpreted as a remote repository")
    .to_string();
    assert!(error.contains("local model directory"), "{error}");
    assert!(error.contains("no supported weights"), "{error}");
    assert!(!error.contains("NoDownload"), "{error}");
    assert!(!cache.root().join("hub").exists());
}

#[tokio::test]
async fn explicit_tokenizer_allows_cached_weights_without_colocated_tokenizer() {
    let cache = CacheFixture::new();
    let weights = cache.snapshot(CURRENT).complete_weights();
    write_semantic(&weights.path);
    cache.select(CURRENT);
    cache.record_optional_template(CURRENT);
    let tokenizer = cache.root().join("explicit-tokenizer");
    write_tokenizer(&tokenizer);

    let resolved = resolve_model_source_with_product_sources(
        REPO,
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
        &ProductSourceArgs {
            semantic_source: None,
            tokenizer_source: Some(tokenizer.clone()),
        },
    )
    .await
    .unwrap();
    let product = resolved.into_product_engine_input();
    let sources = product.model_sources.unwrap();
    assert!(product.source.from_cache);
    assert_eq!(
        sources.weights().path(),
        weights.path.canonicalize().unwrap()
    );
    assert_eq!(
        sources.semantic_root(),
        weights.path.canonicalize().unwrap()
    );
    assert_eq!(sources.tokenizer_root(), tokenizer.canonicalize().unwrap());
    assert!(!weights.path.join("tokenizer.json").exists());
}

#[tokio::test]
async fn tokenizer_override_preserves_the_pinned_repository_for_unchanged_roles() {
    let cache = CacheFixture::new();
    cache.snapshot(CURRENT).complete_weights().metadata();
    cache.select(CURRENT);
    let tokenizer = cache.root().join("external-tokenizer");
    write_tokenizer(&tokenizer);
    let requested = format!("{REPO}@{CURRENT}");

    let product = resolve_model_source_with_product_sources(
        &requested,
        cache.root(),
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
    let sources = product.model_sources.unwrap();
    let originals = sources.original_sources();
    for retained in [&originals.semantic, &originals.weights] {
        assert_eq!(
            retained.kind,
            ferrum_interfaces::vnext::ModelSourceKind::Repository
        );
        assert_eq!(retained.location, REPO);
        assert_eq!(retained.requested_revision.as_deref(), Some(CURRENT));
    }
    assert_eq!(
        originals.tokenizer.kind,
        ferrum_interfaces::vnext::ModelSourceKind::LocalDirectory
    );
    assert_eq!(
        originals.tokenizer.location,
        tokenizer.display().to_string()
    );
    assert!(originals.tokenizer.requested_revision.is_none());
}

#[tokio::test]
async fn explicit_semantics_allow_cached_weights_without_colocated_metadata() {
    let cache = CacheFixture::new();
    let weights = cache.snapshot(CURRENT).complete_weights();
    cache.select(CURRENT);
    cache.record_optional_template(CURRENT);
    let semantic = cache.root().join("explicit-semantics");
    write_semantic(&semantic);
    write_tokenizer(&semantic);

    let resolved = resolve_model_source_with_product_sources(
        REPO,
        cache.root(),
        DownloadPolicy::NoDownload,
        None,
        &ProductSourceArgs {
            semantic_source: Some(semantic.clone()),
            tokenizer_source: None,
        },
    )
    .await
    .unwrap();
    let product = resolved.into_product_engine_input();
    let sources = product.model_sources.unwrap();
    assert!(product.source.from_cache);
    assert_eq!(
        sources.weights().path(),
        weights.path.canonicalize().unwrap()
    );
    assert_eq!(sources.semantic_root(), semantic.canonicalize().unwrap());
    assert_eq!(sources.tokenizer_root(), semantic.canonicalize().unwrap());
    assert!(!weights.path.join("config.json").exists());
    assert!(!weights.path.join("tokenizer.json").exists());
}

#[test]
fn list_accepts_complete_bpe_or_sentencepiece_assets_without_tokenizer_json() {
    let cache = CacheFixture::new();
    let selected = cache.snapshot(CURRENT).complete_weights();
    write_semantic(&selected.path);
    cache.select(CURRENT);
    assert!(!model_cache_ready(&cache.model_dir));

    let vocab = selected.path.join("vocab.json");
    let merges = selected.path.join("merges.txt");
    fs::write(&vocab, b"{}").unwrap();
    assert!(!model_cache_ready(&cache.model_dir));
    fs::write(&merges, b"").unwrap();
    assert!(!model_cache_ready(&cache.model_dir));
    fs::write(&merges, b"a b").unwrap();
    assert!(model_cache_ready(&cache.model_dir));
    fs::remove_file(vocab).unwrap();
    assert!(!model_cache_ready(&cache.model_dir));
    fs::remove_file(merges).unwrap();

    let sentencepiece = selected.path.join("tokenizer.model");
    fs::create_dir(&sentencepiece).unwrap();
    assert!(!model_cache_ready(&cache.model_dir));
    fs::remove_dir(&sentencepiece).unwrap();
    fs::write(&sentencepiece, b"").unwrap();
    assert!(!model_cache_ready(&cache.model_dir));
    fs::write(sentencepiece, b"fixture-sentencepiece").unwrap();
    assert!(model_cache_ready(&cache.model_dir));
    assert!(!selected.path.join("tokenizer.json").exists());
}

#[test]
fn gguf_metadata_lookup_projects_recorded_selection_and_preserves_inventory_errors() {
    let cache = CacheFixture::new();
    let selected = cache.snapshot(CURRENT).metadata();
    cache.select(CURRENT);
    let inventory_dir = cache.model_dir.join(".ferrum-downloads").join(CURRENT);
    fs::create_dir_all(&inventory_dir).unwrap();
    let inventory = inventory_dir.join("sidecars-fixture.json");
    fs::write(
        &inventory,
        serde_json::to_vec(&json!({
            "version": 1,
            "files": {"preprocessor_config.json": 4}
        }))
        .unwrap(),
    )
    .unwrap();

    assert_eq!(
        super::super::find_cached_product_metadata(cache.root(), REPO).unwrap(),
        Some(selected.path.clone())
    );
    assert!(!selected.path.join("preprocessor_config.json").exists());

    fs::write(&inventory, b"{").unwrap();
    let error = super::super::find_cached_product_metadata(cache.root(), REPO)
        .expect_err("invalid inventory JSON cannot be reinterpreted as a cache miss");
    assert!(
        error.to_string().contains("Invalid metadata inventory"),
        "{error}"
    );

    fs::write(
        &inventory,
        serde_json::to_vec(&json!({
            "version": 1,
            "files": {
                "preprocessor_config.json": 4,
                "chat_template.jinja": 8
            }
        }))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        super::super::find_cached_product_metadata(cache.root(), REPO).unwrap(),
        None
    );
    fs::write(selected.path.join("chat_template.jinja"), b"template").unwrap();
    assert_eq!(
        super::super::find_cached_product_metadata(cache.root(), REPO).unwrap(),
        Some(selected.path)
    );
}
