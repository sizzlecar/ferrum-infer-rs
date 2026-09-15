//! Resolve explicit metadata roles without changing the physical weight source.

use super::*;

const SEMANTIC_FILES: &[&str] = &["config.json", "generation_config.json"];
const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
];

enum Selector<'a> {
    Local(&'a Path),
    Repository(PinnedHfRepository<'a>),
}

impl<'a> Selector<'a> {
    fn parse(path: &'a Path) -> Result<Self> {
        // Preserve existing local paths, including names containing '@'. Explicit
        // absolute/relative paths never become repository requests accidentally.
        if path.exists() || path.is_absolute() || path.starts_with(".") || path.starts_with("..") {
            return Ok(Self::Local(path));
        }
        match path.to_str().map(parse_pinned_hf_repository).transpose()? {
            Some(Some(pin)) => Ok(Self::Repository(pin)),
            _ => Ok(Self::Local(path)),
        }
    }
}

pub(super) struct SelectedSources {
    pub arguments: ProductSourceArgs,
    pub semantic_original: Option<OriginalModelSource>,
    pub tokenizer_original: Option<OriginalModelSource>,
    pub downloaded: bool,
}

fn ready(root: &Path, files: &[&str]) -> Result<bool> {
    if ferrum_models::hf_download::inspect_cached_metadata_selection(root, files)?.is_some() {
        return Ok(false);
    }
    for required in ["config.json", "tokenizer.json"] {
        if !files.contains(&required) {
            continue;
        }
        match std::fs::metadata(root.join(required)) {
            Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {}
            Ok(_) => return Ok(false),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(true)
}

async fn resolve_one(
    selector: Selector<'_>,
    files: &[&str],
    cache_dir: &Path,
    download: DownloadPolicy,
) -> Result<(PathBuf, OriginalModelSource, bool)> {
    let pin = match selector {
        Selector::Local(path) => {
            return Ok((
                path.to_path_buf(),
                OriginalModelSource {
                    kind: if path.is_file() {
                        ModelSourceKind::LocalFile
                    } else {
                        ModelSourceKind::LocalDirectory
                    },
                    location: path.display().to_string(),
                    requested_revision: None,
                },
                false,
            ));
        }
        Selector::Repository(pin) => pin,
    };
    let mut root = cache_dir
        .join("hub")
        .join(format!("models--{}", pin.repo_id.replace('/', "--")))
        .join("snapshots")
        .join(&pin.revision);
    let downloaded = !ready(&root, files)?;
    if downloaded {
        if download == DownloadPolicy::NoDownload {
            return Err(FerrumError::model(format!(
                "metadata source {}@{} is not complete in cache and DownloadPolicy::NoDownload is set",
                pin.repo_id, pin.revision
            )));
        }
        let token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();
        root = ferrum_models::HfDownloader::new(cache_dir.to_path_buf(), token)?
            .download_sidecar_files(pin.repo_id, Some(&pin.revision), files)
            .await?;
    }
    pin.verify_snapshot(&root)?;
    if !ready(&root, files)? {
        return Err(FerrumError::model(format!(
            "pinned metadata source {}@{} did not provide its required files",
            pin.repo_id, pin.revision
        )));
    }
    Ok((
        root,
        OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: pin.repo_id.to_owned(),
            requested_revision: Some(pin.revision),
        },
        downloaded,
    ))
}

pub(super) async fn resolve(
    args: &ProductSourceArgs,
    cache_dir: &Path,
    download: DownloadPolicy,
) -> Result<SelectedSources> {
    // Validate both selectors before any download, keeping invalid explicit
    // configuration distinct from an ordinary cold cache.
    let semantic = args
        .semantic_source
        .as_deref()
        .map(Selector::parse)
        .transpose()?;
    let tokenizer = args
        .tokenizer_source
        .as_deref()
        .map(Selector::parse)
        .transpose()?;
    let mut selected = SelectedSources {
        arguments: args.clone(),
        semantic_original: None,
        tokenizer_original: None,
        downloaded: false,
    };
    if let Some(semantic) = semantic {
        let files = if tokenizer.is_some() {
            &SEMANTIC_FILES[..]
        } else {
            &PRODUCT_SOURCE_FILES[..]
        };
        let (root, original, downloaded) =
            resolve_one(semantic, files, cache_dir, download).await?;
        selected.arguments.semantic_source = Some(root);
        selected.semantic_original = Some(original);
        selected.downloaded |= downloaded;
    }
    if let Some(tokenizer) = tokenizer {
        let (root, original, downloaded) =
            resolve_one(tokenizer, TOKENIZER_FILES, cache_dir, download).await?;
        selected.arguments.tokenizer_source = Some(root);
        selected.tokenizer_original = Some(original);
        selected.downloaded |= downloaded;
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(cache: &Path, repo: &str, revision: &str) -> PathBuf {
        let root = cache
            .join("hub")
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn explicit_selector_preserves_local_paths_and_rejects_moving_revisions() {
        let temporary = tempfile::tempdir().unwrap();
        let local = temporary.path().join("local@branch");
        std::fs::create_dir(&local).unwrap();
        assert!(matches!(
            Selector::parse(&local).unwrap(),
            Selector::Local(_)
        ));
        for invalid in [
            "owner/repo@main",
            "owner/repo@v1",
            "owner/repo@",
            "alias@abcdef",
        ] {
            assert!(Selector::parse(Path::new(invalid)).is_err(), "{invalid}");
        }
        let pinned = format!("owner/repo@{}", "A".repeat(40));
        let Selector::Repository(pin) = Selector::parse(Path::new(&pinned)).unwrap() else {
            panic!("pinned repository was interpreted as a path")
        };
        assert_eq!(pin.revision, "a".repeat(40));
    }

    #[tokio::test]
    async fn explicit_metadata_pin_never_falls_back_to_cached_main() {
        let cache = tempfile::tempdir().unwrap();
        let repo = "Original/Checkpoint";
        let old = "a".repeat(40);
        let wanted = "b".repeat(40);
        let old_root = snapshot(cache.path(), repo, &old);
        std::fs::write(old_root.join("config.json"), b"{}").unwrap();
        std::fs::write(old_root.join("tokenizer.json"), b"{}").unwrap();
        let repo_root = old_root.parent().unwrap().parent().unwrap();
        std::fs::create_dir(repo_root.join("refs")).unwrap();
        std::fs::write(repo_root.join("refs/main"), &old).unwrap();
        let args = ProductSourceArgs {
            semantic_source: Some(format!("{repo}@{wanted}").into()),
            ..Default::default()
        };
        assert!(resolve(&args, cache.path(), DownloadPolicy::NoDownload)
            .await
            .is_err());
        let root = snapshot(cache.path(), repo, &wanted);
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        assert!(resolve(&args, cache.path(), DownloadPolicy::NoDownload)
            .await
            .is_err());
        std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        let selected = resolve(&args, cache.path(), DownloadPolicy::NoDownload)
            .await
            .unwrap();
        assert_eq!(
            selected.arguments.semantic_source.as_deref(),
            Some(root.as_path())
        );
        assert!(!selected.downloaded);
        let original = selected.semantic_original.unwrap();
        assert_eq!(original.kind, ModelSourceKind::Repository);
        assert_eq!(original.location, repo);
        assert_eq!(
            original.requested_revision.as_deref(),
            Some(wanted.as_str())
        );
        assert_eq!(
            std::fs::read_to_string(repo_root.join("refs/main")).unwrap(),
            old
        );
    }

    #[tokio::test]
    async fn independently_pinned_roles_do_not_require_each_others_files() {
        let cache = tempfile::tempdir().unwrap();
        let revision = "c".repeat(40);
        let semantic = snapshot(cache.path(), "Original/Semantic", &revision);
        let tokenizer = snapshot(cache.path(), "Original/Tokenizer", &revision);
        std::fs::write(semantic.join("config.json"), b"{}").unwrap();
        std::fs::write(tokenizer.join("tokenizer.json"), b"{}").unwrap();
        let args = ProductSourceArgs {
            semantic_source: Some(format!("Original/Semantic@{revision}").into()),
            tokenizer_source: Some(format!("Original/Tokenizer@{revision}").into()),
            ..Default::default()
        };
        let selected = resolve(&args, cache.path(), DownloadPolicy::NoDownload)
            .await
            .unwrap();
        assert_eq!(
            selected.arguments.semantic_source.as_deref(),
            Some(semantic.as_path())
        );
        assert_eq!(
            selected.arguments.tokenizer_source.as_deref(),
            Some(tokenizer.as_path())
        );
        assert_eq!(
            selected.semantic_original.unwrap().location,
            "Original/Semantic"
        );
        assert_eq!(
            selected.tokenizer_original.unwrap().location,
            "Original/Tokenizer"
        );
    }

    #[tokio::test]
    async fn pinned_metadata_composition_retains_repository_identity_and_weight_selection() {
        let cache = tempfile::tempdir().unwrap();
        let weights_revision = "a".repeat(40);
        let metadata_revision = "b".repeat(40);
        let weights = snapshot(cache.path(), "Quantizer/Checkpoint", &weights_revision);
        let metadata = snapshot(cache.path(), "Original/Checkpoint", &metadata_revision);
        let file = weights.join("model-IQ4_XS.gguf");
        super::super::gguf_repository::tests::write_iq_fixture(&file, &[]);
        std::fs::write(
            metadata.join("config.json"),
            super::super::tests::qwen35_semantic_config(false),
        )
        .unwrap();
        std::fs::write(metadata.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(
            metadata.join("tokenizer_config.json"),
            br#"{"chat_template":"fixture"}"#,
        )
        .unwrap();
        let args = ProductSourceArgs {
            gguf_file: Some("model-IQ4_XS.gguf".into()),
            semantic_source: Some(format!("Original/Checkpoint@{metadata_revision}").into()),
            ..Default::default()
        };
        let product = resolve_model_source_with_product_sources(
            &format!("Quantizer/Checkpoint@{weights_revision}"),
            cache.path(),
            DownloadPolicy::NoDownload,
            None,
            &args,
        )
        .await
        .unwrap()
        .into_product_engine_input();
        let sources = product.model_sources.unwrap();
        assert_eq!(sources.weights().path(), file.canonicalize().unwrap());
        assert_eq!(sources.semantic_root(), metadata.canonicalize().unwrap());
        assert_eq!(sources.tokenizer_root(), metadata.canonicalize().unwrap());
        for source in [
            &sources.original_sources().semantic,
            &sources.original_sources().tokenizer,
        ] {
            assert_eq!(source.kind, ModelSourceKind::Repository);
            assert_eq!(source.location, "Original/Checkpoint");
            assert_eq!(
                source.requested_revision.as_deref(),
                Some(metadata_revision.as_str())
            );
        }
        assert_eq!(
            sources.original_sources().weights.location,
            "Quantizer/Checkpoint"
        );
        assert_eq!(
            sources
                .original_sources()
                .weights
                .requested_revision
                .as_deref(),
            Some(weights_revision.as_str())
        );
        assert_eq!(
            sources.resolved_sources().semantic.resolved_revision,
            metadata_revision
        );
        assert_eq!(
            sources.resolved_sources().tokenizer.resolved_revision,
            metadata_revision
        );
    }
}
