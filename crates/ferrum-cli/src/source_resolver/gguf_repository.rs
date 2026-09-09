//! Compose a selected GGUF repository file with its independent metadata.

use super::*;
use ferrum_quantization::gguf::GgufModelMetadata;

pub(super) mod selection;

#[cfg(test)]
pub(super) mod tests;

pub(super) fn read_metadata(path: &Path) -> Result<GgufModelMetadata> {
    let mut file = std::fs::File::open(path)?;
    GgufModelMetadata::read(&mut file).map_err(|error| {
        FerrumError::model(format!(
            "cannot read GGUF metadata '{}': {error}",
            path.display()
        ))
    })
}

fn declared_metadata_repository(metadata: &GgufModelMetadata) -> Result<Option<String>> {
    // Conversion provenance names the immediate source; a single parent is a
    // fallback when that declaration is absent. Never pick one parent of a merge.
    let url = if let Some(url) = &metadata.source_repository_url {
        Some(url)
    } else {
        if metadata.base_model_count.is_some_and(|count| count > 1) {
            return Err(FerrumError::model(
                "GGUF declares multiple base models; provide --semantic-source DIR",
            ));
        }
        metadata.base_model_repository_urls.get(&0)
    };
    url.map(|url| huggingface_repository_url(url)).transpose()
}

fn huggingface_repository_url(value: &str) -> Result<String> {
    let invalid = || {
        FerrumError::model(
        "GGUF semantic/tokenizer provenance must name an HTTPS Hugging Face model repository; provide --semantic-source DIR",
    )
    };
    let url = reqwest::Url::parse(value).map_err(|_| invalid())?;
    if url.scheme() != "https"
        || url.host_str() != Some("huggingface.co")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid());
    }
    let repo = url
        .path()
        .strip_prefix('/')
        .ok_or_else(invalid)?
        .trim_end_matches('/');
    let valid = |part: &str| {
        !part.is_empty()
            && !part.starts_with(['.', '-'])
            && !part.ends_with(['.', '-'])
            && !part.contains("..")
            && !part.contains("--")
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    };
    if !repo
        .split_once('/')
        .is_some_and(|(owner, name)| valid(owner) && valid(name))
    {
        return Err(invalid());
    }
    Ok(repo.to_owned())
}

pub(super) async fn resolve_metadata(
    requested_model: &str,
    weights_original: OriginalModelSource,
    local_path: &Path,
    cache_dir: &Path,
    download: DownloadPolicy,
    tokenizer_override: Option<&Path>,
) -> Result<(Option<Arc<ProductionModelSourceBundle>>, bool)> {
    // An explicit tokenizer leaves the semantic role intact. It must not force
    // downloading or validating a tokenizer that the caller will replace.
    if let Some(tokenizer) = tokenizer_override {
        let root = local_path.parent().unwrap_or_else(|| Path::new("."));
        if semantic_ready(root)? {
            return compose(
                root,
                tokenizer,
                weights_original.clone(),
                weights_original,
                local_path,
                true,
            );
        }
    }
    let repo = &weights_original.location;
    let metadata_repo = declared_metadata_repository(&read_metadata(local_path)?)?
        .or_else(|| tokenizer_sibling_repo(repo)).ok_or_else(|| {
        FerrumError::model(format!(
            "GGUF repository '{repo}' has no semantic/tokenizer source; provide --semantic-source DIR"
        ))
    })?;
    let cached = if tokenizer_override.is_some() {
        let directory = cache_dir
            .join("hub")
            .join(format!("models--{}", metadata_repo.replace('/', "--")));
        let mut found = None;
        for root in cache::snapshot_candidates(&directory)? {
            if semantic_ready(&root)? {
                found = Some(root);
                break;
            }
        }
        found
    } else {
        find_cached_product_metadata(cache_dir, &metadata_repo)?
    };
    let files = if tokenizer_override.is_some() {
        &SEMANTIC_FILES[..]
    } else {
        &PRODUCT_SOURCE_FILES[..]
    };
    let (metadata_root, metadata_from_cache) = match cached {
            Some(path) => (path, true),
            None if download == DownloadPolicy::AutoDownload => {
                let token = std::env::var("HF_TOKEN")
                    .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                    .ok();
                let downloader = ferrum_models::HfDownloader::new(cache_dir.to_path_buf(), token)?;
                let path = downloader
                    .download_sidecar_files(&metadata_repo, None, files)
                    .await?;
                let ready = if tokenizer_override.is_some() { semantic_ready(&path)? } else { is_complete_product_metadata_snapshot(&path)? };
                if !ready {
                    return Err(FerrumError::model(format!(
                        "semantic/tokenizer source '{metadata_repo}' did not provide the required metadata"
                    )));
                }
                (path, false)
            }
            None => {
                return Err(FerrumError::model(format!(
                    "semantic/tokenizer source '{metadata_repo}' for GGUF '{requested_model}' is not cached and DownloadPolicy::NoDownload is set"
                )))
            }
        };
    let metadata_original = repository_source(metadata_repo);
    compose(
        &metadata_root,
        tokenizer_override.unwrap_or(&metadata_root),
        metadata_original,
        weights_original,
        local_path,
        metadata_from_cache,
    )
}

const SEMANTIC_FILES: [&str; 2] = ["config.json", "generation_config.json"];

fn semantic_ready(root: &Path) -> Result<bool> {
    let config = root.join("config.json");
    if !config.is_file() || std::fs::metadata(&config)?.len() == 0 {
        return Ok(false);
    }
    Ok(
        ferrum_models::hf_download::inspect_cached_metadata_selection(root, &SEMANTIC_FILES)?
            .is_none(),
    )
}

fn compose(
    metadata_root: &Path,
    tokenizer_root: &Path,
    metadata_original: OriginalModelSource,
    weights_original: OriginalModelSource,
    local_path: &Path,
    metadata_from_cache: bool,
) -> Result<(Option<Arc<ProductionModelSourceBundle>>, bool)> {
    let tokenizer_original = if tokenizer_root == metadata_root {
        metadata_original.clone()
    } else {
        OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location: tokenizer_root.display().to_string(),
            requested_revision: None,
        }
    };
    let sources = Arc::new(open_registered_product_sources(
        metadata_root,
        tokenizer_root,
        ProductionWeightArtifact::gguf_file(local_path),
        OriginalModelSources {
            semantic: metadata_original.clone(),
            tokenizer: tokenizer_original,
            weights: weights_original,
        },
    )?);
    Ok((Some(sources), metadata_from_cache))
}
