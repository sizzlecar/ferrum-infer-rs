//! One explicit repository artifact, preserving independent metadata roles.

use super::*;

#[cfg(test)]
mod tests;

pub(crate) async fn resolve(
    model: &str,
    filename: &str,
    cache_dir: &Path,
    download: DownloadPolicy,
    autosize: Option<(AutoSizeProfile, f32)>,
    source_args: &ProductSourceArgs,
) -> Result<Resolved> {
    ferrum_models::hf_download::validate_gguf_filename(filename)?;
    let pin = parse_pinned_hf_repository(model)?;
    let repo = pin.as_ref().map_or(model, |pin| pin.repo_id);
    // Reuse the repository URL contract; local paths and aliases do not name
    // the independent repository selected by this option.
    let validated_repo = huggingface_repository_url(&format!("https://huggingface.co/{repo}"))?;
    if validated_repo != repo {
        return Err(FerrumError::config(
            "--gguf-file requires an explicit owner/repository",
        ));
    }
    let repo_dir = cache_dir
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = match &pin {
        Some(pin) => vec![repo_dir.join("snapshots").join(&pin.revision)],
        None => cache::snapshot_candidates(&repo_dir)?,
    };
    let cached = snapshots
        .into_iter()
        .map(|root| root.join(filename))
        .find(|path| {
            matches!(
                ferrum_models::source::inspect_cached_weights(path),
                Ok(ferrum_models::source::CachedWeights::Ready(
                    ModelFormat::GGUF
                ))
            )
        });
    let revision = pin.as_ref().map(|pin| pin.revision.as_str());
    let (local_path, from_cache) = match cached {
        Some(path) => (path, true),
        None if download == DownloadPolicy::AutoDownload => {
            let token = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")).ok();
            let downloader = ferrum_models::HfDownloader::new(cache_dir.to_path_buf(), token)?;
            (downloader.download_gguf(repo, revision, filename).await?, false)
        }
        None => return Err(FerrumError::model(format!(
            "GGUF file '{filename}' in '{model}' is not cached and DownloadPolicy::NoDownload is set"
        ))),
    };
    if let Some(pin) = &pin {
        pin.verify_snapshot(&local_path)?;
    }
    let original_source = ModelSource::HuggingFace {
        repo_id: repo.to_owned(),
        revision: revision.map(str::to_owned),
        cache_dir: Some(cache_dir.display().to_string()),
    };
    let mut source = ResolvedModelSource {
        original: model.to_owned(),
        local_path,
        format: ModelFormat::GGUF,
        from_cache,
    };
    let model_sources = if source_args.semantic_source.is_some() {
        None
    } else {
        let colocated = if source_args.tokenizer_source.is_none() {
            open_colocated_product_sources(&source, &original_source)?
        } else {
            None
        };
        match colocated {
            Some(sources) => Some(sources),
            None => {
                let (sources, metadata_from_cache) = resolve_metadata(
                    model,
                    original_product_source(&original_source, &source.local_path)?,
                    &source.local_path,
                    cache_dir,
                    download,
                    source_args.tokenizer_source.as_deref(),
                )
                .await?;
                source.from_cache &= metadata_from_cache;
                sources
            }
        }
    };
    Ok(finalize_resolution(
        source,
        original_source,
        model_sources,
        autosize,
    ))
}
