//! One explicit repository artifact, preserving independent metadata roles.

use super::*;

#[cfg(test)]
mod tests;

/// Quantization is part of a repository model name, independently of an
/// optional immutable revision. Existing shorthand aliases are handled first.
pub(in crate::source_resolver) fn quantized_repository(
    model: &str,
) -> Result<Option<(String, &str)>> {
    let (name, revision) = model
        .split_once('@')
        .map_or((model, None), |(name, rev)| (name, Some(rev)));
    let Some((repo, quant)) = name.rsplit_once(':') else {
        return Ok(None);
    };
    if !repo.contains('/') {
        return Ok(None);
    }
    huggingface_repository_url(&format!("https://huggingface.co/{repo}"))?;
    ferrum_models::source::gguf_selection::validate_quantization_label(quant)?;
    let repository = revision.map_or_else(|| repo.to_owned(), |rev| format!("{repo}@{rev}"));
    parse_pinned_hf_repository(&repository)?;
    Ok(Some((repository, quant)))
}

pub(crate) async fn resolve(
    model: &str,
    filename: &str,
    cache_dir: &Path,
    download: DownloadPolicy,
    autosize: Option<(AutoSizeProfile, f32)>,
    source_args: &ProductSourceArgs,
) -> Result<Resolved> {
    let composition = if source_args.semantic_source.is_some() {
        ProductSourceComposition::DeferUntilExplicitSemantic
    } else if let Some(tokenizer) = &source_args.tokenizer_source {
        ProductSourceComposition::ResolveWithExplicitTokenizer(tokenizer)
    } else {
        ProductSourceComposition::ResolveColocated
    };
    resolve_selected(
        model,
        ArtifactSelection::File(filename),
        cache_dir,
        download,
        autosize,
        composition,
    )
    .await
}

#[derive(Clone, Copy)]
enum ArtifactSelection<'a> {
    File(&'a str),
    Quantization(&'a str),
}

pub(crate) async fn resolve_quantization(
    model: &str,
    quantization: &str,
    cache_dir: &Path,
    download: DownloadPolicy,
    autosize: Option<(AutoSizeProfile, f32)>,
    composition: ProductSourceComposition<'_>,
) -> Result<Resolved> {
    resolve_selected(
        model,
        ArtifactSelection::Quantization(quantization),
        cache_dir,
        download,
        autosize,
        composition,
    )
    .await
}

async fn resolve_selected(
    model: &str,
    selection: ArtifactSelection<'_>,
    cache_dir: &Path,
    download: DownloadPolicy,
    autosize: Option<(AutoSizeProfile, f32)>,
    composition: ProductSourceComposition<'_>,
) -> Result<Resolved> {
    match selection {
        ArtifactSelection::File(filename) => {
            ferrum_models::source::gguf_selection::validate_standalone_gguf(filename)?
        }
        ArtifactSelection::Quantization(quant) => {
            ferrum_models::source::gguf_selection::validate_quantization_label(quant)?
        }
    }
    let pin = parse_pinned_hf_repository(model)?;
    let metadata_files = match composition {
        ProductSourceComposition::ResolveColocated => &PRODUCT_SOURCE_FILES[..],
        ProductSourceComposition::ResolveWithExplicitTokenizer(_) => &SEMANTIC_FILES[..],
        ProductSourceComposition::DeferUntilExplicitSemantic => &[][..],
    };
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
    let mut cached = None;
    let mut selection_error = None;
    for root in snapshots {
        let candidate = match selection {
            ArtifactSelection::File(filename) => Some(root.join(filename)),
            ArtifactSelection::Quantization(quant) => {
                let files = cache::cached_gguf_files(&root)?;
                let relative = cache::relative_gguf_paths(&root, &files)?;
                if relative.is_empty() {
                    continue;
                }
                match ferrum_models::source::gguf_selection::select_gguf_file(
                    relative.iter().map(String::as_str),
                    Some(quant),
                ) {
                    Ok(path) => path.map(|path| root.join(path)),
                    Err(error) => {
                        selection_error = Some(error);
                        continue;
                    }
                }
            }
        };
        if let Some(path) = candidate {
            if matches!(
                ferrum_models::source::inspect_cached_weights(&path)?,
                ferrum_models::source::CachedWeights::Ready(ModelFormat::GGUF)
            ) {
                if let Some(reason) = ferrum_models::hf_download::inspect_cached_metadata_selection(
                    &root,
                    metadata_files,
                )? {
                    selection_error = Some(FerrumError::model(reason));
                    continue;
                }
                cached = Some(path);
                break;
            }
        }
    }
    let revision = pin.as_ref().map(|pin| pin.revision.as_str());
    let (local_path, from_cache) = match cached {
        Some(path) => (path, true),
        None if download == DownloadPolicy::AutoDownload => {
            let token = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")).ok();
            let downloader = ferrum_models::HfDownloader::new(cache_dir.to_path_buf(), token)?;
            let request = match selection {
                ArtifactSelection::File(filename) => ferrum_models::hf_download::GgufRequest::File(filename),
                ArtifactSelection::Quantization(quant) => ferrum_models::hf_download::GgufRequest::Quantization(quant),
            };
            let path = downloader.download_selected_gguf(repo, revision, request, metadata_files).await?;
            (path, false)
        }
        None => return Err(selection_error.unwrap_or_else(|| FerrumError::model(format!(
            "Selected model artifact in '{model}' is not cached and DownloadPolicy::NoDownload is set"
        )))),
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
    let model_sources = compose_repository(
        &mut source,
        &original_source,
        model,
        cache_dir,
        download,
        composition,
    )
    .await?;
    Ok(finalize_resolution(
        source,
        original_source,
        model_sources,
        autosize,
    ))
}
