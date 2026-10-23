//! Automatic standard package resolution; the explicit-source API remains available.
use super::*;
use ferrum_models::pipeline_source::{
    ComponentRole, PipelineArchitecture, PipelineComponentSources, PipelineSourceBundle,
    PipelineSourceLimits,
};
use ferrum_quantization::safetensors_archive::SafetensorsNaming;
use std::path::Path;

impl QwenImage21Assets {
    /// Prepare a complete, immutable local model installation before creating
    /// any device runtime. Uses declared architectures and component roles;
    /// neither the directory name nor a public serving alias selects a model.
    /// Missing metadata (including transformer/config.json) is never fabricated.
    /// Standard archive collisions and external symlinks are explicit errors.
    pub fn from_directory(
        root: impl AsRef<Path>,
        limits: PipelineSourceLimits,
    ) -> anyhow::Result<Self> {
        let bundle = PipelineSourceBundle::open(root, limits)?;
        match bundle.declaration().architecture() {
            PipelineArchitecture::QwenImage21 => {}
        }
        let text = unique_component(&bundle, ComponentRole::ConditionEncoder)?;
        let transformer = unique_component(&bundle, ComponentRole::Denoiser)?;
        let vae = unique_component(&bundle, ComponentRole::LatentCodec)?;
        let scheduler = unique_component(&bundle, ComponentRole::Scheduler)?;
        let processor = unique_component(&bundle, ComponentRole::Processor)?;
        let auxiliary = |name: &str| -> anyhow::Result<&[u8]> {
            let path = format!("{}/{}", processor.declaration().directory(), name);
            Ok(processor
                .auxiliary(&path)
                .with_context(|| format!("missing retained processor metadata {path}"))?
                .bytes())
        };
        let open = |component: &PipelineComponentSources| -> anyhow::Result<_> {
            let naming = match component.declaration().architecture().library() {
                "transformers" => SafetensorsNaming::Transformers,
                "diffusers" => SafetensorsNaming::Diffusers,
                other => anyhow::bail!("unsupported component weight library {other}"),
            };
            Ok(Arc::new(SafetensorsArchive::open_standard_component(
                bundle.root(),
                component.declaration().directory(),
                naming,
                limits.configuration_bytes(),
            )?))
        };
        let mut assets = Self::prepare(
            QwenImage21Metadata {
                text_encoder: text.config().bytes(),
                transformer: transformer.config().bytes(),
                vae: vae.config().bytes(),
                model_index: bundle.model_index().bytes(),
                scheduler: scheduler.config().bytes(),
                tokenizer: auxiliary("tokenizer.json")?,
                tokenizer_config: auxiliary("tokenizer_config.json")?,
                chat_template: auxiliary("chat_template.jinja")?,
            },
            QwenImage21Sources {
                text_encoder: open(text)?,
                transformer: open(transformer)?,
                vae: open(vae)?,
            },
        )?;
        // A declared visual tower enables the editing capability only after
        // every required visual/encoder weight and processor field is checked.
        let text_metadata: serde_json::Value = serde_json::from_slice(text.config().bytes())?;
        if text_metadata.get("vision_config").is_some() {
            assets = assets.with_image_editing(
                text.config().bytes(),
                vae.config().bytes(),
                processor.config().bytes(),
            )?;
        }
        // Preserve the identity of every consumed package metadata file,
        // including processor declarations and optional generation metadata.
        for digest in bundle.manifest().files() {
            assets.receipt.metadata_sha256.insert(
                format!("package/{}", digest.path()),
                digest.sha256().to_owned(),
            );
        }
        Ok(assets)
    }
}

fn unique_component(
    bundle: &PipelineSourceBundle,
    role: ComponentRole,
) -> anyhow::Result<&PipelineComponentSources> {
    let mut matches = bundle.components_with_role(role);
    let component = matches
        .next()
        .with_context(|| format!("pipeline has no {role:?} component"))?;
    ensure!(
        matches.next().is_none(),
        "pipeline has multiple {role:?} components; this adapter requires one"
    );
    Ok(component)
}
