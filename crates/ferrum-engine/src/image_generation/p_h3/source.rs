//! Explicit source domain for the shared executor. Original quantized carriers
//! stay borrowed; only the declared text norms, visual weights and VAE adapters
//! may allocate managed WeightBacking.
use super::PH3ImageAssets;
use crate::image_generation::{
    ImagePipelineSourceAdapter, ImageStage, ImageStageDeclaration, ImageWeightOwnerKey,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    p_h3_image::PH3ImageTextWeightSource, p_h3_image_vae::PH3ImageVaeWeightSource,
    qwen3_vl_vision::comfy::Qwen3VLVisionComfyWeightSource,
};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::*;
use std::sync::Arc;

#[derive(Clone)]
pub struct PH3ImageSourceAdapter {
    pub(super) assets: Arc<PH3ImageAssets>,
}
impl PH3ImageSourceAdapter {
    pub fn new(assets: Arc<PH3ImageAssets>) -> Self {
        Self { assets }
    }
}
pub struct PH3ImageStageSource<'a> {
    kind: SourceKind<'a>,
}
enum SourceKind<'a> {
    Canonical(&'a ComfyCanonicalWeightSource),
    Text(PH3ImageTextWeightSource<'a>),
    Vae(PH3ImageVaeWeightSource<'a>),
    Vision(Qwen3VLVisionComfyWeightSource<'a>),
}
fn exact_canonical(
    source: &ComfyCanonicalWeightSource,
    component: &WeightComponentSpec,
) -> Result<()> {
    let [name] = component.external_names.as_slice() else {
        anyhow::bail!("original H3 physical component needs one exact tensor");
    };
    let original = source.archive().tensor(name)?;
    ensure!(
        original.shape() == component.dimensions,
        "H3 original tensor shape differs: {name}"
    );
    let WeightEncoding::Dense { element_type } = component.encoding else {
        anyhow::bail!("H3 physical source requires exact dense carrier specifications");
    };
    let f8_carrier = element_type == ElementType::U8
        && name.ends_with(".weight_s_rel")
        && original.dtype() == safetensors::Dtype::F8_E4M3;
    ensure!(
        original.element_type() == Some(element_type) || f8_carrier,
        "H3 original tensor needs an undeclared conversion: {name}"
    );
    Ok(())
}
impl WeightComponentSource for PH3ImageStageSource<'_> {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        match &self.kind {
            SourceKind::Canonical(source) => {
                exact_canonical(source, component).map_err(|e| {
                    VNextError::InvalidExecutionPlan {
                        reason: e.to_string(),
                    }
                })?;
                source.component(component)
            }
            SourceKind::Text(source) => source.component(component),
            SourceKind::Vae(source) => source.component(component),
            SourceKind::Vision(source) => source.component(component),
        }
    }
}
impl ImagePipelineSourceAdapter for PH3ImageSourceAdapter {
    type Source<'a> = PH3ImageStageSource<'a>;
    fn supports_stage(&self, _stage: ImageStage) -> bool {
        true
    }
    fn weight_owner(&self, stage: ImageStage) -> Option<ImageWeightOwnerKey> {
        Some(match stage {
            ImageStage::TextEncoder => ImageWeightOwnerKey::TextEncoder,
            ImageStage::TextConnector => ImageWeightOwnerKey::TextConnector,
            ImageStage::TextRefiner | ImageStage::Denoiser => ImageWeightOwnerKey::Denoiser,
            ImageStage::VaeEncoder => ImageWeightOwnerKey::VaeEncoder,
            ImageStage::Decoder => ImageWeightOwnerKey::Decoder,
            ImageStage::VisionEncoder => ImageWeightOwnerKey::VisionEncoder,
            ImageStage::Noise => return None,
        })
    }
    fn stage_source<'a>(
        &'a self,
        declaration: &ImageStageDeclaration,
        _profile: &NumericalProfileId,
        host: &'a HostMemoryAccount,
    ) -> Result<Option<Self::Source<'a>>> {
        ensure!(
            declaration.weight_owner == self.weight_owner(declaration.stage),
            "H3 stage declares a foreign static owner"
        );
        let kind = match declaration.stage {
            ImageStage::Noise => return Ok(None),
            ImageStage::TextEncoder => {
                SourceKind::Text(self.assets.text_weights.weight_source(host))
            }
            ImageStage::TextConnector => SourceKind::Canonical(&self.assets.sources.text_encoder),
            ImageStage::TextRefiner | ImageStage::Denoiser => {
                SourceKind::Canonical(&self.assets.sources.transformer)
            }
            ImageStage::VaeEncoder => {
                SourceKind::Vae(self.assets.vae.encoder_weight_source(host.clone()))
            }
            ImageStage::Decoder => {
                SourceKind::Vae(self.assets.vae.decoder_weight_source(host.clone()))
            }
            ImageStage::VisionEncoder => {
                SourceKind::Vision(self.assets.vision_weights.weight_source(host))
            }
        };
        Ok(Some(PH3ImageStageSource { kind }))
    }
    fn prepare_host_backing(
        &self,
        declaration: &ImageStageDeclaration,
        profile: &NumericalProfileId,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<(usize, HostBackingPreparation)> {
        // A source view proves selection before any memory grant. It retains
        // the actual current account only for this call/install, never a cache.
        let _source = self.stage_source(declaration, profile, host)?;
        let extra = match declaration.stage {
            ImageStage::TextEncoder => self
                .assets
                .text_weights
                .maximum_group_backing_bytes(family, plan, host)?,
            ImageStage::VaeEncoder => self
                .assets
                .vae
                .maximum_encoder_group_backing_bytes(family, plan, host)?,
            ImageStage::Decoder => self
                .assets
                .vae
                .maximum_decoder_group_backing_bytes(family, plan, host)?,
            ImageStage::Noise => {
                ensure!(
                    matches!(family.weights(), ModelWeights::None),
                    "noise must be weightless"
                );
                0
            }
            ImageStage::TextConnector | ImageStage::TextRefiner | ImageStage::Denoiser => {
                ensure!(
                    plan.payload().execution_weights().stored()?
                        == &ExecutionWeightPlan::identity(family)?,
                    "H3 canonical sources require the exact identity materializer"
                );
                let source = if declaration.stage == ImageStage::TextConnector {
                    &self.assets.sources.text_encoder
                } else {
                    &self.assets.sources.transformer
                };
                for component in &family.stored_weight_schema()?.components {
                    exact_canonical(source, component)?;
                }
                0
            }
            ImageStage::VisionEncoder => self
                .assets
                .vision_weights
                .maximum_group_backing_bytes(family, plan, host)?,
        };
        let backing = plan.prepare_host_backing(family, None)?;
        Ok((
            backing
                .additional_bytes()
                .checked_add(extra)
                .context("H3 source backing extent overflows")?,
            backing,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_quantization::SafetensorsArchive;
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype,
    };

    #[test]
    fn exact_source_borrows_original_f32_and_refuses_undeclared_dense_promotions() {
        let directory = tempfile::tempdir().unwrap();
        let f32_bytes = [1.25_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat();
        let bf16_bytes = [0x3fa0_u16.to_le_bytes(), 0xc020_u16.to_le_bytes()].concat();
        serialize_to_file(
            [
                (
                    "original",
                    TensorView::new(Dtype::F32, vec![2], &f32_bytes).unwrap(),
                ),
                (
                    "requires_promotion",
                    TensorView::new(Dtype::BF16, vec![2], &bf16_bytes).unwrap(),
                ),
            ],
            &None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let original =
            ComfyCanonicalWeightSource::new(SafetensorsArchive::open(directory.path()).unwrap())
                .unwrap();
        let source = PH3ImageStageSource {
            kind: SourceKind::Canonical(&original),
        };
        let mut component = WeightComponentSpec {
            id: WeightId::new("weight.original").unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec!["original".into()],
            dimensions: vec![2],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        };
        let payload = source.component(&component).unwrap();
        let mapped = original.archive().tensor("original").unwrap();
        assert_eq!(payload.bytes(), f32_bytes);
        assert_eq!(payload.bytes().as_ptr(), mapped.bytes().as_ptr());
        drop(payload);
        component.external_names = vec!["requires_promotion".into()];
        assert!(source.component(&component).is_err());
        component.encoding = WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        };
        assert_eq!(source.component(&component).unwrap().bytes(), bf16_bytes);
        component.dimensions = vec![1, 2];
        assert!(source.component(&component).is_err());
    }
}
