//! Test-only original-format mixed W4/W8 DiT assets, backed by the one model
//! fixture builder used by the model's own Stored execution tests.
use super::*;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use std::sync::Arc;
mod conditioning;
pub use conditioning::TinyPH3ImageConditioningFixture;

/// Keep this owner through the test's session lifetime: it retains the
/// temporary original-format archive as well as its immutable mapped source.
/// This is a synthetic numerical fixture, not a released checkpoint or a
/// substitute for image-quality/capacity/performance qualification.
pub struct TinyPH3ImageDiTFixture {
    _directory: tempfile::TempDir,
    source: Arc<ComfyCanonicalWeightSource>,
    transformer: PH3ImageTransformerConfig,
    transformer_metadata: Vec<u8>,
    limits: PH3ImageDenoiserLimits,
}
impl TinyPH3ImageDiTFixture {
    pub fn new() -> Self {
        Self::with_image_channels(2).expect("checked default fixture channels")
    }
    /// Use the same original-format builder with a declared image-latent
    /// width, for example the VAE's 24 channels. Audio remains an independent
    /// two-channel stream; projection/head shapes come from the real schema.
    pub fn with_image_channels(image_channels: u64) -> anyhow::Result<Self> {
        let transformer =
            dit_forward::test_fixture::transformer_with_image_channels(image_channels)?;
        let config = dit_forward::test_fixture::config_with_transformer(&transformer);
        // Apply the real schema/node/metadata limits before constructing any
        // payload from a caller-selected channel width.
        dit_forward::PH3ImageDiTFamily::new(dit_forward::test_fixture::limits())?
            .compile_options(&config)?;
        let files = dit_forward::test_fixture::files(&config);
        let limits = dit_forward::test_fixture::limits();
        Ok(Self {
            _directory: files._directory,
            source: Arc::new(files.source),
            transformer,
            transformer_metadata: dit_forward::test_fixture::transformer_metadata(image_channels),
            limits: PH3ImageDenoiserLimits {
                max_metadata_bytes: limits.max_metadata_bytes,
                max_plan_nodes: limits.max_plan_nodes,
                max_stored_bytes: limits.max_stored_bytes,
            },
        })
    }
    pub fn source(&self) -> Arc<ComfyCanonicalWeightSource> {
        self.source.clone()
    }
    pub fn transformer(&self) -> &PH3ImageTransformerConfig {
        &self.transformer
    }
    pub fn transformer_metadata(&self) -> &[u8] {
        &self.transformer_metadata
    }
    pub fn limits(&self) -> PH3ImageDenoiserLimits {
        self.limits
    }
    pub fn target_latent(&self) -> PH3ImageReferenceImage {
        PH3ImageReferenceImage::new(4, 4).expect("checked fixture latent")
    }
    pub const fn text_rows(&self) -> usize {
        512
    }
}
impl Default for TinyPH3ImageDiTFixture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_quantization::TensorArchive;
    use ferrum_runtime::*;
    #[test]
    fn complete_conditioning_fixture_uses_one_original_archive_for_visual_text_and_connector() {
        let fixture = TinyPH3ImageConditioningFixture::new().unwrap();
        let raw: serde_json::Value =
            serde_json::from_slice(fixture.text_encoder_metadata()).unwrap();
        assert_eq!(raw["text_config"]["max_position_embeddings"], 512);
        assert_eq!(
            raw["vision_config"]["deepstack_visual_indexes"],
            serde_json::json!([0, 1, 2])
        );
        let source = fixture.source();
        assert!(Arc::ptr_eq(&source, &fixture.source()));
        for name in [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.4.mlp.down_proj.weight",
            "model.visual.patch_embed.proj.weight",
            "model.visual.deepstack_merger_list.2.linear_fc2.weight",
            "text_connector.proj.weight",
        ] {
            assert!(source.archive().tensor(name).is_ok(), "{name}");
        }
        assert!(source.modules().values().any(|m| matches!(
            m,
            ferrum_quantization::comfy_convrot::ComfyQuantizationMetadata::Int8Tensorwise { .. }
        )));
        assert!(source.modules().values().any(|m| matches!(
            m,
            ferrum_quantization::comfy_convrot::ComfyQuantizationMetadata::CodebookW4A8 { .. }
        )));
        let tokenizer = PH3ImageTextTokenizer::from_source_bytes(
            fixture.tokenizer_json(),
            fixture.tokenizer_config(),
            16,
            PH3ImageTextTokenizerLimits {
                max_source_bytes: 1 << 20,
                max_prompt_bytes: 4096,
                max_tokens_per_prompt: 512,
                max_batch: 2,
            },
        )
        .unwrap();
        assert_eq!(tokenizer.prepare(&["cat"]).unwrap().token_ids(), &[2]);
    }
    #[test]
    fn image_channels_parameter_changes_real_projection_and_head_without_changing_audio() {
        let fixture = TinyPH3ImageDiTFixture::with_image_channels(24).unwrap();
        assert_eq!(fixture.transformer().in_channels(), 24);
        assert_eq!(fixture.transformer().patch_input_dim(), 96);
        assert_eq!(fixture.transformer().audio_in_channels(), 2);
        assert_eq!(
            PH3ImageTransformerConfig::from_hf_json(fixture.transformer_metadata()).unwrap(),
            *fixture.transformer()
        );
        let source = fixture.source();
        let adapter = PH3ImageDenoiserAdapter::new(
            fixture.transformer().clone(),
            source.clone(),
            fixture.limits(),
        )
        .unwrap();
        assert!(adapter.stored_bytes() > 0);
        for (name, shape) in [
            ("video_patch_proj.weight", vec![256, 96]),
            ("final_layer.video_out.weight", vec![96, 256]),
            ("audio_patch_proj.weight", vec![256, 2]),
        ] {
            assert_eq!(
                source.archive().tensor(name).unwrap().shape(),
                shape,
                "{name}"
            );
        }
        assert!(TinyPH3ImageDiTFixture::with_image_channels(0).is_err());
        assert!(TinyPH3ImageDiTFixture::with_image_channels(u64::MAX).is_err());
    }
    #[test]
    fn public_fixture_loads_one_original_mixed_source_and_keeps_complete_shared_schema() {
        let fixture = TinyPH3ImageDiTFixture::new();
        let source = fixture.source();
        assert!(source.modules().keys().any(|k| k.starts_with("blocks.0.")));
        assert!(source.modules().keys().any(|k| k.starts_with("blocks.1.")));
        let adapter =
            PH3ImageDenoiserAdapter::new(fixture.transformer().clone(), source, fixture.limits())
                .unwrap();
        let tags = vec![PH3ImageModality::Text; fixture.text_rows()];
        let layout =
            PH3ImageRequestLayout::new(fixture.transformer(), &tags, fixture.target_latent(), &[])
                .unwrap();
        let request = || {
            adapter
                .request(
                    RequestIdentity::new("request.test_support.ph3").unwrap(),
                    &tags,
                    fixture.target_latent(),
                    None,
                    &vec![0; layout.rows()],
                )
                .unwrap()
        };
        let first = adapter
            .prepare(
                PH3ImageDenoiserPhase::InstallFullWeights,
                1,
                vec![request()],
            )
            .unwrap();
        let refine = adapter
            .prepare(PH3ImageDenoiserPhase::RefineFreshText, 1, vec![request()])
            .unwrap();
        let steady = adapter
            .prepare(
                PH3ImageDenoiserPhase::DenoiseRefinedText,
                1,
                vec![request()],
            )
            .unwrap();
        let schema = first.family().stored_weight_schema().unwrap();
        assert_eq!(schema, refine.family().stored_weight_schema().unwrap());
        assert_eq!(schema, steady.family().stored_weight_schema().unwrap());
        assert_eq!(
            adapter.stored_bytes(),
            schema
                .components
                .iter()
                .map(|c| c.physical_bytes().unwrap())
                .sum::<u64>()
        );
        assert_eq!(refine.family().program().outputs().len(), 1);
        assert_eq!(steady.family().program().outputs().len(), 2);
    }
}
