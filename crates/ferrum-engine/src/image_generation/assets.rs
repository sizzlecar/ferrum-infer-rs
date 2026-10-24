//! Device-free preparation of explicitly supplied metadata and mapped sources.
mod directory;
mod edit;
use super::*;
pub(super) use edit::EditAssets;
use ferrum_models::qwen3_vl_text::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use ferrum_models::qwen_image21::ExpectedWeights;
use sha2::{Digest, Sha256};

/// Borrowed metadata from one caller-resolved model installation. This loader
/// does no device work and does not discover paths or download missing assets.
pub struct QwenImage21Metadata<'a> {
    pub text_encoder: &'a [u8],
    pub transformer: &'a [u8],
    pub vae: &'a [u8],
    pub model_index: &'a [u8],
    pub scheduler: &'a [u8],
    pub tokenizer: &'a [u8],
    pub tokenizer_config: &'a [u8],
    pub chat_template: &'a [u8],
}

/// Archives are opened explicitly by the composition root. Mapped weight
/// sources are outside the Host payload account; only managed static staging
/// is charged by component initialization. Sources must remain immutable.
pub struct QwenImage21Sources {
    pub text_encoder: Arc<dyn TensorArchive>,
    pub transformer: Arc<dyn TensorArchive>,
    pub vae: Arc<dyn TensorArchive>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageAssetReceipt {
    pub metadata_sha256: BTreeMap<String, String>,
    /// Text encoder, denoiser, decoder; excludes the mapped CPU embedding table.
    pub required_weight_bytes: [u64; 3],
    /// Header/schema agreement is not authentication of checkpoint payloads.
    pub payload_verification: &'static str,
}

pub struct QwenImage21Assets {
    pub(super) sources: QwenImage21Sources,
    pub(super) text: Qwen3VLTextArchitecture,
    pub(super) transformer: QwenImage21Config,
    pub(super) transformer_encodings: BTreeMap<String, WeightEncoding>,
    pub(super) vae: QwenImage21VaeDecoderConfig,
    pub(super) tokenizer: QwenImage21T2iTokenizer,
    pub(super) statistics: QwenImage21LatentStatistics,
    pub(super) schedule: QwenImage21SamplingSchedule,
    pub(super) editing: Option<EditAssets>,
    receipt: ImageAssetReceipt,
}

impl QwenImage21Assets {
    /// Call this before creating a backend. Checks architecture compatibility,
    /// tokenizer/schedule semantics, and every selected physical weight header;
    /// no full checkpoint payload scan or transcode is performed.
    pub fn prepare(
        metadata: QwenImage21Metadata<'_>,
        sources: QwenImage21Sources,
    ) -> anyhow::Result<Self> {
        let named = [
            ("text_encoder", metadata.text_encoder),
            ("transformer", metadata.transformer),
            ("vae", metadata.vae),
            ("model_index", metadata.model_index),
            ("scheduler", metadata.scheduler),
            ("tokenizer_config", metadata.tokenizer_config),
            ("chat_template", metadata.chat_template),
        ];
        for (name, bytes) in named {
            ensure!(
                !bytes.is_empty() && bytes.len() <= 4 * 1024 * 1024,
                "{name} metadata must be nonempty and <=4 MiB"
            );
        }
        let tokenizer = QwenImage21T2iTokenizer::from_source_bytes(
            metadata.tokenizer,
            metadata.tokenizer_config,
            metadata.chat_template,
        )?;
        let text = Qwen3VLTextArchitecture::from_hf_value(&serde_json::from_slice(
            metadata.text_encoder,
        )?)?;
        let transformer =
            QwenImage21Config::from_hf_json(metadata.transformer, StructureLimits::default())?;
        let vae = QwenImage21VaeDecoderConfig::from_hf_json(metadata.vae, 1, 1, 1)?;
        let statistics =
            QwenImage21LatentStatistics::from_vae_config(&serde_json::from_slice(metadata.vae)?)?;
        let schedule =
            QwenImage21SamplingSchedule::parse(metadata.model_index, metadata.scheduler)?;
        ensure!(
            text.hidden_size() == transformer.context_dim(),
            "text hidden width differs from transformer context width"
        );
        ensure!(
            transformer.in_channels() == 64
                && transformer.out_channels() == 64
                && vae.latent_channels() == 64,
            "this single-frame adapter requires 64-channel latent/velocity/decoder ports"
        );

        let text_config = Qwen3VLTextConfig::new(
            text.clone(),
            Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
        )?;
        let text_schema = Qwen3VLTextEncoderFamily::new()?.weights(&text_config)?;
        let expected_dit = ExpectedWeights::from_config(&transformer, StructureLimits::default())?;
        let transformer_encodings = expected_dit.archive_encodings(sources.transformer.as_ref())?;
        let dit_schema = expected_dit.weight_schema_with_encodings(&transformer_encodings)?;
        let dit_bytes = if transformer_encodings.is_empty() {
            validate_archive(&dit_schema, sources.transformer.as_ref())?
        } else {
            dit_schema
                .components
                .iter()
                .try_fold(0_u64, |total, component| {
                    total
                        .checked_add(component.physical_bytes()?)
                        .context("native denoiser weight bytes overflow")
                })?
        };
        let vae_schema = QwenImage21VaeDecoderFamily::new()?.weights(&vae)?;
        let required_weight_bytes = [
            validate_archive(text_schema.stored_schema()?, sources.text_encoder.as_ref())?,
            dit_bytes,
            validate_archive(vae_schema.stored_schema()?, sources.vae.as_ref())?,
        ];
        let embedding = sources
            .text_encoder
            .tensor(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)?;
        ensure!(
            embedding.element_type() == Some(ElementType::Bf16)
                && embedding.shape() == [text.vocabulary_size(), text.hidden_size()],
            "text embedding metadata differs from architecture"
        );
        let mut metadata_sha256 = named
            .into_iter()
            .map(|(name, bytes)| (name.to_owned(), format!("{:x}", Sha256::digest(bytes))))
            .collect::<BTreeMap<_, _>>();
        metadata_sha256.insert(
            "tokenizer".into(),
            format!("{:x}", Sha256::digest(metadata.tokenizer)),
        );
        if !transformer_encodings.is_empty() {
            metadata_sha256.insert(
                "transformer_weight_encodings".into(),
                format!(
                    "{:x}",
                    Sha256::digest(serde_json::to_vec(&transformer_encodings)?)
                ),
            );
        }
        Ok(Self {
            sources,
            text,
            transformer,
            transformer_encodings,
            vae,
            tokenizer,
            statistics,
            schedule,
            editing: None,
            receipt: ImageAssetReceipt {
                metadata_sha256,
                required_weight_bytes,
                payload_verification: "selected headers only; payload not authenticated",
            },
        })
    }

    pub fn receipt(&self) -> &ImageAssetReceipt {
        &self.receipt
    }

    pub fn spatial_scale(&self) -> anyhow::Result<u32> {
        Ok(u32::try_from(self.vae.output_tensor()?.dimensions()[2])?)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn decoder(
        &self,
        height: u32,
        width: u32,
    ) -> anyhow::Result<QwenImage21VaeDecoderConfig> {
        self.decoder_batch(1, height, width)
    }

    pub(super) fn decoder_batch(
        &self,
        batch: usize,
        height: u32,
        width: u32,
    ) -> anyhow::Result<QwenImage21VaeDecoderConfig> {
        Ok(QwenImage21VaeDecoderConfig::new(
            self.vae.latent_channels(),
            self.vae.base_channels(),
            self.vae.dim_mult().to_vec(),
            self.vae.num_res_blocks(),
            self.vae.temporal_upsample().to_vec(),
            u64::try_from(batch)?,
            u64::from(height),
            u64::from(width),
        )?)
    }
}

pub(super) fn validate_archive(
    schema: &WeightSchema,
    source: &dyn TensorArchive,
) -> anyhow::Result<u64> {
    let mut total = 0_u64;
    for component in &schema.components {
        let [name] = component.external_names.as_slice() else {
            anyhow::bail!("component source requires one physical tensor")
        };
        ensure!(
            matches!(
                component.encoding,
                WeightEncoding::Dense {
                    element_type: ElementType::Bf16
                }
            ),
            "only untransformed BF16 source weights are supported"
        );
        let actual = source.tensor(name)?;
        ensure!(
            actual.element_type() == Some(ElementType::Bf16)
                && actual.shape() == component.dimensions
                && u64::try_from(actual.bytes().len())? == component.physical_bytes()?,
            "physical weight metadata mismatch: {name}"
        );
        total = total
            .checked_add(component.physical_bytes()?)
            .context("required weight bytes overflow")?;
    }
    Ok(total)
}
