use super::*;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
mod image;
pub use image::{
    PH3ImageEditPrompt, PH3_IMAGE_EDIT_PRESENTATION_PROFILE, PH3_IMAGE_EDIT_ROTARY_PROFILE,
};

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageTextTokenizerLimits {
    pub max_source_bytes: usize,
    pub max_prompt_bytes: usize,
    pub max_tokens_per_prompt: u32,
    pub max_batch: usize,
}

/// Uses the repository's HF tokenizer engine directly. The presentation is
/// model capability data, not a client/model-name exception in a chat layer.
/// No chat template is rendered and no start/end token is requested.
pub struct PH3ImageTextTokenizer {
    tokenizer: Tokenizer,
    vocabulary: u64,
    pad: u32,
    forbidden_vision: Vec<u32>,
    source_sha256: [String; 2],
    limits: PH3ImageTextTokenizerLimits,
}
impl PH3ImageTextTokenizer {
    pub fn from_source_bytes(
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        vocabulary: u64,
        limits: PH3ImageTextTokenizerLimits,
    ) -> Result<Self> {
        ensure!(
            limits.max_source_bytes > 0
                && limits.max_prompt_bytes > 0
                && limits.max_tokens_per_prompt > 0
                && limits.max_tokens_per_prompt <= 1 << 24
                && (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&limits.max_batch),
            "invalid text tokenizer capacities"
        );
        let bytes = tokenizer
            .len()
            .checked_add(tokenizer_config.len())
            .context("tokenizer source extent overflows")?;
        ensure!(
            !tokenizer.is_empty()
                && !tokenizer_config.is_empty()
                && bytes <= limits.max_source_bytes,
            "tokenizer source exceeds its declared capacity"
        );
        ensure!(
            vocabulary > 0 && vocabulary <= u32::MAX as u64,
            "invalid embedding vocabulary"
        );
        let config: serde_json::Value = serde_json::from_slice(tokenizer_config)?;
        let disabled = |key: &str| {
            matches!(
                config.get(key),
                None | Some(serde_json::Value::Null) | Some(serde_json::Value::Bool(false))
            )
        };
        ensure!(
            config["tokenizer_class"] == "Qwen2Tokenizer"
                && config["pad_token"] == "<|endoftext|>"
                && config["bos_token"].is_null()
                && disabled("add_bos_token")
                && disabled("add_eos_token")
                && disabled("add_prefix_space")
                && disabled("split_special_tokens"),
            "unsupported raw Qwen tokenizer semantics for P-H3 text conditioning"
        );
        let mut tokenizer_engine = Tokenizer::from_bytes(tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer source: {e}"))?;
        tokenizer_engine.with_padding(None);
        tokenizer_engine
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
        let pad = tokenizer_engine
            .token_to_id("<|endoftext|>")
            .context("missing source PAD token")?;
        ensure!(
            u64::from(pad) < vocabulary
                && tokenizer_engine
                    .get_vocab(true)
                    .values()
                    .all(|&id| u64::from(id) < vocabulary),
            "tokenizer IDs exceed the original embedding vocabulary"
        );
        let encode = |s| {
            tokenizer_engine
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
        };
        ensure!(
            encode("")?.get_ids().is_empty(),
            "raw tokenizer inserts tokens for empty text"
        );
        ensure!(
            encode("<|endoftext|>")?.get_ids() == [pad],
            "source PAD is not one exact token"
        );
        let forbidden_vision = [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        .into_iter()
        .filter_map(|marker| tokenizer_engine.token_to_id(marker))
        .collect();
        Ok(Self {
            tokenizer: tokenizer_engine,
            vocabulary,
            pad,
            forbidden_vision,
            source_sha256: [
                format!("{:x}", Sha256::digest(tokenizer)),
                format!("{:x}", Sha256::digest(tokenizer_config)),
            ],
            limits,
        })
    }
    pub fn presentation_profile(&self) -> &'static str {
        PH3_IMAGE_TEXT_PRESENTATION_PROFILE
    }
    pub fn source_sha256(&self) -> &[String; 2] {
        &self.source_sha256
    }
    /// Derived from the supplied tokenizer's special-token definition. The
    /// pinned official release resolves this to 151643; other tokenizers are
    /// never globally assigned that ID.
    pub fn pad_token_id(&self) -> u32 {
        self.pad
    }
    pub fn prepare(&self, prompts: &[&str]) -> Result<PH3ImageTextPromptBatch> {
        ensure!(
            !prompts.is_empty() && prompts.len() <= self.limits.max_batch,
            "prompt batch exceeds declared capacity"
        );
        let bytes = prompts
            .iter()
            .try_fold(0usize, |sum, s| sum.checked_add(s.len()))
            .context("prompt bytes overflow")?;
        ensure!(
            bytes <= self.limits.max_prompt_bytes,
            "raw prompt bytes exceed declared capacity"
        );
        let mut rows = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            // External embeddings are not raw text/vision and need a distinct
            // capability. Do not silently consume a textual-inversion marker.
            ensure!(!prompt.contains("embedding:"), "external embedding references are not supported by the text-only conditioning interface");
            let encoded = self
                .tokenizer
                .encode(*prompt, false)
                .map_err(|e| anyhow::anyhow!("raw prompt encoding: {e}"))?;
            let mut ids = encoded.get_ids().to_vec();
            ensure!(
                !ids.iter().any(|id| self.forbidden_vision.contains(id)),
                "vision prefixes require the image-editing conditioning interface"
            );
            if ids.is_empty() {
                ids.push(self.pad);
            }
            ensure!(
                ids.len() <= self.limits.max_tokens_per_prompt as usize,
                "prompt token count exceeds declared capacity; truncation is disabled"
            );
            rows.push(ids);
        }
        PH3ImageTextPromptBatch::from_rows(rows, self.pad, self.vocabulary)
    }
}

/// Rectangular device storage with explicit valid prefixes. Short prompts are
/// right-padded, so causal valid outputs match an independent unpadded request;
/// the sink remains token zero. Padding rows are excluded from returned DiT
/// lengths. No empty text, vision input, or invalid ID is silently discarded.
pub struct PH3ImageTextPromptBatch {
    ids: Vec<u32>,
    valid_lengths: Vec<u32>,
    layout: Qwen3VLTextLayout,
    vocabulary: u64,
}
impl PH3ImageTextPromptBatch {
    pub(super) fn from_rows(rows: Vec<Vec<u32>>, pad: u32, vocabulary: u64) -> Result<Self> {
        ensure!(
            !rows.is_empty() && rows.len() <= STRUCTURED_ATTENTION_BF16_MAX_BATCH,
            "invalid raw text batch"
        );
        ensure!(
            vocabulary > 0 && u64::from(pad) < vocabulary,
            "PAD exceeds embedding vocabulary"
        );
        let sequence = rows.iter().map(Vec::len).max().context("empty batch")?;
        ensure!(
            sequence > 0 && sequence <= 1 << 24,
            "text sequence exceeds exact F32 position range"
        );
        let total = rows
            .len()
            .checked_mul(sequence)
            .context("token storage extent overflows")?;
        let mut ids = Vec::with_capacity(total);
        let mut lengths = Vec::with_capacity(rows.len());
        let mut runs = Vec::with_capacity(rows.len());
        for row in rows {
            ensure!(
                !row.is_empty() && row.iter().all(|&id| u64::from(id) < vocabulary),
                "text IDs must be nonempty and within the original embedding vocabulary"
            );
            let length = u32::try_from(row.len())?;
            lengths.push(length);
            let mut sample = vec![Qwen3VLTextKeyRun::new(length, true)];
            if row.len() < sequence {
                sample.push(Qwen3VLTextKeyRun::new((sequence - row.len()) as u32, false));
            }
            runs.push(sample);
            ids.extend_from_slice(&row);
            ids.resize(ids.len() + sequence - row.len(), pad);
        }
        Ok(Self {
            ids,
            valid_lengths: lengths,
            layout: Qwen3VLTextLayout::new(sequence as u32, runs)?,
            vocabulary,
        })
    }
    pub fn token_ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn valid_lengths(&self) -> &[u32] {
        &self.valid_lengths
    }
    pub fn layout(&self) -> &Qwen3VLTextLayout {
        &self.layout
    }
    pub fn managed_inputs(
        &self,
        config: &PH3ImageTextConditioningConfig,
        host: &HostMemoryAccount,
    ) -> Result<Vec<ManagedHostTensor>> {
        let text = config.encoder().text();
        ensure!(
            config.encoder().visual_inputs().is_none()
                && text.layout() == &self.layout
                && text.architecture().vocabulary_size() == self.vocabulary,
            "raw text shape/vocabulary or input mode differs from the compiled conditioning component"
        );
        let specs = config.input_tensor_specs()?;
        let ids_id = ProgramValueId::new(PH3ImageTextConditioningFamily::TOKEN_IDS_ID)?;
        let cos_id = ProgramValueId::new(Qwen3VLTextComfyFamily::COSINE_ID)?;
        let sin_id = ProgramValueId::new(Qwen3VLTextComfyFamily::SINE_ID)?;
        let id_spec = &specs[&ids_id];
        let rotary_spec = &specs[&cos_id];
        let id_bytes = usize::try_from(id_spec.byte_len()?)?;
        let rotary_bytes = usize::try_from(rotary_spec.byte_len()?)?;
        let mut reservations = host
            .reserve_many(&[
                (HostMemoryClass::InputPreprocess, id_bytes),
                (HostMemoryClass::InputPreprocess, rotary_bytes),
                (HostMemoryClass::InputPreprocess, rotary_bytes),
            ])?
            .into_iter();
        let mut ids = reservations.next().unwrap().allocate_zeroed()?;
        let mut cos = reservations.next().unwrap().allocate_zeroed()?;
        let mut sin = reservations.next().unwrap().allocate_zeroed()?;
        for (bytes, id) in ids.as_mut().chunks_exact_mut(4).zip(&self.ids) {
            bytes.copy_from_slice(&id.to_le_bytes());
        }
        let width = text.architecture().head_dim() as usize;
        let sequence = self.layout.sequence_length() as usize;
        let mut inverse = [0.0f32; 128];
        for (pair, value) in inverse[..width / 2].iter_mut().enumerate() {
            *value = 1.0
                / libm::powf(
                    text.architecture().rope_theta() as f32,
                    (2 * pair) as f32 / width as f32,
                );
        }
        for row in 0..self.ids.len() {
            let local = row % sequence;
            let position = if local < self.valid_lengths[row / sequence] as usize {
                local
            } else {
                0
            };
            for (pair, &frequency) in inverse[..width / 2].iter().enumerate() {
                let phase = position as f32 * frequency;
                let c = libm::cosf(phase).to_le_bytes();
                let s = libm::sinf(phase).to_le_bytes();
                for column in [pair, pair + width / 2] {
                    let offset = (row * width + column) * 4;
                    cos.as_mut()[offset..offset + 4].copy_from_slice(&c);
                    sin.as_mut()[offset..offset + 4].copy_from_slice(&s);
                }
            }
        }
        let resolved = |spec: &ProgramTensorSpec| {
            ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                spec.element_type,
                spec.layout.clone(),
            )
        };
        Ok(vec![
            ManagedHostTensor::new(ids_id, resolved(id_spec)?, ids.freeze())?,
            ManagedHostTensor::new(cos_id, resolved(rotary_spec)?, cos.freeze())?,
            ManagedHostTensor::new(sin_id, resolved(rotary_spec)?, sin.freeze())?,
        ])
    }
}
