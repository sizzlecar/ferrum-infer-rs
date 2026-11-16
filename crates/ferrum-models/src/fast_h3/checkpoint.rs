use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use super::{FAST_H3_GRID_MAX_T, FAST_H3_INFERENCE_SCHEMA};

/// Caller-owned read/allocation limits, independent of a particular release.
#[derive(Clone, Copy, Debug)]
pub struct FastH3MetadataLimits {
    pub max_file_bytes: usize,
    pub max_pdd_steps: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InferenceWire {
    schema_version: String,
    schema: String,
    model_type: String,
    transformer_component: String,
    conditioning: String,
    base_model_revision: String,
    pdd_steps: usize,
    pdd_step_indices: Vec<usize>,
    num_inference_steps: usize,
    transformer_forwards: usize,
    grid_max_t: f64,
    video_scheduler_shift: f64,
    audio_scheduler_shift: f64,
    guidance_scale: f64,
    attention_backend: String,
    vsa_sparsity: f64,
    vsa_tile_size: usize,
    vsa_ref_policy: String,
    vsa_ref_keep_rate: f64,
}

/// Dimensions of the actual widened video/audio transformer, without choosing
/// a backend or assuming every projection uses the checkpoint's quantization.
#[derive(Clone, Debug)]
pub struct FastH3TransformerConfig(TransformerWire);

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformerWire {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    version: Option<String>,
    attention_head_dim: usize,
    audio_in_channels: usize,
    ffn_dim: usize,
    final_norm_eps: f64,
    freq_dim: usize,
    hidden_size: usize,
    in_channels: usize,
    norm_eps: f64,
    num_attention_heads: usize,
    num_layers: usize,
    num_refiner_layers: usize,
    patch_size: [usize; 3],
    pdd_steps: usize,
    qk_norm_eps: f64,
    rope_freq_dim: usize,
    rope_theta: f64,
    text_dim: usize,
    time_embed_dim: usize,
    time_embed_hidden_dim: usize,
    #[serde(default)]
    adaln_rank: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulerWire {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    version: Option<String>,
    shift: f64,
    #[serde(default)]
    stochastic_sampling: bool,
    #[serde(default)]
    shift_provided_sigmas: bool,
}

#[derive(Clone, Debug)]
pub struct FastH3Checkpoint {
    contract: InferenceWire,
    transformer: FastH3TransformerConfig,
}

impl FastH3Checkpoint {
    /// Parse the four original byte streams: duplicate fields are errors and
    /// metadata owners must agree before sampling or allocating any tensors.
    pub fn parse(
        inference: &[u8],
        transformer: &[u8],
        video_scheduler: &[u8],
        audio_scheduler: &[u8],
        limits: FastH3MetadataLimits,
    ) -> Result<Self> {
        ensure!(
            limits.max_file_bytes > 0 && limits.max_pdd_steps >= 2,
            "invalid FastH3 metadata limits"
        );
        for bytes in [inference, transformer, video_scheduler, audio_scheduler] {
            ensure!(
                bytes.len() <= limits.max_file_bytes,
                "FastH3 metadata exceeds byte capacity"
            );
            ensure!(
                bytes
                    .iter()
                    .copied()
                    .find(|byte| !byte.is_ascii_whitespace())
                    == Some(b'{'),
                "FastH3 metadata must be a JSON object"
            );
        }
        let contract: InferenceWire =
            serde_json::from_slice(inference).context("invalid FastH3 inference contract")?;
        let transformer: TransformerWire =
            serde_json::from_slice(transformer).context("invalid FastH3 transformer config")?;
        ensure!(
            contract.schema == FAST_H3_INFERENCE_SCHEMA
                && contract.schema_version == FAST_H3_INFERENCE_SCHEMA,
            "unsupported FastH3 inference schema"
        );
        ensure!(
            contract.model_type == "ref2va"
                && contract.transformer_component == "transformer_ref"
                && contract.conditioning == "fixed_ordered_references_target_only_flow",
            "unsupported FastH3 conditioning contract"
        );
        ensure!(
            contract.guidance_scale == 1.0,
            "FastH3 PDD does not use a CFG branch"
        );
        ensure!(
            contract.attention_backend == "VIDEO_SPARSE_ATTN_H3"
                && contract.vsa_ref_policy == "p2_multi_region",
            "unsupported FastH3 attention contract"
        );
        ensure!(
            [64, 128, 256].contains(&contract.vsa_tile_size),
            "unsupported FastH3 VSA tile shape"
        );
        ensure!(
            contract.vsa_sparsity.is_finite() && (0.0..1.0).contains(&contract.vsa_sparsity),
            "VSA sparsity must be in [0, 1)"
        );
        ensure!(
            contract.vsa_ref_keep_rate.is_finite()
                && contract.vsa_ref_keep_rate > 0.0
                && contract.vsa_ref_keep_rate < 1.0,
            "reference keep rate must be in (0, 1)"
        );
        ensure!(
            contract.grid_max_t == FAST_H3_GRID_MAX_T,
            "unsupported FastH3 PDD grid endpoint"
        );
        let base = contract
            .base_model_revision
            .strip_prefix("hf://")
            .context("FastH3 base revision must start with hf://")?;
        let (repo, revision) = base
            .split_once('@')
            .context("FastH3 base revision must include @revision")?;
        ensure!(
            !repo.trim().is_empty() && !revision.trim().is_empty() && !revision.contains('@'),
            "invalid FastH3 base model revision"
        );
        ensure!(
            contract.pdd_steps >= 2 && contract.pdd_steps <= limits.max_pdd_steps,
            "FastH3 fine grid exceeds capacity or has fewer than two intervals"
        );
        transformer.validate()?;
        ensure!(
            transformer.pdd_steps == contract.pdd_steps,
            "PDD grid disagrees with transformer output heads"
        );
        let indices = &contract.pdd_step_indices;
        ensure!(
            indices.len() >= 2
                && indices.first() == Some(&0)
                && indices.last() == Some(&contract.pdd_steps)
                && indices.windows(2).all(|pair| pair[0] < pair[1]),
            "PDD partition must strictly increase from zero to pdd_steps"
        );
        ensure!(
            contract.num_inference_steps == indices.len() - 1
                && contract.transformer_forwards == indices.len() - 1,
            "PDD evaluation counts disagree with the trained partition"
        );
        for (bytes, expected_shift) in [
            (video_scheduler, contract.video_scheduler_shift),
            (audio_scheduler, contract.audio_scheduler_shift),
        ] {
            let scheduler: SchedulerWire =
                serde_json::from_slice(bytes).context("invalid FastH3 scheduler config")?;
            ensure!(
                scheduler.class == "MiniMaxH3Scheduler",
                "unsupported FastH3 scheduler class"
            );
            ensure!(
                scheduler.shift.is_finite()
                    && scheduler.shift > 0.0
                    && scheduler.shift == expected_shift,
                "FastH3 scheduler shift disagrees with inference contract"
            );
            ensure!(
                !scheduler.stochastic_sampling,
                "FastH3 PDD requires deterministic sampling"
            );
            // Explicit sigmas already carry the modality shift. The upstream
            // shift_provided_sigmas flag is informational for this path.
        }
        Ok(Self {
            contract,
            transformer: FastH3TransformerConfig(transformer),
        })
    }

    pub fn transformer(&self) -> &FastH3TransformerConfig {
        &self.transformer
    }
    pub fn base_model_revision(&self) -> &str {
        &self.contract.base_model_revision
    }
    pub fn pdd_step_indices(&self) -> &[usize] {
        &self.contract.pdd_step_indices
    }
    pub fn pdd_steps(&self) -> usize {
        self.contract.pdd_steps
    }
    pub fn evaluation_count(&self) -> usize {
        self.contract.transformer_forwards
    }
    pub fn video_shift(&self) -> f64 {
        self.contract.video_scheduler_shift
    }
    pub fn audio_shift(&self) -> f64 {
        self.contract.audio_scheduler_shift
    }
    pub fn vsa_tile_size(&self) -> usize {
        self.contract.vsa_tile_size
    }
    pub fn vsa_sparsity(&self) -> f64 {
        self.contract.vsa_sparsity
    }
    pub fn vsa_reference_keep_rate(&self) -> f64 {
        self.contract.vsa_ref_keep_rate
    }
}

impl TransformerWire {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.class == "MiniMaxH3Transformer3DModel",
            "unsupported FastH3 transformer class"
        );
        ensure!(
            [
                self.attention_head_dim,
                self.audio_in_channels,
                self.ffn_dim,
                self.freq_dim,
                self.hidden_size,
                self.in_channels,
                self.num_attention_heads,
                self.num_layers,
                self.rope_freq_dim,
                self.text_dim,
                self.time_embed_dim,
                self.time_embed_hidden_dim
            ]
            .iter()
            .all(|value| *value > 0),
            "FastH3 dimensions must be positive"
        );
        ensure!(
            self.patch_size == [1, 2, 2],
            "FastH3 Ref2VA requires patch_size [1, 2, 2]"
        );
        ensure!(self.freq_dim % 2 == 0, "time frequency width must be even");
        ensure!(
            self.rope_freq_dim
                .checked_mul(6)
                .context("rotary width overflow")?
                <= self.attention_head_dim,
            "rotary width exceeds attention head width"
        );
        ensure!(
            [
                self.norm_eps,
                self.qk_norm_eps,
                self.final_norm_eps,
                self.rope_theta
            ]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
            "FastH3 norm/rotary values must be positive and finite"
        );
        ensure!(
            self.adaln_rank
                .is_none_or(|rank| rank > 0 && rank <= self.time_embed_dim),
            "invalid FastH3 AdaLN rank"
        );
        for (left, right) in [
            (self.num_attention_heads, self.attention_head_dim),
            (self.in_channels, 4),
            (self.in_channels, self.pdd_steps),
            (self.audio_in_channels, self.pdd_steps),
            (self.hidden_size, self.ffn_dim),
        ] {
            left.checked_mul(right)
                .context("FastH3 dimension product overflow")?;
        }
        self.in_channels
            .checked_mul(4)
            .and_then(|width| width.checked_mul(self.pdd_steps))
            .and_then(|width| width.checked_mul(self.hidden_size))
            .context("widened video projection size overflow")?;
        self.audio_in_channels
            .checked_mul(self.pdd_steps)
            .and_then(|width| width.checked_mul(self.hidden_size))
            .context("widened audio projection size overflow")?;
        Ok(())
    }
}

impl FastH3TransformerConfig {
    pub fn norm_eps(&self) -> f64 {
        self.0.norm_eps
    }
    pub fn qk_norm_eps(&self) -> f64 {
        self.0.qk_norm_eps
    }
    pub fn final_norm_eps(&self) -> f64 {
        self.0.final_norm_eps
    }
    pub fn rope_freq_dim(&self) -> usize {
        self.0.rope_freq_dim
    }
    pub fn rotary_dim(&self) -> usize {
        // Configuration validation checks this product before construction.
        self.0.rope_freq_dim * 6
    }
    pub fn rope_theta(&self) -> f64 {
        self.0.rope_theta
    }
    pub fn hidden_size(&self) -> usize {
        self.0.hidden_size
    }
    pub fn attention_heads(&self) -> usize {
        self.0.num_attention_heads
    }
    pub fn attention_head_dim(&self) -> usize {
        self.0.attention_head_dim
    }
    pub fn layers(&self) -> usize {
        self.0.num_layers
    }
    pub fn refiner_layers(&self) -> usize {
        self.0.num_refiner_layers
    }
    pub fn ffn_dim(&self) -> usize {
        self.0.ffn_dim
    }
    pub fn frequency_dim(&self) -> usize {
        self.0.freq_dim
    }
    pub fn time_embed_dim(&self) -> usize {
        self.0.time_embed_dim
    }
    pub fn time_embed_hidden_dim(&self) -> usize {
        self.0.time_embed_hidden_dim
    }
    pub fn adaln_rank(&self) -> Option<usize> {
        self.0.adaln_rank
    }
    pub fn text_dim(&self) -> usize {
        self.0.text_dim
    }
    pub fn video_channels(&self) -> usize {
        self.0.in_channels
    }
    pub fn audio_channels(&self) -> usize {
        self.0.audio_in_channels
    }
    /// Both PDD projections are head-major in the pinned FastVideo H3 model.
    pub fn video_head_width(&self) -> usize {
        self.0.in_channels * 4
    }
    pub fn audio_head_width(&self) -> usize {
        self.0.audio_in_channels
    }
    pub fn video_projection_rows(&self) -> usize {
        self.video_head_width() * self.0.pdd_steps
    }
    pub fn audio_projection_rows(&self) -> usize {
        self.audio_head_width() * self.0.pdd_steps
    }
}
