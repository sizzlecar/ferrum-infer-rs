//! Architecture fields from PrunaAI/P-H3-Image revision
//! `7f98acef503cf2bcfba821a78cb1cf1a9928b2df` and Diffusers
//! `c55ec27943462eeeb65c177d6ff7d8272d21d7a7`.

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use super::{PH3_IMAGE_TEXT_CONNECTOR_CLASS, PH3_IMAGE_TRANSFORMER_CLASS};

/// Checked MiniMax-H3 DiT dimensions. The residual hidden width and the
/// attention inner width are independent: the released model uses 5376 and
/// `56 * 128 = 7168` respectively.
#[derive(Clone, Debug, PartialEq)]
pub struct PH3ImageTransformerConfig(TransformerWire);

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformerWire {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    _version: Option<String>,
    attention_head_dim: u64,
    audio_in_channels: u64,
    ffn_dim: u64,
    final_norm_eps: f64,
    freq_dim: u64,
    hidden_size: u64,
    in_channels: u64,
    norm_eps: f64,
    num_attention_heads: u64,
    num_layers: u64,
    num_refiner_layers: u64,
    patch_size: [u64; 3],
    qk_norm_eps: f64,
    rope_freq_dim: u64,
    rope_theta: f64,
    text_dim: u64,
    time_embed_dim: u64,
    time_embed_hidden_dim: u64,
}

impl PH3ImageTransformerConfig {
    /// Parse original JSON bytes so duplicate fields cannot be silently
    /// collapsed. The source loader owns metadata read and allocation limits.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self> {
        ensure_object(bytes)?;
        let wire: TransformerWire =
            serde_json::from_slice(bytes).context("invalid P-H3-Image transformer config")?;
        ensure!(
            wire.class == PH3_IMAGE_TRANSFORMER_CLASS,
            "unsupported P-H3-Image transformer class"
        );
        ensure!(
            [
                wire.attention_head_dim,
                wire.audio_in_channels,
                wire.ffn_dim,
                wire.freq_dim,
                wire.hidden_size,
                wire.in_channels,
                wire.num_attention_heads,
                wire.num_layers,
                wire.rope_freq_dim,
                wire.text_dim,
                wire.time_embed_dim,
                wire.time_embed_hidden_dim
            ]
            .iter()
            .all(|dimension| *dimension > 0),
            "P-H3-Image transformer dimensions must be positive"
        );
        ensure!(
            wire.patch_size.iter().all(|dimension| *dimension > 0),
            "patch dimensions must be positive"
        );
        ensure!(
            wire.patch_size[0] == 1,
            "P-H3-Image supports a single latent frame"
        );
        ensure!(
            wire.freq_dim % 2 == 0,
            "timestep frequency dimension must be even"
        );
        ensure!(
            product(&[6, wire.rope_freq_dim])? <= wire.attention_head_dim,
            "three-axis RoPE rotation exceeds attention head width"
        );
        ensure!(
            [
                wire.norm_eps,
                wire.qk_norm_eps,
                wire.final_norm_eps,
                wire.rope_theta
            ]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
            "normalization epsilons and RoPE theta must be finite and positive"
        );
        // Check all model-derived matrix extents before a loader or graph
        // builder can use them to size an allocation.
        let attention = product(&[wire.num_attention_heads, wire.attention_head_dim])?;
        let patch = product(&wire.patch_size)?;
        let patch_input = product(&[patch, wire.in_channels])?;
        let modulation = product(&[6, 3, wire.hidden_size])?;
        for dimensions in [
            [wire.hidden_size, attention],
            [wire.hidden_size, wire.ffn_dim],
            [wire.hidden_size, product(&[2, wire.ffn_dim])?],
            [wire.hidden_size, wire.hidden_size],
            [wire.hidden_size, patch_input],
            [wire.hidden_size, wire.audio_in_channels],
            [wire.hidden_size, wire.text_dim],
            [wire.time_embed_dim, modulation],
            [wire.time_embed_hidden_dim, wire.freq_dim],
            [wire.time_embed_hidden_dim, wire.time_embed_dim],
        ] {
            product(&dimensions)?;
        }
        wire.num_layers
            .checked_add(wire.num_refiner_layers)
            .context("transformer and refiner layer count overflows")?;
        Ok(Self(wire))
    }

    pub fn attention_head_dim(&self) -> u64 {
        self.0.attention_head_dim
    }
    pub fn attention_heads(&self) -> u64 {
        self.0.num_attention_heads
    }
    pub fn attention_inner_dim(&self) -> u64 {
        self.0.num_attention_heads * self.0.attention_head_dim
    }
    pub fn audio_in_channels(&self) -> u64 {
        self.0.audio_in_channels
    }
    pub fn ffn_dim(&self) -> u64 {
        self.0.ffn_dim
    }
    pub fn final_norm_eps(&self) -> f64 {
        self.0.final_norm_eps
    }
    pub fn freq_dim(&self) -> u64 {
        self.0.freq_dim
    }
    pub fn hidden_size(&self) -> u64 {
        self.0.hidden_size
    }
    pub fn in_channels(&self) -> u64 {
        self.0.in_channels
    }
    pub fn norm_eps(&self) -> f64 {
        self.0.norm_eps
    }
    pub fn layers(&self) -> u64 {
        self.0.num_layers
    }
    pub fn refiner_layers(&self) -> u64 {
        self.0.num_refiner_layers
    }
    pub fn patch_size(&self) -> [u64; 3] {
        self.0.patch_size
    }
    pub fn patch_input_dim(&self) -> u64 {
        self.0.in_channels * self.0.patch_size.iter().product::<u64>()
    }
    pub fn qk_norm_eps(&self) -> f64 {
        self.0.qk_norm_eps
    }
    pub fn rope_freq_dim(&self) -> u64 {
        self.0.rope_freq_dim
    }
    pub fn rope_theta(&self) -> f64 {
        self.0.rope_theta
    }
    pub fn text_dim(&self) -> u64 {
        self.0.text_dim
    }
    pub fn time_embed_dim(&self) -> u64 {
        self.0.time_embed_dim
    }
    pub fn time_embed_hidden_dim(&self) -> u64 {
        self.0.time_embed_hidden_dim
    }
}

/// Hidden-state taps and output width of the FP32 text connector. State index
/// zero denotes the embedding output, as in `output_hidden_states=True`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PH3ImageTextConnectorConfig(ConnectorWire);

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConnectorWire {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    _version: Option<String>,
    hidden_state_indices: Vec<u32>,
    text_hidden_size: u64,
    out_channels: u64,
}

impl PH3ImageTextConnectorConfig {
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self> {
        ensure_object(bytes)?;
        let wire: ConnectorWire =
            serde_json::from_slice(bytes).context("invalid P-H3-Image text connector config")?;
        ensure!(
            wire.class == PH3_IMAGE_TEXT_CONNECTOR_CLASS,
            "unsupported P-H3-Image text connector class"
        );
        ensure!(
            wire.text_hidden_size > 0 && wire.out_channels > 0,
            "text connector dimensions must be positive"
        );
        ensure!(
            !wire.hidden_state_indices.is_empty(),
            "text connector needs at least one hidden-state tap"
        );
        let taps = u64::try_from(wire.hidden_state_indices.len())?;
        product(&[taps, wire.text_hidden_size, wire.out_channels])?;
        Ok(Self(wire))
    }

    /// Validate the connector against the actual loaded encoder and DiT. The
    /// encoder's last entry is its final normalized hidden state, at `layers`.
    pub fn validate_components(
        &self,
        encoder_layers: u32,
        encoder_hidden_size: u64,
        transformer: &PH3ImageTransformerConfig,
    ) -> Result<()> {
        ensure!(
            self.0.text_hidden_size == encoder_hidden_size,
            "text connector input width differs from encoder"
        );
        ensure!(
            self.0.out_channels == transformer.text_dim(),
            "text connector output width differs from transformer"
        );
        ensure!(
            self.0
                .hidden_state_indices
                .iter()
                .all(|index| *index <= encoder_layers),
            "text connector hidden-state tap exceeds encoder layer count"
        );
        Ok(())
    }

    pub fn hidden_state_indices(&self) -> &[u32] {
        &self.0.hidden_state_indices
    }
    pub fn text_hidden_size(&self) -> u64 {
        self.0.text_hidden_size
    }
    pub fn out_channels(&self) -> u64 {
        self.0.out_channels
    }
    pub fn input_dim(&self) -> u64 {
        self.0.hidden_state_indices.len() as u64 * self.0.text_hidden_size
    }
}

fn product(dimensions: &[u64]) -> Result<u64> {
    dimensions.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .context("P-H3-Image matrix extent overflows")
    })
}

pub(super) fn ensure_object(bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace())
            == Some(b'{'),
        "P-H3-Image metadata must be a JSON object"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
