use ferrum_runtime::CanonicalRational;
use ferrum_types::Result;
use serde::{Deserialize, Deserializer, Serialize};

use super::{invalid, StructureLimits, TRANSFORMER_CLASS};

/// Checked semantic configuration. Serde roundtrips this canonical structure;
/// `from_hf_json` separately adapts the Hugging Face configuration format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct QwenImage21Config(ConfigData);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigData {
    layers: u64,
    heads: u64,
    head_dim: u64,
    context_dim: u64,
    in_channels: u64,
    out_channels: u64,
    patch_size: u64,
    mlp_ratio: u64,
    rope_axes: [u64; 3],
    causal_condition: bool,
    epsilon: CanonicalRational,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HfConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    _diffusers_version: Option<String>,
    #[serde(rename = "_name_or_path")]
    _name_or_path: Option<String>,
    num_layers: u64,
    num_attention_heads: u64,
    attention_head_dim: u64,
    context_in_dim: u64,
    in_channels: u64,
    out_channels: u64,
    patch_size: u64,
    mlp_ratio: u64,
    axes_dims_rope: [u64; 3],
    causal_condition: bool,
    eps: serde_json::Number,
}

impl<'de> Deserialize<'de> for QwenImage21Config {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::checked(ConfigData::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl QwenImage21Config {
    pub fn from_hf_json(bytes: &[u8], limits: StructureLimits) -> Result<Self> {
        limits.check_metadata("transformer config", bytes)?;
        let raw: HfConfig = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid transformer config: {error}")))?;
        if raw.class_name != TRANSFORMER_CLASS {
            return Err(invalid(
                "transformer config declares an unsupported architecture",
            ));
        }
        let epsilon = CanonicalRational::from_decimal_str(&raw.eps.to_string())
            .map_err(|error| invalid(format!("invalid transformer eps: {error}")))?;
        Self::checked(ConfigData {
            layers: raw.num_layers,
            heads: raw.num_attention_heads,
            head_dim: raw.attention_head_dim,
            context_dim: raw.context_in_dim,
            in_channels: raw.in_channels,
            out_channels: raw.out_channels,
            patch_size: raw.patch_size,
            mlp_ratio: raw.mlp_ratio,
            rope_axes: raw.axes_dims_rope,
            causal_condition: raw.causal_condition,
            epsilon,
        })
    }

    fn checked(data: ConfigData) -> Result<Self> {
        if [
            data.layers,
            data.heads,
            data.head_dim,
            data.context_dim,
            data.in_channels,
            data.out_channels,
            data.mlp_ratio,
        ]
        .contains(&0)
        {
            return Err(invalid(
                "transformer dimensions and layer count must be positive",
            ));
        }
        if data.patch_size != 1 {
            return Err(invalid(
                "QwenImage21 structure adapter currently supports patch_size=1 only",
            ));
        }
        if data.epsilon.numerator() <= 0 {
            return Err(invalid("transformer eps must be finite and positive"));
        }
        if data
            .rope_axes
            .iter()
            .any(|axis| *axis == 0 || axis % 2 != 0)
            || data
                .rope_axes
                .iter()
                .try_fold(0_u64, |sum, axis| sum.checked_add(*axis))
                != Some(data.head_dim)
        {
            return Err(invalid(
                "three positive even RoPE axes must sum to attention_head_dim",
            ));
        }
        let config = Self(data);
        let h = config
            .0
            .heads
            .checked_mul(config.0.head_dim)
            .ok_or_else(|| invalid("transformer hidden size overflows"))?;
        let m = h
            .checked_mul(config.0.mlp_ratio)
            .ok_or_else(|| invalid("transformer MLP size overflows"))?;
        // Validate every derived matrix and aggregate without allocating an
        // inventory proportional to a potentially enormous layer count.
        let square = h
            .checked_mul(h)
            .ok_or_else(|| invalid("transformer square matrix overflows"))?;
        let mlp = h
            .checked_mul(m)
            .ok_or_else(|| invalid("transformer MLP matrix overflows"))?;
        let per_layer = square
            .checked_mul(4)
            .and_then(|v| mlp.checked_mul(3).and_then(|m| v.checked_add(m)))
            .and_then(|v| {
                config
                    .0
                    .head_dim
                    .checked_mul(2)
                    .and_then(|n| v.checked_add(n))
            })
            .ok_or_else(|| invalid("transformer layer parameter count overflows"))?;
        let global = [
            h.checked_mul(config.0.in_channels),
            square.checked_mul(4),
            h.checked_mul(256),
            Some(square),
            h.checked_mul(config.0.context_dim),
            Some(square),
            Some(config.0.context_dim),
            Some(square),
            h.checked_mul(config.0.out_channels),
        ]
        .into_iter()
        .try_fold(0_u64, |sum, item| {
            item.and_then(|value| sum.checked_add(value))
        })
        .ok_or_else(|| invalid("transformer global parameter count overflows"))?;
        per_layer
            .checked_mul(config.0.layers)
            .and_then(|v| v.checked_add(global))
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("transformer BF16 payload size overflows"))?;
        config
            .0
            .layers
            .checked_mul(9)
            .and_then(|v| v.checked_add(9))
            .ok_or_else(|| invalid("transformer tensor count overflows"))?;
        Ok(config)
    }

    pub const fn layers(&self) -> u64 {
        self.0.layers
    }
    pub const fn heads(&self) -> u64 {
        self.0.heads
    }
    pub const fn head_dim(&self) -> u64 {
        self.0.head_dim
    }
    pub fn hidden_size(&self) -> u64 {
        self.0.heads * self.0.head_dim
    }
    pub fn mlp_hidden_size(&self) -> u64 {
        self.hidden_size() * self.0.mlp_ratio
    }
    pub const fn context_dim(&self) -> u64 {
        self.0.context_dim
    }
    pub const fn in_channels(&self) -> u64 {
        self.0.in_channels
    }
    pub const fn out_channels(&self) -> u64 {
        self.0.out_channels
    }
    pub const fn patch_size(&self) -> u64 {
        self.0.patch_size
    }
    pub const fn rope_axes(&self) -> &[u64; 3] {
        &self.0.rope_axes
    }
    pub const fn causal_condition(&self) -> bool {
        self.0.causal_condition
    }
    pub const fn epsilon(&self) -> CanonicalRational {
        self.0.epsilon
    }
}
