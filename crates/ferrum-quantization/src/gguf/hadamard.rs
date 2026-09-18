//! Validated activation-basis metadata, independent of the stored GGML dtype.
//!
//! Version 1 is the normalized, blockwise Sylvester Walsh-Hadamard transform.
//! Forward projections consume `H(signs * permutation(x))`; latent embedding
//! rows are restored with `signs * H(row)`. The sign vector spans the full input
//! width, not one repeated block. Recognition does not imply runtime support.

use std::collections::{BTreeMap, BTreeSet};

use candle_core::quantized::gguf_file::Value;
use candle_core::{Error, Result};
use serde::Serialize;

const PREFIX: &str = "prism.hadamard.";
const KEYS: &[&str] = &[
    "version",
    "block_size",
    "transform",
    "axis",
    "sign_mode",
    "weight_names",
    "inverse_weight_names",
    "sign_widths",
    "sign_values",
    "gdn_v_grouped",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GgufHadamardDirection {
    BeforeMatmul,
    AfterEmbeddingLookup,
}

/// Reorder features `[head_dim, key_heads, repeats]` to
/// `[head_dim, repeats, key_heads]`, before applying signs and Hadamard.
/// These are fastest-axis-first dimensions, matching GGML's `ne[]` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GgufHadamardGdnPermutation {
    pub head_dim: u64,
    pub key_heads: u64,
    pub repeats: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GgufHadamardWeight {
    direction: GgufHadamardDirection,
    input_width: u64,
    gdn_permutation: Option<GgufHadamardGdnPermutation>,
}

impl GgufHadamardWeight {
    pub fn direction(&self) -> GgufHadamardDirection {
        self.direction
    }
    pub fn input_width(&self) -> u64 {
        self.input_width
    }
    pub fn gdn_permutation(&self) -> Option<GgufHadamardGdnPermutation> {
        self.gdn_permutation
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "mode", content = "by_width", rename_all = "snake_case")]
pub enum GgufHadamardSigns {
    Identity,
    Explicit(BTreeMap<u64, Vec<i8>>),
}

/// Constructed only by validated GGUF parsing. Its deterministic serialization
/// includes every mathematical choice and sign value needed for source identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GgufHadamard {
    version: u32,
    block_size: u32,
    signs: GgufHadamardSigns,
    gdn_v_grouped: bool,
    weights: BTreeMap<String, GgufHadamardWeight>,
}

impl GgufHadamard {
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn block_size(&self) -> u32 {
        self.block_size
    }
    pub fn signs(&self) -> &GgufHadamardSigns {
        &self.signs
    }
    pub fn gdn_v_grouped(&self) -> bool {
        self.gdn_v_grouped
    }
    pub fn weights(&self) -> &BTreeMap<String, GgufHadamardWeight> {
        &self.weights
    }
    pub fn weight(&self, name: &str) -> Option<&GgufHadamardWeight> {
        self.weights.get(name)
    }

    pub(crate) fn parse<'a>(
        metadata: &BTreeMap<String, Value>,
        architecture: &str,
        tensors: impl Iterator<Item = (&'a str, &'a [u64])>,
    ) -> Result<Option<Self>> {
        let declared: Vec<_> = metadata
            .keys()
            .filter(|key| key.starts_with(PREFIX))
            .collect();
        if declared.is_empty() {
            return Ok(None);
        }
        for key in declared {
            if !is_metadata_key(key) {
                return Err(invalid(format!("unsupported field {key}")));
            }
        }
        if u32_value(metadata, "version")? != 1 {
            return Err(invalid("unsupported version (expected 1)"));
        }
        if !matches!(
            architecture,
            "llama" | "qwen3" | "qwen3moe" | "qwen35" | "qwen35moe" | "qwen3next"
        ) {
            return Err(invalid(format!(
                "architecture {architecture:?} has no verified transform roles"
            )));
        }
        let block_size = u32_value(metadata, "block_size")?;
        if !block_size.is_power_of_two() {
            return Err(invalid("block_size must be a nonzero power of two"));
        }
        if string_value(metadata, "transform")? != "normalized-sylvester-walsh-hadamard" {
            return Err(invalid("unsupported transform"));
        }
        if string_value(metadata, "axis")? != "input-last-dimension" {
            return Err(invalid("unsupported axis"));
        }
        let signs = parse_signs(metadata, block_size)?;
        let gdn_v_grouped = match metadata.get(&format!("{PREFIX}gdn_v_grouped")) {
            None => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => return Err(invalid("gdn_v_grouped must be a boolean")),
        };
        if gdn_v_grouped && !matches!(architecture, "qwen35" | "qwen35moe" | "qwen3next") {
            return Err(invalid("gdn_v_grouped requires recurrent head geometry"));
        }
        let tensor_dimensions: BTreeMap<_, _> = tensors.collect();
        let forward = names(metadata, "weight_names", true)?;
        if forward.is_empty() {
            return Err(invalid("weight_names must not be empty"));
        }
        let inverse = names(metadata, "inverse_weight_names", false)?;
        let mut weights = BTreeMap::new();
        for (direction, names) in [
            (GgufHadamardDirection::BeforeMatmul, forward),
            (GgufHadamardDirection::AfterEmbeddingLookup, inverse),
        ] {
            for name in names {
                match direction {
                    GgufHadamardDirection::BeforeMatmul if !is_projection(name) => {
                        return Err(invalid(format!(
                            "{name:?} is not a verified projection role"
                        )))
                    }
                    GgufHadamardDirection::AfterEmbeddingLookup if name != "token_embd.weight" => {
                        return Err(invalid(format!("{name:?} is not a token embedding lookup")))
                    }
                    _ => {}
                }
                let dimensions = tensor_dimensions
                    .get(name)
                    .ok_or_else(|| invalid(format!("tensor {name:?} is absent")))?;
                if !(2..=3).contains(&dimensions.len()) || dimensions.contains(&0) {
                    return Err(invalid(format!(
                        "{name:?} requires nonzero matrix dimensions"
                    )));
                }
                let input_width = *dimensions.last().unwrap();
                if !input_width.is_multiple_of(u64::from(block_size)) {
                    return Err(invalid(format!(
                        "{name:?} input width is not divisible by block_size"
                    )));
                }
                if let GgufHadamardSigns::Explicit(by_width) = &signs {
                    if !by_width.contains_key(&input_width) {
                        return Err(invalid(format!(
                            "{name:?} has no signs for width {input_width}"
                        )));
                    }
                }
                let gdn_permutation = if gdn_v_grouped && name.ends_with(".ssm_out.weight") {
                    let key_heads = geometry(metadata, architecture, "group_count")?;
                    let value_heads = geometry(metadata, architecture, "time_step_rank")?;
                    if !value_heads.is_multiple_of(key_heads)
                        || !input_width.is_multiple_of(value_heads)
                    {
                        return Err(invalid(format!(
                            "{name:?} has incompatible GDN head geometry"
                        )));
                    }
                    Some(GgufHadamardGdnPermutation {
                        head_dim: input_width / value_heads,
                        key_heads,
                        repeats: value_heads / key_heads,
                    })
                } else {
                    None
                };
                if weights
                    .insert(
                        name.to_owned(),
                        GgufHadamardWeight {
                            direction,
                            input_width,
                            gdn_permutation,
                        },
                    )
                    .is_some()
                {
                    return Err(invalid(format!("duplicate transform for {name:?}")));
                }
            }
        }
        Ok(Some(Self {
            version: 1,
            block_size,
            signs,
            gdn_v_grouped,
            weights,
        }))
    }
}

pub(crate) fn is_metadata_key(key: &str) -> bool {
    key.strip_prefix(PREFIX)
        .is_some_and(|suffix| KEYS.contains(&suffix))
}

pub(crate) fn is_geometry_key(key: &str) -> bool {
    ["qwen35", "qwen35moe", "qwen3next"].iter().any(|arch| {
        key == format!("{arch}.ssm.group_count") || key == format!("{arch}.ssm.time_step_rank")
    })
}

pub(crate) fn declares_transform(key: &str) -> bool {
    key.starts_with(PREFIX)
}

/// Temporary fail-closed boundary until execution providers consume the typed
/// transform contract. Shared by legacy and vNext loaders for every family.
pub(crate) fn unsupported_execution() -> Error {
    Error::Msg("GGUF declares Hadamard activation transforms, but this execution path does not yet consume them; refusing to ignore prism.hadamard metadata".into())
}

fn invalid(reason: impl std::fmt::Display) -> Error {
    Error::Msg(format!("invalid GGUF Hadamard metadata: {reason}"))
}
fn required<'a>(metadata: &'a BTreeMap<String, Value>, name: &str) -> Result<&'a Value> {
    metadata
        .get(&format!("{PREFIX}{name}"))
        .ok_or_else(|| invalid(format!("missing {PREFIX}{name}")))
}
fn u32_value(metadata: &BTreeMap<String, Value>, name: &str) -> Result<u32> {
    match required(metadata, name)? {
        Value::U32(value) => Ok(*value),
        _ => Err(invalid(format!("{name} must be uint32"))),
    }
}
fn string_value<'a>(metadata: &'a BTreeMap<String, Value>, name: &str) -> Result<&'a str> {
    match required(metadata, name)? {
        Value::String(value) => Ok(value),
        _ => Err(invalid(format!("{name} must be a string"))),
    }
}
fn array<'a>(metadata: &'a BTreeMap<String, Value>, name: &str) -> Result<&'a [Value]> {
    match required(metadata, name)? {
        Value::Array(value) => Ok(value),
        _ => Err(invalid(format!("{name} must be an array"))),
    }
}
fn names<'a>(
    metadata: &'a BTreeMap<String, Value>,
    name: &str,
    mandatory: bool,
) -> Result<Vec<&'a str>> {
    if !mandatory && !metadata.contains_key(&format!("{PREFIX}{name}")) {
        return Ok(Vec::new());
    }
    array(metadata, name)?
        .iter()
        .map(|value| match value {
            Value::String(name) if !name.is_empty() => Ok(name.as_str()),
            _ => Err(invalid(format!("{name} must contain nonempty strings"))),
        })
        .collect()
}
fn parse_signs(metadata: &BTreeMap<String, Value>, block_size: u32) -> Result<GgufHadamardSigns> {
    match string_value(metadata, "sign_mode")? {
        "identity" => {
            if metadata.contains_key(&format!("{PREFIX}sign_widths"))
                || metadata.contains_key(&format!("{PREFIX}sign_values"))
            {
                return Err(invalid(
                    "identity sign mode must not contain explicit sign tables",
                ));
            }
            Ok(GgufHadamardSigns::Identity)
        }
        "explicit" => {
            let widths = array(metadata, "sign_widths")?;
            let values = array(metadata, "sign_values")?;
            if widths.is_empty() {
                return Err(invalid("explicit signs require nonempty sign_widths"));
            }
            let mut by_width = BTreeMap::new();
            let mut seen = BTreeSet::new();
            let mut offset = 0_usize;
            for width in widths {
                let width = match width {
                    Value::I32(width) if *width > 0 => *width as u64,
                    _ => return Err(invalid("sign_widths must contain positive int32 values")),
                };
                if !width.is_multiple_of(u64::from(block_size)) || !seen.insert(width) {
                    return Err(invalid(
                        "sign_widths must be unique and divisible by block_size",
                    ));
                }
                let end = offset
                    .checked_add(usize::try_from(width).map_err(Error::wrap)?)
                    .ok_or_else(|| invalid("sign table length overflow"))?;
                let slice = values
                    .get(offset..end)
                    .ok_or_else(|| invalid("sign_values is shorter than sign_widths"))?;
                let signs = slice
                    .iter()
                    .map(|value| match value {
                        Value::I32(value @ (-1 | 1)) => Ok(*value as i8),
                        _ => Err(invalid("sign_values must contain int32 +1 or -1")),
                    })
                    .collect::<Result<Vec<_>>>()?;
                by_width.insert(width, signs);
                offset = end;
            }
            if offset != values.len() {
                return Err(invalid("sign_values has unconsumed entries"));
            }
            Ok(GgufHadamardSigns::Explicit(by_width))
        }
        _ => Err(invalid("unsupported sign_mode")),
    }
}
fn geometry(metadata: &BTreeMap<String, Value>, architecture: &str, suffix: &str) -> Result<u64> {
    let key = format!("{architecture}.ssm.{suffix}");
    match metadata.get(&key) {
        Some(Value::U32(value)) if *value > 0 => Ok(u64::from(*value)),
        _ => Err(invalid(format!("{key} must be positive uint32"))),
    }
}
fn is_projection(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let Some((layer, role)) = name
        .strip_prefix("blk.")
        .and_then(|name| name.split_once('.'))
    else {
        return false;
    };
    !layer.is_empty()
        && layer.bytes().all(|b| b.is_ascii_digit())
        && matches!(
            role,
            "attn_q.weight"
                | "attn_k.weight"
                | "attn_v.weight"
                | "attn_qkv.weight"
                | "attn_gate.weight"
                | "attn_output.weight"
                | "ffn_gate.weight"
                | "ffn_up.weight"
                | "ffn_down.weight"
                | "ffn_gate_exps.weight"
                | "ffn_up_exps.weight"
                | "ffn_down_exps.weight"
                | "ffn_gate_up_exps.weight"
                | "ffn_gate_shexp.weight"
                | "ffn_up_shexp.weight"
                | "ffn_down_shexp.weight"
                | "ssm_out.weight"
        )
}
