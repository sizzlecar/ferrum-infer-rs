//! Canonical Comfy INT8 ConvRot / codebook-W4A8 checkpoint inspection.
//!
//! Source contracts: Comfy-Org/comfy-kitchen at
//! `cdab4787a89de5ea9d2bd0bf6ddc6b9e94249d27`, `tensor/int8_utils.py`,
//! `tensor/w4a8_packed.py`, and `backends/eager/w4a8_int8.py`; checkpoint
//! serialization: Comfy-Org/comfy-model-tools at
//! `d6797787e6bdb1a1fb0094d588a26f8e71a1c757`, `quant_int8_convrot.py`.
//! Verified against small byte ranges of PrunaAI/P-H3-Image-Comfy revision
//! `44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543` (DiT metadata and codebook).
//!
//! This module preserves borrowed checkpoint bytes, supplies a validating
//! component source, and provides F32 dequantization and BF16/F32 A8 scalar CPU
//! references. Device execution belongs to the separately versioned runtime
//! operations; these references do not establish CUDA numerical parity.

use ferrum_types::{FerrumError, Result};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};

use crate::{SafetensorsArchive, SafetensorsTensor};

mod reference;
mod source;
pub use source::{ComfyCanonicalWeightSource, COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID};
mod a8_reference;
pub use a8_reference::{ComfyA8Bf16Reference, ComfyA8F32Reference};
mod embedding_reference;
#[cfg(test)]
mod tests;

/// Parsed per-module `.comfy_quant` U8 JSON, rather than file metadata.
/// Unsupported formats and unknown/duplicate fields fail during parsing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "format", deny_unknown_fields)]
pub enum ComfyQuantizationMetadata {
    #[serde(rename = "int8_tensorwise")]
    Int8Tensorwise {
        #[serde(default)]
        convrot: bool,
        #[serde(default = "default_rotation_group")]
        convrot_groupsize: usize,
    },
    #[serde(rename = "asym_w4a8_int8")]
    CodebookW4A8 {
        group_size: usize,
        convrot: bool,
        convrot_groupsize: usize,
    },
}

const fn default_rotation_group() -> usize {
    256
}

impl ComfyQuantizationMetadata {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value: Self = serde_json::from_slice(bytes).map_err(invalid)?;
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<()> {
        if let Self::CodebookW4A8 {
            group_size,
            convrot,
            ..
        } = self
        {
            if !convrot
                || *group_size < 4
                || !(16_usize.is_multiple_of(*group_size) || group_size.is_multiple_of(16))
            {
                return Err(invalid(
                    "W4A8 requires ConvRot and a group size >=4 dividing 16 or a multiple of 16",
                ));
            }
        }
        if let Some(group) = self.rotation_group_size() {
            if group < 4 || !group.is_power_of_two() || !group.trailing_zeros().is_multiple_of(2) {
                return Err(invalid("ConvRot requires a power-of-four group size >=4"));
            }
        }
        Ok(())
    }

    pub fn rotation_group_size(&self) -> Option<usize> {
        match self {
            Self::Int8Tensorwise {
                convrot,
                convrot_groupsize,
            }
            | Self::CodebookW4A8 {
                convrot,
                convrot_groupsize,
                ..
            } => convrot.then_some(*convrot_groupsize),
        }
    }
}

/// Immutable raw tensor view. Construction validates shape and physical bytes;
/// the matrix constructor additionally checks each component's format role.
#[derive(Debug, Clone, Copy)]
pub struct ComfyQuantTensor<'a> {
    dtype: Dtype,
    shape: &'a [u64],
    bytes: &'a [u8],
}

impl<'a> ComfyQuantTensor<'a> {
    pub fn new(dtype: Dtype, shape: &'a [u64], bytes: &'a [u8]) -> Result<Self> {
        let elements = shape.iter().try_fold(1_usize, |count, extent| {
            usize::try_from(*extent)
                .ok()
                .filter(|extent| *extent > 0)
                .and_then(|extent| count.checked_mul(extent))
        });
        let size = match dtype {
            Dtype::I8 | Dtype::U8 | Dtype::F8_E4M3 => 1,
            Dtype::F32 => 4,
            _ => return Err(invalid("unsupported Comfy quantization component dtype")),
        };
        if elements.and_then(|n| n.checked_mul(size)) != Some(bytes.len()) {
            return Err(invalid("component shape, dtype and byte extent differ"));
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
        })
    }

    pub fn from_safetensors(tensor: SafetensorsTensor<'a>) -> Result<Self> {
        Self::new(tensor.dtype(), tensor.shape(), tensor.bytes())
    }

    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub const fn shape(&self) -> &'a [u64] {
        self.shape
    }
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    fn require(&self, dtype: Dtype, shape: &[u64], label: &str) -> Result<()> {
        if self.dtype != dtype || self.shape != shape {
            return Err(invalid(format!(
                "{label} must have dtype {dtype:?} and shape {shape:?}"
            )));
        }
        Ok(())
    }

    fn f32_at(self, index: usize) -> f32 {
        f32::from_le_bytes(self.bytes[index * 4..index * 4 + 4].try_into().unwrap())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ComfyQuantizationComponents<'a> {
    /// Scale is F32 scalar, [N], or [N,1], matching canonical checkpoints.
    Int8 { scale: ComfyQuantTensor<'a> },
    /// Canonical nibble-plane storage, not resident-only MMA-interleaved data.
    /// Asymmetric correction and uniform-code formats are deliberately absent.
    CodebookW4A8 {
        relative_scale: ComfyQuantTensor<'a>,
        channel_scale: ComfyQuantTensor<'a>,
        codebook: ComfyQuantTensor<'a>,
    },
}

/// Validated logical OI matrix borrowing its canonical physical components.
#[derive(Debug)]
pub struct ComfyQuantizedMatrix<'a> {
    metadata: ComfyQuantizationMetadata,
    weight: ComfyQuantTensor<'a>,
    components: ComfyQuantizationComponents<'a>,
    rows: usize,
    columns: usize,
}

impl<'a> ComfyQuantizedMatrix<'a> {
    pub fn new(
        metadata: ComfyQuantizationMetadata,
        weight: ComfyQuantTensor<'a>,
        components: ComfyQuantizationComponents<'a>,
    ) -> Result<Self> {
        metadata.validate()?;
        if weight.dtype != Dtype::I8 || weight.shape.len() != 2 {
            return Err(invalid("canonical quantized weight must be a 2D I8 matrix"));
        }
        let rows = usize::try_from(weight.shape[0]).map_err(invalid)?;
        let packed_columns = usize::try_from(weight.shape[1]).map_err(invalid)?;
        let columns = match (&metadata, components) {
            (
                ComfyQuantizationMetadata::Int8Tensorwise { .. },
                ComfyQuantizationComponents::Int8 { scale },
            ) => {
                if scale.dtype != Dtype::F32
                    || !(scale.shape.is_empty()
                        || scale.shape == [1]
                        || scale.shape == [rows as u64]
                        || scale.shape == [rows as u64, 1])
                {
                    return Err(invalid("INT8 scale must be F32 scalar, [N], or [N,1]"));
                }
                check_f32(scale, true, "INT8 scale")?;
                packed_columns
            }
            (
                ComfyQuantizationMetadata::CodebookW4A8 { group_size, .. },
                ComfyQuantizationComponents::CodebookW4A8 {
                    relative_scale,
                    channel_scale,
                    codebook,
                },
            ) => {
                let columns = packed_columns
                    .checked_mul(2)
                    .ok_or_else(|| invalid("W4A8 logical width overflows"))?;
                if !columns.is_multiple_of(16) || !columns.is_multiple_of(*group_size) {
                    return Err(invalid(
                        "W4A8 logical K must divide into complete 16-wide and scale groups",
                    ));
                }
                relative_scale.require(
                    Dtype::F8_E4M3,
                    &[rows as u64, (columns / group_size) as u64],
                    "W4A8 relative scale",
                )?;
                channel_scale.require(Dtype::F32, &[rows as u64], "W4A8 channel scale")?;
                codebook.require(Dtype::F32, &[16], "W4A8 codebook")?;
                check_f32(channel_scale, true, "W4A8 channel scale")?;
                check_f32(codebook, false, "W4A8 codebook")?;
                // Relative scales can underflow to positive zero in E4M3FN.
                for byte in relative_scale.bytes {
                    let value = decode_e4m3fn(*byte);
                    if !value.is_finite() || value < 0.0 || byte & 0x80 != 0 {
                        return Err(invalid("relative scales must be nonnegative finite E4M3FN"));
                    }
                }
                columns
            }
            _ => {
                return Err(invalid(
                    "metadata and quantization components belong to different formats",
                ))
            }
        };
        if metadata
            .rotation_group_size()
            .is_some_and(|group| !columns.is_multiple_of(group))
        {
            return Err(invalid("ConvRot group size does not divide logical K"));
        }
        rows.checked_mul(columns)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| invalid("logical F32 reference size exceeds address space"))?;
        Ok(Self {
            metadata,
            weight,
            components,
            rows,
            columns,
        })
    }

    /// Inspect a module using its exact original suffixes. Missing metadata,
    /// unknown layouts and correction tensors fail rather than falling back.
    pub fn from_archive(archive: &'a SafetensorsArchive, module: &str) -> Result<Self> {
        let tensor = |suffix: &str| {
            ComfyQuantTensor::from_safetensors(archive.tensor(&format!("{module}.{suffix}"))?)
        };
        let raw_metadata = tensor("comfy_quant")?;
        if raw_metadata.dtype != Dtype::U8 || raw_metadata.shape.len() != 1 {
            return Err(invalid(
                "comfy_quant must be a one-dimensional U8 JSON tensor",
            ));
        }
        let metadata = ComfyQuantizationMetadata::parse(raw_metadata.bytes)?;
        let components = match &metadata {
            ComfyQuantizationMetadata::Int8Tensorwise { .. } => ComfyQuantizationComponents::Int8 {
                scale: tensor("weight_scale")?,
            },
            ComfyQuantizationMetadata::CodebookW4A8 { .. } => {
                let correction_name = format!("{module}.weight_correction");
                if archive.tensor_names().any(|name| name == correction_name) {
                    return Err(invalid("asymmetric W4A8 correction is outside the supported canonical codebook format"));
                }
                ComfyQuantizationComponents::CodebookW4A8 {
                    relative_scale: tensor("weight_s_rel")?,
                    channel_scale: tensor("weight_s_channel")?,
                    codebook: tensor("weight_codebook")?,
                }
            }
        };
        Self::new(metadata, tensor("weight")?, components)
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn columns(&self) -> usize {
        self.columns
    }
    pub const fn metadata(&self) -> &ComfyQuantizationMetadata {
        &self.metadata
    }
    pub const fn weight(&self) -> ComfyQuantTensor<'a> {
        self.weight
    }
    pub const fn components(&self) -> ComfyQuantizationComponents<'a> {
        self.components
    }
}

fn check_f32(tensor: ComfyQuantTensor<'_>, positive: bool, label: &str) -> Result<()> {
    for index in 0..tensor.bytes.len() / 4 {
        let value = tensor.f32_at(index);
        if !value.is_finite() || (positive && value <= 0.0) {
            return Err(invalid(format!(
                "{label} contains a nonfinite or invalid scale"
            )));
        }
    }
    Ok(())
}

/// E4M3FN has no infinities: exponent 15 includes finite values through 448;
/// only mantissa 7 with exponent 15 encodes NaN (either sign).
fn decode_e4m3fn(byte: u8) -> f32 {
    let exponent = (byte >> 3) & 15;
    let mantissa = byte & 7;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else if exponent == 15 && mantissa == 7 {
        f32::NAN
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if byte & 128 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn invalid(reason: impl std::fmt::Display) -> FerrumError {
    FerrumError::model(format!("Comfy ConvRot checkpoint: {reason}"))
}
