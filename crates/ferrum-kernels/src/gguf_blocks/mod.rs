//! Bounded decoding of native GGUF blocks. Callers supply the destination;
//! decoding never expands or caches an entire quantized weight matrix.
//!
//! Layouts and IQ codebooks follow ggml/src/ggml-common.h and ggml-quants.c.
//! Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml for the MIT license.

use ferrum_interfaces::vnext::BlockQuantizationSpec;
use half::f16;

mod iq3s_grid;
pub(crate) use iq3s_grid::IQ3_S_GRID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufBlockFormat {
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    Iq3S,
    Iq4Nl,
    Iq4Xs,
}

impl GgufBlockFormat {
    pub const fn ggml_type_id(self) -> u32 {
        match self {
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8_0 => 8,
            Self::Iq3S => 21,
            Self::Iq4Nl => 20,
            Self::Iq4Xs => 23,
        }
    }

    pub const fn format_id(self) -> &'static str {
        match self {
            Self::Q3K => "quantization.gguf.q3-k",
            Self::Q4K => "quantization.gguf.q4-k",
            Self::Q5K => "quantization.gguf.q5-k",
            Self::Q6K => "quantization.gguf.q6-k",
            Self::Q8_0 => "quantization.gguf.q8-0",
            Self::Iq3S => "quantization.gguf.iq3-s",
            Self::Iq4Nl => "quantization.gguf.iq4-nl",
            Self::Iq4Xs => "quantization.gguf.iq4-xs",
        }
    }

    pub const fn block_values(self) -> usize {
        match self {
            Self::Q8_0 | Self::Iq4Nl => 32,
            _ => 256,
        }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q3K | Self::Iq3S => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8_0 => 34,
            Self::Iq4Nl => 18,
            Self::Iq4Xs => 136,
        }
    }

    pub fn from_spec(spec: &BlockQuantizationSpec) -> Result<Self, String> {
        spec.validate().map_err(|error| error.to_string())?;
        let format = match spec.format_id.as_str() {
            "quantization.gguf.q3-k" => Self::Q3K,
            "quantization.gguf.q4-k" => Self::Q4K,
            "quantization.gguf.q5-k" => Self::Q5K,
            "quantization.gguf.q6-k" => Self::Q6K,
            "quantization.gguf.q8-0" => Self::Q8_0,
            "quantization.gguf.iq3-s" => Self::Iq3S,
            "quantization.gguf.iq4-nl" => Self::Iq4Nl,
            "quantization.gguf.iq4-xs" => Self::Iq4Xs,
            other => return Err(format!("unsupported native GGUF block format {other}")),
        };
        if spec.logical_values_per_block as usize != format.block_values()
            || spec.bytes_per_block as usize != format.block_bytes()
        {
            return Err(format!(
                "native GGUF block ABI differs for {}",
                format.format_id()
            ));
        }
        Ok(format)
    }

    pub fn decode(self, source: &[u8], output: &mut [f32]) -> Result<(), String> {
        if output.is_empty() || !output.len().is_multiple_of(self.block_values()) {
            return Err("native GGUF decode requires complete nonempty logical blocks".into());
        }
        let required = (output.len() / self.block_values())
            .checked_mul(self.block_bytes())
            .ok_or_else(|| "native GGUF block byte length overflows".to_owned())?;
        if required != source.len() {
            return Err(format!(
                "native GGUF decode needs {required} bytes, received {}",
                source.len()
            ));
        }
        for (block, destination) in source
            .chunks_exact(self.block_bytes())
            .zip(output.chunks_exact_mut(self.block_values()))
        {
            for (index, value) in destination.iter_mut().enumerate() {
                *value = self.decode_value(block, index);
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn decode_value(self, block: &[u8], index: usize) -> f32 {
        match self {
            Self::Q3K => {
                let group = index / 16;
                let scales = &block[96..108];
                let low = (scales[group % 8] >> (4 * (group / 8))) & 15;
                let high = (scales[8 + group % 4] >> (2 * (group / 4))) & 3;
                let scale = i32::from(low | (high << 4)) - 32;
                let quant =
                    (block[32 + (index / 128) * 32 + index % 32] >> (2 * ((index % 128) / 32))) & 3;
                let sign_offset = if block[index % 32] & (1 << (index / 32)) == 0 {
                    4
                } else {
                    0
                };
                (half_at(block, 108) * scale as f32) * (i32::from(quant) - sign_offset) as f32
            }
            Self::Q4K | Self::Q5K => {
                let group = index / 32;
                let (scale, minimum) = scale_min(&block[4..16], group);
                let quant_start = if self == Self::Q5K { 48 } else { 16 };
                let mut quant = (block[quant_start + (index / 64) * 32 + index % 32]
                    >> (4 * ((index % 64) / 32)))
                    & 15;
                if self == Self::Q5K && block[16 + index % 32] & (1 << group) != 0 {
                    quant += 16;
                }
                (half_at(block, 0) * f32::from(scale)) * f32::from(quant)
                    - half_at(block, 2) * f32::from(minimum)
            }
            Self::Q6K => {
                let group = (index % 128) / 32;
                let low = (block[(index / 128) * 64 + (group % 2) * 32 + index % 32]
                    >> (4 * (group / 2)))
                    & 15;
                let high = (block[128 + (index / 128) * 32 + index % 32] >> (2 * group)) & 3;
                let quant = i32::from(low | (high << 4)) - 32;
                (half_at(block, 208) * f32::from(block[192 + index / 16] as i8)) * quant as f32
            }
            Self::Q8_0 => half_at(block, 0) * f32::from(block[2 + index] as i8),
            Self::Iq4Nl => {
                let quant = (block[2 + index % 16] >> (4 * (index / 16))) & 15;
                half_at(block, 0) * f32::from(IQ4_NL_VALUES[quant as usize])
            }
            Self::Iq4Xs => {
                let group = index / 32;
                let high_scales = u16::from_le_bytes([block[2], block[3]]);
                let low = (block[4 + group / 2] >> (4 * (group % 2))) & 15;
                let high = ((high_scales >> (2 * group)) & 3) as u8;
                let scale = i32::from(low | (high << 4)) - 32;
                let quant = (block[8 + group * 16 + index % 16] >> (4 * ((index % 32) / 16))) & 15;
                (half_at(block, 0) * scale as f32) * f32::from(IQ4_NL_VALUES[quant as usize])
            }
            Self::Iq3S => {
                let group = index / 32;
                let low = usize::from(block[2 + index / 4]);
                let high = usize::from((block[66 + group] >> ((index % 32) / 4)) & 1);
                let grid = IQ3_S_GRID[low | (high << 8)];
                let quant = ((grid >> (8 * (index % 4))) & 255) as f32;
                let scale = 1 + 2 * ((block[106 + group / 2] >> (4 * (group % 2))) & 15);
                let sign = if block[74 + index / 8] & (1 << (index % 8)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                ((half_at(block, 0) * f32::from(scale)) * quant) * sign
            }
        }
    }
}

pub(crate) const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn half_at(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn scale_min(scales: &[u8], group: usize) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod fixtures;
