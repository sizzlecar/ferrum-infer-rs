//! CPU input adaptation for the homogeneous, single-frame T2I transformer.
//!
//! The structural reference is vLLM-Omni c0551d22674d0db81305c800b66ad83035fdc980,
//! `vllm_omni/diffusion/models/qwen_image_21/qwen_image_21_transformer.py`:
//! TemporalTimesteps L167–185, final feature cast L219–221, Rope L349–414,
//! and the early timestep cast / single shared t0 at L1176–1185.
//! `pipeline_qwen_image_21.py` L1380 / L1434 additionally casts the scheduler
//! timestep to BF16 *before* division by 1000. Those two casts cannot cancel.
//!
//! This implementation declares a libm-0.2.15 F32 profile. Torch CPU `exp`,
//! `pow`, `polar` and device `sin`/`cos` have separate implementation identities;
//! no Torch/libdevice bitwise equivalence is claimed. Output storage belongs
//! to the caller and must enter execution through managed input admission.

use super::{invalid, QwenImage21T2iLayout, QwenImage21TransformerConfig};
use ferrum_runtime::VNextError;
use half::bf16;

/// BF16 timestep at the transformer's normalized input boundary, in [0, 1].
/// A checked value cannot accidentally be interpreted as a raw 0–1000 step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenImage21ModelTimestep(u16);

impl QwenImage21ModelTimestep {
    pub fn from_normalized(value: f32) -> Result<Self, VNextError> {
        checked_range(value, 1.0, "normalized timestep")?;
        Ok(Self(bf16::from_f32(value).to_bits()))
    }

    /// The pipeline bridge is B(RN32(B(t) / 1000)), not B(t / 1000).
    /// The upstream scheduler value itself has already been chosen by its
    /// schedule; this method neither creates a schedule nor advances latents.
    pub fn from_scheduler_timestep(value: f32) -> Result<Self, VNextError> {
        checked_range(value, 1000.0, "scheduler timestep")?;
        Self::from_normalized(bf16::from_f32(value).to_f32() / 1000.0_f32)
    }

    pub const fn bf16_bits(self) -> u16 {
        self.0
    }
    pub fn as_f32(self) -> f32 {
        bf16::from_bits(self.0).to_f32()
    }
}

fn checked_range(value: f32, upper: f32, label: &str) -> Result<(), VNextError> {
    if !value.is_finite() || !(0.0..=upper).contains(&value) {
        return Err(invalid(
            "conditioning",
            format!("{label} must be finite and in [0, {upper}]"),
        ));
    }
    Ok(())
}

/// Fixed 256-channel cos-then-sin projection before the learned time MLP.
/// Its only owned numerical storage is a bounded 128-element frequency table.
#[derive(Debug, Clone)]
pub struct QwenImage21Timestep256 {
    frequencies: [f32; 128],
}

impl QwenImage21Timestep256 {
    pub const WIDTH: usize = 256;
    pub const NUMERICAL_PROFILE_ID: &'static str =
        "qwen_image21.timestep256.bf16_input.libm-0.2.15.f32.bf16_output.v1";

    pub fn new() -> Result<Self, VNextError> {
        let mut frequencies = [0.0; 128];
        // Match the expression order: F32 scalar-times-arange, then /128.
        // Python's F64 math.log scalar enters an F32 torch tensor operation.
        let negative_log = -(libm::log(10000.0_f64) as f32);
        for (index, frequency) in frequencies.iter_mut().enumerate() {
            let exponent = (negative_log * index as f32) / 128.0_f32;
            *frequency = libm::expf(exponent);
            if !frequency.is_finite() || *frequency <= 0.0 {
                return Err(invalid("conditioning", "nonfinite timestep frequency"));
            }
        }
        Ok(Self { frequencies })
    }

    pub fn output_elements(config: &QwenImage21TransformerConfig) -> Result<usize, VNextError> {
        checked_elements::<u16>(config.time_feature_rows(), Self::WIDTH as u64)
    }

    /// Writes BF16 [B + causal_condition, 256]. Causal mode appends exactly
    /// one shared t=0 row. Invalid lengths leave the destination unchanged.
    pub fn write_into(
        &self,
        config: &QwenImage21TransformerConfig,
        timesteps: &[QwenImage21ModelTimestep],
        output: &mut [u16],
    ) -> Result<(), VNextError> {
        let expected = Self::output_elements(config)?;
        if timesteps.len() as u64 != config.layout().batch() || output.len() != expected {
            return Err(invalid(
                "conditioning",
                "timestep rows or feature destination length differs from the typed layout",
            ));
        }
        for (row, destination) in output.chunks_exact_mut(Self::WIDTH).enumerate() {
            let timestep = timesteps.get(row).map_or(0.0, |value| value.as_f32());
            let time = 1000.0_f32 * timestep;
            for (index, frequency) in self.frequencies.iter().enumerate() {
                let angle = time * *frequency;
                // Checked t bounds and positive finite frequency make every
                // argument finite. libm sinf/cosf are finite on that domain.
                destination[index] = bf16::from_f32(libm::cosf(angle)).to_bits();
                destination[128 + index] = bf16::from_f32(libm::sinf(angle)).to_bits();
            }
        }
        Ok(())
    }
}

/// Three-axis complex frequencies, represented as contiguous F32 [S,D/2,2].
/// Each pair stores real (cos), imaginary (sin); axis order is frame/height/width.
/// This descriptor owns no payload buffer and never repeats frequencies over B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenImage21RotaryFrequencies {
    axes: [u32; 3],
    width: u32,
}

impl QwenImage21RotaryFrequencies {
    pub const NUMERICAL_PROFILE_ID: &'static str =
        "qwen_image21.rotary3axis.libm-0.2.15.powf_recip_f32.sincosf.v1";

    pub fn new(axes: [u64; 3]) -> Result<Self, VNextError> {
        if axes
            .iter()
            .any(|axis| *axis == 0 || axis % 2 != 0 || *axis > (1 << 24))
        {
            return Err(invalid(
                "conditioning",
                "rotary axes must be positive even integers representable exactly as F32",
            ));
        }
        let width = axes
            .iter()
            .try_fold(0_u64, |sum, value| sum.checked_add(*value))
            .and_then(|sum| u32::try_from(sum).ok())
            .ok_or_else(|| invalid("conditioning", "rotary channel sum overflows"))?;
        Ok(Self {
            axes: axes.map(|axis| axis as u32),
            width,
        })
    }

    pub const fn axes(self) -> [u32; 3] {
        self.axes
    }

    pub fn output_elements(self, layout: &QwenImage21T2iLayout) -> Result<usize, VNextError> {
        checked_elements::<f32>(layout.sequence_length(), u64::from(self.width))
    }

    /// Uses padded text positions, including invalid text keys. Target frame
    /// position is padded T; centered height/width are independent of T.
    /// All length/geometry checks precede writes; no input-sized scratch exists.
    pub fn write_into(
        self,
        layout: &QwenImage21T2iLayout,
        output: &mut [f32],
    ) -> Result<(), VNextError> {
        let expected = self.output_elements(layout)?;
        if output.len() != expected {
            return Err(invalid(
                "conditioning",
                "rotary destination length differs from [S,D/2,2]",
            ));
        }
        // Validate the bounded frequency domain before any writes. Positions
        // in the checked layout are within [-1024,8191], so angles and all
        // subsequent trig results are finite for every accepted frequency.
        for dimension in self.axes {
            for pair in 0..dimension / 2 {
                let frequency = angular_frequency(pair, dimension);
                if !frequency.is_finite() || frequency <= 0.0 || frequency > 1.0 {
                    return Err(invalid("conditioning", "invalid rotary angular frequency"));
                }
            }
        }
        let stride = self.width as usize;
        let mut axis_start = 0_usize;
        for (axis, dimension) in self.axes.into_iter().enumerate() {
            for pair in 0..dimension / 2 {
                let frequency = angular_frequency(pair, dimension);
                for (row, destination) in output.chunks_exact_mut(stride).enumerate() {
                    let position = layout
                        .rotary_position(row as u64)
                        .expect("exact output length is the checked layout sequence length")[axis];
                    let angle = position as f32 * frequency;
                    let offset = axis_start + 2 * pair as usize;
                    destination[offset] = libm::cosf(angle);
                    destination[offset + 1] = libm::sinf(angle);
                }
            }
            axis_start += dimension as usize;
        }
        Ok(())
    }
}

fn angular_frequency(pair: u32, dimension: u32) -> f32 {
    let exponent = (2 * pair) as f32 / dimension as f32;
    // Keep pow then reciprocal: exp(-log(theta)*exponent) changes rounding.
    1.0_f32 / libm::powf(10000.0_f32, exponent)
}

fn checked_elements<T>(rows: u64, width: u64) -> Result<usize, VNextError> {
    rows.checked_mul(width)
        .filter(|elements| {
            elements
                .checked_mul(std::mem::size_of::<T>() as u64)
                .is_some_and(|bytes| bytes <= isize::MAX as u64)
        })
        .and_then(|elements| usize::try_from(elements).ok())
        .ok_or_else(|| {
            invalid(
                "conditioning",
                "conditioning output exceeds addressable storage",
            )
        })
}

impl QwenImage21TransformerConfig {
    pub fn rotary_frequencies(&self) -> Result<QwenImage21RotaryFrequencies, VNextError> {
        QwenImage21RotaryFrequencies::new(*self.transformer().rope_axes())
    }
}

#[cfg(test)]
#[path = "conditioning_tests.rs"]
mod tests;
