//! Explicit input-table arithmetic for the native F16 causal profile. THW
//! frequency recomposition precedes half-split rotation in the attention op.
//! This is not a claim of Torch CPU/CUDA libm binary equivalence.
use super::*;
use crate::qwen35_config::Qwen35TextConfig;
use crate::vnext::qwen35::Qwen35DensePrecision;
use half::{bf16, f16};

pub const PH3_IMAGE_EXPANSION_ROTARY_PROFILE: &str =
    "qwen35.thw.interleaved_frequencies.half_split.f32_libm_0_2_15.f16_rne.v1";

#[derive(Clone, Debug)]
pub struct PH3ImagePromptExpansionRotary {
    dimension: usize,
    sections: [usize; 3],
    theta: f32,
}
impl PH3ImagePromptExpansionRotary {
    pub fn new(text: &Qwen35TextConfig) -> Result<Self> {
        let dimension = text.full_attention_rope_dim();
        let sections: [usize; 3] = text
            .rope_parameters
            .mrope_section
            .as_ref()
            .and_then(|s| s.as_slice().try_into().ok())
            .ok_or_else(|| {
                invalid("visual rotary requires three explicit source mRoPE sections")
            })?;
        let theta = text.rope_parameters.rope_theta as f32;
        Self::checked(
            dimension,
            sections,
            theta,
            text.rope_parameters.mrope_interleaved,
        )
    }
    fn checked(
        dimension: usize,
        sections: [usize; 3],
        theta: f32,
        interleaved: bool,
    ) -> Result<Self> {
        if dimension == 0
            || dimension % 2 != 0
            || dimension > 256
            || !theta.is_finite()
            || theta <= 0.
            || !interleaved
            || sections.into_iter().try_fold(0usize, usize::checked_add) != Some(dimension / 2)
            || sections.iter().any(|n| n.checked_mul(3).is_none())
        {
            return Err(invalid(
                "visual rotary requires checked default interleaved THW partial frequencies",
            ));
        }
        let plan = Self {
            dimension,
            sections,
            theta,
        };
        let mut actual = [0usize; 3];
        for pair in 0..plan.pairs() {
            actual[plan.axis(pair)] += 1;
        }
        if actual != sections {
            return Err(invalid(
                "mRoPE sections do not fit their interleaved frequency lanes",
            ));
        }
        Ok(plan)
    }
    pub fn pairs(&self) -> usize {
        self.dimension / 2
    }
    fn axis(&self, pair: usize) -> usize {
        if pair % 3 == 1 && pair < self.sections[1] * 3 {
            1
        } else if pair % 3 == 2 && pair < self.sections[2] * 3 {
            2
        } else {
            0
        }
    }
    /// Writes one cosine and sine row into already admitted backing. Callers
    /// allocate the two exact tables through the original Host account.
    pub fn write_row(&self, position: [u64; 3], cosine: &mut [u8], sine: &mut [u8]) -> Result<()> {
        self.write_row_with_precision(position, Qwen35DensePrecision::F16, cosine, sine)
    }
    /// Round directly from the declared F32 rotary arithmetic to the selected
    /// causal input boundary, without an intermediate F16 conversion.
    pub fn write_row_with_precision(
        &self,
        position: [u64; 3],
        precision: Qwen35DensePrecision,
        cosine: &mut [u8],
        sine: &mut [u8],
    ) -> Result<()> {
        let round = |value| match precision {
            Qwen35DensePrecision::F16 => f16::from_f32(value).to_le_bytes(),
            Qwen35DensePrecision::Bf16 => bf16::from_f32(value).to_le_bytes(),
        };
        if cosine.len() != self.pairs() * 2
            || sine.len() != cosine.len()
            || position.iter().any(|&p| p > 1 << 24)
        {
            return Err(invalid(
                "rotary row has the wrong exact extent or inexact F32 coordinates",
            ));
        }
        for pair in 0..self.pairs() {
            let inverse =
                1.0f32 / libm::powf(self.theta, (2 * pair) as f32 / self.dimension as f32);
            let phase = position[self.axis(pair)] as f32 * inverse;
            let cos = round(libm::cosf(phase));
            let sin = round(libm::sinf(phase));
            cosine[2 * pair..2 * pair + 2].copy_from_slice(&cos);
            sine[2 * pair..2 * pair + 2].copy_from_slice(&sin);
        }
        Ok(())
    }
}
fn invalid(message: &str) -> PH3ImagePromptExpansionError {
    PH3ImagePromptExpansionError::Request(message.to_owned())
}

#[cfg(test)]
mod tests;
