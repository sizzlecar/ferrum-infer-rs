use anyhow::{ensure, Context, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostTensor, ProgramValueId,
    ResolvedTensorLayout, ResolvedTensorSpec,
};
use half::bf16;

use super::FastH3PositionPlan;
use crate::fast_h3::FastH3TransformerConfig;

impl FastH3PositionPlan {
    /// Analytic inverse frequencies, F32 angle, libm trig, then BF16. This
    /// declares the host arithmetic and does not assert Torch CUDA bit parity.
    pub const ROTARY_NUMERICAL_PROFILE_ID: &'static str =
        "fast_h3.ref2va.analytic_rotary.libm-0.2.15.f32_angle.bf16.v1";

    /// Add the explicit B1 request axis required by the denoising/backbone
    /// input contract. Both views retain the exact Input owners, IDs, BF16
    /// encodings and byte ranges; this does not copy or reclassify storage.
    pub fn single_request_bf16_rotary_inputs(
        &self,
        config: &FastH3TransformerConfig,
        frequencies: [ManagedHostTensor; 2],
    ) -> Result<[ManagedHostTensor; 2]> {
        let rows = u64::try_from(self.layout.sequence_rows())?;
        let width = u64::try_from(config.rotary_dim())?;
        ensure!(
            frequencies[0].value_id() != frequencies[1].value_id(),
            "RoPE output identities must differ"
        );
        for input in &frequencies {
            ensure!(
                input.tensor().dimensions() == [rows, width]
                    && input.tensor().element_type() == ElementType::Bf16
                    && input.tensor().layout() == &ResolvedTensorLayout::Contiguous
                    && input.bytes().class() == HostMemoryClass::InputPreprocess,
                "single-request RoPE view requires the original packed BF16 Input storage"
            );
        }
        let spec = ResolvedTensorSpec::new(
            vec![1, rows, width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let [cosine, sine] = frequencies;
        Ok([
            ManagedHostTensor::new(
                cosine.value_id().clone(),
                spec.clone(),
                cosine.bytes().clone(),
            )?,
            ManagedHostTensor::new(sine.value_id().clone(), spec, sine.bytes().clone())?,
        ])
    }

    pub fn managed_bf16_frequencies(
        &self,
        config: &FastH3TransformerConfig,
        account: &HostMemoryAccount,
        cosine_id: ProgramValueId,
        sine_id: ProgramValueId,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor)> {
        ensure!(cosine_id != sine_id, "RoPE output identities must differ");
        let rows = self.layout.sequence_rows();
        let width = config.rotary_dim();
        let frequencies = config.rope_freq_dim();
        let theta = config.rope_theta() as f32;
        ensure!(
            theta.is_finite() && theta > 0.0,
            "RoPE theta must fit positive F32"
        );
        let bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(2))
            .context("BF16 RoPE extent overflow")?;
        let mut cosine = account.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let mut sine = account.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        // Frequency-major filling avoids an extra unmanaged parameter/position
        // tensor, and computes pow only once for each inverse frequency.
        for f in 0..frequencies {
            let exponent = (2 * f) as f32 / (2 * frequencies) as f32;
            let inverse = 1.0_f32 / libm::powf(theta, exponent);
            ensure!(
                inverse.is_finite() && inverse > 0.0,
                "RoPE inverse frequency must be positive finite F32"
            );
            for (row, position) in self.positions().enumerate() {
                for (axis, coordinate) in position.into_iter().enumerate() {
                    let angle = coordinate as f32 * inverse;
                    ensure!(angle.is_finite(), "RoPE angle must fit finite F32");
                    let c = bf16::from_f32(libm::cosf(angle)).to_bits().to_le_bytes();
                    let s = bf16::from_f32(libm::sinf(angle)).to_bits().to_le_bytes();
                    let first = row * width + axis * frequencies + f;
                    for column in [first, first + width / 2] {
                        cosine.as_mut()[2 * column..2 * column + 2].copy_from_slice(&c);
                        sine.as_mut()[2 * column..2 * column + 2].copy_from_slice(&s);
                    }
                }
            }
        }
        let spec = ResolvedTensorSpec::new(
            vec![u64::try_from(rows)?, u64::try_from(width)?],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        Ok((
            ManagedHostTensor::new(cosine_id, spec.clone(), cosine.freeze())?,
            ManagedHostTensor::new(sine_id, spec, sine.freeze())?,
        ))
    }
}
