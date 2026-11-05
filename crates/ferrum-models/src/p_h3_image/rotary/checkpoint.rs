//! Checkpoint-backed frequencies for the Comfy BF16 DiT profile.
use super::*;
use anyhow::Context;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostTensor, ProgramValueId,
    ResolvedTensorLayout, ResolvedTensorSpec,
};

/// Retains the original archive provenance; no regenerated frequencies or
/// unaccounted copied parameter table is substituted for `rope.inv_freq`.
pub struct PH3ImageCheckpointRotary<'a> {
    source: &'a ComfyCanonicalWeightSource,
    inverse_bytes: &'a [u8],
    geometry: PH3ImageRotaryEmbedding,
}
impl<'a> PH3ImageCheckpointRotary<'a> {
    pub const NUMERICAL_PROFILE_ID: &'static str =
        "p_h3_image.checkpoint_rotary.libm-0.2.15.f32_angle.bf16_frequencies.v1";

    pub fn new(
        config: &PH3ImageTransformerConfig,
        source: &'a ComfyCanonicalWeightSource,
    ) -> Result<Self> {
        let geometry = PH3ImageRotaryEmbedding::new(config)?;
        let tensor = source.archive().tensor("rope.inv_freq")?;
        ensure!(
            tensor.element_type() == Some(ElementType::F32)
                && tensor.shape() == [geometry.frequencies as u64],
            "checkpoint RoPE requires original F32 inv_freq[rope_freq_dim]"
        );
        let inverse_bytes = tensor.bytes();
        ensure!(
            inverse_bytes.len() == geometry.frequencies * 4,
            "checkpoint inverse-frequency byte extent differs"
        );
        ensure!(
            inverse_bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "checkpoint inverse frequencies must be finite"
        );
        Ok(Self {
            source,
            inverse_bytes,
            geometry,
        })
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
    fn inverse(&self, index: usize) -> f32 {
        f32::from_le_bytes(
            self.inverse_bytes[4 * index..4 * index + 4]
                .try_into()
                .unwrap(),
        )
    }

    /// Positions are caller-owned metadata. Each angle rounds the coordinate
    /// to F32 before its separate F32 product with the actual stored inverse
    /// frequency. libm sinf/cosf round once to BF16; this explicit host profile
    /// does not assert bit equality with an installed Torch GPU trig kernel.
    /// Both retained buffers belong to the supplied existing Host account.
    /// Validation or either allocation failing publishes no partial pair.
    pub fn managed_bf16_frequencies(
        &self,
        account: &HostMemoryAccount,
        positions: &[[f64; 3]],
        cosine_id: ProgramValueId,
        sine_id: ProgramValueId,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor)> {
        ensure!(!positions.is_empty(), "RoPE requires a position row");
        ensure!(cosine_id != sine_id, "RoPE output identities must differ");
        let elements = self.geometry.frequency_elements(positions.len())?;
        let bytes = elements
            .checked_mul(2)
            .context("BF16 RoPE extent overflow")?;
        for position in positions {
            for &coordinate in position {
                ensure!(
                    coordinate.is_finite() && (coordinate as f32).is_finite(),
                    "RoPE position must fit finite F32"
                );
                for f in 0..self.geometry.frequencies {
                    ensure!(
                        ((coordinate as f32) * self.inverse(f)).is_finite(),
                        "RoPE angle must be finite F32"
                    );
                }
            }
        }
        let mut cosine = account.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let mut sine = account.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let width = self.geometry.rotary_dim;
        for (row, position) in positions.iter().enumerate() {
            for (axis, &coordinate) in position.iter().enumerate() {
                for f in 0..self.geometry.frequencies {
                    let angle = (coordinate as f32) * self.inverse(f);
                    let c = bf16::from_f32(libm::cosf(angle)).to_bits().to_le_bytes();
                    let s = bf16::from_f32(libm::sinf(angle)).to_bits().to_le_bytes();
                    let first = row * width + axis * self.geometry.frequencies + f;
                    for column in [first, first + width / 2] {
                        cosine.as_mut()[2 * column..2 * column + 2].copy_from_slice(&c);
                        sine.as_mut()[2 * column..2 * column + 2].copy_from_slice(&s);
                    }
                }
            }
        }
        let spec = ResolvedTensorSpec::new(
            vec![u64::try_from(positions.len())?, u64::try_from(width)?],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        Ok((
            ManagedHostTensor::new(cosine_id, spec.clone(), cosine.freeze())?,
            ManagedHostTensor::new(sine_id, spec, sine.freeze())?,
        ))
    }
}

#[cfg(test)]
mod tests;
