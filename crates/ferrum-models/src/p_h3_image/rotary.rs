//! Three-axis F32 frequencies and leading-channel BF16 half-split RoPE for
//! MiniMax-H3. Coordinate construction remains in `layout`; weights and
//! frequencies never authorize allocations or device execution here.

use anyhow::{ensure, Result};
use half::bf16;

use super::{layout::checked_elements, PH3ImageTransformerConfig};
mod checkpoint;
pub use checkpoint::PH3ImageCheckpointRotary;

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageRotaryEmbedding {
    frequencies: usize,
    rotary_dim: usize,
    head_dim: usize,
    theta: f32,
}

impl PH3ImageRotaryEmbedding {
    pub const NUMERICAL_PROFILE_ID: &'static str =
        "p_h3_image.rotary3axis.libm-0.2.15.f32.partial_half_split.bf16_products.v1";

    pub fn new(config: &PH3ImageTransformerConfig) -> Result<Self> {
        let frequencies = usize::try_from(config.rope_freq_dim())?;
        let rotary_dim = checked_elements::<f32>(&[6, frequencies])?;
        let head_dim = usize::try_from(config.attention_head_dim())?;
        let theta = config.rope_theta() as f32;
        ensure!(
            theta.is_finite() && theta > 0.0,
            "P-H3 RoPE theta is not positive finite FP32"
        );
        ensure!(rotary_dim <= head_dim, "P-H3 RoPE exceeds head width");
        Ok(Self {
            frequencies,
            rotary_dim,
            head_dim,
            theta,
        })
    }
    pub fn rotary_dim(self) -> usize {
        self.rotary_dim
    }
    pub fn head_dim(self) -> usize {
        self.head_dim
    }
    pub fn frequency_elements(self, rows: usize) -> Result<usize> {
        checked_elements::<f32>(&[rows, self.rotary_dim])
    }

    /// Generate `[S,6*rope_freq_dim]` F32 cosine/sine. Each axis contributes
    /// `rope_freq_dim` angles, then the three-axis half is repeated. Coordinates
    /// cast to F32 before multiplication; inverse frequencies are
    /// `1 / theta**(arange(0,2*F,2)/(2*F))`, with FP32 operation boundaries.
    /// libm transcendentals declare a local CPU oracle, not Torch bit parity.
    pub fn write_frequencies(
        self,
        positions: &[[f64; 3]],
        cosine: &mut [f32],
        sine: &mut [f32],
    ) -> Result<()> {
        let elements = self.frequency_elements(positions.len())?;
        ensure!(
            cosine.len() == elements && sine.len() == elements,
            "P-H3 rotary frequency destination differs"
        );
        // Validate every argument before either output is written.
        for position in positions {
            for &coordinate in position {
                ensure!(
                    coordinate.is_finite() && (coordinate as f32).is_finite(),
                    "P-H3 rotary position is not finite FP32"
                );
                for index in 0..self.frequencies {
                    ensure!(
                        self.angle(coordinate as f32, index).is_finite(),
                        "P-H3 rotary angle is not finite"
                    );
                }
            }
        }
        for (row, position) in positions.iter().enumerate() {
            for (axis, &coordinate) in position.iter().enumerate() {
                for index in 0..self.frequencies {
                    let angle = self.angle(coordinate as f32, index);
                    let column = axis * self.frequencies + index;
                    let first = row * self.rotary_dim + column;
                    let second = first + self.rotary_dim / 2;
                    cosine[first] = libm::cosf(angle);
                    cosine[second] = cosine[first];
                    sine[first] = libm::sinf(angle);
                    sine[second] = sine[first];
                }
            }
        }
        Ok(())
    }

    fn angle(self, coordinate: f32, index: usize) -> f32 {
        let exponent = (2 * index) as f32 / (2 * self.frequencies) as f32;
        let inverse = 1.0_f32 / libm::powf(self.theta, exponent);
        coordinate * inverse
    }

    /// Apply the pinned eager BF16 path to `[rows,heads,head_dim]`. The supplied
    /// `[rows,rotary_dim]` F32 frequencies cast to BF16 before multiplication;
    /// each product rounds to BF16, then their sum rounds to BF16. Rotation
    /// splits only the leading rotary_dim channels in half; the remaining
    /// channels pass through bit-for-bit. Frequencies are shared over heads.
    /// Finite inputs/results and exact storage lengths are checked before any
    /// output write. This reference owns no scratch or tensor backing.
    pub fn apply_bf16(
        self,
        rows: usize,
        heads: usize,
        input: &[u16],
        cosine: &[f32],
        sine: &[f32],
        output: &mut [u16],
    ) -> Result<()> {
        ensure!(
            rows > 0 && heads > 0,
            "P-H3 RoPE rows and heads must be positive"
        );
        let elements = checked_elements::<u16>(&[rows, heads, self.head_dim])?;
        let frequencies = self.frequency_elements(rows)?;
        ensure!(
            input.len() == elements
                && output.len() == elements
                && cosine.len() == frequencies
                && sine.len() == frequencies,
            "P-H3 RoPE tensor lengths differ from checked shape"
        );
        ensure!(
            input.iter().all(|bits| bf16::from_bits(*bits).is_finite())
                && cosine.iter().chain(sine).all(|value| value.is_finite()),
            "P-H3 RoPE inputs must be finite"
        );
        for index in 0..elements {
            ensure!(
                bf16::from_bits(self.rotate(index, heads, input, cosine, sine)).is_finite(),
                "P-H3 BF16 rotary result is not finite"
            );
        }
        for (index, target) in output.iter_mut().enumerate() {
            *target = self.rotate(index, heads, input, cosine, sine);
        }
        Ok(())
    }

    fn rotate(
        self,
        index: usize,
        heads: usize,
        input: &[u16],
        cosine: &[f32],
        sine: &[f32],
    ) -> u16 {
        let channel = index % self.head_dim;
        if channel >= self.rotary_dim {
            return input[index];
        }
        let row = index / (heads * self.head_dim);
        let frequency = row * self.rotary_dim + channel;
        let half = self.rotary_dim / 2;
        let (partner, sign) = if channel < half {
            (index + half, -1.0_f32)
        } else {
            (index - half, 1.0_f32)
        };
        let cos = bf16::from_f32(cosine[frequency]).to_f32();
        let sin = bf16::from_f32(sine[frequency]).to_f32();
        let original = bf16::from_f32(bf16::from_bits(input[index]).to_f32() * cos).to_f32();
        let rotated =
            bf16::from_f32((sign * bf16::from_bits(input[partner]).to_f32()) * sin).to_f32();
        bf16::from_f32(original + rotated).to_bits()
    }
}

#[cfg(test)]
mod tests;
