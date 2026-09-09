use super::{half_at, scale_min, GgufBlockFormat};

impl GgufBlockFormat {
    /// Decode one already range-checked physical block. Scale extraction is
    /// shared within each quantization group; the F32 arithmetic order matches
    /// the scalar decoder and the GGML dequantization ABI.
    pub(crate) fn decode_block(self, block: &[u8], output: &mut [f32]) {
        assert_eq!(block.len(), self.block_bytes());
        assert_eq!(output.len(), self.block_values());
        match self {
            Self::Q4K | Self::Q5K => {
                let delta = half_at(block, 0);
                let delta_min = half_at(block, 2);
                let quant_start = if self == Self::Q5K { 48 } else { 16 };
                for (group, destination) in output.chunks_exact_mut(32).enumerate() {
                    let (scale, minimum) = scale_min(&block[4..16], group);
                    let scale = delta * f32::from(scale);
                    let minimum = delta_min * f32::from(minimum);
                    let quants = &block[quant_start + group / 2 * 32..][..32];
                    let shift = 4 * (group % 2);
                    if self == Self::Q5K {
                        for (index, value) in destination.iter_mut().enumerate() {
                            let low = (quants[index] >> shift) & 15;
                            let high = ((block[16 + index] >> group) & 1) << 4;
                            *value = scale * f32::from(low | high) - minimum;
                        }
                    } else {
                        for (value, &quant) in destination.iter_mut().zip(quants) {
                            *value = scale * f32::from((quant >> shift) & 15) - minimum;
                        }
                    }
                }
            }
            Self::Q6K => {
                let delta = half_at(block, 208);
                for (group, destination) in output.chunks_exact_mut(16).enumerate() {
                    let scale = delta * f32::from(block[192 + group] as i8);
                    let half = group / 8;
                    let quarter = (group % 8) / 2;
                    let offset = (group % 2) * 16;
                    let low = &block[half * 64 + (quarter % 2) * 32 + offset..][..16];
                    let high = &block[128 + half * 32 + offset..][..16];
                    for ((value, &low), &high) in destination.iter_mut().zip(low).zip(high) {
                        let low = (low >> (4 * (quarter / 2))) & 15;
                        let high = (high >> (2 * quarter)) & 3;
                        *value = scale * (i32::from(low | (high << 4)) - 32) as f32;
                    }
                }
            }
            _ => {
                for (index, value) in output.iter_mut().enumerate() {
                    *value = self.decode_value(block, index);
                }
            }
        }
    }
}
