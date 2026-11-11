//! Scalar CPU oracles for independent BF16/A8 and full-F32/A8 paths.
//! Source: pinned comfy-kitchen `backends/cuda/ops/int8_linear.cu` (module docs).
//! This computes integer products on CPU and does not establish CUDA parity.
use super::*;
use half::bf16;

#[derive(Debug)]
pub struct ComfyA8Bf16Reference {
    pub output_bf16: Vec<u16>,
    pub activation_i8: Vec<i8>,
    pub row_scales: Vec<f32>,
}
#[derive(Debug)]
pub struct ComfyA8F32Reference {
    pub output_f32: Vec<f32>,
    pub activation_i8: Vec<i8>,
    pub row_scales: Vec<f32>,
}

impl ComfyQuantizedMatrix<'_> {
    /// Independent F32 activation/output oracle. H4 stages, row scale and
    /// numerator/denominator/quotient remain F32, with no BF16 conversions.
    /// Uses nearest-even F32 division and no FMA/FTZ; finite inputs and finite
    /// intermediates are this CPU reference's domain, not a device data scan.
    /// Checkpoint validation/decoding and exact I32 products are shared with
    /// the BF16 oracle, but its activation/output rounding is not selected.
    pub fn linear_f32_a8_reference(
        &self,
        activation: &[f32],
        rows: usize,
    ) -> Result<ComfyA8F32Reference> {
        if rows == 0 || rows.checked_mul(self.columns) != Some(activation.len()) {
            return Err(invalid("activation shape differs from F32 [M,K]"));
        }
        if activation.iter().any(|v| !v.is_finite()) {
            return Err(invalid("F32 A8 reference requires finite activation"));
        }
        let mut rotated = Vec::new();
        rotated
            .try_reserve_exact(activation.len())
            .map_err(invalid)?;
        rotated.extend_from_slice(activation);
        self.linear_a8_reference(rotated, rows, false)
    }
    /// Online regular ConvRot, dynamic per-row A8, exact INT32 dot product,
    /// then RN32(acc * row_scale), RN32(* channel_scale), BF16 RNE.
    /// Bias/correction and stochastic activation rounding are not supported.
    /// The returned intermediate values allow kernel qualification to isolate
    /// activation quantization from integer GEMM and output rounding.
    pub fn linear_bf16_a8_reference(
        &self,
        activation: &[u16],
        rows: usize,
    ) -> Result<ComfyA8Bf16Reference> {
        if rows == 0 || rows.checked_mul(self.columns) != Some(activation.len()) {
            return Err(invalid("activation shape differs from BF16 [M,K]"));
        }
        let mut rotated = Vec::new();
        rotated
            .try_reserve_exact(activation.len())
            .map_err(invalid)?;
        for bits in activation {
            let value = bf16::from_bits(*bits).to_f32();
            if !value.is_finite() {
                return Err(invalid("A8 reference requires finite BF16 activation"));
            }
            rotated.push(value);
        }
        self.linear_f32_activation_a8_reference(rotated, rows)
    }

    /// BF16 gate|up [M,2K] -> F32 SiLU(gate)*up with no intermediate BF16
    /// conversion, then the same H256/A8/integer-dot pipeline. This is a
    /// mathematical CPU oracle using Rust's F32 exponential; its libm does
    /// not establish bitwise parity with CUDA libdevice or upstream fast math.
    /// Row count never changes the activation arithmetic in this reference.
    pub fn linear_swiglu_bf16_a8_reference(
        &self,
        gate_up: &[u16],
        rows: usize,
    ) -> Result<ComfyA8Bf16Reference> {
        let width = self
            .columns
            .checked_mul(2)
            .ok_or_else(|| invalid("gate|up width overflows"))?;
        if rows == 0 || rows.checked_mul(width) != Some(gate_up.len()) {
            return Err(invalid("activation shape differs from BF16 gate|up [M,2K]"));
        }
        let mut activated = Vec::new();
        activated
            .try_reserve_exact(gate_up.len() / 2)
            .map_err(invalid)?;
        for row in gate_up.chunks_exact(width) {
            for column in 0..self.columns {
                let gate = bf16::from_bits(row[column]).to_f32();
                let up = bf16::from_bits(row[self.columns + column]).to_f32();
                if !gate.is_finite() || !up.is_finite() {
                    return Err(invalid("SwiGLU reference requires finite BF16 inputs"));
                }
                let value = (gate / (1.0_f32 + (-gate).exp())) * up;
                if !value.is_finite() {
                    return Err(invalid("SwiGLU F32 intermediate overflow"));
                }
                activated.push(value);
            }
        }
        self.linear_f32_activation_a8_reference(activated, rows)
    }

    fn linear_f32_activation_a8_reference(
        &self,
        rotated: Vec<f32>,
        rows: usize,
    ) -> Result<ComfyA8Bf16Reference> {
        let reference = self.linear_a8_reference(rotated, rows, true)?;
        let mut output_bf16 = Vec::new();
        output_bf16
            .try_reserve_exact(reference.output_f32.len())
            .map_err(invalid)?;
        for value in reference.output_f32 {
            let value = bf16::from_f32(value);
            if !value.is_finite() {
                return Err(invalid("BF16 A8 reference output is nonfinite"));
            }
            output_bf16.push(value.to_bits());
        }
        Ok(ComfyA8Bf16Reference {
            output_bf16,
            activation_i8: reference.activation_i8,
            row_scales: reference.row_scales,
        })
    }

    fn linear_a8_reference(
        &self,
        mut rotated: Vec<f32>,
        rows: usize,
        bf16_boundary: bool,
    ) -> Result<ComfyA8F32Reference> {
        let count = rows
            .checked_mul(self.rows)
            .ok_or_else(|| invalid("reference output size overflows"))?;
        if let Some(group) = self.metadata.rotation_group_size() {
            for values in rotated.chunks_exact_mut(group) {
                rotate_staged(values)?;
            }
        }
        let mut activation_i8 = Vec::new();
        activation_i8
            .try_reserve_exact(rotated.len())
            .map_err(invalid)?;
        let mut row_scales = Vec::new();
        row_scales.try_reserve_exact(rows).map_err(invalid)?;
        for row in rotated.chunks_exact(self.columns) {
            let maximum = row.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            // The CUDA kernel clamps amax to the input dtype's finite maximum;
            // unlike the eager fallback it does not round amax to BF16 first.
            let finite_max = if bf16_boundary {
                bf16::MAX.to_f32()
            } else {
                f32::MAX
            };
            let scale = (maximum.min(finite_max) * (1.0_f32 / 127.0_f32)).max(1.0e-30_f32);
            let denominator = if bf16_boundary {
                bf16::from_f32(scale).to_f32()
            } else {
                scale
            };
            for value in row {
                let numerator = if bf16_boundary {
                    bf16::from_f32(*value).to_f32()
                } else {
                    *value
                };
                let quotient = if bf16_boundary {
                    bf16::from_f32(numerator / denominator).to_f32()
                } else {
                    numerator / denominator
                };
                activation_i8.push(quotient.round_ties_even().clamp(-128.0, 127.0) as i8);
            }
            row_scales.push(scale);
        }
        let weight = self.decoded_rotated_i8_reference()?;
        let mut output_f32 = Vec::new();
        output_f32.try_reserve_exact(count).map_err(invalid)?;
        for row in 0..rows {
            for output in 0..self.rows {
                let mut acc = 0_i32;
                for column in 0..self.columns {
                    let product = i32::from(activation_i8[row * self.columns + column])
                        * i32::from(weight[output * self.columns + column]);
                    acc = acc
                        .checked_add(product)
                        .ok_or_else(|| invalid("INT32 reference accumulation overflow"))?;
                }
                let scale = match self.components {
                    ComfyQuantizationComponents::Int8 { scale } => {
                        scale.f32_at(if scale.bytes.len() == 4 { 0 } else { output })
                    }
                    ComfyQuantizationComponents::CodebookW4A8 { channel_scale, .. } => {
                        channel_scale.f32_at(output)
                    }
                };
                let value = ((acc as f32) * row_scales[row]) * scale;
                if !value.is_finite() {
                    return Err(invalid("A8 reference output is nonfinite"));
                }
                output_f32.push(value);
            }
        }
        Ok(ComfyA8F32Reference {
            output_f32,
            activation_i8,
            row_scales,
        })
    }
}

pub(super) fn rotate_staged(values: &mut [f32]) -> Result<()> {
    let mut stride = 1;
    while stride < values.len() {
        for start in (0..values.len()).step_by(4 * stride) {
            for offset in 0..stride {
                let a = values[start + offset];
                let b = values[start + offset + stride];
                let c = values[start + offset + 2 * stride];
                let d = values[start + offset + 3 * stride];
                values[start + offset] = 0.5 * ((a + b) + c - d);
                values[start + offset + stride] = 0.5 * ((a + b) - c + d);
                values[start + offset + 2 * stride] = 0.5 * ((a - b) + c + d);
                values[start + offset + 3 * stride] = 0.5 * ((-a + b) + c + d);
            }
        }
        stride *= 4;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("ConvRot F32 intermediate overflow"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor<'a>(dtype: Dtype, shape: &'a [u64], bytes: &'a [u8]) -> ComfyQuantTensor<'a> {
        ComfyQuantTensor::new(dtype, shape, bytes).unwrap()
    }

    #[test]
    fn activation_rows_quantize_independently_and_zero_rows_remain_finite() {
        let packed = [0_u8; 128];
        let relative = [0x38_u8; 16];
        let channel = 0.5_f32.to_le_bytes();
        let codebook: Vec<u8> = [1.0_f32; 16]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let matrix = ComfyQuantizedMatrix::new(
            ComfyQuantizationMetadata::CodebookW4A8 {
                group_size: 16,
                convrot: true,
                convrot_groupsize: 256,
            },
            tensor(Dtype::I8, &[1, 128], &packed),
            ComfyQuantizationComponents::CodebookW4A8 {
                relative_scale: tensor(Dtype::F8_E4M3, &[1, 16], &relative),
                channel_scale: tensor(Dtype::F32, &[1], &channel),
                codebook: tensor(Dtype::F32, &[16], &codebook),
            },
        )
        .unwrap();
        let activation: Vec<u16> = [1.0_f32, -0.5, 0.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(bf16::from_f32(value).to_bits(), 256))
            .collect();
        let result = matrix.linear_bf16_a8_reference(&activation, 3).unwrap();
        assert_eq!(
            result.output_bf16,
            [128.0_f32, -64.0, 0.0].map(|value| bf16::from_f32(value).to_bits())
        );
        assert!(result.activation_i8[..256]
            .iter()
            .all(|value| *value == 127));
        assert!(result.activation_i8[256..512]
            .iter()
            .all(|value| *value == -127));
        assert!(result.activation_i8[512..].iter().all(|value| *value == 0));
        assert_eq!(
            result.row_scales,
            [1.0_f32 / 127.0, 0.5_f32 / 127.0, 1.0e-30]
        );
        assert!(matrix
            .linear_bf16_a8_reference(&activation[..255], 1)
            .is_err());
        let mut nonfinite = activation[..256].to_vec();
        nonfinite[0] = bf16::NAN.to_bits();
        assert!(matrix.linear_bf16_a8_reference(&nonfinite, 1).is_err());
    }

    #[test]
    fn staged_rotation_is_regular_and_uses_each_radix4_boundary() {
        let mut row = [1.0, 0.0, 0.0, 0.0];
        rotate_staged(&mut row).unwrap();
        assert_eq!(row, [0.5, 0.5, 0.5, -0.5]);
        rotate_staged(&mut row).unwrap();
        assert_eq!(row, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn signed_int8_weights_include_minimum_code_and_per_output_scales() {
        let weight: Vec<u8> = [-128_i8, 127, -2, 0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value as u8, 256))
            .collect();
        let scale: Vec<u8> = [0.5_f32, 0.25, 0.125, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let matrix = ComfyQuantizedMatrix::new(
            ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: 256,
            },
            tensor(Dtype::I8, &[4, 256], &weight),
            ComfyQuantizationComponents::Int8 {
                scale: tensor(Dtype::F32, &[4, 1], &scale),
            },
        )
        .unwrap();
        let activation: Vec<u16> = [1.0_f32, -0.5, 0.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(bf16::from_f32(value).to_bits(), 256))
            .collect();
        let result = matrix.linear_bf16_a8_reference(&activation, 3).unwrap();
        let expected = [
            -16384.0_f32,
            8128.0,
            -64.0,
            0.0,
            8192.0,
            -4064.0,
            32.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        .map(|value| bf16::from_f32(value).to_bits());
        assert_eq!(result.output_bf16, expected);
        assert_eq!(matrix.decoded_rotated_i8_reference().unwrap()[0], -128);
    }

    #[test]
    fn fused_swiglu_preserves_f32_product_and_does_not_switch_at_512_rows() {
        let packed = [0_u8; 128];
        let relative = [0x38_u8; 16];
        let channel = 0.5_f32.to_le_bytes();
        let codebook: Vec<u8> = [1.0_f32; 16]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let weight = [1_u8; 256];
        for components in [
            ComfyQuantizationComponents::CodebookW4A8 {
                relative_scale: tensor(Dtype::F8_E4M3, &[1, 16], &relative),
                channel_scale: tensor(Dtype::F32, &[1], &channel),
                codebook: tensor(Dtype::F32, &[16], &codebook),
            },
            ComfyQuantizationComponents::Int8 {
                scale: tensor(Dtype::F32, &[1, 1], &channel),
            },
        ] {
            let (metadata, weights) = match components {
                ComfyQuantizationComponents::CodebookW4A8 { .. } => (
                    ComfyQuantizationMetadata::CodebookW4A8 {
                        group_size: 16,
                        convrot: true,
                        convrot_groupsize: 256,
                    },
                    tensor(Dtype::I8, &[1, 128], &packed),
                ),
                ComfyQuantizationComponents::Int8 { .. } => (
                    ComfyQuantizationMetadata::Int8Tensorwise {
                        convrot: true,
                        convrot_groupsize: 256,
                    },
                    tensor(Dtype::I8, &[1, 256], &weight),
                ),
            };
            let matrix = ComfyQuantizedMatrix::new(metadata, weights, components).unwrap();
            let row: Vec<u16> = [0.5_f32, 3.0]
                .into_iter()
                .flat_map(|value| std::iter::repeat_n(bf16::from_f32(value).to_bits(), 256))
                .collect();
            let fused = matrix.linear_swiglu_bf16_a8_reference(&row, 1).unwrap();
            let silu = bf16::from_f32(0.5_f32 / (1.0_f32 + (-0.5_f32).exp()));
            let staged = vec![bf16::from_f32(silu.to_f32() * 3.0).to_bits(); 256];
            let raw = matrix.linear_bf16_a8_reference(&staged, 1).unwrap();
            assert_eq!(bf16::from_bits(raw.output_bf16[0]).to_f32(), 119.0);
            assert_eq!(bf16::from_bits(fused.output_bf16[0]).to_f32(), 119.5);
            assert_ne!(fused.row_scales, raw.row_scales);
            let rows512 = row.repeat(512);
            let a = matrix
                .linear_swiglu_bf16_a8_reference(&rows512[..511 * 512], 511)
                .unwrap();
            let b = matrix
                .linear_swiglu_bf16_a8_reference(&rows512, 512)
                .unwrap();
            assert_eq!(a.output_bf16, b.output_bf16[..511]);
            assert_eq!(a.row_scales, b.row_scales[..511]);
            assert_eq!(a.activation_i8, b.activation_i8[..511 * 256]);
            assert!(matrix
                .linear_swiglu_bf16_a8_reference(&row[..256], 1)
                .is_err());
            let mut invalid = row;
            invalid[0] = bf16::NAN.to_bits();
            assert!(matrix.linear_swiglu_bf16_a8_reference(&invalid, 1).is_err());
        }
    }

    #[test]
    fn full_f32_a8_preserves_scale_quotient_and_output_without_bf16_boundaries() {
        let packed = [0_u8; 256];
        let relative = [0x38_u8; 32];
        let channel = 0.5_f32.to_le_bytes();
        let table: Vec<u8> = [1.0_f32; 16]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let weight = [1_u8; 512];
        for (metadata, weights, components) in [
            (
                ComfyQuantizationMetadata::CodebookW4A8 {
                    group_size: 16,
                    convrot: true,
                    convrot_groupsize: 256,
                },
                tensor(Dtype::I8, &[1, 256], &packed),
                ComfyQuantizationComponents::CodebookW4A8 {
                    relative_scale: tensor(Dtype::F8_E4M3, &[1, 32], &relative),
                    channel_scale: tensor(Dtype::F32, &[1], &channel),
                    codebook: tensor(Dtype::F32, &[16], &table),
                },
            ),
            (
                ComfyQuantizationMetadata::Int8Tensorwise {
                    convrot: true,
                    convrot_groupsize: 256,
                },
                tensor(Dtype::I8, &[1, 512], &weight),
                ComfyQuantizationComponents::Int8 {
                    scale: tensor(Dtype::F32, &[1, 1], &channel),
                },
            ),
        ] {
            let matrix = ComfyQuantizedMatrix::new(metadata, weights, components).unwrap();
            let a = 1.000244140625_f32;
            let b = (63.4_f32 / 127.0) * a;
            let activation: Vec<f32> = [1.0_f32, -0.5, 0.0]
                .into_iter()
                .flat_map(|sign| {
                    [a * sign, b * sign]
                        .into_iter()
                        .flat_map(|value| std::iter::repeat_n(value, 256))
                })
                .collect();
            let f = matrix.linear_f32_a8_reference(&activation, 3).unwrap();
            assert!(f.activation_i8[..256].iter().all(|q| *q == 127));
            assert!(f.activation_i8[256..512].iter().all(|q| *q == 63));
            assert!(f.activation_i8[512..768].iter().all(|q| *q == -127));
            assert!(f.activation_i8[768..1024].iter().all(|q| *q == -63));
            assert!(f.activation_i8[1024..].iter().all(|q| *q == 0));
            let scale = a * (1.0_f32 / 127.0);
            let expected = 48640.0_f32 * scale * 0.5;
            assert_eq!(f.output_f32, [expected, -0.5 * expected, 0.0]);
            assert_eq!(f.row_scales, [scale, scale * 0.5, 1.0e-30]);
            assert_ne!(bf16::from_f32(expected).to_f32(), expected);
            let downcast: Vec<u16> = activation
                .iter()
                .map(|v| bf16::from_f32(*v).to_bits())
                .collect();
            let b = matrix.linear_bf16_a8_reference(&downcast, 3).unwrap();
            assert!(b.activation_i8[256..512].iter().all(|q| *q == 64));
            assert_ne!(f.row_scales, b.row_scales);
            assert!(matrix
                .linear_f32_a8_reference(&activation[..511], 1)
                .is_err());
            let mut nonfinite = activation[..512].to_vec();
            nonfinite[0] = f32::NAN;
            assert!(matrix.linear_f32_a8_reference(&nonfinite, 1).is_err());
            let tiny = matrix.linear_f32_a8_reference(&[1.0e-32; 512], 1).unwrap();
            assert_eq!(tiny.row_scales, [1.0e-30]);
            assert!(tiny.activation_i8.iter().all(|q| *q == 0));
            assert_eq!(tiny.output_f32, [0.0]);
        }
    }
}
