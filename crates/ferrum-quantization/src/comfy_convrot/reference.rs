use super::*;

impl ComfyQuantizedMatrix<'_> {
    /// Decode into the rotated INT8 grid used by the weight GEMM. This is
    /// `round_ties_even(codebook[code] * relative_scale).clamp(-127,127)`,
    /// not a floating-point codebook-times-scale dequantization.
    pub fn decoded_rotated_i8_reference(&self) -> Result<Vec<i8>> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.rows * self.columns)
            .map_err(invalid)?;
        match self.components {
            ComfyQuantizationComponents::Int8 { .. } => {
                output.extend(self.weight.bytes.iter().map(|value| *value as i8));
            }
            ComfyQuantizationComponents::CodebookW4A8 {
                relative_scale,
                codebook,
                ..
            } => {
                let ComfyQuantizationMetadata::CodebookW4A8 { group_size, .. } = self.metadata
                else {
                    unreachable!()
                };
                let groups = self.columns / group_size;
                for row in 0..self.rows {
                    for column in 0..self.columns {
                        let packed = self.weight.bytes[row * (self.columns / 2) + column / 2];
                        let code = if column % 2 == 0 {
                            packed & 15
                        } else {
                            packed >> 4
                        };
                        let scale =
                            decode_e4m3fn(relative_scale.bytes[row * groups + column / group_size]);
                        let value = (codebook.f32_at(usize::from(code)) * scale)
                            .round_ties_even()
                            .clamp(-127.0, 127.0);
                        if !value.is_finite() {
                            return Err(invalid("decoded W4A8 value is nonfinite"));
                        }
                        output.push(value as i8);
                    }
                }
            }
        }
        Ok(output)
    }

    /// Reconstruct F32 logical OI weights, including inverse ConvRot.
    /// Caller-visible allocation is explicit and fallible. This CPU reference
    /// does not emulate activation INT8 quantization or BF16 output rounding.
    pub fn dequantize_f32_reference(&self) -> Result<Vec<f32>> {
        let decoded = self.decoded_rotated_i8_reference()?;
        let mut output = Vec::new();
        output.try_reserve_exact(decoded.len()).map_err(invalid)?;
        for (index, value) in decoded.into_iter().enumerate() {
            let row = index / self.columns;
            let scale = match self.components {
                ComfyQuantizationComponents::Int8 { scale } => {
                    scale.f32_at(if scale.bytes.len() == 4 { 0 } else { row })
                }
                ComfyQuantizationComponents::CodebookW4A8 { channel_scale, .. } => {
                    channel_scale.f32_at(row)
                }
            };
            let value = f32::from(value) * scale;
            if !value.is_finite() {
                return Err(invalid("scaled reference weight is nonfinite"));
            }
            output.push(value);
        }
        if let Some(group_size) = self.metadata.rotation_group_size() {
            for group in output.chunks_exact_mut(group_size) {
                regular_hadamard(group)?;
            }
        }
        Ok(output)
    }
}

/// H4 is symmetric, regular (all row sums are 2), and self-inverse after
/// normalization. This is its Kronecker power, not the Sylvester transform.
fn regular_hadamard(values: &mut [f32]) -> Result<()> {
    let mut stride = 1;
    while stride < values.len() {
        for start in (0..values.len()).step_by(stride * 4) {
            for offset in 0..stride {
                let a = values[start + offset];
                let b = values[start + offset + stride];
                let c = values[start + offset + 2 * stride];
                let d = values[start + offset + 3 * stride];
                values[start + offset] = a + b + c - d;
                values[start + offset + stride] = a + b - c + d;
                values[start + offset + 2 * stride] = a - b + c + d;
                values[start + offset + 3 * stride] = -a + b + c + d;
            }
        }
        stride *= 4;
    }
    let normalization = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= normalization;
        if !value.is_finite() {
            return Err(invalid("rotated reference weight is nonfinite"));
        }
    }
    Ok(())
}
