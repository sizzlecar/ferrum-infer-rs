//! Selected-row CPU oracle; no full-table decoded allocation is constructed.
use super::*;
use ferrum_runtime::ComfyEmbeddingDequantizedDtype;

impl ComfyQuantizedMatrix<'_> {
    /// Match the independently declared selected-row F32 scale/H256 profile,
    /// including the explicit original compute dtype's final store. This CPU
    /// reference requires finite scale/H256 intermediates and does not certify
    /// a Torch F32 matrix reduction's binary. Invalid U32 IDs return zero rows
    /// without indexing the table, as required by this operation's contract.
    pub fn embedding_f32_reference(
        &self,
        ids: &[u32],
        dtype: ComfyEmbeddingDequantizedDtype,
    ) -> Result<Vec<f32>> {
        let ComfyQuantizationMetadata::Int8Tensorwise {
            convrot: true,
            convrot_groupsize: 256,
        } = self.metadata
        else {
            return Err(invalid(
                "embedding reference requires canonical INT8 ConvRot256 metadata",
            ));
        };
        let ComfyQuantizationComponents::Int8 { scale } = self.components else {
            return Err(invalid("embedding reference requires signed INT8 weights"));
        };
        if scale.shape() != [self.rows as u64, 1] || ids.is_empty() {
            return Err(invalid(
                "embedding reference requires nonempty U32 IDs and exact F32[V,1] scales",
            ));
        }
        let count = ids
            .len()
            .checked_mul(self.columns)
            .ok_or_else(|| invalid("selected embedding size overflows"))?;
        let mut output = Vec::new();
        output.try_reserve_exact(count).map_err(invalid)?;
        for id in ids {
            let start = output.len();
            if *id as usize >= self.rows {
                output.resize(start + self.columns, 0.0);
                continue;
            }
            let row = *id as usize;
            let row_scale = scale.f32_at(row);
            for byte in &self.weight.bytes[row * self.columns..(row + 1) * self.columns] {
                let value = f32::from(*byte as i8) * row_scale;
                if !value.is_finite() {
                    return Err(invalid("embedding F32 scaled intermediate is nonfinite"));
                }
                output.push(value);
            }
            for group in output[start..].chunks_exact_mut(256) {
                super::a8_reference::rotate_staged(group)?;
            }
            if dtype == ComfyEmbeddingDequantizedDtype::Bf16ThenF32 {
                for value in &mut output[start..] {
                    *value = half::bf16::from_f32(*value).to_f32();
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t<'a>(shape: &'a [u64], dtype: Dtype, bytes: &'a [u8]) -> ComfyQuantTensor<'a> {
        ComfyQuantTensor::new(dtype, shape, bytes).unwrap()
    }
    fn matrix<'a>(weight: &'a [u8], scale: &'a [u8]) -> ComfyQuantizedMatrix<'a> {
        ComfyQuantizedMatrix::new(
            ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: 256,
            },
            t(&[3, 256], Dtype::I8, weight),
            ComfyQuantizationComponents::Int8 {
                scale: t(&[3, 1], Dtype::F32, scale),
            },
        )
        .unwrap()
    }
    #[test]
    fn selected_rows_preserve_inverse_regular_hadamard_source_dtype_and_invalid_id_rule() {
        let mut weight = vec![0_u8; 3 * 256];
        weight[..256].fill(1);
        weight[256] = (-128_i8) as u8;
        weight[512..516].copy_from_slice(&[3, 1, 1, 1]);
        let scales: Vec<u8> = [1.000244140625_f32, 0.03125, 0.1]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let m = matrix(&weight, &scales);
        let ids = [1, 0, 1, 3, u32::MAX, 2];
        let full = m
            .embedding_f32_reference(&ids, ComfyEmbeddingDequantizedDtype::F32)
            .unwrap();
        let rounded = m
            .embedding_f32_reference(&ids, ComfyEmbeddingDequantizedDtype::Bf16ThenF32)
            .unwrap();
        // Independent dense H256 sign oracle for the dyadic impulse. Every
        // higher base-4 digit uses column zero of H4; normalization is 1/16.
        let h4 = [
            [1_i32, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ];
        for column in 0..256 {
            let mut sign = 1_i32;
            let mut digit = column;
            for _ in 0..4 {
                sign *= h4[digit % 4][0];
                digit /= 4;
            }
            let expected = -4.0_f64 * f64::from(sign) / 16.0;
            assert_eq!(full[column], expected as f32);
            assert_eq!(full[512 + column], expected as f32);
        }
        assert!(full[256..512].iter().all(|v| *v == 1.000244140625));
        assert!(rounded[256..512].iter().all(|v| *v == 1.0));
        assert!(full[3 * 256..5 * 256].iter().all(|v| *v == 0.0));
        // q=[3,1,1,1,0...] with scale .1: staged F32 cancellation retains
        // -2^-30 at column 3. Rotating integers first then scaling gives zero.
        assert_eq!(full[5 * 256 + 3], -2.0_f32.powi(-30));
        assert!(m
            .embedding_f32_reference(&[], ComfyEmbeddingDequantizedDtype::F32)
            .is_err());
        let scalar = 0.1_f32.to_le_bytes();
        let incompatible = ComfyQuantizedMatrix::new(
            ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: 256,
            },
            t(&[3, 256], Dtype::I8, &weight),
            ComfyQuantizationComponents::Int8 {
                scale: t(&[], Dtype::F32, &scalar),
            },
        )
        .unwrap();
        assert!(incompatible
            .embedding_f32_reference(&[0], ComfyEmbeddingDequantizedDtype::F32)
            .is_err());
    }
}
