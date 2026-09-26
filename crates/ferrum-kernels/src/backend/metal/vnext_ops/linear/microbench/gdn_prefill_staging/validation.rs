//! Independent dense oracle and packed-buffer invariants, outside timed loops.
use super::*;

fn halves(buffer: &Buffer) -> &[f16] {
    // SAFETY: callers exclusively own these shared buffers; all GPU commands
    // were waited before validation, and every allocation contains whole halfs.
    unsafe {
        std::slice::from_raw_parts(
            buffer.contents().cast::<f16>(),
            buffer.length() as usize / 2,
        )
    }
}

impl Case {
    pub(super) fn validate(&self, full_cpu_oracle: bool) -> serde_json::Value {
        let g = self.geometry;
        let payload = g.rows as usize * g.packed_width() as usize;
        let mut compared = 0_usize;
        for (set, pair) in self.outputs.iter().enumerate() {
            let outputs = pair.each_ref().map(halves);
            for output in &outputs {
                assert!(output[..PREFIX]
                    .iter()
                    .all(|v| v.to_bits() == GUARD.to_bits()));
                assert!(output[PREFIX + payload..]
                    .iter()
                    .all(|v| v.to_bits() == GUARD.to_bits()));
                assert!(output[PREFIX..PREFIX + payload]
                    .iter()
                    .all(|v| v.is_finite()));
            }
            for (index, (a, b)) in outputs[0][PREFIX..PREFIX + payload]
                .iter()
                .zip(&outputs[1][PREFIX..PREFIX + payload])
                .enumerate()
            {
                assert_eq!(a.to_bits(), b.to_bits(), "QKVZBA set={set} element={index}");
                compared += 1;
            }
            for (i, weights) in self.weights[set].buffers.iter().enumerate() {
                // SAFETY: byte buffers owned here, no live command; length was
                // allocated from this exact prefix + payload + suffix.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        weights.contents().cast::<u8>(),
                        weights.length() as usize,
                    )
                };
                assert!(bytes[..WEIGHT_PREFIX].iter().all(|b| *b == BYTE_GUARD));
                assert_eq!(
                    &bytes[WEIGHT_PREFIX..WEIGHT_PREFIX + self.encoded[i].len()],
                    self.encoded[i]
                );
                assert!(bytes[WEIGHT_PREFIX + self.encoded[i].len()..]
                    .iter()
                    .all(|b| *b == BYTE_GUARD));
            }
        }
        assert_eq!(halves(&self.input), self.input_values);
        let stage = halves(&self.staged);
        assert!(stage[..STAGED_PREFIX]
            .iter()
            .all(|v| v.to_bits() == GUARD.to_bits()));
        assert!(stage[STAGED_PREFIX + g.staging_elements()..]
            .iter()
            .all(|v| v.to_bits() == GUARD.to_bits()));

        // Staging scratch was reused by the Q4 gate last. Check its complete
        // payload, independently decoding the GGUF bytes and F16 rounding.
        let gate = g.shapes()[1];
        let block_len = gate.format.block_values();
        let block_bytes = gate.format.block_bytes();
        for (i, actual) in stage
            [STAGED_PREFIX..STAGED_PREFIX + g.input as usize * g.widths[1] as usize]
            .iter()
            .enumerate()
        {
            let first = i / block_len * block_bytes;
            let expected = f16::from_f32(
                gate.format
                    .decode_value(&self.encoded[1][first..first + block_bytes], i % block_len),
            );
            assert_eq!(actual.to_f32(), expected.to_f32(), "stage coefficient {i}");
        }

        let rows: Vec<usize> = if full_cpu_oracle {
            (0..g.rows as usize).collect()
        } else {
            vec![0, g.rows as usize / 2, g.rows as usize - 1]
        };
        let mut oracle_elements = 0;
        let mut column_start = 0_usize;
        for (i, shape) in g.shapes().into_iter().enumerate() {
            let columns: Vec<usize> = if full_cpu_oracle {
                (0..shape.output as usize).collect()
            } else {
                vec![0, shape.output as usize / 2, shape.output as usize - 1]
            };
            let block_len = shape.format.block_values();
            let block_bytes = shape.format.block_bytes();
            let row_bytes = shape.input as usize / block_len * block_bytes;
            for &row in &rows {
                for &column in &columns {
                    let mut reference = 0.0_f32;
                    for k in 0..shape.input as usize {
                        let offset = column * row_bytes + k / block_len * block_bytes;
                        reference += self.input_values[PREFIX + row * shape.input as usize + k]
                            .to_f32()
                            * shape.format.decode_value(
                                &self.encoded[i][offset..offset + block_bytes],
                                k % block_len,
                            );
                    }
                    let tolerance = linear_tolerance(ElementType::F16, reference);
                    for pair in &self.outputs {
                        for output in pair {
                            let index =
                                PREFIX + row * g.packed_width() as usize + column_start + column;
                            let actual = halves(output)[index].to_f32();
                            assert!(
                                (actual - reference).abs() <= tolerance,
                                "CPU oracle leaf={i} row={row} column={column} actual={actual} expected={reference} tolerance={tolerance}"
                            );
                        }
                    }
                    oracle_elements += 1;
                }
            }
            column_start += shape.output as usize;
        }
        serde_json::json!({
            "bitwise_compared_elements":compared,"bitwise_differences":0,
            "all_outputs_finite":true,"guards_and_immutable_inputs":true,
            "dense_cpu_oracle_full_output":full_cpu_oracle,
            "dense_cpu_oracle_distinct_positions":oracle_elements,
            "final_stage_coefficients_checked":g.input as usize * g.widths[1] as usize,
            "tolerance":"existing_F16_linear_0.002_plus_0.003_abs_reference",
        })
    }
}
