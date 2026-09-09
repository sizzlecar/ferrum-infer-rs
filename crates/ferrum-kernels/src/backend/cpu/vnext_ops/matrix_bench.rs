//! Manual CPU microbenchmark. Run with an optimized Cargo profile and
//! `--ignored --nocapture`; durations are observations, never test thresholds.

use super::matrix::{CpuMatrix, CpuMatrixFormat};
use super::scalar::CpuFloat;
use super::tests::encoded;
use crate::gguf_blocks::{fixtures::oracle_blocks, GgufBlockFormat};
use sha2::{Digest, Sha256};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "manual optimized CPU matrix timings; not a performance gate"]
fn native_matrix_timings_with_dense_output_validation() {
    let columns = 2048;
    let outputs = 512;
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let blocks = oracle_blocks(format);
        let bytes_per_row = columns / format.block_values() * format.block_bytes();
        let mut source = Vec::with_capacity(outputs * bytes_per_row);
        for row in 0..outputs {
            for column_block in 0..columns / format.block_values() {
                let block = (row + column_block) % (blocks.len() / format.block_bytes());
                source.extend_from_slice(
                    &blocks[block * format.block_bytes()..(block + 1) * format.block_bytes()],
                );
            }
        }
        let matrix =
            CpuMatrix::new(&source, CpuMatrixFormat::Native(format), outputs, columns).unwrap();
        // This expanded tensor is only an independent dense-dot reference for
        // the benchmark; production retains the native source above.
        let mut decoded = vec![0.0; outputs * columns];
        format.decode(&source, &mut decoded).unwrap();
        let dense_bytes = encoded(&decoded, CpuFloat::F32);
        let reference = CpuMatrix::new(
            &dense_bytes,
            CpuMatrixFormat::Dense(CpuFloat::F32),
            outputs,
            columns,
        )
        .unwrap();
        for tokens in [1, 32] {
            let input = encoded(
                &(0..tokens * columns)
                    .map(|i| (i as f32 * 0.17).sin() * 0.01)
                    .collect::<Vec<_>>(),
                CpuFloat::F16,
            );
            let mut expected = vec![0; tokens * outputs * 4];
            reference
                .linear(
                    &input,
                    CpuFloat::F16,
                    &mut expected,
                    CpuFloat::F32,
                    tokens,
                    outputs,
                    0,
                )
                .unwrap();
            let mut actual = vec![0; expected.len()];
            matrix
                .linear(
                    &input,
                    CpuFloat::F16,
                    &mut actual,
                    CpuFloat::F32,
                    tokens,
                    outputs,
                    0,
                )
                .unwrap();
            let mut milliseconds = Vec::new();
            for _ in 0..3 {
                let start = Instant::now();
                matrix
                    .linear(
                        black_box(&input),
                        CpuFloat::F16,
                        black_box(&mut actual),
                        CpuFloat::F32,
                        tokens,
                        outputs,
                        0,
                    )
                    .unwrap();
                milliseconds.push(start.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(
                    actual, expected,
                    "native {format:?} differs from dense F32 dot"
                );
            }
            println!(
                "{}",
                serde_json::json!({
                    "benchmark": "cpu_native_matrix", "weight_format": format.format_id(),
                    "input_type": "f16", "output_type": "f32", "accumulation": "f32",
                    "tokens": tokens, "outputs": outputs, "columns": columns,
                    "threads": rayon::current_num_threads(), "os": std::env::consts::OS,
                    "architecture": std::env::consts::ARCH, "milliseconds": milliseconds,
                    "source_sha256": format!("{:x}", Sha256::digest(&source)),
                    "output_sha256": format!("{:x}", Sha256::digest(&actual)),
                    "validation": "all output bytes equal decoded dense F32 dot"
                })
            );
        }
    }
}
