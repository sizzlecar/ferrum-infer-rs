use super::super::super::test_support::Guarded;
use super::*;
use crate::gguf_blocks::{fixtures::oracle_blocks, GgufBlockFormat};
use cudarc::driver::CudaContext;
use ferrum_interfaces::vnext::WeightId;
use half::f16;
use weights::{MatrixFormat, MatrixPart};

#[test]
fn native_attention_projection_accounts_for_partitions_and_cuda_row_capacity() {
    assert_eq!(dispatch_count(7, 3).unwrap(), 7);
    assert_eq!(dispatch_count(7, MAX_ROWS + 1).unwrap(), 14);
    assert!(dispatch_count(0, 1).is_err());
    assert!(dispatch_count(1, 0).is_err());
    assert!(dispatch_count(usize::MAX, u64::MAX).is_err());
    let mut shape = super::super::tests::test_shape();
    shape.hidden_size = 512;
    let scratch = ScratchLayout::new(shape, 3, 2, AttentionProjection::Native).unwrap();
    assert!(scratch.projection_workspace.is_none());
    assert!(scratch.projection_staging.is_none());
    assert_eq!(
        scratch.required_bytes,
        shape.fixed_scratch_bytes().unwrap()
            + 2 * shape.scratch_bytes_per_sequence().unwrap()
            + 3 * shape.scratch_bytes_per_token().unwrap()
    );
}

fn fixture(format: MatrixFormat, rows: usize, columns: usize) -> (Vec<u8>, Vec<f32>) {
    match format {
        MatrixFormat::DenseF16 => {
            let values = (0..rows * columns)
                .map(|i| f16::from_f32((i % 17) as f32 / 16.0 - 0.5))
                .collect::<Vec<_>>();
            (
                values
                    .iter()
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect(),
                values.iter().map(|x| x.to_f32()).collect(),
            )
        }
        MatrixFormat::Block(format) => {
            let fixture = oracle_blocks(format);
            let bytes = fixture
                .chunks_exact(format.block_bytes())
                .cycle()
                .take(rows * columns / format.block_values())
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let mut decoded = vec![0.0; rows * columns];
            format.decode(&bytes, &mut decoded).unwrap();
            (bytes, decoded)
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn native_attention_projection_matches_mixed_matrix_oracle_and_chunk_boundary_on_cuda() {
    let context = CudaContext::new(0).expect("native recurrent projection requires CUDA");
    let stream = context.default_stream();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let mixed = vec![
        MatrixFormat::DenseF16,
        MatrixFormat::Block(GgufBlockFormat::Q3K),
        MatrixFormat::Block(GgufBlockFormat::Q4K),
        MatrixFormat::Block(GgufBlockFormat::Q6K),
        MatrixFormat::Block(GgufBlockFormat::Q8_0),
        MatrixFormat::Block(GgufBlockFormat::Iq3S),
        MatrixFormat::Block(GgufBlockFormat::Iq4Xs),
    ];
    for (columns, tokens, formats) in [
        (256, 3, mixed),
        (3, MAX_ROWS as usize + 1, vec![MatrixFormat::DenseF16]),
    ] {
        let mut matrices = Vec::new();
        let mut parts = Vec::new();
        let mut reference = Vec::new();
        let mut output_features = 0;
        for (index, format) in formats.into_iter().enumerate() {
            let rows = index % 3 + 1;
            let (bytes, decoded) = fixture(format, rows, columns);
            matrices.push(Guarded::new(&stream, &bytes, 0xAB_u8));
            parts.push(MatrixPart {
                component_id: WeightId::new(format!("weight.test.part-{index}")).unwrap(),
                format,
                rows: rows as u32,
                columns: columns as u32,
                output_offset: output_features as u32,
            });
            reference.extend(decoded);
            output_features += rows;
        }
        let input = (0..tokens * columns)
            .map(|i| f16::from_f32((i % 13) as f32 / 1024.0 - 0.005))
            .collect::<Vec<_>>();
        let input_gpu = Guarded::new(&stream, &input, f16::from_f32(73.0));
        let output_gpu = Guarded::new(
            &stream,
            &vec![f16::ZERO; tokens * output_features],
            f16::from_f32(79.0),
        );
        launch_parts(
            &stream,
            &kernels,
            parts
                .iter()
                .zip(&matrices)
                .map(|(part, matrix)| (part, matrix.pointer(&stream))),
            input_gpu.pointer(&stream),
            output_gpu.pointer(&stream),
            tokens as i32,
            output_features as i32,
            columns as i32,
        )
        .unwrap();
        let actual = output_gpu.read(&stream);
        for (token, row) in input.chunks_exact(columns).enumerate() {
            for output in 0..output_features {
                let products = row
                    .iter()
                    .zip(&reference[output * columns..][..columns])
                    .map(|(x, w)| x.to_f64() * f64::from(*w));
                let expected = products.clone().sum::<f64>();
                let accumulation_bound = 2.0
                    * columns as f64
                    * f64::from(f32::EPSILON)
                    * products.map(f64::abs).sum::<f64>();
                let rounding_bound = expected.abs() * 0.0005 + 3.0e-8;
                let actual = actual[token * output_features + output].to_f64();
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() <= accumulation_bound + rounding_bound,
                    "projection[{token}, {output}] actual {actual}, reference {expected}"
                );
            }
        }
        input_gpu.assert_unchanged(&stream);
        for matrix in matrices {
            matrix.assert_unchanged(&stream);
        }
    }
}
