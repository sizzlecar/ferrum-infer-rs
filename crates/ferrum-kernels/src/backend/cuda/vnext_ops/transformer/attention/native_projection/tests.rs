use super::super::super::test_support::Guarded;
use super::*;
use crate::gguf_blocks::{fixtures::oracle_blocks, GgufBlockFormat};
use cudarc::driver::CudaContext;
use ferrum_interfaces::vnext::WeightId;
use half::f16;
use weights::{MatrixFormat, MatrixPart};

#[test]
fn q8_attention_workspace_and_dispatch_account_for_mixed_parts_and_row_chunks() {
    let part = |format, columns, output_offset| MatrixPart {
        component_id: WeightId::new(format!("weight.q8-attention.{output_offset}")).unwrap(),
        format,
        columns,
        rows: 3,
        output_offset,
        transform: None,
        signs_region: None,
    };
    let parts = [
        part(MatrixFormat::Block(GgufBlockFormat::Q5K), 256, 0),
        part(MatrixFormat::Block(GgufBlockFormat::Q4K), 256, 3),
        part(MatrixFormat::Block(GgufBlockFormat::Q8_0), 256, 6),
    ];
    let packed = q8_part_workspace_per_token(&parts).unwrap();
    assert_eq!(packed, 256 + 8 * 4);
    assert_eq!(q8_dispatch_count(&parts, 7).unwrap(), 4);
    assert_eq!(q8_dispatch_count(&parts, MAX_ROWS + 1).unwrap(), 8);
    assert_eq!(q8_part_workspace_per_token(&parts[2..]).unwrap(), 0);
    assert_eq!(q8_dispatch_count(&parts[2..], MAX_ROWS + 1).unwrap(), 2);
    assert!(q8_part_workspace_per_token(&[]).is_err());
    assert!(q8_dispatch_count(&parts, 0).is_err());
    let mut rotated = parts[0].clone();
    rotated.signs_region = Some(1);
    assert!(q8_part_workspace_per_token(&[rotated]).is_err());
    assert!(q8_part_workspace_per_token(&[part(
        MatrixFormat::Block(GgufBlockFormat::Q5K),
        255,
        0
    )])
    .is_err());
    let shape = super::super::tests::test_shape();
    for tokens in [1, 7, 8, 33, MAX_ROWS + 1] {
        let strict = ScratchLayout::new(
            shape,
            tokens,
            3,
            AttentionProjection::Native {
                transform_bytes_per_token: 0,
            },
        )
        .unwrap();
        let q8 = ScratchLayout::new(
            shape,
            tokens,
            3,
            AttentionProjection::NativeQ8 {
                pack_bytes_per_token: packed,
            },
        )
        .unwrap();
        assert_eq!(q8.required_bytes - strict.required_bytes, tokens * packed);
        assert_eq!(
            q8.z_or_activation - q8.projection_staging.unwrap(),
            tokens * packed
        );
        assert_eq!(q8.projection_staging.unwrap() % SCRATCH_ALIGNMENT, 0);
    }
}

#[test]
#[ignore = "requires an actual SM80+ CUDA device"]
fn q8_attention_projection_preserves_pack_and_output_across_row_chunk_on_cuda() {
    use crate::gguf_blocks::q4k_q8_reference::{dot_reference, fixture_block};
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let native = CudaNativeBlockKernels::load(&context).unwrap();
    let q8 = Q8F32ScaleKernels::load(&context).unwrap();
    let tokens = MAX_ROWS as usize + 1;
    let columns = 256;
    let block = fixture_block(1, 0);
    let matrix = Guarded::new(&stream, &block.encode(), 0xab);
    let part = MatrixPart {
        component_id: WeightId::new("weight.q8-chunk").unwrap(),
        format: MatrixFormat::Block(GgufBlockFormat::Q4K),
        rows: 1,
        columns,
        output_offset: 1,
        transform: None,
        signs_region: None,
    };
    let input = (0..tokens)
        .flat_map(|row| vec![f16::from_f32((row as i32 % 7 - 3) as f32 / 128.0); columns as usize])
        .collect::<Vec<_>>();
    let input_gpu = Guarded::new(&stream, &input, f16::from_f32(71.0));
    let sentinel = f16::from_f32(79.0);
    let output = Guarded::new(&stream, &vec![sentinel; tokens * 3], sentinel);
    let bytes = PackLayout::new(MAX_ROWS, u64::from(columns))
        .unwrap()
        .total_bytes;
    let scratch = Guarded::new(&stream, &vec![0xa5_u8; bytes as usize], 0xcd);
    let run = |size| {
        launch_q8_parts(
            &stream,
            &native,
            &q8,
            std::slice::from_ref(&part),
            &[matrix.pointer(&stream)],
            input_gpu.pointer(&stream),
            output.pointer(&stream),
            tokens as i32,
            3,
            columns as i32,
            scratch.pointer(&stream),
            size,
        )
    };
    assert!(run(bytes - 1).is_err());
    run(bytes).unwrap();
    stream.synchronize().unwrap();
    let actual = output.read(&stream);
    let expected = (0..7)
        .map(|row| {
            let dot = dot_reference(
                &input[row * columns as usize..(row + 1) * columns as usize],
                std::slice::from_ref(&block),
            );
            let nu = 11.0 * f64::from(f32::EPSILON);
            let accumulation = nu / (1.0 - nu) * dot.expanded_abs_terms;
            (
                f16::from_f64(dot.policy),
                accumulation
                    + 0.0009765625 * (dot.policy.abs() + accumulation)
                    + f16::from_bits(1).to_f64(),
            )
        })
        .collect::<Vec<_>>();
    for (row, values) in actual.chunks_exact(3).enumerate() {
        assert_eq!(values[0], sentinel);
        assert_eq!(values[2], sentinel);
        let (reference, bound) = expected[row % 7];
        assert!(
            values[1].is_finite() && (values[1].to_f64() - reference.to_f64()).abs() <= bound,
            "row {row}: {} vs {reference}",
            values[1]
        );
    }
    input_gpu.assert_unchanged(&stream);
    matrix.assert_unchanged(&stream);
    scratch.read(&stream);
}

#[test]
fn native_attention_projection_accounts_for_partitions_and_cuda_row_capacity() {
    assert_eq!(dispatch_count(7, 3).unwrap(), 7);
    assert_eq!(dispatch_count(7, MAX_ROWS + 1).unwrap(), 14);
    assert!(dispatch_count(0, 1).is_err());
    assert!(dispatch_count(1, 0).is_err());
    assert!(dispatch_count(usize::MAX, u64::MAX).is_err());
    let mut shape = super::super::tests::test_shape();
    shape.hidden_size = 512;
    let scratch = ScratchLayout::new(
        shape,
        3,
        2,
        AttentionProjection::Native {
            transform_bytes_per_token: 0,
        },
    )
    .unwrap();
    assert!(scratch.projection_workspace.is_none());
    assert!(scratch.projection_staging.is_none());
    assert_eq!(
        scratch.required_bytes,
        shape.fixed_scratch_bytes().unwrap()
            + 2 * shape.scratch_bytes_per_sequence().unwrap()
            + 3 * shape.scratch_bytes_per_token().unwrap()
    );
    let transformed = ScratchLayout::new(
        shape,
        3,
        2,
        AttentionProjection::Native {
            transform_bytes_per_token: 512 * 4,
        },
    )
    .unwrap();
    assert_eq!(
        transformed.required_bytes,
        scratch.required_bytes + 3 * 512 * 4
    );
    assert!(transformed.projection_staging.is_some());
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
        MatrixFormat::Block(GgufBlockFormat::Pq2_0),
    ];
    for (columns, tokens, formats) in [
        (256, 3, mixed),
        (384, 3, vec![MatrixFormat::Block(GgufBlockFormat::Pq2_0)]),
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
                transform: None,
                signs_region: None,
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
                .map(|(part, matrix)| (part, matrix.pointer(&stream), 0)),
            input_gpu.pointer(&stream),
            output_gpu.pointer(&stream),
            tokens as i32,
            output_features as i32,
            columns as i32,
            0,
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
