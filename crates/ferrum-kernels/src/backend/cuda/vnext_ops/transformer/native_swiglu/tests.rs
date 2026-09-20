use super::*;
use crate::gguf_blocks::{fixtures::oracle_blocks, GgufBlockFormat};
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::WeightId;
use half::f16;
use weights::{MatrixFormat, MatrixPart};

#[test]
fn native_swiglu_scratch_accounts_only_for_bounded_activations() {
    let layout = ScratchLayout::new(3, 257).unwrap();
    assert_eq!(layout.activation_elements, 771);
    assert_eq!(layout.gate_up_bytes, 3084);
    assert_eq!(layout.total_bytes, 4626);
    for (tokens, intermediate) in [(0, 1), (1, 0), (u64::MAX, 2), (u64::MAX / 6 + 1, 1)] {
        assert!(ScratchLayout::new(tokens, intermediate).is_err());
    }
}

#[test]
fn native_swiglu_packed_rows_preserve_larger_participant_batches() {
    let max_rows = u64::from(u16::MAX);
    assert_eq!(
        ScratchLayout::new(max_rows, 1)
            .unwrap()
            .packed_rows(max_rows),
        Some(u16::MAX.into())
    );
    assert_eq!(
        ScratchLayout::new(max_rows + 1, 1)
            .unwrap()
            .packed_rows(max_rows + 1),
        None
    );
    // The aggregate cannot use one launch, but its two old participant launches
    // remain within both the grid and the signed gate/up indexing capacity.
    for rows in [max_rows / 2, max_rows / 2 + 1] {
        assert_eq!(
            ScratchLayout::new(rows, 16_385).unwrap().packed_rows(rows),
            Some(rows as u32)
        );
    }
    assert_eq!(
        ScratchLayout::new(max_rows, 16_384)
            .unwrap()
            .packed_rows(max_rows),
        Some(u16::MAX.into())
    );
    assert_eq!(
        ScratchLayout::new(max_rows, 16_385)
            .unwrap()
            .packed_rows(max_rows),
        None
    );
    assert!(native_matrix::single_launch_rows(0).is_none());
    assert!(native_matrix::single_launch_rows(u64::MAX).is_none());
}

fn matrix(format: MatrixFormat, rows: usize, columns: usize, salt: usize) -> (Vec<u8>, Vec<f32>) {
    match format {
        MatrixFormat::DenseF16 => {
            let values = (0..rows * columns)
                .map(|i| f16::from_f32(((i * 7 + salt) % 31) as f32 / 32.0 - 0.5))
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
            assert_eq!(columns % format.block_values(), 0);
            let source = oracle_blocks(format);
            let bytes = source
                .chunks_exact(format.block_bytes())
                .cycle()
                .skip(salt)
                .take(rows * columns / format.block_values())
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let mut values = vec![0.0; rows * columns];
            format.decode(&bytes, &mut values).unwrap();
            (bytes, values)
        }
    }
}

fn assert_projection(
    input: &[f16],
    weight: &[f32],
    actual: &[f16],
    columns: usize,
    outputs: usize,
) {
    for (row, x) in input.chunks_exact(columns).enumerate() {
        for column in 0..outputs {
            let w = &weight[column * columns..][..columns];
            let products = x.iter().zip(w).map(|(x, w)| x.to_f64() * f64::from(*w));
            let reference = products.clone().sum::<f64>();
            let bound = (columns as f64 * f32::EPSILON as f64 + 0.0009765625)
                * products.map(f64::abs).sum::<f64>()
                + f16::from_bits(1).to_f64();
            let actual = actual[row * outputs + column].to_f64();
            assert!(
                actual.is_finite() && (actual - reference).abs() <= bound,
                "row {row} column {column}: actual {actual}, F64 {reference}, bound {bound}"
            );
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn native_swiglu_mixed_matrices_match_stage_oracles_on_cuda() {
    use GgufBlockFormat::*;
    use MatrixFormat::{Block as Q, DenseF16 as D};
    let context = CudaContext::new(0).expect("native SwiGLU conformance requires CUDA");
    let stream = context.default_stream();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    let cases = [
        (Q(Q4K), Q(Q6K), Q(Q8_0), 256, 256),
        (Q(Iq3S), Q(Iq4Xs), D, 256, 256),
        (D, Q(Q3K), Q(Q5K), 256, 256),
        (Q(Iq4Nl), D, Q(Q4K), 256, 256),
        (Q(Pq2_0), D, Q(Pq2_0), 128, 384),
        (Q(Pq2_0), Q(Pq2_0), Q(Pq2_0), 384, 128),
        (D, D, D, 33, 17),
    ];
    for (gate_format, up_format, down_format, hidden, intermediate) in cases {
        let (gate_bytes, gate_values) = matrix(gate_format, intermediate, hidden, 0);
        let (up_bytes, up_values) = matrix(up_format, intermediate, hidden, 1);
        let (down_bytes, down_values) = matrix(down_format, hidden, intermediate, 2);
        let mut weight_gpu = Vec::new();
        for bytes in [&gate_bytes, &up_bytes, &down_bytes] {
            let mut padded = vec![0xAB_u8; 16];
            padded.extend_from_slice(bytes);
            weight_gpu.push(stream.clone_htod(&padded).unwrap());
        }
        let part = |format, rows, columns, offset| MatrixPart {
            transform: None,
            signs_region: None,
            component_id: WeightId::new("component.swiglu").unwrap(),
            format,
            rows: rows as u32,
            columns: columns as u32,
            output_offset: offset as u32,
        };
        let gate_up = [
            part(gate_format, intermediate, hidden, 0),
            part(up_format, intermediate, hidden, intermediate),
        ];
        let down = [part(down_format, hidden, intermediate, 0)];
        for tokens in [1, 3] {
            let input = (0..tokens * hidden)
                .map(|i| f16::from_f32(((i * 13 % 7) as f32 - 3.0) / 16384.0))
                .collect::<Vec<_>>();
            let x = stream.clone_htod(&input).unwrap();
            let canary = f16::from_f32(-12345.0);
            let mut gate_out = stream
                .clone_htod(&vec![canary; 16 + tokens * 2 * intermediate])
                .unwrap();
            let mut activation = stream
                .clone_htod(&vec![canary; 16 + tokens * intermediate])
                .unwrap();
            let mut output = stream
                .clone_htod(&vec![canary; 16 + tokens * hidden])
                .unwrap();
            let (xp, _x_guard) = x.device_ptr(&stream);
            let ptrs = weight_gpu
                .iter()
                .map(|buffer| buffer.device_ptr(&stream))
                .collect::<Vec<_>>();
            let pointers = ptrs.iter().map(|(p, _)| p + 16).collect::<Vec<_>>();
            let (gp, gg) = gate_out.device_ptr_mut(&stream);
            let (ap, ag) = activation.device_ptr_mut(&stream);
            let (yp, yg) = output.device_ptr_mut(&stream);
            launch(
                &kernels,
                &silu,
                &stream,
                &gate_up,
                &down,
                &pointers,
                xp,
                yp + 16,
                gp + 16,
                ap + 16,
                tokens as u32,
                hidden as u32,
                intermediate as u32,
                0,
            )
            .unwrap();
            drop((gg, ag, yg));
            let gate_actual = stream.clone_dtoh(&gate_out).unwrap();
            let act_actual = stream.clone_dtoh(&activation).unwrap();
            let out_actual = stream.clone_dtoh(&output).unwrap();
            for buffer in [&gate_actual, &act_actual, &out_actual] {
                assert!(buffer[..8]
                    .iter()
                    .chain(&buffer[buffer.len() - 8..])
                    .all(|x| *x == canary));
            }
            let gates = &gate_actual[8..gate_actual.len() - 8];
            let activations = &act_actual[8..act_actual.len() - 8];
            let values = gate_values
                .iter()
                .chain(&up_values)
                .copied()
                .collect::<Vec<_>>();
            assert_projection(&input, &values, gates, hidden, 2 * intermediate);
            // Each stage uses an independent F64 scalar oracle over its actual
            // admitted F16 input; all compressed weights decode on the CPU.
            for row in 0..tokens {
                for column in 0..intermediate {
                    let g = gates[row * 2 * intermediate + column].to_f64();
                    let u = gates[row * 2 * intermediate + intermediate + column].to_f64();
                    let reference = g / (1.0 + (-g).exp()) * u;
                    let actual = activations[row * intermediate + column].to_f64();
                    let bound = 0.001 * reference.abs() + f16::from_bits(1).to_f64();
                    assert!(
                        actual.is_finite() && (actual - reference).abs() <= bound,
                        "SiLU row {row} col {column}: {actual} vs {reference}, bound {bound}"
                    );
                }
            }
            assert_projection(
                activations,
                &down_values,
                &out_actual[8..out_actual.len() - 8],
                intermediate,
                hidden,
            );
            assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
            for (gpu, bytes) in weight_gpu.iter().zip([&gate_bytes, &up_bytes, &down_bytes]) {
                let actual = stream.clone_dtoh(gpu).unwrap();
                assert!(actual[..16].iter().all(|&byte| byte == 0xAB));
                assert_eq!(&actual[16..], bytes);
            }
        }
    }
}

/// Exercises the native launchers, not BatchedOperationInvocation admission.
/// The packed rows must preserve the old independent matrix computations even
/// when a row tile crosses a request boundary or a source starts after row zero.
#[test]
#[ignore = "requires an actual CUDA device"]
fn native_swiglu_packed_rows_match_independent_source_slices_on_cuda() {
    use super::super::test_support::Guarded;

    let context = CudaContext::new(0).expect("packed native SwiGLU requires CUDA");
    let stream = context.default_stream();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    let hidden = 256;
    let intermediate = 256;
    let formats = [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q6K,
        GgufBlockFormat::Q4K,
    ];
    let matrices = formats
        .into_iter()
        .enumerate()
        .map(|(index, format)| {
            let (bytes, _) = matrix(MatrixFormat::Block(format), 256, 256, index);
            Guarded::new(&stream, &bytes, 0xAB_u8)
        })
        .collect::<Vec<_>>();
    let pointers = matrices
        .iter()
        .map(|matrix| matrix.pointer(&stream))
        .collect::<Vec<_>>();
    let part = |index: usize, offset| MatrixPart {
        transform: None,
        signs_region: None,
        component_id: WeightId::new(format!("component.packed.{index}")).unwrap(),
        format: MatrixFormat::Block(formats[index]),
        rows: 256,
        columns: 256,
        output_offset: offset,
    };
    let gate_up = [part(0, 0), part(1, intermediate as u32)];
    let down = [part(2, 0)];
    let run = |input, count: usize| {
        let guard = f16::from_f32(-12345.0);
        let gate = Guarded::new(&stream, &vec![f16::NAN; count * intermediate * 2], guard);
        let activation = Guarded::new(&stream, &vec![f16::NAN; count * intermediate], guard);
        let output = Guarded::new(&stream, &vec![f16::NAN; count * hidden], guard);
        launch(
            &kernels,
            &silu,
            &stream,
            &gate_up,
            &down,
            &pointers,
            input,
            output.pointer(&stream),
            gate.pointer(&stream),
            activation.pointer(&stream),
            count as u32,
            hidden as u32,
            intermediate as u32,
            0,
        )
        .unwrap();
        [
            gate.read(&stream),
            activation.read(&stream),
            output.read(&stream),
        ]
    };
    for counts in [
        vec![1, 1],
        vec![7, 1],
        vec![3, 11, 1],
        vec![64, 1],
        vec![1, 64],
    ] {
        let mut packed = Vec::new();
        let mut separate = [Vec::new(), Vec::new(), Vec::new()];
        for (participant, &count) in counts.iter().enumerate() {
            let source_start = 3 + participant;
            let source = (0..(source_start + count + 2) * hidden)
                .map(|index| {
                    f16::from_f32(((index * 13 + participant * 3) % 17) as f32 / 16384.0 - 0.0005)
                })
                .collect::<Vec<_>>();
            // Separate physical input allocation and a nonzero source offset.
            let source_gpu = Guarded::new(&stream, &source, f16::from_f32(73.0));
            let stages = run(
                source_gpu.pointer(&stream) + (source_start * hidden * 2) as u64,
                count,
            );
            for (actual, expected) in separate.iter_mut().zip(stages) {
                actual.extend(expected);
            }
            packed
                .extend_from_slice(&source[source_start * hidden..(source_start + count) * hidden]);
            source_gpu.assert_unchanged(&stream);
        }
        let packed_gpu = Guarded::new(&stream, &packed, f16::from_f32(79.0));
        let combined = run(packed_gpu.pointer(&stream), counts.iter().sum());
        for (stage, (combined, separate)) in combined.into_iter().zip(separate).enumerate() {
            assert!(combined.iter().all(|value| value.is_finite()));
            assert_eq!(
                combined, separate,
                "native packed stage {stage}, participant rows {counts:?}"
            );
        }
        packed_gpu.assert_unchanged(&stream);
    }
    for matrix in matrices {
        matrix.assert_unchanged(&stream);
    }
}
