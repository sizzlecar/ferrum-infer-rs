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
