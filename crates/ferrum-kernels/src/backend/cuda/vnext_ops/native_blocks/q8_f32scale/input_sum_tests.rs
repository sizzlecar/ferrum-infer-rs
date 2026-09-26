//! Real production-policy launches with independent numeric references.
use super::*;
use crate::gguf_blocks::{q4k_q8_reference, q56k_q8_reference, q8_input_sum_reference};
use cudarc::driver::{CudaSlice, DevicePtr};
use ferrum_interfaces::vnext::WeightId;
use half::f16;

struct Guarded {
    gpu: CudaSlice<u8>,
    payload: Vec<u8>,
}

impl Guarded {
    fn new(stream: &Arc<CudaStream>, payload: Vec<u8>) -> Self {
        let mut bytes = vec![0xab; 16];
        bytes.extend(&payload);
        bytes.extend([0xab; 16]);
        Self {
            gpu: stream.clone_htod(&bytes).unwrap(),
            payload,
        }
    }
    fn pointer(&self, stream: &CudaStream) -> u64 {
        self.gpu.device_ptr(stream).0 + 16
    }
    fn read(&self, stream: &Arc<CudaStream>) -> Vec<u8> {
        stream.synchronize().unwrap();
        let bytes = stream.clone_dtoh(&self.gpu).unwrap();
        assert!(bytes[..16]
            .iter()
            .chain(&bytes[16 + self.payload.len()..])
            .all(|&b| b == 0xab));
        bytes[16..16 + self.payload.len()].to_vec()
    }
}

fn halves(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

fn check_pack(bytes: &[u8], input: &[f16], rows: usize, columns: usize) {
    let packed = q8_input_sum_reference::pack_rows(input, rows, columns);
    let layout = PackLayout::with_policy(rows as u64, columns as u64, Q8SumPolicy::Input).unwrap();
    let floats = |start: usize, expected: &[f32]| {
        for (got, &want) in bytes[start..start + expected.len() * 4]
            .chunks_exact(4)
            .zip(expected)
        {
            let got = f32::from_le_bytes(got.try_into().unwrap());
            if want.is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
    };
    floats(0, &packed.scales);
    floats(layout.scales_bytes as usize, &packed.input_sums);
    assert_eq!(
        &bytes[layout.words_offset as usize..],
        packed.quants.iter().map(|q| *q as u8).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires SM80+ CUDA; pack nonfinite/subnormal/cancellation contract"]
fn q8_input_sum_pack_edge_values_on_cuda() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let q8 = Q8F32ScaleKernels::load_with_policy(&context, Q8SumPolicy::Input).unwrap();
    let mut input = vec![f16::ZERO; 256];
    input[32..64].fill(f16::NEG_ZERO);
    for i in 64..128 {
        input[i] = f16::from_bits(1 + ((i - 64) % 31) as u16);
    }
    for i in 128..192 {
        input[i] = f16::from_f32(if i % 2 == 0 { 65504.0 } else { -65472.0 });
    }
    input[192] = f16::INFINITY;
    input[224] = f16::NAN;
    let x = Guarded::new(&stream, halves(&input));
    let layout = PackLayout::with_policy(1, 256, Q8SumPolicy::Input).unwrap();
    let scratch = Guarded::new(&stream, vec![0x7e; layout.total_bytes as usize]);
    let xp = x.pointer(&stream);
    let sp = scratch.pointer(&stream);
    let wp = sp + layout.words_offset;
    let sums = sp + layout.scales_bytes;
    let mut launch = stream.launch_builder(&q8.pack);
    launch
        .arg(&xp)
        .arg(&sp)
        .arg(&wp)
        .arg(&1_u32)
        .arg(&256_u32)
        .arg(&sums);
    // SAFETY: three disjoint spans are held by scratch; complete K256 input.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (2, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
    check_pack(&scratch.read(&stream), &input, 1, 256);
    assert_eq!(x.read(&stream), x.payload);
}

#[test]
#[ignore = "requires SM80+ CUDA; actual selector, Q4/Q5/Q6 and output tails/strides"]
fn q8_input_sum_projection_routes_on_cuda() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let strict = CudaNativeBlockKernels::load(&context).unwrap();
    let candidate = Q8F32ScaleKernels::load_with_policy(&context, Q8SumPolicy::Input).unwrap();
    let old = Q8F32ScaleKernels::load(&context).unwrap();
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        for rows in [1, 4, 8, 33] {
            let columns = 512;
            let outputs = 17;
            let stride = outputs + 5;
            let offset = 2;
            let input: Vec<_> = (0..rows * columns)
                .map(|i| f16::from_f32(((i * 13 % 67) as f32 - 29.0) / 127.0))
                .collect();
            let mut raw = vec![0xcc; 5]; // Native block decode must remain byte-safe.
            for col in 0..outputs {
                for block in 0..columns / 256 {
                    match format {
                        GgufBlockFormat::Q4K => {
                            raw.extend(q4k_q8_reference::fixture_block(col, block).encode())
                        }
                        GgufBlockFormat::Q5K => {
                            raw.extend(q56k_q8_reference::fixture_q5(col, block).encode())
                        }
                        GgufBlockFormat::Q6K => {
                            raw.extend(q56k_q8_reference::fixture_q6(col, block).encode())
                        }
                        _ => unreachable!(),
                    }
                }
            }
            raw.extend([0xcc; 5]);
            let w = Guarded::new(&stream, raw);
            let x = Guarded::new(&stream, halves(&input));
            let initial = vec![f16::from_f32(-12345.0); rows * stride];
            let y = Guarded::new(&stream, halves(&initial));
            let layout =
                PackLayout::with_policy(rows as u64, columns as u64, Q8SumPolicy::Input).unwrap();
            let scratch = Guarded::new(&stream, vec![0x7e; layout.total_bytes as usize]);
            let part = weights::MatrixPart {
                component_id: WeightId::new("component.input-sum-test").unwrap(),
                format: weights::MatrixFormat::Block(format),
                rows: outputs as u32,
                columns: columns as u32,
                output_offset: offset as u32,
                transform: None,
                signs_region: None,
            };
            candidate
                .launch(
                    &strict,
                    &stream,
                    &[part.clone()],
                    &[w.pointer(&stream) + 5],
                    x.pointer(&stream),
                    y.pointer(&stream),
                    rows as u32,
                    columns as u32,
                    stride as u32,
                    scratch.pointer(&stream),
                )
                .unwrap();
            let result = y.read(&stream);
            let result_half: Vec<_> = result
                .chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                .collect();
            for row in 0..rows {
                for col in 0..stride {
                    let actual = result_half[row * stride + col];
                    if !(offset..offset + outputs).contains(&col) {
                        assert_eq!(actual.to_bits(), initial[0].to_bits());
                        continue;
                    }
                    let col = col - offset;
                    let xrow = &input[row * columns..][..columns];
                    let (policy, absolute) = match format {
                        GgufBlockFormat::Q4K => {
                            let r = q8_input_sum_reference::dot_q4(
                                xrow,
                                &(0..columns / 256)
                                    .map(|b| q4k_q8_reference::fixture_block(col, b))
                                    .collect::<Vec<_>>(),
                            );
                            (r.policy, r.expanded_abs_terms)
                        }
                        GgufBlockFormat::Q5K => {
                            let r = q8_input_sum_reference::dot_q5(
                                xrow,
                                &(0..columns / 256)
                                    .map(|b| q56k_q8_reference::fixture_q5(col, b))
                                    .collect::<Vec<_>>(),
                            );
                            (r.policy, r.expanded_abs_terms)
                        }
                        GgufBlockFormat::Q6K => {
                            let r = q56k_q8_reference::dot_q6(
                                xrow,
                                &(0..columns / 256)
                                    .map(|b| q56k_q8_reference::fixture_q6(col, b))
                                    .collect::<Vec<_>>(),
                            );
                            (r.policy, r.expanded_abs_terms)
                        }
                        _ => unreachable!(),
                    };
                    let nu = ((columns / 32).div_ceil(4) + 9) as f64 * f64::from(f32::EPSILON);
                    let error = nu / (1.0 - nu) * absolute;
                    let bound =
                        error + 0.0009765625 * (policy.abs() + error) + f16::from_bits(1).to_f64();
                    assert!(actual.is_finite() && (actual.to_f64() - f16::from_f64(policy).to_f64()).abs() <= bound, "{format:?} rows={rows} row={row} col={col}: actual={actual} policy={policy} bound={bound}");
                }
            }
            check_pack(&scratch.read(&stream), &input, rows, columns);
            assert_eq!(x.read(&stream), x.payload);
            assert_eq!(w.read(&stream), w.payload);
            if format == GgufBlockFormat::Q6K {
                let control = Guarded::new(&stream, halves(&initial));
                old.launch(
                    &strict,
                    &stream,
                    &[part],
                    &[w.pointer(&stream) + 5],
                    x.pointer(&stream),
                    control.pointer(&stream),
                    rows as u32,
                    columns as u32,
                    stride as u32,
                    scratch.pointer(&stream),
                )
                .unwrap();
                assert_eq!(
                    control.read(&stream),
                    result,
                    "Q6 must preserve the old arithmetic"
                );
                scratch.read(&stream);
            }
        }
    }
}
