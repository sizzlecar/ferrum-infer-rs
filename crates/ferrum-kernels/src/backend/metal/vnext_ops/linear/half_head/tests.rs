use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use crate::gguf_blocks::fixtures::oracle_blocks;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};
use half::f16;
use metal::{CommandQueueRef, MTLCommandBufferStatus};

const PREFIX: usize = 8;
const GUARD: f32 = -123.0;

fn retained<T: Copy>(
    runtime: &MetalDeviceRuntime,
    name: &str,
    data: &[T],
    dtype: ElementType,
) -> MetalBufferRegion {
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                std::mem::size_of_val(data) as u64,
                64,
                BufferUsage::Transfer,
                dtype,
            )
            .unwrap(),
        )
        .unwrap();
    // SAFETY: fresh shared allocation has exactly data's byte length.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            std::mem::size_of_val(data),
        );
    }
    region
}

fn read<T: Copy>(region: &MetalBufferRegion) -> Vec<T> {
    // SAFETY: test-owned shared allocation, original scalar type, after GPU completion.
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize)
                .cast::<T>(),
            region.length_bytes() as usize / std::mem::size_of::<T>(),
        )
        .to_vec()
    }
}

struct Case {
    launch: LinearLaunch,
    regions: Vec<MetalBufferRegion>,
    parents: Vec<MetalBufferRegion>,
    input: Vec<f32>,
    output: Vec<f32>,
    weights: Vec<u8>,
    expected: Vec<f64>,
    bounds: Vec<f64>,
}

impl Case {
    fn new(runtime: &MetalDeviceRuntime, rows: u32, width: u32, outputs: u32) -> Self {
        let k = width as usize;
        let n = outputs as usize;
        let m = rows as usize;
        let stride = n + 5;
        let mut input = vec![GUARD; PREFIX + m * k + PREFIX];
        for (index, value) in input[PREFIX..PREFIX + m * k].iter_mut().enumerate() {
            let magnitude =
                0.015625 + ((index * 17 + index / k * 11) % 251) as f32 / 2048.0 + 0.000123;
            *value = if index % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
        }
        let templates = oracle_blocks(GgufBlockFormat::Q6K);
        let template_count = templates.len() / 210;
        let mut weights = vec![0xcc; 16 + n * (k / 256) * 210 + 16];
        let weight_end = weights.len() - 16;
        for (index, block) in weights[16..weight_end].chunks_exact_mut(210).enumerate() {
            let source = (index % template_count) * 210;
            block.copy_from_slice(&templates[source..source + 210]);
            let d = if index % 3 == 0 {
                -1.0 / 512.0
            } else {
                1.0 / 256.0
            };
            block[208..210].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        }
        let mut output = vec![GUARD; PREFIX + m * stride + PREFIX];
        for row in 0..m {
            output[PREFIX + row * stride + 2..PREFIX + row * stride + 2 + n].fill(f32::NAN);
        }
        let mut expected = Vec::with_capacity(m * n);
        let mut bounds = Vec::with_capacity(m * n);
        let ku = k as f64 * 2.0_f64.powi(-24);
        assert!(ku < 1.0);
        let gamma = (ku / (1.0 - ku)).next_up();
        // Half inputs <=1 and weights on a 2^-9 lattice give products on a
        // 2^-33 lattice. The bounded absolute sum is exactly representable in F64.
        assert!(k as f64 * 16.0 * 2.0_f64.powi(33) <= 2.0_f64.powi(53));
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0_f64;
                let mut sum_abs = 0.0_f64;
                for inner in 0..k {
                    let x = f16::from_f32(input[PREFIX + row * k + inner]).to_f32();
                    let offset = 16 + (col * (k / 256) + inner / 256) * 210;
                    let decoded = GgufBlockFormat::Q6K
                        .decode_value(&weights[offset..offset + 210], inner % 256);
                    let w = f16::from_f32(decoded).to_f32();
                    assert!(x.is_finite() && x.abs() <= 1.0);
                    assert!(x == 0.0 || x.abs() >= f16::MIN_POSITIVE.to_f32());
                    assert!(w.is_finite() && w.abs() <= 16.0 && (w * 512.0).fract() == 0.0);
                    let product = f64::from(x) * f64::from(w);
                    sum += product;
                    sum_abs += product.abs();
                }
                expected.push(sum);
                bounds.push(if sum_abs == 0.0 {
                    0.0
                } else {
                    (gamma * sum_abs).next_up()
                });
            }
        }
        let parents = vec![
            retained(runtime, "half.input", &input, ElementType::F32),
            retained(runtime, "half.weight", &weights, ElementType::U8),
            retained(runtime, "half.output", &output, ElementType::F32),
        ];
        let regions = vec![
            parents[0]
                .test_subregion(16..((input.len() - PREFIX) * 4) as u64)
                .unwrap(),
            parents[1].test_subregion(16..weight_end as u64).unwrap(),
            parents[2]
                .test_subregion(16..((output.len() - PREFIX) * 4) as u64)
                .unwrap(),
        ];
        let launch = linear_launch_typed(
            PreparedLinearPart {
                region: 1,
                format: LinearPhysicalFormat::Q6K,
                output_offset: 2,
                out_features: outputs,
                transform: None,
            },
            0,
            2,
            u64::from(rows),
            u64::from(width),
            stride as u64,
            16,
            16,
            ElementType::F32,
        )
        .unwrap();
        validate_half_launch(&regions, launch, &[]).unwrap();
        Self {
            launch,
            regions,
            parents,
            input,
            output,
            weights,
            expected,
            bounds,
        }
    }

    fn run(&self, pipelines: &HalfHeadPipelines, queue: &CommandQueueRef) {
        // SAFETY: reset the complete owned output after the previous command completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.output.as_ptr().cast::<u8>(),
                self.parents[2]
                    .buffer()
                    .contents()
                    .cast::<u8>()
                    .add(self.parents[2].offset_bytes() as usize),
                self.output.len() * 4,
            );
        }
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        pipelines.dispatch(encoder, &self.regions, self.launch);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }

    fn validate(&self) -> Vec<u32> {
        assert_eq!(read::<f32>(&self.parents[0]), self.input);
        assert_eq!(read::<u8>(&self.parents[1]), self.weights);
        let actual = read::<f32>(&self.parents[2]);
        let p = self.launch.params;
        for (index, &value) in actual.iter().enumerate() {
            let logical = index
                .checked_sub(PREFIX)
                .filter(|&x| x < p.rows as usize * p.output_stride as usize)
                .filter(|&x| {
                    (2..2 + p.out_features as usize).contains(&(x % p.output_stride as usize))
                });
            if let Some(x) = logical {
                let index = (x / p.output_stride as usize) * p.out_features as usize
                    + x % p.output_stride as usize
                    - 2;
                let error = (f64::from(value) - self.expected[index]).abs();
                assert!(
                    value.is_finite() && error <= self.bounds[index],
                    "M={} index={index} actual={value} expected={} bound={}",
                    p.rows,
                    self.expected[index],
                    self.bounds[index]
                );
            } else {
                assert_eq!(
                    value.to_bits(),
                    self.output[index].to_bits(),
                    "guard {index}"
                );
            }
        }
        actual.iter().map(|x| x.to_bits()).collect()
    }
}

#[test]
fn half_head_shape_rejects_overflow_and_accounts_for_f32_gather_scratch() {
    let p = LinearParams {
        rows: 32,
        in_features: 4096,
        out_features: 248320,
        output_stride: 248320,
        output_column_offset: 0,
    };
    half_params(p).unwrap();
    for invalid in [
        LinearParams { rows: 0, ..p },
        LinearParams {
            in_features: 257,
            ..p
        },
        LinearParams {
            rows: u32::MAX,
            ..p
        },
        LinearParams {
            out_features: u32::MAX,
            output_stride: u32::MAX,
            ..p
        },
        LinearParams {
            output_column_offset: 1,
            ..p
        },
    ] {
        assert!(half_params(invalid).is_err());
    }
    for count in [1, 2, 3, 7, 8, 32, 33] {
        let layout =
            LastTokenPackedScratchLayout::new(count, 4096, 248320, ElementType::F32).unwrap();
        let estimate = LAST_TOKEN_SCRATCH_PADDING_BYTES
            + count
                * last_token_scratch_bytes_per_sequence(4096, 248320, ElementType::F32).unwrap();
        assert!(layout.required_bytes <= estimate);
        assert_eq!(layout.output_offset_bytes % VALUE_ALIGNMENT_BYTES, 0);
    }
}

#[test]
fn half_head_small_and_tiled_match_independent_half_operand_oracle() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.half-head.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = HalfHeadPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for (rows, width, outputs) in [
        (1, 256, 65),
        (2, 512, 67),
        (3, 768, 129),
        (4, 768, 129),
        (5, 768, 129),
        (7, 768, 129),
        (8, 768, 65),
        (31, 768, 129),
        (32, 768, 128),
        (33, 512, 129),
    ] {
        let case = Case::new(runtime, rows, width, outputs);
        case.run(&pipelines, &queue);
        let first = case.validate();
        case.run(&pipelines, &queue);
        assert_eq!(
            case.validate(),
            first,
            "same half-head route must repeat its bits"
        );
        let mut unaligned = case.launch;
        unaligned.input_offset_bytes += 4;
        assert!(validate_half_launch(&case.regions, unaligned, &[]).is_err());
        for index in 0..3 {
            let mut regions = case.regions.clone();
            let len = regions[index].length_bytes();
            regions[index] = regions[index].test_subregion(0..len - 4).unwrap();
            assert!(validate_half_launch(&regions, case.launch, &[]).is_err());
        }
        let mut wrong_format = case.launch;
        wrong_format.format = LinearPhysicalFormat::Q4K;
        assert!(validate_half_launch(&case.regions, wrong_format, &[]).is_err());
    }
}

#[test]
fn half_head_f32_gather_and_scatter_preserve_final_rows_and_guards() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.half-head.gather").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = HalfHeadPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    let case = Case::new(runtime, 8, 256, 65);
    let p = case.launch.params;
    let mut source = vec![GUARD; 8 * 2 * 256];
    for row in 0..8 {
        source[(row * 2 + 1) * 256..(row * 2 + 2) * 256]
            .copy_from_slice(&case.input[PREFIX + row * 256..PREFIX + (row + 1) * 256]);
    }
    let source = retained(runtime, "half.gather-source", &source, ElementType::F32);
    // The destination must not already contain the expected rows: omitting the
    // gather or selecting a participant's first token must fail this test.
    unsafe {
        std::slice::from_raw_parts_mut(
            case.parents[0]
                .buffer()
                .contents()
                .cast::<u8>()
                .add(case.parents[0].offset_bytes() as usize)
                .cast::<f32>()
                .add(PREFIX),
            8 * 256,
        )
        .fill(f32::NAN);
    }
    let outputs: Vec<_> = (0..8)
        .map(|_| {
            retained(
                runtime,
                "half.scatter-output",
                &vec![GUARD; 67],
                ElementType::F32,
            )
        })
        .collect();
    let command = queue.new_command_buffer();
    let gather = command.new_blit_command_encoder();
    for row in 0..8_u64 {
        gather.copy_from_buffer(
            source.buffer(),
            source.offset_bytes() + (row * 2 + 1) * 256 * 4,
            case.parents[0].buffer(),
            case.parents[0].offset_bytes() + (PREFIX as u64 + row * 256) * 4,
            256 * 4,
        );
    }
    gather.end_encoding();
    let encoder = command.new_compute_command_encoder();
    pipelines.dispatch(encoder, &case.regions, case.launch);
    encoder.end_encoding();
    let scatter = command.new_blit_command_encoder();
    for (row, output) in outputs.iter().enumerate() {
        scatter.copy_from_buffer(
            case.parents[2].buffer(),
            case.parents[2].offset_bytes()
                + (PREFIX as u64 + row as u64 * u64::from(p.output_stride) + 2) * 4,
            output.buffer(),
            output.offset_bytes() + 4,
            u64::from(p.out_features) * 4,
        );
    }
    scatter.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let packed = case.validate();
    for (row, output) in outputs.iter().enumerate() {
        let actual = read::<f32>(output);
        assert_eq!(actual[0], GUARD);
        assert_eq!(actual[66], GUARD);
        for col in 0..65 {
            assert_eq!(
                actual[col + 1].to_bits(),
                packed[PREFIX + row * p.output_stride as usize + 2 + col]
            );
        }
    }
}

/// Exact constant Q6 fixtures exercise the arithmetic domain independently of
/// the normal-only dense oracle's error bound. All finite sums here are binary
/// powers times an integer <=256 and are exactly representable in F32.
fn assert_edge_projection(
    runtime: &MetalDeviceRuntime,
    pipelines: &HalfHeadPipelines,
    queue: &CommandQueueRef,
    rows: u32,
    input_value: f32,
    delta: f16,
    scale: i8,
    quant: i8,
    sparse_input: bool,
) {
    let k = 256_usize;
    let n = 65_usize;
    let stride = n + 3;
    let mut input = vec![GUARD; PREFIX + rows as usize * k + PREFIX];
    for row in 0..rows as usize {
        let payload = &mut input[PREFIX + row * k..PREFIX + (row + 1) * k];
        payload.fill(if sparse_input { 0.0 } else { input_value });
        payload[0] = input_value;
    }
    let code = u8::try_from(i16::from(quant) + 32).unwrap();
    assert!(code < 64);
    let mut block = [0_u8; 210];
    block[..128].fill((code & 15) * 17);
    block[128..192].fill((code >> 4) * 85);
    block[192..208].fill(scale as u8);
    block[208..210].copy_from_slice(&delta.to_le_bytes());
    let mut weights = vec![0xcc; 16 + n * 210 + 16];
    for destination in weights[16..16 + n * 210].chunks_exact_mut(210) {
        destination.copy_from_slice(&block);
    }
    let mut expected = 0.0_f32;
    for inner in 0..k {
        let x = f16::from_f32(input[PREFIX + inner]).to_f32();
        let w = f16::from_f32(GgufBlockFormat::Q6K.decode_value(&block, inner)).to_f32();
        expected += x * w;
    }
    let mut output = vec![GUARD; PREFIX + rows as usize * stride + PREFIX];
    for row in 0..rows as usize {
        // A NaN expected result must be produced by the shader, not inherited
        // from the output poison.
        output[PREFIX + row * stride + 1..PREFIX + row * stride + 1 + n].fill(12345.0);
    }
    let parents = [
        retained(runtime, "half.edge-input", &input, ElementType::F32),
        retained(runtime, "half.edge-weight", &weights, ElementType::U8),
        retained(runtime, "half.edge-output", &output, ElementType::F32),
    ];
    let regions = vec![
        parents[0]
            .test_subregion(16..((input.len() - PREFIX) * 4) as u64)
            .unwrap(),
        parents[1]
            .test_subregion(16..(16 + n * 210) as u64)
            .unwrap(),
        parents[2]
            .test_subregion(16..((output.len() - PREFIX) * 4) as u64)
            .unwrap(),
    ];
    let launch = linear_launch_typed(
        PreparedLinearPart {
            region: 1,
            format: LinearPhysicalFormat::Q6K,
            output_offset: 1,
            out_features: n as u32,
            transform: None,
        },
        0,
        2,
        u64::from(rows),
        k as u64,
        stride as u64,
        16,
        16,
        ElementType::F32,
    )
    .unwrap();
    validate_half_launch(&regions, launch, &[]).unwrap();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    pipelines.dispatch(encoder, &regions, launch);
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert_eq!(
        read::<f32>(&parents[0])
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        input.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(read::<u8>(&parents[1]), weights);
    for (index, actual) in read::<f32>(&parents[2]).into_iter().enumerate() {
        let logical = index
            .checked_sub(PREFIX)
            .filter(|&i| i < rows as usize * stride)
            .is_some_and(|i| (1..1 + n).contains(&(i % stride)));
        if logical {
            assert!(
                if expected.is_nan() { actual.is_nan() } else { actual == expected },
                "M={rows} x={input_value:?} d={delta:?} scale={scale} q={quant} sparse={sparse_input} actual={actual:?} expected={expected:?} index={index}"
            );
        } else {
            assert_eq!(
                actual.to_bits(),
                output[index].to_bits(),
                "edge guard {index}"
            );
        }
    }
}

#[test]
fn half_head_small_and_tiled_preserve_rounding_subnormals_and_nonfinite_classes() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.half-head.edge").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = HalfHeadPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    let tie_to_zero = 2.0_f32.powi(-25);
    let tie_to_one = 1.000_488_3_f32;
    for rows in [1, 8, 32] {
        for input in [
            0.0,
            -0.0,
            tie_to_zero,
            tie_to_zero.next_up(),
            f16::from_bits(1).to_f32(),
            -f16::from_bits(1).to_f32(),
            f16::from_bits(0x03ff).to_f32(),
            f16::MIN_POSITIVE.to_f32(),
            tie_to_one,
            tie_to_one.next_up(),
            65_504.0,
            65_520.0,
            -65_520.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert_edge_projection(
                runtime,
                &pipelines,
                &queue,
                rows,
                input,
                f16::ONE,
                1,
                1,
                true,
            );
        }
        for (delta, scale, quant, input) in [
            (f16::from_bits(1), 1, 1, 1.0),
            (f16::from_bits(1), 1, 1, f16::from_bits(1).to_f32()),
            (f16::from_bits(0x03ff), 1, -1, 1.0),
            (f16::MAX, 2, 1, 1.0),
            (f16::MAX, 2, -1, 1.0),
            (f16::INFINITY, 1, 1, 1.0),
            (f16::NEG_INFINITY, 1, 1, 1.0),
            (f16::NAN, 1, 1, 1.0),
            (f16::INFINITY, 0, 1, 1.0),
            (f16::INFINITY, 1, 0, 1.0),
        ] {
            assert_edge_projection(
                runtime, &pipelines, &queue, rows, input, delta, scale, quant, false,
            );
        }
    }
}
