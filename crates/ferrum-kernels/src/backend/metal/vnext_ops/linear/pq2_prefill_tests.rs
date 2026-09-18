//! Same-tile PQ2 format specialization with F32 inputs, operands and accumulation.

use super::*;
use half::f16;
use metal::{Buffer, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

const INPUT_PREFIX: usize = 4;
const WEIGHT_PREFIX: usize = 18;
const OUTPUT_PREFIX: usize = 8;
const GUARD: f32 = -123.0;
const PACKED_CODES: [u8; 4] = [0xe4, 0x1b, 0xff, 0x55];
// Literal LSB-first PQ2 coefficients, independent of the device/CPU decoders.
const COEFFICIENTS: [[f64; 4]; 4] = [
    [-1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, 0.0, -1.0],
    [2.0, 2.0, 2.0, 2.0],
    [0.0, 0.0, 0.0, 0.0],
];

fn buffer<T>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[derive(Clone, Copy)]
enum PrefillTile {
    GenericM32,
    SpecializedM32,
    SpecializedM64,
    Production,
}

struct Fixture {
    params: LinearParams,
    input: Buffer,
    weight: Buffer,
    output: Buffer,
    input_values: Vec<f32>,
    weight_bytes: Vec<u8>,
    initial_output: Vec<f16>,
    // Each row has four different literal weight patterns, repeated across N.
    // This bounds the independent full-output oracle at realistic matrix sizes.
    reference: Vec<[(f64, f64); 4]>,
}

impl Fixture {
    fn new(device: &Device, rows: u32, width: u32, outputs: u32, wide_scale: bool) -> Self {
        assert!(width > 0 && width.is_multiple_of(128));
        let scales = if wide_scale {
            [
                (0x7bff_u16, 65504.0),
                (0xfbff, -65504.0),
                (0x3800, 0.5),
                (0xb400, -0.25),
            ]
        } else {
            [
                (0x3800_u16, 0.5),
                (0xb400, -0.25),
                (0, 0.0),
                (0x0400, 1.0 / 16384.0),
            ]
        };
        let mut input_values = vec![GUARD; INPUT_PREFIX];
        for i in 0..rows * width {
            let centered = ((i * 37 % 63) as i32 - 31) as f32;
            input_values.push(if wide_scale {
                centered / 16777216.0
            } else {
                centered / 512.0 + (i % 7) as f32 / 1048576.0
            });
        }
        input_values.extend([GUARD; 4]);
        let mut weight_bytes = vec![0xcc_u8; WEIGHT_PREFIX];
        for column in 0..outputs {
            for block in 0..width / 128 {
                weight_bytes
                    .extend_from_slice(&scales[((column + block) % 4) as usize].0.to_le_bytes());
                for byte in 0..32 {
                    weight_bytes.push(PACKED_CODES[((column + block + byte) % 4) as usize]);
                }
            }
        }
        weight_bytes.extend([0xcc; 16]);
        let reference = (0..rows as usize)
            .map(|row| {
                std::array::from_fn(|pattern| {
                    let mut sum = 0.0;
                    let mut absolute_sum = 0.0;
                    for k in 0..width as usize {
                        let block = k / 128;
                        let byte = (k % 128) / 4;
                        let w = scales[(pattern + block) % 4].1
                            * COEFFICIENTS[(pattern + block + byte) % 4][k % 4];
                        let x = f64::from(input_values[INPUT_PREFIX + row * width as usize + k]);
                        let product = x * w;
                        sum += product;
                        absolute_sum += product.abs();
                    }
                    (sum, absolute_sum)
                })
            })
            .collect();
        let params = LinearParams {
            rows,
            in_features: width,
            out_features: outputs,
            output_stride: outputs + 5,
            output_column_offset: 2,
        };
        let mut initial_output =
            vec![f16::from_f32(GUARD); OUTPUT_PREFIX + (rows * params.output_stride) as usize + 8];
        for row in 0..rows as usize {
            let first = OUTPUT_PREFIX + row * params.output_stride as usize + 2;
            initial_output[first..first + outputs as usize].fill(f16::NAN);
        }
        Self {
            params,
            input: buffer(device, &input_values),
            weight: buffer(device, &weight_bytes),
            output: buffer(device, &initial_output),
            input_values,
            weight_bytes,
            initial_output,
            reference,
        }
    }

    fn run(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        specialized: bool,
        dispatches: usize,
    ) -> Option<f64> {
        self.run_tile(
            pipelines,
            queue,
            if specialized {
                PrefillTile::SpecializedM32
            } else {
                PrefillTile::GenericM32
            },
            dispatches,
        )
    }

    fn run_tile(
        &self,
        pipelines: &MetalLinearPipelines,
        queue: &CommandQueueRef,
        tile: PrefillTile,
        dispatches: usize,
    ) -> Option<f64> {
        assert!(dispatches > 0);
        // SAFETY: no pending command uses this shared output; every run waits
        // for completion. Its declared length/type match initial_output.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.initial_output.as_ptr(),
                self.output.contents().cast::<f16>(),
                self.initial_output.len(),
            );
        }
        let (pipeline, dispatch) = match tile {
            PrefillTile::GenericM32 => (
                &pipelines.native.gemm_input_f32_output_f16,
                LinearDispatchKind::NativeTiledGemm,
            ),
            PrefillTile::SpecializedM32 => (
                &pipelines.native.pq2_gemm_input_f32_output_f16,
                LinearDispatchKind::NativeTiledGemm,
            ),
            PrefillTile::SpecializedM64 => (
                pipelines
                    .native
                    .pq2_gemm_input_f32_output_f16_m64
                    .as_ref()
                    .expect("PQ2 M64 experiment requires a supported M64 pipeline"),
                LinearDispatchKind::NativeTiledGemmM64,
            ),
            PrefillTile::Production => {
                pipelines.mixed_input_tiled_pipeline(GgufBlockFormat::Pq2_0, self.params)
            }
        };
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.input), 16);
        encoder.set_buffer(1, Some(&self.weight), WEIGHT_PREFIX as u64);
        encoder.set_buffer(2, Some(&self.output), 16);
        encoder.set_bytes(
            3,
            std::mem::size_of::<LinearParams>() as u64,
            &self.params as *const _ as *const c_void,
        );
        bind_native_block(encoder, GgufBlockFormat::Pq2_0, 4);
        for _ in 0..dispatches {
            dispatch_linear_grid(encoder, self.params, dispatch);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        super::microbench::gpu_elapsed_ns(command).map(|ns| ns / dispatches as f64)
    }

    fn validate(&self) -> Vec<u16> {
        // SAFETY: these shared allocations have the recorded lengths/types and
        // every submitted command has completed before this method is called.
        let actual = unsafe {
            std::slice::from_raw_parts(
                self.output.contents().cast::<f16>(),
                self.initial_output.len(),
            )
        };
        let input = unsafe {
            std::slice::from_raw_parts(self.input.contents().cast::<f32>(), self.input_values.len())
        };
        let weights = unsafe {
            std::slice::from_raw_parts(self.weight.contents().cast::<u8>(), self.weight_bytes.len())
        };
        assert_eq!(
            input, self.input_values,
            "mixed prefill modified its F32 input"
        );
        assert_eq!(
            weights, self.weight_bytes,
            "mixed prefill modified packed weights"
        );
        let stride = self.params.output_stride as usize;
        let columns = self.params.out_features as usize;
        for (index, value) in actual.iter().enumerate() {
            let location = index
                .checked_sub(OUTPUT_PREFIX)
                .filter(|index| *index < self.params.rows as usize * stride)
                .and_then(|index| {
                    (2..columns + 2)
                        .contains(&(index % stride))
                        .then(|| (index / stride, index % stride - 2))
                });
            let Some((row, column)) = location else {
                assert_eq!(*value, f16::from_f32(GUARD), "output guard at {index}");
                continue;
            };
            let (expected, absolute_sum) = self.reference[row][column % 4];
            // Bound a sequential F32 multiply/add chain and final F16 store,
            // as in native_tests. No F16 operand-rounding allowance is added.
            let n_u = (2 * self.params.in_features + 8) as f64 * f64::from(f32::EPSILON) / 2.0;
            let accumulation = n_u / (1.0 - n_u) * absolute_sum;
            let rounding = (expected.abs() + accumulation) * 2_f64.powi(-11) + 2_f64.powi(-25);
            assert!(
                value.is_finite()
                    && (f64::from(value.to_f32()) - expected).abs() <= accumulation + rounding,
                "PQ2 mixed prefill ({row},{column}) {} != {expected}, bound={}",
                value.to_f32(),
                accumulation + rounding,
            );
        }
        actual.iter().map(|value| value.to_bits()).collect()
    }
}

#[test]
fn pq2_mixed_prefill_specialization_preserves_f32_operands_tiles_and_guards_on_metal() {
    let device = Device::system_default().expect("PQ2 mixed prefill conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let m32_params = LinearParams {
        rows: 32,
        in_features: 384,
        out_features: 1024,
        output_stride: 1024,
        output_column_offset: 0,
    };
    assert!(std::ptr::eq(
        pipelines
            .mixed_input_tiled_pipeline(GgufBlockFormat::Pq2_0, m32_params)
            .0,
        &pipelines.native.pq2_gemm_input_f32_output_f16,
    ));
    assert!(std::ptr::eq(
        pipelines
            .mixed_input_tiled_pipeline(GgufBlockFormat::Iq4Xs, m32_params)
            .0,
        &pipelines.native.gemm_input_f32_output_f16,
    ));
    // K384 uses a 102-byte row pitch; offset18 and odd rows forbid assuming
    // vector-aligned blocks. Cover both M32/N64 tails and the product row count.
    for (rows, width, outputs, wide_scale) in [
        (31, 384, 65, false),
        (32, 384, 1024, false),
        (33, 384, 1025, false),
        (128, 1024, 1025, false),
        (33, 384, 1025, true),
    ] {
        let fixture = Fixture::new(&device, rows, width, outputs, wide_scale);
        let _ = fixture.run(&pipelines, &queue, false, 1);
        let reference = fixture.validate();
        let _ = fixture.run(&pipelines, &queue, true, 1);
        assert_eq!(
            fixture.validate(),
            reference,
            "generic/specialized M32 changed output bits"
        );
    }
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_mixed_prefill_specialization_gpu_microbench() {
    let device = Device::system_default().expect("PQ2 mixed prefill timing requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    const DISPATCHES: usize = 8;
    const WARMUP_PAIRS: usize = 2;
    const PAIRS: usize = 10;
    // Real Bonsai principal projection dimensions; synthetic packed weights.
    // This isolates the same M32 computation, not whole-model prefill quality.
    for (name, width, outputs) in [
        ("ffn_gate_or_up", 5120, 17408),
        ("ffn_down", 17408, 5120),
        ("gdn_output", 6144, 5120),
    ] {
        let fixture = Fixture::new(&device, 128, width, outputs, false);
        let _ = fixture.run(&pipelines, &queue, false, 1);
        let reference = fixture.validate();
        let _ = fixture.run(&pipelines, &queue, true, 1);
        assert_eq!(fixture.validate(), reference);
        for pair in 0..WARMUP_PAIRS {
            let order = if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for specialized in order {
                let _ = fixture.run(&pipelines, &queue, specialized, DISPATCHES);
                assert_eq!(fixture.validate(), reference);
            }
        }
        let mut samples = Vec::new();
        for pair in 0..PAIRS {
            let order = if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            let mut times = [0.0_f64; 2];
            for specialized in order {
                let elapsed = fixture
                    .run(&pipelines, &queue, specialized, DISPATCHES)
                    .expect("completed GPU command must expose valid GPU timestamps");
                assert_eq!(fixture.validate(), reference);
                times[usize::from(specialized)] = elapsed;
            }
            samples.push(serde_json::json!({
                "pair": pair, "specialized_first": order[0],
                "generic_gpu_ns_per_dispatch": times[0],
                "specialized_gpu_ns_per_dispatch": times[1],
                "specialized_over_generic": times[1] / times[0],
            }));
        }
        println!(
            "{}",
            serde_json::json!({
                "schema_version": 1,
                "kind": "pq2_mixed_prefill_specialization_gpu_microbench",
                "device": device.name(), "shape": name,
                "rows": 128, "in_features": width, "out_features": outputs,
                "weight_format": "quantization.gguf.pq2-0",
                "input_dtype": "f32", "operand_dtype": "f32", "accumulator_dtype": "f32",
                "output_dtype": "f16", "tile_rows": 32,
            "dispatches_per_command": DISPATCHES,
            "warmup_pairs": WARMUP_PAIRS,
                "output_bitwise_equal": true, "independent_oracle": "literal-pq2-f64",
                "working_set": "one-resident-matrix-per-shape",
                "synthetic_weight_row_period": 4, "samples": samples,
            })
        );
    }
}

#[test]
fn pq2_mixed_prefill_m64_selection_requires_aligned_rows_wide_output_and_capability() {
    for (rows, outputs, aligned_wide) in [
        (0, 4096, false),
        (32, 4096, false),
        (63, 4096, false),
        (64, 4095, false),
        (64, 4096, true),
        (65, 4096, false),
        (96, 4096, false),
        (128, 1024, false),
        (128, 4096, true),
        (129, 4096, false),
        (192, 5120, true),
    ] {
        let params = LinearParams {
            rows,
            in_features: 5120,
            out_features: outputs,
            output_stride: outputs,
            output_column_offset: 0,
        };
        for format in [GgufBlockFormat::Pq2_0, GgufBlockFormat::Iq4Xs] {
            for available in [false, true] {
                assert_eq!(
                    pq2_mixed_prefill_m64_supported(format, params, available),
                    aligned_wide && format == GgufBlockFormat::Pq2_0 && available,
                );
            }
        }
    }
    use super::super::native_blocks::supports_m64_threadgroup;
    assert!(supports_m64_threadgroup(32, 256, 0, 16384));
    assert!(supports_m64_threadgroup(32, 256, 1024, 17408));
    for (simd, threads, static_bytes, device_bytes) in [
        (16, 256, 0, 16384),
        (32, 255, 0, 16384),
        (32, 256, 0, 16383),
        (32, 256, 1024, 17407),
        (32, 256, u64::MAX, u64::MAX),
    ] {
        assert!(!supports_m64_threadgroup(
            simd,
            threads,
            static_bytes,
            device_bytes
        ));
    }
}

#[test]
fn pq2_mixed_prefill_production_dispatch_and_m32_fallback_preserve_output() {
    let device = Device::system_default().expect("PQ2 production conformance requires Metal");
    let mut pipelines = MetalLinearPipelines::new(&device).unwrap();
    assert!(pipelines.native.pq2_gemm_input_f32_output_f16_m64.is_some());
    let queue = device.new_command_queue();
    for (rows, outputs, expected_dispatch) in [
        (64, 4096, LinearDispatchKind::NativeTiledGemmM64),
        (128, 4096, LinearDispatchKind::NativeTiledGemmM64),
        (96, 4096, LinearDispatchKind::NativeTiledGemm),
        (65, 4096, LinearDispatchKind::NativeTiledGemm),
        (64, 1024, LinearDispatchKind::NativeTiledGemm),
        (64, 4095, LinearDispatchKind::NativeTiledGemm),
    ] {
        let fixture = Fixture::new(&device, rows, 384, outputs, false);
        assert_eq!(
            pipelines
                .mixed_input_tiled_pipeline(GgufBlockFormat::Pq2_0, fixture.params)
                .1,
            expected_dispatch
        );
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::GenericM32, 1);
        let reference = fixture.validate();
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::Production, 1);
        assert_eq!(fixture.validate(), reference);
    }
    // Exercise an actual M32 submission for the same M64-eligible shape when
    // device/PSO capabilities cannot supply the optional pipeline.
    pipelines.native.pq2_gemm_input_f32_output_f16_m64 = None;
    let fixture = Fixture::new(&device, 64, 384, 4096, true);
    let (pipeline, dispatch) =
        pipelines.mixed_input_tiled_pipeline(GgufBlockFormat::Pq2_0, fixture.params);
    assert_eq!(dispatch, LinearDispatchKind::NativeTiledGemm);
    assert!(std::ptr::eq(
        pipeline,
        &pipelines.native.pq2_gemm_input_f32_output_f16
    ));
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::GenericM32, 1);
    let reference = fixture.validate();
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::Production, 1);
    assert_eq!(fixture.validate(), reference);
}

#[test]
fn pq2_mixed_prefill_m64_preserves_f32_operands_tiles_and_guards_on_metal() {
    let device = Device::system_default().expect("PQ2 M64 conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let pipeline = pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64
        .as_ref()
        .expect("PQ2 M64 conformance requires a supported M64 pipeline");
    assert!(super::super::native_blocks::supports_m64_threadgroup(
        pipeline.thread_execution_width(),
        pipeline.max_total_threads_per_threadgroup(),
        pipeline.static_threadgroup_memory_length(),
        device.max_threadgroup_memory_length(),
    ));
    let queue = device.new_command_queue();
    for (rows, width, outputs, wide_scale) in [
        (31, 384, 65, false),
        (32, 384, 1024, false),
        (63, 384, 1025, false),
        (64, 384, 1024, false),
        (65, 384, 1025, false),
        (128, 1024, 1025, false),
        (129, 384, 65, true),
    ] {
        let fixture = Fixture::new(&device, rows, width, outputs, wide_scale);
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::GenericM32, 1);
        let reference = fixture.validate();
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::SpecializedM64, 1);
        assert_eq!(
            fixture.validate(),
            reference,
            "PQ2 M32/M64 changed output bits"
        );
    }
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_mixed_prefill_m64_gpu_microbench() {
    measure_m64_shapes(&[
        ("ffn_gate_or_up", 128, 5120, 17408),
        ("ffn_down", 128, 17408, 5120),
        ("gdn_output", 128, 6144, 5120),
    ]);
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_mixed_prefill_m64_boundary_gpu_microbench() {
    measure_m64_shapes(&[
        ("aligned_m64_n4096", 64, 5120, 4096),
        ("aligned_m64_n1024", 64, 5120, 1024),
        ("partial_m96_n4096", 96, 5120, 4096),
        ("partial_m96_n1024", 96, 5120, 1024),
        ("aligned_m128_n4096", 128, 5120, 4096),
        ("attention_k_or_v_n1024", 128, 5120, 1024),
    ]);
}

fn measure_m64_shapes(shapes: &[(&str, u32, u32, u32)]) {
    let device = Device::system_default().expect("PQ2 M64 timing requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    const TILES: [PrefillTile; 3] = [
        PrefillTile::GenericM32,
        PrefillTile::SpecializedM32,
        PrefillTile::SpecializedM64,
    ];
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [2, 0, 1],
        [1, 0, 2],
    ];
    const DISPATCHES: usize = 8;
    const WARMUP_ROUNDS: usize = 2;
    const ROUNDS: usize = 12;
    for &(name, rows, width, outputs) in shapes {
        let fixture = Fixture::new(&device, rows, width, outputs, false);
        let _ = fixture.run_tile(&pipelines, &queue, TILES[0], 1);
        let reference = fixture.validate();
        for order in ORDERS.iter().take(WARMUP_ROUNDS) {
            for &index in order {
                let _ = fixture.run_tile(&pipelines, &queue, TILES[index], DISPATCHES);
                assert_eq!(fixture.validate(), reference);
            }
        }
        let mut samples = Vec::new();
        for round in 0..ROUNDS {
            let order = ORDERS[round % ORDERS.len()];
            let mut times = [0.0_f64; 3];
            for index in order {
                times[index] = fixture
                    .run_tile(&pipelines, &queue, TILES[index], DISPATCHES)
                    .expect("completed GPU command must expose valid GPU timestamps");
                assert_eq!(fixture.validate(), reference);
            }
            samples.push(serde_json::json!({
                "round": round, "order": order,
                "generic_m32_gpu_ns_per_dispatch": times[0],
                "specialized_m32_gpu_ns_per_dispatch": times[1],
                "specialized_m64_gpu_ns_per_dispatch": times[2],
                "m64_over_generic_m32": times[2] / times[0],
                "m64_over_specialized_m32": times[2] / times[1],
            }));
        }
        println!(
            "{}",
            serde_json::json!({
                "schema_version": 1,
                "kind": "pq2_mixed_prefill_m64_gpu_microbench",
                "device": device.name(), "shape": name,
            "rows": rows, "in_features": width, "out_features": outputs,
                "weight_format": "quantization.gguf.pq2-0",
                "input_dtype": "f32", "operand_dtype": "f32", "accumulator_dtype": "f32",
                "output_dtype": "f16", "variant_tile_rows": [32, 32, 64],
                "variant_names": ["generic_m32", "specialized_m32", "specialized_m64"],
                "dispatches_per_command": DISPATCHES, "warmup_rounds": WARMUP_ROUNDS,
                "output_bitwise_equal": true, "independent_oracle": "literal-pq2-f64",
                "working_set": "one-resident-matrix-per-shape",
                "synthetic_weight_row_period": 4, "samples": samples,
            })
        );
    }
}
