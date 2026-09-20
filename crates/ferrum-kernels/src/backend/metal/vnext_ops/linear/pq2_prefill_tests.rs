//! Same-tile PQ2 format specialization with F32 inputs, operands and accumulation.

use super::*;
use half::f16;
use metal::{Buffer, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

mod prism_benchmark;
mod prism_reference;
mod vector_input_tests;
mod weight_thread_tests;

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
    Pq2CooperativeGemv,
    GenericM32,
    SpecializedM32,
    SpecializedM64,
    Production,
    ScalarWeightControlM64,
    WeightThreadsM64,
}

struct Fixture {
    params: LinearParams,
    input: Buffer,
    input_offset_bytes: u64,
    weight: Buffer,
    output: Buffer,
    input_values: Vec<f32>,
    weight_bytes: Vec<u8>,
    initial_output: Vec<f16>,
    // Each row has four different literal weight patterns, repeated across N.
    // This bounds the independent full-output oracle at realistic matrix sizes.
    reference: Vec<[(f64, f64); 4]>,
    column_reference: Option<Vec<Vec<(f64, f64)>>>,
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
            input_offset_bytes: (INPUT_PREFIX * 4) as u64,
            weight: buffer(device, &weight_bytes),
            output: buffer(device, &initial_output),
            input_values,
            weight_bytes,
            initial_output,
            reference,
            column_reference: None,
        }
    }

    fn use_distinct_weight_rows(&mut self, device: &Device) {
        let width = self.params.in_features as usize;
        let outputs = self.params.out_features as usize;
        let mut scales = vec![vec![0.0_f64; width / 128]; outputs];
        // Small layout fixtures must distinguish columns separated by 4 or 8,
        // which the periodic large-matrix oracle intentionally does not.
        for (column, row_scales) in scales.iter_mut().enumerate() {
            for (block, scale) in row_scales.iter_mut().enumerate() {
                let value = f16::from_f32((column + 1) as f32 * (block % 3 + 1) as f32 / 2048.0);
                *scale = f64::from(value.to_f32());
                let offset = WEIGHT_PREFIX + (column * (width / 128) + block) * 34;
                self.weight_bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
        self.column_reference = Some(
            (0..self.params.rows as usize)
                .map(|row| {
                    scales
                        .iter()
                        .enumerate()
                        .map(|(column, row_scales)| {
                            let mut sum = 0.0;
                            let mut absolute_sum = 0.0;
                            for k in 0..width {
                                let block = k / 128;
                                let byte = (k % 128) / 4;
                                let coefficient = COEFFICIENTS[(column + block + byte) % 4][k % 4];
                                let input =
                                    f64::from(self.input_values[INPUT_PREFIX + row * width + k]);
                                let product = input * row_scales[block] * coefficient;
                                sum += product;
                                absolute_sum += product.abs();
                            }
                            (sum, absolute_sum)
                        })
                        .collect()
                })
                .collect(),
        );
        self.weight = buffer(device, &self.weight_bytes);
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
            PrefillTile::Pq2CooperativeGemv => (
                &pipelines.native.pq2_linear_f32_f16,
                LinearDispatchKind::Pq2CooperativeGemv,
            ),
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
            PrefillTile::Production => pipelines.hadamard_native_dispatch(
                GgufBlockFormat::Pq2_0,
                ElementType::F16,
                self.params,
            ),
            PrefillTile::ScalarWeightControlM64 | PrefillTile::WeightThreadsM64 => {
                weight_thread_tests::pipeline(
                    pipelines,
                    self,
                    matches!(tile, PrefillTile::WeightThreadsM64),
                )
            }
        };
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.input), self.input_offset_bytes);
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
        self.validate_output_values(actual)
    }

    fn validate_output_values(&self, actual: &[f16]) -> Vec<u16> {
        assert_eq!(actual.len(), self.initial_output.len());
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
            let (expected, absolute_sum) = self
                .column_reference
                .as_ref()
                .map_or(self.reference[row][column % 4], |reference| {
                    reference[row][column]
                });
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
fn pq2_hadamard_prefill_selection_preserves_format_dtype_and_shape_boundaries() {
    for (rows, outputs, pq2_f16, other_f16) in [
        (0, 4096, false, false),
        (19, 4096, false, false),
        (20, 4095, false, false),
        (20, 4096, true, false),
        (31, 4095, false, false),
        (31, 4096, true, false),
        (32, 1023, false, false),
        (32, 1024, true, true),
        (32, 4095, true, true),
        (64, 4096, true, true),
    ] {
        let params = LinearParams {
            rows,
            in_features: 5120,
            out_features: outputs,
            output_stride: outputs,
            output_column_offset: 0,
        };
        for (format, expected_f16) in [
            (GgufBlockFormat::Pq2_0, pq2_f16),
            (GgufBlockFormat::Iq4Xs, other_f16),
        ] {
            assert_eq!(
                hadamard_tiled_gemm_supported(format, ElementType::F16, params),
                expected_f16,
                "{format:?}: rows={rows}, outputs={outputs}",
            );
            assert!(
                !hadamard_tiled_gemm_supported(format, ElementType::F32, params),
                "F32 output must retain its F32 GEMV ABI",
            );
        }
    }
}

#[test]
fn pq2_mixed_prefill_production_dispatch_and_m32_fallback_preserve_output() {
    let device = Device::system_default().expect("PQ2 production conformance requires Metal");
    let mut pipelines = MetalLinearPipelines::new(&device).unwrap();
    assert!(pipelines.native.pq2_gemm_input_f32_output_f16_m64.is_some());
    assert!(pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles
        .is_some());
    let queue = device.new_command_queue();
    for (rows, outputs, expected_dispatch) in [
        (19, 4096, LinearDispatchKind::Pq2CooperativeGemv),
        (20, 4095, LinearDispatchKind::Pq2CooperativeGemv),
        (20, 4096, LinearDispatchKind::NativeTiledGemm),
        (31, 4095, LinearDispatchKind::Pq2CooperativeGemv),
        (31, 4096, LinearDispatchKind::NativeTiledGemm),
        (32, 1023, LinearDispatchKind::CooperativeGemv),
        (32, 1024, LinearDispatchKind::NativeTiledGemm),
        (32, 4095, LinearDispatchKind::NativeTiledGemm),
        (64, 4096, LinearDispatchKind::NativeTiledGemmM64),
        (64, 4097, LinearDispatchKind::NativeTiledGemmM64),
        (128, 4096, LinearDispatchKind::NativeTiledGemmM64),
        (96, 4096, LinearDispatchKind::NativeTiledGemm),
        (65, 4096, LinearDispatchKind::NativeTiledGemm),
        (64, 1024, LinearDispatchKind::NativeTiledGemm),
        (64, 4095, LinearDispatchKind::NativeTiledGemm),
    ] {
        let fixture = Fixture::new(&device, rows, 384, outputs, false);
        let (pipeline, dispatch) = pipelines.hadamard_native_dispatch(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
        );
        assert_eq!(dispatch, expected_dispatch);
        if dispatch == LinearDispatchKind::NativeTiledGemmM64 {
            let expected = if outputs.is_multiple_of(64) {
                pipelines
                    .native
                    .pq2_gemm_input_f32_output_f16_m64_full_tiles
                    .as_ref()
            } else {
                pipelines.native.pq2_gemm_input_f32_output_f16_m64.as_ref()
            };
            assert!(std::ptr::eq(pipeline, expected.unwrap()));
        }
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::GenericM32, 1);
        let reference = fixture.validate();
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::Production, 1);
        let actual = fixture.validate();
        if matches!(
            expected_dispatch,
            LinearDispatchKind::NativeTiledGemm | LinearDispatchKind::NativeTiledGemmM64
        ) {
            assert_eq!(
                actual, reference,
                "same tiled reduction changed output bits"
            );
        }
        // GEMV uses a different reduction tree. Both results independently
        // satisfy the F64 oracle above; only a repeat of the same path must
        // preserve every output bit and guard.
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::Production, 1);
        assert_eq!(fixture.validate(), actual);
    }
    // Shape rejection is checked without executing invalid packed widths.
    for width in [0, 32, 127, 128, 129, 256, 384] {
        let params = LinearParams {
            rows: 64,
            in_features: width,
            out_features: 4096,
            output_stride: 4096,
            output_column_offset: 0,
        };
        let (pipeline, dispatch) =
            pipelines.hadamard_native_dispatch(GgufBlockFormat::Pq2_0, ElementType::F16, params);
        assert_eq!(dispatch, LinearDispatchKind::NativeTiledGemmM64);
        let expected = if width > 0 && width.is_multiple_of(128) {
            pipelines
                .native
                .pq2_gemm_input_f32_output_f16_m64_full_tiles
                .as_ref()
        } else {
            pipelines.native.pq2_gemm_input_f32_output_f16_m64.as_ref()
        };
        assert!(std::ptr::eq(pipeline, expected.unwrap()));
    }
    // Check the real outer selector, not only the tiled eligibility predicate:
    // neither F32 outputs nor another packed format gets the early M32 route.
    for rows in [19, 20, 31, 32, 64] {
        let params = LinearParams {
            rows,
            in_features: 5120,
            out_features: 4096,
            output_stride: 4096,
            output_column_offset: 0,
        };
        let (pipeline, dispatch) =
            pipelines.hadamard_native_dispatch(GgufBlockFormat::Pq2_0, ElementType::F32, params);
        if rows < 32 {
            assert_eq!(dispatch, LinearDispatchKind::Pq2CooperativeGemv);
            assert!(std::ptr::eq(
                pipeline,
                pipelines.native.pq2_linear_f32_complete.as_ref().unwrap()
            ));
        } else {
            assert_eq!(dispatch, LinearDispatchKind::CooperativeGemv);
            assert!(std::ptr::eq(
                pipeline,
                pipelines.native.linear_f32(GgufBlockFormat::Pq2_0),
            ));
        }
        let (pipeline, dispatch) =
            pipelines.hadamard_native_dispatch(GgufBlockFormat::Iq4Xs, ElementType::F16, params);
        if rows < 32 {
            assert_eq!(dispatch, LinearDispatchKind::CooperativeGemv);
            assert!(std::ptr::eq(
                pipeline,
                pipelines.native.linear_f32_f16(GgufBlockFormat::Iq4Xs),
            ));
        } else {
            assert_eq!(dispatch, LinearDispatchKind::NativeTiledGemm);
            assert!(std::ptr::eq(
                pipeline,
                &pipelines.native.gemm_input_f32_output_f16,
            ));
        }
    }
    // A missing full-tile PSO must still submit the original guarded M64.
    let full_tiles = pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles
        .take();
    let fixture = Fixture::new(&device, 64, 384, 4096, true);
    let (pipeline, dispatch) = pipelines.hadamard_native_dispatch(
        GgufBlockFormat::Pq2_0,
        ElementType::F16,
        fixture.params,
    );
    assert_eq!(dispatch, LinearDispatchKind::NativeTiledGemmM64);
    assert!(std::ptr::eq(
        pipeline,
        pipelines
            .native
            .pq2_gemm_input_f32_output_f16_m64
            .as_ref()
            .unwrap()
    ));
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::GenericM32, 1);
    let reference = fixture.validate();
    let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::Production, 1);
    assert_eq!(fixture.validate(), reference);
    pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles = full_tiles;
    // Exercise an actual M32 submission for the same M64-eligible shape when
    // device/PSO capabilities cannot supply the optional pipeline.
    pipelines.native.pq2_gemm_input_f32_output_f16_m64 = None;
    assert!(pipelines
        .native
        .pq2_gemm_input_f32_output_f16_m64_full_tiles
        .is_some());
    let fixture = Fixture::new(&device, 64, 384, 4096, true);
    let (pipeline, dispatch) = pipelines.hadamard_native_dispatch(
        GgufBlockFormat::Pq2_0,
        ElementType::F16,
        fixture.params,
    );
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
fn pq2_mixed_prefill_distinct_columns_match_literal_oracle_on_metal() {
    let device = Device::system_default().expect("PQ2 distinct-column conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    // Both M32/M64 row tails and N64 column tails must preserve columns that
    // differ beyond the four-column period used by the large timing fixtures.
    for (rows, outputs) in [(31, 65), (65, 1025)] {
        let mut fixture = Fixture::new(&device, rows, 384, outputs, false);
        fixture.use_distinct_weight_rows(&device);
        let _ = fixture.run_tile(&pipelines, &queue, PrefillTile::GenericM32, 1);
        let control = fixture.validate();
        for tile in [PrefillTile::SpecializedM32, PrefillTile::SpecializedM64] {
            let _ = fixture.run_tile(&pipelines, &queue, tile, 1);
            assert_eq!(fixture.validate(), control);
        }
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

fn crossover_numerics(fixture: &Fixture, gemv: &[u16], m32: &[u16]) -> serde_json::Value {
    assert_eq!(gemv.len(), fixture.initial_output.len());
    assert_eq!(m32.len(), fixture.initial_output.len());
    let mut max_abs_difference = 0.0_f64;
    let mut max_gemv_reference_error = 0.0_f64;
    let mut max_m32_reference_error = 0.0_f64;
    let mut different_output_bits = 0_usize;
    for row in 0..fixture.params.rows as usize {
        for column in 0..fixture.params.out_features as usize {
            let index = OUTPUT_PREFIX
                + row * fixture.params.output_stride as usize
                + fixture.params.output_column_offset as usize
                + column;
            let gemv_value = f64::from(f16::from_bits(gemv[index]).to_f32());
            let m32_value = f64::from(f16::from_bits(m32[index]).to_f32());
            let reference = fixture.reference[row][column % 4].0;
            max_abs_difference = max_abs_difference.max((gemv_value - m32_value).abs());
            max_gemv_reference_error = max_gemv_reference_error.max((gemv_value - reference).abs());
            max_m32_reference_error = max_m32_reference_error.max((m32_value - reference).abs());
            different_output_bits += usize::from(gemv[index] != m32[index]);
        }
    }
    serde_json::json!({
        "output_elements": fixture.params.rows as u64 * fixture.params.out_features as u64,
        "different_output_bits": different_output_bits,
        "max_abs_gemv_m32_difference": max_abs_difference,
        "max_abs_gemv_f64_reference_error": max_gemv_reference_error,
        "max_abs_m32_f64_reference_error": max_m32_reference_error,
        "both_within_existing_forward_error_budget": true,
        "all_outputs_finite": true,
        "guards_and_immutable_inputs_verified": true,
    })
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_small_row_gemv_m32_crossover_gpu_microbench() {
    // GDN QKV has 2 * 16 * 128 + 48 * 128 = 10240 outputs in the
    // pinned 27B geometry. Its separate Z projection has 6144 outputs.
    const ROWS: &[u32] = &[1, 2, 4, 8, 12, 16, 24, 31, 32];
    measure_small_row_crossover(&[
        ("ffn_gate_or_up", 5120, 17408, ROWS),
        ("ffn_down", 17408, 5120, ROWS),
        ("gdn_qkv", 5120, 10240, ROWS),
    ]);
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_small_row_projection_boundary_gpu_microbench() {
    measure_small_row_crossover(&[
        ("gdn_z", 5120, 6144, &[16, 20, 24]),
        ("gdn_output", 6144, 5120, &[16, 20, 24]),
        ("wide_output_boundary", 5120, 4096, &[16, 20, 24]),
        ("attention_k_or_v", 5120, 1024, &[16, 20, 32]),
    ]);
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_mixed_projection_crossover_boundary_gpu_microbench() {
    measure_small_row_crossover(&[
        ("small_n1024", 5120, 1024, &[32, 64, 128]),
        ("small_n2048", 5120, 2048, &[32, 64, 128]),
        ("wide_n4096", 5120, 4096, &[32, 64, 128]),
        ("ffn_gate_or_up", 5120, 17408, &[20]),
        ("ffn_down", 17408, 5120, &[20]),
        ("gdn_qkv", 5120, 10240, &[20]),
    ]);
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn pq2_small_width_intermediate_rows_gpu_microbench() {
    measure_small_row_crossover(&[
        ("small_n1024", 5120, 1024, &[40, 48, 56, 63]),
        ("small_n2048", 5120, 2048, &[40, 48, 56, 63]),
    ]);
}

fn measure_small_row_crossover(shapes: &[(&str, u32, u32, &[u32])]) {
    let device = Device::system_default().expect("PQ2 small-row crossover requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    const TILES: [PrefillTile; 2] = [PrefillTile::Pq2CooperativeGemv, PrefillTile::SpecializedM32];
    const DISPATCHES: usize = 8;
    const WARMUP_PAIRS: usize = 2;
    const PAIRS: usize = 8;
    for &(name, width, outputs, rows) in shapes {
        for &rows in rows {
            let fixture = Fixture::new(&device, rows, width, outputs, false);
            let mut references = Vec::with_capacity(TILES.len());
            for tile in TILES {
                let _ = fixture.run_tile(&pipelines, &queue, tile, 1);
                // Each reduction order must meet the same independent F64
                // forward bound and final F16 rounding budget. Do not require
                // bitwise equality between the two different reductions.
                references.push(fixture.validate());
            }
            let numerics = crossover_numerics(&fixture, &references[0], &references[1]);
            for pair in 0..WARMUP_PAIRS {
                let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
                for index in order {
                    let _ = fixture.run_tile(&pipelines, &queue, TILES[index], DISPATCHES);
                    assert_eq!(fixture.validate(), references[index]);
                }
            }
            let mut samples = Vec::new();
            for pair in 0..PAIRS {
                let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
                let mut times = [0.0_f64; 2];
                for index in order {
                    times[index] = fixture
                        .run_tile(&pipelines, &queue, TILES[index], DISPATCHES)
                        .expect("completed GPU command must expose valid GPU timestamps");
                    assert_eq!(fixture.validate(), references[index]);
                }
                samples.push(serde_json::json!({
                    "pair": pair, "m32_first": order[0] == 1,
                    "pq2_gemv_gpu_ns_per_dispatch": times[0],
                    "specialized_m32_gpu_ns_per_dispatch": times[1],
                    "m32_over_gemv": times[1] / times[0],
                }));
            }
            println!(
                "{}",
                serde_json::json!({
                    "schema_version": 1,
                    "kind": "pq2_small_row_gemv_m32_crossover_gpu_microbench",
                    "device": device.name(), "shape": name,
                    "rows": rows, "in_features": width, "out_features": outputs,
                    "weight_format": "quantization.gguf.pq2-0",
                    "input_dtype": "f32", "operand_dtype": "f32", "accumulator_dtype": "f32",
                    "output_dtype": "f16", "m32_tile_rows": 32,
                    "variant_names": ["production_pq2_cooperative_gemv", "specialized_m32"],
                    "dispatches_per_command": DISPATCHES, "warmup_pairs": WARMUP_PAIRS,
                    "independent_oracle": "literal-pq2-f64",
                    "error_budget": "existing-f32-forward-bound-plus-final-f16-rounding",
                    "working_set": "one-resident-matrix-per-shape",
                    "synthetic_weight_row_period": 4, "numerics": numerics, "samples": samples,
                })
            );
        }
    }
}
