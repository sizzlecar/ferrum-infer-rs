//! Complete synthetic 9B-geometry FFN experiment; production routing is intact.
//! Run ignored in release mode with exclusive Metal access. CPU reference work
//! covers every coefficient/output and is outside all command timing intervals.
use super::*;
use crate::backend::metal::vnext_ops::numerical_tolerance;

pub(super) mod oracle;
pub(super) mod oracle64;

const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 12288;
const PACKED: usize = 2 * INTERMEDIATE;
const PREFIX: usize = 16;
const GUARD: usize = 16;
const WEIGHT_PREFIX: usize = 16;
const WEIGHT_GUARD: u8 = 0xa7;
const OUTPUT_STRIDE: usize = HIDDEN + 16;
const OUTPUT_COLUMN: usize = 4;
const LAYERS: usize = 4;
const FFNS_PER_COMMAND: usize = 4;
const FULL_FFN_TOLERANCE: &str =
    "runtime-vnext.metal.dense-swiglu.v1.operation.fp16.gguf-q4-k-q6-k.full-pipeline";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Production,
    CooperativeCandidate,
}
#[derive(Clone, Copy)]
enum Role {
    General,
    GateUp,
    Down,
}

#[derive(Clone, Copy)]
struct Qualification {
    role: Role,
    original_rows: u32,
    params: LinearParams,
    format: LinearPhysicalFormat,
    activation_type: ElementType,
    transformed: bool,
    two_part_layout: bool,
}
impl Qualification {
    fn eligible(self) -> bool {
        matches!(self.role, Role::GateUp)
            && self.original_rows == 4
            && self.params.rows == self.original_rows
            && self.params.in_features == HIDDEN as u32
            && self.params.out_features == INTERMEDIATE as u32
            && self.params.output_stride == PACKED as u32
            && matches!(self.params.output_column_offset, 0 | 12288)
            && self.format == LinearPhysicalFormat::Q4K
            && self.activation_type == ElementType::F16
            && !self.transformed
            && self.two_part_layout
    }
}

fn gate_shape() -> Shape {
    Shape {
        name: "gate_or_up",
        input: HIDDEN as u32,
        output: INTERMEDIATE as u32,
        format: GgufBlockFormat::Q4K,
    }
}
fn down_shape(format: GgufBlockFormat) -> Shape {
    Shape {
        name: "down",
        input: INTERMEDIATE as u32,
        output: HIDDEN as u32,
        format,
    }
}
fn gate_params(rows: u32, column: u32) -> LinearParams {
    LinearParams {
        rows,
        in_features: HIDDEN as u32,
        out_features: INTERMEDIATE as u32,
        output_stride: PACKED as u32,
        output_column_offset: column,
    }
}
fn gate_qualification(rows: u32, column: u32) -> Qualification {
    Qualification {
        role: Role::GateUp,
        original_rows: rows,
        params: gate_params(rows, column),
        format: LinearPhysicalFormat::Q4K,
        activation_type: ElementType::F16,
        transformed: false,
        two_part_layout: true,
    }
}

#[test]
fn q4_b4_ffn_candidate_requires_whole_wave_role_format_and_packed_geometry() {
    for column in [0, INTERMEDIATE as u32] {
        let base = gate_qualification(4, column);
        assert!(base.eligible());
        assert!(
            !Qualification {
                original_rows: 8,
                ..base
            }
            .eligible(),
            "an M8 wave split into M4 must stay on production"
        );
        assert!(!Qualification {
            role: Role::General,
            ..base
        }
        .eligible());
        assert!(!Qualification {
            role: Role::Down,
            ..base
        }
        .eligible());
        assert!(!Qualification {
            format: LinearPhysicalFormat::Q6K,
            ..base
        }
        .eligible());
        assert!(!Qualification {
            activation_type: ElementType::F32,
            ..base
        }
        .eligible());
        assert!(!Qualification {
            transformed: true,
            ..base
        }
        .eligible());
        assert!(!Qualification {
            two_part_layout: false,
            ..base
        }
        .eligible());
        for params in [
            LinearParams {
                in_features: 256,
                ..base.params
            },
            LinearParams {
                out_features: PACKED as u32,
                ..base.params
            },
            LinearParams {
                output_stride: PACKED as u32 + 1,
                ..base.params
            },
            LinearParams {
                output_column_offset: 1,
                ..base.params
            },
        ] {
            assert!(!Qualification { params, ..base }.eligible());
        }
    }
    for rows in [1, 2, 3, 5, 8] {
        assert!(!gate_qualification(rows, 0).eligible());
    }
}

#[test]
fn q4_shared_weight_dense_batched_matches_full_projection_oracle() {
    let device = Device::system_default().expect("dense Q4 shared regression requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    // Retain the failing experiment's dense input and coefficient sequence,
    // with a smaller output span that still exercises production selection
    // and the final incomplete group. Both independent output rows must agree
    // with the oracle for every activation row, including the B4 first row.
    let shape = Shape {
        name: "dense_q4_shared",
        input: HIDDEN as u32,
        output: 1025,
        format: GgufBlockFormat::Q4K,
    };
    let input_values = oracle::input();
    let input = buffer(&device, &input_values);
    for seed in [2, 5] {
        let encoded = oracle::matrix(shape, seed);
        let expected = oracle::project(shape, &encoded, &input_values);
        let weights = [buffer(&device, &encoded)];
        for rows in [2_u32, 3, 4] {
            let logical = rows as usize * shape.output as usize;
            let output = buffer(&device, &vec![f16::from_f32(OUTPUT_GUARD); logical + GUARD]);
            let case = super::Case {
                queue: &queue,
                pipelines: &pipelines,
                shape,
                rows,
                activation_type: ElementType::F16,
                input: &input,
                weights: &weights,
                output: &output,
            };
            case.poison_output();
            // This existing harness variant selects and asserts the actual
            // production SharedWeightGemv route; it does not supply a test PSO.
            case.run(Dispatch::NativeSharedGemv, 1);
            let expected: Vec<_> = expected[..logical].iter().map(|v| v.to_f32()).collect();
            case.validate(&expected);
        }
    }
}

struct Encoded {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}
struct Layer {
    gate: Buffer,
    up: Buffer,
    down: Buffer,
}
impl Layer {
    fn new(device: &Device, encoded: &Encoded) -> Self {
        fn guarded(device: &Device, bytes: &[u8]) -> Buffer {
            let mut values = vec![WEIGHT_GUARD; WEIGHT_PREFIX];
            values.extend_from_slice(bytes);
            values.extend([WEIGHT_GUARD; GUARD]);
            buffer(device, &values)
        }
        Self {
            gate: guarded(device, &encoded.gate),
            up: guarded(device, &encoded.up),
            down: guarded(device, &encoded.down),
        }
    }
    fn validate(&self, expected: &Encoded) -> (bool, serde_json::Value) {
        let mut report = serde_json::Map::new();
        let mut passed = true;
        for (name, buffer, bytes) in [
            ("gate", &self.gate, &expected.gate),
            ("up", &self.up, &expected.up),
            ("down", &self.down, &expected.down),
        ] {
            // SAFETY: shared byte allocation; all callers wait for completion.
            let actual = unsafe {
                std::slice::from_raw_parts(buffer.contents().cast::<u8>(), buffer.length() as usize)
            };
            let prefix_ok = actual[..WEIGHT_PREFIX]
                .iter()
                .all(|&byte| byte == WEIGHT_GUARD);
            let immutable = &actual[WEIGHT_PREFIX..WEIGHT_PREFIX + bytes.len()] == bytes.as_slice();
            let suffix_ok = actual[WEIGHT_PREFIX + bytes.len()..]
                .iter()
                .all(|&byte| byte == WEIGHT_GUARD);
            passed &= prefix_ok && immutable && suffix_ok;
            report.insert(
                name.to_owned(),
                serde_json::json!({"prefix_ok":prefix_ok,
                "immutable":immutable,"suffix_ok":suffix_ok}),
            );
        }
        (passed, report.into())
    }
}

struct Validation {
    values: Vec<f16>,
    passed: bool,
    report: serde_json::Value,
}

struct Case<'a> {
    queue: &'a CommandQueueRef,
    pipelines: &'a MetalLinearPipelines,
    layers: &'a [Layer],
    rows: usize,
    down_format: GgufBlockFormat,
    input: Buffer,
    input_storage: Vec<f16>,
    scratch: Buffer,
    output: Buffer,
}
impl Case<'_> {
    fn activation_start(&self) -> usize {
        PREFIX + self.rows * PACKED + GUARD
    }
    fn reset(&self) {
        // SAFETY: shared allocations, no command is in flight at call sites.
        unsafe {
            let scratch = std::slice::from_raw_parts_mut(
                self.scratch.contents().cast::<f16>(),
                self.scratch.length() as usize / 2,
            );
            scratch.fill(f16::from_f32(OUTPUT_GUARD));
            scratch[PREFIX..PREFIX + self.rows * PACKED].fill(f16::NAN);
            scratch[self.activation_start()..self.activation_start() + self.rows * INTERMEDIATE]
                .fill(f16::NAN);
            let output = std::slice::from_raw_parts_mut(
                self.output.contents().cast::<f16>(),
                self.output.length() as usize / 2,
            );
            output.fill(f16::from_f32(OUTPUT_GUARD));
            for row in 0..self.rows {
                let start = PREFIX + row * OUTPUT_STRIDE + OUTPUT_COLUMN;
                output[start..start + HIDDEN].fill(f16::NAN);
            }
        }
    }

    fn run(&self, arm: Arm, start_layer: usize, ffns: usize) -> serde_json::Value {
        let started = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let mut gate_kind = LinearDispatchKind::SharedWeightGemv;
        let down_params = LinearParams {
            rows: self.rows as u32,
            in_features: INTERMEDIATE as u32,
            out_features: HIDDEN as u32,
            output_stride: OUTPUT_STRIDE as u32,
            output_column_offset: OUTPUT_COLUMN as u32,
        };
        // This exact production selector is shared by both experiment arms.
        let (down_pipeline, down_kind) = self.pipelines.plain_linear_dispatch(
            physical(self.down_format),
            ElementType::F16,
            down_params,
        );
        for index in 0..ffns {
            let layer = &self.layers[(start_layer + index) % self.layers.len()];
            for (weight, column) in [(&layer.gate, 0), (&layer.up, INTERMEDIATE as u32)] {
                let qualification = gate_qualification(self.rows as u32, column);
                let (pipeline, kind) =
                    if arm == Arm::CooperativeCandidate && qualification.eligible() {
                        (
                            &self.pipelines.q4_k_gemv,
                            LinearDispatchKind::CooperativeGemv,
                        )
                    } else {
                        self.pipelines.plain_linear_dispatch(
                            LinearPhysicalFormat::Q4K,
                            ElementType::F16,
                            qualification.params,
                        )
                    };
                gate_kind = kind;
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(&self.input), (PREFIX * 2) as u64);
                encoder.set_buffer(1, Some(weight), WEIGHT_PREFIX as u64);
                encoder.set_buffer(2, Some(&self.scratch), (PREFIX * 2) as u64);
                bind_linear_params(
                    encoder,
                    qualification.params,
                    LinearPhysicalFormat::Q4K,
                    ElementType::F16,
                );
                dispatch_linear_grid(encoder, qualification.params, kind);
            }
            // Same production activation kernel and parameters. Scratch uses
            // real packed rows, with a guarded gap before the activation span.
            encoder.set_compute_pipeline_state(&self.pipelines.swiglu);
            encoder.set_buffer(0, Some(&self.scratch), (PREFIX * 2) as u64);
            encoder.set_buffer(1, Some(&self.scratch), (self.activation_start() * 2) as u64);
            let activation = SwiGluParams {
                rows: self.rows as u32,
                intermediate_size: INTERMEDIATE as u32,
                gate_up_stride: PACKED as u32,
            };
            encoder.set_bytes(
                2,
                std::mem::size_of::<SwiGluParams>() as u64,
                &activation as *const _ as *const c_void,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    (self.rows as u64 * INTERMEDIATE as u64).div_ceil(THREADS_PER_GROUP),
                    1,
                    1,
                ),
                MTLSize::new(THREADS_PER_GROUP, 1, 1),
            );
            encoder.set_compute_pipeline_state(down_pipeline);
            encoder.set_buffer(0, Some(&self.scratch), (self.activation_start() * 2) as u64);
            encoder.set_buffer(1, Some(&layer.down), WEIGHT_PREFIX as u64);
            encoder.set_buffer(2, Some(&self.output), (PREFIX * 2) as u64);
            bind_linear_params(
                encoder,
                down_params,
                physical(self.down_format),
                ElementType::F16,
            );
            dispatch_linear_grid(encoder, down_params, down_kind);
        }
        encoder.end_encoding();
        let host_encode_ns = started.elapsed().as_nanos() as u64;
        let submitted = Instant::now();
        command.commit();
        command.wait_until_completed();
        let host_submit_wait_ns = submitted.elapsed().as_nanos() as u64;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(
            gate_kind,
            if arm == Arm::CooperativeCandidate && self.rows == 4 {
                LinearDispatchKind::CooperativeGemv
            } else {
                LinearDispatchKind::SharedWeightGemv
            }
        );
        serde_json::json!({"arm":format!("{arm:?}"), "gate_up_dispatch":format!("{gate_kind:?}"), "down_dispatch":format!("{down_kind:?}"), "ffns":ffns,
            "compute_dispatches":ffns * 4, "start_layer":start_layer,
            "host_encode_ns":host_encode_ns, "host_submit_wait_ns":host_submit_wait_ns,
            "device_command_ns":gpu_elapsed_ns(command)})
    }

    fn validate(
        &self,
        reference: &oracle::GateUp,
        expected: &[f16],
        encoded: &Encoded,
        reference_f64: &oracle::GateUp,
        expected_f64: &[f16],
    ) -> Validation {
        // SAFETY: complete shared buffers, after command completion.
        let scratch = unsafe {
            std::slice::from_raw_parts(
                self.scratch.contents().cast::<f16>(),
                self.scratch.length() as usize / 2,
            )
        };
        let output = unsafe {
            std::slice::from_raw_parts(
                self.output.contents().cast::<f16>(),
                self.output.length() as usize / 2,
            )
        };
        let input = unsafe {
            std::slice::from_raw_parts(
                self.input.contents().cast::<f16>(),
                self.input.length() as usize / 2,
            )
        };
        let input_immutable = input == self.input_storage.as_slice();
        let projection_range = PREFIX..PREFIX + self.rows * PACKED;
        let activation_range =
            self.activation_start()..self.activation_start() + self.rows * INTERMEDIATE;
        let scratch_guard_failures = scratch
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                !projection_range.contains(index)
                    && !activation_range.contains(index)
                    && value.to_bits() != f16::from_f32(OUTPUT_GUARD).to_bits()
            })
            .count();
        let projected = &scratch[projection_range.clone()];
        let activated = &scratch[activation_range.clone()];
        let mut stages = serde_json::Map::new();
        let mut passed = input_immutable && scratch_guard_failures == 0;
        for (name, column_offset, weights) in [
            ("gate", 0, &encoded.gate),
            ("up", INTERMEDIATE, &encoded.up),
        ] {
            let extract = |values: &[f16]| -> Vec<f16> {
                (0..self.rows)
                    .flat_map(|row| {
                        values[row * PACKED + column_offset
                            ..row * PACKED + column_offset + INTERMEDIATE]
                            .iter()
                            .copied()
                    })
                    .collect()
            };
            let actual = extract(projected);
            let expected = extract(&reference.projected);
            let mut metrics = oracle::metrics(&actual, &expected, INTERMEDIATE);
            passed &= metrics["linear_bound_violations"].as_u64() == Some(0);
            let scalar: Vec<_> = oracle::diagnostic_indices(&metrics).into_iter().map(|index| {
                let row = index / INTERMEDIATE;
                let column = index % INTERMEDIATE;
                let sum = oracle::scalar_dot(gate_shape(), weights,
                    &self.input_storage[PREFIX + row * HIDDEN..PREFIX + (row + 1) * HIDDEN], column);
                serde_json::json!({"row":row,"column":column,"f64_dot":sum,
                    "f64_dot_fp16":f16::from_f64(sum).to_f32(),"block_f32_oracle":expected[index].to_f32(),
                    "actual":actual[index].to_f32()})
            }).collect();
            metrics["scalar_coefficient_decode_f64_checks"] = scalar.into();
            stages.insert(name.to_owned(), metrics);
        }
        let mut activation_metrics = oracle::metrics(
            activated,
            &reference.activated[..self.rows * INTERMEDIATE],
            INTERMEDIATE,
        );
        passed &= activation_metrics["linear_bound_violations"].as_u64() == Some(0);
        let activation_scalar: Vec<_> = oracle::diagnostic_indices(&activation_metrics).into_iter().map(|index| {
            let row = index / INTERMEDIATE;
            let column = index % INTERMEDIATE;
            let gate = row * PACKED + column;
            let up = gate + INTERMEDIATE;
            serde_json::json!({"row":row,"column":column,
                "from_reference_gate_up_f64":oracle::scalar_swiglu(reference.projected[gate],reference.projected[up]),
                "from_actual_gate_up_f64":oracle::scalar_swiglu(projected[gate],projected[up]),
                "reference":reference.activated[index].to_f32(),"actual":activated[index].to_f32()})
        }).collect();
        activation_metrics["scalar_f64_checks"] = activation_scalar.into();
        stages.insert("activation".to_owned(), activation_metrics);
        let mut dense = Vec::with_capacity(self.rows * HIDDEN);
        let mut output_guard_failures = 0;
        for (index, &value) in output.iter().enumerate() {
            let logical = index
                .checked_sub(PREFIX)
                .filter(|relative| *relative < self.rows * OUTPUT_STRIDE)
                .is_some_and(|relative| {
                    (OUTPUT_COLUMN..OUTPUT_COLUMN + HIDDEN).contains(&(relative % OUTPUT_STRIDE))
                });
            if logical {
                dense.push(value);
            } else {
                output_guard_failures +=
                    usize::from(value.to_bits() != f16::from_f32(OUTPUT_GUARD).to_bits());
            }
        }
        passed &= output_guard_failures == 0;
        let mut down_metrics = oracle::metrics(&dense, &expected[..self.rows * HIDDEN], HIDDEN);
        let down_scalar: Vec<_> = oracle::diagnostic_indices(&down_metrics).into_iter().map(|index| {
            let row = index / HIDDEN;
            let column = index % HIDDEN;
            let span = row * INTERMEDIATE..(row + 1) * INTERMEDIATE;
            let reference_sum = oracle::scalar_dot(down_shape(self.down_format), &encoded.down, &reference.activated[span.clone()], column);
            let actual_input_sum = oracle::scalar_dot(down_shape(self.down_format), &encoded.down, &activated[span], column);
            serde_json::json!({"row":row,"column":column,"from_reference_activation_f64":reference_sum,
                "from_reference_activation_fp16":f16::from_f64(reference_sum).to_f32(),
                "from_actual_activation_f64":actual_input_sum,"from_actual_activation_fp16":f16::from_f64(actual_input_sum).to_f32(),
                "block_f32_oracle":expected[index].to_f32(),"actual":dense[index].to_f32()})
        }).collect();
        down_metrics["scalar_coefficient_decode_f64_checks"] = down_scalar.into();
        stages.insert("down".to_owned(), down_metrics);
        let mut parallel_f64 = oracle64::report(
            self.rows,
            projected,
            activated,
            &dense,
            reference_f64,
            expected_f64,
            reference,
            expected,
        );
        let layout_passed =
            input_immutable && scratch_guard_failures == 0 && output_guard_failures == 0;
        parallel_f64["layout_passed"] = layout_passed.into();
        parallel_f64["passed"] =
            (parallel_f64["passed"].as_bool().unwrap() && layout_passed).into();
        let actual: Vec<_> = dense.iter().map(|value| value.to_f32()).collect();
        let expected: Vec<_> = expected[..self.rows * HIDDEN]
            .iter()
            .map(|value| value.to_f32())
            .collect();
        let full_pipeline = numerical_tolerance::assert_matches(
            "complete FFN dense CPU quantized oracle (unchanged full-pipeline bounds)",
            &actual,
            &[self.rows, HIDDEN],
            &expected,
            &[self.rows, HIDDEN],
            numerical_tolerance::LogicalDtype::Fp16,
            FULL_FFN_TOLERANCE,
        );
        passed &= full_pipeline.is_ok();
        let mut all = scratch[PREFIX..PREFIX + self.rows * PACKED].to_vec();
        all.extend_from_slice(
            &scratch[self.activation_start()..self.activation_start() + self.rows * INTERMEDIATE],
        );
        all.extend(dense);
        Validation {
            values: all,
            passed,
            report: serde_json::json!({"passed":passed,"stages":stages,
                "input_immutable":input_immutable,"scratch_guard_failures":scratch_guard_failures,
                "output_guard_failures":output_guard_failures,"full_pipeline_tolerance":FULL_FFN_TOLERANCE,
                "full_pipeline_error":full_pipeline.err(),"parallel_f64":parallel_f64}),
        }
    }
}

#[test]
#[ignore = "complete 9B-geometry dense FFN oracle and paired timing; release mode, exclusive Metal access"]
fn q4_b4_complete_ffn_cooperative_microbench() {
    let device = Device::system_default().expect("complete FFN experiment requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let input = oracle::input();
    assert!(input
        .iter()
        .all(|value| value.is_finite() && *value != f16::ZERO));
    let gate = oracle::matrix(gate_shape(), 2);
    let up = oracle::matrix(gate_shape(), 5);
    println!(
        "{}",
        serde_json::json!({"kind":"complete_ffn_experiment_plan", "device":device.name(),
        "hidden":HIDDEN, "intermediate":INTERMEDIATE, "rows":[4,3], "down_formats":["Q4K","Q6K"],
        "layer_working_sets":[1,LAYERS], "warmup_rounds":WARMUP_ROUNDS, "paired_rounds":MEASURED_ROUNDS,
        "ffns_per_command":FFNS_PER_COMMAND, "cpu_oracle":"dense_all_coefficients_block_decode_fp32_with_fp16_writebacks", "decoded_weight_workspace_bytes":256 * 4,
        "total_dense_oracle_mac_count":4_u64 * 4 * HIDDEN as u64 * INTERMEDIATE as u64,
        "parallel_oracle":"direct_encoded_coefficient_fp64_dot_silu_and_fp16_writebacks",
        "parallel_decoded_weight_workspace_bytes":256 * 8,
        "parallel_dense_oracle_mac_count":4_u64 * 4 * HIDDEN as u64 * INTERMEDIATE as u64,
        "performance_gate":"unchanged_original_fp32_oracle",
        "scope":"synthetic_complete_swiglu_not_provider_admission_or_9b_model_quality"})
    );
    let oracle_started = Instant::now();
    let reference = oracle::gate_up(&gate, &up, &input);
    let gate_oracle_ns = oracle_started.elapsed().as_nanos() as u64;
    let oracle_started = Instant::now();
    let reference_f64 = oracle64::gate_up(&gate, &up, &input);
    let gate_oracle_f64_ns = oracle_started.elapsed().as_nanos() as u64;
    let down_references: Vec<_> = [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K]
        .into_iter()
        .map(|format| {
            let down = oracle::matrix(down_shape(format), 11);
            let oracle_started = Instant::now();
            let expected = oracle::project(down_shape(format), &down, &reference.activated);
            assert!(expected.iter().any(|value| value.to_f32().abs() > 1.0e-5));
            let down_oracle_ns = oracle_started.elapsed().as_nanos() as u64;
            let oracle_started = Instant::now();
            let expected_f64 =
                oracle64::project(down_shape(format), &down, &reference_f64.activated);
            let down_oracle_f64_ns = oracle_started.elapsed().as_nanos() as u64;
            (
                format,
                down,
                expected,
                down_oracle_ns,
                expected_f64,
                down_oracle_f64_ns,
            )
        })
        .collect();
    let mut failures = 0_usize;
    let mut checked_commands = 0_usize;
    let mut parallel_f64_failures = 0_usize;
    // Complete the entire conformance matrix before permitting any warmup or
    // timing loop. A failing early cell must not hide the other arm or device
    // error pattern, and later failures must not leave earlier timing samples.
    for measure in [false, true] {
        if measure {
            println!(
                "{}",
                serde_json::json!({"kind":"complete_ffn_conformance_summary",
                "checked_commands":checked_commands,"failures":failures,
                "parallel_f64_failures":parallel_f64_failures,
                "performance_loops_permitted":failures == 0})
            );
            assert_eq!(
                failures, 0,
                "complete FFN conformance failed; every performance loop skipped"
            );
        }
        for (format, down, expected, down_oracle_ns, expected_f64, down_oracle_f64_ns) in
            &down_references
        {
            let format = *format;
            let encoded = Encoded {
                gate: gate.clone(),
                up: up.clone(),
                down: down.clone(),
            };
            let weight_bytes = encoded.gate.len() + encoded.up.len() + encoded.down.len();
            let weight_hashes = serde_json::json!({
                "gate":format!("{:x}", Sha256::digest(&encoded.gate)),
                "up":format!("{:x}", Sha256::digest(&encoded.up)),
                "down":format!("{:x}", Sha256::digest(&encoded.down)),
            });
            let layers: Vec<_> = (0..LAYERS).map(|_| Layer::new(&device, &encoded)).collect();
            let allocated_weight_bytes: u64 = layers
                .iter()
                .map(|layer| layer.gate.length() + layer.up.length() + layer.down.length())
                .sum();
            for rows in [4, 3] {
                let mut input_storage = vec![f16::from_f32(OUTPUT_GUARD); PREFIX];
                input_storage.extend_from_slice(&input[..rows * HIDDEN]);
                input_storage.extend([f16::from_f32(OUTPUT_GUARD); GUARD]);
                for layer_count in [1, LAYERS] {
                    let case = Case {
                        queue: &queue,
                        pipelines: &pipelines,
                        layers: &layers[..layer_count],
                        rows,
                        down_format: format,
                        input: buffer(&device, &input_storage),
                        input_storage: input_storage.clone(),
                        scratch: buffer(
                            &device,
                            &vec![
                                f16::from_f32(OUTPUT_GUARD);
                                PREFIX + rows * PACKED + GUARD + rows * INTERMEDIATE + GUARD
                            ],
                        ),
                        output: buffer(
                            &device,
                            &vec![
                                f16::from_f32(OUTPUT_GUARD);
                                PREFIX + rows * OUTPUT_STRIDE + GUARD
                            ],
                        ),
                    };
                    let mut conformance = Vec::new();
                    if !measure {
                        for layer in 0..layer_count {
                            let mut pair = Vec::new();
                            for arm in [Arm::Production, Arm::CooperativeCandidate] {
                                case.reset();
                                let sample = case.run(arm, layer, 1);
                                let validation = case.validate(
                                    &reference,
                                    expected,
                                    &encoded,
                                    &reference_f64,
                                    expected_f64,
                                );
                                let (weights_ok, weights_report) = layers[layer].validate(&encoded);
                                failures += usize::from(!validation.passed || !weights_ok);
                                parallel_f64_failures += usize::from(
                                    validation.report["parallel_f64"]["passed"].as_bool()
                                        != Some(true)
                                        || !weights_ok,
                                );
                                checked_commands += 1;
                                println!(
                                    "{}",
                                    serde_json::json!({"kind":"complete_ffn_stage_diagnostics",
                            "device":device.name(),"rows":rows,"down_format":format.format_id(),
                            "active_layer_working_set":layer_count,"layer":layer,"arm":format!("{arm:?}"),
                            "numerics":validation.report,"weights":weights_report,"command":sample})
                                );
                                pair.push(validation.values);
                                conformance.push(sample);
                            }
                            let differences = pair[0]
                                .iter()
                                .zip(&pair[1])
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            if rows == 3 {
                                failures += usize::from(differences != 0);
                            }
                            println!(
                                "{}",
                                serde_json::json!({"kind":"complete_ffn_pair_numerics", "rows":rows, "down_format":format.format_id(), "layer":layer, "compared_gate_up_activation_output_values":pair[0].len(), "bitwise_differences":differences})
                            );
                        }
                    }
                    let mut samples = Vec::new();
                    if measure {
                        case.reset();
                        for round in 0..WARMUP_ROUNDS + MEASURED_ROUNDS {
                            let order = if round % 2 == 0 {
                                [Arm::Production, Arm::CooperativeCandidate]
                            } else {
                                [Arm::CooperativeCandidate, Arm::Production]
                            };
                            for (order_index, arm) in order.into_iter().enumerate() {
                                let mut sample =
                                    case.run(arm, round % layer_count, FFNS_PER_COMMAND);
                                sample["round"] = round.into();
                                sample["order_index"] = order_index.into();
                                sample["warmup"] = (round < WARMUP_ROUNDS).into();
                                samples.push(sample);
                            }
                        }
                        let validation = case.validate(
                            &reference,
                            expected,
                            &encoded,
                            &reference_f64,
                            expected_f64,
                        );
                        println!(
                            "{}",
                            serde_json::json!({"kind":"complete_ffn_post_timing_diagnostics",
                    "rows":rows,"down_format":format.format_id(),"active_layer_working_set":layer_count,
                    "numerics":validation.report})
                        );
                        assert!(
                            validation.passed,
                            "post-timing complete FFN conformance failed"
                        );
                        for layer in &layers {
                            assert!(
                                layer.validate(&encoded).0,
                                "post-timing immutable weights or guards changed"
                            );
                        }
                    }
                    println!(
                        "{}",
                        serde_json::json!({"kind":"complete_ffn_cooperative_microbench", "device":device.name(), "rows":rows, "down_format":format.format_id(),
                    "gate_up_format":"Q4K", "gate_up_stride":PACKED, "gate_up_columns":[0,INTERMEDIATE],
                    "input_prefix_elements":PREFIX, "weight_prefix_bytes":WEIGHT_PREFIX, "output_stride":OUTPUT_STRIDE, "output_column":OUTPUT_COLUMN,
                    "gate_oracle_ns":gate_oracle_ns, "down_oracle_ns":down_oracle_ns, "weight_bytes_per_layer":weight_bytes,
                    "gate_oracle_f64_ns":gate_oracle_f64_ns,"down_oracle_f64_ns":down_oracle_f64_ns,
                    "active_layer_working_set":layer_count, "active_weight_payload_bytes":weight_bytes * layer_count, "allocated_weight_buffer_bytes":allocated_weight_bytes,
                    "quantized_weight_sha256":weight_hashes,
                    "weight_content":"independent_allocations_identical_synthetic_coefficients", "oracle_all_outputs_checked":true,
                    "phase":if measure {"performance"} else {"conformance"},
                    "conformance_commands":conformance, "samples":samples})
                    );
                }
            }
        }
    }
}
