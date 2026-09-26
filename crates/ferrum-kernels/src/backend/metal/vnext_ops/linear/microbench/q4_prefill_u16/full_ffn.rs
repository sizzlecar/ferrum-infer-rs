//! Complete dense FFN, original production selector plus one explicit test PSO.
use super::*;

#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    hidden: usize,
    intermediate: usize,
}
impl Geometry {
    fn gate(self) -> Shape {
        Shape {
            name: "gate_up",
            input: self.hidden as u32,
            output: self.intermediate as u32,
            format: GgufBlockFormat::Q4K,
        }
    }
    fn down(self, format: GgufBlockFormat) -> Shape {
        Shape {
            name: "down",
            input: self.intermediate as u32,
            output: self.hidden as u32,
            format,
        }
    }
    fn packed(self) -> usize {
        2 * self.intermediate
    }
    fn activation_start(self) -> usize {
        HALF_PREFIX + self.rows * self.packed() + GUARD
    }
    fn output_stride(self) -> usize {
        self.hidden + 7
    }
}

struct Scratch {
    intermediate: Buffer,
    output: Buffer,
}

struct Case<'a> {
    geometry: Geometry,
    down_format: GgufBlockFormat,
    queue: &'a CommandQueueRef,
    production: &'a MetalLinearPipelines,
    experimental: &'a ExperimentalPipelines,
    input: Buffer,
    input_storage: Vec<f16>,
    encoded: [&'a [u8]; 3],
    weights: [Buffer; 3],
    scratch: [Scratch; 2],
}

#[derive(Clone)]
struct Snapshot {
    gate: Vec<f16>,
    up: Vec<f16>,
    activation: Vec<f16>,
    down: Vec<f16>,
    guards: bool,
    immutable: bool,
}
impl Snapshot {
    fn finite(&self) -> bool {
        [&self.gate, &self.up, &self.activation, &self.down]
            .into_iter()
            .flatten()
            .all(|v| v.is_finite())
    }
    fn equivalent(&self, other: &Self) -> bool {
        self.guards
            && other.guards
            && self.immutable
            && other.immutable
            && self.finite()
            && other.finite()
            && bitwise_equal(&self.gate, &other.gate)
            && bitwise_equal(&self.up, &other.up)
            && bitwise_equal(&self.activation, &other.activation)
            && bitwise_equal(&self.down, &other.down)
    }
    fn report(&self, expected: &Self, geometry: Geometry) -> serde_json::Value {
        serde_json::json!({"gate":bits_report(&self.gate, &expected.gate, geometry.intermediate),
            "up":bits_report(&self.up, &expected.up, geometry.intermediate),
            "activation":bits_report(&self.activation, &expected.activation, geometry.intermediate),
            "down":bits_report(&self.down, &expected.down, geometry.hidden),
            "guards_passed":self.guards,"inputs_and_weights_immutable":self.immutable,
            "finite":self.finite()})
    }
}

impl<'a> Case<'a> {
    fn new(
        device: &Device,
        queue: &'a CommandQueueRef,
        production: &'a MetalLinearPipelines,
        experimental: &'a ExperimentalPipelines,
        geometry: Geometry,
        down_format: GgufBlockFormat,
        encoded: [&'a [u8]; 3],
        input_values: &[f16],
    ) -> Self {
        assert_eq!(input_values.len(), geometry.rows * geometry.hidden);
        let mut input_storage = vec![HALF_GUARD; HALF_PREFIX];
        input_storage.extend_from_slice(input_values);
        input_storage.extend([HALF_GUARD; GUARD]);
        let scratch = std::array::from_fn(|_| Scratch {
            intermediate: buffer(
                device,
                &vec![
                    HALF_GUARD;
                    geometry.activation_start() + geometry.rows * geometry.intermediate + GUARD
                ],
            ),
            output: buffer(
                device,
                &vec![HALF_GUARD; HALF_PREFIX + geometry.rows * geometry.output_stride() + GUARD],
            ),
        });
        Self {
            geometry,
            down_format,
            queue,
            production,
            experimental,
            input: buffer(device, &input_storage),
            input_storage,
            weights: encoded.map(|values| byte_buffer(device, values)),
            encoded,
            scratch,
        }
    }

    fn output(&self, arm: Arm) -> &Scratch {
        &self.scratch[usize::from(arm == Arm::U16Bytes)]
    }

    fn reset(&self, arm: Arm) {
        let g = self.geometry;
        let scratch = self.output(arm);
        // SAFETY: exclusively owned shared buffers; previous command completed.
        unsafe {
            let intermediate = std::slice::from_raw_parts_mut(
                scratch.intermediate.contents().cast::<f16>(),
                scratch.intermediate.length() as usize / 2,
            );
            intermediate.fill(HALF_GUARD);
            intermediate[HALF_PREFIX..HALF_PREFIX + g.rows * g.packed()].fill(f16::NAN);
            intermediate[g.activation_start()..g.activation_start() + g.rows * g.intermediate]
                .fill(f16::NAN);
            let output = std::slice::from_raw_parts_mut(
                scratch.output.contents().cast::<f16>(),
                scratch.output.length() as usize / 2,
            );
            output.fill(HALF_GUARD);
            for row in 0..g.rows {
                let start = HALF_PREFIX + row * g.output_stride() + 3;
                output[start..start + g.hidden].fill(f16::NAN);
            }
        }
    }

    fn projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        arm: Arm,
        format: GgufBlockFormat,
        params: LinearParams,
        input: (&Buffer, usize),
        weight: &Buffer,
        output: (&Buffer, usize),
    ) -> bool {
        let (control, kind) =
            self.production
                .plain_linear_dispatch(physical(format), ElementType::F16, params);
        let selected = selects_candidate(arm, format, kind);
        encoder.set_compute_pipeline_state(if selected {
            &self.experimental.fused
        } else {
            control
        });
        encoder.set_buffer(0, Some(input.0), (input.1 * 2) as u64);
        encoder.set_buffer(1, Some(weight), WEIGHT_PREFIX as u64);
        encoder.set_buffer(2, Some(output.0), (output.1 * 2) as u64);
        bind_linear_params(encoder, params, physical(format), ElementType::F16);
        dispatch_linear_grid(encoder, params, kind);
        selected
    }

    fn run(&self, arm: Arm) -> serde_json::Value {
        let g = self.geometry;
        let scratch = self.output(arm);
        self.reset(arm); // Buffer poisoning is outside every measured interval.
        let started = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let mut changed = 0;
        for (weight, column) in [(&self.weights[0], 0), (&self.weights[1], g.intermediate)] {
            changed += usize::from(self.projection(
                encoder,
                arm,
                GgufBlockFormat::Q4K,
                LinearParams {
                    rows: g.rows as u32,
                    in_features: g.hidden as u32,
                    out_features: g.intermediate as u32,
                    output_stride: g.packed() as u32,
                    output_column_offset: column as u32,
                },
                (&self.input, HALF_PREFIX),
                weight,
                (&scratch.intermediate, HALF_PREFIX),
            ));
        }
        encoder.set_compute_pipeline_state(&self.production.swiglu);
        encoder.set_buffer(0, Some(&scratch.intermediate), (HALF_PREFIX * 2) as u64);
        encoder.set_buffer(
            1,
            Some(&scratch.intermediate),
            (g.activation_start() * 2) as u64,
        );
        let activation = SwiGluParams {
            rows: g.rows as u32,
            intermediate_size: g.intermediate as u32,
            gate_up_stride: g.packed() as u32,
        };
        encoder.set_bytes(
            2,
            std::mem::size_of::<SwiGluParams>() as u64,
            (&activation as *const SwiGluParams).cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (g.rows as u64 * g.intermediate as u64).div_ceil(THREADS_PER_GROUP),
                1,
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
        changed += usize::from(self.projection(
            encoder,
            arm,
            self.down_format,
            LinearParams {
                rows: g.rows as u32,
                in_features: g.intermediate as u32,
                out_features: g.hidden as u32,
                output_stride: g.output_stride() as u32,
                output_column_offset: 3,
            },
            (&scratch.intermediate, g.activation_start()),
            &self.weights[2],
            (&scratch.output, HALF_PREFIX),
        ));
        encoder.end_encoding();
        let host_encode_ns = started.elapsed().as_nanos() as u64;
        let submitted = Instant::now();
        command.commit();
        command.wait_until_completed();
        let host_submit_wait_ns = submitted.elapsed().as_nanos() as u64;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let expected_changed = if arm == Arm::Production {
            0
        } else {
            2 + usize::from(self.down_format == GgufBlockFormat::Q4K)
        };
        assert_eq!(
            changed, expected_changed,
            "the tested FFN must actually reach the declared PSOs"
        );
        serde_json::json!({"arm":arm.label(self.experimental.experiment),"compute_dispatches":4,
            "changed_q4_dispatches":changed,"weight_offset_bytes":WEIGHT_PREFIX,
            "host_encode_ns":host_encode_ns,"host_submit_wait_ns":host_submit_wait_ns,
            "device_command_ns":gpu_elapsed_ns(command)})
    }

    fn snapshot(&self, arm: Arm) -> Snapshot {
        let g = self.geometry;
        let scratch = self.output(arm);
        let intermediate = read_halves(&scratch.intermediate);
        let output = read_halves(&scratch.output);
        let packed = HALF_PREFIX..HALF_PREFIX + g.rows * g.packed();
        let activation = g.activation_start()..g.activation_start() + g.rows * g.intermediate;
        let mut guards = intermediate.iter().enumerate().all(|(i, v)| {
            packed.contains(&i) || activation.contains(&i) || v.to_bits() == HALF_GUARD.to_bits()
        });
        let mut gate = Vec::with_capacity(g.rows * g.intermediate);
        let mut up = Vec::with_capacity(g.rows * g.intermediate);
        let mut down = Vec::with_capacity(g.rows * g.hidden);
        for row in 0..g.rows {
            let start = HALF_PREFIX + row * g.packed();
            gate.extend_from_slice(&intermediate[start..start + g.intermediate]);
            up.extend_from_slice(&intermediate[start + g.intermediate..start + g.packed()]);
        }
        for (index, &value) in output.iter().enumerate() {
            let logical = index
                .checked_sub(HALF_PREFIX)
                .filter(|i| *i < g.rows * g.output_stride())
                .is_some_and(|i| (3..3 + g.hidden).contains(&(i % g.output_stride())));
            if logical {
                down.push(value);
            } else {
                guards &= value.to_bits() == HALF_GUARD.to_bits();
            }
        }
        Snapshot {
            gate,
            up,
            down,
            activation: intermediate[activation].to_vec(),
            guards,
            immutable: bitwise_equal(read_halves(&self.input), &self.input_storage)
                && self
                    .weights
                    .iter()
                    .zip(self.encoded)
                    .all(|(buffer, bytes)| immutable_bytes(buffer, bytes)),
        }
    }
}

fn baseline_report(actual: &Snapshot, reference: &Snapshot, g: Geometry) -> serde_json::Value {
    let actual_down: Vec<_> = actual.down.iter().map(|v| v.to_f32()).collect();
    let reference_down: Vec<_> = reference.down.iter().map(|v| v.to_f32()).collect();
    let catalog = numerical_tolerance::assert_matches(
        "independent F16-coefficient F64 FFN diagnostic",
        &actual_down,
        &[g.rows, g.hidden],
        &reference_down,
        &[g.rows, g.hidden],
        numerical_tolerance::LogicalDtype::Fp16,
        FULL_TOLERANCE,
    );
    let stages = actual.report(reference, g);
    let stage_bounds_passed = ["gate", "up", "activation", "down"]
        .into_iter()
        .all(|name| stages[name]["linear_bound_violations"].as_u64() == Some(0));
    serde_json::json!({"role":"baseline_arithmetic_diagnostic_not_timing_qualification",
        "arithmetic":"encoded_exact_f64_to_f16_coefficients_f64_dot_silu_f16_stage_writebacks",
        "full_output_independent_oracle":true,"stages":stages,
        "arithmetic_diagnostic_passed":stage_bounds_passed && catalog.is_ok(),
        "catalog_tolerance":FULL_TOLERANCE,"catalog_passed":catalog.is_ok(),
        "catalog_error":catalog.err(),"old_b4_experiment_status":"unchanged",
        "release_approved":false})
}

#[test]
fn q4_prefill_u16_small_full_ffn_preserves_stages_layout_and_q6_down() {
    small_full_ffn(Experiment::U16Bytes);
}

#[test]
fn q4_prefill_direct_f16_small_full_ffn_preserves_stages_layout_and_q6_down() {
    small_full_ffn(Experiment::DirectF16);
}

fn small_full_ffn(experiment: Experiment) {
    let device = Device::system_default().expect("Metal FFN test requires device");
    let queue = device.new_command_queue();
    let production = MetalLinearPipelines::new(&device).unwrap();
    let experimental = ExperimentalPipelines::for_experiment(&device, experiment);
    // Real short-row selector remains GEMV; the isolated candidate cannot
    // silently turn an unmeasured M7 tail into a tiled GEMM.
    let (_, short_kind) = production.plain_linear_dispatch(
        LinearPhysicalFormat::Q4K,
        ElementType::F16,
        LinearParams {
            rows: 7,
            in_features: 4096,
            out_features: 12288,
            output_stride: 24576,
            output_column_offset: 12288,
        },
    );
    assert!(!selects_candidate(
        Arm::U16Bytes,
        GgufBlockFormat::Q4K,
        short_kind
    ));
    let g = Geometry {
        rows: 33,
        hidden: 256,
        intermediate: 256,
    };
    let input = reference::dense_input(g.rows, g.hidden);
    let gate = matrix(g.gate(), 2);
    let up = matrix(g.gate(), 5);
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let down = matrix(g.down(format), 11);
        let case = Case::new(
            &device,
            &queue,
            &production,
            &experimental,
            g,
            format,
            [&gate, &up, &down],
            &input,
        );
        case.run(Arm::Production);
        let control = case.snapshot(Arm::Production);
        case.run(Arm::U16Bytes);
        let candidate = case.snapshot(Arm::U16Bytes);
        experiment.emit(
            serde_json::json!({"kind":"q4_prefill_u16_small_full_ffn",
            "down_format":format!("{format:?}"),"candidate_vs_control":candidate.report(&control,g),
            "candidate_equivalence_qualified":candidate.equivalent(&control),"release_approved":false})
        );
        assert!(candidate.equivalent(&control));
    }
}

#[test]
#[ignore = "complete dense M128 FFN diagnostic; release mode and exclusive Metal access"]
fn q4_prefill_u16_m128_complete_ffn_microbench() {
    complete_ffn(Experiment::U16Bytes);
}

#[test]
#[ignore = "complete dense M128 FFN diagnostic; release mode and exclusive Metal access"]
fn q4_prefill_direct_f16_m128_complete_ffn_microbench() {
    complete_ffn(Experiment::DirectF16);
}

fn complete_ffn(experiment: Experiment) {
    let device = Device::system_default().expect("Metal experiment requires device");
    let queue = device.new_command_queue();
    let production = MetalLinearPipelines::new(&device).unwrap();
    let experimental = ExperimentalPipelines::for_experiment(&device, experiment);
    let g = Geometry {
        rows: 128,
        hidden: 4096,
        intermediate: 12288,
    };
    let input = reference::dense_input(g.rows, g.hidden);
    let gate = matrix(g.gate(), 2);
    let up = matrix(g.gate(), 5);
    let downs = [
        matrix(g.down(GgufBlockFormat::Q4K), 11),
        matrix(g.down(GgufBlockFormat::Q6K), 11),
    ];
    let mut input_hasher = Sha256::new();
    for value in &input {
        input_hasher.update(value.to_le_bytes());
    }
    let weight_hashes: Vec<_> = ["gate", "up", "q4_down", "q6_down"]
        .into_iter()
        .zip([&gate, &up, &downs[0], &downs[1]])
        .map(|(name, bytes)| {
            serde_json::json!({"name":name,"bytes":bytes.len(),
            "sha256":format!("{:x}",Sha256::digest(bytes))})
        })
        .collect();
    experiment.emit(serde_json::json!({"kind":"q4_prefill_u16_predeclared",
        "rows":g.rows,"hidden":g.hidden,"intermediate":g.intermediate,
        "warmup_rounds":2,"measured_rounds":6,"paired_order":"alternate_every_round",
        "timing_scope":"one_complete_ffn_four_dispatches_not_full_model",
        "weight_worksets":1,"dense_nonzero_input":input.iter().all(|v|*v!=f16::ZERO),
        "input_half_le_sha256":format!("{:x}",input_hasher.finalize()),
        "encoded_weights":weight_hashes,"device_name":device.name(),
        "activation_and_coefficient_dtype":"f16","accumulation_dtype":"f32",
        "changed_mechanism":if experiment == Experiment::DirectF16 {
            "full_tile_f32_fragment_to_f16_device_store_original_scalar_q4_decoder"
        } else { "q4_packed_byte_loads_via_u16_original_shared_epilogue" },
        "production_selector":"unchanged_test_only_pso_override_on_existing_q4_tiled_route",
        "tile_mnk":[32,64,32],"threads":128,"dynamic_shared_bytes":8192,
        "qualification":"gpu_coefficients_and_all_ffn_stages_bitwise_finite_immutable_guards",
        "independent_oracle_role":"baseline_diagnostic_not_candidate_equivalence",
        "old_b4_oracle_and_catalog_status":"unchanged_not_overridden",
        "release_approved":false,"serving_performance_evidence":false}));
    let mut coefficients_passed = true;
    let layout_passed = experiment != Experiment::DirectF16
        || direct_store::check(&device, &queue, &production, &experimental);
    for (name, bytes) in [("gate", &gate), ("up", &up), ("q4_down", &downs[0])] {
        coefficients_passed &= coefficients::check(
            &device,
            &queue,
            &production.k_quant_gemm.stage_q4_k,
            &experimental.coefficients,
            experiment,
            name,
            bytes,
        );
    }
    experiment.emit(serde_json::json!({"kind":"q4_prefill_u16_oracle_started",
        "scope":"complete_cpu_f64_outside_all_gpu_timing"}));
    let oracle_gate = reference::project(g.gate(), &gate, &input);
    let oracle_up = reference::project(g.gate(), &up, &input);
    let oracle_activation = reference::activate(&oracle_gate, &oracle_up);
    let oracle_down = [
        reference::project(g.down(GgufBlockFormat::Q4K), &downs[0], &oracle_activation),
        reference::project(g.down(GgufBlockFormat::Q6K), &downs[1], &oracle_activation),
    ];
    let cases: Vec<_> = [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K]
        .into_iter()
        .enumerate()
        .map(|(index, format)| {
            Case::new(
                &device,
                &queue,
                &production,
                &experimental,
                g,
                format,
                [&gate, &up, &downs[index]],
                &input,
            )
        })
        .collect();
    let mut controls = Vec::new();
    let mut qualified = coefficients_passed && layout_passed;
    for (index, case) in cases.iter().enumerate() {
        let reference = Snapshot {
            gate: oracle_gate.clone(),
            up: oracle_up.clone(),
            activation: oracle_activation.clone(),
            down: oracle_down[index].clone(),
            guards: true,
            immutable: true,
        };
        case.run(Arm::Production);
        let control = case.snapshot(Arm::Production);
        case.run(Arm::U16Bytes);
        let candidate = case.snapshot(Arm::U16Bytes);
        let pair_qualified = candidate.equivalent(&control);
        qualified &= pair_qualified;
        experiment.emit(
            serde_json::json!({"kind":"q4_prefill_u16_complete_ffn_preflight",
            "rows":g.rows,"down_format":format!("{:?}",case.down_format),
            "candidate_equivalence_qualified":pair_qualified,
            "candidate_vs_control":candidate.report(&control,g),
            "baseline_oracle_status":baseline_report(&control,&reference,g),
            "candidate_oracle_status":baseline_report(&candidate,&reference,g),
            "release_approved":false}),
        );
        controls.push(control);
    }
    // The actual long-tail request also reached M106. Its rows are a prefix
    // of the same dense input/reference, so no second CPU dot is necessary.
    let tail = Geometry { rows: 106, ..g };
    for (index, format) in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K]
        .into_iter()
        .enumerate()
    {
        let case = Case::new(
            &device,
            &queue,
            &production,
            &experimental,
            tail,
            format,
            [&gate, &up, &downs[index]],
            &input[..tail.rows * tail.hidden],
        );
        case.run(Arm::Production);
        let control = case.snapshot(Arm::Production);
        case.run(Arm::U16Bytes);
        let candidate = case.snapshot(Arm::U16Bytes);
        let tail_reference = Snapshot {
            gate: oracle_gate[..tail.rows * tail.intermediate].to_vec(),
            up: oracle_up[..tail.rows * tail.intermediate].to_vec(),
            activation: oracle_activation[..tail.rows * tail.intermediate].to_vec(),
            down: oracle_down[index][..tail.rows * tail.hidden].to_vec(),
            guards: true,
            immutable: true,
        };
        qualified &= candidate.equivalent(&control);
        experiment.emit(serde_json::json!({"kind":"q4_prefill_u16_tail_preflight",
            "rows":tail.rows,"down_format":format!("{format:?}"),
            "candidate_equivalence_qualified":candidate.equivalent(&control),
            "candidate_vs_control":candidate.report(&control,tail),
            "baseline_oracle_status":baseline_report(&control,&tail_reference,tail),
            "candidate_oracle_status":baseline_report(&candidate,&tail_reference,tail),
            "timing_scope":"untimed_tail_correctness_only","release_approved":false}));
    }
    experiment.emit(
        serde_json::json!({"kind":"q4_prefill_u16_timing_qualification",
        "coefficients_qualified":coefficients_passed,"layout_qualified":layout_passed,
        "candidate_equivalence_qualified":qualified,
        "diagnostic_timing_eligible":qualified,"release_approved":false}),
    );
    assert!(
        qualified,
        "candidate equivalence failed; ALL performance loops skipped"
    );
    for (index, case) in cases.iter().enumerate() {
        for round in 0..8 {
            let order = if round % 2 == 0 {
                [Arm::Production, Arm::U16Bytes]
            } else {
                [Arm::U16Bytes, Arm::Production]
            };
            for arm in order {
                let timing = case.run(arm);
                let actual = case.snapshot(arm);
                let unchanged = actual.equivalent(&controls[index]);
                let timing_complete = timing["device_command_ns"]
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value > 0.0);
                experiment.emit(
                    serde_json::json!({"kind":"q4_prefill_u16_complete_ffn_timing",
                    "rows":g.rows,"down_format":format!("{:?}",case.down_format),
                    "round":round,"warmup":round<2,"timing":timing,
                    "full_stages_bitwise_match_preflight":unchanged,
                    "gpu_timing_complete":timing_complete,
                    "diagnostic_only":true,"release_approved":false}),
                );
                assert!(
                    unchanged,
                    "timed command changed full stage outputs or guards"
                );
                assert!(
                    timing_complete,
                    "missing/nonpositive GPU interval; performance evidence incomplete"
                );
            }
        }
    }
}
