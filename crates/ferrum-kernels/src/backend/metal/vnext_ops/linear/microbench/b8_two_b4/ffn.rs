//! Full gate/up -> unchanged SwiGLU -> down, four dispatches in both arms.
use super::*;
use crate::backend::metal::vnext_ops::numerical_tolerance;

const FULL_TOLERANCE: &str =
    "runtime-vnext.metal.dense-swiglu.v1.operation.fp16.gguf-q4-k-q6-k.full-pipeline";

pub(in super::super) struct Output {
    pub(in super::super) packed: Halves,
    pub(in super::super) activated: Halves,
    pub(in super::super) down: Halves,
}
pub(in super::super) struct Case {
    pub(in super::super) hidden: usize,
    pub(in super::super) intermediate: usize,
    pub(in super::super) input: Halves,
    input_values: Vec<f16>,
    pub(in super::super) gate: Projection,
    pub(in super::super) up: Projection,
    pub(in super::super) down: Projection,
    pub(in super::super) output: [Output; 2],
}

pub(in super::super) struct Snapshot {
    pub(in super::super) gate: Vec<f16>,
    pub(in super::super) up: Vec<f16>,
    pub(in super::super) activated: Vec<f16>,
    pub(in super::super) down: Vec<f16>,
}

impl Case {
    pub(in super::super) fn new(
        device: &Device,
        hidden: usize,
        intermediate: usize,
        down_format: GgufBlockFormat,
    ) -> Self {
        let shape = |name, input, output, format| Shape {
            name,
            input: input as u32,
            output: output as u32,
            format,
        };
        let input_values = dense_input(hidden);
        Self {
            hidden,
            intermediate,
            input: Halves::new(device, &input_values),
            input_values,
            gate: Projection::new(
                device,
                shape("gate", hidden, intermediate, GgufBlockFormat::Q4K),
                0,
            ),
            up: Projection::new(
                device,
                shape("up", hidden, intermediate, GgufBlockFormat::Q4K),
                1,
            ),
            down: Projection::new(device, shape("down", intermediate, hidden, down_format), 2),
            output: std::array::from_fn(|_| Output {
                packed: Halves::empty(device, ROWS * intermediate * 2),
                activated: Halves::empty(device, ROWS * intermediate),
                down: Halves::new(device, &vec![HALF_GUARD; ROWS * (hidden + 9)]),
            }),
        }
    }

    fn output(&self, arm: Arm) -> &Output {
        &self.output[usize::from(arm != Arm::ProductionMma)]
    }

    pub(in super::super) fn reset(&self, arm: Arm) {
        let out = self.output(arm);
        out.packed.reset(f16::NAN);
        out.activated.reset(f16::NAN);
        out.down.reset(HALF_GUARD);
    }

    pub(in super::super) fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &Pipelines,
        arm: Arm,
    ) {
        let out = self.output(arm);
        let packed = (self.intermediate * 2) as u32;
        self.gate
            .encode(encoder, pipelines, arm, &self.input, &out.packed, packed, 0);
        self.up.encode(
            encoder,
            pipelines,
            arm,
            &self.input,
            &out.packed,
            packed,
            self.intermediate as u32,
        );
        self.encode_activation(encoder, &pipelines.production, arm);
        self.down.encode(
            encoder,
            pipelines,
            arm,
            &out.activated,
            &out.down,
            (self.hidden + 9) as u32,
            4,
        );
    }

    pub(in super::super) fn encode_activation(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &MetalLinearPipelines,
        arm: Arm,
    ) {
        let out = self.output(arm);
        encoder.set_compute_pipeline_state(&pipelines.swiglu);
        encoder.set_threadgroup_memory_length(0, 0);
        encoder.set_buffer(0, Some(&out.packed.buffer), (PREFIX * 2) as u64);
        encoder.set_buffer(1, Some(&out.activated.buffer), (PREFIX * 2) as u64);
        let activation = SwiGluParams {
            rows: ROWS as u32,
            intermediate_size: self.intermediate as u32,
            gate_up_stride: (self.intermediate * 2) as u32,
        };
        encoder.set_bytes(
            2,
            std::mem::size_of::<SwiGluParams>() as u64,
            (&activation as *const SwiGluParams).cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (ROWS as u64 * self.intermediate as u64).div_ceil(THREADS_PER_GROUP),
                1,
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    pub(in super::super) fn snapshot(&self, arm: Arm) -> Snapshot {
        let out = self.output(arm);
        let packed = out.packed.values();
        let extract = |offset| {
            (0..ROWS)
                .flat_map(|row| {
                    packed[row * self.intermediate * 2 + offset
                        ..row * self.intermediate * 2 + offset + self.intermediate]
                        .iter()
                        .copied()
                })
                .collect()
        };
        let all_down = out.down.values();
        let mut down = Vec::with_capacity(ROWS * self.hidden);
        for row in all_down.chunks_exact(self.hidden + 9) {
            assert!(row[..4]
                .iter()
                .chain(&row[4 + self.hidden..])
                .all(|v| v.to_bits() == HALF_GUARD.to_bits()));
            down.extend_from_slice(&row[4..4 + self.hidden]);
        }
        Snapshot {
            gate: extract(0),
            up: extract(self.intermediate),
            activated: out.activated.values().to_vec(),
            down,
        }
    }

    pub(in super::super) fn reference(&self, half_weights: bool) -> Snapshot {
        let gate = oracle::project(&self.gate, &self.input_values, half_weights);
        let up = oracle::project(&self.up, &self.input_values, half_weights);
        let activated = oracle::activate(&gate, &up);
        let down = oracle::project(&self.down, &activated, half_weights);
        Snapshot {
            gate,
            up,
            activated,
            down,
        }
    }

    pub(in super::super) fn immutable(&self) {
        exact_bits(self.input.values(), &self.input_values);
        for projection in [&self.gate, &self.up, &self.down] {
            projection.immutable();
        }
    }

    pub(in super::super) fn weight_addresses(&self) -> [usize; 3] {
        [
            self.gate.weight.contents() as usize,
            self.up.weight.contents() as usize,
            self.down.weight.contents() as usize,
        ]
    }
}

pub(in super::super) fn qualify(
    actual: &Snapshot,
    reference: &Snapshot,
    case: &Case,
) -> serde_json::Value {
    let mut stages = serde_json::Map::new();
    for (name, a, e, width) in [
        ("gate", &actual.gate, &reference.gate, case.intermediate),
        ("up", &actual.up, &reference.up, case.intermediate),
        (
            "activation",
            &actual.activated,
            &reference.activated,
            case.intermediate,
        ),
        ("down", &actual.down, &reference.down, case.hidden),
    ] {
        stages.insert(name.to_owned(), metrics(a, e, width));
    }
    let a: Vec<_> = actual.down.iter().map(|v| v.to_f32()).collect();
    let e: Vec<_> = reference.down.iter().map(|v| v.to_f32()).collect();
    let catalog = numerical_tolerance::assert_matches(
        "B8 encoded coefficient FP64 whole FFN",
        &a,
        &[ROWS, case.hidden],
        &e,
        &[ROWS, case.hidden],
        numerical_tolerance::LogicalDtype::Fp16,
        FULL_TOLERANCE,
    );
    let passed = catalog.is_ok() && stages.values().all(|s| s["linear_bound_violations"] == 0);
    serde_json::json!({"stages":stages,"catalog":FULL_TOLERANCE,"catalog_passed":catalog.is_ok(),"catalog_error":catalog.err(),"passed":passed})
}

fn experiment(hidden: usize, intermediate: usize, timing: bool) {
    let device = Device::system_default().expect("B8 experiment requires Metal");
    let queue = device.new_command_queue();
    let pipelines = Pipelines::new(&device);
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        // These are independent physical allocations, not aliases of one hot weight.
        let cases: Vec<_> = (0..if timing { 4 } else { 1 })
            .map(|_| Case::new(&device, hidden, intermediate, format))
            .collect();
        println!(
            "{}",
            serde_json::json!({"kind":"b8_two_b4_ffn_configuration","device":device.name(),"rows":ROWS,"hidden":hidden,"intermediate":intermediate,
            "down_format":format.format_id(),"small_batch_shader_sha256":format!("{:x}",Sha256::digest(small_batch::SHADER_SOURCE.as_bytes())),
            "candidate":"one_dispatch_two_independent_B4_row_tiles","production":"actual_B8_TiledGemm","full_ffn_dispatches":4,
            "packed_gate_up_stride":intermediate*2,"output_stride":hidden+9,"output_column":4,"weight_byte_offset":WEIGHT_PREFIX,
            "independent_weight_addresses":cases.iter().map(|c| [c.gate.weight.contents() as usize,c.up.weight.contents() as usize,c.down.weight.contents() as usize]).collect::<Vec<_>>(),
            "comparison":"common_exact_coefficient_F64_and_unchanged_F16_catalog","warmup_rounds":2,"measured_rounds":8,"release_approved":false})
        );
        let reference = cases[0].reference(false);
        let half_reference = cases[0].reference(true);
        let mut snapshots = Vec::new();
        let mut qualified = true;
        for arm in [Arm::ProductionMma, Arm::TwoB4] {
            for case in &cases {
                case.reset(arm);
            }
            run(&queue, arm, cases.len(), 4, |encoder| {
                for case in &cases {
                    case.encode(encoder, &pipelines, arm);
                }
            });
            let mut arm_snapshots = Vec::new();
            for (index, case) in cases.iter().enumerate() {
                case.immutable();
                let snapshot = case.snapshot(arm);
                let common = qualify(&snapshot, &reference, case);
                qualified &= common["passed"] == true;
                println!(
                    "{}",
                    serde_json::json!({"kind":"b8_two_b4_ffn_qualification","arm":format!("{arm:?}"),"rows":ROWS,"hidden":hidden,"intermediate":intermediate,"down":format.format_id(),"workset":index,
                    "common_exact_coefficient_f64":common,"half_coefficient_f64_diagnostic":qualify(&snapshot,&half_reference,case),
                    "candidate_vs_control_bitwise_required":false,"release_approved":false})
                );
                arm_snapshots.push(snapshot);
            }
            snapshots.push(arm_snapshots);
        }
        // Do not reinterpret a pre-existing baseline error as candidate quality.
        assert!(
            qualified,
            "both complete FFN arms must satisfy the unchanged common numerical gate before timing"
        );
        if timing {
            for count in [1, 4] {
                for round in 0..10 {
                    let order = if round % 2 == 0 {
                        [Arm::ProductionMma, Arm::TwoB4]
                    } else {
                        [Arm::TwoB4, Arm::ProductionMma]
                    };
                    for arm in order {
                        for case in &cases[..count] {
                            case.reset(arm);
                        }
                        let measurement = run(&queue, arm, count, 4, |encoder| {
                            for case in &cases[..count] {
                                case.encode(encoder, &pipelines, arm);
                            }
                        });
                        for (index, case) in cases[..count].iter().enumerate() {
                            let actual = case.snapshot(arm);
                            let expected = &snapshots[usize::from(arm == Arm::TwoB4)][index];
                            for (a, e) in [
                                (&actual.gate, &expected.gate),
                                (&actual.up, &expected.up),
                                (&actual.activated, &expected.activated),
                                (&actual.down, &expected.down),
                            ] {
                                exact_bits(a, e);
                            }
                        }
                        println!(
                            "{}",
                            serde_json::json!({"kind":"b8_two_b4_complete_ffn_timing","round":round,"warmup":round<2,"hidden":hidden,"intermediate":intermediate,"down":format.format_id(),"measurement":measurement})
                        );
                    }
                }
            }
        }
        for case in &cases {
            case.immutable();
        }
    }
}

#[test]
fn b8_two_b4_small_complete_ffn_independent_f64_and_guards() {
    experiment(256, 512, false);
}

#[test]
#[ignore = "complete B8 FFN candidate: exclusive GPU; numerical qualification precedes timing"]
fn b8_two_b4_qwen_complete_ffn_microbench() {
    experiment(4096, 12288, true);
}
