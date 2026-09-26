//! Actual GDN projection dimensions; this does not measure recurrence/state.
use super::*;

pub(in super::super) struct Case {
    pub(in super::super) projections: Vec<Projection>,
    pub(in super::super) input: Halves,
    input_values: Vec<f16>,
    pub(in super::super) output: [Halves; 2],
    pub(in super::super) width: usize,
}

impl Case {
    pub(in super::super) fn new(
        device: &Device,
        input_width: usize,
        widths: &[usize],
        formats: &[GgufBlockFormat],
    ) -> Self {
        assert_eq!(widths.len(), formats.len());
        let input_values = dense_input(input_width);
        let width = widths.iter().sum();
        Self {
            projections: widths
                .iter()
                .zip(formats)
                .enumerate()
                .map(|(i, (&output, &format))| {
                    Projection::new(
                        device,
                        Shape {
                            name: "gdn_projection",
                            input: input_width as u32,
                            output: output as u32,
                            format,
                        },
                        i,
                    )
                })
                .collect(),
            input: Halves::new(device, &input_values),
            input_values,
            output: std::array::from_fn(|_| Halves::empty(device, ROWS * width)),
            width,
        }
    }

    pub(in super::super) fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &Pipelines,
        arm: Arm,
    ) {
        let mut column = 0;
        for p in &self.projections {
            p.encode(
                encoder,
                pipelines,
                arm,
                &self.input,
                &self.output[usize::from(arm != Arm::ProductionMma)],
                self.width as u32,
                column,
            );
            column += p.shape.output;
        }
    }

    pub(in super::super) fn snapshot(&self, arm: Arm) -> Vec<Vec<f16>> {
        let out = self.output[usize::from(arm != Arm::ProductionMma)].values();
        let mut column = 0;
        self.projections
            .iter()
            .map(|p| {
                let width = p.shape.output as usize;
                let result = (0..ROWS)
                    .flat_map(|row| {
                        out[row * self.width + column..row * self.width + column + width]
                            .iter()
                            .copied()
                    })
                    .collect();
                column += width;
                result
            })
            .collect()
    }

    pub(in super::super) fn immutable(&self) {
        exact_bits(self.input.values(), &self.input_values);
        for p in &self.projections {
            p.immutable();
        }
    }

    pub(in super::super) fn reset(&self, arm: Arm) {
        self.output[usize::from(arm != Arm::ProductionMma)].reset(f16::NAN);
    }

    pub(in super::super) fn reference(&self, half_weights: bool) -> Vec<Vec<f16>> {
        self.projections
            .iter()
            .map(|p| oracle::project(p, &self.input_values, half_weights))
            .collect()
    }

    pub(in super::super) fn weight_addresses(&self) -> Vec<usize> {
        self.projections
            .iter()
            .map(|p| p.weight.contents() as usize)
            .collect()
    }
}

fn experiment(
    input: usize,
    widths: &[usize],
    formats: &[GgufBlockFormat],
    label: &str,
    timing: bool,
) {
    let device = Device::system_default().expect("B8 GDN projections require Metal");
    let queue = device.new_command_queue();
    let pipelines = Pipelines::new(&device);
    let cases: Vec<_> = (0..if timing { 4 } else { 1 })
        .map(|_| Case::new(&device, input, widths, formats))
        .collect();
    println!(
        "{}",
        serde_json::json!({"kind":"b8_two_b4_gdn_configuration","device":device.name(),"scope":label,"rows":ROWS,"input":input,"widths":widths,
        "formats":formats.iter().map(|f| f.format_id()).collect::<Vec<_>>(),"packed_stride":cases[0].width,"physical_dispatches_per_workset":widths.len(),
        "small_batch_shader_sha256":format!("{:x}",Sha256::digest(small_batch::SHADER_SOURCE.as_bytes())),
        "independent_weight_addresses":cases.iter().map(|c| c.projections.iter().map(|p| p.weight.contents() as usize).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "q8_route":"unchanged_actual_B8_TiledGemm","includes_recurrence":false,"release_approved":false})
    );
    let expected: Vec<_> = cases[0]
        .projections
        .iter()
        .map(|p| oracle::project(p, &cases[0].input_values, false))
        .collect();
    let mut qualified = true;
    let mut snapshots = Vec::new();
    for arm in [Arm::ProductionMma, Arm::TwoB4] {
        run(&queue, arm, cases.len(), widths.len(), |encoder| {
            for case in &cases {
                case.encode(encoder, &pipelines, arm);
            }
        });
        let mut arm_snapshots = Vec::new();
        for (index, case) in cases.iter().enumerate() {
            case.immutable();
            let snapshot = case.snapshot(arm);
            for (leaf, actual) in snapshot.iter().enumerate() {
                let measured = metrics(actual, &expected[leaf], widths[leaf]);
                qualified &= measured["linear_bound_violations"] == 0;
                println!(
                    "{}",
                    serde_json::json!({"kind":"b8_two_b4_gdn_projection_qualification","scope":label,"arm":format!("{arm:?}"),"leaf":leaf,"format":formats[leaf].format_id(),"shape":[ROWS,input,widths[leaf]],"packed_stride":case.width,"workset":index,"independent_exact_coefficient_f64":measured,"release_approved":false})
                );
            }
            arm_snapshots.push(snapshot);
        }
        snapshots.push(arm_snapshots);
    }
    // The small Q8 leaves retain exactly their original pipeline and math.
    for (leaf, format) in formats.iter().enumerate() {
        if *format == GgufBlockFormat::Q8_0 {
            for i in 0..cases.len() {
                exact_bits(&snapshots[0][i][leaf], &snapshots[1][i][leaf]);
            }
        }
    }
    assert!(
        qualified,
        "unchanged F16 linear threshold is required before GDN projection timing"
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
                        case.output[usize::from(arm == Arm::TwoB4)].reset(f16::NAN);
                    }
                    let measurement = run(&queue, arm, count, widths.len(), |encoder| {
                        for case in &cases[..count] {
                            case.encode(encoder, &pipelines, arm);
                        }
                    });
                    for (i, case) in cases[..count].iter().enumerate() {
                        let actual = case.snapshot(arm);
                        for (a, e) in actual
                            .iter()
                            .zip(&snapshots[usize::from(arm == Arm::TwoB4)][i])
                        {
                            exact_bits(a, e);
                        }
                    }
                    println!(
                        "{}",
                        serde_json::json!({"kind":"b8_two_b4_gdn_projection_timing","scope":label,"round":round,"warmup":round<2,"packed_stride":cases[0].width,"measurement":measurement,"includes_recurrence":false})
                    );
                }
            }
        }
    }
    for case in &cases {
        case.immutable();
    }
}

#[test]
fn b8_two_b4_gdn_packed_tails_independent_f64_and_guards() {
    experiment(
        256,
        &[129, 65, 32, 32],
        &[
            GgufBlockFormat::Q5K,
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q8_0,
            GgufBlockFormat::Q8_0,
        ],
        "gdn_qkvzba_tails",
        false,
    );
}

#[test]
#[ignore = "B8 actual GDN projection geometry: exclusive GPU, independent oracle before timing"]
fn b8_two_b4_qwen_gdn_projections_microbench() {
    experiment(
        4096,
        &[8192, 4096, 32, 32],
        &[
            GgufBlockFormat::Q5K,
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q8_0,
            GgufBlockFormat::Q8_0,
        ],
        "gdn_qkvzba_four_projections",
        true,
    );
    experiment(
        4096,
        &[4096],
        &[GgufBlockFormat::Q5K],
        "gdn_output_projection_only",
        true,
    );
}
