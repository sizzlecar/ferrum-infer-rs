//! Real projection geometry only; not a recurrent-state or whole-GDN benchmark.
use super::super::b8_two_b4::gdn::Case;
use super::*;

fn assert_snapshot(actual: &[Vec<f16>], expected: &[Vec<f16>]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_finite_bits(a, e, &format!("GDN projection {index}"));
    }
}

fn experiment(
    input: usize,
    widths: &[usize],
    formats: &[GgufBlockFormat],
    label: &str,
    timing: bool,
) {
    let device = Device::system_default().expect("M8 GDN projection screen requires Metal");
    let queue = device.new_command_queue();
    let pipelines = pipelines(&device);
    let cases: Vec<_> = (0..if timing { 4 } else { 1 })
        .map(|_| Case::new(&device, input, widths, formats))
        .collect();
    let stride: usize = widths.iter().sum();
    println!(
        "{}",
        serde_json::json!({"kind":"b8_m8_mma_gdn_configuration","identity":identity(),
        "device":device.name(),"scope":label,"input":input,"widths":widths,
        "formats":formats.iter().map(|f| f.format_id()).collect::<Vec<_>>(),"packed_stride":stride,
        "physical_dispatches_per_workset":widths.len(),"includes_recurrence":false,
        "independent_weight_addresses":cases.iter().map(Case::weight_addresses).collect::<Vec<_>>() })
    );
    let raw_reference = cases[0].reference(false);
    let half_reference = cases[0].reference(true);
    let mut baseline: Vec<Vec<Vec<f16>>> = Vec::new();
    for arm in [Arm::ProductionMma, Arm::M8Mma] {
        for case in &cases {
            case.reset(arm);
        }
        run(&queue, arm, cases.len(), widths.len(), |encoder| {
            for case in &cases {
                case.encode(encoder, &pipelines, arm);
            }
        });
        for (index, case) in cases.iter().enumerate() {
            case.immutable();
            let snapshot = case.snapshot(arm);
            if arm == Arm::ProductionMma {
                assert_snapshot(&snapshot, &snapshot);
            } else {
                assert_snapshot(&snapshot, &baseline[index]);
            }
            for (leaf, actual) in snapshot.iter().enumerate() {
                println!(
                    "{}",
                    serde_json::json!({"kind":"b8_m8_mma_gdn_qualification","scope":label,
                    "arm":format!("{arm:?}"),"workset":index,"leaf":leaf,"format":formats[leaf].format_id(),
                    "shape":[ROWS,input,widths[leaf]],"packed_stride":stride,
                    "raw_coefficient_f64":metrics(actual,&raw_reference[leaf],widths[leaf]),
                    "half_coefficient_f64_diagnostic":metrics(actual,&half_reference[leaf],widths[leaf]),
                    "bitwise_equal_to_control":true,"absolute_numerical_quality_claimed":false,"release_approved":false})
                );
            }
            if arm == Arm::ProductionMma {
                baseline.push(snapshot);
            }
        }
    }
    if !timing {
        return;
    }
    for count in [1, 4] {
        for round in 0..WARMUP_ROUNDS + MEASURED_ROUNDS {
            for arm in order(round) {
                for case in &cases[..count] {
                    case.reset(arm);
                }
                let measurement = run(&queue, arm, count, widths.len(), |encoder| {
                    for case in &cases[..count] {
                        case.encode(encoder, &pipelines, arm);
                    }
                });
                for (index, case) in cases[..count].iter().enumerate() {
                    assert_snapshot(&case.snapshot(arm), &baseline[index]);
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":"b8_m8_mma_gdn_projection_timing","scope":label,
                    "round":round,"warmup":round<WARMUP_ROUNDS,"measurement":measurement,"packed_stride":stride,
                    "all_elements_bitwise_rechecked":true,"includes_recurrence":false,"absolute_numerical_quality_claimed":false})
                );
            }
        }
    }
    // Preserve cache behavior between paired commands; immutable allocations
    // are checked before the timing phase and after its last command.
    for case in &cases {
        case.immutable();
    }
}

#[test]
fn b8_m8_mma_gdn_packed_tails_bitwise_and_f64_diagnostic() {
    experiment(
        512,
        &[65, 67, 32, 32],
        &[
            GgufBlockFormat::Q5K,
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q8_0,
            GgufBlockFormat::Q8_0,
        ],
        "packed_tails",
        false,
    );
}

#[test]
#[ignore = "exclusive Metal GPU: actual GDN input/output projection shapes, bitwise admission"]
fn b8_m8_mma_qwen_gdn_projections_microbench() {
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
