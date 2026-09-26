//! Reuse the original full four-dispatch fixture and unchanged catalog evaluation.
use super::super::b8_two_b4::ffn::{qualify, Case, Snapshot};
use super::*;

fn assert_snapshot(actual: &Snapshot, expected: &Snapshot) {
    for (name, a, b) in [
        ("gate", &actual.gate, &expected.gate),
        ("up", &actual.up, &expected.up),
        ("activation", &actual.activated, &expected.activated),
        ("down", &actual.down, &expected.down),
    ] {
        assert_finite_bits(a, b, name);
    }
}

fn experiment(hidden: usize, intermediate: usize, timing: bool) {
    let device = Device::system_default().expect("M8 MMA screen requires Metal");
    let queue = device.new_command_queue();
    let pipelines = pipelines(&device);
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let cases: Vec<_> = (0..if timing { 4 } else { 1 })
            .map(|_| Case::new(&device, hidden, intermediate, format))
            .collect();
        println!(
            "{}",
            serde_json::json!({"kind":"b8_m8_mma_ffn_configuration",
            "identity":identity(),"device":device.name(),"hidden":hidden,"intermediate":intermediate,
            "down_format":format.format_id(),"full_ffn_dispatches":4,
            "packed_gate_up_stride":intermediate*2,"output_stride":hidden+9,"output_column":4,
            "independent_weight_addresses":cases.iter().map(Case::weight_addresses).collect::<Vec<_>>(),
            "warmup_rounds":WARMUP_ROUNDS,"measured_rounds":MEASURED_ROUNDS,
            "catalog_failures_are_reported_not_relabelled":true})
        );
        let raw_reference = cases[0].reference(false);
        let half_reference = cases[0].reference(true);
        let mut baseline: Vec<Snapshot> = Vec::new();
        for arm in [Arm::ProductionMma, Arm::M8Mma] {
            for case in &cases {
                case.reset(arm);
            }
            run(&queue, arm, cases.len(), 4, |encoder| {
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
                println!(
                    "{}",
                    serde_json::json!({"kind":"b8_m8_mma_ffn_qualification",
                    "arm":format!("{arm:?}"),"hidden":hidden,"intermediate":intermediate,
                    "down_format":format.format_id(),"workset":index,
                    "raw_coefficient_f64_catalog":qualify(&snapshot,&raw_reference,case),
                    "half_coefficient_f64_diagnostic":qualify(&snapshot,&half_reference,case),
                    "bitwise_equal_to_control":true,"absolute_numerical_quality_claimed":false,
                    "release_approved":false})
                );
                if arm == Arm::ProductionMma {
                    baseline.push(snapshot);
                }
            }
        }
        if !timing {
            continue;
        }
        for count in [1, 4] {
            for round in 0..WARMUP_ROUNDS + MEASURED_ROUNDS {
                for arm in order(round) {
                    for case in &cases[..count] {
                        case.reset(arm);
                    }
                    let measurement = run(&queue, arm, count, 4, |encoder| {
                        for case in &cases[..count] {
                            case.encode(encoder, &pipelines, arm);
                        }
                    });
                    for (index, case) in cases[..count].iter().enumerate() {
                        assert_snapshot(&case.snapshot(arm), &baseline[index]);
                    }
                    println!(
                        "{}",
                        serde_json::json!({"kind":"b8_m8_mma_complete_ffn_timing",
                        "round":round,"warmup":round<WARMUP_ROUNDS,"hidden":hidden,
                        "intermediate":intermediate,"down_format":format.format_id(),"measurement":measurement,
                        "all_stages_bitwise_rechecked":true,"absolute_numerical_quality_claimed":false})
                    );
                }
            }
        }
        // Check large input/weight allocations outside the paired timing loop.
        // Scanning them between commands would disturb the next GPU cache state.
        for case in &cases {
            case.immutable();
        }
    }
}

#[test]
fn b8_m8_mma_small_complete_ffn_bitwise_and_catalog_diagnostic() {
    experiment(256, 512, false);
}

#[test]
#[ignore = "exclusive Metal GPU: complete four-dispatch FFN, bitwise admission; raw catalog reported unchanged"]
fn b8_m8_mma_qwen_complete_ffn_microbench() {
    experiment(4096, 12288, true);
}
