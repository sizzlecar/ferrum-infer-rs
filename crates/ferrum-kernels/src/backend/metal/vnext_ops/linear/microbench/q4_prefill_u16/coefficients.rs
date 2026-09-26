//! Real GPU control/candidate coefficient readback, including 2-byte offset.
use super::*;

pub(super) fn check(
    device: &Device,
    queue: &CommandQueueRef,
    control: &ComputePipelineState,
    candidate: &ComputePipelineState,
    experiment: Experiment,
    name: &str,
    bytes: &[u8],
) -> bool {
    let block_bytes = GgufBlockFormat::Q4K.block_bytes();
    assert_eq!(bytes.len() % block_bytes, 0);
    let blocks = u32::try_from(bytes.len() / block_bytes).unwrap();
    let count = blocks as usize * 256;
    let weight = byte_buffer(device, bytes);
    let mut initial = vec![HALF_GUARD; HALF_PREFIX + count + GUARD];
    initial[HALF_PREFIX..HALF_PREFIX + count].fill(f16::NAN);
    let outputs = [buffer(device, &initial), buffer(device, &initial)];
    let mut immutable_after_command = Vec::new();
    for (pipeline, output) in [(control, &outputs[0]), (candidate, &outputs[1])] {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weight), WEIGHT_PREFIX as u64);
        encoder.set_buffer(1, Some(output), (HALF_PREFIX * 2) as u64);
        encoder.set_bytes(2, 4, (&blocks as *const u32).cast());
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(blocks).div_ceil(8), 1, 1),
            MTLSize::new(128, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        immutable_after_command.push(immutable_bytes(&weight, bytes));
    }
    let stored = [read_halves(&outputs[0]), read_halves(&outputs[1])];
    let guards = stored.iter().all(|values| {
        values[..HALF_PREFIX]
            .iter()
            .chain(&values[HALF_PREFIX + count..])
            .all(|v| v.to_bits() == HALF_GUARD.to_bits())
    });
    let actual = [
        &stored[0][HALF_PREFIX..HALF_PREFIX + count],
        &stored[1][HALF_PREFIX..HALF_PREFIX + count],
    ];
    let mut differences = 0_usize;
    let mut nonfinite = [0_usize; 2];
    let mut zero_coefficients = [0_usize; 2];
    let mut independent_nonzero_differences = [0_usize; 2];
    let mut independent_zero_sign_differences = [0_usize; 2];
    let mut first_difference = None;
    let mut first_oracle_difference = [None, None];
    let mut max_abs_error = [0.0_f64; 2];
    for (block_index, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let reference = reference::half_coefficients(GgufBlockFormat::Q4K, block);
        for (offset, expected) in reference.into_iter().enumerate() {
            let index = block_index * 256 + offset;
            if actual[0][index].to_bits() != actual[1][index].to_bits() {
                differences += 1;
                first_difference.get_or_insert(serde_json::json!({
                    "block":block_index,"coefficient":offset,
                    "control_bits":actual[0][index].to_bits(),
                    "candidate_bits":actual[1][index].to_bits()}));
            }
            for arm in 0..2 {
                let value = actual[arm][index];
                nonfinite[arm] += usize::from(!value.is_finite());
                zero_coefficients[arm] += usize::from(value == f16::ZERO);
                if value.to_bits() != expected.to_bits() {
                    if value == f16::ZERO && expected == f16::ZERO {
                        independent_zero_sign_differences[arm] += 1;
                    } else {
                        independent_nonzero_differences[arm] += 1;
                    }
                    first_oracle_difference[arm].get_or_insert(serde_json::json!({
                        "block":block_index,"coefficient":offset,
                        "actual_bits":value.to_bits(),"expected_bits":expected.to_bits()}));
                }
                if value.is_finite() && expected.is_finite() {
                    max_abs_error[arm] =
                        max_abs_error[arm].max((value.to_f64() - expected.to_f64()).abs());
                }
            }
        }
    }
    let immutable = immutable_after_command.iter().all(|v| *v);
    let qualified = differences == 0 && nonfinite == [0, 0] && guards && immutable;
    experiment.emit(serde_json::json!({"kind":"q4_prefill_u16_coefficients",
        "name":name,"elements":count,"weight_offset_bytes":WEIGHT_PREFIX,
        "control":"stage_q4k_f16","candidate":experiment.coefficient_probe(),
        "scope":"untimed_coefficient_readback_not_a_staged_performance_route",
        "gpu_control_candidate_bitwise_differences":differences,
        "first_gpu_difference":first_difference,"nonfinite":nonfinite,
        "zero_coefficients":zero_coefficients,
        "guards_passed":guards,"weight_immutable":immutable,
        "weight_immutable_after_each_command":immutable_after_command,
        "independent_exact_coefficient_f64_to_f16":{
            "nonzero_bit_differences":independent_nonzero_differences,
            "zero_sign_differences":independent_zero_sign_differences,
            "max_abs_error":max_abs_error,"first_difference":first_oracle_difference,
            "role":"baseline_arithmetic_diagnostic_not_candidate_equivalence"},
        "candidate_equivalence_qualified":qualified,"release_approved":false}));
    qualified
}

fn boundary_blocks() -> Vec<u8> {
    let mut result = Vec::new();
    // Every byte value appears in every qs position. Scale fields exercise all
    // low/high packed bits; d/dmin include signed zero and subnormal operands.
    let scales = [
        0x0000_u16, 0x8000, 0x0001, 0x8001, 0x0400, 0x8400, 0x1401, 0x9401, 0x4800, 0xc800,
    ];
    for byte in 0..=255_u16 {
        for (case, &d) in scales.iter().enumerate() {
            let mut block = vec![0_u8; GgufBlockFormat::Q4K.block_bytes()];
            block[..2].copy_from_slice(&d.to_le_bytes());
            block[2..4].copy_from_slice(&scales[(case + 3) % scales.len()].to_le_bytes());
            for (index, value) in block[4..16].iter_mut().enumerate() {
                *value = (byte as u8).wrapping_add((index * 37) as u8);
            }
            for (index, value) in block[16..].iter_mut().enumerate() {
                *value = (byte as u8).wrapping_add((index * 29) as u8);
            }
            result.extend(block);
        }
    }
    result
}

#[test]
fn q4_prefill_u16_real_metal_coefficients_are_bitwise_equal_at_offset_two() {
    let device = Device::system_default().expect("Metal coefficient test requires device");
    let queue = device.new_command_queue();
    let production = MetalKQuantGemmPipelines::new(&device).unwrap();
    let experimental = ExperimentalPipelines::new(&device);
    assert!(check(
        &device,
        &queue,
        &production.stage_q4_k,
        &experimental.coefficients,
        Experiment::U16Bytes,
        "all_bytes_packed_scales_signed_zero_subnormal",
        &boundary_blocks()
    ));
}
