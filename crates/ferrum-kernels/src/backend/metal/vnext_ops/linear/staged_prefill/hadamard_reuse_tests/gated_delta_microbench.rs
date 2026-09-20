//! Resident QKV + Z projection timing, excluding the remainder of GDN.

use super::*;
use crate::backend::metal::vnext_ops::linear::microbench::gpu_elapsed_ns;
use metal::CommandQueueRef;

fn run(
    fixture: &GatedDeltaProjectionCase,
    pipelines: &MetalLinearPipelines,
    queue: &CommandQueueRef,
    reuse: bool,
    repetitions: usize,
) -> f64 {
    metal::objc::rc::autoreleasepool(|| {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for _ in 0..repetitions {
            fixture.encode(pipelines, encoder, reuse);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let ns = gpu_elapsed_ns(command).expect("completed Metal GPU timestamp interval");
        assert!(ns.is_finite() && ns > 0.0);
        ns / 1_000.0 / repetitions as f64
    })
}

#[test]
#[ignore = "isolated Metal paired GPU timing; exclude compiler and other GPU work"]
fn gated_delta_qkv_z_hadamard_reuse_paired_gpu_timing() {
    const REPETITIONS: usize = 64;
    const WARMUPS: usize = 2;
    const PAIRS: usize = 5;
    // Dimensions of the local Bonsai 2 27B projections, not selection gates.
    const WIDTH: usize = 5120;
    const OUTPUTS: [usize; 2] = [10240, 6144];
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.gated-delta.hadamard.timing").unwrap())
            .unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    let fixture =
        GatedDeltaProjectionCase::new(runtime, &pipelines, 1, WIDTH, &OUTPUTS, ElementType::F32);

    // Validate both routes independently before warming the resident workload.
    fixture.reset();
    run(&fixture, &pipelines, &queue, false, 1);
    let expected = fixture.checked_output();
    fixture.reset();
    run(&fixture, &pipelines, &queue, true, 1);
    assert_eq!(fixture.checked_output(), expected);
    fixture.assert_sources_unchanged();
    for _ in 0..WARMUPS {
        for reuse in [false, true] {
            run(&fixture, &pipelines, &queue, reuse, REPETITIONS);
        }
    }

    let mut samples = Vec::new();
    for pair in 0..PAIRS {
        let order = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut durations = [0.0_f64; 2];
        for reuse in order {
            durations[usize::from(reuse)] = run(&fixture, &pipelines, &queue, reuse, REPETITIONS);
            assert_eq!(fixture.checked_output(), expected);
        }
        samples.push(serde_json::json!({
            "pair": pair,
            "order": order.map(|reuse| if reuse { "shared_transform" } else { "old_loop" }),
            "control_gpu_us": durations[0],
            "candidate_gpu_us": durations[1],
            "candidate_over_control": durations[1] / durations[0],
        }));
    }
    fixture.assert_sources_unchanged();
    eprintln!(
        "GDN_QKV_Z_HADAMARD_GPU {}",
        serde_json::json!({
            "schema_version": 1,
            "kind": "gated_delta_qkv_z_hadamard_reuse_paired_gpu_microbench",
            "device": runtime.device().name(),
            "rows": 1, "in_features": WIDTH, "projection_outputs": OUTPUTS,
            "input_dtype": "f32", "output_dtype": "f32", "accumulator_dtype": "f32",
            "weight_format": "quantization.gguf.pq2-0",
            "weight_values": "deterministic_synthetic_valid_pq2",
            "sign_values": "deterministic_synthetic_plus_minus_one",
            "hadamard_block_size": 128,
            "hadamard_inverse": false, "hadamard_permutation": null,
            "control": "old_staged_prefill_dispatch_loop",
            "candidate": "adjacent_projection_steps_shared_transform",
            "control_dispatches_per_iteration": fixture.dispatch_count(false),
            "candidate_dispatches_per_iteration": fixture.dispatch_count(true),
            "warmup_pairs": WARMUPS, "pairs": PAIRS,
            "qkv_z_iterations_per_sample": REPETITIONS,
            "working_set": "resident_qkv_and_z_weights_input_signs_and_invocation_scratch",
            "timing_scope": "GPU_completed_command_per_QKV_plus_Z_pair; excludes_other_GDN_stages_upload_readback_host_encode",
            "validation": "complete_scratch_output_bits_input_and_canaries_each_sample; weight_and_sign_bits_before_and_after",
            "samples": samples,
        })
    );
}
