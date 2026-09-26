//! Test-only comparison of staged GEMM output stores. Both variants perform
//! fresh quantized-to-F16 staging and the identical M32/N64/K32 MMA reduction.

use super::*;

fn staged_buffer(device: &Device, case: &PrefillStagingCase) -> Buffer {
    let mut values = vec![
        PrefillStagingCase::GUARD;
        PrefillStagingCase::STAGED_PREFIX + case.staged_elements() + 17
    ];
    values[PrefillStagingCase::STAGED_PREFIX
        ..PrefillStagingCase::STAGED_PREFIX + case.staged_elements()]
        .fill(f16::NAN);
    buffer(device, &values)
}

fn staged_coefficient_matches(actual: f16, expected: f16) -> bool {
    // The existing Q6 GPU decoder and CPU decoder can produce opposite zero
    // signs through their different arithmetic expressions. The finite-value
    // coefficient oracle treats only +/-0 as equal; no subnormal is flushed
    // and every nonzero coefficient must retain its exact half bits.
    actual.is_finite()
        && expected.is_finite()
        && (actual.to_bits() == expected.to_bits()
            || (actual == f16::ZERO && expected == f16::ZERO))
}

#[test]
fn staged_coefficient_oracle_preserves_nonzero_bits_and_rejects_nonfinite() {
    let negative_zero = f16::from_bits(0x8000);
    assert!(staged_coefficient_matches(f16::ZERO, negative_zero));
    assert!(staged_coefficient_matches(negative_zero, f16::ZERO));
    for bits in [0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff] {
        let value = f16::from_bits(bits);
        assert!(staged_coefficient_matches(value, value));
        assert!(!staged_coefficient_matches(value, f16::from_bits(bits ^ 1)));
    }
    for bits in [0x7c00, 0xfc00, 0x7e00] {
        let value = f16::from_bits(bits);
        assert!(!staged_coefficient_matches(value, value));
    }
}

fn validate_staged(case: &PrefillStagingCase, staged: &Buffer, check_coefficients: bool) {
    // SAFETY: every caller waits for the command before examining the shared
    // allocation, whose complete byte length is an integral number of halves.
    let values = unsafe {
        std::slice::from_raw_parts(
            staged.contents().cast::<f16>(),
            staged.length() as usize / 2,
        )
    };
    let start = PrefillStagingCase::STAGED_PREFIX;
    let end = start + case.staged_elements();
    for (index, &value) in values[..start].iter().enumerate().chain(
        values[end..]
            .iter()
            .enumerate()
            .map(|(index, value)| (end + index, value)),
    ) {
        assert_eq!(
            value.to_bits(),
            PrefillStagingCase::GUARD.to_bits(),
            "staged guard {index}"
        );
    }
    if check_coefficients {
        let format = case.shape.format;
        for (position, &value) in values[start..end].iter().enumerate() {
            let block = position / format.block_values() * format.block_bytes();
            let expected = f16::from_f32(format.decode_value(
                &case.quantized_bytes[block..block + format.block_bytes()],
                position % format.block_values(),
            ));
            assert!(
                staged_coefficient_matches(value, expected),
                "{format:?} staged coefficient {position}: actual={:#06x}, expected={:#06x}",
                value.to_bits(),
                expected.to_bits(),
            );
        }
    }
}

fn run_variant(
    case: &PrefillStagingCase,
    queue: &CommandQueueRef,
    pipelines: &MetalKQuantGemmPipelines,
    contiguous: &metal::ComputePipelineState,
    staged: &Buffer,
    candidate: bool,
) -> serde_json::Value {
    let mut sample = case.run_with_staged_pipeline(
        queue,
        pipelines,
        staged,
        true,
        if candidate {
            contiguous
        } else {
            &pipelines.staged_f16
        },
        usize::from(candidate),
    );
    sample["variant"] = if candidate {
        "staged_contiguous_128_thread_store"
    } else {
        "staged_existing_32_thread_row_store"
    }
    .into();
    sample
}

fn validate_pair(case: &PrefillStagingCase, dense_cpu_oracle: bool) -> serde_json::Value {
    let validation = case.validate(dense_cpu_oracle);
    // Existing validation checks every output, all row padding and outer
    // guards, and immutable input/quantized storage. F16 -> F32 preserves all
    // finite half bits, including signed zero, for its bitwise comparison.
    assert_eq!(
        validation["bitwise_differences"].as_u64(),
        Some(0),
        "contiguous stores must preserve every old staged output bit: {validation}"
    );
    validation
}

#[test]
fn prefill_staged_contiguous_store_matches_old_staged_bits_and_cpu_with_guards() {
    let device = Device::system_default().expect("prefill conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalKQuantGemmPipelines::new(&device).unwrap();
    let contiguous = MetalKQuantGemmPipelines::staged_contiguous_store_for_test(&device).unwrap();
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        // Exercise partial and complete M32/N64 tiles, multiple K blocks,
        // nonzero input/weight/output/staging offsets, and padded output rows.
        // These finite fixtures do not target signed-zero outputs or nonfinite
        // arithmetic. Output comparison remains bitwise, including zero signs.
        // These are kernel safety cases, not a proposed production work gate.
        for (rows, input, output) in [
            (1, 256, 1),
            (7, 256, 31),
            (31, 512, 63),
            (32, 256, 64),
            (33, 512, 67),
            (65, 256, 129),
        ] {
            let case = PrefillStagingCase::new(
                &device,
                Shape {
                    name: "contiguous_store_tail",
                    input,
                    output,
                    format,
                },
                rows,
            );
            let staged = staged_buffer(&device, &case);
            for candidate in [false, true] {
                run_variant(&case, &queue, &pipelines, &contiguous, &staged, candidate);
                validate_staged(&case, &staged, true);
            }
            validate_pair(&case, true);
        }
    }
}

#[test]
#[ignore = "GPU performance experiment: coordinate exclusive device access"]
fn quantized_prefill_staged_contiguous_store_9b_ffn_microbench() {
    let device = Device::system_default().expect("prefill microbench requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalKQuantGemmPipelines::new(&device).unwrap();
    let contiguous = MetalKQuantGemmPipelines::staged_contiguous_store_for_test(&device).unwrap();
    // One projection lives at a time. Gate/up writes one half of the real
    // concatenated intermediate row; down writes a contiguous hidden row.
    // M603 is an observed partial-tile prefill; M2048 is the large work budget.
    // Neither size changes production staging policy or establishes a SLO.
    for (name, input, output, format, stride, column) in [
        (
            "ffn_gate_or_up",
            4096,
            12288,
            GgufBlockFormat::Q4K,
            24576,
            12288,
        ),
        ("ffn_down_q4", 12288, 4096, GgufBlockFormat::Q4K, 4096, 0),
        ("ffn_down_q6", 12288, 4096, GgufBlockFormat::Q6K, 4096, 0),
    ] {
        for rows in [603, 2048] {
            let case = PrefillStagingCase::with_output_layout(
                &device,
                Shape {
                    name,
                    input,
                    output,
                    format,
                },
                rows,
                stride,
                column,
            );
            let staged = staged_buffer(&device, &case);
            let mut samples = Vec::new();
            let mut validation = serde_json::Value::Null;
            let warmup_rounds = 2;
            let measured_rounds = 6;
            for round in 0..warmup_rounds + measured_rounds {
                let order = if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for (position, candidate) in order.into_iter().enumerate() {
                    let mut sample =
                        run_variant(&case, &queue, &pipelines, &contiguous, &staged, candidate);
                    // Only guard inspection between paired commands; the
                    // full CPU scan follows the pair and is never GPU timing.
                    validate_staged(&case, &staged, false);
                    sample["round"] = round.into();
                    sample["position_in_pair"] = position.into();
                    sample["warmup"] = (round < warmup_rounds).into();
                    samples.push(sample);
                }
                validation = validate_pair(&case, false);
            }
            println!(
                "{}",
                serde_json::json!({
                    "schema_version": 1,
                    "kind": "quantized_prefill_staged_contiguous_store_9b_ffn_microbench",
                    "device": device.name(), "shape": name, "weight_format": format.format_id(),
                    "rows": rows, "input_features": input, "output_features": output,
                    "output_stride": stride, "output_column_offset": column,
                    "activation_and_staged_weight_type": "f16", "accumulation_type": "f32",
                    "output_type": "f16", "tile_m_n_k": [32, 64, 32],
                    "threads_per_threadgroup": 128, "threadgroup_bytes": 8192,
                    "quantized_bytes": case.quantized_bytes.len(),
                    "staged_payload_bytes": case.staged_elements() * 2,
                    "gpu_buffer_bytes": case.input.length() + case.quantized.length()
                        + staged.length() + case.outputs.iter().map(|buffer| buffer.length()).sum::<u64>(),
                    "existing_pipeline_max_threads": pipelines.staged_f16.max_total_threads_per_threadgroup(),
                    "candidate_pipeline_max_threads": contiguous.max_total_threads_per_threadgroup(),
                    "candidate_pipeline_execution_width": contiguous.thread_execution_width(),
                    "warmup_paired_rounds": warmup_rounds, "measured_paired_rounds": measured_rounds,
                    "dispatches_per_command": 2,
                    "correctness": "poison_each_output_before_command; full_old_staged_bitwise_equality_and_input_weight_output_guards_each_pair; staging_guards_each_command",
                    "oracle": "old_staged_full_output_bits; independent_dense_CPU_and_decoded_staged_coefficients_in_small_conformance_test",
                    "timing_scope": "completed_GPU_command_including_fresh_dequant_and_full_GEMM; host_encode_and_submit_wait_separate",
                    "comparison": "same_process_same_buffers_alternating_pair_order",
                    "scope": "synthetic_hot_projection_not_model_TTFT_ITL_or_throughput",
                    "production_dispatch_changed": false,
                    "gpu_timing_unavailable_samples": samples.iter().filter(|sample| sample["device_command_ns"].is_null()).count(),
                    "validation": validation, "samples": samples,
                })
            );
        }
    }
}
