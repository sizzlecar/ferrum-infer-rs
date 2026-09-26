//! Untimed qualification of the direct fragment-store mapping and layout.
use super::*;

fn guarded_output(device: &Device, params: LinearParams) -> Buffer {
    let mut values = vec![
        HALF_GUARD;
        HALF_PREFIX + params.rows as usize * params.output_stride as usize + GUARD
    ];
    for row in 0..params.rows as usize {
        let start = HALF_PREFIX
            + row * params.output_stride as usize
            + params.output_column_offset as usize;
        values[start..start + params.out_features as usize].fill(f16::NAN);
    }
    buffer(device, &values)
}

fn read_output(output: &Buffer, params: LinearParams) -> (Vec<f16>, bool) {
    let mut logical = Vec::new();
    let mut guards = true;
    for (index, &value) in read_halves(output).iter().enumerate() {
        let valid = index.checked_sub(HALF_PREFIX).is_some_and(|relative| {
            relative < params.rows as usize * params.output_stride as usize
                && (params.output_column_offset as usize
                    ..(params.output_column_offset + params.out_features) as usize)
                    .contains(&(relative % params.output_stride as usize))
        });
        if valid {
            logical.push(value);
        } else {
            guards &= value.to_bits() == HALF_GUARD.to_bits();
        }
    }
    (logical, guards)
}

fn rounding_value(index: usize) -> f32 {
    // Exact half endpoints and their F32 midpoint, with neighbors on either
    // side. Include normal/subnormal transitions and preserve signed zero.
    let half_bits = [
        0_u16, 1, 2, 0x03ff, 0x0400, 0x3554, 0x3555, 0x3bff, 0x3c00, 0x63fe,
    ];
    let bits = half_bits[(index / 5) % half_bits.len()];
    let low = f16::from_bits(bits).to_f32();
    let high = f16::from_bits(bits + 1).to_f32();
    let mid = (low + high) * 0.5;
    let value = match index % 5 {
        0 => low,
        1 => high,
        2 => mid,
        3 => mid - (high - low) / 16.0,
        _ => mid + (high - low) / 16.0,
    };
    if (index / (5 * half_bits.len())) % 2 == 0 {
        value
    } else {
        -value
    }
}

fn probe(
    device: &Device,
    queue: &CommandQueueRef,
    experimental: &ExperimentalPipelines,
    rounding: bool,
) -> bool {
    let params = LinearParams {
        rows: 32,
        in_features: 64,
        out_features: 64,
        output_stride: 73,
        output_column_offset: 3,
    };
    let values: Vec<f32> = (0..32 * 64)
        .map(|index| {
            if rounding {
                rounding_value(index)
            } else {
                (index as f32 - 1024.0) * 0.5
            }
        })
        .collect();
    let expected: Vec<_> = values.iter().copied().map(f16::from_f32).collect();
    let mut source_values = vec![-123.25_f32];
    source_values.extend_from_slice(&values);
    source_values.extend([-123.25_f32; GUARD]);
    let source = buffer(device, &source_values);
    let output = guarded_output(device, params);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(experimental.direct_store_probe.as_ref().unwrap());
    encoder.set_buffer(0, Some(&source), 4);
    encoder.set_buffer(1, Some(&output), (HALF_PREFIX * 2) as u64);
    encoder.set_bytes(
        2,
        std::mem::size_of::<LinearParams>() as u64,
        (&params as *const LinearParams).cast(),
    );
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: completed command, unchanged shared F32 allocation held alive.
    let source_readback =
        unsafe { std::slice::from_raw_parts(source.contents().cast::<f32>(), source_values.len()) };
    let immutable = source_readback
        .iter()
        .zip(&source_values)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let (actual, guards) = read_output(&output, params);
    let finite = actual.iter().all(|value| value.is_finite());
    let passed = bitwise_equal(&actual, &expected) && guards && immutable && finite;
    experimental.experiment.emit(serde_json::json!({
        "kind":"q4_prefill_direct_f16_fragment_store_probe",
        "case":if rounding { "half_rounding_boundaries_signed_zero_subnormal" } else { "unique_position_tags" },
        "source_offset_bytes":4,"output_offset_bytes":HALF_PREFIX*2,
        "output_stride":params.output_stride,"output_column_offset":params.output_column_offset,
        "actual_vs_independent_half_conversion":bits_report(&actual,&expected,64),
        "guards_passed":guards,"source_immutable":immutable,"finite":finite,
        "qualified":passed,"diagnostic_only":true,"release_approved":false
    }));
    passed
}

fn projection(
    device: &Device,
    queue: &CommandQueueRef,
    production: &MetalLinearPipelines,
    experimental: &ExperimentalPipelines,
    rows: u32,
    columns: u32,
) -> bool {
    let params = LinearParams {
        rows,
        in_features: 256,
        out_features: columns,
        output_stride: columns + 9,
        output_column_offset: 3,
    };
    let shape = Shape {
        name: "full_tail_store_layout",
        input: 256,
        output: columns,
        format: GgufBlockFormat::Q4K,
    };
    let encoded = matrix(shape, 2);
    let weight = byte_buffer(device, &encoded);
    let values = reference::dense_input(rows as usize, 256);
    let mut input_values = vec![HALF_GUARD; HALF_PREFIX];
    input_values.extend_from_slice(&values);
    input_values.extend([HALF_GUARD; GUARD]);
    let input = buffer(device, &input_values);
    let (control, kind) =
        production.plain_linear_dispatch(LinearPhysicalFormat::Q4K, ElementType::F16, params);
    assert!(selects_candidate(Arm::U16Bytes, GgufBlockFormat::Q4K, kind));
    let outputs = [
        guarded_output(device, params),
        guarded_output(device, params),
    ];
    let mut immutable = true;
    for (pipeline, output) in [(control, &outputs[0]), (&experimental.fused, &outputs[1])] {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input), (HALF_PREFIX * 2) as u64);
        encoder.set_buffer(1, Some(&weight), WEIGHT_PREFIX as u64);
        encoder.set_buffer(2, Some(output), (HALF_PREFIX * 2) as u64);
        bind_linear_params(encoder, params, LinearPhysicalFormat::Q4K, ElementType::F16);
        dispatch_linear_grid(encoder, params, kind);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        immutable &=
            immutable_bytes(&weight, &encoded) && bitwise_equal(read_halves(&input), &input_values);
    }
    let (control, control_guards) = read_output(&outputs[0], params);
    let (candidate, candidate_guards) = read_output(&outputs[1], params);
    let finite = control
        .iter()
        .chain(&candidate)
        .all(|value| value.is_finite());
    let passed = bitwise_equal(&control, &candidate)
        && control_guards
        && candidate_guards
        && immutable
        && finite;
    experimental.experiment.emit(serde_json::json!({
        "kind":"q4_prefill_direct_f16_full_tail_projection","rows":rows,"columns":columns,
        "input_offset_bytes":HALF_PREFIX*2,"output_offset_bytes":HALF_PREFIX*2,
        "weight_offset_bytes":WEIGHT_PREFIX,"output_stride":params.output_stride,
        "output_column_offset":params.output_column_offset,
        "candidate_vs_control":bits_report(&candidate,&control,columns as usize),
        "guards_passed":[control_guards,candidate_guards],"inputs_and_weights_immutable":immutable,
        "finite":finite,"qualified":passed,"diagnostic_only":true,"release_approved":false
    }));
    passed
}

pub(super) fn check(
    device: &Device,
    queue: &CommandQueueRef,
    production: &MetalLinearPipelines,
    experimental: &ExperimentalPipelines,
) -> bool {
    assert_eq!(experimental.experiment, Experiment::DirectF16);
    let mut passed = true;
    for rounding in [false, true] {
        passed &= probe(device, queue, experimental, rounding);
    }
    for (rows, columns) in [(32, 64), (33, 65), (64, 128)] {
        passed &= projection(device, queue, production, experimental, rows, columns);
    }
    passed
}

#[test]
fn q4_prefill_direct_f16_fragment_mapping_rounding_and_full_tail_layout() {
    let device = Device::system_default().expect("Metal fragment layout requires device");
    let queue = device.new_command_queue();
    let production = MetalLinearPipelines::new(&device).unwrap();
    let experimental = ExperimentalPipelines::for_experiment(&device, Experiment::DirectF16);
    assert!(check(&device, &queue, &production, &experimental));
}
