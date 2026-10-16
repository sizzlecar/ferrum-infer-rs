//! Reference calibration only. The existing complete-FFN qualification tests
//! and their frozen terminal tolerance remain unchanged in the parent module.
use super::*;

// Flattened indices from the recorded generation-zero/one FP16/FP32-partial
// failures, plus first/last-element controls. Selection is frozen before this
// diagnostic; no search for a convenient new reference or passing subset.
const GATE_POINTS: [usize; 5] = [0, 27, 3_937, 202_277, 557_055];
const UP_POINTS: [usize; 5] = [0, 5, 1_646, 510_324, 557_055];
const DOWN_POINTS: [usize; 10] = [
    0, 22, 135, 279, 376, 8_439, 105_514, 127_628, 144_130, 163_839,
];

/// Independently decode the original logical q/z/scale arrays at one output
/// coordinate. Never read repacked Marlin weights or an actual GPU activation.
fn scalar_point(f: Fixture, host: &HostFixture, input: &[f16], index: usize) -> (f64, f64, f32) {
    assert!(index < f.rows * f.output_features);
    assert_eq!(input.len(), f.rows * f.input_features);
    let (row, col) = (index / f.output_features, index % f.output_features);
    let groups = f.input_features / GROUP_SIZE;
    let (mut sum, mut sum_absolute, mut sequential_f32) = (0.0_f64, 0.0_f64, 0.0_f32);
    for k in 0..f.input_features {
        let q = ((host.qweight_gptq[(k / 8) * f.output_features + col] as u32 >> (4 * (k % 8)))
            & 15) as i32;
        let z = ((host.zero_points_compressed_tensors[(col / 8) * groups + k / GROUP_SIZE] as u32
            >> (4 * (col % 8)))
            & 15) as i32;
        let scale = host.scales_grouped[(k / GROUP_SIZE) * f.output_features + col];
        let weight = f16::from_f32((q - z) as f32 * scale.to_f32());
        let x = input[row * f.input_features + k];
        let product = x.to_f64() * weight.to_f64();
        sum += product;
        sum_absolute += product.abs();
        sequential_f32 += x.to_f32() * weight.to_f32();
    }
    (sum, sum_absolute, sequential_f32)
}

fn ulps(actual: f16, expected: f16) -> Option<u32> {
    if !actual.is_finite() || !expected.is_finite() {
        return None;
    }
    let ordered = |value: f16| {
        let bits = u32::from(value.to_bits());
        if bits & 0x8000 != 0 {
            0x8000 - (bits & 0x7fff)
        } else {
            0x8000 + bits
        }
    };
    Some(ordered(actual).abs_diff(ordered(expected)))
}

fn report_points(
    stage: &str,
    f: Fixture,
    host: &HostFixture,
    input: &[f16],
    actual: &[f16],
    reference: &[f64],
    points: &[usize],
) -> bool {
    assert_eq!(actual.len(), reference.len());
    let mut calibrated = true;
    for &index in points {
        let (scalar, sum_absolute, sequential_f32) = scalar_point(f, host, input, index);
        let dgemm = reference[index];
        // Same independent F64-to-F64 crosscheck bound as the existing small
        // fixture. This is not a replacement for the terminal relative gate.
        let scalar_matches = (scalar - dgemm).abs() <= 1e-10 * (1.0 + scalar.abs());
        calibrated &= scalar_matches && scalar.is_finite() && dgemm.is_finite();
        let expected = f16::from_f64(dgemm);
        let error = (actual[index].to_f64() - expected.to_f64()).abs();
        println!(
            "{}",
            json!({
                "kind":"baseline_ffn_reference_point", "stage":stage,"index":index,
                "row":index/f.output_features,"column":index%f.output_features,
                "actual_f16_bits":actual[index].to_bits(),"reference_f64":dgemm,
                "reference_f16_bits":expected.to_bits(),"scalar_f64":scalar,
                "scalar_vs_dgemm_absolute":(scalar-dgemm).abs(),"scalar_matches_dgemm":scalar_matches,
                "sum_absolute_products":sum_absolute,
                "cancellation_ratio":(scalar!=0.0).then(||sum_absolute/scalar.abs()),
                "sequential_f32_diagnostic":sequential_f32,
                "sequential_f32_f16_bits":f16::from_f32(sequential_f32).to_bits(),
                "absolute_error_vs_f16_boundary":error,
                "relative_error_nonzero":(expected.to_f64()!=0.0).then(||error/expected.to_f64().abs()),
                "f16_ulps":ulps(actual[index],expected),
                "strict_relative_pass":actual[index].is_finite() && error<=RELATIVE_LIMIT*expected.to_f64().abs(),
                "selection":"prior worst indices and fixed endpoint controls; diagnostic only"
            })
        );
    }
    calibrated
}

fn report_activation_points(
    projection: &[f16],
    reference: &[f16],
    actual: &[f16],
    isolated: &[f16],
    intermediate: usize,
) {
    for index in [0, 5, 27, 1_646, 3_937, 202_277, 510_324, 557_055] {
        let (row, col) = (index / intermediate, index % intermediate);
        println!(
            "{}",
            json!({"kind":"baseline_ffn_silu_point","index":index,
            "independent_gate_f16_bits":projection[row*2*intermediate+col].to_bits(),
            "independent_up_f16_bits":projection[row*2*intermediate+intermediate+col].to_bits(),
            "reference_activation_f16_bits":reference[index].to_bits(),
            "complete_arm_activation_f16_bits":actual[index].to_bits(),
            "isolated_gpu_activation_f16_bits":isolated[index].to_bits(),
            "complete_arm_f16_ulps":ulps(actual[index],reference[index]),
            "isolated_gpu_f16_ulps":ulps(isolated[index],reference[index]),
            "cancellation_scale":"not a dot product; independent gate/up projection scales recorded separately"})
        );
    }
}

#[test]
#[ignore = "requires actual CUDA and pinned Marlin; original-arm reference attribution only"]
fn marlin_ct_original_ffn_reference_audit_on_cuda() {
    let gate = Fixture {
        name: "original-m32-reference-audit",
        rows: 32,
        input_features: 5120,
        output_features: 34816,
    };
    let down = Fixture {
        name: gate.name,
        rows: gate.rows,
        input_features: gate.output_features / 2,
        output_features: 5120,
    };
    let context = CudaContext::new(0).expect("reference audit requires CUDA");
    let stream = context.new_stream().unwrap();
    let blas = CudaBlas::new(Arc::clone(&stream)).unwrap();
    let runtime = MarlinProjectionRuntime {
        multiprocessor_count: context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap(),
        device_ordinal: 0,
    };
    let gate_host = source(gate);
    let down_host = build_host_fixture(down);
    let gate_weights = repack_source(gate, &gate_host);
    let down_weights = Repacked::new(
        &down_host.qweight_gptq,
        &down_host.scales_grouped,
        &down_host.zero_points_compressed_tensors,
        down.input_features,
        down.output_features,
    );
    let reference_peak = [gate, down]
        .iter()
        .map(|f| {
            8 * (f.input_features * f.output_features
                + f.rows * (f.input_features + f.output_features))
        })
        .max()
        .unwrap();
    let device_peak_bound = reference_peak
        + gate_weights[2].bytes()
        + down_weights.bytes()
        + 8 * gate.rows
            * (gate.input_features
                + gate.output_features
                + down.input_features
                + down.output_features)
        + runtime.workspace_bytes().unwrap() as usize
        + 1024 * 1024;
    assert!(
        device_peak_bound < DEVICE_BUDGET_BYTES - 256 * 1024 * 1024,
        "reference audit device budget"
    );
    let gate_up = Arm::new(
        &stream,
        runtime,
        gate,
        &gate_host.input,
        &gate_weights,
        false,
    );
    let down_projection = Projection::new(
        &stream,
        &down_weights,
        down.rows,
        down.output_features,
        gate_up.projections[0].pointers[4],
    );
    let arm = CompleteArm {
        gate_up,
        down: down_projection,
        fp32_reduction: None,
    };
    let _drain = Drain(Arc::clone(&stream));
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let planar = module.load_function(PLANAR_SILU_MUL_FUNCTION_NAME).unwrap();
    let merged = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    println!(
        "{}",
        json!({"kind":"baseline_ffn_reference_audit_setup",
        "geometry":[gate.rows,gate.input_features,down.input_features,down.output_features],
        "generations":1,"arm":"original separate gate/up","atomic_add":false,"fp32_reduce":false,
        "weight_reference":"RoundedF16Weight","relative_limit":RELATIVE_LIMIT,
        "gate_points":GATE_POINTS,"up_points":UP_POINTS,"down_points":DOWN_POINTS,
        "owned_device_peak_bound_bytes":device_peak_bound,
        "scope":"reference calibration and independent-input stage attribution; no candidate, no timing, no model qualification"})
    );
    arm.enqueue(runtime, &stream, gate, down, &planar, &merged);
    let eager = arm.read(&stream, gate);
    let graph = Graph::new(&stream, || {
        arm.enqueue(runtime, &stream, gate, down, &planar, &merged)
    });
    arm.refill(&stream, &gate_host.input, gate);
    graph.launch();
    let actual = arm.read(&stream, gate);
    assert!(same_bits(&actual.projection, &eager.projection));
    assert!(same_bits(&actual.activation, &eager.activation));
    assert!(same_bits(&actual.output, &eager.output));
    arm.guards(&stream);
    assert!(same_bits(
        &arm.gate_up.input.read(&stream),
        &gate_host.input
    ));

    let reference_projection = super::reference_projection(
        &stream,
        &blas,
        gate,
        &gate_host,
        &gate_host.input,
        PRIMARY_REFERENCE,
        false,
    );
    let reference_activation =
        super::reference_activation(&reference_projection, gate.rows, down.input_features);
    let reference_output = super::reference_projection(
        &stream,
        &blas,
        down,
        &down_host,
        &reference_activation,
        PRIMARY_REFERENCE,
        false,
    );
    let reference_projection16: Vec<_> = reference_projection
        .iter()
        .map(|v| f16::from_f64(*v))
        .collect();
    assert!(
        gate_host
            .input
            .iter()
            .chain(&reference_projection16)
            .chain(&reference_activation)
            .all(|v| v.is_finite()),
        "reference audit inputs and independent canonical boundaries must be finite"
    );
    let reference_output16: Vec<_> = reference_output
        .iter()
        .map(|v| f16::from_f64(*v).to_f64())
        .collect();
    let reference_activation64 = half_values(&reference_activation);
    let (projection_errors, projection_pass) =
        element_errors(&actual.projection, &half_values(&reference_projection16));
    let (activation_errors, activation_pass) =
        element_errors(&actual.activation, &reference_activation64);
    let (output_errors, output_pass) = element_errors(&actual.output, &reference_output16);
    let gate_points: Vec<_> = GATE_POINTS
        .iter()
        .map(|&i| i / down.input_features * gate.output_features + i % down.input_features)
        .collect();
    let up_points: Vec<_> = UP_POINTS
        .iter()
        .map(|&i| {
            i / down.input_features * gate.output_features
                + down.input_features
                + i % down.input_features
        })
        .collect();
    let gate_calibrated = report_points(
        "gate",
        gate,
        &gate_host,
        &gate_host.input,
        &actual.projection,
        &reference_projection,
        &gate_points,
    );
    let up_calibrated = report_points(
        "up",
        gate,
        &gate_host,
        &gate_host.input,
        &actual.projection,
        &reference_projection,
        &up_points,
    );
    let down_calibrated = report_points(
        "complete_output_vs_independent_chain",
        down,
        &down_host,
        &reference_activation,
        &actual.output,
        &reference_output,
        &DOWN_POINTS,
    );

    // Isolate the nonlinear kernel using only independent canonical projection
    // inputs. Neither this output nor any actual arm activation enters the oracle.
    let independent_halves = projection_halves(&reference_projection16, down.input_features);
    let independent_gate = Guarded::new(&stream, &independent_halves[0], f16::from_f32(-117.0));
    let independent_up = Guarded::new(&stream, &independent_halves[1], f16::from_f32(-117.0));
    let isolated_activation = Guarded::new(
        &stream,
        &vec![f16::NAN; reference_activation.len()],
        f16::from_f32(-117.0),
    );
    launch_planar_silu_mul(
        &stream,
        &planar,
        independent_gate.pointer(&stream),
        independent_up.pointer(&stream),
        isolated_activation.pointer(&stream),
        reference_activation.len() as u64,
    )
    .unwrap();
    let isolated_activation_values = isolated_activation.read(&stream);
    independent_gate.assert_unchanged(&stream);
    independent_up.assert_unchanged(&stream);
    let (isolated_silu_errors, isolated_silu_pass) =
        element_errors(&isolated_activation_values, &reference_activation64);
    report_activation_points(
        &reference_projection16,
        &reference_activation,
        &actual.activation,
        &isolated_activation_values,
        down.input_features,
    );
    // The unchanged original down projection sees the independent reference
    // activation, not the actual complete arm or isolated GPU SiLU output.
    let independent_activation =
        Guarded::new(&stream, &reference_activation, f16::from_f32(-117.0));
    arm.down.launch(
        runtime,
        &stream,
        independent_activation.pointer(&stream),
        down,
    );
    let isolated_down_values = arm.down.output.read(&stream);
    independent_activation.assert_unchanged(&stream);
    let (isolated_down_errors, isolated_down_pass) =
        element_errors(&isolated_down_values, &reference_output16);
    let isolated_down_calibrated = report_points(
        "isolated_down",
        down,
        &down_host,
        &reference_activation,
        &isolated_down_values,
        &reference_output,
        &DOWN_POINTS,
    );
    let all_finite = actual
        .projection
        .iter()
        .chain(&actual.activation)
        .chain(&actual.output)
        .chain(&isolated_activation_values)
        .chain(&isolated_down_values)
        .all(|v| v.is_finite());
    arm.guards(&stream);
    println!(
        "{}",
        json!({"kind":"baseline_ffn_reference_audit_complete",
        "original_projection":projection_errors,"original_activation":activation_errors,"original_terminal":output_errors,
        "original_all_stages_relative_pass":projection_pass&&activation_pass&&output_pass,
        "original_terminal_relative_pass":output_pass,
        "isolated_silu_from_independent_projection":isolated_silu_errors,"isolated_silu_relative_pass":isolated_silu_pass,
        "isolated_silu_kernel":"same original planar SiLU as complete arm",
        "isolated_down_from_independent_activation":isolated_down_errors,"isolated_down_relative_pass":isolated_down_pass,
        "selected_scalar_dgemm_crosschecks_pass":gate_calibrated&&up_calibrated&&down_calibrated&&isolated_down_calibrated,
        "all_actual_stages_finite":all_finite,"eager_graph_all_bits":true,"guards_and_inputs_preserved":true,
        "old_complete_qualification_unchanged":true,
        "test_pass_scope":"only independent selected F64 reference crosschecks, finite inputs and execution/guards; numerical qualification results above are not overridden",
        "model_quality_or_performance_claim":false})
    );
    // This diagnostic must still succeed in reporting a rejected original
    // numerical qualification. The two old selectors retain their strict assert.
    assert!(
        all_finite
            && gate_calibrated
            && up_calibrated
            && down_calibrated
            && isolated_down_calibrated,
        "finite-stage or independent scalar/DGEMM reference crosscheck failed"
    );
    drop(graph);
}
