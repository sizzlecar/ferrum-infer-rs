use super::*;
use serde_json::{json, Value};

fn config(tokens: usize, intermediate: usize, k: usize) -> SubmissionConfig {
    SubmissionConfig {
        tokens,
        intermediate,
        k,
        seed: 42,
        max_nmse: 1e-7,
    }
}

// A producer-shaped serialization fixture for the consumer, not hardware
// execution evidence. Both summaries are computed from the raw test arrays.
fn fixture(expected: &SubmissionConfig) -> Value {
    let len = expected.segment_len().unwrap();
    let reference: Vec<f32> = (0..len * SUBMISSION_PHASES.len())
        .map(|index| (index as f32 + 1.0) * 0.25)
        .collect();
    let actual: Vec<f32> = reference.iter().map(|value| value * 1.00001).collect();
    let precision = || Precision {
        reference_storage: "f32".into(),
        backend_input_storage: "f32".into(),
        backend_output_storage: "f32".into(),
        kernel_entrypoint: expected.kernel_entrypoint().into(),
    };
    serde_json::to_value(Document {
        schema_version: 1,
        config: ProbeConfig {
            require_backend: Backend::Metal,
            op: "metal-context".into(),
            tokens: expected.tokens,
            intermediate: expected.intermediate,
            k: expected.k,
            seed: expected.seed,
            max_nmse: expected.max_nmse,
        },
        output_shape: [SUBMISSION_PHASES.len(), len],
        expected_output_elements: reference.len(),
        configured_precision: precision(),
        executed_precision: Some(precision()),
        execution_path: PROBE_EXECUTION_PATH.into(),
        _coverage: "fixture, no hardware execution".into(),
        submission_phases: Some(SUBMISSION_PHASES.to_vec()),
        submission_metrics: Some(
            compare_submission_segments(len, &reference, &actual, expected.max_nmse).unwrap(),
        ),
        started_at: "2026-09-06T00:00:00Z".into(),
        completed: true,
        finished_at: Some("2026-09-06T00:00:01Z".into()),
        execution_elapsed_ms: Some(1000.0),
        result: Some(ProbeResult {
            schema_version: 1,
            op: "metal_context".into(),
            backend: Backend::Metal,
            seed: expected.seed,
            tolerance: Some(expected.max_nmse),
            tolerance_f64_bits: expected.max_nmse.to_bits(),
            status: Status::Passed,
            reason: None,
            reference: Some(RawOutput::from_f32(&reference)),
            actual: Some(RawOutput::from_f32(&actual)),
            metrics: Some(compare_outputs(&reference, &actual, expected.max_nmse).unwrap()),
        }),
    })
    .unwrap()
}

fn verify_value(value: &Value, expected: &SubmissionConfig) -> Result<VerifiedSubmission, String> {
    verify(&serde_json::to_vec(value).unwrap(), expected)
}

#[test]
fn replay_accepts_registered_shapes_and_only_returns_submission_scope() {
    for expected in [config(1, 3, 5), config(3, 7, 9)] {
        let result = verify_value(&fixture(&expected), &expected).unwrap();
        assert_eq!(result.checker_id(), SUBMISSION_CHECK_ID);
        assert_eq!(result.scope(), submission_scope());
        assert_eq!(result.config(), &expected);
        assert_eq!(result.metrics().len(), SUBMISSION_PHASES.len());
        assert!(result
            .metrics()
            .iter()
            .all(|metric| metric.element_count == expected.tokens * expected.intermediate));
        assert_eq!(result.elapsed_ms(), 1000.0);
    }
}

#[test]
fn missing_failed_or_unexecuted_reports_cannot_become_verified() {
    let expected = config(3, 7, 9);
    for (pointer, replacement) in [
        ("/completed", json!(false)),
        ("/finished_at", Value::Null),
        ("/execution_elapsed_ms", Value::Null),
        ("/execution_elapsed_ms", json!(-1.0)),
        ("/result", Value::Null),
        ("/result/status", json!("not_run")),
        ("/result/status", json!("failed")),
        ("/result/reason", json!("device failure")),
        ("/executed_precision", Value::Null),
        ("/result/reference", Value::Null),
        ("/result/actual", Value::Null),
        ("/submission_metrics", Value::Null),
    ] {
        let mut value = fixture(&expected);
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            verify_value(&value, &expected).is_err(),
            "accepted {pointer}"
        );
    }
    let mut value = fixture(&expected);
    value.as_object_mut().unwrap().remove("completed");
    assert!(verify_value(&value, &expected).is_err());
}

#[test]
fn other_backends_operations_and_unchecked_paths_cannot_satisfy_submission() {
    let expected = config(3, 7, 9);
    for (pointer, replacement) in [
        ("/config/require_backend", json!("cuda")),
        ("/result/backend", json!("cpu")),
        ("/config/op", json!("rms-norm")),
        ("/result/op", json!("rms_norm")),
        ("/execution_path", json!("production-plan-runtime")),
        ("/configured_precision/backend_input_storage", json!("f16")),
        ("/executed_precision/backend_output_storage", json!("f16")),
        ("/executed_precision/kernel_entrypoint", json!("gemv_f32")),
        ("/schema_version", json!(2)),
        ("/result/schema_version", json!(2)),
    ] {
        let mut value = fixture(&expected);
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            verify_value(&value, &expected).is_err(),
            "accepted {pointer}"
        );
    }
}

#[test]
fn declared_phase_order_and_full_output_shape_are_required() {
    let expected = config(3, 7, 9);
    for replacement in [
        Value::Null,
        json!(["initial", "reused_context", "independent_context"]),
        json!([
            "initial",
            "reused_context",
            "independent_context",
            "independent_context"
        ]),
        json!([
            "reused_context",
            "initial",
            "independent_context",
            "drop_flush"
        ]),
    ] {
        let mut value = fixture(&expected);
        value["submission_phases"] = replacement;
        assert!(verify_value(&value, &expected).is_err());
    }
    for (pointer, replacement) in [
        ("/output_shape", json!([4, 20])),
        ("/expected_output_elements", json!(20)),
        ("/result/actual/f32_bits", json!([0])),
        ("/result/reference/f32_bits", json!([])),
    ] {
        let mut value = fixture(&expected);
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            verify_value(&value, &expected).is_err(),
            "accepted {pointer}"
        );
    }
}

#[test]
fn report_cannot_change_registered_inputs_or_loosen_tolerance() {
    let expected = config(3, 7, 9);
    for (pointer, replacement) in [
        ("/config/tokens", json!(1)),
        ("/config/intermediate", json!(0)),
        ("/config/k", json!(11)),
        ("/config/seed", json!(43)),
        ("/result/seed", json!(43)),
        ("/config/max_nmse", json!(1e-3)),
        ("/result/tolerance", json!(1e-3)),
        ("/result/tolerance_f64_bits", json!((1e-3_f64).to_bits())),
    ] {
        let mut value = fixture(&expected);
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            verify_value(&value, &expected).is_err(),
            "accepted {pointer}"
        );
    }
    for max_nmse in [0.0, -1.0, 1e-6, f64::NAN, f64::INFINITY] {
        let mut bad_expected = expected.clone();
        bad_expected.max_nmse = max_nmse;
        assert!(verify_value(&fixture(&expected), &bad_expected).is_err());
    }
}

#[test]
fn passed_summary_cannot_hide_nonfinite_or_numerically_wrong_raw_output() {
    let expected = config(3, 7, 9);
    for bits in [
        f32::NAN.to_bits(),
        f32::INFINITY.to_bits(),
        1000.0_f32.to_bits(),
    ] {
        let mut value = fixture(&expected);
        value["result"]["actual"]["f32_bits"][0] = json!(bits);
        assert!(verify_value(&value, &expected).is_err());
    }
}

#[test]
fn aggregate_success_cannot_hide_a_failed_low_energy_phase() {
    let expected = config(1, 2, 3);
    let len = expected.segment_len().unwrap();
    let mut reference = vec![1000.0_f32; len * SUBMISSION_PHASES.len()];
    reference[..len].fill(0.001);
    let mut actual = reference.clone();
    actual[..len].fill(0.002);
    // Large correct phases dilute aggregate error, so only checking that
    // number (or a passed summary) would wrongly accept a failed initial phase.
    let overall = compare_outputs(&reference, &actual, expected.max_nmse).unwrap();
    assert!(compare_submission_segments(len, &reference, &actual, expected.max_nmse).is_err());
    let mut value = fixture(&expected);
    value["result"]["reference"] = json!(RawOutput::from_f32(&reference));
    value["result"]["actual"] = json!(RawOutput::from_f32(&actual));
    value["result"]["metrics"] = json!(overall);
    let error = verify_value(&value, &expected).unwrap_err();
    assert!(error.contains("Initial:"), "{error}");
}

#[test]
fn aggregate_and_phase_metrics_must_both_match_raw_arrays() {
    let expected = config(3, 7, 9);
    for (pointer, replacement) in [
        ("/result/metrics", Value::Null),
        ("/result/metrics/nmse", json!(0.0)),
        ("/result/metrics/reference_mse", json!(1.0)),
        ("/result/metrics/max_abs", json!(0.0)),
        ("/result/metrics/element_count", json!(1)),
        ("/submission_metrics/0/nmse", json!(0.0)),
        ("/submission_metrics/0/uses_absolute_mse", json!(true)),
    ] {
        let mut value = fixture(&expected);
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            verify_value(&value, &expected).is_err(),
            "accepted {pointer}"
        );
    }
    let mut value = fixture(&expected);
    value["submission_metrics"].as_array_mut().unwrap().pop();
    assert!(verify_value(&value, &expected).is_err());
}
