use super::config::{Operation, StoragePrecision};
use super::*;
use ferrum_testkit::op_diff::required::{compare_outputs, RawOutput};
use ferrum_testkit::op_diff::{required::RequiredBackend, NMSE_FP16_TOL, NMSE_FP32_TOL};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrum-backend-numerics-{}-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn argv(backend: &str, report: &str) -> Vec<String> {
    [
        "--require-backend",
        backend,
        "--op",
        "rms-norm",
        "--report",
        report,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

#[test]
fn explicit_backend_and_operator_select_typed_precision_defaults() {
    for (backend, tolerance, storage) in [
        ("metal", NMSE_FP32_TOL, StoragePrecision::F32),
        ("cuda", NMSE_FP16_TOL, StoragePrecision::F16),
    ] {
        let args = parse_args(argv(backend, "report.json")).unwrap().unwrap();
        assert_eq!(args.config.max_nmse, tolerance);
        assert_eq!(args.config.validate().unwrap(), 4 * 128);
        assert_eq!(args.config.precision().backend_input_storage, storage);
    }
    assert!(parse_args(Vec::new()).is_err());
    assert!(parse_args(argv("cpu", "report.json")).is_err());
    let mut unsupported = argv("metal", "report.json");
    unsupported[3] = "quantized-marlin".into();
    assert!(parse_args(unsupported).is_err());
}

#[test]
fn malformed_arguments_do_not_start_execution() {
    for extra in [
        vec!["--tokens", "0"],
        vec!["--dim", "0"],
        vec!["--eps", "NaN"],
        vec!["--eps", "0"],
        vec!["--max-nmse", "inf"],
        vec!["--max-nmse", "-1"],
        vec!["--seed", "-1"],
        vec!["--unknown", "1"],
        vec!["--tokens"],
        vec!["--dim", "--seed"],
        vec!["--op", "rms-norm"],
    ] {
        let mut args = argv("metal", "must-not-exist.json");
        args.extend(extra.into_iter().map(str::to_owned));
        assert!(parse_args(args).is_err());
    }
    let mut args = argv("metal", "must-not-exist.json");
    args.extend([
        "--tokens".into(),
        usize::MAX.to_string(),
        "--dim".into(),
        "2".into(),
    ]);
    assert!(parse_args(args).is_err());
    assert_eq!(
        entry(["--require-backend".into(), "metal".into()]),
        ExitCode::FAILURE
    );
}

#[test]
fn explicit_tolerance_and_tail_shape_are_preserved() {
    let mut args = argv("metal", "report.json");
    args.extend(
        [
            "--tokens",
            "3",
            "--dim",
            "33",
            "--eps",
            "1e-5",
            "--seed",
            "9",
            "--max-nmse",
            "2e-8",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    let args = parse_args(args).unwrap().unwrap();
    assert_eq!(args.config.validate().unwrap(), 99);
    assert_eq!(args.config.seed, 9);
    assert_eq!(args.config.max_nmse, 2e-8);
}

#[test]
fn shape_binding_rejects_equally_truncated_outputs_after_numerical_success() {
    let reference = vec![1.0];
    let actual = vec![1.0];
    let mut report = RequiredReport {
        schema_version: 1,
        op: "rms_norm".into(),
        backend: RequiredBackend::Metal,
        seed: 7,
        tolerance: Some(1e-7),
        tolerance_f64_bits: 1e-7_f64.to_bits(),
        status: RequiredStatus::Passed,
        reason: None,
        reference: Some(RawOutput::from_f32(&reference)),
        actual: Some(RawOutput::from_f32(&actual)),
        metrics: Some(compare_outputs(&reference, &actual, 1e-7).unwrap()),
    };
    bind_output_shape(&mut report, 1);
    assert!(report.is_passed());
    bind_output_shape(&mut report, 2);
    assert_eq!(report.status, RequiredStatus::Failed);
    assert!(report
        .reason
        .unwrap()
        .contains("configured shape requires 2"));
}

#[test]
fn report_creation_failure_exits_nonzero_and_preserves_existing_evidence() {
    let directory = TempDir::new();
    let path = directory.0.join("existing.json");
    fs::write(&path, "existing evidence").unwrap();
    assert_eq!(
        entry(argv("metal", path.to_str().unwrap())),
        ExitCode::FAILURE
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "existing evidence");
    let missing_parent = directory.0.join("absent/report.json");
    assert_eq!(
        entry(argv("metal", missing_parent.to_str().unwrap())),
        ExitCode::FAILURE
    );
}

#[test]
fn new_operations_bind_shape_entrypoint_and_reject_irrelevant_flags() {
    for (op, shape_args, shape, metal_entry, cuda_entry) in [
        (
            "gemm",
            vec!["--m", "64", "--n", "33", "--k", "35"],
            [64, 33],
            "gemm_f32_v2",
            "cublasGemmEx(CUBLAS_COMPUTE_32F_FAST_16F)",
        ),
        (
            "gemm",
            vec!["--m", "1", "--n", "3", "--k", "5"],
            [1, 3],
            "gemv_f32",
            "cublasGemmEx(CUBLAS_COMPUTE_32F_FAST_16F)",
        ),
        (
            "silu-mul",
            vec!["--tokens", "3", "--intermediate", "257"],
            [3, 257],
            "silu_mul_split_f32",
            "fused_silu_mul_interleaved_f16",
        ),
    ] {
        for (backend, entrypoint) in [("metal", metal_entry), ("cuda", cuda_entry)] {
            let mut arguments = argv(backend, "report.json");
            arguments[3] = op.into();
            arguments.extend(shape_args.iter().map(|value| (*value).to_owned()));
            let args = parse_args(arguments).unwrap().unwrap();
            assert_eq!(args.config.op.output_shape(), shape);
            assert_eq!(args.config.validate().unwrap(), shape[0] * shape[1]);
            assert_eq!(args.config.precision().kernel_entrypoint, entrypoint);
            let json = serde_json::to_value(&args.config).unwrap();
            assert_eq!(json["op"], op);
            assert!(json.get("eps").is_none());
        }
    }
    for (op, option, value) in [
        ("gemm", "--tokens", "4"),
        ("gemm", "--eps", "1e-6"),
        ("silu-mul", "--dim", "128"),
        ("silu-mul", "--m", "64"),
        ("rms-norm", "--intermediate", "256"),
        ("gemm", "--m", "0"),
        ("gemm", "--n", "0"),
        ("gemm", "--k", "0"),
        ("silu-mul", "--tokens", "0"),
        ("silu-mul", "--intermediate", "0"),
    ] {
        let mut arguments = argv("metal", "must-not-exist.json");
        arguments[3] = op.into();
        arguments.extend([option.into(), value.into()]);
        assert!(parse_args(arguments).is_err(), "{op}: {option}={value}");
    }
    for (op, dimension) in [("gemm", "--k"), ("silu-mul", "--intermediate")] {
        let mut arguments = argv("metal", "must-not-exist.json");
        arguments[3] = op.into();
        arguments.extend([dimension.into(), usize::MAX.to_string()]);
        assert!(parse_args(arguments).is_err());
    }
}

#[test]
fn rms_norm_configuration_keeps_the_original_public_shape() {
    let args = parse_args(argv("metal", "report.json")).unwrap().unwrap();
    assert_eq!(
        args.config.op,
        Operation::RmsNorm {
            tokens: 4,
            dim: 128,
            eps: 1e-6
        }
    );
    assert_eq!(
        serde_json::to_value(&args.config).unwrap(),
        serde_json::json!({
            "require_backend": "metal", "op": "rms-norm", "tokens": 4,
            "dim": 128, "eps": 1e-6_f32, "seed": 42, "max_nmse": NMSE_FP32_TOL
        })
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn each_uncompiled_required_operation_writes_not_run_and_exits_nonzero() {
    let directory = TempDir::new();
    for (op, report_op, shape) in [
        ("rms-norm", "rms_norm", [4, 128]),
        ("gemm", "gemm", [64, 32]),
        ("silu-mul", "fused_silu_mul", [4, 256]),
    ] {
        let path = directory.0.join(format!("{op}.json"));
        let mut arguments = argv("cuda", path.to_str().unwrap());
        arguments[3] = op.into();
        assert_eq!(entry(arguments), ExitCode::FAILURE);
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(document["completed"], true);
        assert_eq!(document["result"]["op"], report_op);
        assert_eq!(document["result"]["status"], "not_run");
        assert!(document["result"]["reference"].is_null());
        assert!(document["result"]["actual"].is_null());
        assert!(document["executed_precision"].is_null());
        assert!(document["execution_elapsed_ms"].as_f64().unwrap() >= 0.0);
        assert_eq!(document["output_shape"], serde_json::json!(shape));
    }
}

#[test]
fn metal_context_requires_metal_and_records_lifecycle_output_segments() {
    let mut arguments = argv("metal", "report.json");
    arguments[3] = "metal-context".into();
    let args = parse_args(arguments.clone()).unwrap().unwrap();
    assert_eq!(
        args.config.op.output_shape(),
        [SUBMISSION_PHASES.len(), 3 * 33]
    );
    assert_eq!(
        args.config.validate().unwrap(),
        SUBMISSION_PHASES.len() * 3 * 33
    );
    assert_eq!(
        args.config.precision().backend_input_storage,
        StoragePrecision::F32
    );
    assert!(args.config.op.execution_path().contains("MetalContext"));
    arguments[1] = "cuda".into();
    assert!(parse_args(arguments)
        .unwrap_err()
        .contains("requires the Metal backend"));
    for (flag, value) in [
        ("--tokens", "0"),
        ("--intermediate", "0"),
        ("--k", "0"),
        ("--dim", "33"),
        ("--eps", "1e-6"),
    ] {
        let mut arguments = argv("metal", "report.json");
        arguments[3] = "metal-context".into();
        arguments.extend([flag.into(), value.into()]);
        assert!(parse_args(arguments).is_err());
    }
}

#[test]
fn context_segment_failure_overrides_a_successful_aggregate_report() {
    let reference = [1e8, 1.0, 2.0, 3.0];
    let actual = [1e8, 0.0, 2.0, 3.0];
    let config = Config {
        require_backend: RequiredBackend::Metal,
        op: Operation::MetalContext {
            tokens: 1,
            intermediate: 1,
            k: 1,
        },
        seed: 7,
        max_nmse: NMSE_FP32_TOL,
    };
    let mut report = RequiredReport {
        schema_version: 1,
        op: "metal_context".into(),
        backend: RequiredBackend::Metal,
        seed: 7,
        tolerance: Some(NMSE_FP32_TOL),
        tolerance_f64_bits: NMSE_FP32_TOL.to_bits(),
        status: RequiredStatus::Passed,
        reason: None,
        reference: Some(RawOutput::from_f32(&reference)),
        actual: Some(RawOutput::from_f32(&actual)),
        metrics: Some(compare_outputs(&reference, &actual, NMSE_FP32_TOL).unwrap()),
    };
    bind_output_shape(&mut report, config.validate().unwrap());
    assert!(report.is_passed());
    assert!(bind_submission_segments(&mut report, &config).is_none());
    assert_eq!(report.status, RequiredStatus::Failed);
    assert!(report.reason.unwrap().contains("ReusedContext"));
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[test]
fn uncompiled_metal_context_writes_not_run_without_output_or_segment_metrics() {
    let directory = TempDir::new();
    let path = directory.0.join("metal-context.json");
    let mut arguments = argv("metal", path.to_str().unwrap());
    arguments[3] = "metal-context".into();
    assert_eq!(entry(arguments), ExitCode::FAILURE);
    let document: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(document["result"]["status"], "not_run");
    assert!(document["result"]["actual"].is_null());
    assert!(document["submission_metrics"].is_null());
    assert_eq!(
        document["submission_phases"],
        serde_json::json!([
            "initial",
            "reused_context",
            "independent_context",
            "drop_flush"
        ])
    );
}
