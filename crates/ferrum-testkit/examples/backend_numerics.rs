//! Execute an explicitly required backend op with strict numerical evidence.
use ferrum_testkit::op_diff::{
    required::{run_required, RequiredBackend, RequiredReport, RequiredStatus},
    rms_norm::RmsNormOp,
    NMSE_FP16_TOL, NMSE_FP32_TOL,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "backend_numerics --require-backend metal|cuda --op rms-norm --report PATH \
    [--tokens N] [--dim N] [--eps VALUE] [--seed N] [--max-nmse VALUE]\n\
    Defaults: tokens=4, dim=128, eps=1e-6, seed=42; max-nmse=1e-7 for Metal F32, 1e-6 for CUDA F16.\n\
    The report must be a new file in an existing directory. NotRun and Failed exit nonzero.\n\
    This checks the selected Backend::rms_norm fixture, not all model/runtime/precision paths or performance.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Operation {
    RmsNorm,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Config {
    require_backend: RequiredBackend,
    op: Operation,
    tokens: usize,
    dim: usize,
    eps: f32,
    seed: u64,
    max_nmse: f64,
}

impl Config {
    fn rms_norm(&self) -> RmsNormOp {
        RmsNormOp {
            tokens: self.tokens,
            dim: self.dim,
            eps: self.eps,
        }
    }

    fn validate(&self) -> Result<usize, String> {
        if !self.max_nmse.is_finite() || self.max_nmse <= 0.0 {
            return Err("--max-nmse must be finite and strictly positive".into());
        }
        self.rms_norm().expected_output_len()
    }
}

#[derive(Debug, PartialEq)]
struct Args {
    config: Config,
    report: PathBuf,
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let arguments: Vec<_> = arguments.into_iter().collect();
    if arguments == ["--help"] || arguments == ["-h"] {
        return Ok(None);
    }
    let mut fields = BTreeMap::new();
    let mut arguments = arguments.into_iter();
    while let Some(key) = arguments.next() {
        if !matches!(
            key.as_str(),
            "--require-backend"
                | "--op"
                | "--report"
                | "--tokens"
                | "--dim"
                | "--eps"
                | "--seed"
                | "--max-nmse"
        ) {
            return Err(format!("unknown option {key:?}\n{USAGE}"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if fields.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option {key}"));
        }
    }
    let backend = match fields.remove("--require-backend").as_deref() {
        Some("metal") => RequiredBackend::Metal,
        Some("cuda") => RequiredBackend::Cuda,
        Some(other) => return Err(format!("unsupported required backend {other:?}")),
        None => {
            return Err(
                "--require-backend is required; CPU cannot substitute for an accelerator".into(),
            )
        }
    };
    let op = match fields.remove("--op").as_deref() {
        Some("rms-norm") => Operation::RmsNorm,
        Some(other) => return Err(format!("unsupported operator {other:?}")),
        None => return Err("--op is required".into()),
    };
    let report = fields.remove("--report").ok_or("--report is required")?;
    let max_nmse = match fields.remove("--max-nmse") {
        Some(value) => value.parse().map_err(|_| "--max-nmse must be a float")?,
        None => match backend {
            RequiredBackend::Metal => NMSE_FP32_TOL,
            RequiredBackend::Cuda => NMSE_FP16_TOL,
        },
    };
    let mut parse = |key: &str, default: &str| fields.remove(key).unwrap_or_else(|| default.into());
    let config = Config {
        require_backend: backend,
        op,
        tokens: parse("--tokens", "4")
            .parse()
            .map_err(|_| "--tokens must be a nonnegative integer")?,
        dim: parse("--dim", "128")
            .parse()
            .map_err(|_| "--dim must be a nonnegative integer")?,
        eps: parse("--eps", "1e-6")
            .parse()
            .map_err(|_| "--eps must be a float")?,
        seed: parse("--seed", "42")
            .parse()
            .map_err(|_| "--seed must be a nonnegative integer")?,
        max_nmse,
    };
    config.validate()?;
    Ok(Some(Args {
        config,
        report: report.into(),
    }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StoragePrecision {
    F32,
    F16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Precision {
    reference_storage: StoragePrecision,
    backend_input_storage: StoragePrecision,
    backend_output_storage: StoragePrecision,
    kernel_entrypoint: &'static str,
}

fn precision(backend: RequiredBackend) -> Precision {
    let (storage, kernel) = match backend {
        RequiredBackend::Metal => (StoragePrecision::F32, "rms_norm_f32"),
        RequiredBackend::Cuda => (StoragePrecision::F16, "rms_norm_f16"),
    };
    Precision {
        reference_storage: StoragePrecision::F32,
        backend_input_storage: storage,
        backend_output_storage: storage,
        kernel_entrypoint: kernel,
    }
}

#[derive(Serialize)]
struct Document {
    schema_version: u32,
    config: Config,
    output_shape: [usize; 2],
    expected_output_elements: usize,
    configured_precision: Precision,
    /// Filled only when this concrete production adapter returned actual data.
    /// Its fixed buffer construction, not a model name or probe, defines dtype.
    executed_precision: Option<Precision>,
    execution_path: &'static str,
    coverage: &'static str,
    started_at: String,
    completed: bool,
    finished_at: Option<String>,
    /// Probe, CPU reference, required accelerator, comparison and shape binding;
    /// excludes Cargo compilation and report I/O. This is not a kernel benchmark.
    execution_elapsed_ms: Option<f64>,
    result: Option<RequiredReport>,
}

fn bind_output_shape(report: &mut RequiredReport, expected_elements: usize) {
    let mut errors = Vec::new();
    for (label, output) in [("reference", &report.reference), ("actual", &report.actual)] {
        match output {
            Some(output) if output.f32_bits.len() != expected_elements => errors.push(format!(
                "{label} output has {} elements; configured shape requires {expected_elements}",
                output.f32_bits.len()
            )),
            None if report.status == RequiredStatus::Passed => {
                errors.push(format!("passed result is missing {label} output"))
            }
            _ => {}
        }
    }
    if !errors.is_empty() {
        if let Some(previous) = report.reason.take() {
            errors.insert(0, previous);
        }
        report.reason = Some(errors.join("; "));
        report.status = RequiredStatus::Failed;
    }
}

fn write_document(file: &mut File, document: &Document) -> Result<(), String> {
    // Serialize first, so a serialization error leaves the previous incomplete
    // record intact. A write failure still exits nonzero and cannot authorize use.
    let mut bytes = serde_json::to_vec_pretty(document)
        .map_err(|error| format!("serialize report: {error}"))?;
    bytes.push(b'\n');
    file.rewind()
        .map_err(|error| format!("seek report: {error}"))?;
    file.set_len(0)
        .map_err(|error| format!("truncate report: {error}"))?;
    file.write_all(&bytes)
        .map_err(|error| format!("write report: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("flush report: {error}"))
}

fn run(args: Args) -> Result<(), String> {
    let expected_elements = args.config.validate()?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report)
        .map_err(|error| format!("create report {}: {error}", args.report.display()))?;
    let mut document = Document {
        schema_version: 1, output_shape: [args.config.tokens, args.config.dim], expected_output_elements: expected_elements,
        configured_precision: precision(args.config.require_backend), executed_precision: None,
        execution_path: "Backend::rms_norm",
        coverage: "Only the configured RMSNorm shape and fixed adapter precision; excludes vNext attention/recurrent/MoE, quantization, full models and performance claims",
        started_at: chrono::Utc::now().to_rfc3339(), completed: false, finished_at: None,
        execution_elapsed_ms: None, result: None, config: args.config,
    };
    write_document(&mut file, &document)?;
    let started = Instant::now();
    let mut result = run_required(
        &document.config.rms_norm(),
        document.config.require_backend,
        document.config.seed,
        document.config.max_nmse,
    );
    bind_output_shape(&mut result, expected_elements);
    document.execution_elapsed_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    if result.actual.is_some() {
        document.executed_precision = Some(precision(document.config.require_backend));
    }
    document.completed = true;
    document.finished_at = Some(chrono::Utc::now().to_rfc3339());
    let passed = result.is_passed();
    let failure = result
        .reason
        .clone()
        .unwrap_or_else(|| format!("required backend result: {:?}", result.status));
    document.result = Some(result);
    write_document(&mut file, &document)?;
    if passed {
        println!(
            "Numerical check completed; report: {}",
            args.report.display()
        );
        Ok(())
    } else {
        Err(format!("{failure}; report: {}", args.report.display()))
    }
}

fn entry(arguments: impl IntoIterator<Item = String>) -> ExitCode {
    let result = match parse_args(arguments) {
        Ok(Some(args)) => run(args),
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(error) => Err(error),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("backend numerics: {error}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    entry(std::env::args().skip(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_testkit::op_diff::required::{compare_outputs, RawOutput};
    use std::fs;
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
            assert_eq!(
                precision(args.config.require_backend).backend_input_storage,
                storage
            );
        }
        assert!(parse_args(Vec::new()).is_err());
        assert!(parse_args(argv("cpu", "report.json")).is_err());
        let mut unsupported = argv("metal", "report.json");
        unsupported[3] = "gemm".into();
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

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn uncompiled_required_backend_writes_not_run_and_exits_nonzero() {
        let directory = TempDir::new();
        let path = directory.0.join("not-run.json");
        assert_eq!(
            entry(argv("cuda", path.to_str().unwrap())),
            ExitCode::FAILURE
        );
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(document["completed"], true);
        assert_eq!(document["result"]["status"], "not_run");
        assert!(document["result"]["reference"].is_null());
        assert!(document["result"]["actual"].is_null());
        assert!(document["executed_precision"].is_null());
        assert!(document["execution_elapsed_ms"].as_f64().unwrap() >= 0.0);
        assert_eq!(document["output_shape"], serde_json::json!([4, 128]));
    }
}
