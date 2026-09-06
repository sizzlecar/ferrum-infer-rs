//! Execute an explicitly required backend op with strict numerical evidence.
#[path = "backend_numerics/config.rs"]
mod config;
use config::{parse_args, Args, Config, Precision, USAGE};
use ferrum_testkit::op_diff::metal_context::{
    compare_submission_segments, MetalContextOp, SubmissionPhase, SUBMISSION_PHASES,
};
use ferrum_testkit::op_diff::required::{NumericalMetrics, RequiredReport, RequiredStatus};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{Seek, Write};
use std::process::ExitCode;
use std::time::Instant;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    submission_phases: Option<&'static [SubmissionPhase]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    submission_metrics: Option<Vec<NumericalMetrics>>,
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

fn bind_submission_segments(
    report: &mut RequiredReport,
    config: &Config,
) -> Option<Vec<NumericalMetrics>> {
    let config::Operation::MetalContext {
        tokens,
        intermediate,
        k,
    } = config.op
    else {
        return None;
    };
    let (Some(reference), Some(actual)) = (&report.reference, &report.actual) else {
        return None;
    };
    match compare_submission_segments(
        &MetalContextOp {
            tokens,
            intermediate,
            k,
        },
        &reference.to_f32(),
        &actual.to_f32(),
        config.max_nmse,
    ) {
        Ok(metrics) => Some(metrics),
        Err(error) => {
            report.status = RequiredStatus::Failed;
            report.reason = Some(match report.reason.take() {
                Some(previous) => format!("{previous}; {error}"),
                None => error,
            });
            None
        }
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
    let is_context = matches!(args.config.op, config::Operation::MetalContext { .. });
    let mut document = Document {
        schema_version: 1,
        output_shape: args.config.op.output_shape(),
        expected_output_elements: expected_elements,
        configured_precision: args.config.precision(),
        executed_precision: None,
        execution_path: args.config.op.execution_path(),
        coverage: if is_context {
            "Legacy MetalContext F32 compute/blit/compute, checked initial/reused/independent submissions and post-Drop data. Drop exposes no driver status. Excludes quantized kernels, production-plan dispatch and model performance."
        } else {
            "Only the selected Backend trait operation, shape and fixed adapter precision; excludes production-plan dispatch, quantized Marlin, paged attention, full models and performance claims"
        },
        submission_phases: is_context.then_some(&SUBMISSION_PHASES),
        submission_metrics: None,
        started_at: chrono::Utc::now().to_rfc3339(),
        completed: false,
        finished_at: None,
        execution_elapsed_ms: None,
        result: None,
        config: args.config,
    };
    write_document(&mut file, &document)?;
    let started = Instant::now();
    let mut result = document.config.execute();
    bind_output_shape(&mut result, expected_elements);
    document.submission_metrics = bind_submission_segments(&mut result, &document.config);
    document.execution_elapsed_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    if result.actual.is_some() {
        document.executed_precision = Some(document.config.precision());
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
#[path = "backend_numerics/tests.rs"]
mod tests;
