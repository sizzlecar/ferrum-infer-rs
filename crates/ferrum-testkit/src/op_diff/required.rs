//! Strict checks for an explicitly required accelerator. Compilation and CPU
//! reference execution cannot substitute for the requested backend's output.
//!
//! The operator adapter remains responsible for checked device synchronization:
//! a Vec-returning trait cannot independently attest driver completion. This
//! module catches unwind failures; process aborts require the caller to reject
//! missing/incomplete reports. Neither an op label nor this report proves an
//! unexecuted shape, precision, execution path, model, or performance claim.
use super::{OpUnderTest, Output};
pub use ferrum_bench_core::release_regression::numerics::{
    compare_outputs, NumericalFailure, NumericalMetrics, RawOutput,
};
use ferrum_bench_core::release_regression::numerics::{
    validate_reference as valid_reference, validate_tolerance as valid_tolerance,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequiredBackend {
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequiredStatus {
    Passed,
    Failed,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredReport {
    pub schema_version: u32,
    pub op: String,
    pub backend: RequiredBackend,
    pub seed: u64,
    /// None only when the submitted tolerance was non-finite. Its original
    /// bits are always retained below so invalid inputs remain auditable.
    pub tolerance: Option<f64>,
    pub tolerance_f64_bits: u64,
    pub status: RequiredStatus,
    pub reason: Option<String>,
    pub reference: Option<RawOutput>,
    pub actual: Option<RawOutput>,
    pub metrics: Option<NumericalMetrics>,
}

impl RequiredReport {
    /// This convenience describes a freshly generated report. Consumers of
    /// stored reports must replay the raw data and verify execution provenance;
    /// deserializing a self-reported status is not a release authorization.
    pub fn is_passed(&self) -> bool {
        self.status == RequiredStatus::Passed
    }

    fn failed(&mut self, reason: impl Into<String>) {
        self.status = RequiredStatus::Failed;
        self.reason = Some(reason.into());
    }
}

#[derive(Debug)]
enum Availability {
    #[cfg_attr(
        not(any(test, feature = "cuda", all(target_os = "macos", feature = "metal"))),
        allow(dead_code)
    )]
    Available,
    Unavailable(String),
}

fn availability(backend: RequiredBackend) -> Result<Availability, String> {
    match backend {
        RequiredBackend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                if ferrum_kernels::attention::metal::is_available() {
                    Ok(Availability::Available)
                } else {
                    Ok(Availability::Unavailable(
                        "no Metal device is available".into(),
                    ))
                }
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Ok(Availability::Unavailable(
                    "Metal backend is not compiled for this platform".into(),
                ))
            }
        }
        RequiredBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match ferrum_kernels::cuda_device_count()? {
                    0 => Ok(Availability::Unavailable(
                        "no CUDA device is available".into(),
                    )),
                    _ => Ok(Availability::Available),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(Availability::Unavailable(
                    "CUDA backend is not compiled".into(),
                ))
            }
        }
    }
}

fn backend_output(op: &dyn OpUnderTest, backend: RequiredBackend, seed: u64) -> Option<Output> {
    // The branches deliberately dispatch only the explicitly required backend.
    // This does not call compare_backends, which runs every compiled accelerator.
    let _ = (op, seed);
    match backend {
        RequiredBackend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Some(op.run_metal(seed))
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                None
            }
        }
        RequiredBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Some(op.run_cuda(seed))
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        }
    }
}

fn panic_text(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).into()
    } else {
        "non-string panic payload".into()
    }
}

/// Run the CPU reference and exactly one requested backend. Uncompiled or
/// unavailable backends return NotRun; failed probes, driver/dispatch panics,
/// invalid outputs and excessive error return Failed. No fallback is a pass.
pub fn run_required(
    op: &dyn OpUnderTest,
    backend: RequiredBackend,
    seed: u64,
    tolerance: f64,
) -> RequiredReport {
    run_with(
        op.name(),
        backend,
        seed,
        tolerance,
        || availability(backend),
        || op.run_cpu(seed),
        || backend_output(op, backend, seed),
    )
}

fn run_with(
    op: &str,
    backend: RequiredBackend,
    seed: u64,
    tolerance: f64,
    probe: impl FnOnce() -> Result<Availability, String>,
    reference: impl FnOnce() -> Output,
    execute: impl FnOnce() -> Option<Output>,
) -> RequiredReport {
    let mut report = RequiredReport {
        schema_version: 1,
        op: op.into(),
        backend,
        seed,
        tolerance: tolerance.is_finite().then_some(tolerance),
        tolerance_f64_bits: tolerance.to_bits(),
        status: RequiredStatus::NotRun,
        reason: None,
        reference: None,
        actual: None,
        metrics: None,
    };
    if let Err(error) = valid_tolerance(tolerance) {
        report.failed(error.reason);
        return report;
    }
    match catch_unwind(AssertUnwindSafe(probe)) {
        Ok(Ok(Availability::Available)) => {}
        Ok(Ok(Availability::Unavailable(reason))) => {
            report.reason = Some(reason);
            return report;
        }
        Ok(Err(error)) => {
            report.failed(format!("backend availability query failed: {error}"));
            return report;
        }
        Err(payload) => {
            report.failed(format!(
                "backend availability query panicked: {}",
                panic_text(payload)
            ));
            return report;
        }
    }
    let reference = match catch_unwind(AssertUnwindSafe(reference)) {
        Ok(output) => output,
        Err(payload) => {
            report.failed(format!("CPU reference panicked: {}", panic_text(payload)));
            return report;
        }
    };
    report.reference = Some(RawOutput::from_f32(&reference));
    if let Err(error) = valid_reference(&reference) {
        report.failed(error.reason);
        return report;
    }
    let actual = match catch_unwind(AssertUnwindSafe(execute)) {
        Ok(Some(output)) => output,
        Ok(None) => {
            report.reason =
                Some("required backend was not executed; no output was produced".into());
            return report;
        }
        Err(payload) => {
            report.failed(format!(
                "required backend execution panicked: {}",
                panic_text(payload)
            ));
            return report;
        }
    };
    report.actual = Some(RawOutput::from_f32(&actual));
    match compare_outputs(&reference, &actual, tolerance) {
        Ok(metrics) => {
            report.metrics = Some(metrics);
            report.status = RequiredStatus::Passed;
        }
        Err(error) => {
            report.metrics = error.metrics;
            report.failed(error.reason);
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn missing_backend_does_not_even_run_the_reference() {
        let report = run_with(
            "fixture",
            RequiredBackend::Metal,
            7,
            1e-7,
            || Ok(Availability::Unavailable("no device".into())),
            || panic!("reference must not run"),
            || panic!("backend must not run"),
        );
        assert_eq!(report.status, RequiredStatus::NotRun);
        assert!(report.reference.is_none());
        assert!(report.actual.is_none());
        assert!(!report.is_passed());
    }

    #[test]
    fn cpu_reference_alone_cannot_pass_required_backend() {
        let report = run_with(
            "fixture",
            RequiredBackend::Cuda,
            7,
            1e-7,
            || Ok(Availability::Available),
            || vec![1.0, 2.0],
            || None,
        );
        assert_eq!(report.status, RequiredStatus::NotRun);
        assert_eq!(report.reference.unwrap().to_f32(), [1.0, 2.0]);
        assert!(report.actual.is_none());
        assert!(report.metrics.is_none());
    }

    #[test]
    fn failed_probe_and_panicking_execution_are_failed_not_not_run() {
        let probe = run_with(
            "fixture",
            RequiredBackend::Cuda,
            7,
            1e-7,
            || Err("driver query failed".into()),
            || panic!("no reference"),
            || panic!("no backend"),
        );
        assert_eq!(probe.status, RequiredStatus::Failed);
        let dispatch = run_with(
            "fixture",
            RequiredBackend::Metal,
            7,
            1e-7,
            || Ok(Availability::Available),
            || vec![1.0],
            || panic!("checked GPU completion failed"),
        );
        assert_eq!(dispatch.status, RequiredStatus::Failed);
        assert!(dispatch.reference.is_some());
        assert!(dispatch.actual.is_none());
        assert!(dispatch
            .reason
            .unwrap()
            .contains("checked GPU completion failed"));
    }

    #[test]
    fn runner_and_replay_share_the_same_comparator() {
        for actual in [vec![1.0, 2.0], vec![2.0, 4.0], vec![], vec![f32::NAN, 2.0]] {
            let result = run_with(
                "fixture",
                RequiredBackend::Metal,
                7,
                1e-7,
                || Ok(Availability::Available),
                || vec![1.0, 2.0],
                || Some(actual),
            );
            let reference = result.reference.as_ref().unwrap().to_f32();
            let actual = result.actual.as_ref().unwrap().to_f32();
            let replay = compare_outputs(
                &reference,
                &actual,
                f64::from_bits(result.tolerance_f64_bits),
            );
            assert_eq!(result.is_passed(), replay.is_ok());
            assert_eq!(
                result.metrics,
                match replay {
                    Ok(metrics) => Some(metrics),
                    Err(error) => error.metrics,
                }
            );
        }
    }

    #[test]
    fn invalid_reference_or_tolerance_prevents_backend_execution() {
        let dispatched = Cell::new(false);
        let result = run_with(
            "fixture",
            RequiredBackend::Metal,
            7,
            1e-7,
            || Ok(Availability::Available),
            || vec![],
            || {
                dispatched.set(true);
                Some(vec![])
            },
        );
        assert_eq!(result.status, RequiredStatus::Failed);
        assert!(!dispatched.get());
        let result = run_with(
            "fixture",
            RequiredBackend::Metal,
            7,
            f64::NAN,
            || panic!("invalid input must not probe"),
            || panic!("no reference"),
            || panic!("no backend"),
        );
        assert_eq!(result.status, RequiredStatus::Failed);
        assert_eq!(result.tolerance, None);
        assert_eq!(result.tolerance_f64_bits, f64::NAN.to_bits());
    }

    struct MustNotExecute;
    impl OpUnderTest for MustNotExecute {
        fn name(&self) -> &str {
            "fixture"
        }
        fn run_cpu(&self, _: u64) -> Output {
            panic!("CPU reference must not execute for an uncompiled backend")
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        fn run_metal(&self, _: u64) -> Output {
            panic!("unrequested Metal must not execute")
        }
        #[cfg(feature = "cuda")]
        fn run_cuda(&self, _: u64) -> Output {
            panic!("unrequested CUDA must not execute")
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    #[test]
    fn missing_metal_feature_is_not_run_through_public_api() {
        let result = run_required(&MustNotExecute, RequiredBackend::Metal, 7, 1e-7);
        assert_eq!(result.status, RequiredStatus::NotRun);
        assert!(result.actual.is_none());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn missing_cuda_feature_is_not_run_through_public_api() {
        let result = run_required(&MustNotExecute, RequiredBackend::Cuda, 7, 1e-7);
        assert_eq!(result.status, RequiredStatus::NotRun);
        assert!(result.actual.is_none());
    }
}
