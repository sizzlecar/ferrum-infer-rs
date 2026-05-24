//! Integration tests for the op-diff NMSE harness (PLAYBOOK § 3.1).
//!
//! Runs each op-under-test on CPU plus every accelerator the current
//! build supports, then asserts cross-backend NMSE is below the
//! relevant tolerance bucket (fp32 / fp16).
//!
//! Run:
//!     cargo test -p ferrum-testkit                       # CPU sanity only
//!     cargo test -p ferrum-testkit --features metal     # adds Metal cells
//!     cargo test -p ferrum-testkit --features cuda      # adds CUDA cells
//!
//! NMSE never breaks the test on CPU-only builds — it just verifies the
//! harness still runs and produces a sensible reference output.

use ferrum_testkit::op_diff::{
    compare_backends, rms_norm::RmsNormOp, silu_mul::SiluMulOp, NMSE_FP16_TOL,
};

#[test]
fn rms_norm_small_shape() {
    let op = RmsNormOp {
        tokens: 4,
        dim: 128,
        eps: 1e-6,
    };
    let report = compare_backends(&op, 42);
    assert_eq!(report.cpu.len(), op.tokens * op.dim);
    assert!(
        report.cpu.iter().any(|&x| x != 0.0),
        "CPU reference output is all zeros — harness misconfigured"
    );
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn rms_norm_llama_shape() {
    // Matches Llama-3 8B hidden dim with small token count to keep
    // the CPU reference fast.
    let op = RmsNormOp {
        tokens: 4,
        dim: 4096,
        eps: 1e-5,
    };
    let report = compare_backends(&op, 123);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn silu_mul_small_shape() {
    let op = SiluMulOp {
        tokens: 4,
        intermediate: 256,
    };
    let report = compare_backends(&op, 99);
    assert_eq!(report.cpu.len(), op.tokens * op.intermediate);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn silu_mul_llama_shape() {
    // Llama-3 8B intermediate dim. Small token count for CPU-loop speed.
    let op = SiluMulOp {
        tokens: 2,
        intermediate: 14336,
    };
    let report = compare_backends(&op, 7);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

fn check_accelerator_tolerance(report: &ferrum_testkit::op_diff::NmseReport, tol: f64) {
    if let Some(n) = report.metal_nmse {
        assert!(
            n < tol,
            "{}: Metal NMSE {:.3e} exceeds fp16 tol {:.3e} (seed={})",
            report.op,
            n,
            tol,
            report.seed
        );
        eprintln!("  {} metal NMSE: {:.3e}", report.op, n);
    }
    if let Some(n) = report.cuda_nmse {
        assert!(
            n < tol,
            "{}: CUDA NMSE {:.3e} exceeds fp16 tol {:.3e} (seed={})",
            report.op,
            n,
            tol,
            report.seed
        );
        eprintln!("  {} cuda NMSE: {:.3e}", report.op, n);
    }
}
