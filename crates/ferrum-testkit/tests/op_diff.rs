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
    compare_backends, fused_add_rms_norm::FusedAddRmsNormOp, gemm::GemmOp,
    qk_norm_rope::QkNormRopeOp, residual_add::ResidualAddOp, rms_norm::RmsNormOp,
    silu_mul::SiluMulOp, NMSE_FP16_TOL,
};

#[test]
fn residual_add_small_shape() {
    let op = ResidualAddOp { len: 4 * 256 };
    let report = compare_backends(&op, 7);
    assert_eq!(report.cpu.len(), op.len);
    assert!(
        report.cpu.iter().any(|&x| x != 0.0),
        "CPU reference output is all zeros — harness misconfigured"
    );
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn fused_add_rms_norm_small_shape() {
    let op = FusedAddRmsNormOp {
        tokens: 4,
        dim: 256,
        eps: 1e-6,
    };
    let report = compare_backends(&op, 31);
    // Output is [residual_after, out] concatenated.
    assert_eq!(report.cpu.len(), 2 * op.tokens * op.dim);
    assert!(report.cpu.iter().any(|&x| x != 0.0));
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

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

#[test]
fn gemm_small_shape() {
    // m × n × k = 4 × 128 × 128 — keep CPU reference fast.
    let op = GemmOp {
        m: 4,
        n: 128,
        k: 128,
    };
    let report = compare_backends(&op, 17);
    assert_eq!(report.cpu.len(), op.m * op.n);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn gemm_qkv_shape() {
    // Llama-3 8B QKV: m=tokens, n=hidden, k=hidden. Small token count
    // since CPU O(m*n*k) loop is cubic.
    let op = GemmOp {
        m: 2,
        n: 4096,
        k: 4096,
    };
    let report = compare_backends(&op, 31);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn qk_norm_rope_small_mode_0() {
    // Mode 0 = plain transpose; tests that the head-major output
    // matches across backends.
    let op = QkNormRopeOp {
        tokens: 4,
        heads: 8,
        head_dim: 64,
        pos_offset: 0,
        eps: 1e-5,
        mode: 0,
    };
    let report = compare_backends(&op, 51);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn qk_norm_rope_small_mode_1() {
    // Mode 1 = fused rms_norm + RoPE pairs + transpose.
    let op = QkNormRopeOp {
        tokens: 4,
        heads: 8,
        head_dim: 64,
        pos_offset: 0,
        eps: 1e-5,
        mode: 1,
    };
    let report = compare_backends(&op, 53);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn qk_norm_rope_llama_shape_with_offset() {
    // Llama-3 8B head_dim=128. Position offset matters for decode at
    // non-zero generation index.
    let op = QkNormRopeOp {
        tokens: 2,
        heads: 32,
        head_dim: 128,
        pos_offset: 64,
        eps: 1e-5,
        mode: 1,
    };
    let report = compare_backends(&op, 71);
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
