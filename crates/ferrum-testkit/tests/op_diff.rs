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
    argmax_rows::ArgmaxRowsOp, compare_backends, embedding_lookup::EmbeddingLookupOp,
    fused_add_rms_norm::FusedAddRmsNormOp, gemm::GemmOp, qk_norm_rope::QkNormRopeOp,
    residual_add::ResidualAddOp, rms_norm::RmsNormOp, silu_mul::SiluMulOp, split_qkv::SplitQkvOp,
    transpose_head_to_token::TransposeHeadToTokenOp, NMSE_FP16_TOL,
};

#[test]
fn argmax_rows_spiked() {
    let op = ArgmaxRowsOp { m: 8, n: 256 };
    let report = compare_backends(&op, 19);
    assert_eq!(report.cpu.len(), op.m);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn flash_attention_small_shape() {
    use ferrum_testkit::op_diff::flash_attention::FlashAttentionOp;
    let op = FlashAttentionOp {
        batch: 1,
        q_len: 8,
        kv_len: 8,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 32,
    };
    let report = compare_backends(&op, 41);
    assert_eq!(
        report.cpu.len(),
        op.batch * op.q_len * op.num_heads * op.head_dim
    );
    assert!(report.cpu.iter().any(|&x| x != 0.0));
    check_accelerator_tolerance(&report, 5e-3); // attention fp16 accumulation
}

#[test]
fn activation_bridge_roundtrip() {
    use ferrum_testkit::op_diff::activation_bridge::ActivationBridgeOp;
    let op = ActivationBridgeOp { len: 4 * 128 };
    let report = compare_backends(&op, 3);
    assert_eq!(report.cpu.len(), op.len);
    check_accelerator_tolerance(&report, 3e-3); // fp16 storage roundtrip
}

#[test]
fn kv_cache_append_small_shape() {
    use ferrum_testkit::op_diff::kv_cache_append::KvCacheAppendOp;
    let op = KvCacheAppendOp {
        nkv: 2,
        hd: 8,
        capacity: 16,
        cache_len: 2,
        new_tokens: 4,
    };
    let report = compare_backends(&op, 23);
    assert_eq!(report.cpu.len(), 2 * op.nkv * op.capacity * op.hd);
    // K/V is stored at the backend's cache dtype (f16 on Metal/CUDA); use the
    // fp16-storage tolerance, not the fp32 compute bucket.
    check_accelerator_tolerance(&report, 3e-3);
}

#[test]
fn embedding_lookup_small_shape() {
    let op = EmbeddingLookupOp {
        vocab: 64,
        dim: 128,
        tokens: 8,
    };
    let report = compare_backends(&op, 5);
    assert_eq!(report.cpu.len(), op.tokens * op.dim);
    assert!(report.cpu.iter().any(|&x| x != 0.0));
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn transpose_head_to_token_small_shape() {
    let op = TransposeHeadToTokenOp {
        tokens: 4,
        heads: 4,
        dim: 16,
    };
    let report = compare_backends(&op, 11);
    assert_eq!(report.cpu.len(), op.tokens * op.heads * op.dim);
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

#[test]
fn split_qkv_small_shape() {
    let op = SplitQkvOp {
        tokens: 4,
        q_dim: 64,
        kv_dim: 32,
    };
    let report = compare_backends(&op, 13);
    assert_eq!(report.cpu.len(), op.tokens * (op.q_dim + 2 * op.kv_dim));
    check_accelerator_tolerance(&report, NMSE_FP16_TOL);
}

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
