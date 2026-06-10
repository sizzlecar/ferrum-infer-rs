//! Cross-backend op-diff harness — PLAYBOOK § 3 L1.
//!
//! Runs the same op on CPU (reference) and on each available accelerator
//! (Metal / CUDA), then reports the **NMSE** (normalized mean-squared
//! error) of each accelerator's output relative to CPU's. Modelled on
//! llama.cpp's `tests/test-backend-ops.cpp` `NMSE = mse(a,b) / mse(a,0)`
//! comparison rather than naive max-abs-diff: NMSE is invariant to the
//! magnitude of the reference output, so a well-tuned kernel will sit
//! at the same NMSE regardless of input scaling.
//!
//! # Usage
//!
//! ```ignore
//! use ferrum_testkit::op_diff::{compare_backends, NMSE_FP16_TOL, rms_norm::RmsNormOp};
//!
//! let report = compare_backends(&RmsNormOp { tokens: 4, dim: 4096, eps: 1e-6 }, 42);
//! if let Some(nmse) = report.metal_nmse {
//!     assert!(nmse < NMSE_FP16_TOL, "metal rms_norm NMSE {nmse} exceeds fp16 tol");
//! }
//! ```
//!
//! # Tolerance buckets (PLAYBOOK § 3.1)
//!
//! - `NMSE_FP32_TOL = 1e-7` — fp32 kernels must agree with CPU below this.
//! - `NMSE_FP16_TOL = 1e-6` — fp16 kernels (Metal default storage).
//!
//! Tighter bucketing per op is welcome — define op-specific constants
//! once empirical baselines are stable.

pub mod fused_add_rms_norm;
pub mod gemm;
pub mod marlin_matmul; // stub — see file docs
pub mod paged_varlen_attn; // stub — see file docs
pub mod qk_norm_rope;
pub mod residual_add;
pub mod rms_norm;
pub mod silu_mul;

/// fp32 kernels — should agree with CPU below this.
pub const NMSE_FP32_TOL: f64 = 1e-7;
/// fp16 storage / Metal accumulation — slightly larger tol.
pub const NMSE_FP16_TOL: f64 = 1e-6;

/// Normalized mean-squared error.
///
/// NMSE = mse(a, b) / mse(a, 0). Returns the raw `mse(a, b)` when the
/// reference is degenerate (all zeros) — falls back gracefully so tests
/// for ops that legitimately output zero don't divide by zero.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
pub fn nmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "nmse: length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let mse_ab: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum::<f64>()
        / n;
    let mse_a0: f64 = a
        .iter()
        .map(|x| {
            let d = *x as f64;
            d * d
        })
        .sum::<f64>()
        / n;
    if mse_a0 < 1e-30 {
        return mse_ab;
    }
    mse_ab / mse_a0
}

/// Output of a single op invocation. Each backend produces its own
/// `Vec<f32>` after `to_vec()`-ing its buffer to host.
pub type Output = Vec<f32>;

/// A single op-under-test: knows how to run itself on each backend.
pub trait OpUnderTest {
    /// Display name (used in test failure messages).
    fn name(&self) -> &str;

    /// Run on CPU (reference). Always available.
    fn run_cpu(&self, seed: u64) -> Output;

    /// Run on Metal. Only available with `cfg(all(target_os = "macos", feature = "metal"))`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output;

    /// Run on CUDA. Only available with `cfg(feature = "cuda")`.
    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output;
}

/// Cross-backend comparison result.
///
/// `cpu` is the reference output. `metal_nmse` / `cuda_nmse` are `None`
/// on builds that don't include that backend.
#[derive(Debug)]
pub struct NmseReport {
    pub op: String,
    pub seed: u64,
    pub cpu: Vec<f32>,
    pub metal_nmse: Option<f64>,
    pub cuda_nmse: Option<f64>,
}

impl NmseReport {
    /// True if every available accelerator matches CPU below `tol`.
    pub fn within_tol(&self, tol: f64) -> bool {
        self.metal_nmse.map_or(true, |n| n < tol) && self.cuda_nmse.map_or(true, |n| n < tol)
    }
}

/// Run `op` on every backend the current build supports and assemble
/// the comparison report.
pub fn compare_backends(op: &dyn OpUnderTest, seed: u64) -> NmseReport {
    let cpu = op.run_cpu(seed);
    let metal_nmse = run_metal_nmse(op, &cpu, seed);
    let cuda_nmse = run_cuda_nmse(op, &cpu, seed);
    NmseReport {
        op: op.name().to_string(),
        seed,
        cpu,
        metal_nmse,
        cuda_nmse,
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn run_metal_nmse(op: &dyn OpUnderTest, cpu: &[f32], seed: u64) -> Option<f64> {
    Some(nmse(cpu, &op.run_metal(seed)))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn run_metal_nmse(_op: &dyn OpUnderTest, _cpu: &[f32], _seed: u64) -> Option<f64> {
    None
}

#[cfg(feature = "cuda")]
fn run_cuda_nmse(op: &dyn OpUnderTest, cpu: &[f32], seed: u64) -> Option<f64> {
    Some(nmse(cpu, &op.run_cuda(seed)))
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_nmse(_op: &dyn OpUnderTest, _cpu: &[f32], _seed: u64) -> Option<f64> {
    None
}

/// Convenience: deterministic uniform-random `Vec<f32>` in `[lo, hi)`.
pub fn random_vec(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(lo..hi)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nmse_identical_is_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(nmse(&a, &a) < 1e-30);
    }

    #[test]
    fn nmse_scaled_b_proportional() {
        // b = 1.01 * a → relative error 0.01, NMSE ≈ 0.0001
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = a.iter().map(|x| x * 1.01).collect();
        let n = nmse(&a, &b);
        // NMSE = mse(0.01*a, 0) / mse(a, 0) = 0.0001
        assert!((n - 1e-4).abs() < 1e-5);
    }

    #[test]
    fn nmse_zero_reference_falls_back() {
        // a all-zero: NMSE returns raw MSE.
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.1, 0.1, 0.1];
        let n = nmse(&a, &b);
        assert!((n - 0.01).abs() < 1e-9);
    }

    #[test]
    fn random_vec_determinism() {
        let a = random_vec(100, -1.0, 1.0, 42);
        let b = random_vec(100, -1.0, 1.0, 42);
        assert_eq!(a, b);
    }
}
