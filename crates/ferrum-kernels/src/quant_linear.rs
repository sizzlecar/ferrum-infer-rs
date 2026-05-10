//! Concrete `Linear<B>` impls for quantized weights.
//!
//! Phase 3e moves the per-backend kernel-dispatch logic out of the
//! `BackendQuantMarlin` / `BackendQuantGguf` trait method bodies and
//! into these concrete `Linear<B>` types. Each impl owns the
//! cudarc / metal / cpu kernel call directly — no more `B::gemm_gptq`
//! indirection.
//!
//! Why these live in `ferrum-kernels` rather than `ferrum-quantization`:
//! the `forward()` body needs cudarc / metal-rs types, and pulling
//! those into ferrum-quantization would create a dep cycle (kernels
//! → quantization → kernels). ferrum-quantization stays as the
//! weight-format parser layer; backend-specific Linear impls live here.

#[cfg(feature = "cuda")]
pub mod cuda_marlin;
