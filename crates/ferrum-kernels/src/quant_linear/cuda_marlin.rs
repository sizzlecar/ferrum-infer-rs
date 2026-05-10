//! `Linear<CudaBackend>` impls for GPTQ weights routed through Marlin.
//!
//! Two concrete types:
//! - [`CudaMarlinLinear`] — single-tensor projection (the GPTQ analogue
//!   of `DenseLinear`).
//! - [`CudaMarlinStackedExpertLinear`] — column-slice view into a stacked
//!   Marlin tile (the MoE analogue; one per expert).
//!
//! Each type's `forward()` body owns the kernel dispatch directly —
//! the old `BackendQuantMarlin::gemm_gptq` / `gemm_gptq_with_offset`
//! trait methods are now redundant and will be deleted in 3e/2.

use crate::backend::cuda::{CudaBackend, GptqStoreCuda};
use crate::Linear;
use cudarc::driver::CudaSlice;
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::sync::Arc;

/// Single-tensor GPTQ-Marlin Linear projection.
///
/// Holds a `GptqStoreCuda` (Marlin tiles, optionally Triton view) plus
/// an optional bias. `forward()` dispatches to the Marlin kernel via
/// `crate::backend::cuda::marlin_gemm_with_perm` (made `pub` in 3e/1)
/// or to the Triton w4a16 launcher when the store is `Triton(_)`.
pub struct CudaMarlinLinear {
    pub store: GptqStoreCuda,
    pub bias: Option<CudaSlice<f16>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear<CudaBackend> for CudaMarlinLinear {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    #[allow(clippy::needless_return)]
    fn forward(
        &self,
        ctx: &mut <CudaBackend as crate::backend::Backend>::Context,
        input: &<CudaBackend as crate::backend::Backend>::Buffer,
        out: &mut <CudaBackend as crate::backend::Backend>::Buffer,
        m: usize,
    ) {
        // Body migrated from `BackendQuantMarlin::gemm_gptq` impl on
        // CudaBackend (cuda.rs:3440). cfg branches preserved verbatim.
        let res: Result<()> = {
            #[cfg(feature = "marlin")]
            {
                #[cfg(feature = "triton-kernels")]
                {
                    match &self.store {
                        GptqStoreCuda::Marlin(mw) => {
                            crate::backend::cuda::marlin_gemm_with_perm(ctx, input, mw, out, m)
                        }
                        GptqStoreCuda::Triton(tw) => {
                            let func = ctx.func(
                                "triton_w4a16_gptq",
                                crate::triton_ptx::w4a16_gptq_f16::PTX,
                                crate::triton_w4a16::fn_name(),
                            );
                            let stream = ctx.stream.clone();
                            crate::triton_w4a16::launch_w4a16_gptq_triton(
                                &stream, &func, input, tw, out, m as i32,
                            )
                            .map_err(|e| FerrumError::model(format!("triton w4a16: {e}")))
                        }
                    }
                }
                #[cfg(not(feature = "triton-kernels"))]
                {
                    crate::backend::cuda::marlin_gemm_with_perm(ctx, input, &self.store, out, m)
                }
            }
            #[cfg(all(not(feature = "marlin"), feature = "triton-kernels"))]
            {
                match &self.store {
                    GptqStoreCuda::Marlin(_) => Err(FerrumError::unsupported(
                        "cargo feature `marlin` disabled — Marlin variant unusable; \
                         set FERRUM_TRITON_INT4=1 to force the triton path",
                    )),
                    GptqStoreCuda::Triton(tw) => {
                        let func = ctx.func(
                            "triton_w4a16_gptq",
                            crate::triton_ptx::w4a16_gptq_f16::PTX,
                            crate::triton_w4a16::fn_name(),
                        );
                        let stream = ctx.stream.clone();
                        crate::triton_w4a16::launch_w4a16_gptq_triton(
                            &stream, &func, input, tw, out, m as i32,
                        )
                        .map_err(|e| FerrumError::model(format!("triton w4a16: {e}")))
                    }
                }
            }
            #[cfg(all(not(feature = "marlin"), not(feature = "triton-kernels")))]
            {
                let _ = (ctx, input, out, m);
                Err(FerrumError::unsupported(
                    "cargo features `marlin` and `triton-kernels` both disabled — \
                     GPTQ not available",
                ))
            }
        };
        res.unwrap_or_else(|e| panic!("CudaMarlinLinear forward failed: {e}"));

        if let Some(bias) = &self.bias {
            <CudaBackend as crate::backend::Backend>::add_bias(
                ctx,
                out,
                bias,
                m,
                self.out_features,
            );
        }
    }
}

/// View into one expert's column slab of a shared stacked Marlin tile.
///
/// `store` is an `Arc` of the shared tile (one big repacked Marlin
/// tensor concatenating all experts' weights). `expert_offset` selects
/// which expert's columns to dispatch via `marlin_gemm_with_offset`.
pub struct CudaMarlinStackedExpertLinear {
    pub store: Arc<GptqStoreCuda>,
    pub expert_offset: usize,
    pub expert_n: usize,
    pub k: usize,
    pub bias: Option<CudaSlice<f16>>,
}

impl Linear<CudaBackend> for CudaMarlinStackedExpertLinear {
    fn in_features(&self) -> usize {
        self.k
    }

    fn out_features(&self) -> usize {
        self.expert_n
    }

    fn forward(
        &self,
        ctx: &mut <CudaBackend as crate::backend::Backend>::Context,
        input: &<CudaBackend as crate::backend::Backend>::Buffer,
        out: &mut <CudaBackend as crate::backend::Backend>::Buffer,
        m: usize,
    ) {
        // Body migrated from `BackendQuantMarlin::gemm_gptq_with_offset`
        // impl on CudaBackend (cuda.rs:3522).
        let res: Result<()> = {
            #[cfg(feature = "marlin")]
            {
                #[cfg(feature = "triton-kernels")]
                let mw = match self.store.as_ref() {
                    GptqStoreCuda::Marlin(mw) => mw,
                    GptqStoreCuda::Triton(_) => {
                        panic!(
                            "CudaMarlinStackedExpertLinear: Triton w4a16 store has no \
                             stride-aware variant; load MoE via Marlin (default)"
                        );
                    }
                };
                #[cfg(not(feature = "triton-kernels"))]
                let mw: &crate::marlin::MarlinWeight = self.store.as_ref();
                let stream = ctx.stream.clone();
                crate::marlin::marlin_gemm_with_offset(
                    &stream,
                    input,
                    mw,
                    out,
                    m as i32,
                    self.expert_offset as i32,
                    self.expert_n as i32,
                )
                .map_err(|e| FerrumError::model(format!("marlin offset gemm: {e}")))
            }
            #[cfg(not(feature = "marlin"))]
            {
                let _ = (ctx, input, out, m);
                Err(FerrumError::unsupported(
                    "cargo feature `marlin` disabled — \
                     CudaMarlinStackedExpertLinear unavailable",
                ))
            }
        };
        res.unwrap_or_else(|e| panic!("CudaMarlinStackedExpertLinear forward failed: {e}"));

        if let Some(bias) = &self.bias {
            <CudaBackend as crate::backend::Backend>::add_bias(ctx, out, bias, m, self.expert_n);
        }
    }
}
