//! Stage 15: fused MoE GPTQ INT4 × FP16 GEMM via Triton-rs PTX.
//!
//! Single launch processes ALL `(token, expert)` pairs of one MoE phase:
//! `expert_ids[pid_m]` selects which expert tile this 16-row block uses;
//! `sorted_token_ids[pid_m * BM .. ]` gathers M-axis input rows from
//! `A`. Padding/sentinel rows are masked at load + store.
//!
//! Stacked weight layout (raw GPTQ, NOT marlin-repacked):
//!   - qweight: `[E, K/8, N]` int32, per-expert stride = `(K/8) * N` int32
//!   - scales:  `[E, K/G, N]` fp16,  per-expert stride = `(K/G) * N` fp16
//!   - qzeros:  `[E, K/G, N/8]` int32, per-expert stride = `(K/G) * (N/8)` int32
//!
//! Caller builds `sorted_token_ids` + `expert_ids` via `B::moe_align_block_size`
//! (ferrum's existing kernel from Stage 10 / PR #94) and passes them
//! directly. With our pre-gathered A from Stage 12.1 path, the input
//! "size_m" is `total_pairs` and `top_k=1` — sorted_token_ids[i] = i for
//! lanes pointing at real rows, sentinel `total_pairs` for padding.

#[cfg(feature = "triton-kernels")]
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
#[cfg(feature = "triton-kernels")]
use std::sync::Arc;

#[cfg(feature = "triton-kernels")]
use crate::triton_meta::parse_meta;
#[cfg(feature = "triton-kernels")]
use crate::triton_ptx::fused_moe_w4a16_f16_bm16;

/// Per-tile shape. MUST match `<BM, BN, BK>` template params used to
/// generate the PTX (`fused_moe_w4a16_typed::<f16, 16, 64, 32>` in
/// `crates/triton-dsl/examples/ferrum_fused_moe_w4a16.rs`).
pub const BM: i32 = 16;
pub const BN: i32 = 64;
#[cfg_attr(not(feature = "triton-kernels"), allow(dead_code))]
pub const BK: i32 = 32;

/// PTX text for the kernel. Loaded via `cudarc::nvrtc::Ptx::from_src`.
#[cfg(feature = "triton-kernels")]
pub const FUSED_MOE_W4A16_PTX: &str = fused_moe_w4a16_f16_bm16::PTX;

/// Kernel function name in the PTX module — must match `tt.func` name
/// emitted by the DSL macro on `fused_moe_w4a16_typed`.
pub fn fn_name() -> &'static str {
    "fused_moe_w4a16_typed"
}

/// Stacked GPTQ INT4 weight in raw on-disk layout (no Marlin repack).
/// Sized for `num_experts` experts of identical shape `(K, N)`.
#[cfg(feature = "triton-kernels")]
pub struct TritonStackedGptqWeight {
    /// `[E * (K/8) * N]` i32, contiguous per-expert tile.
    pub qweight: CudaSlice<i32>,
    /// `[E * (K/G) * N]` f16.
    pub scales: CudaSlice<half::f16>,
    /// `[E * (K/G) * (N/8)]` i32.
    pub qzeros: CudaSlice<i32>,
    pub num_experts: usize,
    pub k: usize,
    pub n: usize,
    pub group_size: i32,
}

/// Build a `TritonStackedGptqWeight` from per-expert raw GPTQ tensors.
///
/// Skips Marlin repack — the Triton kernel reads the on-disk layout
/// directly. All experts MUST share `(k, n_per_expert, group_size, bits)`.
///
/// CPU side:
///   - qweights[e]: `[K/8, n_per_expert]` i32
///   - scales[e]:   `[K/G, n_per_expert]` f32 (caller's GPTQ loader format)
///   - qzeros[e]:   `[K/G, n_per_expert/8]` i32
///
/// Output buffers are concatenated per-expert with NO permutation —
/// expert e's tile is at offset `e * per_expert_stride`.
#[cfg(feature = "triton-kernels")]
pub fn load_stacked_gptq_raw(
    stream: &Arc<CudaStream>,
    qweights: &[&[i32]],
    scales_f32: &[&[f32]],
    qzeros: &[&[i32]],
    bits: u32,
    group_size: usize,
    k: usize,
    n_per_expert: usize,
) -> candle_core::Result<TritonStackedGptqWeight> {
    if bits != 4 {
        return Err(candle_core::Error::Msg(format!(
            "TritonStackedGptqWeight: only bits=4 supported (got {bits})"
        )));
    }
    let num_experts = qweights.len();
    if num_experts == 0 || scales_f32.len() != num_experts || qzeros.len() != num_experts {
        return Err(candle_core::Error::Msg(format!(
            "TritonStackedGptqWeight: shape mismatch qw={} sc={} qz={}",
            num_experts,
            scales_f32.len(),
            qzeros.len()
        )));
    }

    let qw_per = (k / 8) * n_per_expert;
    let groups = k / group_size;
    let sc_per = groups * n_per_expert;
    let qz_per = groups * (n_per_expert / 8);

    // Validate per-expert sizes.
    for (e, qw) in qweights.iter().enumerate() {
        if qw.len() != qw_per {
            return Err(candle_core::Error::Msg(format!(
                "TritonStacked: qweight[{e}].len()={} expected {qw_per}",
                qw.len()
            )));
        }
    }
    for (e, sc) in scales_f32.iter().enumerate() {
        if sc.len() != sc_per {
            return Err(candle_core::Error::Msg(format!(
                "TritonStacked: scales[{e}].len()={} expected {sc_per}",
                sc.len()
            )));
        }
    }
    for (e, qz) in qzeros.iter().enumerate() {
        if qz.len() != qz_per {
            return Err(candle_core::Error::Msg(format!(
                "TritonStacked: qzeros[{e}].len()={} expected {qz_per}",
                qz.len()
            )));
        }
    }

    // Concat host-side, then one h2d copy each.
    let mut qw_flat: Vec<i32> = Vec::with_capacity(num_experts * qw_per);
    for qw in qweights {
        qw_flat.extend_from_slice(qw);
    }
    let mut sc_flat_f16: Vec<half::f16> = Vec::with_capacity(num_experts * sc_per);
    for sc in scales_f32 {
        sc_flat_f16.extend(sc.iter().map(|&x| half::f16::from_f32(x)));
    }
    let mut qz_flat: Vec<i32> = Vec::with_capacity(num_experts * qz_per);
    for qz in qzeros {
        qz_flat.extend_from_slice(qz);
    }

    let qw_dev = stream
        .clone_htod(&qw_flat)
        .map_err(|e| candle_core::Error::Msg(format!("triton qw htod: {e}")))?;
    let sc_dev = stream
        .clone_htod(&sc_flat_f16)
        .map_err(|e| candle_core::Error::Msg(format!("triton sc htod: {e}")))?;
    let qz_dev = stream
        .clone_htod(&qz_flat)
        .map_err(|e| candle_core::Error::Msg(format!("triton qz htod: {e}")))?;

    Ok(TritonStackedGptqWeight {
        qweight: qw_dev,
        scales: sc_dev,
        qzeros: qz_dev,
        num_experts,
        k,
        n: n_per_expert,
        group_size: group_size as i32,
    })
}

/// Launch the fused MoE Triton kernel. ONE launch covers all experts.
///
/// `func` is a pre-loaded `CudaFunction` for `fused_moe_w4a16_typed`,
/// usually obtained via `CudaState::func` for caching.
///
/// `sorted_token_ids` length = `N_padded`; sentinel value = `size_m`.
/// `expert_ids` length = `N_padded / BM` — one expert id per 16-row tile.
/// `num_tokens_post_padded` is consumed only for grid-dim derivation
/// (we read it host-side from caller's bucket plan, kernel-side mask
/// derives from `m_idx < size_m`).
#[cfg(feature = "triton-kernels")]
#[allow(clippy::too_many_arguments)]
pub fn launch_fused_moe_w4a16_triton(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    input: &CudaSlice<half::f16>, // [size_m, K] fp16
    weight: &TritonStackedGptqWeight,
    output: &mut CudaSlice<half::f16>, // [size_m * top_k, N] fp16
    sorted_token_ids: &CudaSlice<i32>,
    expert_ids: &CudaSlice<i32>,
    num_padded_tokens: i32, // = sorted_token_ids.len() == N_padded
    size_m: i32,            // unique input rows (sentinel boundary)
) -> candle_core::Result<()> {
    let k = weight.k as i32;
    let n = weight.n as i32;
    let gs = weight.group_size;

    // Per-expert strides (int32 / fp16 elements, NOT bytes).
    let qw_per_expert = ((weight.k / 8) * weight.n) as i32;
    let groups = (weight.k as i32) / gs;
    let s_per_expert = (groups as i64 * weight.n as i64) as i32;
    let qz_per_expert = (groups * (weight.n as i32) / 8) as i32;

    // Row-major contiguous strides (matching the PTX kernel's expected layout).
    let stride_am = k;
    let stride_ak = 1i32;
    let stride_qwk = n;
    let stride_qwn = 1i32;
    let stride_sk = n;
    let stride_sn = 1i32;
    let stride_qzk = n / 8;
    let stride_qzn = 1i32;
    let stride_cm = n;
    let stride_cn = 1i32;

    // num_valid_tokens = size_m * top_k. With our pre-gathered path top_k=1.
    let num_valid_tokens = size_m;

    // Implicit Triton scratch buffers (1 byte each, zero-sized in practice).
    let global_scratch: CudaSlice<u8> = stream
        .alloc_zeros::<u8>(1)
        .map_err(|e| candle_core::Error::Msg(format!("triton fused_moe scratch: {e}")))?;
    let profile_scratch: CudaSlice<u8> = stream
        .alloc_zeros::<u8>(1)
        .map_err(|e| candle_core::Error::Msg(format!("triton fused_moe profile: {e}")))?;

    let inp = input.slice(..);
    let qw = weight.qweight.slice(..);
    let sc = weight.scales.slice(..);
    let qz = weight.qzeros.slice(..);
    let st = sorted_token_ids.slice(..);
    let eid = expert_ids.slice(..);

    let mut b = stream.launch_builder(func);
    b.arg(&inp);
    b.arg(&qw);
    b.arg(&sc);
    b.arg(&qz);
    b.arg(&st);
    b.arg(&eid);
    b.arg(output);
    b.arg(&num_valid_tokens);
    b.arg(&n);
    b.arg(&k);
    b.arg(&gs);
    b.arg(&qw_per_expert);
    b.arg(&s_per_expert);
    b.arg(&qz_per_expert);
    b.arg(&stride_am);
    b.arg(&stride_ak);
    b.arg(&stride_qwk);
    b.arg(&stride_qwn);
    b.arg(&stride_sk);
    b.arg(&stride_sn);
    b.arg(&stride_qzk);
    b.arg(&stride_qzn);
    b.arg(&stride_cm);
    b.arg(&stride_cn);
    b.arg(&global_scratch);
    b.arg(&profile_scratch);

    let lp = launch_params();
    let blocks_m = num_padded_tokens.div_ceil(BM) as u32;
    let blocks_n = (n + BN - 1) / BN;
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (blocks_m, blocks_n as u32, 1),
            block_dim: (lp.num_warps * 32, 1, 1),
            shared_mem_bytes: lp.shared_mem_bytes,
        })
    }
    .map_err(|e| candle_core::Error::Msg(format!("triton fused_moe launch: {e}")))?;
    Ok(())
}

#[cfg(feature = "triton-kernels")]
struct LaunchParams {
    num_warps: u32,
    shared_mem_bytes: u32,
}

#[cfg(feature = "triton-kernels")]
fn launch_params() -> LaunchParams {
    let meta = parse_meta(fused_moe_w4a16_f16_bm16::META).expect("parse fused_moe_w4a16 meta");
    LaunchParams {
        num_warps: meta.num_warps as u32,
        shared_mem_bytes: meta.shared_mem_bytes as u32,
    }
}
