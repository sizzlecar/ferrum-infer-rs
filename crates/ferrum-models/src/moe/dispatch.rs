//! Expert dispatch — load per-layer expert weights from a GGUF file and run
//! the per-token MoE forward (top-K experts per token, weighted combine).
//!
//! Phase 2 ships a CPU-only implementation (`moe_forward_cpu`). The
//! algorithm is:
//!
//! ```text
//! for each token b in batch:
//!     route token b → (expert_ids[K], weights[K])
//!     out[b] = 0
//!     for each (expert_id, weight) pair:
//!         gate_up = experts.gate_up[expert_id].forward(x[b])     # [2*ffn]
//!         silu_mul = silu(gate_up[..ffn]) * gate_up[ffn..]       # [ffn]
//!         contribution = experts.down[expert_id].forward(silu_mul) # [hidden]
//!         out[b] += weight * contribution
//! ```
//!
//! The fused `gate || up` per-expert layout means we can call
//! `Backend::fused_silu_mul_split` directly on the projection's output
//! — same kernel ferrum already uses for dense Llama-family models.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::quantized::GgmlDType;
use candle_core::{Device, Result as CandleResult};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::{Backend, GgufQuantType};
use ferrum_kernels::Linear;
use ferrum_quantization::gguf::GgufFile;
use ferrum_quantization::{DenseLinear, QuantLinear};
use ferrum_types::{FerrumError, Result};

use crate::moe::router::RouterOutput;

/// MoE per-op timers. Public so the model wrapper can drain + print at
/// end of decode. Times are in microseconds, atomically accumulated.
/// Toggle via env `FERRUM_MOE_PROFILE=1`.
pub static MOE_SYNC_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_SYNC_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_GATE_UP_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_GATE_UP_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_SILU_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_SILU_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_DOWN_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_SCALED_ADD_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_SCALED_ADD_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_COPY_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_COPY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_HOST_TOPK_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_HOST_TOPK_CALLS: AtomicU64 = AtomicU64::new(0);

fn moe_profile_enabled() -> bool {
    std::env::var("FERRUM_MOE_PROFILE").is_ok()
}

/// Per-layer expert weights, materialised as `[num_experts]`-long vectors
/// of `Box<dyn Linear<B>>`. Each entry runs the corresponding expert's
/// fused `[gate; up]` projection or its `down` projection.
///
/// `B::Buffer` is hidden behind `Linear<B>` so this struct is generic
/// over backend, but Phase 2's only consumer (`moe_forward_cpu`) is CPU-
/// only — generic `moe_forward<B>` is deferred until the trait gains
/// scaled-accumulate + cheap buffer slicing.
pub struct ExpertStack<B: Backend> {
    /// Fused `[gate; up]` projection per expert. Output shape per token:
    /// `[2 * expert_intermediate]` — the lower half is gate, upper is up.
    pub gate_up: Vec<Box<dyn Linear<B>>>,
    /// `down` projection per expert. Output shape per token: `[hidden_size]`.
    pub down: Vec<Box<dyn Linear<B>>>,
    /// Stacked-experts representation for backends that have a batched
    /// MoE indirect-dispatch kernel (Metal `gemv_q4kw_moe_id_f32` /
    /// `gemv_q6kw_moe_id_f32`). Holds **all experts** for one matmul
    /// role in a single `B::QuantStore` with byte stride between expert
    /// slabs, so a single dispatch can cover all selected (token, expert)
    /// pairs at decode m=1.
    ///
    /// `None` on backends without the kernel (CPU, CUDA-without-MoE-kernel)
    /// and on quant flavours that don't have a stacked path yet — callers
    /// fall back to the per-expert `gate_up` / `down` Linears in those
    /// cases.
    pub gate_stacked: Option<B::QuantStore>,
    pub up_stacked: Option<B::QuantStore>,
    pub down_stacked: Option<B::QuantStore>,
}

impl<B: Backend> ExpertStack<B> {
    /// Build from raw fp32 stacked tensors (test helper). Caller has
    /// already dequantised and laid out the data:
    ///   `gate_stack`: `[num_experts * expert_inter * hidden]`
    ///   `up_stack`:   `[num_experts * expert_inter * hidden]`
    ///   `down_stack`: `[num_experts * hidden * expert_inter]`
    /// Each per-expert slice is row-major in the natural Linear shape.
    pub fn from_dense_stacks(
        gate_stack: &[f32],
        up_stack: &[f32],
        down_stack: &[f32],
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Self> {
        let gate_up_per_expert = expert_intermediate * hidden_size;
        let down_per_expert = hidden_size * expert_intermediate;

        check_size(
            gate_stack.len(),
            num_experts * gate_up_per_expert,
            "gate_stack",
        )?;
        check_size(up_stack.len(), num_experts * gate_up_per_expert, "up_stack")?;
        check_size(
            down_stack.len(),
            num_experts * down_per_expert,
            "down_stack",
        )?;

        let mut gate_up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let g_off = e * gate_up_per_expert;
            let g_slice = &gate_stack[g_off..g_off + gate_up_per_expert];
            let u_slice = &up_stack[g_off..g_off + gate_up_per_expert];

            // Fused [gate; up] is [2 * expert_inter, hidden] row-major.
            // We concatenate row-blocks so the first expert_inter rows are
            // gate, the next expert_inter rows are up — the layout
            // fused_silu_mul_split expects.
            let mut fused = Vec::with_capacity(2 * gate_up_per_expert);
            fused.extend_from_slice(g_slice);
            fused.extend_from_slice(u_slice);
            gate_up.push(Box::new(DenseLinear::<B>::from_rows(
                &fused,
                2 * expert_intermediate,
                hidden_size,
            )) as Box<dyn Linear<B>>);

            let d_off = e * down_per_expert;
            let d_slice = &down_stack[d_off..d_off + down_per_expert];
            down.push(Box::new(DenseLinear::<B>::from_rows(
                d_slice,
                hidden_size,
                expert_intermediate,
            )) as Box<dyn Linear<B>>);
        }
        Ok(Self {
            gate_up,
            down,
            gate_stacked: None,
            up_stacked: None,
            down_stacked: None,
        })
    }

    /// Load all experts for one MoE layer from a GGUF file. Names follow
    /// the GGUF convention: `blk.{layer_idx}.ffn_{gate,up,down}_exps.weight`.
    ///
    /// The loader picks between two strategies based on the on-disk dtype
    /// of the expert tensors:
    ///
    ///   - **Quantised path** (Q4_K / Q6_K only): each expert's
    ///     `gate || up` becomes a single `QuantLinear<B>` (Fused
    ///     QuantStore — gate + up share `n_cols = hidden`), and `down` is
    ///     a plain `QuantLinear<B>`. Block bytes stay compressed in
    ///     backend memory; per-call dequant happens inside `gemm_quant`.
    ///   - **Dense fallback** (everything else, e.g. F32 / F16 / Q5_K
    ///     until a kernel ships): eager-dequant to fp32 and wrap
    ///     `DenseLinear<B>`. Memory inflates ~7× vs Q4_K_M but the
    ///     algorithm is correctness-equivalent and this is the path the
    ///     synthetic-MoE test fixtures need.
    ///
    /// The runtime dispatcher (`moe_forward<B>`) doesn't see which path
    /// was taken — it just calls `Linear::forward` per (token, expert).
    pub fn load_from_gguf(
        gguf: &GgufFile,
        layer_idx: usize,
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Self> {
        if let Some(quant) = Self::try_load_quantised(
            gguf,
            layer_idx,
            num_experts,
            hidden_size,
            expert_intermediate,
        )? {
            if std::env::var("FERRUM_MOE_LOAD_TRACE").is_ok() {
                eprintln!("[moe-load] layer {layer_idx} → quantised expert path");
            }
            return Ok(quant);
        }

        if std::env::var("FERRUM_MOE_LOAD_TRACE").is_ok() {
            eprintln!("[moe-load] layer {layer_idx} → eager fp32 dense fallback ⚠");
        }

        let device = Device::Cpu;
        let gate = read_dequant_flat(
            gguf,
            &format!("blk.{layer_idx}.ffn_gate_exps.weight"),
            &device,
        )?;
        let up = read_dequant_flat(
            gguf,
            &format!("blk.{layer_idx}.ffn_up_exps.weight"),
            &device,
        )?;
        let down = read_dequant_flat(
            gguf,
            &format!("blk.{layer_idx}.ffn_down_exps.weight"),
            &device,
        )?;
        // Eager-dense path leaves stacked variants as None — no MoE
        // fast path for synthesised / non-quantised expert tensors.
        Self::from_dense_stacks(
            &gate,
            &up,
            &down,
            num_experts,
            hidden_size,
            expert_intermediate,
        )
    }

    /// Attempt the quantised path. Returns `Ok(None)` if any of the three
    /// tensors isn't a supported k-quant flavour (Q4_K / Q6_K) or if the
    /// shape doesn't match the expected per-expert tile size — caller
    /// then takes the eager-dequant fallback. Returns `Err` only on a
    /// genuine load failure (missing tensor, byte-count mismatch).
    fn try_load_quantised(
        gguf: &GgufFile,
        layer_idx: usize,
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Option<Self>> {
        let device = Device::Cpu;

        let gate_name = format!("blk.{layer_idx}.ffn_gate_exps.weight");
        let up_name = format!("blk.{layer_idx}.ffn_up_exps.weight");
        let down_name = format!("blk.{layer_idx}.ffn_down_exps.weight");

        // Inspect tensor info up front — if any tensor isn't a k-quant
        // flavour the backend can dispatch on, bail to the dense path
        // before paying the byte-read cost.
        let gate_kind = match quant_kind(gguf, &gate_name)? {
            Some(k) => k,
            None => return Ok(None),
        };
        let up_kind = match quant_kind(gguf, &up_name)? {
            Some(k) => k,
            None => return Ok(None),
        };
        let down_kind = match quant_kind(gguf, &down_name)? {
            Some(k) => k,
            None => return Ok(None),
        };

        // Read the three 3-D quantised tensors. `qt.data()` exposes the
        // raw block-byte payload — the same bytes that live on disk.
        let gate_qt = gguf
            .read_tensor(&gate_name, &device)
            .map_err(candle_to_ferrum)?;
        let up_qt = gguf
            .read_tensor(&up_name, &device)
            .map_err(candle_to_ferrum)?;
        let down_qt = gguf
            .read_tensor(&down_name, &device)
            .map_err(candle_to_ferrum)?;
        let gate_bytes = gate_qt.data().map_err(candle_to_ferrum)?;
        let up_bytes = up_qt.data().map_err(candle_to_ferrum)?;
        let down_bytes = down_qt.data().map_err(candle_to_ferrum)?;
        let gate_bytes = gate_bytes.as_ref();
        let up_bytes = up_bytes.as_ref();
        let down_bytes = down_bytes.as_ref();

        // Per-expert byte stride for each tensor. The 3-D layout is
        // contiguous, [num_experts, rows, cols] row-major, so each
        // expert's slab is exactly `total_bytes / num_experts`.
        let gate_per = block_bytes_for(
            gate_kind,
            expert_intermediate * hidden_size,
            "ffn_gate_exps",
        )?;
        let up_per = block_bytes_for(up_kind, expert_intermediate * hidden_size, "ffn_up_exps")?;
        let down_per = block_bytes_for(
            down_kind,
            hidden_size * expert_intermediate,
            "ffn_down_exps",
        )?;

        check_size(
            gate_bytes.len(),
            num_experts * gate_per,
            "ffn_gate_exps bytes",
        )?;
        check_size(up_bytes.len(), num_experts * up_per, "ffn_up_exps bytes")?;
        check_size(
            down_bytes.len(),
            num_experts * down_per,
            "ffn_down_exps bytes",
        )?;

        // Try the stacked-experts fast path FIRST. If the backend has a
        // batched MoE kernel (Metal `gemv_q*kw_moe_id_f32`), we want to
        // hold the experts only as one big stacked buffer per role —
        // not as 128 per-expert MetalQuantStores PLUS the stacked one
        // (that would double-allocate ~17 GB on a 32 GB Mac, which on
        // Qwen3-30B-A3B Q4_K_M sends the model into swap and tanks
        // both load and forward time).
        let gate_stacked = B::load_quant_experts(
            gate_kind,
            gate_bytes,
            num_experts,
            expert_intermediate,
            hidden_size,
        )
        .ok();
        let up_stacked = B::load_quant_experts(
            up_kind,
            up_bytes,
            num_experts,
            expert_intermediate,
            hidden_size,
        )
        .ok();
        let down_stacked = B::load_quant_experts(
            down_kind,
            down_bytes,
            num_experts,
            hidden_size,
            expert_intermediate,
        )
        .ok();

        // Decide the storage shape:
        //   * Stacked-only (Metal MoE fast path): all three stacked
        //     loaders succeeded — skip per-expert and use stacked
        //     for both decode and prefill. Cuts memory in half.
        //   * Per-expert: stacked path is incomplete or unsupported —
        //     load 128-per-layer QuantLinears and let `moe_forward`
        //     drive the per-(token, expert) loop on top of them.
        let stacked_complete =
            gate_stacked.is_some() && up_stacked.is_some() && down_stacked.is_some();

        let (gate_up, down) = if stacked_complete {
            // No per-expert needed — `moe_forward_stacked_decode_impl`
            // and the per-token prefill loop both use the stacked buffers.
            (Vec::new(), Vec::new())
        } else {
            let mut gate_up: Vec<Box<dyn Linear<B>>> = Vec::with_capacity(num_experts);
            let mut down: Vec<Box<dyn Linear<B>>> = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let g_slice = &gate_bytes[e * gate_per..(e + 1) * gate_per];
                let u_slice = &up_bytes[e * up_per..(e + 1) * up_per];
                let d_slice = &down_bytes[e * down_per..(e + 1) * down_per];

                let parts: [(GgufQuantType, &[u8], usize); 2] = [
                    (gate_kind, g_slice, expert_intermediate),
                    (up_kind, u_slice, expert_intermediate),
                ];
                let gate_up_e = match QuantLinear::<B>::from_gguf_fused(&parts, hidden_size) {
                    Ok(q) => q,
                    Err(_) => return Ok(None),
                };
                gate_up.push(Box::new(gate_up_e) as Box<dyn Linear<B>>);

                let down_e = match QuantLinear::<B>::from_gguf_bytes(
                    down_kind,
                    d_slice,
                    hidden_size,
                    expert_intermediate,
                ) {
                    Ok(q) => q,
                    Err(_) => return Ok(None),
                };
                down.push(Box::new(down_e) as Box<dyn Linear<B>>);
            }
            (gate_up, down)
        };

        Ok(Some(Self {
            gate_up,
            down,
            gate_stacked,
            up_stacked,
            down_stacked,
        }))
    }

    /// Convenience: open a GGUF and load layer `layer_idx`. The GGUF
    /// stays open inside this call only — for multi-layer loads use
    /// [`Self::load_from_gguf`] with a shared [`GgufFile`].
    pub fn open_and_load(
        path: impl AsRef<Path>,
        layer_idx: usize,
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Self> {
        let gguf = GgufFile::open(path).map_err(candle_to_ferrum)?;
        Self::load_from_gguf(
            &gguf,
            layer_idx,
            num_experts,
            hidden_size,
            expert_intermediate,
        )
    }

    /// `num_experts` for the layer (consistency check helper).
    ///
    /// Returns the per-expert Vec length, OR — when the stacked-only
    /// path is in effect (Metal MoE fast path with empty per-expert
    /// Vecs) — falls back to a stored count via the stacked variants.
    /// In the stacked-only case there's no Vec to count, so this method
    /// is mostly used by tests on the per-expert path.
    pub fn num_experts(&self) -> usize {
        debug_assert_eq!(
            self.gate_up.len(),
            self.down.len(),
            "ExpertStack: gate_up and down disagree on expert count"
        );
        self.gate_up.len()
    }
}

/// Backend-generic MoE forward.
///
/// Equivalent of [`moe_forward_cpu`] but parameterised on `B: Backend`
/// so Metal / CUDA paths can dispatch the same per-(token, expert) loop
/// using their own kernels for the gemv + silu + scaled-add primitives.
///
/// The caller pre-supplies all scratch buffers — this function does no
/// allocation, which matters because it's invoked from inside the
/// transformer's `forward_layer` where allocation during graph capture
/// (CUDA) would corrupt the captured graph.
///
/// Buffer contract (lengths, sized at scratch alloc time):
///   - `x`            : `[batch * hidden]` post-RMSNorm activations
///   - `router_logits`: `[batch * num_experts]` raw router output
///   - `out`          : `[batch * hidden]` — caller is responsible for
///                      zeroing this before the call (we accumulate,
///                      not assign)
///   - `x_single`     : `[hidden]` per-token input slice
///   - `acc_buf`      : `[hidden]` per-token output accumulator (kept
///                      separate from `x_single` so the gate_up gemv
///                      can consume `x_single` repeatedly across the
///                      top_k loop without an inter-pair restore)
///   - `gate_up_buf`  : `[2 * expert_inter]` per-(token, expert) gemv out
///   - `silu_buf`     : `[expert_inter]`
///   - `down_buf`     : `[hidden]` per-(token, expert) accumulate src
///
/// Routing (softmax + top-K + optional renorm) runs host-side using
/// `B::to_vec(router_logits, …)` — the routing computation is small
/// (`batch * num_experts` floats) and the top-K is a sort, both of
/// which dwarf in cost any plausible host↔device transfer.
///
/// Per-pair dispatch budget (m=1, Metal):
///   gate_up Fused gemv (2 parts) + silu + down gemv + scaled_add
///   = 5 dispatches/pair. Plus 2 copy_slice/token (load x_single,
///   write acc_buf back to out[b]). With top_k=8 and 48 layers, that's
///   8×5 + 2 = 42 dispatches/layer × 48 ≈ 2k/token (vs. ~3.5k in the
///   pre-PR scheme that round-tripped through `out` per pair).
#[allow(clippy::too_many_arguments)]
pub fn moe_forward<B: Backend>(
    ctx: &mut B::Context,
    x: &B::Buffer,
    router_logits: &B::Buffer,
    out: &mut B::Buffer,
    batch: usize,
    hidden_size: usize,
    expert_intermediate: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    experts: &ExpertStack<B>,
    x_single: &mut B::Buffer,
    acc_buf: &mut B::Buffer,
    gate_up_buf: &mut B::Buffer,
    silu_buf: &mut B::Buffer,
    down_buf: &mut B::Buffer,
    zero_hidden: &B::Buffer,
) -> Result<()> {
    let n_experts = experts.num_experts();
    if n_experts != num_experts {
        return Err(FerrumError::model(format!(
            "moe_forward: experts.num_experts() = {n_experts} != cfg.num_experts = {num_experts}"
        )));
    }

    let prof = moe_profile_enabled();

    // Routing on host. Sized batch*num_experts (e.g. 512*128 = 64k floats
    // per layer for Qwen3-30B-A3B prefill); cheap relative to the per-
    // expert gemvs that follow.
    let t0 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    B::sync(ctx);
    if let Some(t) = t0 {
        MOE_SYNC_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        MOE_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    let t0 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let logits_host = B::to_vec(router_logits, batch * num_experts);
    let route_out =
        crate::moe::router::route(&logits_host, batch, num_experts, top_k, norm_topk_prob);
    if let Some(t) = t0 {
        MOE_HOST_TOPK_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        MOE_HOST_TOPK_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    for b in 0..batch {
        // Load x[b] into x_single + reset accumulator.
        let t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::copy_slice(ctx, x, b * hidden_size, x_single, 0, hidden_size);
        B::copy_slice(ctx, zero_hidden, 0, acc_buf, 0, hidden_size);
        if let Some(t) = t0 {
            MOE_COPY_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            MOE_COPY_CALLS.fetch_add(2, Ordering::Relaxed);
        }

        for k in 0..top_k {
            let pair = b * top_k + k;
            let expert_id = route_out.expert_ids[pair] as usize;
            let weight = route_out.expert_weights[pair];
            if expert_id >= num_experts {
                return Err(FerrumError::model(format!(
                    "moe_forward: routed expert {expert_id} >= num_experts {num_experts}"
                )));
            }

            // Fused gate||up gemv → [2 * expert_inter]
            let t0 = if prof {
                B::sync(ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
            experts.gate_up[expert_id].forward(ctx, x_single, gate_up_buf, 1);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_GEMV_GATE_UP_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_GEMV_GATE_UP_CALLS.fetch_add(1, Ordering::Relaxed);
            }

            // SiLU(gate) * up → [expert_inter]
            let t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            B::fused_silu_mul_split(ctx, gate_up_buf, silu_buf, 1, expert_intermediate);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_SILU_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_SILU_CALLS.fetch_add(1, Ordering::Relaxed);
            }

            // down gemv → [hidden]
            let t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            experts.down[expert_id].forward(ctx, silu_buf, down_buf, 1);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_GEMV_DOWN_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_GEMV_DOWN_CALLS.fetch_add(1, Ordering::Relaxed);
            }

            // acc_buf += weight * down_buf
            let t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            B::scaled_add_inplace(ctx, acc_buf, down_buf, weight, hidden_size);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_SCALED_ADD_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_SCALED_ADD_CALLS.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Final write: out[b] = acc_buf
        let t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::copy_slice(ctx, acc_buf, 0, out, b * hidden_size, hidden_size);
        if let Some(t) = t0 {
            MOE_COPY_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            MOE_COPY_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    Ok(())
}

/// Run MoE forward on CPU.
///
/// Inputs:
///   - `x`: `[batch, hidden_size]` row-major hidden states (post-attention,
///          post-residual — i.e. what the dense MLP would normally see).
///   - `router`: top-K assignments + weights from [`super::router::route`].
///   - `experts`: per-layer expert weights from [`ExpertStack::load_from_gguf`].
///
/// Output:
///   - `out`: `[batch, hidden_size]`. Resized + zero-initialised.
///
/// The function recomputes its scratch buffers each call. For tight
/// inner loops, callers will eventually want a pre-allocated workspace
/// (Phase 2F refactor). For now, this is the readable reference.
pub fn moe_forward_cpu(
    x: &[f32],
    batch: usize,
    hidden_size: usize,
    expert_intermediate: usize,
    top_k: usize,
    router: &RouterOutput,
    experts: &ExpertStack<CpuBackend>,
    out: &mut Vec<f32>,
) -> Result<()> {
    let n_experts = experts.num_experts();

    if x.len() != batch * hidden_size {
        return Err(FerrumError::model(format!(
            "moe_forward_cpu: x len {} doesn't match batch*hidden = {}*{} = {}",
            x.len(),
            batch,
            hidden_size,
            batch * hidden_size
        )));
    }
    if router.expert_ids.len() != batch * top_k {
        return Err(FerrumError::model(format!(
            "moe_forward_cpu: router has {} expert_ids but expected batch*top_k = {}*{} = {}",
            router.expert_ids.len(),
            batch,
            top_k,
            batch * top_k
        )));
    }

    out.clear();
    out.resize(batch * hidden_size, 0.0);

    let mut ctx = <CpuBackend as Backend>::new_context();
    let mut x_b: Vec<f32> = vec![0.0; hidden_size];
    let mut gate_up_buf: Vec<f32> = vec![0.0; 2 * expert_intermediate];
    let mut silu_mul_buf: Vec<f32> = vec![0.0; expert_intermediate];
    let mut down_out: Vec<f32> = vec![0.0; hidden_size];

    for b in 0..batch {
        x_b.copy_from_slice(&x[b * hidden_size..(b + 1) * hidden_size]);

        for k in 0..top_k {
            let pair_idx = b * top_k + k;
            let expert_id = router.expert_ids[pair_idx] as usize;
            let weight = router.expert_weights[pair_idx];

            if expert_id >= n_experts {
                return Err(FerrumError::model(format!(
                    "moe_forward_cpu: router selected expert {expert_id} >= num_experts {n_experts}"
                )));
            }

            // Gate||Up projection (fused) → [1, 2*expert_inter]
            experts.gate_up[expert_id].forward(&mut ctx, &x_b, &mut gate_up_buf, 1);

            // SiLU(gate) * up → [1, expert_inter]
            <CpuBackend as Backend>::fused_silu_mul_split(
                &mut ctx,
                &gate_up_buf,
                &mut silu_mul_buf,
                1,
                expert_intermediate,
            );

            // Down projection → [1, hidden]
            experts.down[expert_id].forward(&mut ctx, &silu_mul_buf, &mut down_out, 1);

            // Weighted accumulate into out[b, :]. Done host-side because
            // CpuBackend::Buffer = Vec<f32> and the trait doesn't yet
            // expose scaled-add.
            let out_row = &mut out[b * hidden_size..(b + 1) * hidden_size];
            for (o, d) in out_row.iter_mut().zip(down_out.iter()) {
                *o += weight * *d;
            }
        }
    }

    Ok(())
}

fn check_size(actual: usize, expected: usize, label: &str) -> Result<()> {
    if actual != expected {
        return Err(FerrumError::model(format!(
            "ExpertStack: {label} size mismatch (got {actual}, expected {expected})"
        )));
    }
    Ok(())
}

/// Map candle's `GgmlDType` to the kernel-side `GgufQuantType` for the
/// dtypes a backend can dispatch on. Returns `None` for any other dtype
/// (callers fall back to eager dequant).
fn quant_kind(gguf: &GgufFile, name: &str) -> Result<Option<GgufQuantType>> {
    let info = gguf.tensor_info(name).ok_or_else(|| {
        FerrumError::model(format!("ExpertStack: tensor info missing for '{name}'"))
    })?;
    Ok(match info.ggml_dtype {
        GgmlDType::Q4K => Some(GgufQuantType::Q4K),
        GgmlDType::Q6K => Some(GgufQuantType::Q6K),
        _ => None,
    })
}

/// Per-expert block-byte count for a given k-quant flavour and element
/// count. Q4_K = 144 B / 256 elems, Q6_K = 210 B / 256 elems. Errors if
/// `n_elems` is not a multiple of the super-block size (256) — a Q-quant
/// invariant.
fn block_bytes_for(kind: GgufQuantType, n_elems: usize, label: &str) -> Result<usize> {
    const QK_K: usize = 256;
    if n_elems % QK_K != 0 {
        return Err(FerrumError::model(format!(
            "ExpertStack {label}: per-expert element count {n_elems} not a multiple of {QK_K}"
        )));
    }
    let block_bytes = match kind {
        GgufQuantType::Q4K => 144,
        GgufQuantType::Q6K => 210,
        // Other k-quants are filtered out earlier via `quant_kind`; reaching here
        // with one would be a programming error.
        other => {
            return Err(FerrumError::model(format!(
                "ExpertStack {label}: unsupported k-quant flavour {other:?}"
            )))
        }
    };
    Ok((n_elems / QK_K) * block_bytes)
}

fn read_dequant_flat(gguf: &GgufFile, name: &str, device: &Device) -> Result<Vec<f32>> {
    let qt = gguf.read_tensor(name, device).map_err(candle_to_ferrum)?;
    let dense = qt.dequantize(device).map_err(candle_to_ferrum)?;
    let flat = dense.flatten_all().map_err(candle_to_ferrum)?;
    flat.to_vec1::<f32>().map_err(candle_to_ferrum)
}

fn candle_to_ferrum(e: candle_core::Error) -> FerrumError {
    FerrumError::model(format!("candle: {e}"))
}

// Suppress unused-import warning when this module compiles standalone in
// the lib (the candle Result alias is only used via map_err in Phase 2).
#[allow(dead_code)]
type _CandleResult<T> = CandleResult<T>;
