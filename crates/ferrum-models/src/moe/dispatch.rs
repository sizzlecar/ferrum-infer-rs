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

use candle_core::{Device, Result as CandleResult};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::Backend;
use ferrum_kernels::Linear;
use ferrum_quantization::gguf::GgufFile;
use ferrum_quantization::DenseLinear;
use ferrum_types::{FerrumError, Result};

use crate::moe::router::RouterOutput;

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
        Ok(Self { gate_up, down })
    }

    /// Load all experts for one MoE layer from a GGUF file. Names follow
    /// the GGUF convention: `blk.{layer_idx}.ffn_{gate,up,down}_exps.weight`.
    /// Tensors are dequantised on CPU (Phase 2 dispatch is CPU-only).
    pub fn load_from_gguf(
        gguf: &GgufFile,
        layer_idx: usize,
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Self> {
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
        Self::from_dense_stacks(
            &gate,
            &up,
            &down,
            num_experts,
            hidden_size,
            expert_intermediate,
        )
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
    pub fn num_experts(&self) -> usize {
        debug_assert_eq!(
            self.gate_up.len(),
            self.down.len(),
            "ExpertStack: gate_up and down disagree on expert count"
        );
        self.gate_up.len()
    }
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
