//! Parity: `moe_forward_bucketed` vs `moe_forward` on synthetic GPTQ
//! experts, CPU backend. Both paths should produce numerically equal
//! output for the same input/router state — they're computing the same
//! sum, just in different dispatch orders.

use std::sync::Arc;

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::BackendQuantMarlin;
use ferrum_models::moe::{moe_forward, moe_forward_bucketed, ExpertStack, MoeRouteScratch};
use ferrum_quantization::{Linear, StackedExpertLinear};

/// Tiny PRNG (same as gptq_parity_test).
fn rnd_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

fn rnd_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = (rnd_u32(state) & 0x00FF_FFFF) as f32 / 16_777_216.0;
    lo + u * (hi - lo)
}

fn synth_gptq(k: usize, n: usize, group_size: usize, seed: u64) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let mut rs = seed;
    let num_groups = k / group_size;
    let mut qweight = vec![0i32; (k / 8) * n];
    for qw in qweight.iter_mut() {
        *qw = rnd_u32(&mut rs) as i32;
    }
    let mut scales = vec![0f32; num_groups * n];
    for s in scales.iter_mut() {
        *s = rnd_f32(&mut rs, 0.01, 0.1);
    }
    let mut qzeros = vec![0i32; num_groups * (n / 8)];
    for qz in qzeros.iter_mut() {
        let mut word: u32 = 0;
        for j in 0..8 {
            word |= ((rnd_u32(&mut rs) & 0xF) as u32) << (j * 4);
        }
        *qz = word as i32;
    }
    (qweight, scales, qzeros)
}

/// Build a stacked GPTQ store covering `num_experts` experts × N cols
/// each. Phase 3e/2: uses the proper `BackendQuantMarlin::load_gptq_stacked`
/// API (which CpuBackend now implements) instead of abusing `load_gptq`
/// on a hand-concatenated tensor.
fn build_stacked(
    k: usize,
    n_per_expert: usize,
    num_experts: usize,
    group_size: usize,
    base_seed: u64,
) -> Arc<dyn ferrum_kernels::MarlinExpertStack<CpuBackend>> {
    let parts: Vec<_> = (0..num_experts)
        .map(|e| synth_gptq(k, n_per_expert, group_size, base_seed + e as u64))
        .collect();
    let qweights: Vec<&[i32]> = parts.iter().map(|p| p.0.as_slice()).collect();
    let scales: Vec<&[f32]> = parts.iter().map(|p| p.1.as_slice()).collect();
    let qzeros: Vec<&[i32]> = parts.iter().map(|p| p.2.as_slice()).collect();
    <CpuBackend as BackendQuantMarlin>::load_gptq_stacked(
        &qweights,
        &scales,
        &qzeros,
        None,
        4,
        group_size,
        k,
        n_per_expert,
    )
    .expect("CPU stacked load_gptq_stacked")
}

#[test]
fn bucketed_matches_per_pair_dispatch() {
    let hidden = 64;
    let inter = 32;
    let num_experts = 4;
    let top_k = 2;
    let batch = 5;
    let group_size = 32;

    // Build stacked MarlinExpertStacks (Phase C step 4e: build_stacked now
    // returns the trait object directly).
    let gate_up_stack = build_stacked(hidden, 2 * inter, num_experts, group_size, 0xA);
    let down_stack = build_stacked(inter, hidden, num_experts, group_size, 0xB);

    // Per-expert StackedExpertLinear views built from the trait-object
    // stack (Phase C step 4b). Used by per-pair dispatch path.
    let mut gate_up_per_expert: Vec<Box<dyn Linear<CpuBackend>>> = Vec::with_capacity(num_experts);
    let mut down_per_expert: Vec<Box<dyn Linear<CpuBackend>>> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        gate_up_per_expert.push(Box::new(
            StackedExpertLinear::<CpuBackend>::new(gate_up_stack.clone(), e * 2 * inter, 2 * inter)
                .expect("gate_up StackedExpertLinear"),
        ));
        down_per_expert.push(Box::new(
            StackedExpertLinear::<CpuBackend>::new(down_stack.clone(), e * hidden, hidden)
                .expect("down StackedExpertLinear"),
        ));
    }
    let experts = ExpertStack::<CpuBackend> {
        gate_up: gate_up_per_expert,
        down: down_per_expert,
        gate_stacked: None,
        up_stacked: None,
        down_stacked: None,
        gate_up_marlin_stack: Some(gate_up_stack),
        down_marlin_stack: Some(down_stack),
    };

    // Synthetic input + router logits.
    let mut rs = 0xC0FFEEu64;
    let x: Vec<f32> = (0..batch * hidden)
        .map(|_| rnd_f32(&mut rs, -1.0, 1.0))
        .collect();
    let router_logits: Vec<f32> = (0..batch * num_experts)
        .map(|_| rnd_f32(&mut rs, -2.0, 2.0))
        .collect();

    // ── Path A: per-pair moe_forward ───────────────────────────────────
    let mut out_a = vec![0.0f32; batch * hidden];
    let mut x_single = vec![0.0f32; hidden];
    let mut acc_buf = vec![0.0f32; hidden];
    let mut gate_up_buf = vec![0.0f32; 2 * inter];
    let mut silu_buf = vec![0.0f32; inter];
    let mut down_buf = vec![0.0f32; hidden];
    let zero_hidden = vec![0.0f32; hidden];
    let mut ctx_a = ();
    moe_forward::<CpuBackend>(
        &mut ctx_a,
        &x,
        &router_logits,
        &mut out_a,
        batch,
        hidden,
        inter,
        num_experts,
        top_k,
        true,
        &experts,
        &mut x_single,
        &mut acc_buf,
        &mut gate_up_buf,
        &mut silu_buf,
        &mut down_buf,
        &zero_hidden,
    )
    .expect("per-pair forward");

    // ── Path B: moe_forward_bucketed ───────────────────────────────────
    let total_pairs = batch * top_k;
    let mut out_b = vec![0.0f32; batch * hidden];
    let mut x_packed = vec![0.0f32; total_pairs * hidden];
    let mut gate_up_packed = vec![0.0f32; total_pairs * 2 * inter];
    let mut silu_packed = vec![0.0f32; total_pairs * inter];
    let mut down_packed = vec![0.0f32; total_pairs * hidden];
    let mut ctx_b = ();
    let mut route_scratch = MoeRouteScratch::new();
    moe_forward_bucketed::<CpuBackend>(
        &mut ctx_b,
        &x,
        &router_logits,
        &mut out_b,
        batch,
        hidden,
        inter,
        num_experts,
        top_k,
        true,
        &experts,
        &mut x_packed,
        &mut gate_up_packed,
        &mut silu_packed,
        &mut down_packed,
        &mut route_scratch,
        None,
    )
    .expect("bucketed forward");

    // ── Compare ────────────────────────────────────────────────────────
    let max_diff = out_a
        .iter()
        .zip(&out_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mag = out_a.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-6);
    let rel = max_diff / mag;
    eprintln!("moe_forward vs moe_forward_bucketed: max|diff|={max_diff:.6e} rel={rel:.4e}");
    assert!(
        rel < 1e-4,
        "bucketed path drift too large: rel={rel:.4e}, max|diff|={max_diff:.6e}"
    );
}
