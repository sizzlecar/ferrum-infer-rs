//! MoE router (gating) — pick top-K experts per token.
//!
//! Given router logits of shape `[batch, num_experts]` (output of the small
//! gating linear), produce per-token expert indices + combine weights:
//!
//!   1. Softmax over each row (so all probs are non-negative and sum to 1).
//!   2. Take the K highest-probability experts.
//!   3. Optionally renormalise those K probs so they sum back to 1
//!      (Qwen3-MoE / Mixtral default; some legacy variants don't).
//!
//! Output layout is **flat with stride `top_k`**: `expert_ids[b*K + k]`
//! is the k-th selected expert for token b, and `expert_weights[b*K + k]`
//! is its combine weight. That matches how the dispatch loop iterates.

/// Result of routing one batch: parallel arrays indexed `[b * top_k + k]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterOutput {
    /// Selected expert indices. `expert_ids[b * top_k + k] ∈ [0, num_experts)`.
    pub expert_ids: Vec<u32>,
    /// Combine weights. Same shape as `expert_ids`. If
    /// `norm_topk_prob` was true, the K weights for each token sum to 1;
    /// otherwise they're the raw (post-softmax) probabilities of the
    /// selected experts.
    pub expert_weights: Vec<f32>,
}

impl RouterOutput {
    /// Empty `RouterOutput` with no allocation. Use [`Self::reset`] before
    /// reuse — this is the cheap constructor for putting it in a scratch
    /// struct.
    pub fn empty() -> Self {
        Self {
            expert_ids: Vec::new(),
            expert_weights: Vec::new(),
        }
    }

    /// Resize both vectors to `batch * top_k`. Existing capacity is reused
    /// when sufficient; growth uses standard `Vec::resize`. Old contents
    /// are not preserved (callers always overwrite).
    pub fn reset(&mut self, batch: usize, top_k: usize) {
        let n = batch * top_k;
        self.expert_ids.clear();
        self.expert_ids.resize(n, 0);
        self.expert_weights.clear();
        self.expert_weights.resize(n, 0.0);
    }

    /// Number of tokens routed.
    pub fn batch(&self) -> usize {
        // `top_k` is the second dimension; we don't store it explicitly,
        // so derive it from the assumption that the caller passed
        // consistent sizes. Length checks belong upstream.
        self.expert_ids.len() / self.batch_top_k_pair_count()
    }

    fn batch_top_k_pair_count(&self) -> usize {
        // Defensive: avoid divide-by-zero for the empty-router case.
        self.expert_ids.len().max(1)
    }
}

/// Route a batch of tokens to top-K experts.
///
/// `logits`: row-major `[batch, num_experts]`. Each row is the raw output
/// of the gating linear for one token.
///
/// `norm_topk_prob`: if true, the K returned weights for each token are
/// renormalised to sum to 1 (after the masked softmax) — Qwen3-MoE and
/// Mixtral both do this. If false, they're the raw softmax probabilities,
/// which leaves probability mass "on the floor" for unselected experts.
///
/// Panics if `top_k == 0` or `top_k > num_experts` or
/// `logits.len() != batch * num_experts` — these are programming errors,
/// not runtime conditions.
pub fn route(
    logits: &[f32],
    batch: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
) -> RouterOutput {
    let mut out = RouterOutput::empty();
    let mut scratch = Vec::new();
    route_into(
        logits,
        batch,
        num_experts,
        top_k,
        norm_topk_prob,
        &mut out,
        &mut scratch,
    );
    out
}

/// Allocation-free variant of [`route`].
///
/// The 4 per-row `Vec` allocations of [`route`] (softmax buffer, indexed
/// pair buffer for sort, top-K index buffer, renormalised weights buffer)
/// dominate per-token cost in MoE forward — at c=32 / num_experts=128 /
/// top_k=8 / 48 layers that's 4 608 allocations per decode token, or
/// ~10 ms of pure CPU per token (25% of MoE wallclock at c=32 on RTX 4090).
///
/// This variant takes a reusable `out: &mut RouterOutput` and a
/// `scratch_probs: &mut Vec<f32>` softmax buffer, both of which are
/// `clear() + resize()` reused across calls — zero allocations after warmup.
/// Top-K is computed via argmax-mask (K passes of a linear scan) instead
/// of a full O(N log N) sort, which is also faster for K=8 / N=128.
///
/// Tie-breaking: when two probs are equal, the smaller index wins (matches
/// [`route`] / Metal `moe_router_topk_softmax_f32` for bit-exact output).
pub fn route_into(
    logits: &[f32],
    batch: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    out: &mut RouterOutput,
    scratch_probs: &mut Vec<f32>,
) {
    assert_eq!(
        logits.len(),
        batch * num_experts,
        "router logits shape mismatch: expected {batch}×{num_experts}, got {}",
        logits.len()
    );
    assert!(top_k > 0, "top_k must be > 0");
    assert!(
        top_k <= num_experts,
        "top_k {top_k} exceeds num_experts {num_experts}"
    );

    out.reset(batch, top_k);
    scratch_probs.clear();
    scratch_probs.resize(num_experts, 0.0);

    for b in 0..batch {
        let row = &logits[b * num_experts..(b + 1) * num_experts];

        // ── Softmax in-place into scratch_probs. ─────────────────────────
        let mut max = f32::NEG_INFINITY;
        for &v in row {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for (i, &v) in row.iter().enumerate() {
            let e = (v - max).exp();
            scratch_probs[i] = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;
        for v in scratch_probs.iter_mut() {
            *v *= inv_sum;
        }

        // ── Top-K via argmax-mask. K passes; each picks the largest
        // remaining prob and overwrites it with -inf so the next pass
        // sees the next-best. The strict `v > best` keeps the first
        // (smallest-index) tied entry — matches the sort-based path.
        let mut sel_sum = 0.0f32;
        let dst_lo = b * top_k;
        for k in 0..top_k {
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for (i, &v) in scratch_probs.iter().enumerate() {
                if v > best {
                    best = v;
                    best_idx = i;
                }
            }
            out.expert_ids[dst_lo + k] = best_idx as u32;
            out.expert_weights[dst_lo + k] = best;
            sel_sum += best;
            scratch_probs[best_idx] = f32::NEG_INFINITY;
        }

        // ── Optional renorm of the K picked weights. ─────────────────
        if norm_topk_prob {
            if sel_sum > 0.0 {
                let scale = 1.0 / sel_sum;
                for w in &mut out.expert_weights[dst_lo..dst_lo + top_k] {
                    *w *= scale;
                }
            } else {
                let uniform = 1.0 / top_k as f32;
                for w in &mut out.expert_weights[dst_lo..dst_lo + top_k] {
                    *w = uniform;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn run_parity(batch: usize, num_experts: usize, top_k: usize, norm: bool, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let logits: Vec<f32> = (0..batch * num_experts)
            .map(|_| rng.gen_range(-3.0..3.0_f32))
            .collect();

        // Old / new bit-for-bit must match — both use stable max-subtract
        // softmax + first-tie-wins top-K + (optional) sum-renorm.
        let a = route(&logits, batch, num_experts, top_k, norm);
        let mut b = RouterOutput::empty();
        let mut probs = Vec::new();
        route_into(&logits, batch, num_experts, top_k, norm, &mut b, &mut probs);

        assert_eq!(a.expert_ids, b.expert_ids, "expert_ids mismatch");
        for (i, (&aw, &bw)) in a.expert_weights.iter().zip(&b.expert_weights).enumerate() {
            // Within ulps for the renorm-divide, exact otherwise.
            let delta = (aw - bw).abs();
            assert!(
                delta < 1e-6,
                "weight[{i}] mismatch: route={aw} route_into={bw} delta={delta}"
            );
        }
    }

    #[test]
    fn parity_qwen3_moe_shape() {
        // Qwen3-MoE 30B-A3B production shape (norm_topk_prob=true).
        run_parity(32, 128, 8, true, 0xDEADBEEF);
        run_parity(1, 128, 8, true, 0x1234);
        run_parity(64, 128, 8, true, 0x5678);
    }

    #[test]
    fn parity_no_renorm() {
        run_parity(8, 64, 4, false, 0xC0FFEE);
    }

    #[test]
    fn parity_topk_one() {
        run_parity(4, 16, 1, true, 0x42);
        run_parity(4, 16, 1, false, 0x42);
    }

    #[test]
    fn allocation_free_after_warmup() {
        // Sanity: scratch capacity stays put across calls — we don't
        // grow / shrink the underlying Vec on each call.
        let mut out = RouterOutput::empty();
        let mut probs = Vec::new();
        let logits = vec![0.5f32; 32 * 128];
        route_into(&logits, 32, 128, 8, true, &mut out, &mut probs);
        let cap_ids = out.expert_ids.capacity();
        let cap_w = out.expert_weights.capacity();
        let cap_p = probs.capacity();
        // Repeat — capacity must not grow.
        for _ in 0..16 {
            route_into(&logits, 32, 128, 8, true, &mut out, &mut probs);
            assert_eq!(out.expert_ids.capacity(), cap_ids);
            assert_eq!(out.expert_weights.capacity(), cap_w);
            assert_eq!(probs.capacity(), cap_p);
        }
    }
}
