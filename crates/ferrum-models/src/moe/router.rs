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

    let mut expert_ids = Vec::with_capacity(batch * top_k);
    let mut expert_weights = Vec::with_capacity(batch * top_k);

    for b in 0..batch {
        let row = &logits[b * num_experts..(b + 1) * num_experts];
        let probs = softmax(row);
        let topk = top_k_indices(&probs, top_k);

        // Optionally renorm the K selected weights. If norm is off we
        // emit the raw post-softmax probs (unselected mass discarded).
        let combine_weights = if norm_topk_prob {
            renormalise(&topk, &probs)
        } else {
            topk.iter().map(|&i| probs[i]).collect::<Vec<_>>()
        };

        for (i, &exp_id) in topk.iter().enumerate() {
            expert_ids.push(exp_id as u32);
            expert_weights.push(combine_weights[i]);
        }
    }

    RouterOutput {
        expert_ids,
        expert_weights,
    }
}

/// Numerically-stable softmax over a single row.
fn softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    // sum is guaranteed > 0 because at least one term is exp(max-max) = 1.
    for v in &mut exp {
        *v /= sum;
    }
    exp
}

/// Return the indices of the K largest entries, sorted by value descending,
/// breaking ties by smaller index first (stable / reproducible).
fn top_k_indices(probs: &[f32], top_k: usize) -> Vec<usize> {
    // Pair each prob with its index, sort by (-prob, index) then truncate.
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.truncate(top_k);
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Renormalise the K selected probabilities so they sum to 1.
fn renormalise(selected: &[usize], probs: &[f32]) -> Vec<f32> {
    let sum: f32 = selected.iter().map(|&i| probs[i]).sum();
    // Guard against degenerate sum=0 (shouldn't happen with finite logits).
    if sum > 0.0 {
        selected.iter().map(|&i| probs[i] / sum).collect()
    } else {
        // Fallback: uniform 1/K.
        let k = selected.len() as f32;
        vec![1.0 / k; selected.len()]
    }
}
