//! Router unit tests — softmax + top-K selection + optional re-norm.

use ferrum_models::moe::{route, RouterOutput};

#[test]
fn top_k_one_picks_argmax() {
    // 4-expert logits, expert 2 dominates by a wide margin.
    let logits = vec![0.0_f32, 0.5, 5.0, 0.1];
    let out = route(&logits, 1, 4, 1, true);
    assert_eq!(out.expert_ids, vec![2]);
    // Single-expert top-1 with norm → weight is 1.0 regardless of softmax.
    assert!((out.expert_weights[0] - 1.0).abs() < 1e-6);
}

#[test]
fn top_k_two_orders_by_weight_descending() {
    let logits = vec![0.0_f32, 5.0, 1.0, 0.0];
    let out = route(&logits, 1, 4, 2, false);
    // Selected: index 1 (largest, ~5.0) then index 2 (second largest, 1.0).
    assert_eq!(out.expert_ids, vec![1, 2]);
    // Without renorm, weights are softmax probs of those two.
    let p1 = (5.0_f32).exp() / (0.0_f32.exp() + 5.0_f32.exp() + 1.0_f32.exp() + 0.0_f32.exp());
    let p2 = (1.0_f32).exp() / (0.0_f32.exp() + 5.0_f32.exp() + 1.0_f32.exp() + 0.0_f32.exp());
    assert!(
        (out.expert_weights[0] - p1).abs() < 1e-5,
        "expected {p1}, got {}",
        out.expert_weights[0]
    );
    assert!(
        (out.expert_weights[1] - p2).abs() < 1e-5,
        "expected {p2}, got {}",
        out.expert_weights[1]
    );
}

#[test]
fn norm_topk_prob_sums_selected_to_one() {
    let logits = vec![0.0_f32, 5.0, 1.0, 0.0];
    let out = route(&logits, 1, 4, 2, true);
    let total: f32 = out.expert_weights.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "norm-topk weights must sum to 1, got {total}"
    );
}

#[test]
fn raw_softmax_sums_lt_one_when_some_mass_dropped() {
    let logits = vec![0.0_f32, 0.0, 0.0, 0.0]; // uniform — each prob = 0.25
    let out = route(&logits, 1, 4, 2, false);
    let total: f32 = out.expert_weights.iter().sum();
    // Two of four uniform 0.25s = 0.5
    assert!((total - 0.5).abs() < 1e-5, "got {total}");
}

#[test]
fn batch_dim_routes_each_row_independently() {
    // Two tokens. Token 0 prefers expert 1; token 1 prefers expert 3.
    let logits = vec![
        0.0_f32, 5.0, 0.0, 0.0, // token 0
        0.0, 0.0, 0.0, 5.0, // token 1
    ];
    let out = route(&logits, 2, 4, 1, true);
    assert_eq!(out.expert_ids, vec![1, 3]);
    for &w in &out.expert_weights {
        assert!((w - 1.0).abs() < 1e-6);
    }
}

#[test]
fn top_k_equal_to_num_experts_returns_all_with_renorm_to_one() {
    // top_k = num_experts means router degenerates: every expert is
    // selected with the full softmax probability, and renorm keeps the
    // sum at 1 (which it already was).
    let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
    let out = route(&logits, 1, 4, 4, true);
    assert_eq!(out.expert_ids.len(), 4);
    // After renorm, the four weights still sum to 1.0.
    let total: f32 = out.expert_weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-5);
    // Highest-logit expert (index 3) should have the largest weight.
    let max_pos = out
        .expert_weights
        .iter()
        .cloned()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(out.expert_ids[max_pos], 3);
}

#[test]
fn ties_break_by_smaller_index_first() {
    // All logits equal → softmax uniform. Top-2 must pick indices 0, 1
    // (tie-break: smaller index first) regardless of FP order.
    let logits = vec![1.0_f32; 8];
    let out = route(&logits, 1, 8, 2, true);
    assert_eq!(out.expert_ids, vec![0, 1]);
}

#[test]
#[should_panic(expected = "top_k must be > 0")]
fn top_k_zero_panics() {
    let _ = route(&[1.0_f32; 4], 1, 4, 0, true);
}

#[test]
#[should_panic(expected = "exceeds num_experts")]
fn top_k_greater_than_num_experts_panics() {
    let _ = route(&[1.0_f32; 4], 1, 4, 5, true);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn wrong_logits_size_panics() {
    let _ = route(&[1.0_f32; 7], 1, 4, 2, true);
}

#[test]
fn router_output_can_be_partially_consumed() {
    // Spot-check: indexing matches the documented stride convention.
    let logits = vec![0.0_f32, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0];
    let out: RouterOutput = route(&logits, 2, 4, 2, true);
    // batch=2, top_k=2 → 4 entries
    assert_eq!(out.expert_ids.len(), 4);
    assert_eq!(out.expert_weights.len(), 4);
    // Token 0's selections occupy [0..2]; token 1's occupy [2..4].
    assert_eq!(out.expert_ids[0], 1); // token 0, k=0
    assert_eq!(out.expert_ids[2], 3); // token 1, k=0
}
