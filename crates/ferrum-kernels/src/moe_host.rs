//! Backend-agnostic MoE host-side helpers — used by all backends and
//! across all builds (no `cfg(feature = "metal")` gate).
//!
//! `compute_ids_tpe` builds the `ids[num_experts, max_per_expert]` /
//! `tpe[num_experts]` arrays from per-token expert assignments. It's
//! pure CPU code: a small bucket-sort by expert id over the
//! `[batch, top_k]` selected_experts table.

/// Host-side computation of `tpe[num_experts]` and
/// `ids[num_experts, max_per_expert]` from per-token expert IDs.
///
/// Input: `selected_experts[batch * top_k]` — flat array of expert IDs
/// where index `b * top_k + k` is token `b`'s `k`-th selected expert.
///
/// Returns `(tpe, ids, max_per_expert)`:
/// - `tpe[e]` = number of (token, slot) pairs assigned to expert `e`.
/// - `ids[e * max_per_expert + slot]` = global pair index `b * top_k + k`
///   that landed in expert `e`'s slot `slot`.
/// - `max_per_expert` is the largest count across all experts (defines
///   the row stride of the `ids` array).
pub fn compute_ids_tpe(
    selected_experts: &[u32],
    num_experts: usize,
    batch: usize,
    top_k: usize,
) -> (Vec<i32>, Vec<i32>, usize) {
    debug_assert_eq!(selected_experts.len(), batch * top_k);

    let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); num_experts];
    for b in 0..batch {
        for k in 0..top_k {
            let pair_idx = (b * top_k + k) as i32;
            let e = selected_experts[b * top_k + k] as usize;
            if e < num_experts {
                buckets[e].push(pair_idx);
            }
        }
    }

    let max_per_expert = buckets.iter().map(|v| v.len()).max().unwrap_or(0);
    let max_per_expert = max_per_expert.max(1);

    let mut tpe = vec![0i32; num_experts];
    let mut ids = vec![0i32; num_experts * max_per_expert];
    for (e, bucket) in buckets.iter().enumerate() {
        tpe[e] = bucket.len() as i32;
        let off = e * max_per_expert;
        for (slot, &pair) in bucket.iter().enumerate() {
            ids[off + slot] = pair;
        }
    }
    (tpe, ids, max_per_expert)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_ids_tpe_simple() {
        // 2 tokens, top_k=2, num_experts=4.
        // Token 0 picks experts (1, 3); token 1 picks (1, 0).
        let selected = vec![1u32, 3, 1, 0];
        let (tpe, ids, mpe) = compute_ids_tpe(&selected, 4, 2, 2);
        assert_eq!(tpe, vec![1, 2, 0, 1]);
        assert_eq!(mpe, 2);
        assert_eq!(ids[0], 3);
        assert_eq!(ids[2], 0);
        assert_eq!(ids[3], 2);
        assert_eq!(ids[6], 1);
    }
}
