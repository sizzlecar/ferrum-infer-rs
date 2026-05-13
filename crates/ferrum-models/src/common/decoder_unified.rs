//! Shared helpers for decoder-only unified mixed-batch forward.
//!
//! The Llama / Qwen3-MoE / future decoder families all share the same
//! outer scaffolding for unified forward: cu_seqlens construction,
//! block-table stacking, final-token index lookup, graph-cache keying.
//! These are pure functions — no kernel calls, no model state — extracted
//! here so each family's `unified_forward_internal` reads as
//! "scaffolding + family-specific layer loop", not "scaffolding +
//! 700 lines of scaffolding clone".
//!
//! Per `docs/decoder-unified-runner-abstraction.md`. Phase 2A.

/// Cumulative q-token counts: `cu_seqlens_q[i+1] - cu_seqlens_q[i] =
/// items[i].q_tokens.len()`. The varlen attention + paged-KV-write
/// kernels read this to find each sequence's slice of the flat
/// `[M_total, *]` tensor.
///
/// Also returns the flat `q_lens[i] = items[i].q_tokens.len()` and
/// `m_total = sum(q_lens)`.
pub fn compute_cu_seqlens_q(
    items: &[(String, Vec<u32>, usize, bool)],
) -> (Vec<usize>, Vec<u32>, usize) {
    let q_lens: Vec<usize> = items.iter().map(|it| it.1.len()).collect();
    let mut cu_seqlens_q: Vec<u32> = Vec::with_capacity(items.len() + 1);
    cu_seqlens_q.push(0);
    for &l in &q_lens {
        let prev = *cu_seqlens_q.last().unwrap();
        cu_seqlens_q.push(prev + l as u32);
    }
    let m_total = *cu_seqlens_q.last().unwrap() as usize;
    (q_lens, cu_seqlens_q, m_total)
}

/// Per-item starting absolute KV position for the FIRST q-token in
/// `items[i].q_tokens`. Zero for fresh prefill, prior `kv_len` for
/// chunked-prefill continuations or decode steps. Returned as `u32`
/// to match the device-side index buffers the varlen kernels read.
pub fn compute_pos_offsets(items: &[(String, Vec<u32>, usize, bool)]) -> Vec<u32> {
    items.iter().map(|it| it.2 as u32).collect()
}

/// Causal max over `(pos_offset + q_len)` — needed for the varlen
/// attention kernel's shared-mem sizing (must fit the longest reachable
/// `kv_pos` across all items in the batch).
pub fn compute_max_kv_len(items: &[(String, Vec<u32>, usize, bool)]) -> usize {
    items
        .iter()
        .map(|it| it.2 + it.1.len())
        .max()
        .unwrap_or(0)
}

/// Flatten all items' q-tokens into one concatenated `[M_total]` vec.
/// Caller passes this to `embedding_lookup` so the entire batch's
/// embeddings end up contiguous in the unified residual buffer.
pub fn concat_q_tokens(items: &[(String, Vec<u32>, usize, bool)]) -> Vec<u32> {
    items.iter().flat_map(|it| it.1.iter().copied()).collect()
}

/// Pack per-(seq, layer-0) page indices into the dense
/// `[num_seqs, max_blocks_per_seq]` layout that the varlen attention
/// kernel reads. Layer indexing is "first layer's block table"
/// because in ferrum's paged-KV layout every layer shares the same
/// block-table list (the layer-specific data lives inside each KV
/// pool; the table itself is per-sequence).
///
/// `lookup` returns the block-indices slice for each item's cache_id;
/// the caller wires this to its model's `kv_caches.get(cid)`.
pub fn stack_block_tables<F: Fn(&str) -> Vec<u32>>(
    items: &[(String, Vec<u32>, usize, bool)],
    max_blocks_per_seq: usize,
    lookup: F,
) -> Vec<u32> {
    let mut stacked: Vec<u32> = vec![0u32; items.len() * max_blocks_per_seq];
    for (i, (cid, _, _, _)) in items.iter().enumerate() {
        let blocks = lookup(cid);
        let n_to_copy = blocks.len().min(max_blocks_per_seq);
        stacked[i * max_blocks_per_seq..i * max_blocks_per_seq + n_to_copy]
            .copy_from_slice(&blocks[..n_to_copy]);
    }
    stacked
}

/// For each `is_final_chunk = true` item, return `(orig_index, global_token_index)`
/// where `global_token_index` is the position in the flat `[M_total, hidden]`
/// residual buffer of that item's LAST q-token. The final-norm + lm_head
/// stages slice these rows out for sampling.
pub fn compute_final_indices(
    items: &[(String, Vec<u32>, usize, bool)],
    cu_seqlens_q: &[u32],
) -> Vec<(usize, usize)> {
    items
        .iter()
        .enumerate()
        .filter(|(_, it)| it.3)
        .map(|(orig_idx, it)| {
            let last_token_local = it.1.len() - 1;
            let global = (cu_seqlens_q[orig_idx] as usize) + last_token_local;
            (orig_idx, global)
        })
        .collect()
}

/// Graph cache key for a unified mixed-batch forward. High bit set so we
/// never collide with legacy decode/batched keys (which use the low 63
/// bits for `m_padded` / `SINGLE_ITEM = 0`).
///
/// Keyed by `(m_total, num_seqs)` because the captured kernel launches
/// bake in grid_dim / per-seq scratch indexing for that specific shape;
/// reusing a graph for a different shape leads to wrong-shape memory
/// access. (See memory `project_moe_phase3_graph_bug.md` for the same
/// rationale in the legacy MoE path.)
pub const fn unified_graph_key(m_total: usize, num_seqs: usize) -> u64 {
    (1u64 << 63) | ((m_total as u64) << 32) | (num_seqs as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(cid: &str, q_len: usize, pos: usize, final_chunk: bool) -> (String, Vec<u32>, usize, bool) {
        (cid.to_string(), vec![0u32; q_len], pos, final_chunk)
    }

    #[test]
    fn cu_seqlens_q_mixed_lengths() {
        let items = vec![item("a", 5, 0, true), item("b", 1, 100, true), item("c", 3, 10, false)];
        let (q_lens, cu, m_total) = compute_cu_seqlens_q(&items);
        assert_eq!(q_lens, vec![5, 1, 3]);
        assert_eq!(cu, vec![0, 5, 6, 9]);
        assert_eq!(m_total, 9);
    }

    #[test]
    fn pos_offsets_and_max_kv_len() {
        let items = vec![item("a", 5, 0, true), item("b", 1, 100, true), item("c", 3, 10, false)];
        assert_eq!(compute_pos_offsets(&items), vec![0u32, 100, 10]);
        assert_eq!(compute_max_kv_len(&items), 101); // b: 100 + 1
    }

    #[test]
    fn final_indices_only_final_chunks() {
        let items = vec![
            item("a", 5, 0, true),  // last token at global 4
            item("b", 1, 100, true), // last at global 5
            item("c", 3, 10, false), // not final
        ];
        let (_, cu, _) = compute_cu_seqlens_q(&items);
        let fi = compute_final_indices(&items, &cu);
        assert_eq!(fi, vec![(0, 4), (1, 5)]);
    }

    #[test]
    fn graph_key_high_bit_set() {
        let k = unified_graph_key(32, 4);
        assert!(k & (1u64 << 63) != 0, "high bit must be set");
        // Legacy key with same low bits should differ.
        let legacy = ((32u64) << 32) | 4u64;
        assert_ne!(k, legacy);
    }

    #[test]
    fn stack_block_tables_pads_and_truncates() {
        let items = vec![item("a", 1, 0, true), item("b", 1, 0, true)];
        // Item a has 2 blocks; b has 5 but max_blocks_per_seq=3
        let stacked = stack_block_tables(&items, 3, |cid| match cid {
            "a" => vec![10u32, 11u32],
            "b" => vec![20u32, 21u32, 22u32, 23u32, 24u32],
            _ => unreachable!(),
        });
        // a: [10, 11, 0]  (padded with 0)
        // b: [20, 21, 22] (truncated to 3)
        assert_eq!(stacked, vec![10, 11, 0, 20, 21, 22]);
    }

    #[test]
    fn empty_items() {
        let items: Vec<(String, Vec<u32>, usize, bool)> = Vec::new();
        let (q_lens, cu, m_total) = compute_cu_seqlens_q(&items);
        assert!(q_lens.is_empty());
        assert_eq!(cu, vec![0]);
        assert_eq!(m_total, 0);
    }
}
