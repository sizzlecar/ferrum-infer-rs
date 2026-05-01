# Paged attention Phase 4-5 plan — multi-seq + concurrent bench

**State after PRs #68 + #69 + #70**: paged-KV kernels + LlamaFamily integration are correct end-to-end for single-seq decode + prefill. Byte-identical greedy decode vs contig, no perf regression. Llama-3.x and Qwen3 both work.

**Goal of Phase 4-5**: unlock multi-seq concurrent serving on Metal — close the capability gap with mistral.rs/vLLM, get a 16-concurrent throughput number.

## Phase 4: multi-seq pool sharing

### What changes

The current `LlamaFamilyModel`:
- `kv_caches: HashMap<cache_id, Vec<KvCache>>` — each cache_id has its own pool
- Each `KvCache` allocates a separate K/V pool (~1 GB per request for Llama-8B 4096 ctx)

Phase 4 architecture:
- Single shared pool per layer: `Vec<KvCache>` indexed by layer, where K/V are sized to `MAX_TOTAL_BLOCKS`
- Block allocator (per-model, not per-request) hands out free blocks
- Per-cache_id state: `Vec<u32>` (block_table — list of physical blocks) + `usize` (current len)

```rust
pub struct LlamaFamilyModel<B: Backend> {
    // ...existing fields...
    /// Shared pools per layer. Sized for total concurrent KV across
    /// MAX_RUNNING_REQUESTS sequences × MAX_PER_SEQ_BLOCKS blocks each.
    pub kv_pools: Option<Vec<KvPool<B>>>,
    /// Per-request block table + len. Multiple cache_ids index into
    /// the shared kv_pools above.
    pub seq_state: HashMap<String, SeqState<B>>,
    /// Block allocator: tracks which physical blocks are free.
    pub block_allocator: BlockAllocator,
}

pub struct KvPool<B: Backend> {
    pub k: B::Buffer,  // [MAX_BLOCKS, num_kv_heads, block_size, head_dim]
    pub v: B::Buffer,
}

pub struct SeqState<B: Backend> {
    pub block_table: B::Buffer,  // u32[max_blocks_per_seq]
    pub block_table_host: Vec<u32>,  // shadow on host for allocator updates
    pub context_len: usize,
}
```

### LlamaFamilyModel.forward_layer for multi-seq

Currently single-seq:
```rust
B::paged_decode_attention(
    ctx,
    &q_head_major,       // shape [num_heads * tokens * head_dim] for single seq
    &kv_pool.k, &kv_pool.v,
    &mut attn_out,
    &seq.block_table,    // [max_blocks_per_seq]
    &seq.context_lens,   // [1]
    1,                   // num_seqs ← hardcoded
    nh, nkv, hd, block_size, max_blocks_per_seq, q_len,
)
```

Multi-seq:
```rust
// Fan-in N sequences' Q into one batched Q buffer; allocate
// stacked block_table/context_lens.
let stacked_q = stack_per_seq_q(seqs);  // [num_seqs, num_heads, head_dim] for q_len=1
let stacked_block_tables = ...;          // [num_seqs, max_blocks_per_seq]
let stacked_context_lens = ...;          // [num_seqs]

B::paged_decode_attention(
    ctx,
    &stacked_q,
    &kv_pool.k, &kv_pool.v,
    &mut stacked_attn_out,
    &stacked_block_tables,
    &stacked_context_lens,
    seqs.len(),          // num_seqs
    ...
);

// Fan-out attn_out back to per-seq buffers.
unstack_per_seq_o(stacked_attn_out, seqs);
```

The kernel from PR #68 already handles `num_seqs > 1` via `tgpig.z` indexing. The work is in the Rust layer: collecting Q/K/V/block_tables/context_lens across requests into one batched dispatch.

### Block allocator

Simple bitmap or free-list. Block size 16 → for 4096 max ctx per seq, max_blocks_per_seq = 256. With 16 concurrent seqs at 4096 ctx each: total = 4096 blocks. Per-block size for Llama-8B (8 kv heads × 128 dim) = 16 KB. Total pool = 64 MB per layer × 32 layers = 2 GB. Fits comfortably in M1 Max's 32 GB.

Block reclamation when a seq finishes: return its blocks to the free pool.

Prefix sharing (later optimisation): when two seqs share a prefix, the allocator can let them share the same physical blocks for those positions.

### Estimated effort

~700-1000 LOC. Key pieces:
- BlockAllocator (~200 LOC, including tests)
- LlamaFamilyModel restructure (~300-400 LOC)
- Scratch buffer reshape for batched dispatch (~100-200 LOC)
- Engine integration: `ContinuousBatchScheduler` passes seq IDs to model's forward (~100-200 LOC)

## Phase 5: 16-concurrent throughput bench

### Bench harness

`bench/scripts/bench_concurrent.py` is drafted. mistralrs entry already
works (uses `Runner(max_seqs=N, paged_attn=True)` + threaded sends).

llama.cpp side: `llama-server -np N` + parallel HTTP `/v1/completions`.

ferrum side blocked on Phase 4 — needs server-mode multi-seq.

### Metrics

Per concurrency level (1, 4, 8, 16):
- **Aggregate throughput**: total decoded tokens / wall time
- **Per-request decode rate**: decode tps when N requests share GPU
- **TTFT p50 / p99**: time-to-first-token
- **Memory footprint**: peak RSS / swap delta

### Expected results

mistral.rs is the reference for "what's possible" on Mac. Their paged attention + scheduler should give near-linear scaling up to bandwidth cap.

llama.cpp's cell-based KV scales but with more fragmentation — somewhere between 1× and N× depending on context length variance.

ferrum target after Phase 4: match mistral.rs at N=16 for short prompts (4-token + 128-decode). Long context comparisons need fp16 KV cache (Phase 6).

### Group B report

Output format mirrors Group A:
- `bench/group-b-report-XXX.md`
- Table: 3 engines × 3 models × 4 concurrency levels (1/4/8/16) × 3 metrics
- Memory monitoring via existing capture_env.sh

## Stopgaps available before Phase 4

If the user wants a partial concurrent number now (single-seq paged repeated):
- `for i in 1..N; do ferrum run model.gguf ... & done; wait` — N independent processes, each loading the model. Approximates concurrent throughput at the cost of NxModel memory.
- Doesn't validate engine-level batching but shows raw kernel scaling.

## Why Phase 4 is the real unlock

Single-seq paged (Phases 1-3) is a no-op vs contig for performance. The architectural payoff is exclusively from sharing the pool across concurrent requests:
1. **Memory**: 16 seqs × 4 KB-per-block instead of 16 × 1 GB pre-allocated → orders of magnitude less wasted VRAM.
2. **Prefix caching**: same system prompt shared at the block level.
3. **Variable-length batching**: short and long sequences batch together without padding.

Without Phase 4, ferrum on Metal can't run useful server workloads — it'd OOM on a 32 GB Mac at 4-5 concurrent Llama-8B requests.

## Recommended order

1. Land PRs #68/#69/#70 (kernels + single-seq integration).
2. Phase 4 in a single PR (~1 week of focused work).
3. Run bench_concurrent.py for all 3 engines, write Group B report.
4. Optional Phase 6: fp16 KV cache for long-context efficiency.
