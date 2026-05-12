//! Multi-sequence paged-KV pool — Phase 4 of Metal paged attention.
//!
//! Replaces the per-cache_id `KvCache` allocation model with a
//! shared-pool architecture matching vLLM / mistral.rs:
//!
//! - **One pool per layer** holds K and V for *all* concurrent sequences.
//!   Sized to `MAX_TOTAL_BLOCKS × num_kv_heads × block_size × head_dim`.
//! - **Per-cache_id state** ([`PagedSeqState`]) carries that sequence's
//!   logical → physical block mapping (`block_table`) plus its current
//!   length (`len`). Multiple cache_ids can index into the same pool
//!   without colliding because their block_tables point at disjoint
//!   physical blocks (or shared ones, if prefix-caching is enabled
//!   later).
//! - **[`BlockAllocator`]** is a free-list owning physical block indices.
//!   `allocate` pops, `free` pushes back. Out-of-memory surfaces as
//!   `Result::Err` so the scheduler can refuse the request rather than
//!   panicking deep in the model forward.
//!
//! ## Ref counting (Phase 1 of block-level prefix cache)
//!
//! Each physical block has a `u16` ref count. `allocate` / `allocate_n`
//! create a block with `ref_count = 1`. `free` decrements; when the count
//! reaches zero the block returns to the free list (physical reuse).
//! New API `acquire(block)` increments — used by the upcoming prefix cache
//! when another sequence wants to share an already-resident block.
//!
//! Backwards compatible: pre-prefix-cache callers don't touch `acquire`,
//! so every block stays at ref=1 and `free` behaves identically to the
//! old single-ref free.
//!
//! ## What's *not* here yet (deferred to Phase 2-4 of prefix cache):
//! - **Block hash chain** + global hash → block_id table.
//! - **Engine integration**: hashing incoming prompt blocks, looking up
//!   matching prefix, splicing the existing blocks into the new seq's
//!   block_table.
//! - **Eviction policy**: when ref hits 0 the block becomes immediately
//!   reusable; an LRU "soft-free" tier would let us keep recently-evicted
//!   blocks hot for opportunistic prefix-cache hits before they're
//!   overwritten by a new allocate.
//! - **Preemption**: when blocks run out we still just `Err`; real
//!   scheduler would refuse-and-queue or evict.
//! - Cross-process or cross-model pooling.

use ferrum_kernels::backend::Backend;
use ferrum_types::{FerrumError, Result};
use std::sync::atomic::{AtomicUsize, Ordering};

/// LIFO free-list block allocator. `O(1)` allocate / free, no fragmentation
/// (all blocks are uniform size).
///
/// `capacity` is the total physical block count baked into the pool at
/// load time. The allocator is independent per-model (block index space
/// is not portable across models).
pub struct BlockAllocator {
    free_list: Vec<u32>,
    capacity: u32,
    /// Watermark: how many blocks have been live at peak, useful for
    /// pool-sizing diagnostics in the bench harness.
    peak_in_use: AtomicUsize,
    /// Per-block reference count. `ref_counts[i] == 0` ⟺ block i is on
    /// the free list. `allocate` sets to 1; `acquire` increments;
    /// `free` decrements and (only when reaching 0) returns the block
    /// to the free list. Width 16 bits — supports up to 65535 concurrent
    /// sharers of the same physical KV block, far past any realistic
    /// continuous batching cap.
    ref_counts: Vec<u16>,
}

impl BlockAllocator {
    /// Create a fresh allocator. All `num_blocks` blocks start free.
    /// Free-list is built so `allocate()` returns block 0 first, then 1,
    /// etc. — predictable for tests and ensures the lower physical
    /// blocks see the most reuse (better cache locality on M1's SLC).
    pub fn new(num_blocks: u32) -> Self {
        let mut free_list: Vec<u32> = (0..num_blocks).collect();
        free_list.reverse(); // pop() yields 0 first
        Self {
            free_list,
            capacity: num_blocks,
            peak_in_use: AtomicUsize::new(0),
            ref_counts: vec![0u16; num_blocks as usize],
        }
    }

    /// Allocate a single physical block. Returns `Err` when the pool is
    /// exhausted — caller is expected to refuse the request and queue
    /// it (or evict another seq, when that's wired up).
    ///
    /// New block starts with `ref_count = 1`. Drop one ref via `free()`
    /// to return it to the pool, or call `acquire()` for additional
    /// sharers.
    pub fn allocate(&mut self) -> Result<u32> {
        match self.free_list.pop() {
            Some(b) => {
                debug_assert!(
                    self.ref_counts[b as usize] == 0,
                    "allocate yielded block {b} with non-zero ref_count {}",
                    self.ref_counts[b as usize]
                );
                self.ref_counts[b as usize] = 1;
                let in_use = self.capacity as usize - self.free_list.len();
                self.peak_in_use.fetch_max(in_use, Ordering::Relaxed);
                Ok(b)
            }
            None => Err(FerrumError::resource_exhausted(format!(
                "paged KV pool exhausted (capacity={} blocks, all in use)",
                self.capacity
            ))),
        }
    }

    /// Bulk allocate. Atomic: either all `n` succeed or none are taken.
    /// Each returned block starts at `ref_count = 1`.
    pub fn allocate_n(&mut self, n: usize) -> Result<Vec<u32>> {
        if self.free_list.len() < n {
            return Err(FerrumError::resource_exhausted(format!(
                "paged KV pool exhausted: need {n} blocks but only {} free",
                self.free_list.len()
            )));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let b = self.free_list.pop().unwrap();
            debug_assert!(
                self.ref_counts[b as usize] == 0,
                "allocate_n yielded block {b} with non-zero ref_count"
            );
            self.ref_counts[b as usize] = 1;
            out.push(b);
        }
        let in_use = self.capacity as usize - self.free_list.len();
        self.peak_in_use.fetch_max(in_use, Ordering::Relaxed);
        Ok(out)
    }

    /// Increment the ref count for an already-allocated block. Used by
    /// prefix-caching paths when a new sequence wants to share a block
    /// that's already resident from a prior request.
    ///
    /// Panics in debug builds if the block is not currently allocated
    /// (ref_count==0) — a release-build path that calls `acquire` on a
    /// free block is a memory-safety bug we'd rather catch loudly.
    pub fn acquire(&mut self, block: u32) {
        let bi = block as usize;
        debug_assert!(
            self.ref_counts[bi] > 0,
            "acquire on block {block} with ref_count=0 (not currently allocated)"
        );
        self.ref_counts[bi] = self.ref_counts[bi]
            .checked_add(1)
            .expect("BlockAllocator ref_count u16 overflow (>65535 sharers)");
    }

    /// Bulk acquire — convenience for prefix-cache hit paths that take a
    /// list of physical block ids to share.
    pub fn acquire_many(&mut self, blocks: &[u32]) {
        for &b in blocks {
            self.acquire(b);
        }
    }

    /// Drop one ref from each block. Blocks whose ref_count hits 0 are
    /// returned to the free list and become available for the next
    /// `allocate*` call.
    ///
    /// Pre-prefix-cache callers see no behavioural change: every block
    /// they hold starts at ref=1 (from `allocate`), and `free` drops it
    /// to 0 — physically frees on the same call, same as before.
    pub fn free(&mut self, blocks: &[u32]) {
        for &b in blocks {
            let bi = b as usize;
            debug_assert!(
                self.ref_counts[bi] > 0,
                "free on block {b} with ref_count=0 (double-free)"
            );
            self.ref_counts[bi] -= 1;
            if self.ref_counts[bi] == 0 {
                self.free_list.push(b);
            }
        }
    }

    /// Read current ref count for a block. 0 means free, ≥1 means in
    /// use by that many sequences.
    #[inline]
    pub fn ref_count(&self, block: u32) -> u16 {
        self.ref_counts[block as usize]
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn peak_in_use(&self) -> usize {
        self.peak_in_use.load(Ordering::Relaxed)
    }
}

/// Per-sequence paged-KV state.
///
/// Holds the logical→physical block mapping for ONE sequence (one
/// `cache_id`) plus its current token count. The mapping is stored as
/// both:
/// - `blocks: Vec<u32>` — the host-side source of truth, used by the
///   block allocator + grow logic.
/// - `block_table_buf: B::Buffer` — a device-side u32 buffer that mirrors
///   `blocks` and is read directly by the paged Metal kernels (PR #68 /
///   #69). Kept in sync via [`Self::ensure_capacity`].
///
/// `context_lens_buf` is a 1-element u32 device buffer holding `len`.
/// The kernel reads it each forward; we update it via `B::write_u32`.
pub struct PagedSeqState<B: Backend> {
    pub blocks: Vec<u32>,
    pub block_table_buf: B::Buffer,
    pub context_lens_buf: B::Buffer,
    pub len: usize,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,
}

impl<B: Backend> PagedSeqState<B> {
    /// Allocate buffers for a sequence that hasn't yet allocated any
    /// blocks. The allocator isn't touched here — the first call to
    /// [`Self::ensure_capacity`] does the real work.
    pub fn new(block_size: usize, max_blocks_per_seq: usize) -> Self {
        let block_table_buf =
            B::alloc_typed(ferrum_kernels::backend::Dtype::U32, max_blocks_per_seq);
        let context_lens_buf = B::alloc_typed(ferrum_kernels::backend::Dtype::U32, 1);
        // Initialise context_lens to 0 so a forward dispatched before
        // any token has been written sees an empty context.
        let mut ctx = B::new_context();
        let mut cl = context_lens_buf;
        B::write_typed::<u32>(&mut ctx, &mut cl, &[0u32]);
        B::sync(&mut ctx);
        Self {
            blocks: Vec::with_capacity(max_blocks_per_seq),
            block_table_buf,
            context_lens_buf: cl,
            len: 0,
            block_size,
            max_blocks_per_seq,
        }
    }

    /// Ensure the seq has enough blocks to hold `target_len` tokens.
    /// Allocates additional blocks from the pool if needed and re-syncs
    /// `block_table_buf` to the device. Idempotent if already big enough.
    pub fn ensure_capacity(
        &mut self,
        ctx: &mut B::Context,
        alloc: &mut BlockAllocator,
        target_len: usize,
    ) -> Result<()> {
        let needed = target_len.div_ceil(self.block_size);
        if needed > self.max_blocks_per_seq {
            return Err(FerrumError::model(format!(
                "paged KV: target_len={target_len} would need {needed} blocks, exceeds max_blocks_per_seq={}",
                self.max_blocks_per_seq
            )));
        }
        while self.blocks.len() < needed {
            let block = alloc.allocate()?;
            self.blocks.push(block);
        }
        // Mirror the host-side blocks list into the device buffer. We
        // write the FULL `max_blocks_per_seq` entries — unused slots
        // beyond `needed` are never read by the kernel (it only walks
        // `[0, ceil(context_len / block_size))`), but writing them
        // keeps the buffer's content predictable.
        let mut padded = self.blocks.clone();
        padded.resize(self.max_blocks_per_seq, 0);
        B::write_typed::<u32>(ctx, &mut self.block_table_buf, &padded);
        Ok(())
    }

    /// Update the on-device `context_lens_buf` to the current `self.len`.
    /// Call this after [`Self::ensure_capacity`] but before dispatching
    /// the paged attention kernel for this seq.
    pub fn sync_context_len(&mut self, ctx: &mut B::Context) {
        B::write_typed::<u32>(ctx, &mut self.context_lens_buf, &[self.len as u32]);
    }

    /// Release all blocks back to the allocator. Buffers are kept (cheap
    /// to reuse for a future cache_id), but blocks become available for
    /// other sequences. Sets `len` back to 0.
    pub fn release(&mut self, alloc: &mut BlockAllocator) {
        alloc.free(&self.blocks);
        self.blocks.clear();
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_basic() {
        let mut a = BlockAllocator::new(4);
        assert_eq!(a.free_count(), 4);
        assert_eq!(a.allocate().unwrap(), 0);
        assert_eq!(a.allocate().unwrap(), 1);
        assert_eq!(a.allocate().unwrap(), 2);
        assert_eq!(a.allocate().unwrap(), 3);
        assert!(a.allocate().is_err());
        assert_eq!(a.free_count(), 0);

        a.free(&[1, 3]);
        assert_eq!(a.free_count(), 2);
        // LIFO: most recently freed comes back first.
        assert_eq!(a.allocate().unwrap(), 3);
        assert_eq!(a.allocate().unwrap(), 1);
    }

    #[test]
    fn allocator_atomic_n_failure() {
        let mut a = BlockAllocator::new(3);
        let _ = a.allocate().unwrap(); // 1 left in free_list... wait, 2 left
        let _ = a.allocate().unwrap();
        // 1 free, asking for 2 should fail without consuming the 1.
        assert!(a.allocate_n(2).is_err());
        assert_eq!(a.free_count(), 1);
    }

    #[test]
    fn allocator_peak_tracking() {
        let mut a = BlockAllocator::new(8);
        let blocks = a.allocate_n(5).unwrap();
        assert_eq!(a.peak_in_use(), 5);
        a.free(&blocks);
        assert_eq!(a.peak_in_use(), 5); // peak doesn't decrease
        let _ = a.allocate_n(3).unwrap();
        assert_eq!(a.peak_in_use(), 5);
    }

    #[test]
    fn refcount_allocate_starts_at_one() {
        let mut a = BlockAllocator::new(4);
        let b = a.allocate().unwrap();
        assert_eq!(a.ref_count(b), 1);
    }

    #[test]
    fn refcount_acquire_increments() {
        let mut a = BlockAllocator::new(4);
        let b = a.allocate().unwrap();
        a.acquire(b);
        a.acquire(b);
        assert_eq!(a.ref_count(b), 3);
    }

    #[test]
    fn refcount_free_decrements_no_physical_release() {
        let mut a = BlockAllocator::new(4);
        let b = a.allocate().unwrap();
        a.acquire(b); // ref=2
        a.free(&[b]); // ref=1 — NOT yet returned to free_list
        assert_eq!(a.ref_count(b), 1);
        assert_eq!(a.free_count(), 3); // only 3 of original 4 blocks free
    }

    #[test]
    fn refcount_free_physical_release_at_zero() {
        let mut a = BlockAllocator::new(4);
        let b = a.allocate().unwrap();
        a.acquire(b); // ref=2
        a.free(&[b]); // ref=1
        a.free(&[b]); // ref=0 → physical release
        assert_eq!(a.ref_count(b), 0);
        assert_eq!(a.free_count(), 4);
    }

    #[test]
    fn refcount_legacy_single_ref_behaviour_unchanged() {
        // Pre-prefix-cache code never calls `acquire`. Verify that
        // allocate+free behaves identically to the old version: block
        // immediately returns to the pool.
        let mut a = BlockAllocator::new(2);
        let b = a.allocate().unwrap();
        assert_eq!(a.free_count(), 1);
        a.free(&[b]);
        assert_eq!(a.free_count(), 2);
        assert_eq!(a.ref_count(b), 0);
        // Re-allocation reuses the slot (LIFO).
        let b2 = a.allocate().unwrap();
        assert_eq!(b2, b);
    }

    #[test]
    fn refcount_bulk_acquire_and_release() {
        let mut a = BlockAllocator::new(8);
        let blocks = a.allocate_n(3).unwrap();
        a.acquire_many(&blocks); // each ref=2
        for &b in &blocks {
            assert_eq!(a.ref_count(b), 2);
        }
        a.free(&blocks); // each ref=1, none physically released
        assert_eq!(a.free_count(), 5);
        a.free(&blocks); // each ref=0, all released
        assert_eq!(a.free_count(), 8);
    }
}
