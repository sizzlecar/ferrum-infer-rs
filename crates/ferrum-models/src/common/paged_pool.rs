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
//! What's *not* here yet (deferred to Phase 4c / 5):
//! - Prefix sharing: today a fresh `PagedSeqState` always allocates new
//!   blocks even if its prompt overlaps another live sequence's.
//! - Eviction / preemption: when blocks run out we just `Err`. A real
//!   scheduler would either refuse-and-queue or evict the least-recently-used.
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
        }
    }

    /// Allocate a single physical block. Returns `Err` when the pool is
    /// exhausted — caller is expected to refuse the request and queue
    /// it (or evict another seq, when that's wired up).
    pub fn allocate(&mut self) -> Result<u32> {
        match self.free_list.pop() {
            Some(b) => {
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
    pub fn allocate_n(&mut self, n: usize) -> Result<Vec<u32>> {
        if self.free_list.len() < n {
            return Err(FerrumError::resource_exhausted(format!(
                "paged KV pool exhausted: need {n} blocks but only {} free",
                self.free_list.len()
            )));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.free_list.pop().unwrap());
        }
        let in_use = self.capacity as usize - self.free_list.len();
        self.peak_in_use.fetch_max(in_use, Ordering::Relaxed);
        Ok(out)
    }

    /// Return blocks to the free list. Caller is responsible for
    /// ensuring no live sequence still references them; freeing a block
    /// while it's still in a `PagedSeqState::blocks` will silently
    /// corrupt the next allocation that gets it.
    pub fn free(&mut self, blocks: &[u32]) {
        self.free_list.extend_from_slice(blocks);
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
        let block_table_buf = B::alloc_u32(max_blocks_per_seq);
        let context_lens_buf = B::alloc_u32(1);
        // Initialise context_lens to 0 so a forward dispatched before
        // any token has been written sees an empty context.
        let mut ctx = B::new_context();
        let mut cl = context_lens_buf;
        B::write_u32(&mut ctx, &mut cl, &[0u32]);
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
        B::write_u32(ctx, &mut self.block_table_buf, &padded);
        Ok(())
    }

    /// Update the on-device `context_lens_buf` to the current `self.len`.
    /// Call this after [`Self::ensure_capacity`] but before dispatching
    /// the paged attention kernel for this seq.
    pub fn sync_context_len(&mut self, ctx: &mut B::Context) {
        B::write_u32(ctx, &mut self.context_lens_buf, &[self.len as u32]);
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
}
