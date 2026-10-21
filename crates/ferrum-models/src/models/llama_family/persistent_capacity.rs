//! Complete original-owner proof over allocated paged payload and metadata.

use super::*;
use crate::common::llm::ModelPersistentCapacityQuery;
use ferrum_execution::model_executor::{
    PersistentCapacityUnknown as U, MAX_PERSISTENT_CAPACITY_MEMBERS,
};

fn bytes<B: Backend>(buffer: &B::Buffer, required: usize) -> std::result::Result<(), U> {
    match B::buffer_storage_bytes(buffer) {
        Some(actual) if actual >= required => Ok(()),
        Some(_) => Err(U::InsufficientBacking),
        None => Err(U::Unsupported),
    }
}

impl<B: MoeLlmBackend> LlamaFamilyModel<B, KvFp16> {
    pub(super) fn observe_persistent_capacity(
        &self,
        query: &ModelPersistentCapacityQuery<'_>,
    ) -> std::result::Result<(), U> {
        let budget = query.inspection_budget;
        budget.validate()?;
        if query.through_tokens.is_empty()
            || query.through_tokens.len() > MAX_PERSISTENT_CAPACITY_MEMBERS
            || query.through_tokens.contains(&0)
            || query.max_live_sequences == 0
            || query.max_live_sequences > MAX_PERSISTENT_CAPACITY_MEMBERS
            || query.max_wave_sequences == 0
            || query.max_wave_sequences > query.max_live_sequences
            || query.residents.len() > query.max_live_sequences
            || query.residents.len() > query.through_tokens.len()
        {
            return Err(U::InvalidQuery);
        }
        if self.pending_device_frame.is_some()
            || self.execution_context.is_none()
            || self.unified_graph_retirement_pending
        {
            return Err(U::PendingOwner);
        }
        if !self.lora_cache_adapters.is_empty()
            || self.source_layer_range() != (0..self.cfg.num_layers)
        {
            return Err(U::Unsupported);
        }
        if self.kv_caches.len() != query.residents.len()
            || self
                .prepared_prefixes
                .keys()
                .any(|key| !self.kv_caches.contains_key(key))
        {
            return Err(U::MembershipChanged);
        }
        let (max_seqs, blocks_per_seq) = self.paged_dims.ok_or(U::UnsupportedPacking)?;
        // The physical admission ceiling must cover this same concurrency
        // contract; a scheduler-only bound cannot constrain retiring owners.
        if max_seqs != self.runtime_env.paged_max_seqs || max_seqs != query.max_live_sequences {
            return Err(U::UnsupportedPacking);
        }
        let slots = self
            .kv_caches
            .len()
            .checked_add(self.kv_free_pool.len())
            .ok_or(U::Arithmetic)?;
        if slots > max_seqs {
            return Err(U::InvalidOwner);
        }
        if slots < max_seqs.min(query.through_tokens.len()) {
            return Err(U::InsufficientBacking);
        }
        let layers = self.local_layer_count();
        let pools = self.paged_pools.as_ref().ok_or(U::InsufficientBacking)?;
        if layers == 0 || pools.len() != layers {
            return Err(U::InvalidOwner);
        }
        if layers > budget.max_pools
            || slots
                .checked_mul(layers)
                .and_then(|n| n.checked_add(layers + 2))
                .is_none_or(|n| n > budget.max_descriptors)
        {
            return Err(U::InspectionBudget);
        }
        let table_bytes = blocks_per_seq.checked_mul(4).ok_or(U::Arithmetic)?;
        let capacity = blocks_per_seq
            .checked_mul(LLAMA_PAGED_BLOCK_SIZE)
            .ok_or(U::Arithmetic)?;
        if blocks_per_seq == 0
            || query
                .through_tokens
                .iter()
                .any(|&n| n > self.kv_capacity_limit() || n > capacity)
        {
            return Err(U::InsufficientBacking);
        }
        // Persistent dispatch tables are actual buffers, not just dimensions
        // recorded when an earlier scratch object was allocated.
        bytes::<B>(
            self.scratch
                .paged_batch_block_tables
                .as_ref()
                .ok_or(U::InsufficientBacking)?,
            table_bytes.checked_mul(max_seqs).ok_or(U::Arithmetic)?,
        )?;
        bytes::<B>(
            self.scratch
                .paged_batch_context_lens
                .as_ref()
                .ok_or(U::InsufficientBacking)?,
            max_seqs.checked_mul(4).ok_or(U::Arithmetic)?,
        )?;
        let alloc = self
            .paged_block_alloc
            .as_ref()
            .ok_or(U::InsufficientBacking)?
            .try_lock()
            .map_err(|_| U::Contended)?;
        let physical_blocks = alloc.capacity() as usize;
        let scalar = match <KvFp16 as KvLayer<B>>::physical_storage() {
            Some(ferrum_types::KvCacheStorage::F32) => 4usize,
            Some(ferrum_types::KvCacheStorage::F16) => 2usize,
            _ => return Err(U::Unsupported),
        };
        let payload_bytes = physical_blocks
            .checked_mul(LLAMA_PAGED_BLOCK_SIZE)
            .and_then(|n| n.checked_mul(self.cfg.num_kv_heads))
            .and_then(|n| n.checked_mul(self.cfg.head_dim))
            .and_then(|n| n.checked_mul(scalar))
            .ok_or(U::Arithmetic)?;
        for (k, v) in pools {
            bytes::<B>(k, payload_bytes)?;
            bytes::<B>(v, payload_bytes)?;
        }
        // Bound all extent inspection and scratch before allocation. Hash
        // entries are intentionally irrelevant: soft-free hashes pin nothing.
        let mut extents = physical_blocks.checked_mul(3).ok_or(U::Arithmetic)?;
        for caches in self.kv_caches.values().chain(self.kv_free_pool.iter()) {
            if caches.len() != layers {
                return Err(U::InvalidOwner);
            }
            for cache in caches {
                extents = extents
                    .checked_add(cache.paged_block_indices.len())
                    .ok_or(U::Arithmetic)?;
            }
        }
        let scratch = physical_blocks
            .checked_mul(std::mem::size_of::<u16>() + std::mem::size_of::<usize>())
            .and_then(|n| {
                n.checked_add(MAX_PERSISTENT_CAPACITY_MEMBERS * (std::mem::size_of::<usize>() + 1))
            })
            .ok_or(U::Arithmetic)?;
        if extents > budget.max_extents || scratch > budget.max_scratch_bytes {
            return Err(U::InspectionBudget);
        }
        let mut refs = Vec::<u16>::new();
        let mut seen = Vec::<usize>::new();
        refs.try_reserve_exact(physical_blocks)
            .map_err(|_| U::InspectionBudget)?;
        seen.try_reserve_exact(physical_blocks)
            .map_err(|_| U::InspectionBudget)?;
        refs.resize(physical_blocks, 0);
        seen.resize(physical_blocks, 0);
        let mut matched = [false; MAX_PERSISTENT_CAPACITY_MEMBERS];
        for (index, resident) in query.residents.iter().enumerate() {
            if query.residents[..index]
                .iter()
                .any(|prior| prior.cache_id == resident.cache_id)
                || resident.consumed_tokens > resident.through_tokens
            {
                return Err(U::InvalidQuery);
            }
            let position = query
                .through_tokens
                .iter()
                .enumerate()
                .position(|(i, n)| !matched[i] && *n == resident.through_tokens)
                .ok_or(U::InvalidQuery)?;
            matched[position] = true;
            let caches = self
                .kv_caches
                .get(resident.cache_id)
                .ok_or(U::MembershipChanged)?;
            let mapping = &caches[0].paged_block_indices;
            if mapping.len() > resident.through_tokens.div_ceil(LLAMA_PAGED_BLOCK_SIZE) {
                return Err(U::InvalidOwner);
            }
            for cache in caches {
                self.check_capacity_cache(cache, capacity, table_bytes)?;
                if cache.len != resident.consumed_tokens
                    || cache.paged_block_indices != *mapping
                    || cache.len
                        > mapping
                            .len()
                            .checked_mul(LLAMA_PAGED_BLOCK_SIZE)
                            .ok_or(U::Arithmetic)?
                {
                    return Err(U::InvalidOwner);
                }
            }
            for &block in mapping {
                let block = block as usize;
                if block >= physical_blocks || seen[block] == index + 1 {
                    return Err(U::InvalidOwner);
                }
                seen[block] = index + 1;
                refs[block] = refs[block].checked_add(1).ok_or(U::Arithmetic)?;
            }
        }
        for caches in &self.kv_free_pool {
            for cache in caches {
                self.check_capacity_cache(cache, capacity, table_bytes)?;
                if !cache.paged_block_indices.is_empty() {
                    return Err(U::InvalidOwner);
                }
            }
        }
        if !alloc.matches_complete_references(&refs, &mut seen) {
            return Err(U::InvalidOwner);
        }
        let mut largest = [0usize; MAX_PERSISTENT_CAPACITY_MEMBERS];
        for (out, &tokens) in largest.iter_mut().zip(query.through_tokens) {
            *out = tokens.div_ceil(LLAMA_PAGED_BLOCK_SIZE);
        }
        largest[..query.through_tokens.len()].sort_unstable_by(|a, b| b.cmp(a));
        let demand = largest[..max_seqs.min(query.through_tokens.len())]
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            .ok_or(U::Arithmetic)?;
        if demand > physical_blocks {
            return Err(U::InsufficientBacking);
        }
        Ok(())
    }

    fn check_capacity_cache(
        &self,
        cache: &ferrum_kernels::backend::KvCache<B>,
        capacity: usize,
        table_bytes: usize,
    ) -> std::result::Result<(), U> {
        if cache.block_size != LLAMA_PAGED_BLOCK_SIZE
            || cache.capacity != capacity
            || cache.num_kv_heads != self.cfg.num_kv_heads
            || cache.head_dim != self.cfg.head_dim
            || cache.len > capacity
        {
            return Err(U::UnsupportedPacking);
        }
        bytes::<B>(
            cache.block_table.as_ref().ok_or(U::InsufficientBacking)?,
            table_bytes,
        )?;
        bytes::<B>(
            cache.context_lens.as_ref().ok_or(U::InsufficientBacking)?,
            4,
        )
    }
}

#[cfg(test)]
mod tests;
