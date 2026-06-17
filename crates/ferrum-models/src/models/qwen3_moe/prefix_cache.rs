use super::*;

impl<B: MoeLlmBackend, K: KvDtypeKind> Qwen3MoeModel<B, K> {
    pub(crate) fn record_prefix_cache_probe(&mut self, saved_tokens: usize) {
        if saved_tokens > 0 {
            self.prefix_cache_hits += 1;
            self.prefix_cache_saved_prefill_tokens += saved_tokens as u64;
        } else {
            self.prefix_cache_misses += 1;
        }
    }

    pub(crate) fn prefix_cache_snapshot_json(&self) -> serde_json::Value {
        let (entries, block_size) = self
            .paged_block_alloc
            .as_ref()
            .and_then(|alloc| {
                let alloc = alloc.lock().ok()?;
                let block_size = self
                    .kv_caches
                    .values()
                    .find_map(|layers| layers.first().map(|cache| cache.block_size))
                    .unwrap_or(16);
                Some((alloc.hash_table_size() as u64, block_size))
            })
            .unwrap_or((0, 16));
        let bytes_per_entry = (block_size
            * self.cfg.base.num_layers
            * self.cfg.base.num_kv_heads
            * self.cfg.base.head_dim
            * K::BYTES_PER_ELEM
            * 2) as u64;
        serde_json::json!({
            "position": "real-kv-reuse",
            "source": "qwen3-moe-paged-block-prefix-cache",
            "enabled": self.runtime_env.prefix_cache,
            "hits": self.prefix_cache_hits,
            "misses": self.prefix_cache_misses,
            "evictions": 0u64,
            "saved_prefill_tokens": self.prefix_cache_saved_prefill_tokens,
            "entries": entries,
            "bytes": entries.saturating_mul(bytes_per_entry),
            "block_size": block_size,
            "kv_dtype": K::NAME,
        })
    }

    /// Block-level prefix cache: try to splice cached prefix blocks into
    /// `cache_id`'s KV state. Hashes `tokens` block-by-block, looks each
    /// hash up in the shared `paged_block_alloc`'s hash table, and on
    /// hit:
    ///   1. acquires the cached physical block (ref+=1, resurrecting from
    ///      soft-free if needed)
    ///   2. swaps the fresh block (from prior `ensure_kv`) out of this
    ///      seq's `block_indices[i]`, returns it to the pool
    ///   3. writes the cached block id into `block_indices[i]` instead
    ///
    /// Returns the number of tokens that were already cached. After this
    /// call the cache_id has `cache.len = returned * block_size`, so
    /// `prefill_internal` reading `pos_offset` from `cache.len` naturally
    /// continues from where prefix cache left off — the caller's prefill
    /// only needs to process `tokens[returned..]`.
    ///
    /// MUST be called after `ensure_kv(cache_id)`. Returns 0 if non-paged
    /// or paged_block_alloc is None.
    pub(crate) fn try_acquire_prefix_cache(&mut self, cache_id: &str, tokens: &[u32]) -> usize {
        let Some(alloc_arc) = self.paged_block_alloc.as_ref() else {
            return 0;
        };
        let caches = match self.kv_caches.get(cache_id) {
            Some(c) => c,
            None => return 0,
        };
        let block_size = caches.first().map(|c| c.block_size).unwrap_or(0);
        if block_size == 0 {
            return 0;
        }

        // Hash chain on input tokens (only full blocks — trailing partial
        // block can't be cached as standalone).
        let token_ids: Vec<ferrum_types::TokenId> = tokens
            .iter()
            .map(|&t| ferrum_types::TokenId::new(t))
            .collect();
        let hashes: Vec<BlockHash> = block_hash_chain(&token_ids, block_size);
        if hashes.is_empty() {
            return 0;
        }

        // Probe matches from the front. Stop at first miss — we want the
        // longest contiguous prefix; gaps would break the kv_len contract.
        let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
        let mut matched: Vec<u32> = Vec::with_capacity(hashes.len());
        for &h in &hashes {
            match alloc.try_acquire_by_hash(h) {
                Some(b) => matched.push(b),
                None => break,
            }
        }
        if matched.is_empty() {
            return 0;
        }
        let n_matched = matched.len();

        // Free the freshly-allocated blocks that we're displacing —
        // those `block_indices[0..n_matched]` slots get the cached IDs
        // instead.
        let displaced: Vec<u32> = caches
            .first()
            .map(|c| {
                let existing = &c.paged_block_indices;
                existing[..n_matched.min(existing.len())].to_vec()
            })
            .unwrap_or_default();
        alloc.free(&displaced);
        drop(alloc);

        // Splice matched into each layer's cache state and mirror to the
        // device block_table buffer + context_lens.
        let caches_mut = self.kv_caches.get_mut(cache_id).expect("cache present");
        let max_blocks = caches_mut
            .first()
            .map(|c| c.capacity / c.block_size)
            .unwrap_or(0);
        let new_len = n_matched * block_size;
        let mut ctx_tmp = B::new_context();
        for c in caches_mut.iter_mut() {
            // Replace first n_matched entries with cached IDs.
            if c.paged_block_indices.len() < matched.len() {
                c.paged_block_indices.resize(matched.len(), 0);
            }
            for (i, &b) in matched.iter().enumerate() {
                c.paged_block_indices[i] = b;
            }
            c.len = new_len;
            if let Some(bt) = c.block_table.as_mut() {
                let mut padded = c.paged_block_indices.clone();
                padded.resize(max_blocks, 0);
                B::write_typed::<u32>(&mut ctx_tmp, bt, &padded);
            }
            if let Some(cl) = c.context_lens.as_mut() {
                B::write_typed::<u32>(&mut ctx_tmp, cl, &[new_len as u32]);
            }
        }
        B::sync(&mut ctx_tmp);

        new_len
    }

    /// Register block hashes for content that was just written by the
    /// prefill kernel. Called AFTER `prefill_internal`'s forward pass
    /// completes so the resulting blocks can be cache-hit by future
    /// requests with the same prompt prefix.
    ///
    /// `all_tokens` is the FULL prompt (the same `tokens` passed to
    /// prefill, before prefix-cache slicing).
    /// `prior_cached_tokens` is the count returned by
    /// `try_acquire_prefix_cache` — those blocks already had their hashes
    /// registered (we just acquired them); we only need to register the
    /// NEW blocks past that point.
    pub(crate) fn register_prefix_cache(
        &mut self,
        cache_id: &str,
        all_tokens: &[u32],
        prior_cached_tokens: usize,
    ) {
        let Some(alloc_arc) = self.paged_block_alloc.as_ref() else {
            return;
        };
        let caches = match self.kv_caches.get(cache_id) {
            Some(c) => c,
            None => return,
        };
        let cache0 = match caches.first() {
            Some(c) => c,
            None => return,
        };
        let block_size = cache0.block_size;
        if block_size == 0 {
            return;
        }

        let token_ids: Vec<ferrum_types::TokenId> = all_tokens
            .iter()
            .map(|&t| ferrum_types::TokenId::new(t))
            .collect();
        let hashes: Vec<BlockHash> = block_hash_chain(&token_ids, block_size);
        if hashes.is_empty() {
            return;
        }

        let start_block = prior_cached_tokens / block_size;
        let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
        for i in start_block..hashes.len().min(cache0.paged_block_indices.len()) {
            // Only register blocks whose content actually got written.
            // After prefill the cache covers `all_tokens.len()` tokens;
            // a hash at index i represents block i which holds
            // tokens[i*bs .. (i+1)*bs]. That block is "fully written"
            // iff (i+1)*bs <= cache.len (the actual fully-prefilled
            // position). If only a partial block was written we don't
            // register it (its content depends on the next prefill).
            let block_end_token = (i + 1) * block_size;
            if block_end_token > cache0.len {
                break;
            }
            let block_id = cache0.paged_block_indices[i];
            alloc.register_block_hash(block_id, hashes[i]);
        }
    }
}
