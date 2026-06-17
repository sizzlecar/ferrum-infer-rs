use super::*;

impl<B: MoeLlmBackend, K: KvDtypeKind> Qwen3MoeModel<B, K> {
    pub(crate) fn ensure_scratch(&mut self, tokens: usize) {
        if self.scratch.max_tokens < tokens {
            {
                let mut ctx = B::new_context();
                B::reset_all_graphs(&mut ctx);
            }
            // Scratch realloc invalidates captured graph addresses —
            // clear the cache so the next decode_batch starts a fresh
            // capture cycle.
            self.batched_graph_keys_seen.clear();
            self.batched_graph_warmup = 0;
            self.batched_graph_failed = false;
            self.scratch = Qwen3MoeScratch::alloc(&self.cfg, tokens);
            // Realloc wiped paged_batch_*. Re-enable using the dims
            // pinned at first ensure_kv. Without this, the next
            // `forward_layer_batched_decode` panics on
            // `paged_batch_block_tables missing` (regression manifests
            // at c≥16 when batch growth triggers scratch realloc
            // between `ensure_kv` and the batched-decode entry point).
            if let Some((max_seqs, max_blocks_per_seq)) = self.paged_dims {
                self.scratch
                    .enable_paged_batch(&self.cfg, max_seqs, max_blocks_per_seq);
            }
        }
    }

    pub(crate) fn ensure_kv(&mut self, cache_id: &str) {
        if self.kv_caches.contains_key(cache_id) {
            return;
        }
        let nkv = self.cfg.base.num_kv_heads;
        let hd = self.cfg.base.head_dim;
        // 512 in 0.7.2 — same value the published bench used to hit 79
        // tok/s at c=16 on this exact MoE model. See
        // `LlamaFamilyModel::ensure_kv` for the full rationale.
        let model_max = self.cfg.base.max_seq_len;
        let max = self.runtime_env.kv_capacity(model_max);

        // Paged-KV mode: `FERRUM_METAL_PAGED_KV=1` switches caches into
        // block-table-indirect layout. Mirrors LlamaFamilyModel's path so
        // the existing `paged_decode_attention` Metal kernel can fire
        // once at num_seqs=m for batched decode (replacing the per-item
        // attention loop that currently dominates `attn_peritem` in the
        // c=16 profile).
        // Default ON when the backend supports paged-KV (Metal). Users
        // can force off with `FERRUM_METAL_PAGED_KV=0`. The flag was
        // opt-in pre-0.7.2; flipping the default so default `ferrum
        // serve` matches the bench-quality numbers without requiring
        // env-var knowledge.
        let paged = self
            .runtime_env
            .metal_paged_kv_enabled(B::supports_paged_kv());
        const PAGED_BLOCK_SIZE: usize = 16;

        // Paged KV uses a vLLM-style global physical block pool. `max`
        // defines the per-sequence logical table stride; `kv_max_blocks`
        // defines the physical pool size. Sequences grow their block lists
        // on demand before each KV write.
        let max_seqs = self.runtime_env.paged_max_seqs;
        let max_blocks_per_seq = max.div_ceil(PAGED_BLOCK_SIZE);
        let total_pool_blocks = self.runtime_env.paged_total_blocks(max_blocks_per_seq);

        // Lazy-allocate the shared paged pools on the first paged
        // ensure_kv call.
        if paged && self.paged_pools.is_none() {
            let mut pools = Vec::with_capacity(self.cfg.base.num_layers);
            for _ in 0..self.cfg.base.num_layers {
                let pool_floats = total_pool_blocks * nkv * PAGED_BLOCK_SIZE * hd;
                pools.push((B::alloc(pool_floats), B::alloc(pool_floats)));
            }
            self.paged_pools = Some(pools);
            self.paged_block_alloc = Some(std::sync::Mutex::new(
                crate::common::paged_pool::BlockAllocator::new(total_pool_blocks as u32),
            ));
        }
        if paged
            && self.use_vllm_paged_attn
            && (self.runtime_env.fa_layout_varlen
                || self.runtime_env.fa2_direct_ffi
                || self.runtime_env.fa2_source)
            && self.paged_fa_pools.is_none()
        {
            let mut pools = Vec::with_capacity(self.cfg.base.num_layers);
            for _ in 0..self.cfg.base.num_layers {
                let pool_floats = total_pool_blocks * nkv * PAGED_BLOCK_SIZE * hd;
                pools.push((B::alloc(pool_floats), B::alloc(pool_floats)));
            }
            self.paged_fa_pools = Some(pools);
        }
        if paged {
            self.scratch
                .enable_paged_batch(&self.cfg, max_seqs, max_blocks_per_seq);
            // Pin dims on the model so `ensure_scratch`'s realloc can
            // re-call `enable_paged_batch` after wiping scratch.
            self.paged_dims = Some((max_seqs, max_blocks_per_seq));
        }

        let mut caches = self.kv_free_pool.pop().unwrap_or_else(|| {
            (0..self.cfg.base.num_layers)
                .map(|_| {
                    if paged {
                        // Paged mode: cache holds metadata only. K/V are
                        // 1-element placeholders. Real data lives in
                        // `self.paged_pools[li].{k,v}`.
                        let mut block_table =
                            B::alloc_typed(ferrum_kernels::backend::Dtype::U32, max_blocks_per_seq);
                        let _ = &mut block_table; // suppress unused-mut on backends that no-op write_u32
                        let mut context_lens =
                            B::alloc_typed(ferrum_kernels::backend::Dtype::U32, 1);
                        let mut bt_ctx = B::new_context();
                        B::write_typed::<u32>(&mut bt_ctx, &mut context_lens, &[0u32]);
                        B::sync(&mut bt_ctx);
                        KvCache {
                            k: B::alloc(1),
                            v: B::alloc(1),
                            len: 0,
                            capacity: max_blocks_per_seq * PAGED_BLOCK_SIZE,
                            num_kv_heads: nkv,
                            head_dim: hd,
                            block_size: PAGED_BLOCK_SIZE,
                            block_table: Some(block_table),
                            context_lens: Some(context_lens),
                            paged_block_indices: Vec::new(),
                            _kv_dtype: std::marker::PhantomData,
                        }
                    } else {
                        KvCache {
                            k: B::alloc(nkv * max * hd),
                            v: B::alloc(nkv * max * hd),
                            len: 0,
                            capacity: max,
                            num_kv_heads: nkv,
                            head_dim: hd,
                            block_size: 0,
                            block_table: None,
                            context_lens: None,
                            paged_block_indices: Vec::new(),
                            _kv_dtype: std::marker::PhantomData,
                        }
                    }
                })
                .collect()
        });

        // Physical blocks are allocated on demand before each forward writes
        // KV. At creation time the cache owns no blocks.
        if paged {
            let padded = vec![0u32; max_blocks_per_seq];
            let mut ctx_tmp = B::new_context();
            for c in caches.iter_mut() {
                if let Some(bt) = c.block_table.as_mut() {
                    B::write_typed::<u32>(&mut ctx_tmp, bt, &padded);
                }
                c.paged_block_indices.clear();
            }
            B::sync(&mut ctx_tmp);
        }

        for c in caches.iter_mut() {
            c.len = 0;
            if let Some(cl) = c.context_lens.as_mut() {
                let mut ctx_tmp = B::new_context();
                B::write_typed::<u32>(&mut ctx_tmp, cl, &[0u32]);
                B::sync(&mut ctx_tmp);
            }
        }
        self.kv_caches.insert(cache_id.to_string(), caches);
    }

    pub(crate) fn ensure_paged_kv_capacity_for_cache_id(
        &mut self,
        ctx: &mut B::Context,
        cache_id: &str,
        target_len: usize,
    ) -> Result<()> {
        if self.paged_pools.is_none() {
            return Ok(());
        }

        let (block_size, max_blocks_per_seq, mut block_indices) = {
            let caches = self.kv_caches.get(cache_id).ok_or_else(|| {
                FerrumError::model(format!(
                    "paged KV grow called before ensure_kv for cache_id={cache_id:?}"
                ))
            })?;
            let cache = caches.first().ok_or_else(|| {
                FerrumError::model(format!(
                    "paged KV grow found empty layer cache for cache_id={cache_id:?}"
                ))
            })?;
            if cache.block_size == 0 {
                return Ok(());
            }
            (
                cache.block_size,
                cache.capacity / cache.block_size,
                cache.paged_block_indices.clone(),
            )
        };

        let needed_blocks = target_len.div_ceil(block_size);
        if needed_blocks > max_blocks_per_seq {
            return Err(FerrumError::model(format!(
                "paged KV: target_len={target_len} needs {needed_blocks} blocks, exceeds per-seq table capacity {max_blocks_per_seq} for cache_id={cache_id:?}"
            )));
        }
        if block_indices.len() >= needed_blocks {
            return Ok(());
        }

        let extra_blocks = needed_blocks - block_indices.len();
        let new_blocks = {
            let alloc_arc = self.paged_block_alloc.as_ref().ok_or_else(|| {
                FerrumError::model("paged KV grow missing block allocator while paged_pools is set")
            })?;
            let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
            alloc.allocate_n(extra_blocks)?
        };
        block_indices.extend(new_blocks);

        let mut padded = block_indices.clone();
        padded.resize(max_blocks_per_seq, 0);
        let caches = self.kv_caches.get_mut(cache_id).ok_or_else(|| {
            FerrumError::model(format!(
                "paged KV grow lost cache after allocation for cache_id={cache_id:?}"
            ))
        })?;
        for cache in caches.iter_mut() {
            cache.paged_block_indices = block_indices.clone();
            if let Some(block_table) = cache.block_table.as_mut() {
                B::write_typed::<u32>(ctx, block_table, &padded);
            }
        }

        Ok(())
    }

    pub(crate) fn reserve_paged_kv_slots(
        &mut self,
        requests: &[KvSlotRequest],
    ) -> Result<Option<KvSlotReservation>> {
        if requests.is_empty() {
            return Ok(None);
        }

        let mut targets: Vec<(String, usize)> = Vec::new();
        for request in requests {
            if let Some((_, target)) = targets
                .iter_mut()
                .find(|(cache_id, _)| cache_id == &request.cache_id)
            {
                *target = (*target).max(request.target_len);
            } else {
                targets.push((request.cache_id.clone(), request.target_len));
            }
        }

        for (cache_id, _) in &targets {
            self.ensure_kv(cache_id);
        }
        if self.paged_pools.is_none() {
            return Ok(None);
        }

        let mut plans = Vec::with_capacity(targets.len());
        let mut block_size = 0usize;
        let mut total_new_blocks = 0usize;
        for (cache_id, target_len) in &targets {
            let caches = self.kv_caches.get(cache_id).ok_or_else(|| {
                FerrumError::model(format!(
                    "paged KV reservation missing cache for cache_id={cache_id:?}"
                ))
            })?;
            let cache = caches.first().ok_or_else(|| {
                FerrumError::model(format!(
                    "paged KV reservation found empty layer cache for cache_id={cache_id:?}"
                ))
            })?;
            if cache.block_size == 0 {
                return Ok(None);
            }
            if block_size == 0 {
                block_size = cache.block_size;
            } else if block_size != cache.block_size {
                return Err(FerrumError::model(format!(
                    "paged KV reservation saw mixed block sizes: {block_size} and {}",
                    cache.block_size
                )));
            }

            let max_blocks_per_seq = cache.capacity / cache.block_size;
            let blocks_before = cache.paged_block_indices.len();
            let blocks_after = target_len.div_ceil(cache.block_size);
            if blocks_after > max_blocks_per_seq {
                return Err(FerrumError::model(format!(
                    "paged KV reservation: target_len={target_len} needs {blocks_after} blocks, exceeds per-seq table capacity {max_blocks_per_seq} for cache_id={cache_id:?}"
                )));
            }
            let new_blocks = blocks_after.saturating_sub(blocks_before);
            total_new_blocks += new_blocks;
            plans.push(KvSlotAllocation {
                cache_id: cache_id.clone(),
                blocks_before,
                blocks_after,
                new_blocks,
            });
        }

        let alloc_arc = self.paged_block_alloc.as_ref().ok_or_else(|| {
            FerrumError::model(
                "paged KV reservation missing block allocator while paged_pools is set",
            )
        })?;
        let free_blocks_before = {
            let alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
            alloc.free_count()
        };
        if total_new_blocks > free_blocks_before {
            return Err(FerrumError::resource_exhausted(format!(
                "paged KV admission: need {total_new_blocks} new blocks but only {free_blocks_before} free"
            )));
        }

        let mut ctx = B::new_context();
        for plan in &plans {
            if plan.new_blocks == 0 {
                continue;
            }
            let new_blocks = {
                let alloc_arc = self.paged_block_alloc.as_ref().ok_or_else(|| {
                    FerrumError::model(
                        "paged KV reservation lost block allocator while paged_pools is set",
                    )
                })?;
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                alloc.allocate_n(plan.new_blocks)?
            };
            let caches = self.kv_caches.get_mut(&plan.cache_id).ok_or_else(|| {
                FerrumError::model(format!(
                    "paged KV reservation lost cache for cache_id={:?}",
                    plan.cache_id
                ))
            })?;
            let max_blocks_per_seq = caches
                .first()
                .map(|cache| cache.capacity / block_size)
                .unwrap_or(0);
            let mut block_indices = caches
                .first()
                .map(|cache| cache.paged_block_indices.clone())
                .unwrap_or_default();
            block_indices.extend(new_blocks);
            let mut padded = block_indices.clone();
            padded.resize(max_blocks_per_seq, 0);
            for cache in caches.iter_mut() {
                cache.paged_block_indices = block_indices.clone();
                if let Some(block_table) = cache.block_table.as_mut() {
                    B::write_typed::<u32>(&mut ctx, block_table, &padded);
                }
            }
        }
        B::sync(&mut ctx);

        let (total_blocks, free_blocks_after) = {
            let alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
            (alloc.capacity() as usize, alloc.free_count())
        };
        Ok(Some(KvSlotReservation {
            block_size,
            total_blocks,
            free_blocks_before,
            free_blocks_after,
            allocations: plans,
        }))
    }
}
