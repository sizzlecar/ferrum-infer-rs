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

        // Default 32: covers c=16 burst with 2× headroom for the
        // fresh-cache-id-per-request pattern that bench/server harnesses
        // use. Pool memory unchanged from pre-0.7.2 default because
        // DEFAULT_KV_CAPACITY dropped 4096 → 2048 in lockstep.
        let max_seqs = self.runtime_env.paged_max_seqs;
        let max_blocks_per_seq = max.div_ceil(PAGED_BLOCK_SIZE);
        let total_pool_blocks = max_seqs * max_blocks_per_seq;

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
            && (self.runtime_env.fa_layout_varlen || self.runtime_env.fa2_direct_ffi)
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

        // Allocate physical blocks for THIS cache_id from the shared pool.
        if paged {
            let alloc_arc = self
                .paged_block_alloc
                .as_ref()
                .expect("paged_block_alloc must be initialised when paged=true");
            let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
            let block_indices = match alloc.allocate_n(max_blocks_per_seq) {
                Ok(idx) => idx,
                Err(e) => {
                    drop(alloc);
                    self.kv_free_pool.push(caches);
                    eprintln!(
                        "[ferrum] paged KV pool exhausted on ensure_kv for \
                         cache_id={cache_id:?}: {e}. Increase \
                         FERRUM_PAGED_MAX_SEQS (currently {max_seqs}) or \
                         throttle concurrent requests.",
                    );
                    return;
                }
            };
            let mut padded = block_indices.clone();
            padded.resize(max_blocks_per_seq, 0);
            let mut ctx_tmp = B::new_context();
            for c in caches.iter_mut() {
                if let Some(bt) = c.block_table.as_mut() {
                    B::write_typed::<u32>(&mut ctx_tmp, bt, &padded);
                }
                c.paged_block_indices = block_indices.clone();
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
}
