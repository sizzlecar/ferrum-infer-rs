use super::*;

impl<B: MoeLlmBackend + BackendPagedKv, K: KvDtypeKind> DecoderOnlyLLM for Qwen3MoeModel<B, K> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        // Eager scratch + KV cache grow + a 1-token forward warmup so
        // the first real prefill / decode doesn't pay the cold-start
        // ~25-MTLBuffer scratch alloc + ~96-MTLBuffer KV alloc + Metal
        // pipeline-state first-bind costs (~265 ms total on Qwen3-MoE
        // 30B-A3B / M1 Max). Mirrors what llama-bench's --warmup does
        // (which runs a same-shape forward before the timer).
        self.ensure_scratch(max_tokens);
        self.ensure_kv(cache_id);

        // Warmup forward through all 48 layers under a scratch cache_id
        // so the real `cache_id` starts at pos_offset=0. Token 0 is
        // valid for any tokenizer (BOS or pad).
        const WARMUP_CACHE: &str = "__ferrum_warmup__";
        let _ = self.prefill_internal(WARMUP_CACHE, &[0u32]);
        // Drop the warmup KV cache slot — real cache_id is unaffected.
        if let Some(caches) = self.kv_caches.remove(WARMUP_CACHE) {
            self.kv_free_pool.push(caches);
        }
    }

    fn kv_capacity(&self) -> usize {
        // Mirror the bound `ensure_kv` will use when allocating the cache.
        let model_max = self.cfg.base.max_seq_len;
        self.runtime_env.kv_capacity(model_max)
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        self.prefill_internal(cache_id, tokens)
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.decode_internal(cache_id, token, pos)
    }

    // decode_batch is gated to use the batched path only when it's a
    // measurable win. The crossover depends on M:
    //
    //   - At low M (≤ ~8) the per-item `decode_internal` loop wins
    //     because: (a) it stays at scratch offset 0 (no copy_slice
    //     overhead), (b) it preserves the cross-layer rms_norm fusion
    //     fast path (`weighted_sum_residual_norm_stacked`).
    //   - At high M (≥ ~12) the batched path wins because the dense
    //     GEMM batching (qkv_proj, o_proj, router, lm_head at m=M) and
    //     the prefill-batched MoE dispatch (one `gemm_quant_moe_id` for
    //     all tokens) amortise the ~48-dispatch lost-fusion penalty.
    //
    // Default ON in 0.7.2+. On CUDA with paged KV + vLLM MoE, the
    // crossover is now M=4: 2026-05-28/29 Vast RTX 4090 random-256/128
    // probes saw the old threshold=8 stay on sequential per-token decode
    // (~89-122 tok/s), while threshold=4 measured 425.6 ± 36.6 tok/s.
    // `FERRUM_MOE_BATCHED=0` forces the
    // legacy loop; `FERRUM_MOE_BATCH_THRESHOLD` remains an escape hatch
    // for future hardware/backends.
    fn decode_batch(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        let m = batch.len();
        let opted_in = self.runtime_env.moe_batched_enabled;
        let threshold = self.runtime_env.moe_batch_threshold;
        if opted_in && m >= threshold {
            self.decode_batch_internal(batch)
        } else {
            batch
                .iter()
                .map(|(cid, tok, p)| self.decode(cid, *tok, *p))
                .collect()
        }
    }

    fn unified_forward(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> std::result::Result<Vec<Option<Vec<f32>>>, FerrumError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if self.runtime_env.qwen_unified_trace {
            let lens: Vec<usize> = items.iter().map(|it| it.1.len()).collect();
            let positions: Vec<usize> = items.iter().map(|it| it.2).collect();
            let finals: Vec<bool> = items.iter().map(|it| it.3).collect();
            eprintln!(
                "[qwen-unified] items={} lens={:?} positions={:?} finals={:?} use_vllm_paged_attn={}",
                items.len(),
                lens,
                positions,
                finals,
                self.use_vllm_paged_attn
            );
        }
        if !B::supports_varlen_qkv() {
            return Err(FerrumError::unsupported(
                "Qwen3MoeModel::unified_forward: backend lacks varlen QKV kernels. \
                 Engine will fall back to legacy paths.",
            ));
        }
        // Pure-decode shortcut: every item is q_len=1 + is_final_chunk.
        // For this shape, ferrum's legacy `forward_layer_batched_decode`
        // path (with FERRUM_MOE_GRAPH=1 graph capture + decode-tuned
        // moe_forward_stacked) is faster than our generic varlen +
        // bucketed-MoE unified path. Returning Unsupported routes the
        // engine to the legacy decode_batch path via LlmExecutor's
        // fallback partition.
        let all_decode = items.iter().all(|it| it.1.len() == 1 && it.3);
        if all_decode {
            return Err(FerrumError::unsupported(
                "Qwen3MoeModel::unified_forward: pure-decode batch — \
                 routed to legacy decode_batch (faster for q_len=1)",
            ));
        }
        if items.len() == 1 && items[0].1.len() > 1 {
            return Err(FerrumError::unsupported(
                "Qwen3MoeModel::unified_forward: single-seq prefill — \
                 routed to specialized prefill path",
            ));
        }
        if !self.runtime_env.qwen_unified_prefill && items.iter().any(|it| it.1.len() > 1) {
            return Err(FerrumError::unsupported(
                "Qwen3MoeModel::unified_forward: prefill disabled by \
                 FERRUM_QWEN_UNIFIED_PREFILL=0",
            ));
        }
        // Any prefill chunk (q_len > 1) OR non-final-chunk item:
        // unified path wins by collapsing N serial prefills into one
        // [M_total, hidden] forward.
        if self.paged_pools.is_none() {
            return Err(FerrumError::unsupported(
                "Qwen3MoeModel::unified_forward: paged KV required \
                 (set FERRUM_METAL_PAGED_KV=1).",
            ));
        }
        let m_total: usize = items.iter().map(|it| it.1.len()).sum();
        if m_total > self.scratch.max_tokens {
            return Err(FerrumError::unsupported(format!(
                "Qwen3MoeModel::unified_forward: m_total={} > scratch.max_tokens={}",
                m_total, self.scratch.max_tokens,
            )));
        }
        Ok(self.unified_forward_internal(items))
    }

    fn release(&mut self, cache_id: &str) {
        // Mirror LlamaFamilyModel::release — do NOT reset the captured
        // graphs here. Graphs reference paged_pool addresses (model-
        // level + stable) and paged_batch_* scratch addresses (also
        // model-level + stable); the per-cache_id state (paged_block_
        // indices) lives in `kv_caches` and never appears in graph
        // node args. Wiping graphs on release would invalidate them
        // mid-flight (a release between capture and the next replay
        // → CUDA_ERROR_INVALID_VALUE on cuGraphLaunch).
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        if let Some(mut caches) = self.kv_caches.remove(cache_id) {
            // Paged mode: return the cache_id's blocks to the shared
            // allocator so other sequences can reuse them. Without this,
            // every request consumes max_blocks_per_seq blocks
            // permanently — pool exhausts after FERRUM_PAGED_MAX_SEQS
            // requests and subsequent ensure_kv panics with
            // "scratch residual missing" (the cascade panic from a
            // failed ensure_kv path leaving scratch poisoned).
            if let Some(alloc_arc) = self.paged_block_alloc.as_ref() {
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(c0) = caches.first() {
                    if !c0.paged_block_indices.is_empty() {
                        alloc.free(&c0.paged_block_indices);
                    }
                }
                for c in caches.iter_mut() {
                    c.paged_block_indices.clear();
                }
            }
            self.kv_free_pool.push(caches);
        }
    }

    fn reset(&mut self) {
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_all_graphs(&mut ctx);
        self.batched_graph_keys_seen.clear();
        self.batched_graph_warmup = 0;
        self.batched_graph_failed = false;
        B::sync(&mut ctx);
        self.kv_caches.clear();
        self.kv_free_pool.clear();
    }
}
