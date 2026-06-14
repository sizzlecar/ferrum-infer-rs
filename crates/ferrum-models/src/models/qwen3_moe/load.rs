use super::*;

impl<B: MoeLlmBackend, K: KvDtypeKind> Qwen3MoeModel<B, K> {
    /// Build a Qwen3-MoE model from a generic `WeightLoader<B>` plus a
    /// GGUF reader for the experts (which `WeightLoader` doesn't model
    /// directly — its API is rank-2 only).
    ///
    /// `loader` provides: token embedding, attention projections, layer
    /// norms, lm_head — all the rank-2 weights.
    /// `gguf` provides: the rank-3 expert tensors, sliced per-expert
    /// inside [`ExpertStack::load_from_gguf`].
    pub fn new(
        cfg: Qwen3MoeConfig,
        loader: &dyn WeightLoader<B>,
        gguf: &ferrum_quantization::gguf::GgufFile,
    ) -> Result<Self> {
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let runtime_env = Qwen3MoeRuntimeEnv::from_env();
        let rope = build_rope_cache::<B>(&cfg.base);
        // GGUF/Metal uses the legacy decode path, not the CUDA varlen
        // unified path. Keep the historical small scratch allocation here
        // and let ensure_scratch grow to the actual batch size (for the
        // README c=16 row this means 16, not 2048). A 2048-token scratch
        // allocates multi-GB batch_logits/MoE temporaries on Apple Silicon
        // and regresses Qwen3-30B-A3B by ~4x through memory pressure.
        let initial_scratch_tokens = if B::supports_varlen_qkv() {
            runtime_env.initial_scratch_tokens
        } else {
            1
        };
        let scratch = Qwen3MoeScratch::alloc(&cfg, initial_scratch_tokens);

        let embed = loader.load_tensor("model.embed_tokens.weight")?;

        let mut attn_layers = Vec::with_capacity(cfg.base.num_layers);
        let mut moe_layers = Vec::with_capacity(cfg.base.num_layers);
        for li in 0..cfg.base.num_layers {
            let prefix = format!("model.layers.{li}");
            let input_ln_w = loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
            let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
            let post_ln_w =
                loader.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;

            // Dense gate_up_proj / down_proj are absent in MoE GGUFs —
            // we synthesise stub Linears so the LlamaFamilyLayer struct
            // type-checks. They're never invoked because forward_layer
            // calls the MoE path. Cheap: tiny zero-sized DenseLinears.
            let gate_up_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(2 * cfg.expert_intermediate_size, cfg.base.hidden_size);
            let down_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(cfg.base.hidden_size, cfg.expert_intermediate_size);

            let (q_norm_w, k_norm_w) = if cfg.base.has_qk_norm {
                let q = loader
                    .load_tensor(&format!("{prefix}.self_attn.q_norm.weight"))
                    .ok();
                let k = loader
                    .load_tensor(&format!("{prefix}.self_attn.k_norm.weight"))
                    .ok();
                (q, k)
            } else {
                (None, None)
            };

            attn_layers.push(LlamaFamilyLayer {
                input_ln_w,
                qkv_proj,
                q_norm_w,
                k_norm_w,
                o_proj,
                post_ln_w,
                post_attn_ln_w: None,
                post_ffn_ln_w: None,
                gate_up_proj,
                down_proj,
            });

            // Router lives at `model.layers.{li}.mlp.router.weight` in
            // ferrum-name space (see ferrum_to_gguf mapping). It's a
            // plain rank-2 linear so the standard loader path covers
            // it without going through the MoE-specific GGUF helper.
            let router = loader.load_linear(&format!("{prefix}.mlp.router"))?;
            if router.in_features() != cfg.base.hidden_size {
                return Err(FerrumError::model(format!(
                    "router layer {li}: in_features {} != hidden {}",
                    router.in_features(),
                    cfg.base.hidden_size
                )));
            }
            if router.out_features() != cfg.num_experts {
                return Err(FerrumError::model(format!(
                    "router layer {li}: out_features {} != num_experts {}",
                    router.out_features(),
                    cfg.num_experts
                )));
            }

            let experts = ExpertStack::<B>::load_from_gguf(
                gguf,
                li,
                cfg.num_experts,
                cfg.base.hidden_size,
                cfg.expert_intermediate_size,
            )?;

            moe_layers.push(Qwen3MoeLayerState { router, experts });
        }

        let final_norm_w = loader.load_tensor("model.norm.weight")?;
        let lm_head = if loader.has_tensor("lm_head.weight") {
            loader.load_linear("lm_head")?
        } else {
            // Tied embeddings — same as dense path.
            tracing::info!(
                "Qwen3MoeModel: tied embeddings — loading model.embed_tokens.weight as lm_head"
            );
            loader.load_linear("model.embed_tokens")?
        };

        let runtime_cfg = cfg.base.to_runtime();
        let use_vllm_paged_attn = B::supports_vllm_paged_attn() && runtime_env.use_vllm_paged_attn;
        Ok(Self {
            cfg,
            runtime_cfg,
            runtime_env,
            supports_varlen_qkv: B::supports_varlen_qkv(),
            supports_batched_moe_gemv: B::supports_batched_moe_gemv(),
            embed,
            attn_layers,
            moe_layers,
            final_norm_w,
            lm_head,
            rope,
            scratch,
            kv_caches: HashMap::new(),
            kv_free_pool: Vec::new(),
            paged_pools: None,
            paged_fa_pools: None,
            paged_block_alloc: None,
            paged_dims: None,
            batched_graph_warmup: 0,
            batched_graph_failed: false,
            batched_graph_keys_seen: std::collections::HashSet::new(),
            prefix_cache_hits: 0,
            prefix_cache_misses: 0,
            prefix_cache_saved_prefill_tokens: 0,
            use_vllm_paged_attn,
        })
    }

    /// Build from a HuggingFace safetensors model directory (GPTQ-INT4
    /// expected). Mirrors [`Self::new`] but with a STACKED expert loader:
    /// reads all `num_experts` experts' raw GPTQ tensors per layer once,
    /// concats along N host-side, single `B::load_gptq` repacks the
    /// whole thing into one Marlin tile per (layer, role).
    ///
    /// 128 experts × 48 layers × 3 projs would otherwise trigger 18 432
    /// per-call Marlin repacks (~30+ minute cold start at ~100 ms each
    /// on Llama-MoE shapes). The stacked path drops that to 96 repacks
    /// — one per (layer × {gate_up, down}) — and dispatch slices per
    /// expert via `B::gemm_gptq_with_offset`.
    pub fn new_safetensors(
        cfg: Qwen3MoeConfig,
        loader: &ferrum_quantization::NativeSafetensorsLoader<B>,
    ) -> Result<Self> {
        use ferrum_quantization::WeightLoader as _;
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let runtime_env = Qwen3MoeRuntimeEnv::from_env();
        let rope = build_rope_cache::<B>(&cfg.base);
        let scratch = Qwen3MoeScratch::alloc(&cfg, runtime_env.initial_scratch_tokens);
        let embed = loader.load_tensor("model.embed_tokens.weight")?;
        let mut attn_layers = Vec::with_capacity(cfg.base.num_layers);
        let mut moe_layers = Vec::with_capacity(cfg.base.num_layers);
        for li in 0..cfg.base.num_layers {
            let prefix = format!("model.layers.{li}");
            let input_ln_w = loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
            let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
            let post_ln_w =
                loader.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let gate_up_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(2 * cfg.expert_intermediate_size, cfg.base.hidden_size);
            let down_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(cfg.base.hidden_size, cfg.expert_intermediate_size);
            let (q_norm_w, k_norm_w) = if cfg.base.has_qk_norm {
                let q = loader
                    .load_tensor(&format!("{prefix}.self_attn.q_norm.weight"))
                    .ok();
                let k = loader
                    .load_tensor(&format!("{prefix}.self_attn.k_norm.weight"))
                    .ok();
                (q, k)
            } else {
                (None, None)
            };
            attn_layers.push(LlamaFamilyLayer {
                input_ln_w,
                qkv_proj,
                q_norm_w,
                k_norm_w,
                o_proj,
                post_ln_w,
                post_attn_ln_w: None,
                post_ffn_ln_w: None,
                gate_up_proj,
                down_proj,
            });

            // Router: rank-2 linear, standard load.
            let router = loader.load_linear(&format!("{prefix}.mlp.gate"))?;
            if router.in_features() != cfg.base.hidden_size {
                return Err(ferrum_types::FerrumError::model(format!(
                    "router layer {li}: in_features {} != hidden {}",
                    router.in_features(),
                    cfg.base.hidden_size
                )));
            }
            if router.out_features() != cfg.num_experts {
                return Err(ferrum_types::FerrumError::model(format!(
                    "router layer {li}: out_features {} != num_experts {}",
                    router.out_features(),
                    cfg.num_experts
                )));
            }

            // Stacked GPTQ Marlin load via per-expert-repack-then-concat
            // (B::load_gptq_stacked). Each expert's Marlin-packed bytes
            // are contiguous in the GPU buffer, so offset GEMM
            // dispatches via pointer offset alone — no stride magic.
            let expert_prefix = format!("{prefix}.mlp.experts.{{e}}.");
            let probe_split =
                loader.has_tensor(&format!("{prefix}.mlp.experts.0.gate_proj.qweight"));
            let gate_up_projs: &[&str] = if probe_split {
                &["gate_proj", "up_proj"]
            } else {
                &["gate_up_proj"]
            };
            // Phase C step 4e: load_stacked_gptq_experts returns the
            // trait-object MarlinExpertStack directly (no intermediate
            // GptqStore type). The loader internally calls
            // B::load_gptq_stacked which now returns
            // Arc<dyn MarlinExpertStack<B>>.
            let (gate_up_marlin, gate_up_n_per_expert, gate_up_k) =
                loader.load_stacked_gptq_experts(&expert_prefix, cfg.num_experts, gate_up_projs)?;
            let (down_marlin, down_n_per_expert, down_k) = loader.load_stacked_gptq_experts(
                &expert_prefix,
                cfg.num_experts,
                &["down_proj"],
            )?;

            // Per-expert Linear views — used by code paths that go
            // through `ExpertStack::gate_up[i]` / `down[i]` (single
            // expert, non-bucketed). StackedExpertLinear wraps the
            // MarlinExpertStack trait object and dispatches via
            // `stack.make_expert_linear(...)` per Phase C step 4b.
            let mut gate_up: Vec<Box<dyn ferrum_quantization::Linear<B>>> =
                Vec::with_capacity(cfg.num_experts);
            let mut down: Vec<Box<dyn ferrum_quantization::Linear<B>>> =
                Vec::with_capacity(cfg.num_experts);
            for e in 0..cfg.num_experts {
                gate_up.push(Box::new(
                    ferrum_quantization::StackedExpertLinear::<B>::new(
                        gate_up_marlin.clone(),
                        e * gate_up_n_per_expert,
                        gate_up_n_per_expert,
                    )?,
                ));
                down.push(Box::new(
                    ferrum_quantization::StackedExpertLinear::<B>::new(
                        down_marlin.clone(),
                        e * down_n_per_expert,
                        down_n_per_expert,
                    )?,
                ));
            }

            let experts = crate::moe::ExpertStack::<B> {
                gate_up,
                down,
                gate_stacked: None,
                up_stacked: None,
                down_stacked: None,
                gate_up_marlin_stack: Some(gate_up_marlin),
                down_marlin_stack: Some(down_marlin),
            };
            moe_layers.push(Qwen3MoeLayerState { router, experts });

            if li == 0 || li.is_multiple_of(8) || li == cfg.base.num_layers - 1 {
                tracing::info!(
                    "Qwen3MoeModel safetensors: layer {li}/{} loaded \
                     (stacked: gate_up={}x{} k={}, down={}x{} k={})",
                    cfg.base.num_layers,
                    cfg.num_experts,
                    gate_up_n_per_expert,
                    gate_up_k,
                    cfg.num_experts,
                    down_n_per_expert,
                    down_k,
                );
            }
        }

        let final_norm_w = loader.load_tensor("model.norm.weight")?;
        let lm_head = if loader.has_tensor("lm_head.weight") {
            loader.load_linear("lm_head")?
        } else {
            tracing::info!(
                "Qwen3MoeModel safetensors: tied embeddings — using model.embed_tokens as lm_head"
            );
            loader.load_linear("model.embed_tokens")?
        };

        let runtime_cfg = cfg.base.to_runtime();
        let use_vllm_paged_attn = B::supports_vllm_paged_attn() && runtime_env.use_vllm_paged_attn;
        Ok(Self {
            cfg,
            runtime_cfg,
            runtime_env,
            supports_varlen_qkv: B::supports_varlen_qkv(),
            supports_batched_moe_gemv: B::supports_batched_moe_gemv(),
            embed,
            attn_layers,
            moe_layers,
            final_norm_w,
            lm_head,
            rope,
            scratch,
            kv_caches: HashMap::new(),
            kv_free_pool: Vec::new(),
            paged_pools: None,
            paged_fa_pools: None,
            paged_block_alloc: None,
            paged_dims: None,
            batched_graph_warmup: 0,
            batched_graph_failed: false,
            batched_graph_keys_seen: std::collections::HashSet::new(),
            prefix_cache_hits: 0,
            prefix_cache_misses: 0,
            prefix_cache_saved_prefill_tokens: 0,
            use_vllm_paged_attn,
        })
    }

    /// Read-only access to the captured-graph warmup counter. Bumps once
    /// per non-replay `decode_batch_internal` call under
    /// `FERRUM_MOE_GRAPH=1`; capture starts on the 4th call (warmup>=3).
    /// Test helper — production code should not branch on this.
    pub fn batched_graph_warmup(&self) -> usize {
        self.batched_graph_warmup
    }

    /// True iff CUDA Graph capture failed at some point — backend
    /// returns Err from begin/end/replay or replay produced wrong
    /// output. Once true, subsequent calls stay eager. Test helper.
    pub fn batched_graph_failed(&self) -> bool {
        self.batched_graph_failed
    }

    /// Set of `m_padded` keys for which a graph has been captured.
    /// Empty until the first successful capture; cleared on
    /// `reset()`, `release()`, or scratch realloc. Test helper.
    pub fn batched_graph_keys_seen(&self) -> &std::collections::HashSet<u64> {
        &self.batched_graph_keys_seen
    }

    pub(super) fn moe_graph_enabled_graph_clean(&self) -> bool {
        if !self.runtime_env.moe_graph_requested {
            return false;
        }
        if self.runtime_env.moe_graph_vllm_clean {
            return true;
        }

        if !MOE_GRAPH_UNCLEAN_WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[moe-graph] disabled: capture requires FERRUM_VLLM_MOE=1 and FERRUM_MOE_HOST_ROUTE!=1"
            );
        }
        false
    }
}

/// Build a stub Linear<B> with the given shape but zero weights. Used to
/// fill the dense `gate_up_proj` / `down_proj` slots in `LlamaFamilyLayer`
/// for MoE models — those slots are never invoked because the MoE FFN
/// path runs through `moe_layer.experts` instead. The stub's only purpose
/// is to satisfy the struct's type signature with minimal memory cost.
fn stub_linear<B: QuantLlmBackend + BackendMoeFused>(
    out_features: usize,
    in_features: usize,
) -> Box<dyn ferrum_quantization::Linear<B>> {
    // Zero-init: out_features * in_features f32. For a 30B-A3B layer
    // this is 2*768*2048 = 3.1M f32 = 12 MB → fine; per-layer overhead
    // ≈ 12 MB × 48 = 576 MB. Marginal vs the experts (~16 GB).
    let zeros = vec![0.0f32; out_features * in_features];
    Box::new(ferrum_quantization::DenseLinear::<B>::from_rows(
        &zeros,
        out_features,
        in_features,
    ))
}

fn build_rope_cache<B: QuantLlmBackend + BackendMoeFused>(cfg: &LlamaFamilyConfig) -> RopeCache<B> {
    let hd = cfg.head_dim;
    let half = hd / 2;
    let max = cfg.max_seq_len;
    let mut cos = vec![0.0f32; max * half];
    let mut sin = vec![0.0f32; max * half];
    for pos in 0..max {
        for i in 0..half {
            let freq = 1.0f64 / cfg.rope_theta.powf((2 * i) as f64 / hd as f64);
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    RopeCache {
        cos: B::from_slice(&cos),
        sin: B::from_slice(&sin),
    }
}
