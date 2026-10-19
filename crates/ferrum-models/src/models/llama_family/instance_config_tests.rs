use std::sync::Mutex;

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::Backend;
use ferrum_quantization::{DenseLinear, Linear, QuantConfig, WeightLoader};
use ferrum_types::{EngineConfig, LlamaExecutionConfig, Result, RuntimeConfigSnapshot};

use super::{
    LlamaFamilyConfig, LlamaFamilyLayerStageConfig, LlamaFamilyModel, LlamaFamilyRuntimeEnv,
    LLAMA_PAGED_BLOCK_SIZE,
};
use crate::models::llama_family_forward_batched::LlamaBatchedRuntimeConfig;

type Model = LlamaFamilyModel<CpuBackend>;

/// Run another complete construction at the first weight read, after the
/// outer constructor has received its execution config. Taking the callback
/// before calling it avoids holding a lock across the nested construction.
#[derive(Default)]
struct FirstLoadLoader<'a> {
    first_load: Mutex<Option<Box<dyn FnOnce() + Send + 'a>>>,
    contexts: Mutex<Vec<ferrum_types::BackendContextConfig>>,
}

impl FirstLoadLoader<'_> {
    fn on_load(&self, ctx: &<CpuBackend as Backend>::Context) {
        self.contexts
            .lock()
            .unwrap()
            .push(CpuBackend::context_config(ctx).clone());
        let callback = self.first_load.lock().unwrap().take();
        if let Some(callback) = callback {
            callback();
        }
    }
}

impl WeightLoader<CpuBackend> for FirstLoadLoader<'_> {
    fn load_tensor(
        &self,
        ctx: &mut <CpuBackend as ferrum_kernels::backend::Backend>::Context,
        _name: &str,
    ) -> Result<Vec<f32>> {
        self.on_load(ctx);
        Ok(vec![1.0])
    }

    fn load_linear(
        &self,
        ctx: &mut <CpuBackend as ferrum_kernels::backend::Backend>::Context,
        name: &str,
    ) -> Result<Box<dyn Linear<CpuBackend>>> {
        self.on_load(ctx);
        let rows = if name.ends_with("qkv_proj") {
            3
        } else if name.ends_with("gate_up_proj") {
            2
        } else {
            1
        };
        Ok(Box::new(DenseLinear::<CpuBackend>::from_rows_in_context(
            ctx,
            &vec![1.0; rows],
            rows,
            1,
        )))
    }

    fn has_tensor(&self, _name: &str) -> bool {
        false
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        None
    }
}

fn tiny_config() -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        hidden_size: 1,
        intermediate_size: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 1,
        num_layers: 2,
        vocab_size: 1,
        max_seq_len: 64,
        ..Default::default()
    }
}

fn configured_engine(capacity: usize, blocks: usize, max_seqs: usize, flags: bool) -> EngineConfig {
    let flag = if flags { "1" } else { "0" };
    let mut entries = vec![
        ("FERRUM_KV_CAPACITY", capacity.to_string()),
        ("FERRUM_KV_MAX_BLOCKS", blocks.to_string()),
        ("FERRUM_PAGED_MAX_SEQS", max_seqs.to_string()),
        ("FERRUM_METAL_PAGED_KV", "1".to_owned()),
        ("FERRUM_PREFIX_CACHE", flag.to_owned()),
        ("FERRUM_BATCHED_GRAPH", flag.to_owned()),
        ("FERRUM_UNIFIED_GRAPH", flag.to_owned()),
        ("FERRUM_UNIFIED_GRAPH_LAYERS_ONLY", flag.to_owned()),
        ("FERRUM_UNIFIED_GRAPH_LM_HEAD_EAGER", flag.to_owned()),
        ("FERRUM_GREEDY_ARGMAX", flag.to_owned()),
        ("FERRUM_SPLIT_K_ATTN", flag.to_owned()),
        ("FERRUM_VLLM_MOE", flag.to_owned()),
        ("FERRUM_MOE_HOST_ROUTE", flag.to_owned()),
        (
            "FERRUM_METAL_DTYPE",
            if flags { "f16" } else { "f32" }.to_owned(),
        ),
        (
            "FERRUM_NAN_TRACE",
            if flags { "layer1" } else { "0" }.to_owned(),
        ),
    ];
    if flags {
        entries.extend([
            ("FERRUM_CUDA_GRAPH", "1".to_owned()),
            ("FERRUM_OP_DUMP", "instance-ops".to_owned()),
            ("FERRUM_LAYER_DUMP", "instance-layers".to_owned()),
        ]);
    }
    let mut engine = EngineConfig::default();
    engine
        .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars(entries))
        .unwrap();
    engine
}

fn assert_frozen_config(model: &Model, expected: &LlamaExecutionConfig) {
    assert_eq!(
        CpuBackend::context_config(model.execution_context.as_ref().unwrap()),
        &expected.backend_context
    );
    assert_eq!(
        model.runtime_env,
        LlamaFamilyRuntimeEnv::from_execution_config(expected)
    );
    assert_eq!(
        model.batched_cfg,
        LlamaBatchedRuntimeConfig::from_execution_config(expected)
    );
}

fn assert_initial_paged_allocation(
    model: &mut Model,
    capacity: usize,
    blocks: usize,
    max_seqs: usize,
) {
    assert_eq!(crate::common::DecoderOnlyLLM::kv_capacity(model), capacity);
    model.ensure_kv("instance-cache");
    let table_blocks = capacity.div_ceil(LLAMA_PAGED_BLOCK_SIZE);
    assert_eq!(model.paged_dims, Some((max_seqs, table_blocks)));
    assert_eq!(model.scratch.paged_max_seqs, max_seqs);
    assert_eq!(model.scratch.paged_max_blocks_per_seq, table_blocks);
    let allocator = model.paged_block_alloc.as_ref().unwrap().lock().unwrap();
    assert_eq!(allocator.capacity() as usize, blocks);
    assert_eq!(allocator.free_count(), blocks);
    let pools = model.paged_pools.as_ref().unwrap();
    assert_eq!(pools.len(), model.local_layer_count());
    for (k, v) in pools {
        assert_eq!(k.len(), blocks * LLAMA_PAGED_BLOCK_SIZE);
        assert_eq!(v.len(), blocks * LLAMA_PAGED_BLOCK_SIZE);
    }
    for cache in &model.kv_caches["instance-cache"] {
        assert_eq!(cache.capacity, table_blocks * LLAMA_PAGED_BLOCK_SIZE);
        assert!(cache.paged_block_indices.is_empty());
    }
}

#[test]
fn construction_keeps_instance_config_across_another_engine_apply_and_model_load() {
    let engine_a = configured_engine(17, 3, 2, true);
    let execution_a = engine_a.llama_execution_config();
    let nested = Mutex::new(None);
    let loader = FirstLoadLoader {
        first_load: Mutex::new(Some(Box::new(|| {
            let engine_b = configured_engine(33, 5, 3, false);
            let execution_b = engine_b.llama_execution_config();
            let model_b = Model::new_with_execution_config(
                tiny_config(),
                &FirstLoadLoader::default(),
                execution_b.clone(),
            )
            .unwrap();
            *nested.lock().unwrap() = Some((model_b, execution_b));
        }))),
        ..Default::default()
    };
    let mut model_a =
        Model::new_with_execution_config(tiny_config(), &loader, execution_a.clone()).unwrap();
    let (mut model_b, execution_b) = nested
        .lock()
        .unwrap()
        .take()
        .expect("first load interleaved");

    assert!(!loader.contexts.lock().unwrap().is_empty());
    assert!(loader
        .contexts
        .lock()
        .unwrap()
        .iter()
        .all(|context| context == &execution_a.backend_context));
    assert_frozen_config(&model_a, &execution_a);
    assert_frozen_config(&model_b, &execution_b);
    assert_ne!(model_a.runtime_env, model_b.runtime_env);
    assert_ne!(model_a.batched_cfg, model_b.batched_cfg);
    // CPU exercises real allocation and capacity accounting here. No paged
    // attention forward is run or claimed as GPU numerical validation.
    assert_initial_paged_allocation(&mut model_a, 17, 3, 2);
    assert_initial_paged_allocation(&mut model_b, 33, 5, 3);
}

#[test]
fn default_library_constructors_ignore_applied_engine_options() {
    let _configured = configured_engine(17, 3, 2, true);
    let loader = FirstLoadLoader::default();
    let mut models = [
        Model::new(tiny_config(), &loader).unwrap(),
        Model::new_backbone_only(tiny_config(), &loader).unwrap(),
        Model::new_layer_stage(
            tiny_config(),
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(1..2, false, true),
        )
        .unwrap(),
    ];
    for model in &mut models {
        assert_frozen_config(model, &LlamaExecutionConfig::default());
        assert_eq!(crate::common::DecoderOnlyLLM::kv_capacity(model), 64);
        model.ensure_kv("default-cache");
        assert!(model.paged_pools.is_none());
        for cache in &model.kv_caches["default-cache"] {
            assert_eq!(cache.capacity, 64);
            assert_eq!(cache.k.len(), 64);
            assert_eq!(cache.v.len(), 64);
        }
    }
}

#[test]
fn layer_stages_keep_the_same_frozen_config_across_interleaved_construction() {
    let engine_a = configured_engine(17, 3, 2, true);
    let execution_a = engine_a.llama_execution_config();
    let nested = Mutex::new(None);
    let loader = FirstLoadLoader {
        first_load: Mutex::new(Some(Box::new(|| {
            let engine_b = configured_engine(33, 5, 3, false);
            let execution_b = engine_b.llama_execution_config();
            let stage_b = Model::new_layer_stage_with_execution_config(
                tiny_config(),
                &FirstLoadLoader::default(),
                LlamaFamilyLayerStageConfig::pipeline_stage(1..2, false, true),
                execution_b.clone(),
            )
            .unwrap();
            *nested.lock().unwrap() = Some((stage_b, execution_b));
        }))),
        ..Default::default()
    };
    let first = Model::new_layer_stage_with_execution_config(
        tiny_config(),
        &loader,
        LlamaFamilyLayerStageConfig::pipeline_stage(0..1, true, false),
        execution_a.clone(),
    )
    .unwrap();
    let last = Model::new_layer_stage_with_execution_config(
        tiny_config(),
        &FirstLoadLoader::default(),
        LlamaFamilyLayerStageConfig::pipeline_stage(1..2, false, true),
        execution_a.clone(),
    )
    .unwrap();
    let (mut stage_b, execution_b) = nested
        .lock()
        .unwrap()
        .take()
        .expect("first load interleaved");
    for mut stage in [first, last] {
        assert_frozen_config(&stage, &execution_a);
        assert_initial_paged_allocation(&mut stage, 17, 3, 2);
    }
    assert_frozen_config(&stage_b, &execution_b);
    assert_initial_paged_allocation(&mut stage_b, 33, 5, 3);
}

#[test]
fn invalid_instance_capacity_is_rejected_before_any_weight_load() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut invalid = vec![
        LlamaExecutionConfig {
            kv_capacity: Some(0),
            ..Default::default()
        },
        LlamaExecutionConfig {
            paged_max_seqs: 0,
            ..Default::default()
        },
    ];
    let mut zero_blocks = LlamaExecutionConfig::default();
    zero_blocks.options.kv_max_blocks = Some(0);
    invalid.push(zero_blocks);
    let mut zero_streams = LlamaExecutionConfig::default();
    zero_streams.backend_context.moe.moe_streams = 0;
    invalid.push(zero_streams);
    let mut invalid_block = LlamaExecutionConfig::default();
    invalid_block.backend_context.moe.moe_block_size = Some(12);
    invalid.push(invalid_block);
    for execution in invalid {
        let reads = AtomicUsize::new(0);
        let loader = FirstLoadLoader {
            first_load: Mutex::new(Some(Box::new(|| {
                reads.fetch_add(1, Ordering::Relaxed);
            }))),
            ..Default::default()
        };
        for result in [
            Model::new_with_execution_config(tiny_config(), &loader, execution.clone()),
            Model::new_backbone_only_with_execution_config(
                tiny_config(),
                &loader,
                execution.clone(),
            ),
            Model::new_layer_stage_with_execution_config(
                tiny_config(),
                &loader,
                LlamaFamilyLayerStageConfig::pipeline_stage(1..2, false, true),
                execution.clone(),
            ),
        ] {
            assert!(matches!(
                result,
                Err(ferrum_types::FerrumError::Config { .. })
            ));
        }
        assert_eq!(reads.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn selected_kv_layout_cpu_contiguous_crosses_block_estimate_with_full_logits() {
    use crate::common::DecoderOnlyLLM;
    use crate::test_support::SyntheticLlamaLoader;
    use ferrum_types::{FerrumConfigBuilder, SelectedKvCacheLayout};
    let cfg = LlamaFamilyConfig {
        hidden_size: 16,
        intermediate_size: 32,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 8,
        num_layers: 2,
        vocab_size: 32,
        max_seq_len: 32,
        ..Default::default()
    };
    let request = RuntimeConfigSnapshot::from_env_vars([
        ("FERRUM_KV_CAPACITY", "32"),
        ("FERRUM_KV_MAX_BLOCKS", "1"),
        ("FERRUM_PAGED_MAX_SEQS", "1"),
        ("FERRUM_MAX_MODEL_LEN", "32"),
        ("FERRUM_MAX_BATCHED_TOKENS", "32"),
    ]);
    let mut input = EngineConfig::default();
    input.apply_runtime_config_snapshot(&request).unwrap();
    let selected = cfg.selected_kv_cache_layout::<CpuBackend>(&input.runtime.llama);
    assert_eq!(selected, SelectedKvCacheLayout::Contiguous);
    let definition = crate::ModelDefinition {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_attention_heads: cfg.num_heads,
        num_key_value_heads: Some(cfg.num_kv_heads),
        num_hidden_layers: cfg.num_layers,
        vocab_size: cfg.vocab_size,
        max_position_embeddings: cfg.max_seq_len,
        ..Default::default()
    };
    assert_eq!(
        crate::legacy_capabilities::selected_kv_cache_layout(
            &definition,
            &ferrum_types::Device::CPU,
            &input.runtime.llama,
        ),
        selected
    );
    let resolved = FerrumConfigBuilder::new(request)
        .with_selected_kv_cache_layout(selected)
        .resolve()
        .unwrap();
    let mut engine = EngineConfig::default();
    engine
        .apply_runtime_config_snapshot(&resolved.runtime_config)
        .unwrap();
    let loader = SyntheticLlamaLoader::new(cfg.clone());
    let mut candidate =
        Model::new_with_execution_config(cfg.clone(), &loader, engine.llama_execution_config())
            .unwrap();
    let mut reference = Model::new(cfg.clone(), &loader).unwrap();
    let tokens: Vec<_> = (0..17).map(|index| (index % 7 + 1) as u32).collect();
    let actual = candidate.prefill_internal("candidate", &tokens);
    let expected = reference.prefill_internal("reference", &tokens);
    assert_eq!(actual.len(), cfg.vocab_size);
    assert!(actual.iter().all(|value| value.is_finite()));
    assert_eq!(actual, expected);
    assert!(candidate.paged_pools.is_none());
    assert!(candidate.paged_block_alloc.is_none());
    assert!(candidate.kv_caches["candidate"]
        .iter()
        .all(|cache| cache.capacity >= 17));
    let actual = candidate.decode_internal("candidate", 3, 17);
    let expected = reference.decode_internal("reference", 3, 17);
    assert_eq!(actual.len(), cfg.vocab_size);
    assert!(actual.iter().all(|value| value.is_finite()));
    assert_eq!(actual, expected);
    candidate.release("candidate");
    reference.release("reference");
    candidate.close().unwrap();
    reference.close().unwrap();
    assert!(candidate.execution_context.is_none());
    assert!(reference.execution_context.is_none());
    assert!(candidate.kv_caches.is_empty());
    assert!(reference.kv_caches.is_empty());
}

#[test]
fn selected_kv_layout_preserves_paged_allocator_geometry_and_layer_fallback() {
    use ferrum_types::{FerrumConfigBuilder, SelectedKvCacheLayout};
    let engine = configured_engine(32, 2, 1, false);
    let mut cfg = tiny_config();
    cfg.max_seq_len = 32;
    let selected = cfg.selected_kv_cache_layout::<CpuBackend>(&engine.runtime.llama);
    assert_eq!(
        selected,
        SelectedKvCacheLayout::Paged {
            block_size_tokens: LLAMA_PAGED_BLOCK_SIZE
        }
    );
    let mut model = Model::new_with_execution_config(
        cfg.clone(),
        &FirstLoadLoader::default(),
        engine.llama_execution_config(),
    )
    .unwrap();
    // This checks the existing real allocation path, not GPU numerical eligibility.
    assert_initial_paged_allocation(&mut model, 32, 2, 1);
    let actual_pool_tokens = model
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .capacity() as usize
        * LLAMA_PAGED_BLOCK_SIZE;
    assert_eq!(actual_pool_tokens, 32);
    assert!(
        FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_env_vars([
            ("FERRUM_KV_MAX_BLOCKS", "1"),
            ("FERRUM_PAGED_MAX_SEQS", "1"),
            ("FERRUM_MAX_BATCHED_TOKENS", "16"),
            ("FERRUM_MAX_MODEL_LEN", "32"),
        ]))
        .with_selected_kv_cache_layout(selected)
        .resolve()
        .is_err()
    );
    model.close().unwrap();
    // The same requested paged mode falls back for an unsupported layer schedule.
    cfg.sliding_window_pattern = 6;
    assert_eq!(
        cfg.selected_kv_cache_layout::<CpuBackend>(&engine.runtime.llama),
        SelectedKvCacheLayout::Contiguous
    );
}
