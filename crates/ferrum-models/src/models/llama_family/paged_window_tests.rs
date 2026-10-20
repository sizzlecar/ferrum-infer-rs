//! Actual layout selection and resource-free rejection of missing KV semantics.
use super::*;
use crate::test_support::SyntheticLlamaLoader;
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_types::{LlamaExecutionConfig, SelectedKvCacheLayout};

fn config() -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 2,
        vocab_size: 128,
        max_seq_len: 32,
        sliding_window: 2,
        ..Default::default()
    }
}

fn windowed_paged_falls_back_to_real_contiguous<B: MoeLlmBackend>() {
    use crate::test_support::numerical::assert_full_logits_qualification;
    assert!(!B::supports_windowed_paged_attention());
    for pattern in [0, 2] {
        let mut cfg = config();
        cfg.sliding_window_pattern = pattern;
        cfg.rope_local_theta = Some(100.0);
        let loader = SyntheticLlamaLoader::new(cfg.clone());
        let mut execution = LlamaExecutionConfig::default();
        execution.options.metal_paged_kv = Some(true);
        execution.options.cuda_graph = false;
        execution.options.unified_graph = false;
        assert_eq!(
            cfg.selected_kv_cache_layout::<B>(&execution.options),
            SelectedKvCacheLayout::Contiguous
        );
        let mut actual = LlamaFamilyModel::<B>::new_with_execution_config(
            cfg.clone(),
            &loader,
            execution.clone(),
        )
        .unwrap();
        execution.options.metal_paged_kv = Some(false);
        let mut reference =
            LlamaFamilyModel::<CpuBackend>::new_with_execution_config(cfg, &loader, execution)
                .unwrap();
        let mut outputs = vec![(
            actual.prefill_internal("a", &[1, 2, 3, 4, 5]),
            reference.prefill_internal("r", &[1, 2, 3, 4, 5]),
        )];
        assert!(actual.paged_pools.is_none());
        assert!(actual.paged_block_alloc.is_none());
        assert!(actual.kv_caches["a"]
            .iter()
            .all(|cache| cache.block_size == 0));
        for position in 5..9 {
            outputs.push((
                actual.decode_internal("a", position + 10, position),
                reference.decode_internal("r", position + 10, position),
            ));
        }
        assert_eq!(actual.cache_len("a"), 9);
        actual.close().unwrap();
        reference.close().unwrap();
        for (actual, expected) in outputs {
            assert_eq!(actual.len(), 128);
            assert_eq!(expected.len(), 128);
            assert_full_logits_qualification(&actual, &expected);
        }
    }
}

#[test]
fn paged_window_layout_and_execution_fallback_on_cpu() {
    windowed_paged_falls_back_to_real_contiguous::<CpuBackend>();
    let mut cfg = config();
    cfg.sliding_window = 0;
    let mut options = LlamaExecutionOptions::default();
    options.metal_paged_kv = Some(true);
    // Preserve the existing explicit full-attention allocation selection.
    assert!(matches!(
        cfg.selected_kv_cache_layout::<CpuBackend>(&options),
        SelectedKvCacheLayout::Paged { .. }
    ));
    cfg.sliding_window = 2;
    cfg.sliding_window_pattern = 1;
    assert!(!cfg.requires_windowed_attention(0..cfg.num_layers));
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal device"]
fn paged_window_layout_and_execution_fallback_on_metal() {
    windowed_paged_falls_back_to_real_contiguous::<ferrum_kernels::backend::metal::MetalBackend>();
}

#[cfg(feature = "cuda")]
#[test]
fn paged_window_int8_is_rejected_before_weight_loading() {
    use ferrum_interfaces::kv_dtype::KvInt8;
    use ferrum_kernels::backend::cuda::CudaBackend;
    use ferrum_quantization::{QuantConfig, WeightLoader};
    struct NeverLoaded;
    impl WeightLoader<CudaBackend> for NeverLoaded {
        fn load_tensor(
            &self,
            _: &mut <CudaBackend as Backend>::Context,
            _: &str,
        ) -> Result<<CudaBackend as Backend>::Buffer> {
            panic!("unsupported KV semantics must be rejected before weights");
        }
        fn load_linear(
            &self,
            _: &mut <CudaBackend as Backend>::Context,
            _: &str,
        ) -> Result<ferrum_kernels::LoadedLinear<CudaBackend>> {
            panic!("unsupported KV semantics must be rejected before projections");
        }
        fn has_tensor(&self, _: &str) -> bool {
            false
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }
    for pattern in [0, 2] {
        let mut cfg = config();
        cfg.sliding_window_pattern = pattern;
        let mut execution = LlamaExecutionConfig::default();
        execution.options.metal_paged_kv = Some(true);
        // The backend FP16 path has the capability; INT8 must not borrow it.
        assert!(matches!(
            cfg.selected_kv_cache_layout::<CudaBackend>(&execution.options),
            SelectedKvCacheLayout::Paged { .. }
        ));
        let error = match LlamaFamilyModel::<CudaBackend, KvInt8>::new_with_execution_config(
            cfg,
            &NeverLoaded,
            execution,
        ) {
            Err(error) => error,
            Ok(_) => panic!("unsupported INT8 window must not construct a model"),
        };
        assert!(matches!(
            error,
            ferrum_types::FerrumError::Unsupported { .. }
        ));
    }
}
