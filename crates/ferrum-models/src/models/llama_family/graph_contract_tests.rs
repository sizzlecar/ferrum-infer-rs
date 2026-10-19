use ferrum_kernels::{
    backend::{cpu::CpuBackend, Backend},
    LoadedLinear,
};
use ferrum_quantization::{QuantConfig, WeightLoader};
use ferrum_types::{FerrumError, LlamaExecutionConfig, Result};

use super::{LlamaFamilyConfig, LlamaFamilyModel};
use crate::test_support::SyntheticLlamaLoader;

fn config() -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        hidden_size: 32,
        intermediate_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        num_layers: 2,
        vocab_size: 48,
        max_seq_len: 128,
        ..Default::default()
    }
}

struct EagerOnlyLoader(SyntheticLlamaLoader);

impl WeightLoader<CpuBackend> for EagerOnlyLoader {
    fn load_tensor(
        &self,
        ctx: &mut <CpuBackend as Backend>::Context,
        name: &str,
    ) -> Result<Vec<f32>> {
        <SyntheticLlamaLoader as WeightLoader<CpuBackend>>::load_tensor(&self.0, ctx, name)
    }
    fn load_linear(
        &self,
        ctx: &mut <CpuBackend as Backend>::Context,
        name: &str,
    ) -> Result<LoadedLinear<CpuBackend>> {
        let original =
            <SyntheticLlamaLoader as WeightLoader<CpuBackend>>::load_linear(&self.0, ctx, name)?;
        // Wrapping a loaded object does not silently inherit its credential.
        Ok(LoadedLinear::new(original.into_eager_box()))
    }
    fn has_tensor(&self, name: &str) -> bool {
        <SyntheticLlamaLoader as WeightLoader<CpuBackend>>::has_tensor(&self.0, name)
    }
    fn quant_config(&self) -> Option<&QuantConfig> {
        None
    }
}

#[test]
fn graph_contract_safe_custom_loader_keeps_eager_full_logits() {
    let config = config();
    let loader = EagerOnlyLoader(SyntheticLlamaLoader::new(config.clone()));
    let reference_loader = SyntheticLlamaLoader::new(config.clone());
    let mut actual = LlamaFamilyModel::<CpuBackend>::new(config.clone(), &loader).unwrap();
    let mut expected = LlamaFamilyModel::<CpuBackend>::new(config, &reference_loader).unwrap();
    for (left, right) in [
        (
            actual.prefill_internal("actual", &[1, 2, 3]),
            expected.prefill_internal("reference", &[1, 2, 3]),
        ),
        (
            actual.decode_internal("actual", 4, 3),
            expected.decode_internal("reference", 4, 3),
        ),
    ] {
        assert_eq!(left.len(), actual.model_config().vocab_size);
        assert!(left.iter().chain(&right).all(|value| value.is_finite()));
        assert_eq!(left, right);
    }
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
fn graph_contract_explicit_graph_rejects_unqualified_projection() {
    for route in 0..3 {
        let config = config();
        let mut execution = LlamaExecutionConfig::default();
        match route {
            0 => execution.options.cuda_graph = true,
            1 => execution.batched_graph = true,
            _ => execution.options.unified_graph = true,
        }
        let loader = EagerOnlyLoader(SyntheticLlamaLoader::new(config.clone()));
        let error = match LlamaFamilyModel::<CpuBackend>::new_with_execution_config(
            config.clone(),
            &loader,
            execution.clone(),
        ) {
            Ok(mut model) => {
                model.close().unwrap();
                panic!("unqualified graph request accepted");
            }
            Err(error) => error,
        };
        assert!(matches!(error, FerrumError::Unsupported { .. }), "{error}");
        assert!(error
            .to_string()
            .contains("capture-stable loaded projections"));
        // The same explicit option remains available to audited built-in
        // projections; this is not a blanket disabling of graph configuration.
        let audited = SyntheticLlamaLoader::new(config.clone());
        let mut model =
            LlamaFamilyModel::<CpuBackend>::new_with_execution_config(config, &audited, execution)
                .unwrap();
        model.close().unwrap();
    }
}
