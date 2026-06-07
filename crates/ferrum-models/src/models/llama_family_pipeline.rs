use ferrum_kernels::backend::{KvLayer, MoeLlmBackend};
use ferrum_types::{FerrumError, Result};

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};

use super::llama_family::LlamaFamilyModel;

pub struct LlamaFamilyPipelineModel<B: MoeLlmBackend, K: KvLayer<B>> {
    stages: Vec<LlamaFamilyModel<B, K>>,
    runtime_cfg: LlmRuntimeConfig,
}

impl<B: MoeLlmBackend, K: KvLayer<B>> LlamaFamilyPipelineModel<B, K> {
    pub fn new(stages: Vec<LlamaFamilyModel<B, K>>) -> Result<Self> {
        if stages.is_empty() {
            return Err(FerrumError::model(
                "LlamaFamilyPipelineModel requires at least one stage",
            ));
        }
        if stages.first().is_some_and(|stage| stage.embed.is_none()) {
            return Err(FerrumError::model(
                "first Llama pipeline stage must load embedding weights",
            ));
        }
        if stages.last().is_some_and(|stage| stage.lm_head.is_none()) {
            return Err(FerrumError::model(
                "last Llama pipeline stage must load lm_head weights",
            ));
        }

        let runtime_cfg = stages[0].runtime_cfg.clone();
        let mut expected_start = 0usize;
        for stage in &stages {
            if stage.runtime_cfg.hidden_size != runtime_cfg.hidden_size
                || stage.runtime_cfg.vocab_size != runtime_cfg.vocab_size
                || stage.runtime_cfg.num_kv_heads != runtime_cfg.num_kv_heads
                || stage.runtime_cfg.head_dim != runtime_cfg.head_dim
                || stage.runtime_cfg.max_seq_len != runtime_cfg.max_seq_len
            {
                return Err(FerrumError::model(
                    "Llama pipeline stages must share runtime dimensions",
                ));
            }
            let range = stage.source_layer_range();
            if range.start != expected_start {
                return Err(FerrumError::model(format!(
                    "Llama pipeline stage range starts at {}, expected {expected_start}",
                    range.start
                )));
            }
            expected_start = range.end;
        }
        if expected_start != runtime_cfg.num_layers {
            return Err(FerrumError::model(format!(
                "Llama pipeline stages cover {expected_start} layers but model has {}",
                runtime_cfg.num_layers
            )));
        }

        Ok(Self {
            stages,
            runtime_cfg,
        })
    }

    pub fn stages(&self) -> &[LlamaFamilyModel<B, K>] {
        &self.stages
    }

    fn last_hidden_row<'a>(&self, hidden: &'a [f32], seq_len: usize) -> &'a [f32] {
        let h = self.runtime_cfg.hidden_size;
        &hidden[(seq_len - 1) * h..seq_len * h]
    }
}

impl<B, K> DecoderOnlyLLM for LlamaFamilyPipelineModel<B, K>
where
    B: MoeLlmBackend,
    K: KvLayer<B>,
    LlamaFamilyModel<B, K>: DecoderOnlyLLM,
{
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "position": "llama-layer-split-pipeline",
            "stage_count": self.stages.len() as u64,
            "transport": "host-hidden-bridge",
        }))
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        for stage in &mut self.stages {
            stage.ensure_scratch(max_tokens);
            stage.ensure_kv(cache_id);
        }
    }

    fn kv_capacity(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.kv_capacity())
            .min()
            .unwrap_or(self.runtime_cfg.max_seq_len)
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        assert!(!tokens.is_empty(), "pipeline prefill called with no tokens");
        let pos_offset = self.stages[0].cache_len(cache_id);
        let mut hidden =
            self.stages[0].prefill_stage_tokens_to_hidden(cache_id, tokens, pos_offset);
        for stage in self.stages.iter_mut().skip(1) {
            hidden =
                stage.prefill_stage_hidden_from_host(cache_id, &hidden, tokens.len(), pos_offset);
        }
        let last_hidden = self.last_hidden_row(&hidden, tokens.len()).to_vec();
        self.stages
            .last_mut()
            .expect("validated non-empty stages")
            .logits_from_hidden(&last_hidden)
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        let mut hidden = self.stages[0].decode_stage_token_to_hidden(cache_id, token, pos);
        for stage in self.stages.iter_mut().skip(1) {
            hidden = stage.decode_stage_hidden_from_host(cache_id, &hidden, pos);
        }
        self.stages
            .last_mut()
            .expect("validated non-empty stages")
            .logits_from_hidden(&hidden)
    }

    fn release(&mut self, cache_id: &str) {
        for stage in &mut self.stages {
            stage.release(cache_id);
        }
    }

    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        for stage in &mut self.stages {
            stage.truncate_kv(cache_id, new_len);
        }
    }

    fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use ferrum_interfaces::kv_dtype::KvFp16;
    use ferrum_kernels::backend::cpu::CpuBackend;
    use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
    use ferrum_types::Result;

    use super::*;
    use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyLayerStageConfig};

    #[derive(Default)]
    struct TinyLoader;

    impl WeightLoader<CpuBackend> for TinyLoader {
        fn load_tensor(&self, _name: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0])
        }

        fn load_linear(
            &self,
            _name: &str,
        ) -> Result<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
            Ok(Box::new(DenseLinear::<CpuBackend>::from_rows(&[1.0], 1, 1)))
        }

        fn has_tensor(&self, _name: &str) -> bool {
            false
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }

    fn tiny_config(num_layers: usize) -> LlamaFamilyConfig {
        LlamaFamilyConfig {
            hidden_size: 1,
            intermediate_size: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            num_layers,
            vocab_size: 1,
            max_seq_len: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rope_interleaved: false,
            has_qk_norm: false,
            sliding_window: 0,
        }
    }

    #[test]
    fn pipeline_prefill_matches_full_model_on_tiny_cpu_model() {
        let cfg = tiny_config(2);
        let loader = TinyLoader;
        let mut full = LlamaFamilyModel::<CpuBackend, KvFp16>::new(cfg.clone(), &loader).unwrap();
        let stage0 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg.clone(),
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(0..1, true, false),
        )
        .unwrap();
        let stage1 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(1..2, false, true),
        )
        .unwrap();
        let mut pipeline = LlamaFamilyPipelineModel::new(vec![stage0, stage1]).unwrap();

        let full_logits = full.prefill("full", &[0]);
        let pipeline_logits = pipeline.prefill("pipe", &[0]);

        assert_eq!(pipeline.config().num_layers, 2);
        assert_eq!(pipeline.stages().len(), 2);
        assert_eq!(full_logits.len(), pipeline_logits.len());
        assert!((full_logits[0] - pipeline_logits[0]).abs() < 1e-5);
    }

    #[test]
    fn pipeline_decode_matches_full_model_on_tiny_cpu_model() {
        let cfg = tiny_config(2);
        let loader = TinyLoader;
        let mut full = LlamaFamilyModel::<CpuBackend, KvFp16>::new(cfg.clone(), &loader).unwrap();
        let stage0 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg.clone(),
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(0..1, true, false),
        )
        .unwrap();
        let stage1 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(1..2, false, true),
        )
        .unwrap();
        let mut pipeline = LlamaFamilyPipelineModel::new(vec![stage0, stage1]).unwrap();

        let _ = full.prefill("full", &[0]);
        let _ = pipeline.prefill("pipe", &[0]);
        let full_logits = full.decode("full", 0, 1);
        let pipeline_logits = pipeline.decode("pipe", 0, 1);

        assert_eq!(full_logits.len(), pipeline_logits.len());
        assert!((full_logits[0] - pipeline_logits[0]).abs() < 1e-5);
    }
}
