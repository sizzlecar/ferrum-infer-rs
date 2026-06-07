use ferrum_kernels::backend::{KvLayer, MoeLlmBackend};
use ferrum_types::{FerrumError, Result};

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};

use super::llama_family::LlamaFamilyModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPipelineTransport {
    HostHiddenBridge,
}

impl LlamaPipelineTransport {
    fn as_str(self) -> &'static str {
        match self {
            Self::HostHiddenBridge => "host-hidden-bridge",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaPipelineStagePlacement {
    pub backend_device_ordinal: Option<usize>,
}

impl LlamaPipelineStagePlacement {
    pub fn default_backend_device() -> Self {
        Self {
            backend_device_ordinal: None,
        }
    }

    pub fn backend_device(ordinal: usize) -> Self {
        Self {
            backend_device_ordinal: Some(ordinal),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaPipelinePlacement {
    stages: Vec<LlamaPipelineStagePlacement>,
    transport: LlamaPipelineTransport,
}

impl LlamaPipelinePlacement {
    pub fn unplaced(stage_count: usize) -> Self {
        Self {
            stages: vec![LlamaPipelineStagePlacement::default_backend_device(); stage_count],
            transport: LlamaPipelineTransport::HostHiddenBridge,
        }
    }

    pub fn from_backend_device_ordinals(stage_device_ordinals: Vec<Option<usize>>) -> Self {
        Self {
            stages: stage_device_ordinals
                .into_iter()
                .map(|backend_device_ordinal| LlamaPipelineStagePlacement {
                    backend_device_ordinal,
                })
                .collect(),
            transport: LlamaPipelineTransport::HostHiddenBridge,
        }
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    pub fn stage(&self, idx: usize) -> LlamaPipelineStagePlacement {
        self.stages[idx]
    }

    pub fn stages(&self) -> &[LlamaPipelineStagePlacement] {
        &self.stages
    }

    pub fn transport(&self) -> LlamaPipelineTransport {
        self.transport
    }

    pub fn stage_device_ordinals(&self) -> Vec<Option<usize>> {
        self.stages
            .iter()
            .map(|stage| stage.backend_device_ordinal)
            .collect()
    }

    fn has_explicit_device_ordinals(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| stage.backend_device_ordinal.is_some())
    }
}

pub struct LlamaFamilyPipelineModel<B: MoeLlmBackend, K: KvLayer<B>> {
    stages: Vec<LlamaFamilyModel<B, K>>,
    placement: LlamaPipelinePlacement,
    runtime_cfg: LlmRuntimeConfig,
}

impl<B: MoeLlmBackend, K: KvLayer<B>> LlamaFamilyPipelineModel<B, K> {
    pub fn new(stages: Vec<LlamaFamilyModel<B, K>>) -> Result<Self> {
        let placement = LlamaPipelinePlacement::unplaced(stages.len());
        Self::new_with_placement(stages, placement)
    }

    pub fn new_with_backend_device_ordinals(
        stages: Vec<LlamaFamilyModel<B, K>>,
        stage_device_ordinals: Vec<Option<usize>>,
    ) -> Result<Self> {
        Self::new_with_placement(
            stages,
            LlamaPipelinePlacement::from_backend_device_ordinals(stage_device_ordinals),
        )
    }

    pub fn new_with_placement(
        stages: Vec<LlamaFamilyModel<B, K>>,
        placement: LlamaPipelinePlacement,
    ) -> Result<Self> {
        if stages.is_empty() {
            return Err(FerrumError::model(
                "LlamaFamilyPipelineModel requires at least one stage",
            ));
        }
        if placement.len() != stages.len() {
            return Err(FerrumError::model(format!(
                "Llama pipeline stage device count {} must match stage count {}",
                placement.len(),
                stages.len()
            )));
        }
        if placement.has_explicit_device_ordinals() && !B::supports_device_ordinal_scope() {
            return Err(FerrumError::unsupported(
                "Llama layer-split pipeline requested explicit backend device ordinals, \
                 but the selected backend does not support device-scoped execution",
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
            placement,
            runtime_cfg,
        })
    }

    pub fn stages(&self) -> &[LlamaFamilyModel<B, K>] {
        &self.stages
    }

    pub fn placement(&self) -> &LlamaPipelinePlacement {
        &self.placement
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
            "stage_device_ordinals": self.placement.stage_device_ordinals(),
            "transport": self.placement.transport().as_str(),
        }))
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            let device = self.placement.stage(idx).backend_device_ordinal;
            B::with_device_ordinal(device, || {
                stage.ensure_scratch(max_tokens);
                stage.ensure_kv(cache_id);
            });
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
            B::with_device_ordinal(self.placement.stage(0).backend_device_ordinal, || {
                self.stages[0].prefill_stage_tokens_to_hidden(cache_id, tokens, pos_offset)
            });
        for idx in 1..self.stages.len() {
            let device = self.placement.stage(idx).backend_device_ordinal;
            hidden = B::with_device_ordinal(device, || {
                self.stages[idx].prefill_stage_hidden_from_host(
                    cache_id,
                    &hidden,
                    tokens.len(),
                    pos_offset,
                )
            });
        }
        let last_hidden = self.last_hidden_row(&hidden, tokens.len()).to_vec();
        let last_idx = self.stages.len() - 1;
        B::with_device_ordinal(
            self.placement.stage(last_idx).backend_device_ordinal,
            || self.stages[last_idx].logits_from_hidden(&last_hidden),
        )
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        let mut hidden =
            B::with_device_ordinal(self.placement.stage(0).backend_device_ordinal, || {
                self.stages[0].decode_stage_token_to_hidden(cache_id, token, pos)
            });
        for idx in 1..self.stages.len() {
            let device = self.placement.stage(idx).backend_device_ordinal;
            hidden = B::with_device_ordinal(device, || {
                self.stages[idx].decode_stage_hidden_from_host(cache_id, &hidden, pos)
            });
        }
        let last_idx = self.stages.len() - 1;
        B::with_device_ordinal(
            self.placement.stage(last_idx).backend_device_ordinal,
            || self.stages[last_idx].logits_from_hidden(&hidden),
        )
    }

    fn release(&mut self, cache_id: &str) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            B::with_device_ordinal(self.placement.stage(idx).backend_device_ordinal, || {
                stage.release(cache_id);
            });
        }
    }

    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            B::with_device_ordinal(self.placement.stage(idx).backend_device_ordinal, || {
                stage.truncate_kv(cache_id, new_len);
            });
        }
    }

    fn reset(&mut self) {
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            B::with_device_ordinal(self.placement.stage(idx).backend_device_ordinal, || {
                stage.reset();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use ferrum_interfaces::kv_dtype::KvFp16;
    use ferrum_kernels::backend::cpu::CpuBackend;
    use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
    use ferrum_types::{FerrumError, Result};

    use super::*;
    use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyLayerStageConfig};

    struct ParityLoader {
        cfg: LlamaFamilyConfig,
    }

    impl ParityLoader {
        fn new(cfg: LlamaFamilyConfig) -> Self {
            Self { cfg }
        }

        fn deterministic_values(name: &str, len: usize, base: f32, scale: f32) -> Vec<f32> {
            let mut hash = 0x811c9dc5u32;
            for byte in name.bytes() {
                hash ^= byte as u32;
                hash = hash.wrapping_mul(0x01000193);
            }
            (0..len)
                .map(|idx| {
                    let mixed = hash
                        .wrapping_add((idx as u32).wrapping_mul(0x9e3779b9))
                        .rotate_left((idx % 17) as u32);
                    let centered = (mixed % 23) as f32 - 11.0;
                    base + centered * scale
                })
                .collect()
        }

        fn layer_norm_values(&self, name: &str) -> Vec<f32> {
            Self::deterministic_values(name, self.cfg.hidden_size, 1.0, 0.005)
        }

        fn linear_dims(&self, name: &str) -> Result<(usize, usize)> {
            let q_dim = self.cfg.num_heads * self.cfg.head_dim;
            let kv_dim = self.cfg.num_kv_heads * self.cfg.head_dim;
            if name.ends_with(".self_attn.qkv_proj") {
                Ok((q_dim + 2 * kv_dim, self.cfg.hidden_size))
            } else if name.ends_with(".self_attn.o_proj") {
                Ok((self.cfg.hidden_size, q_dim))
            } else if name.ends_with(".mlp.gate_up_proj") {
                Ok((2 * self.cfg.intermediate_size, self.cfg.hidden_size))
            } else if name.ends_with(".mlp.down_proj") {
                Ok((self.cfg.hidden_size, self.cfg.intermediate_size))
            } else if name == "lm_head" || name == "model.embed_tokens" {
                Ok((self.cfg.vocab_size, self.cfg.hidden_size))
            } else {
                Err(FerrumError::model(format!(
                    "unexpected linear requested by parity loader: {name}"
                )))
            }
        }
    }

    impl WeightLoader<CpuBackend> for ParityLoader {
        fn load_tensor(&self, name: &str) -> Result<Vec<f32>> {
            if name == "model.embed_tokens.weight" {
                return Ok(Self::deterministic_values(
                    name,
                    self.cfg.vocab_size * self.cfg.hidden_size,
                    0.0,
                    0.02,
                ));
            }
            if name == "model.norm.weight"
                || name.ends_with(".input_layernorm.weight")
                || name.ends_with(".post_attention_layernorm.weight")
            {
                return Ok(self.layer_norm_values(name));
            }
            Err(FerrumError::model(format!(
                "unexpected tensor requested by parity loader: {name}"
            )))
        }

        fn load_linear(
            &self,
            name: &str,
        ) -> Result<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
            let (out_features, in_features) = self.linear_dims(name)?;
            let weights = Self::deterministic_values(name, out_features * in_features, 0.0, 0.015);
            Ok(Box::new(DenseLinear::<CpuBackend>::from_rows(
                &weights,
                out_features,
                in_features,
            )))
        }

        fn has_tensor(&self, name: &str) -> bool {
            name == "lm_head.weight"
        }

        fn quant_config(&self) -> Option<&QuantConfig> {
            None
        }
    }

    fn parity_config(num_layers: usize) -> LlamaFamilyConfig {
        LlamaFamilyConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers,
            vocab_size: 7,
            max_seq_len: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rope_interleaved: false,
            has_qk_norm: false,
            sliding_window: 0,
        }
    }

    fn build_full_and_pipeline() -> (
        LlamaFamilyModel<CpuBackend, KvFp16>,
        LlamaFamilyPipelineModel<CpuBackend, KvFp16>,
    ) {
        let cfg = parity_config(3);
        let loader = ParityLoader::new(cfg.clone());
        let full = LlamaFamilyModel::<CpuBackend, KvFp16>::new(cfg.clone(), &loader).unwrap();
        let stage0 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg.clone(),
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(0..1, true, false),
        )
        .unwrap();
        let stage1 = LlamaFamilyModel::<CpuBackend, KvFp16>::new_layer_stage(
            cfg,
            &loader,
            LlamaFamilyLayerStageConfig::pipeline_stage(1..3, false, true),
        )
        .unwrap();
        let pipeline = LlamaFamilyPipelineModel::new(vec![stage0, stage1]).unwrap();
        (full, pipeline)
    }

    fn assert_logits_close(label: &str, expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
        let max_diff = expected
            .iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "{label} logits diverged: max_diff={max_diff} expected={expected:?} actual={actual:?}"
        );
    }

    #[test]
    fn pipeline_prefill_matches_full_model_on_multi_token_cpu_model() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let full_logits = full.prefill("full", &[0, 1, 2, 3]);
        let pipeline_logits = pipeline.prefill("pipe", &[0, 1, 2, 3]);

        assert_eq!(pipeline.config().num_layers, 3);
        assert_eq!(pipeline.stages().len(), 2);
        assert_eq!(
            pipeline.placement().stage_device_ordinals(),
            vec![None, None]
        );
        assert_eq!(
            pipeline.placement().transport(),
            LlamaPipelineTransport::HostHiddenBridge
        );
        assert_logits_close("multi-token prefill", &full_logits, &pipeline_logits);
    }

    #[test]
    fn pipeline_decode_after_multi_token_prefill_matches_full_model() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let _ = full.prefill("full", &[0, 1, 2]);
        let _ = pipeline.prefill("pipe", &[0, 1, 2]);
        let full_logits_3 = full.decode("full", 3, 3);
        let pipeline_logits_3 = pipeline.decode("pipe", 3, 3);
        assert_logits_close("decode pos 3", &full_logits_3, &pipeline_logits_3);

        let full_logits_4 = full.decode("full", 4, 4);
        let pipeline_logits_4 = pipeline.decode("pipe", 4, 4);
        assert_logits_close("decode pos 4", &full_logits_4, &pipeline_logits_4);
    }

    #[test]
    fn pipeline_incremental_prefill_matches_full_model_position_offset() {
        let (mut full, mut pipeline) = build_full_and_pipeline();

        let _ = full.prefill("full", &[0, 1]);
        let _ = pipeline.prefill("pipe", &[0, 1]);
        let full_logits = full.prefill("full", &[2, 3]);
        let pipeline_logits = pipeline.prefill("pipe", &[2, 3]);

        assert_logits_close("incremental prefill", &full_logits, &pipeline_logits);
    }

    #[test]
    fn pipeline_rejects_device_ordinals_without_backend_scope_support() {
        let cfg = parity_config(2);
        let loader = ParityLoader::new(cfg.clone());
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

        let err = match LlamaFamilyPipelineModel::new_with_backend_device_ordinals(
            vec![stage0, stage1],
            vec![Some(0), Some(1)],
        ) {
            Ok(_) => panic!("pipeline unexpectedly accepted unsupported device ordinals"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("does not support device-scoped execution"));
    }
}
