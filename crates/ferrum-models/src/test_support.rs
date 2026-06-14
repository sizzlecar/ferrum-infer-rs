//! Deterministic tiny-model construction for full-stack tests.
//!
//! Gated behind the `test-support` feature so it never ships in release
//! builds. Downstream test code (e.g. `ferrum-engine/tests/tiny_stack.rs`)
//! uses this to assemble a real `ContinuousBatchEngine` over a real model
//! forward + real paged KV on the CPU backend — no GPU, no model download,
//! milliseconds per forward.
//!
//! The synthetic loader produces deterministic weights from a hash of each
//! tensor name, so a given [`TinyLlamaConfig`] always yields the same model
//! and therefore the same greedy token stream. Tests compute that stream
//! once and assert engine behavior (EOS, stop, streaming) against it.

use std::sync::Arc;

use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_interfaces::tokenizer::{TokenizerInfo, TokenizerType};
use ferrum_interfaces::{ModelExecutor, Tokenizer};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_quantization::{DenseLinear, QuantConfig, WeightLoader};
use ferrum_types::{
    DataType, Device, FerrumError, ModelInfo, ModelType, Result, SpecialTokens, TokenId,
};

use crate::models::llama_family::LlamaFamilyConfig;
use crate::models::LlamaFamilyModel;
use crate::LlmExecutor;

/// Shape of a deterministic tiny LLaMA-family model.
///
/// Defaults are larger than the op-parity micro config (vocab 7, hidden 4):
/// big enough that the greedy token stream varies across positions and
/// distinct tokens can be designated EOS / stop, small enough that a forward
/// is sub-millisecond on CPU.
#[derive(Debug, Clone)]
pub struct TinyLlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl Default for TinyLlamaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 32,
            intermediate_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            num_layers: 2,
            vocab_size: 48,
            max_seq_len: 128,
        }
    }
}

impl TinyLlamaConfig {
    fn to_llama_config(&self) -> LlamaFamilyConfig {
        LlamaFamilyConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            num_layers: self.num_layers,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rope_interleaved: false,
            has_qk_norm: false,
            sliding_window: 0,
            ..Default::default()
        }
    }
}

/// Deterministic synthetic weight loader. Promoted from the private
/// `ParityLoader` in `llama_family_pipeline.rs` tests so engine-level tests
/// can build the same model without copying the loader.
pub struct SyntheticLlamaLoader {
    cfg: LlamaFamilyConfig,
}

impl SyntheticLlamaLoader {
    pub fn new(cfg: LlamaFamilyConfig) -> Self {
        Self { cfg }
    }

    fn deterministic_values(name: &str, len: usize, base: f32, scale: f32) -> Vec<f32> {
        let mut hash = 0x811c_9dc5u32;
        for byte in name.bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        (0..len)
            .map(|idx| {
                let mixed = hash
                    .wrapping_add((idx as u32).wrapping_mul(0x9e37_79b9))
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
                "unexpected linear requested by synthetic loader: {name}"
            )))
        }
    }
}

impl WeightLoader<CpuBackend> for SyntheticLlamaLoader {
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
            "unexpected tensor requested by synthetic loader: {name}"
        )))
    }

    fn load_linear(&self, name: &str) -> Result<Box<dyn ferrum_quantization::Linear<CpuBackend>>> {
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

/// Identifier used for the tiny model everywhere (engine, executor, info).
pub const TINY_MODEL_ID: &str = "tiny-llama-test";

/// Build the tiny CPU model directly. Most tests want
/// [`tiny_llama_executor`] instead; this is for op-level access.
pub fn tiny_llama_model(cfg: &TinyLlamaConfig) -> LlamaFamilyModel<CpuBackend, KvFp16> {
    let loader = SyntheticLlamaLoader::new(cfg.to_llama_config());
    LlamaFamilyModel::<CpuBackend, KvFp16>::new(cfg.to_llama_config(), &loader)
        .expect("tiny llama model construction")
}

/// [`ModelInfo`] matching a [`TinyLlamaConfig`].
pub fn tiny_model_info(cfg: &TinyLlamaConfig) -> ModelInfo {
    ModelInfo {
        model_id: TINY_MODEL_ID.into(),
        model_type: ModelType::Custom("tiny-llama".into()),
        num_parameters: 0,
        hidden_size: cfg.hidden_size,
        num_layers: cfg.num_layers,
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.num_kv_heads,
        vocab_size: cfg.vocab_size,
        max_sequence_length: cfg.max_seq_len,
        dtype: DataType::FP16,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: Default::default(),
    }
}

/// Build a real [`ModelExecutor`] wrapping the tiny model — the seam the
/// engine scheduler calls. Real forward, real KV, no mock.
pub fn tiny_llama_executor(cfg: &TinyLlamaConfig) -> Arc<dyn ModelExecutor> {
    let model = tiny_llama_model(cfg);
    Arc::new(LlmExecutor::new(Box::new(model), tiny_model_info(cfg)))
}

/// Build a [`TinyTokenizer`] whose vocab matches the model's.
pub fn tiny_tokenizer(cfg: &TinyLlamaConfig) -> Arc<TinyTokenizer> {
    Arc::new(TinyTokenizer::new(cfg.vocab_size))
}

/// In-memory tokenizer with a fixed vocab and a real EOS token.
///
/// Each token id maps to a deterministic short text. By default token texts
/// are single-segment (`"t<id>"`); [`TinyTokenizer::with_composite_token`]
/// overrides a chosen id with a multi-character text so a stop string can be
/// embedded *inside* one token's decoded contribution — the exact shape the
/// composite-stop-detection fix (hb-02) guards. `decode` concatenates token
/// texts with no separator so embedded substrings survive.
pub struct TinyTokenizer {
    vocab_size: usize,
    special_tokens: SpecialTokens,
    texts: Vec<String>,
}

impl TinyTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        assert!(vocab_size >= 4, "tiny tokenizer needs at least 4 tokens");
        let eos = TokenId::new((vocab_size - 1) as u32);
        let bos = TokenId::new((vocab_size - 2) as u32);
        let texts = (0..vocab_size).map(|id| format!("t{id}")).collect();
        Self {
            vocab_size,
            special_tokens: SpecialTokens {
                bos_token: Some(bos),
                eos_token: Some(eos),
                unk_token: Some(TokenId::new(0)),
                pad_token: None,
                sep_token: None,
                cls_token: None,
                mask_token: None,
                extra_eos_tokens: Vec::new(),
            },
            texts,
        }
    }

    /// Override the decoded text of a single token id. Used to make a token
    /// the greedy model actually emits decode to a composite string that
    /// embeds a stop marker.
    pub fn with_composite_token(mut self, id: u32, text: &str) -> Self {
        let idx = id as usize;
        assert!(idx < self.vocab_size, "composite id out of range");
        self.texts[idx] = text.to_string();
        self
    }

    /// Override the EOS token id. The greedy stream is fixed by the model, so
    /// tests observe the stream first, then point EOS at a token the model
    /// actually emits to drive deterministic EOS termination.
    pub fn with_eos(mut self, id: u32) -> Self {
        assert!((id as usize) < self.vocab_size, "eos id out of range");
        self.special_tokens.eos_token = Some(TokenId::new(id));
        self
    }

    /// Parse a decoded string produced by this tokenizer's default
    /// (`"t<id>"`) text mapping back into token ids. Only valid when no
    /// composite overrides are in effect — the scenario helpers use a plain
    /// tokenizer for this.
    pub fn parse_default_ids(text: &str) -> Vec<u32> {
        text.split('t')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<u32>().ok())
            .collect()
    }

    pub fn eos_token_id(&self) -> u32 {
        self.special_tokens.eos_token.unwrap().get()
    }

    fn text_for(&self, id: u32) -> &str {
        self.texts
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("<unk>")
    }
}

impl Tokenizer for TinyTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        if add_special {
            if let Some(bos) = self.special_tokens.bos_token {
                tokens.push(bos);
            }
        }
        for word in text.split_whitespace() {
            let hash = word
                .bytes()
                .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            let id = 1 + (hash % (self.vocab_size as u32 - 3));
            tokens.push(TokenId::new(id));
        }
        if tokens.is_empty() {
            tokens.push(TokenId::new(1));
        }
        Ok(tokens)
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let mut out = String::new();
        for tok in tokens {
            if skip_special && self.is_special_token(*tok) {
                continue;
            }
            out.push_str(self.text_for(tok.get()));
        }
        Ok(out)
    }

    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        if self.is_special_token(next) {
            return Ok(String::new());
        }
        Ok(self.text_for(next.get()).to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.texts
            .iter()
            .position(|t| t == text)
            .map(|idx| TokenId::new(idx as u32))
    }

    fn token_text(&self, token_id: TokenId) -> Option<&str> {
        self.texts.get(token_id.get() as usize).map(String::as_str)
    }

    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::Custom,
            vocab_size: self.vocab_size,
            special_tokens: self.special_tokens.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: Some(64),
            model_name: Some(TINY_MODEL_ID.into()),
        }
    }
}
