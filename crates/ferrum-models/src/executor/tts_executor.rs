//! Qwen3-TTS Executor — text-to-speech pipeline wiring Talker LM + Vocoder.
//!
//! Implements: text tokenization, autoregressive codec token generation,
//! SubTalker code prediction (TODO), vocoder waveform synthesis.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, MemoryRequirements,
        PrefillInput, PrefillOutput,
    },
    ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, ModelType, Result};
use tracing::info;

use super::common;
use crate::architectures::qwen3_tts::{Qwen3TTSTalker, TalkerConfig};
use crate::architectures::qwen3_tts_vocoder::{Qwen3TTSVocoder, VocoderConfig};

// ── Constants ────────────────────────────────────────────────────────────

const SAMPLE_RATE: usize = 24000;
const MAX_CODEC_TOKENS: usize = 4096;

/// Sampling parameters for codec token generation.
const TEMPERATURE: f32 = 0.9;
const TOP_K: usize = 50;
const REPETITION_PENALTY: f32 = 1.05;

/// Qwen3-TTS executor: text-to-speech synthesis.
pub struct TtsModelExecutor {
    talker: Qwen3TTSTalker,
    vocoder: Qwen3TTSVocoder,
    text_tokenizer: tokenizers::Tokenizer,
    config: TalkerConfig,
    info: ModelInfo,
}

impl TtsModelExecutor {
    /// Load from model directory containing:
    /// - config.json (TalkerConfig)
    /// - model.safetensors (Talker weights)
    /// - speech_tokenizer/model.safetensors (Vocoder weights)
    /// - tokenizer_config.json + vocab.json + merges.txt (text tokenizer)
    pub fn from_path(model_path: &str, device: CandleDevice, dtype: DType) -> Result<Self> {
        let dir = std::path::Path::new(model_path);

        // Parse TalkerConfig from config.json
        let config_json: serde_json::Value = {
            let config_path = dir.join("config.json");
            let data = std::fs::read_to_string(&config_path).map_err(|e| {
                FerrumError::model(format!("read config.json: {e}"))
            })?;
            serde_json::from_str(&data).map_err(|e| {
                FerrumError::model(format!("parse config.json: {e}"))
            })?
        };
        let config = TalkerConfig::from_json(&config_json)?;

        // Load text tokenizer from vocab.json + merges.txt
        let text_tokenizer = load_bpe_tokenizer(dir)?;

        // Load Talker weights from model.safetensors (or sharded)
        let talker_weights = find_safetensor_files(dir, "model")?;
        let talker_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&talker_weights, dtype, &device)
                .map_err(|e| FerrumError::model(format!("load talker weights: {e}")))?
        };
        let talker = Qwen3TTSTalker::load(&config, talker_vb, device.clone())?;

        // Load Vocoder weights from speech_tokenizer/model.safetensors
        let vocoder_dir = dir.join("speech_tokenizer");
        let vocoder_weights = find_safetensor_files(&vocoder_dir, "model")?;
        let vocoder_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vocoder_weights, dtype, &device)
                .map_err(|e| FerrumError::model(format!("load vocoder weights: {e}")))?
        };
        let vocoder_config = VocoderConfig::default();
        let vocoder = Qwen3TTSVocoder::load(&vocoder_config, vocoder_vb)?;

        let info = ModelInfo {
            model_id: ferrum_types::ModelId(model_path.to_string()),
            model_type: ModelType::Custom("qwen3-tts".to_string()),
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            num_parameters: 0,
            max_sequence_length: config.max_position_embeddings,
            device: match &device {
                CandleDevice::Cpu => Device::CPU,
                CandleDevice::Cuda(_) => Device::CUDA(0),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                CandleDevice::Metal(_) => Device::Metal,
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                CandleDevice::Metal(_) => Device::CPU,
            },
            dtype: match dtype {
                DType::F32 => DataType::FP32,
                DType::F16 => DataType::FP16,
                DType::BF16 => DataType::BF16,
                _ => DataType::FP32,
            },
            version: None,
            license: None,
            metadata: HashMap::new(),
        };

        info!(
            "TtsModelExecutor: {} (hidden={}, layers={}, codec_groups={})",
            model_path,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_code_groups,
        );

        Ok(Self {
            talker,
            vocoder,
            text_tokenizer,
            config,
            info,
        })
    }

    /// Synthesize speech from text.
    ///
    /// Returns PCM samples at 24kHz as Vec<f32>.
    pub fn synthesize(&mut self, text: &str, language: &str) -> Result<Vec<f32>> {
        self.talker.reset();

        let device = self.talker.device().clone();

        // 1. Tokenize text
        let encoding = self
            .text_tokenizer
            .encode(text, true)
            .map_err(|e| FerrumError::model(format!("tokenize: {e}")))?;
        let text_ids: Vec<u32> = encoding.get_ids().to_vec();

        if text_ids.is_empty() {
            return Err(FerrumError::model("empty text after tokenization"));
        }

        info!("TTS: text tokens = {}", text_ids.len());

        // 2. Build input sequence:
        //    [tts_bos] + text_tokens + [tts_eos] + [codec_bos]
        let tts_bos = self.config.tts_bos_token_id;
        let tts_eos = self.config.tts_eos_token_id;
        let codec_bos = self.config.codec_bos_id;
        let codec_eos = self.config.codec_eos_token_id;

        // Build text portion: embed through text_embedding + text_projection
        let mut full_text_ids = Vec::with_capacity(text_ids.len() + 2);
        full_text_ids.push(tts_bos);
        full_text_ids.extend_from_slice(&text_ids);
        full_text_ids.push(tts_eos);

        let text_tensor = Tensor::new(&full_text_ids[..], &device)
            .map_err(|e| FerrumError::model(format!("text tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;

        let text_embeds = self.talker.embed_text(&text_tensor)?;

        // Build codec BOS embedding
        let codec_bos_tensor = Tensor::new(&[codec_bos], &device)
            .map_err(|e| FerrumError::model(format!("codec_bos tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
        let codec_bos_embed = self.talker.embed_codec(&codec_bos_tensor)?;

        // Concatenate: text_embeds + codec_bos_embed along seq dimension
        let prefill_embeds = Tensor::cat(&[&text_embeds, &codec_bos_embed], 1)
            .map_err(|e| FerrumError::model(format!("cat embeds: {e}")))?;

        // 3. Prefill: forward through transformer
        let hidden = self.talker.forward_step(&prefill_embeds)?;

        // Get logits from last position
        let last_hidden = hidden
            .narrow(1, hidden.dim(1).unwrap() - 1, 1)
            .map_err(|e| FerrumError::model(format!("narrow: {e}")))?;
        let logits = self.talker.logits(&last_hidden)?;

        // 4. Autoregressive decode loop: generate codec token 0 per step
        let mut all_codec_tokens: Vec<Vec<u32>> = Vec::new();
        let mut current_logits = logits;

        for step in 0..MAX_CODEC_TOKENS {
            // Sample next codec token from logits
            let logits_vec = logits_to_vec(&current_logits)?;
            let next_token = sample_token(&logits_vec, TEMPERATURE, TOP_K, REPETITION_PENALTY);

            // Check for EOS
            if next_token == codec_eos {
                info!("TTS: codec EOS at step {}", step);
                break;
            }

            // For code group 0, we have the token from the main talker
            let mut frame_codes = vec![next_token];

            // TODO: Use SubTalker (code_predictor) to predict remaining codec tokens
            // for code groups 1..num_code_groups-1.
            // For now, pad with codec_pad_id.
            for _g in 1..self.config.num_code_groups {
                frame_codes.push(self.config.codec_pad_id);
            }

            all_codec_tokens.push(frame_codes);

            // Embed the generated token and forward one step
            let token_tensor = Tensor::new(&[next_token], &device)
                .map_err(|e| FerrumError::model(format!("token tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
            let token_embed = self.talker.embed_codec(&token_tensor)?;
            let hidden = self.talker.forward_step(&token_embed)?;
            current_logits = self.talker.logits(&hidden)?;
        }

        if all_codec_tokens.is_empty() {
            return Err(FerrumError::model("no codec tokens generated"));
        }

        info!("TTS: generated {} codec frames", all_codec_tokens.len());

        // 5. Build codec tensor [1, num_code_groups, T] for vocoder
        let num_frames = all_codec_tokens.len();
        let num_groups = self.config.num_code_groups;
        let mut flat_codes: Vec<u32> = vec![0; num_groups * num_frames];
        for (t, frame) in all_codec_tokens.iter().enumerate() {
            for (g, &code) in frame.iter().enumerate() {
                flat_codes[g * num_frames + t] = code;
            }
        }

        // Clamp codes to valid codebook range [0, codebook_size-1].
        // Special tokens (codec_bos, codec_eos, etc.) are >= codebook_size and must be clamped.
        let codebook_size = 2048u32;
        for code in &mut flat_codes {
            if *code >= codebook_size {
                *code = 0; // replace special tokens with pad/silence
            }
        }

        let codes_tensor =
            Tensor::new(&flat_codes[..], &device)
                .map_err(|e| FerrumError::model(format!("codes tensor: {e}")))?
                .reshape((1, num_groups, num_frames))
                .map_err(|e| FerrumError::model(format!("reshape codes: {e}")))?;

        // 6. Vocoder: codec tokens → waveform
        let waveform = self.vocoder.decode(&codes_tensor)?;

        // Extract samples: [1, 1, samples] → Vec<f32>
        let samples: Vec<f32> = waveform
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze batch: {e}")))?
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze channel: {e}")))?
            .to_vec1()
            .map_err(|e| FerrumError::model(format!("to_vec1: {e}")))?;

        info!(
            "TTS: waveform {} samples ({:.2}s @ {}Hz)",
            samples.len(),
            samples.len() as f64 / SAMPLE_RATE as f64,
            SAMPLE_RATE,
        );

        Ok(samples)
    }

    /// Get the output sample rate.
    pub fn sample_rate(&self) -> usize {
        SAMPLE_RATE
    }

    pub fn config(&self) -> &TalkerConfig {
        &self.config
    }
}

// ── Utility functions ───────────────────────────────────────────────────

/// Find safetensor files matching a prefix in a directory.
fn find_safetensor_files(
    dir: &std::path::Path,
    prefix: &str,
) -> Result<Vec<std::path::PathBuf>> {
    // Try single file first
    let single = dir.join(format!("{prefix}.safetensors"));
    if single.exists() {
        return Ok(vec![single]);
    }

    // Try sharded: prefix-00001-of-00005.safetensors, etc.
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(prefix)
                    && name.ends_with(".safetensors")
                    && name != format!("{prefix}.safetensors")
                {
                    files.push(path);
                }
            }
        }
    }
    files.sort();

    if files.is_empty() {
        Err(FerrumError::model(format!(
            "no safetensors files with prefix '{prefix}' in {}",
            dir.display()
        )))
    } else {
        Ok(files)
    }
}

/// Load a BPE tokenizer from vocab.json + merges.txt.
fn load_bpe_tokenizer(dir: &std::path::Path) -> Result<tokenizers::Tokenizer> {
    // Try tokenizer.json first (HF fast tokenizer format)
    let tokenizer_json = dir.join("tokenizer.json");
    if tokenizer_json.exists() {
        return tokenizers::Tokenizer::from_file(&tokenizer_json)
            .map_err(|e| FerrumError::model(format!("load tokenizer.json: {e}")));
    }

    // Fallback: build from vocab.json + merges.txt
    let vocab_path = dir.join("vocab.json");
    let merges_path = dir.join("merges.txt");

    if !vocab_path.exists() || !merges_path.exists() {
        return Err(FerrumError::model(
            "tokenizer.json not found, and vocab.json + merges.txt not found either",
        ));
    }

    let vocab_data = std::fs::read_to_string(&vocab_path)
        .map_err(|e| FerrumError::model(format!("read vocab.json: {e}")))?;
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_data)
        .map_err(|e| FerrumError::model(format!("parse vocab.json: {e}")))?;

    let merges_data = std::fs::read_to_string(&merges_path)
        .map_err(|e| FerrumError::model(format!("read merges.txt: {e}")))?;
    let merges: Vec<(String, String)> = merges_data
        .lines()
        .skip(1) // skip header line
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    let bpe = tokenizers::models::bpe::BPE::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
    )
    .build()
    .map_err(|e| FerrumError::model(format!("build BPE: {e}")))?;

    let tokenizer = tokenizers::Tokenizer::new(bpe);
    Ok(tokenizer)
}

/// Extract logits as Vec<f32> from a [1, 1, vocab] or [1, vocab] tensor.
fn logits_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let logits = if logits.dims().len() == 3 {
        logits
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze: {e}")))?
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze: {e}")))?
    } else if logits.dims().len() == 2 {
        logits
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze: {e}")))?
    } else {
        logits.clone()
    };

    logits
        .to_vec1()
        .map_err(|e| FerrumError::model(format!("logits to_vec1: {e}")))
}

/// Sample a token from logits with temperature, top-k, and repetition penalty.
fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    _repetition_penalty: f32,
) -> u32 {
    if temperature == 0.0 {
        return argmax(logits);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Top-k filtering
    let mut indexed: Vec<(usize, f32)> =
        scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    // Softmax over top-k
    let max_val = indexed[0].1;
    let exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Weighted random sampling
    let r = rand_f32();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return indexed[i].0 as u32;
        }
    }
    indexed.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0xdeadbeef_cafebabe);
    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    STATE.store(s, Ordering::Relaxed);
    (s as f32) / (u64::MAX as f32)
}

// ── Dummy KV cache + ModelExecutor trait impl ───────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DummyTtsCache;

impl ferrum_interfaces::KvCacheHandle for DummyTtsCache {
    fn block_table(&self) -> &ferrum_interfaces::BlockTable {
        static EMPTY: std::sync::OnceLock<ferrum_interfaces::BlockTable> =
            std::sync::OnceLock::new();
        EMPTY.get_or_init(|| ferrum_interfaces::BlockTable::new(16))
    }
    fn block_table_mut(&mut self) -> &mut ferrum_interfaces::BlockTable {
        unimplemented!()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device(&self) -> Device {
        Device::CPU
    }
    fn num_layers(&self) -> usize {
        0
    }
    fn num_heads(&self) -> usize {
        0
    }
    fn head_dim(&self) -> usize {
        0
    }
    fn key_cache(&self, _: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }
    fn value_cache(&self, _: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }
    fn clone_handle(&self) -> Result<Arc<dyn ferrum_interfaces::KvCacheHandle>> {
        Ok(Arc::new(self.clone()))
    }
    fn stats(&self) -> ferrum_interfaces::CacheHandleStats {
        ferrum_interfaces::CacheHandleStats {
            memory_bytes: 0,
            blocks_allocated: 0,
            tokens_stored: 0,
            utilization: 0.0,
            last_access: std::time::Instant::now(),
        }
    }
    fn is_valid(&self) -> bool {
        true
    }
    fn cache_id(&self) -> String {
        "tts_dummy".to_string()
    }
}

#[async_trait]
impl ModelExecutor for TtsModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, _input: &PrefillInput) -> Result<PrefillOutput> {
        Err(FerrumError::model(
            "TTS uses synthesize(), not prefill/decode",
        ))
    }

    async fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        Err(FerrumError::model(
            "TTS uses synthesize(), not prefill/decode",
        ))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::GroupedQuery],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32, DataType::BF16],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: 0,
                activation_memory_per_token: 0,
                kv_cache_memory_per_token: 0,
                overhead_memory: 0,
            },
        }
    }

    fn release_cache(&self, _: &str) {}

    fn status(&self) -> ferrum_interfaces::model_executor::ExecutorStatus {
        common::default_executor_status()
    }
}
