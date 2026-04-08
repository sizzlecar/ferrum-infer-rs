//! Whisper ASR Executor for speech-to-text transcription.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device as CandleDevice};
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
use crate::architectures::whisper::WhisperModelWrapper;
use crate::audio_processor;

/// Whisper executor for speech-to-text.
pub struct WhisperModelExecutor {
    model: WhisperModelWrapper,
    tokenizer: tokenizers::Tokenizer,
    info: ModelInfo,
    // Special token IDs
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
    language_tokens: HashMap<String, u32>,
}

impl WhisperModelExecutor {
    /// Load from model directory.
    pub fn from_path(model_path: &str, device: CandleDevice, dtype: DType) -> Result<Self> {
        let dir = std::path::Path::new(model_path);

        let model = WhisperModelWrapper::from_model_dir(dir, device, dtype)?;

        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| FerrumError::model(format!("load tokenizer: {e}")))?;

        // Resolve special token IDs
        let sot_token = token_id(&tokenizer, "<|startoftranscript|>");
        let eot_token = token_id(&tokenizer, "<|endoftext|>");
        let transcribe_token = token_id(&tokenizer, "<|transcribe|>");
        let no_timestamps_token = token_id(&tokenizer, "<|notimestamps|>");

        // Build language token map
        let mut language_tokens = HashMap::new();
        for lang in &[
            "en", "zh", "ja", "ko", "fr", "de", "es", "ru", "ar", "pt", "it",
        ] {
            let token_str = format!("<|{lang}|>");
            if let Some(id) = tokenizer.token_to_id(&token_str) {
                language_tokens.insert(lang.to_string(), id);
            }
        }

        let info = ModelInfo {
            model_id: ferrum_types::ModelId(model_path.to_string()),
            model_type: ModelType::Custom("whisper".to_string()),
            hidden_size: model.config().d_model,
            vocab_size: model.config().vocab_size,
            num_layers: model.config().encoder_layers + model.config().decoder_layers,
            num_heads: model.config().encoder_attention_heads,
            num_kv_heads: model.config().decoder_attention_heads,
            num_parameters: 0,
            max_sequence_length: model.config().max_target_positions,
            device: Device::CPU,
            dtype: DataType::FP32,
            version: None,
            license: None,
            metadata: HashMap::new(),
        };

        info!(
            "WhisperModelExecutor: {} (d_model={}, languages={})",
            model_path,
            model.config().d_model,
            language_tokens.len()
        );

        Ok(Self {
            model,
            tokenizer,
            info,
            sot_token,
            eot_token,
            transcribe_token,
            no_timestamps_token,
            language_tokens,
        })
    }

    /// Transcribe audio file → text.
    pub fn transcribe_file(&self, audio_path: &str, language: Option<&str>) -> Result<String> {
        let pcm = audio_processor::load_audio(audio_path)?;
        self.transcribe_pcm(&pcm, language)
    }

    /// Transcribe raw audio bytes (WAV) → text.
    pub fn transcribe_bytes(&self, audio_data: &[u8], language: Option<&str>) -> Result<String> {
        let pcm = audio_processor::load_audio_bytes(audio_data)?;
        self.transcribe_pcm(&pcm, language)
    }

    /// Transcribe PCM samples → text. Automatically chunks long audio into 30s segments.
    fn transcribe_pcm(&self, pcm: &[f32], language: Option<&str>) -> Result<String> {
        let lang_token = language
            .and_then(|l| self.language_tokens.get(l).copied())
            .unwrap_or_else(|| {
                self.language_tokens
                    .get("en")
                    .copied()
                    .unwrap_or(self.sot_token + 1)
            });

        let chunks = audio_processor::chunk_pcm(pcm);
        let total = chunks.len();
        let mut all_text = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            if total > 1 {
                tracing::info!("Transcribing chunk {}/{} ...", i + 1, total);
            }

            let token_ids = self.model.transcribe(
                chunk,
                lang_token,
                self.transcribe_token,
                self.no_timestamps_token,
                self.eot_token,
                self.sot_token,
                448,
            )?;

            let text = self
                .tokenizer
                .decode(&token_ids, true)
                .map_err(|e| FerrumError::model(format!("decode tokens: {e}")))?;

            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                all_text.push(trimmed);
            }
        }

        Ok(all_text.join(" "))
    }
}

fn token_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> u32 {
    tokenizer.token_to_id(token).unwrap_or(0)
}

// Dummy KV cache (same pattern as CLIP/BERT)
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DummyWhisperCache;

impl ferrum_interfaces::KvCacheHandle for DummyWhisperCache {
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
        "whisper_dummy".to_string()
    }
}

#[async_trait]
impl ModelExecutor for WhisperModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, _input: &PrefillInput) -> Result<PrefillOutput> {
        Err(FerrumError::model(
            "Whisper uses transcribe(), not prefill/decode",
        ))
    }

    async fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        Err(FerrumError::model(
            "Whisper uses transcribe(), not prefill/decode",
        ))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32],
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
