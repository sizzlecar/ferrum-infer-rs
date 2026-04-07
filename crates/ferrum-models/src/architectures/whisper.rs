//! Whisper ASR model wrapper.
//!
//! Wraps candle-transformers' Whisper for speech-to-text transcription.
//! Supports all Whisper variants (tiny, base, small, medium, large-v3).

use candle_core::{DType, Device as CandleDevice, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, model::Whisper, Config};
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use tracing::info;

/// Whisper model wrapper for speech-to-text.
pub struct WhisperModelWrapper {
    model: Mutex<Whisper>,
    config: Config,
    mel_filters: Vec<f32>,
    device: CandleDevice,
    dtype: DType,
}

impl WhisperModelWrapper {
    /// Load Whisper from VarBuilder + config.
    pub fn new(
        vb: VarBuilder,
        config: Config,
        mel_filters: Vec<f32>,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!(
            "Loading Whisper (d_model={}, encoder_layers={}, decoder_layers={})",
            config.d_model, config.encoder_layers, config.decoder_layers
        );
        let model = Whisper::load(&vb, config.clone())
            .map_err(|e| FerrumError::model(format!("Whisper load: {e}")))?;
        Ok(Self {
            model: Mutex::new(model),
            config,
            mel_filters,
            device,
            dtype,
        })
    }

    /// Load from model directory (config.json + safetensors + mel_filters).
    pub fn from_model_dir(
        model_dir: &std::path::Path,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| FerrumError::model(format!("read config: {e}")))?,
        )
        .map_err(|e| FerrumError::model(format!("parse config: {e}")))?;

        // Load mel filters
        let mel_bytes = match config.num_mel_bins {
            128 => include_bytes!("mel_filters128.bin").as_slice(),
            _ => include_bytes!("mel_filters80.bin").as_slice(),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        for (i, chunk) in mel_bytes.chunks_exact(4).enumerate() {
            mel_filters[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        // Load weights
        let safetensors: Vec<_> = std::fs::read_dir(model_dir)
            .map_err(|e| FerrumError::model(format!("read dir: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();

        if safetensors.is_empty() {
            return Err(FerrumError::model("No safetensors files found"));
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors, dtype, &device)
                .map_err(|e| FerrumError::model(format!("load weights: {e}")))?
        };

        Self::new(vb, config, mel_filters, device, dtype)
    }

    /// Convert PCM audio samples (16kHz, mono, f32) to mel spectrogram tensor.
    pub fn pcm_to_mel_tensor(&self, pcm: &[f32]) -> Result<Tensor> {
        let mel = whisper::audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let mel_len = mel.len() / self.config.num_mel_bins;
        Tensor::from_vec(mel, (1, self.config.num_mel_bins, mel_len), &self.device)
            .map_err(|e| FerrumError::model(format!("mel tensor: {e}")))
    }

    /// Encode audio: mel spectrogram → encoder hidden states.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let mut model = self.model.lock();
        model.reset_kv_cache();
        model
            .encoder
            .forward(mel, true)
            .map_err(|e| FerrumError::model(format!("encode: {e}")))
    }

    /// Decode one token given encoder output and previous tokens.
    pub fn decode_token(&self, token: &Tensor, encoder_out: &Tensor) -> Result<Tensor> {
        let mut model = self.model.lock();
        let logits = model
            .decoder
            .forward(token, encoder_out, false)
            .map_err(|e| FerrumError::model(format!("decode: {e}")))?;
        model
            .decoder
            .final_linear(
                &logits
                    .i((
                        ..,
                        logits
                            .dim(1)
                            .map_err(|e| FerrumError::model(format!("dim: {e}")))?
                            - 1..,
                    ))
                    .map_err(|e| FerrumError::model(format!("slice: {e}")))?,
            )
            .map_err(|e| FerrumError::model(format!("final_linear: {e}")))
    }

    /// Full transcription: PCM → text token IDs.
    /// Returns Vec of token IDs (caller decodes with tokenizer).
    pub fn transcribe(
        &self,
        pcm: &[f32],
        language_token: u32,
        task_token: u32,
        no_timestamps_token: u32,
        eot_token: u32,
        sot_token: u32,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mel = self.pcm_to_mel_tensor(pcm)?;
        let encoder_out = self.encode(&mel)?;

        // Initial tokens: SOT, language, task, no_timestamps
        let mut tokens: Vec<u32> = vec![sot_token, language_token, task_token, no_timestamps_token];
        let mut all_tokens: Vec<u32> = Vec::new();

        // Reset decoder KV cache for fresh decode
        {
            let mut model = self.model.lock();
            model.decoder.reset_kv_cache();
        }

        for _ in 0..max_tokens {
            let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| FerrumError::model(format!("token tensor: {e}")))?;

            let logits = self.decode_token(&token_tensor, &encoder_out)?;
            let logits = logits
                .squeeze(0)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| FerrumError::model(format!("squeeze: {e}")))?;

            let next_token = logits
                .argmax(0)
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| FerrumError::model(format!("argmax: {e}")))?;

            if next_token == eot_token {
                break;
            }

            all_tokens.push(next_token);
            // For subsequent iterations, only feed the new token (KV cache handles history)
            tokens = vec![next_token];
        }

        Ok(all_tokens)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &CandleDevice {
        &self.device
    }
}
