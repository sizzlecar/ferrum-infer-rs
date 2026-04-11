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
use crate::architectures::qwen3_tts::{Qwen3TTSTalker, SubTalker, TalkerConfig};
use crate::architectures::qwen3_tts_vocoder::{Qwen3TTSVocoder, VocoderConfig};
use crate::architectures::speaker_encoder::{mel_spectrogram_speaker_encoder, SpeakerEncoder};
use crate::architectures::speech_tokenizer_encoder::SpeechTokenizerEncoder;

// ── Constants ────────────────────────────────────────────────────────────

const SAMPLE_RATE: usize = 24000;
const MAX_CODEC_TOKENS: usize = 2000;

/// Sampling parameters for codec token generation.
const TEMPERATURE: f32 = 0.9;
const TOP_K: usize = 50;
const REPETITION_PENALTY: f32 = 1.05;

/// Qwen3-TTS executor: text-to-speech synthesis.
pub struct TtsModelExecutor {
    talker: Qwen3TTSTalker,
    sub_talker: SubTalker,
    vocoder: Qwen3TTSVocoder,
    text_tokenizer: tokenizers::Tokenizer,
    config: TalkerConfig,
    info: ModelInfo,
    speaker_encoder: Option<SpeakerEncoder>,
    speech_tokenizer_encoder: Option<SpeechTokenizerEncoder>,
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
            let data = std::fs::read_to_string(&config_path)
                .map_err(|e| FerrumError::model(format!("read config.json: {e}")))?;
            serde_json::from_str(&data)
                .map_err(|e| FerrumError::model(format!("parse config.json: {e}")))?
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
        let talker = Qwen3TTSTalker::load(&config, talker_vb.clone(), device.clone())?;

        // Load SubTalker (code predictor) from same weights file
        let sub_talker = SubTalker::load(&config, talker_vb.clone(), device.clone())?;

        // Load Speaker Encoder (for voice cloning, base models only)
        let speaker_encoder = SpeakerEncoder::load(talker_vb.pp("speaker_encoder"))
            .map_err(|e| {
                tracing::warn!("Speaker encoder not available: {e}");
                e
            })
            .ok();

        // Load Vocoder weights from speech_tokenizer/model.safetensors
        let vocoder_dir = dir.join("speech_tokenizer");
        let vocoder_weights = find_safetensor_files(&vocoder_dir, "model")?;
        let vocoder_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vocoder_weights, dtype, &device)
                .map_err(|e| FerrumError::model(format!("load vocoder weights: {e}")))?
        };
        let vocoder_config = VocoderConfig::default();
        let vocoder = Qwen3TTSVocoder::load(&vocoder_config, vocoder_vb.clone())?;

        // Load Speech Tokenizer Encoder on CPU — Metal float32 accumulation order
        // causes transformer output divergence that amplifies through RVQ codebook lookup.
        // CPU is exact and encoder only runs once per reference audio.
        let speech_tokenizer_encoder = if vocoder_dir.join("config.json").exists() {
            let cpu_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&vocoder_weights, dtype, &CandleDevice::Cpu)
                    .map_err(|e| FerrumError::model(format!("load encoder cpu: {e}")))?
            };
            SpeechTokenizerEncoder::load(cpu_vb.pp("encoder"), CandleDevice::Cpu)
                .map_err(|e| {
                    tracing::warn!("Speech tokenizer encoder not available: {e}");
                    e
                })
                .ok()
        } else {
            None
        };

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
            model_path, config.hidden_size, config.num_hidden_layers, config.num_code_groups,
        );

        Ok(Self {
            talker,
            sub_talker,
            vocoder,
            text_tokenizer,
            config,
            info,
            speaker_encoder,
            speech_tokenizer_encoder,
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
        let mut hidden = self.talker.forward_step(&prefill_embeds)?;

        // Get logits from last position
        let current_logits = self.talker.logits(
            &hidden
                .narrow(1, hidden.dim(1).unwrap() - 1, 1)
                .map_err(|e| FerrumError::model(format!("narrow: {e}")))?,
        )?;

        // 4. Autoregressive decode loop: generate codec token 0 per step
        let mut all_codec_tokens: Vec<Vec<u32>> = Vec::new();
        let mut current_logits = current_logits;

        for step in 0..MAX_CODEC_TOKENS {
            // Sample next codec token from logits
            let logits_vec = logits_to_vec(&current_logits)?;
            let next_token = sample_token(&logits_vec, TEMPERATURE, TOP_K, REPETITION_PENALTY);

            // Check for EOS
            if next_token == codec_eos {
                info!("TTS: codec EOS at step {}", step);
                break;
            }

            // Get last hidden state from talker for SubTalker
            let last_hidden = hidden
                .narrow(1, hidden.dim(1).unwrap() - 1, 1)
                .map_err(|e| FerrumError::model(format!("last_hidden: {e}")))?;

            // Embed first codec token
            let token_tensor = Tensor::new(&[next_token], &device)
                .map_err(|e| FerrumError::model(format!("token tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
            let first_codec_embed = self.talker.embed_codec(&token_tensor)?;

            // SubTalker: predict remaining codec tokens 1..num_code_groups-1
            let extra_codes =
                self.sub_talker
                    .predict(&last_hidden, &first_codec_embed, TEMPERATURE, TOP_K)?;

            let mut frame_codes = vec![next_token];
            frame_codes.extend_from_slice(&extra_codes);
            all_codec_tokens.push(frame_codes);

            // Build combined embedding for next talker step:
            // sum of all codec embeddings (token 0 from main talker + tokens 1-15 from sub-talker)
            let mut combined_embed = first_codec_embed.clone();
            for (i, &code) in extra_codes.iter().enumerate() {
                let code_t = Tensor::new(&[code], &device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| FerrumError::model(format!("code_t: {e}")))?;
                let sub_embed = code_t
                    .apply(&self.sub_talker.codec_embeddings[i])
                    .map_err(|e| FerrumError::model(format!("sub_embed: {e}")))?;
                combined_embed = (combined_embed + sub_embed)
                    .map_err(|e| FerrumError::model(format!("add embed: {e}")))?;
            }

            // Forward combined embedding through talker
            hidden = self.talker.forward_step(&combined_embed)?;
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

        let codes_tensor = Tensor::new(&flat_codes[..], &device)
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

    /// Synthesize speech with voice cloning from a reference audio.
    ///
    /// Uses ICL (in-context learning) prompting: the reference audio is
    /// encoded to codec tokens and prepended to the generation prompt,
    /// along with a speaker embedding extracted via ECAPA-TDNN.
    ///
    /// Returns PCM samples at 24kHz as Vec<f32>.
    pub fn synthesize_voice_clone(
        &mut self,
        text: &str,
        language: &str,
        ref_audio_path: &str,
        ref_text: &str,
    ) -> Result<Vec<f32>> {
        let device = self.talker.device().clone();

        // Step 1: Load and process reference audio at 24kHz
        let ref_pcm = crate::audio_processor::load_audio_at_rate(ref_audio_path, 24000)?;
        info!(
            "TTS voice clone: loaded ref audio {} samples ({:.2}s)",
            ref_pcm.len(),
            ref_pcm.len() as f64 / 24000.0
        );

        let t0 = std::time::Instant::now();
        // Step 2: Extract speaker embedding via ECAPA-TDNN
        let speaker_encoder = self
            .speaker_encoder
            .as_ref()
            .ok_or_else(|| FerrumError::model("speaker encoder not loaded"))?;
        let mel = mel_spectrogram_speaker_encoder(&ref_pcm);
        let n_mel_frames = mel.len() / 128;
        let mel_tensor = Tensor::from_vec(mel, (1, n_mel_frames, 128), &device)
            .map_err(|e| FerrumError::model(format!("mel tensor: {e}")))?;
        let spk_embed = speaker_encoder.forward(&mel_tensor)?;
        // spk_embed shape: [1024] -> reshape to [1, 1, 1024]
        let spk_embed = spk_embed
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("spk unsqueeze(0): {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("spk unsqueeze(0) 2: {e}")))?;

        info!("Step 2 (speaker embed): {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        let t1 = std::time::Instant::now();
        // Step 3: Encode reference audio to codec tokens (ICL)
        let speech_enc = self
            .speech_tokenizer_encoder
            .as_ref()
            .ok_or_else(|| FerrumError::model("speech tokenizer encoder not loaded"))?;
        // Allow pre-computed codec tokens for debugging (FERRUM_REF_CODES=/path/to/codes.bin)
        let ref_codes = if let Ok(path) = std::env::var("FERRUM_REF_CODES") {
            let data = std::fs::read(&path)
                .map_err(|e| FerrumError::model(format!("read ref codes: {e}")))?;
            let u32s: Vec<u32> = data.chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let ncb = self.config.num_code_groups;
            let nframes = u32s.len() / ncb;
            info!("Loaded pre-computed ref codes: {} frames from {}", nframes, path);
            u32s.chunks(ncb).map(|c| c.to_vec()).collect()
        } else {
            speech_enc.encode(&ref_pcm)?
        };
        let ref_frames = ref_codes.len();
        info!(
            "TTS voice clone: ref_frames={}, spk_embed loaded",
            ref_frames
        );
        // Debug: dump first 5 codec frames for comparison with Python
        for i in 0..ref_frames.min(5) {
            info!("  rust codec frame {}: {:?}", i, &ref_codes[i]);
        }

        info!("Step 3 (speech tokenizer): {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);
        let t2 = std::time::Instant::now();
        // Step 4: Tokenize target text with chat template
        let chat_text = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let encoding = self
            .text_tokenizer
            .encode(chat_text.as_str(), false)
            .map_err(|e| FerrumError::model(format!("tokenize: {e}")))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        // role = input_ids[..3], text_content = input_ids[3..input_ids.len()-5]
        let role_ids = &input_ids[..3];
        let text_content_ids = &input_ids[3..input_ids.len().saturating_sub(5)];

        // Tokenize ref_text
        let ref_chat_text = format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n");
        let ref_encoding = self
            .text_tokenizer
            .encode(ref_chat_text.as_str(), false)
            .map_err(|e| FerrumError::model(format!("tokenize ref: {e}")))?;
        let ref_ids: Vec<u32> = ref_encoding.get_ids().to_vec();
        // ref text content: ref_ids[3..ref_ids.len()-2]
        let ref_text_ids = &ref_ids[3..ref_ids.len().saturating_sub(2)];

        // Step 5: Build prefill prompt (dual-stream text+codec summing)
        self.talker.reset();

        let tts_bos = self.config.tts_bos_token_id;
        let tts_eos = self.config.tts_eos_token_id;
        let tts_pad = self.config.tts_pad_token_id;
        let codec_bos = self.config.codec_bos_id;
        let codec_eos = self.config.codec_eos_token_id;
        let codec_pad = self.config.codec_pad_id;

        // Helper: embed codec and text tokens
        let embed_codec_ids = |ids: &[u32]| -> Result<Tensor> {
            let t = Tensor::new(ids, &device)
                .map_err(|e| FerrumError::model(format!("codec tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("codec unsqueeze: {e}")))?;
            self.talker.embed_codec(&t)
        };
        let embed_text_ids = |ids: &[u32]| -> Result<Tensor> {
            let t = Tensor::new(ids, &device)
                .map_err(|e| FerrumError::model(format!("text tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("text unsqueeze: {e}")))?;
            self.talker.embed_text(&t)
        };

        // Get tts special embeddings
        let tts_special = embed_text_ids(&[tts_bos, tts_eos, tts_pad])?;
        let tts_bos_embed = tts_special
            .narrow(1, 0, 1)
            .map_err(|e| FerrumError::model(format!("tts_bos narrow: {e}")))?;
        let tts_eos_embed = tts_special
            .narrow(1, 1, 1)
            .map_err(|e| FerrumError::model(format!("tts_eos narrow: {e}")))?;
        let tts_pad_embed = tts_special
            .narrow(1, 2, 1)
            .map_err(|e| FerrumError::model(format!("tts_pad narrow: {e}")))?;

        // Resolve language_id — "auto" defaults to "chinese"
        let resolved_lang = if language.eq_ignore_ascii_case("auto") {
            "chinese"
        } else {
            language
        };
        let language_id = self.config.codec_language_id.get(&resolved_lang.to_lowercase());

        // Codec prefix: [think, think_bos, lang, think_eos] or [nothink, think_bos, think_eos]
        let codec_prefix_ids = if let Some(&lang_id) = language_id {
            vec![
                self.config.codec_think_id,
                self.config.codec_think_bos_id,
                lang_id,
                self.config.codec_think_eos_id,
            ]
        } else {
            vec![
                self.config.codec_nothink_id,
                self.config.codec_think_bos_id,
                self.config.codec_think_eos_id,
            ]
        };
        let codec_prefix_embed = embed_codec_ids(&codec_prefix_ids)?;

        // Codec suffix: [pad, bos]
        let codec_suffix_embed = embed_codec_ids(&[codec_pad, codec_bos])?;

        // Speaker embed inserted between prefix and suffix
        let codec_input = Tensor::cat(&[&codec_prefix_embed, &spk_embed, &codec_suffix_embed], 1)
            .map_err(|e| FerrumError::model(format!("codec_input cat: {e}")))?;
        let codec_len = codec_input
            .dim(1)
            .map_err(|e| FerrumError::model(format!("codec_len dim: {e}")))?;

        // Role embedding
        let role_embed = embed_text_ids(role_ids)?;

        // Text-codec prefix: (codec_len - 2) pads + tts_bos, summed with codec[:-1]
        let n_pads = codec_len - 2;
        let mut text_prefix_parts = Vec::new();
        for _ in 0..n_pads {
            text_prefix_parts.push(tts_pad_embed.clone());
        }
        text_prefix_parts.push(tts_bos_embed.clone());
        let text_prefix_refs: Vec<&Tensor> = text_prefix_parts.iter().collect();
        let text_prefix = Tensor::cat(&text_prefix_refs, 1)
            .map_err(|e| FerrumError::model(format!("text_prefix cat: {e}")))?;
        let codec_prefix_part = codec_input
            .narrow(1, 0, codec_len - 1)
            .map_err(|e| FerrumError::model(format!("codec prefix narrow: {e}")))?;
        let text_codec_prefix = (&text_prefix + &codec_prefix_part)
            .map_err(|e| FerrumError::model(format!("text+codec prefix sum: {e}")))?;

        let mut talker_input = Tensor::cat(&[&role_embed, &text_codec_prefix], 1)
            .map_err(|e| FerrumError::model(format!("talker_input cat: {e}")))?;

        // ICL prompt: embed ref text + target text, ref codec, and sum them
        // Text stream: text_projection(text_embed([ref_text_content, text_content]))
        let all_text_ids: Vec<u32> = ref_text_ids
            .iter()
            .chain(text_content_ids.iter())
            .copied()
            .collect();
        let text_embed = embed_text_ids(&all_text_ids)?;
        let text_embed_with_eos = Tensor::cat(&[&text_embed, &tts_eos_embed], 1)
            .map_err(|e| FerrumError::model(format!("text+eos cat: {e}")))?;
        let text_len = text_embed_with_eos
            .dim(1)
            .map_err(|e| FerrumError::model(format!("text_len dim: {e}")))?;

        // Codec stream: codec_bos + sum of all 16 codebook embeddings for ref codes
        let mut codec_frame_embeds = Vec::new();
        for frame in &ref_codes {
            // First codebook: main codec embedding
            let first_embed = embed_codec_ids(&[frame[0]])?;
            let mut frame_embed = first_embed;
            // Remaining codebooks: sub-talker embeddings
            for (i, &code) in frame[1..self.config.num_code_groups].iter().enumerate() {
                let code_t = Tensor::new(&[code], &device)
                    .map_err(|e| FerrumError::model(format!("ref code_t: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| FerrumError::model(format!("ref code unsqueeze: {e}")))?;
                let sub_embed = code_t
                    .apply(&self.sub_talker.codec_embeddings[i])
                    .map_err(|e| FerrumError::model(format!("ref sub_embed: {e}")))?;
                frame_embed = (frame_embed + sub_embed)
                    .map_err(|e| FerrumError::model(format!("ref frame_embed add: {e}")))?;
            }
            codec_frame_embeds.push(frame_embed);
        }
        let codec_frames_refs: Vec<&Tensor> = codec_frame_embeds.iter().collect();
        let codec_frames_cat = Tensor::cat(&codec_frames_refs, 1)
            .map_err(|e| FerrumError::model(format!("codec_frames cat: {e}")))?;

        // Prepend codec_bos to codec frames: [codec_bos_embed, codec_frames]
        let codec_bos_for_icl = embed_codec_ids(&[codec_bos])?;
        let icl_codec = Tensor::cat(&[&codec_bos_for_icl, &codec_frames_cat], 1)
            .map_err(|e| FerrumError::model(format!("icl_codec cat: {e}")))?;
        let codec_icl_len = icl_codec
            .dim(1)
            .map_err(|e| FerrumError::model(format!("codec_icl_len dim: {e}")))?;

        // Merge text and codec (element-wise sum, handling different lengths)
        let trailing_text: Option<Tensor>;
        if text_len > codec_icl_len {
            // text is longer: sum first codec_len positions, trailing text is separate
            let text_part = text_embed_with_eos
                .narrow(1, 0, codec_icl_len)
                .map_err(|e| FerrumError::model(format!("text_part narrow: {e}")))?;
            let summed = (&text_part + &icl_codec)
                .map_err(|e| FerrumError::model(format!("text+codec sum: {e}")))?;
            let trailing = text_embed_with_eos
                .narrow(1, codec_icl_len, text_len - codec_icl_len)
                .map_err(|e| FerrumError::model(format!("trailing narrow: {e}")))?;
            talker_input = Tensor::cat(&[&talker_input, &summed], 1)
                .map_err(|e| FerrumError::model(format!("talker+summed cat: {e}")))?;
            trailing_text = Some(trailing);
        } else {
            // codec is longer or equal: pad text with tts_pad
            let mut text_padded_parts = vec![text_embed_with_eos.clone()];
            for _ in 0..(codec_icl_len - text_len) {
                text_padded_parts.push(tts_pad_embed.clone());
            }
            let padded_refs: Vec<&Tensor> = text_padded_parts.iter().collect();
            let text_padded = Tensor::cat(&padded_refs, 1)
                .map_err(|e| FerrumError::model(format!("text_padded cat: {e}")))?;
            let summed = (&text_padded + &icl_codec)
                .map_err(|e| FerrumError::model(format!("padded+codec sum: {e}")))?;
            talker_input = Tensor::cat(&[&talker_input, &summed], 1)
                .map_err(|e| FerrumError::model(format!("talker+summed cat: {e}")))?;
            trailing_text = None;
        }

        // Add trailing text if any (in streaming mode, trailing text is fed during decode)
        // Note: Python streaming ICL does NOT add a decode_start position.
        // The first token is generated from the last ICL position directly.
        if let Some(trailing) = &trailing_text {
            talker_input = Tensor::cat(&[&talker_input, trailing], 1)
                .map_err(|e| FerrumError::model(format!("talker+trailing cat: {e}")))?;
        }

        let prefill_len = talker_input
            .dim(1)
            .map_err(|e| FerrumError::model(format!("prefill dim: {e}")))?;
        info!("TTS voice clone: prefill seq_len={}", prefill_len);

        // Debug: dump key positions for comparison with Python
        for pos in [0usize, 8, 9, 10, 20, 40, 60, 70, 71, 72] {
            if pos < prefill_len {
                if let Ok(vals) = talker_input
                    .narrow(0, 0, 1)
                    .and_then(|t| t.narrow(1, pos, 1))
                    .and_then(|t| t.narrow(2, 0, 5))
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                {
                    info!("  prefill pos {}: {:?}", pos, vals);
                }
            }
        }

        info!("Steps 4-5 (tokenize+prompt): {:.1}ms", t2.elapsed().as_secs_f64() * 1000.0);
        let t3 = std::time::Instant::now();
        // Step 6: Prefill and decode
        let mut hidden = self.talker.forward_step(&talker_input)?;
        info!("Prefill ({} tokens, {} layers): {:.1}ms", prefill_len, self.config.num_hidden_layers, t3.elapsed().as_secs_f64() * 1000.0);
        let hidden_len = hidden
            .dim(1)
            .map_err(|e| FerrumError::model(format!("hidden dim: {e}")))?;
        let current_logits = self.talker.logits(
            &hidden
                .narrow(1, hidden_len - 1, 1)
                .map_err(|e| FerrumError::model(format!("narrow last: {e}")))?,
        )?;

        // Decode loop
        let mut all_codec_tokens: Vec<Vec<u32>> = Vec::new();
        let mut current_logits = current_logits;

        // Suppress special tokens [vocab_size-1024, vocab_size) except EOS
        let suppress_start = self.config.vocab_size.saturating_sub(1024);
        let suppress_end = self.config.vocab_size;

        for step in 0..MAX_CODEC_TOKENS {
            let mut logits_vec = logits_to_vec(&current_logits)?;
            // Suppress special tokens
            for i in suppress_start..suppress_end.min(logits_vec.len()) {
                if i as u32 != codec_eos {
                    logits_vec[i] = f32::NEG_INFINITY;
                }
            }
            let next_token = sample_token(&logits_vec, TEMPERATURE, TOP_K, REPETITION_PENALTY);

            if step < 3 {
                // Dump top-5 logits for comparison
                let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
                info!("  decode step {}: token={}, top5={:?}", step, next_token, top5);
            }

            if next_token == codec_eos {
                info!("TTS: codec EOS at step {}", step);
                break;
            }

            let cur_hidden_len = hidden
                .dim(1)
                .map_err(|e| FerrumError::model(format!("hidden dim: {e}")))?;
            let last_hidden = hidden
                .narrow(1, cur_hidden_len - 1, 1)
                .map_err(|e| FerrumError::model(format!("last_hidden: {e}")))?;

            if step < 2 {
                if let Ok(vals) = last_hidden.narrow(0, 0, 1)
                    .and_then(|t| t.narrow(1, 0, 1))
                    .and_then(|t| t.narrow(2, 0, 5))
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>()) {
                    info!("  step {} past_hidden first 5: {:?}", step, vals);
                }
            }

            let token_tensor = Tensor::new(&[next_token], &device)
                .map_err(|e| FerrumError::model(format!("token tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
            let first_codec_embed = self.talker.embed_codec(&token_tensor)?;

            let extra_codes =
                self.sub_talker
                    .predict(&last_hidden, &first_codec_embed, TEMPERATURE, TOP_K)?;

            let mut frame_codes = vec![next_token];
            frame_codes.extend_from_slice(&extra_codes);
            all_codec_tokens.push(frame_codes);

            // Sum all codebook embeddings
            let mut combined_embed = first_codec_embed.clone();
            for (i, &code) in extra_codes.iter().enumerate() {
                let code_t = Tensor::new(&[code], &device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| FerrumError::model(format!("code_t: {e}")))?;
                let sub_embed = code_t
                    .apply(&self.sub_talker.codec_embeddings[i])
                    .map_err(|e| FerrumError::model(format!("sub_embed: {e}")))?;
                combined_embed = (combined_embed + sub_embed)
                    .map_err(|e| FerrumError::model(format!("add embed: {e}")))?;
            }

            if step < 3 {
                let mut all_ids = vec![next_token];
                all_ids.extend_from_slice(&extra_codes);
                info!("  step {} codec_ids: {:?}", step, all_ids);
            }

            // Add tts_pad to codec embedding (dual-stream sum)
            combined_embed = (combined_embed + &tts_pad_embed)
                .map_err(|e| FerrumError::model(format!("add tts_pad: {e}")))?;

            hidden = self.talker.forward_step(&combined_embed)?;
            current_logits = self.talker.logits(&hidden)?;
        }

        if all_codec_tokens.is_empty() {
            return Err(FerrumError::model("no codec tokens generated"));
        }
        info!(
            "TTS voice clone: generated {} codec frames",
            all_codec_tokens.len()
        );

        // Step 7: Prepend ref codes and decode with vocoder
        let mut all_codes_with_ref = ref_codes.clone();
        all_codes_with_ref.extend_from_slice(&all_codec_tokens);

        let num_frames = all_codes_with_ref.len();
        let num_groups = self.config.num_code_groups;

        // Build codec tensor [1, num_groups, T]
        let mut flat_codes: Vec<u32> = vec![0; num_groups * num_frames];
        for (t, frame) in all_codes_with_ref.iter().enumerate() {
            for (g, &code) in frame.iter().take(num_groups).enumerate() {
                flat_codes[g * num_frames + t] = code;
            }
        }

        // Clamp to valid range
        let codebook_size = 2048u32;
        for code in &mut flat_codes {
            if *code >= codebook_size {
                *code = 0;
            }
        }

        let codes_tensor = Tensor::new(&flat_codes[..], &device)
            .map_err(|e| FerrumError::model(format!("codes tensor: {e}")))?
            .reshape((1, num_groups, num_frames))
            .map_err(|e| FerrumError::model(format!("reshape codes: {e}")))?;

        let waveform = self.vocoder.decode(&codes_tensor)?;

        let samples: Vec<f32> = waveform
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze batch: {e}")))?
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze channel: {e}")))?
            .to_vec1()
            .map_err(|e| FerrumError::model(format!("to_vec1: {e}")))?;

        // Trim reference portion
        let ref_ratio = ref_frames as f64 / num_frames as f64;
        let cut = (ref_ratio * samples.len() as f64) as usize;
        let output_samples = samples[cut..].to_vec();

        info!(
            "TTS voice clone: waveform {} samples ({:.2}s), trimmed ref {} samples",
            output_samples.len(),
            output_samples.len() as f64 / SAMPLE_RATE as f64,
            cut,
        );

        Ok(output_samples)
    }
}

// ── Utility functions ───────────────────────────────────────────────────

/// Find safetensor files matching a prefix in a directory.
fn find_safetensor_files(dir: &std::path::Path, prefix: &str) -> Result<Vec<std::path::PathBuf>> {
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
pub fn sample_token(
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
    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
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
