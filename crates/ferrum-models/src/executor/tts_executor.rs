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
use crate::architectures::qwen3_tts_backbone::TalkerBackboneBackend;
use crate::architectures::qwen3_tts_vocoder::{Qwen3TTSVocoder, VocoderConfig};
use crate::architectures::speaker_encoder::{mel_spectrogram_speaker_encoder, SpeakerEncoder};
use crate::architectures::speech_tokenizer_encoder::SpeechTokenizerEncoder;
use ferrum_quantization::NativeSafetensorsLoader;

/// Install `Qwen3TtsTalker`/`SubTalker` Backend<CudaBackend> overrides so
/// the transformer stack runs via ferrum-kernels cuBLAS + CUDA kernels
/// instead of the broken fused-on-Linux CPU fallback.
#[cfg(feature = "cuda")]
fn install_cuda_backend_overrides(
    cfg: &TalkerConfig,
    model_dir: &std::path::Path,
    talker: &mut Qwen3TTSTalker,
    sub_talker: &mut SubTalker,
) -> Result<()> {
    use ferrum_kernels::backend::cuda::CudaBackend;
    let loader: NativeSafetensorsLoader<CudaBackend> = NativeSafetensorsLoader::open(model_dir)?;
    let talker_bb = TalkerBackboneBackend::<CudaBackend>::new(cfg, &loader)?;
    talker.set_backend_override(Box::new(talker_bb));
    let sub_bb = TalkerBackboneBackend::<CudaBackend>::new_code_predictor(cfg, &loader)?;
    sub_talker.set_backend_override(Box::new(sub_bb));
    Ok(())
}

// ── Constants ────────────────────────────────────────────────────────────

const SAMPLE_RATE: usize = 24000;
const MAX_CODEC_TOKENS: usize = 2000;

/// Sampling parameters for codec token generation.
/// FERRUM_TTS_TEMP env var overrides (0.0 = greedy, 0.9 = default sampling)
const TEMPERATURE: f32 = 0.9;
const TOP_K: usize = 50;
const REPETITION_PENALTY: f32 = 1.05;

fn tts_temperature() -> f32 {
    std::env::var("FERRUM_TTS_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TEMPERATURE)
}

fn st_temperature() -> f32 {
    std::env::var("FERRUM_ST_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(tts_temperature())
}

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
        let mut talker = Qwen3TTSTalker::load(&config, talker_vb.clone(), device.clone())?;

        // Load SubTalker (code predictor) from same weights file
        let mut sub_talker = SubTalker::load(&config, talker_vb.clone(), device.clone())?;

        // Load Speaker Encoder (for voice cloning, base models only)
        let spk_enc_dim = config_json
            .get("speaker_encoder_config")
            .and_then(|c| c.get("enc_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize;
        let speaker_encoder =
            SpeakerEncoder::load_with_dim(talker_vb.pp("speaker_encoder"), spk_enc_dim)
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

        // Install a Backend<B>-backed transformer for the Talker/SubTalker
        // when running on CUDA. The legacy `ferrum_attention::FusedTransformer`
        // CUDA module is a stub and its Linux CPU fallback uses naive fp64
        // matmul — the CUDA-only voice-clone regression traces back to that
        // path accumulating precision drift through the 20-layer decoder.
        // Routing through LlamaFamilyModel<CudaBackend> gives us cuBLAS +
        // ferrum-kernels for the transformer stack while keeping candle
        // embeddings / projection / codec_head unchanged.
        #[cfg(feature = "cuda")]
        if matches!(&device, CandleDevice::Cuda(_)) {
            match install_cuda_backend_overrides(&config, dir, &mut talker, &mut sub_talker) {
                Ok(()) => {
                    tracing::info!(
                        "TtsModelExecutor: Backend<CudaBackend> installed for Talker + SubTalker"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "TtsModelExecutor: Backend<CudaBackend> install failed ({e}); \
                         falling back to candle/fused path (CUDA voice-clone may produce garbage)"
                    );
                }
            }
        }

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
    ///
    /// Prompt structure (matches Python/qwen3-tts-rs):
    ///   Prefill: [role_prefix(3)] + [tts_text_prefix(6) + codec_prefix(6)] + [first_text + codec_bos]
    ///   Trailing: text_projection(remaining_text + tts_eos) — added per decode step
    pub fn synthesize(&mut self, text: &str, language: &str) -> Result<Vec<f32>> {
        self.talker.reset();

        let device = self.talker.device().clone();

        // 1. Tokenize text (raw content only, no chat template)
        let encoding = self
            .text_tokenizer
            .encode(text, false)
            .map_err(|e| FerrumError::model(format!("tokenize: {e}")))?;
        let content_ids: Vec<u32> = encoding.get_ids().to_vec();

        if content_ids.is_empty() {
            return Err(FerrumError::model("empty text after tokenization"));
        }

        info!("TTS: content tokens = {}", content_ids.len());

        let codec_eos = self.config.codec_eos_token_id;
        let tts_pad = self.config.tts_pad_token_id;
        let tts_bos = self.config.tts_bos_token_id;
        let tts_eos = self.config.tts_eos_token_id;

        // Helper: embed text token IDs through text_embedding + text_projection
        let embed_text_ids = |ids: &[u32]| -> Result<Tensor> {
            let t = Tensor::new(ids, &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| FerrumError::model(format!("text ids: {e}")))?;
            self.talker.embed_text(&t)
        };
        let embed_codec_ids = |ids: &[u32]| -> Result<Tensor> {
            let t = Tensor::new(ids, &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| FerrumError::model(format!("codec ids: {e}")))?;
            self.talker.embed_codec(&t)
        };

        // 2. Build prefill (matching qwen3-tts-rs prefill_custom_voice)

        // Role prefix: text_projection([<|im_start|>, assistant, \n])
        // These are fixed special token IDs from the tokenizer
        let im_start_id = 151644u32; // <|im_start|>
        let assistant_id = 77091u32; // "assistant"
        let newline_id = 198u32; // "\n"
        let role_prefix_ids = [im_start_id, assistant_id, newline_id];

        info!(
            "TTS: role_prefix={} content={} tokens",
            role_prefix_ids.len(),
            content_ids.len()
        );

        // Role prefix embedding (text_projection)
        let role_embed = embed_text_ids(&role_prefix_ids)?;

        // Codec prefix: [think, think_bos, lang, think_eos, speaker, pad]
        let resolved_lang = if language == "auto" {
            "chinese"
        } else {
            language
        };
        let language_id = self
            .config
            .codec_language_id
            .get(&resolved_lang.to_lowercase());
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
        // Codec sequence: [think, think_bos, lang, think_eos, SPEAKER, pad, bos]
        // Speaker: use Vivian (3065) for Chinese, Ryan (3061) for English
        let speaker_token = if resolved_lang == "chinese" {
            3065u32
        } else {
            3061u32
        };
        let codec_full = {
            let mut v = codec_prefix_ids.clone();
            v.push(speaker_token);
            v.push(self.config.codec_pad_id);
            v.push(self.config.codec_bos_id);
            v
        };
        let codec_embed = embed_codec_ids(&codec_full)?;

        // tts_text_prefix: [tts_pad * (codec_len-1), tts_bos]
        let n_codec = codec_full.len();
        let mut tts_prefix_ids = vec![tts_pad; n_codec - 1];
        tts_prefix_ids.push(tts_bos);
        let tts_prefix_embed = embed_text_ids(&tts_prefix_ids)?;

        // Sum: tts_prefix + codec_prefix[:-1]
        let codec_first = codec_embed
            .narrow(1, 0, n_codec - 1)
            .map_err(|e| FerrumError::model(format!("codec narrow: {e}")))?;
        // Actually we need codec_first to have same length as tts_prefix
        // tts_prefix has n_codec elements, codec_first has n_codec-1
        // Let me re-read the reference... codec_embed has 6 tokens [think..pad,bos], tts_prefix has 6 [pad*5,bos]
        // They sum first 6 codec with first 6 tts_prefix, then codec_bos is separate
        // codec_full has [think, think_bos, lang, think_eos, speaker, pad, bos] = 7 tokens
        // tts_prefix: [pad*5, bos] overlaid on codec[0:6] (first 6, excluding last bos)
        // Then: first_text + codec_bos (last token) summed
        let n_prefix = n_codec - 1; // 6: everything except codec_bos
        let codec_prefix_part = codec_embed
            .narrow(1, 0, n_prefix)
            .map_err(|e| FerrumError::model(format!("codec narrow: {e}")))?;

        // tts text prefix: [pad * (n_prefix-1), bos]
        let mut tts_text_prefix_ids = vec![tts_pad; n_prefix - 1];
        tts_text_prefix_ids.push(tts_bos);
        let tts_text_embed = embed_text_ids(&tts_text_prefix_ids)?;

        let codec_hidden = (&tts_text_embed + &codec_prefix_part)
            .map_err(|e| FerrumError::model(format!("prefix sum: {e}")))?;

        // codec_bos is the last element of codec_full
        let codec_bos_embed = codec_embed
            .narrow(1, n_prefix, 1)
            .map_err(|e| FerrumError::model(format!("codec bos: {e}")))?;

        // First text token + codec_bos (summed)
        let first_text_combined = if !content_ids.is_empty() {
            let first_text_embed = embed_text_ids(&content_ids[..1])?;
            (&first_text_embed + &codec_bos_embed)
                .map_err(|e| FerrumError::model(format!("first text+bos: {e}")))?
        } else {
            codec_bos_embed.clone()
        };

        // Full prefill: [role_prefix, codec_hidden, first_text+codec_bos]
        let prefill_embeds = Tensor::cat(&[&role_embed, &codec_hidden, &first_text_combined], 1)
            .map_err(|e| FerrumError::model(format!("prefill cat: {e}")))?;

        let plen = prefill_embeds.dim(1).unwrap_or(0);
        info!("TTS: prefill_len = {}", plen);
        // Dump prefill input for comparison with reference
        if let Ok(v) = prefill_embeds
            .narrow(0, 0, 1)
            .and_then(|t| t.narrow(1, 0, 1))
            .and_then(|t| t.narrow(2, 0, 5))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
        {
            info!("  prefill_input pos0[:5] = {:?}", v);
        }
        if plen > 0 {
            if let Ok(v) = prefill_embeds
                .narrow(0, 0, 1)
                .and_then(|t| t.narrow(1, plen - 1, 1))
                .and_then(|t| t.narrow(2, 0, 5))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
            {
                info!("  prefill_input pos-1[:5] = {:?}", v);
            }
        }

        // 3. Build trailing text: text_projection(remaining_content + tts_eos)
        let mut trailing_ids: Vec<u32> = if content_ids.len() > 1 {
            content_ids[1..].to_vec()
        } else {
            Vec::new()
        };
        trailing_ids.push(tts_eos);
        let trailing_text_embeds = embed_text_ids(&trailing_ids)?;
        let trailing_text_len = trailing_text_embeds
            .dim(1)
            .map_err(|e| FerrumError::model(format!("trailing dim: {e}")))?;
        let tts_pad_embed = embed_text_ids(&[tts_pad])?;

        info!("TTS: trailing_text_len = {}", trailing_text_len);

        // 4. Prefill: forward through transformer
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

        // Token suppression: mask [vocab_size-1024, vocab_size) except EOS
        let suppress_start = self.config.vocab_size.saturating_sub(1024);
        let suppress_end = self.config.vocab_size;
        let mut generated_tokens: Vec<u32> = Vec::new();

        for step in 0..MAX_CODEC_TOKENS {
            // Sample next codec token with suppression + repetition penalty
            let mut logits_vec = logits_to_vec(&current_logits)?;
            // Suppress special tokens
            for i in suppress_start..suppress_end.min(logits_vec.len()) {
                if i as u32 != codec_eos {
                    logits_vec[i] = f32::NEG_INFINITY;
                }
            }
            // Repetition penalty (matching Python's repetition_penalty=1.05)
            for &prev_tok in &generated_tokens {
                let idx = prev_tok as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= REPETITION_PENALTY;
                    } else {
                        logits_vec[idx] *= REPETITION_PENALTY;
                    }
                }
            }
            let next_token =
                sample_token(&logits_vec, tts_temperature(), TOP_K, REPETITION_PENALTY);

            if step < 10 {
                // Find argmax (greedy token)
                let argmax_tok = logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, v)| (i, *v))
                    .unwrap_or((0, 0.0));
                info!(
                    "TOKEN step={} sampled={} argmax=({}, {:.2})",
                    step, next_token, argmax_tok.0, argmax_tok.1
                );
            }

            generated_tokens.push(next_token);

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
            let st_t0 = std::time::Instant::now();
            let extra_codes = self.sub_talker.predict(
                &last_hidden,
                &first_codec_embed,
                st_temperature(),
                TOP_K,
            )?;
            if step == 0 {
                info!(
                    "  SubTalker: {:.1}ms",
                    st_t0.elapsed().as_secs_f64() * 1000.0
                );
            }

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

            if step == 0 {
                if let Ok(v) = combined_embed
                    .flatten_all()
                    .and_then(|t| t.narrow(0, 0, 5))
                    .and_then(|t| t.to_vec1::<f32>())
                {
                    info!("STEP0 codec_sum[:5] = {:?} (before trailing)", v);
                }
            }
            // Add trailing text embedding (guides generation toward target text)
            // Python: inputs_embeds = codec_sum + trailing_text_hidden[:, gen_step]
            // trailing_text = text_projection(remaining_text) + tts_eos
            // For basic TTS, trailing covers all text tokens after the first
            if step < trailing_text_len {
                let trail = trailing_text_embeds
                    .narrow(1, step, 1)
                    .map_err(|e| FerrumError::model(format!("trailing narrow: {e}")))?;
                combined_embed = (combined_embed + trail)
                    .map_err(|e| FerrumError::model(format!("add trailing: {e}")))?;
            } else {
                combined_embed = (combined_embed + &tts_pad_embed)
                    .map_err(|e| FerrumError::model(format!("add tts_pad: {e}")))?;
            }

            // Debug: dump step 0 components
            if step == 0 {
                if let Ok(v) = first_codec_embed
                    .flatten_all()
                    .and_then(|t| t.narrow(0, 0, 5))
                    .and_then(|t| t.to_vec1::<f32>())
                {
                    info!("STEP0 semantic[:5] = {:?}", v);
                }
                if let Ok(v) = combined_embed
                    .flatten_all()
                    .and_then(|t| t.narrow(0, 0, 5))
                    .and_then(|t| t.to_vec1::<f32>())
                {
                    info!("STEP0 combined[:5] = {:?}", v);
                }
            }
            // Forward combined embedding through talker
            let tk_t0 = std::time::Instant::now();
            hidden = self.talker.forward_step(&combined_embed)?;
            current_logits = self.talker.logits(&hidden)?;
            if step == 0 {
                info!(
                    "  Talker step: {:.1}ms",
                    tk_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
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
        let ref_pcm = if let Ok(path) = std::env::var("FERRUM_REF_PCM") {
            let data = std::fs::read(&path)
                .map_err(|e| FerrumError::model(format!("read ref pcm: {e}")))?;
            let pcm: Vec<f32> = data
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            info!("Loaded ref PCM override: {} samples", pcm.len());
            pcm
        } else {
            crate::audio_processor::load_audio_at_rate(ref_audio_path, 24000)?
        };
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
        // mel_spectrogram_speaker_encoder returns [T, 128] row-major
        // forward() internally transposes [1, T, 128] → [1, 128, T]
        let mel_tensor = Tensor::from_vec(mel, (1, n_mel_frames, 128), &device)
            .map_err(|e| FerrumError::model(format!("mel tensor: {e}")))?;
        let spk_embed = speaker_encoder.forward(&mel_tensor)?;
        info!(
            "Step 2 (speaker embed): {:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        // spk_embed shape: [enc_dim] -> reshape to [1, 1, hidden_size]
        let spk_embed = spk_embed
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("spk unsqueeze(0): {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("spk unsqueeze(0) 2: {e}")))?;

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
            let u32s: Vec<u32> = data
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let ncb = self.config.num_code_groups;
            let nframes = u32s.len() / ncb;
            info!(
                "Loaded pre-computed ref codes: {} frames from {}",
                nframes, path
            );
            u32s.chunks(ncb).map(|c| c.to_vec()).collect()
        } else {
            let codes = speech_enc.encode(&ref_pcm)?;
            info!(
                "Step 3 (speech tokenizer): {:.1}ms",
                t1.elapsed().as_secs_f64() * 1000.0
            );
            codes
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

        info!(
            "Step 3 (speech tokenizer): {:.1}ms",
            t1.elapsed().as_secs_f64() * 1000.0
        );
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
        let language_id = self
            .config
            .codec_language_id
            .get(&resolved_lang.to_lowercase());

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

        // ICL mode: prefill is 9 positions (no first_text+codec_bos)
        let prefill_embed = Tensor::cat(&[&role_embed, &text_codec_prefix], 1)
            .map_err(|e| FerrumError::model(format!("prefill cat: {e}")))?;

        let t3 = std::time::Instant::now();

        // Step 6b: Build ICL block — [ref_text, target_text, tts_eos] + [codec_bos, ref_codec]
        // Streaming mode: element-wise sum, trailing = text beyond codec length
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

        // Codec stream: batch all codebook embeddings in one shot
        // Collect all first-codebook IDs → single batch embed
        let t_codec_start = std::time::Instant::now();
        let ncg = self.config.num_code_groups;
        let first_codes: Vec<u32> = ref_codes.iter().map(|f| f[0]).collect();
        let codec_frames_cat = {
            // Batch embed main codec: [1, nframes, hidden]
            let mut sum = embed_codec_ids(&first_codes)?;
            // Batch embed each sub-codebook and accumulate
            for cb in 0..(ncg - 1) {
                let codes: Vec<u32> = ref_codes.iter().map(|f| f[cb + 1]).collect();
                let codes_t = Tensor::new(codes.as_slice(), &device)
                    .map_err(|e| FerrumError::model(format!("batch codes: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| FerrumError::model(format!("batch unsqueeze: {e}")))?;
                let sub_embed = codes_t
                    .apply(&self.sub_talker.codec_embeddings[cb])
                    .map_err(|e| FerrumError::model(format!("batch sub_embed: {e}")))?;
                sum =
                    (sum + sub_embed).map_err(|e| FerrumError::model(format!("batch add: {e}")))?;
            }
            sum
        };
        info!(
            "Codec embedding: {:.1}ms ({} frames × {} codebooks)",
            t_codec_start.elapsed().as_secs_f64() * 1000.0,
            ref_codes.len(),
            ncg
        );

        // Prepend codec_bos to codec frames: [codec_bos_embed, codec_frames]
        let t_merge = std::time::Instant::now();
        let codec_bos_for_icl = embed_codec_ids(&[codec_bos])?;
        let icl_codec = Tensor::cat(&[&codec_bos_for_icl, &codec_frames_cat], 1)
            .map_err(|e| FerrumError::model(format!("icl_codec cat: {e}")))?;
        let codec_icl_len = icl_codec
            .dim(1)
            .map_err(|e| FerrumError::model(format!("codec_icl_len dim: {e}")))?;

        // Build ICL embed: element-wise sum of text and codec (streaming mode)
        let icl_trailing: Tensor;
        let icl_embed: Tensor;
        if text_len > codec_icl_len {
            let text_part = text_embed_with_eos
                .narrow(1, 0, codec_icl_len)
                .map_err(|e| FerrumError::model(format!("text_part narrow: {e}")))?;
            icl_embed = (&text_part + &icl_codec)
                .map_err(|e| FerrumError::model(format!("text+codec sum: {e}")))?;
            icl_trailing = text_embed_with_eos
                .narrow(1, codec_icl_len, text_len - codec_icl_len)
                .map_err(|e| FerrumError::model(format!("trailing narrow: {e}")))?;
        } else {
            let n_pad = codec_icl_len - text_len;
            let text_padded = if n_pad > 0 {
                let pad_block = tts_pad_embed
                    .expand((1, n_pad, self.config.hidden_size))
                    .map_err(|e| FerrumError::model(format!("pad expand: {e}")))?;
                Tensor::cat(&[&text_embed_with_eos, &pad_block], 1)
                    .map_err(|e| FerrumError::model(format!("text_padded cat: {e}")))?
            } else {
                text_embed_with_eos.clone()
            };
            icl_embed = (&text_padded + &icl_codec)
                .map_err(|e| FerrumError::model(format!("padded+codec sum: {e}")))?;
            icl_trailing = tts_pad_embed.clone();
        }
        let trailing_text_len = icl_trailing
            .dim(1)
            .map_err(|e| FerrumError::model(format!("trailing dim: {e}")))?;

        // Debug: dump values for comparison with reference project
        // Step 6c: Run prefill then ICL block as SEPARATE forward passes
        let _prefill_out = self.talker.forward_step(&prefill_embed)?;
        let t_icl = std::time::Instant::now();
        let icl_hidden = self.talker.forward_step(&icl_embed)?;
        let icl_len = icl_hidden
            .dim(1)
            .map_err(|e| FerrumError::model(format!("icl_hidden dim: {e}")))?;
        info!(
            "ICL block: {:.1}ms ({} tokens), trailing={}",
            t_icl.elapsed().as_secs_f64() * 1000.0,
            icl_len,
            trailing_text_len
        );

        // Use ICL hidden output for logits and decode
        let mut hidden = icl_hidden;
        let hidden_len = hidden
            .dim(1)
            .map_err(|e| FerrumError::model(format!("hidden dim: {e}")))?;
        let last_hidden = hidden
            .narrow(1, hidden_len - 1, 1)
            .map_err(|e| FerrumError::model(format!("narrow last: {e}")))?;
        if let Ok(v) = last_hidden.flatten_all().and_then(|t| t.to_vec1::<f32>()) {}
        let current_logits = self.talker.logits(&last_hidden)?;
        {}

        // Decode loop
        let mut all_codec_tokens: Vec<Vec<u32>> = Vec::new();
        let mut current_logits = current_logits;

        // Suppress special tokens [vocab_size-1024, vocab_size) except EOS
        let suppress_start = self.config.vocab_size.saturating_sub(1024);
        let suppress_end = self.config.vocab_size;

        // ICL mode: stronger repetition penalty (matching reference Rust project)
        // + repetition detection for early stop.
        const ICL_REPETITION_PENALTY: f32 = 1.5;
        const ICL_FRAMES_PER_TOKEN: usize = 6;
        const ICL_MIN_FRAMES: usize = 75;
        let max_icl_tokens = ICL_MIN_FRAMES.max(text_content_ids.len() * ICL_FRAMES_PER_TOKEN);
        let mut generated_tokens: Vec<u32> = Vec::new();

        for step in 0..max_icl_tokens {
            let mut logits_vec = logits_to_vec(&current_logits)?;
            // Suppress special tokens [vocab-1024, vocab) except EOS
            for i in suppress_start..suppress_end.min(logits_vec.len()) {
                if i as u32 != codec_eos {
                    logits_vec[i] = f32::NEG_INFINITY;
                }
            }
            // min_new_tokens: suppress EOS until we've generated a minimum
            // number of frames. Reference Python uses sentence-length heuristic.
            // FERRUM_TTS_MIN_FRAMES env lets us tune without rebuilding.
            let min_frames: usize = std::env::var("FERRUM_TTS_MIN_FRAMES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| text_content_ids.len() * ICL_FRAMES_PER_TOKEN);
            if step < min_frames {
                if let Some(v) = logits_vec.get_mut(codec_eos as usize) {
                    *v = f32::NEG_INFINITY;
                }
            }
            // Repetition penalty with token history
            for &prev_tok in &generated_tokens {
                let idx = prev_tok as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= ICL_REPETITION_PENALTY;
                    } else {
                        logits_vec[idx] *= ICL_REPETITION_PENALTY;
                    }
                }
            }
            let next_token = sample_token(
                &logits_vec,
                tts_temperature(),
                TOP_K,
                ICL_REPETITION_PENALTY,
            );

            generated_tokens.push(next_token);

            if next_token == codec_eos {
                info!("TTS voice clone: codec EOS at step {}", step);
                break;
            }

            // Repetition detection: check for repeating patterns of length 1-4
            if generated_tokens.len() >= 6 {
                let n = generated_tokens.len();
                let mut is_repeat = false;
                for pat_len in 1..=4 {
                    if n >= pat_len * 3 {
                        let a = &generated_tokens[n - pat_len * 3..n - pat_len * 2];
                        let b = &generated_tokens[n - pat_len * 2..n - pat_len];
                        let c = &generated_tokens[n - pat_len..n];
                        if a == b && b == c {
                            is_repeat = true;
                            break;
                        }
                    }
                }
                if is_repeat {
                    info!(
                        "TTS voice clone: repetition detected at step {}, stopping",
                        step
                    );
                    break;
                }
            }

            let cur_hidden_len = hidden
                .dim(1)
                .map_err(|e| FerrumError::model(format!("hidden dim: {e}")))?;
            let last_hidden = hidden
                .narrow(1, cur_hidden_len - 1, 1)
                .map_err(|e| FerrumError::model(format!("last_hidden: {e}")))?;

            let token_tensor = Tensor::new(&[next_token], &device)
                .map_err(|e| FerrumError::model(format!("token tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
            let first_codec_embed = self.talker.embed_codec(&token_tensor)?;

            let extra_codes = self.sub_talker.predict(
                &last_hidden,
                &first_codec_embed,
                st_temperature(),
                TOP_K,
            )?;

            let mut frame_codes = vec![next_token];
            frame_codes.extend_from_slice(&extra_codes);
            all_codec_tokens.push(frame_codes);

            // Sum all codebook embeddings on GPU
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

            // Add trailing text or tts_pad (matching reference streaming mode)
            if step < trailing_text_len {
                let trail = icl_trailing
                    .narrow(1, step, 1)
                    .map_err(|e| FerrumError::model(format!("trailing narrow: {e}")))?;
                combined_embed = (combined_embed + trail)
                    .map_err(|e| FerrumError::model(format!("add trailing: {e}")))?;
            } else {
                combined_embed = (combined_embed + &tts_pad_embed)
                    .map_err(|e| FerrumError::model(format!("add tts_pad: {e}")))?;
            }

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
/// Sample a token matching qwen3-tts-rs reference:
/// 1. temperature scaling
/// 2. top-k filter (keep top_k, rest = -inf)
/// 3. top-p filter (keep smallest set with cumprob > top_p, rest = -inf)
/// 4. softmax over filtered logits
/// 5. multinomial sample from distribution
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    _repetition_penalty: f32,
) -> u32 {
    if temperature < 0.01 {
        return argmax(logits);
    }

    let vocab = logits.len();

    // 1. Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // 2. Top-k filter: keep top_k values, set rest to -inf
    let mut filtered = scaled.clone();
    if top_k > 0 && top_k < vocab {
        let mut sorted = scaled.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[top_k - 1];
        for v in &mut filtered {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // 3. Top-p filter: keep smallest set of tokens whose cumulative prob >= top_p
    const TOP_P: f32 = 0.9;
    {
        let mut indices: Vec<usize> = (0..vocab).collect();
        indices.sort_unstable_by(|&a, &b| {
            filtered[b]
                .partial_cmp(&filtered[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Softmax over sorted values for cumulative prob
        let max_val = filtered[indices[0]];
        let exp_sorted: Vec<f32> = indices
            .iter()
            .map(|&i| (filtered[i] - max_val).exp())
            .collect();
        let sum: f32 = exp_sorted.iter().sum();
        let probs_sorted: Vec<f32> = exp_sorted.iter().map(|e| e / sum).collect();

        // Find cutoff
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = vocab;
        for (i, &p) in probs_sorted.iter().enumerate() {
            cumsum += p;
            if cumsum > TOP_P {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Mask out tokens beyond cutoff
        for &idx in &indices[cutoff_idx..] {
            filtered[idx] = f32::NEG_INFINITY;
        }
    }

    // 4. Softmax over filtered logits
    let max_val = filtered.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = filtered.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // 5. Multinomial sample
    let r = rand_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return i as u32;
        }
    }
    // Fallback
    argmax(&probs)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// RNG matching qwen3-tts-rs: LCG with subsec_nanos seed + counter
fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    let state = seed
        .wrapping_add(count)
        .wrapping_mul(1103515245)
        .wrapping_add(12345);
    (state as f32) / (u64::MAX as f32)
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
