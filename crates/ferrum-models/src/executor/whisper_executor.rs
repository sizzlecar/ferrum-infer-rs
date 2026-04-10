//! Whisper ASR Executor — full decode pipeline matching Python whisper.
//!
//! Implements: timestamp-based sequential decode, logit suppression (SuppressBlank,
//! SuppressTokens, ApplyTimestampRules), temperature fallback, compression ratio
//! check, seek-based segmentation.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device as CandleDevice, Tensor};
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

// ── Token constants ─────────────────────────────────────────────────────
// These match Python whisper's tokenizer for multilingual models.

const TIMESTAMP_BEGIN: u32 = 50364;
const INPUT_STRIDE: usize = 2; // mel frames per output token (N_FRAMES / n_audio_ctx = 3000/1500)
const TIME_PRECISION: f64 = 0.02; // seconds per timestamp token (INPUT_STRIDE * HOP_LENGTH / SAMPLE_RATE)

/// Non-speech token IDs to suppress (from Python whisper tokenizer.non_speech_tokens).
/// These are symbols that shouldn't appear in transcription (from Python whisper).
const NON_SPEECH_TOKENS: &[u32] = &[
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357,
    366, 438, 532, 685, 691, 1060, 1258, 1261, 1435, 1436, 1652, 2028, 2029, 2150, 2404, 2932,
    3292, 3455, 3723, 4100, 5751, 6283, 6347, 6436, 6615, 7579, 8765, 9929, 10563, 10813, 11318,
    12380, 14117, 14397, 14734, 15003, 15068, 15206, 16450, 16805, 17193, 17832, 19063, 19438,
    19635, 20203, 21111, 24220, 24408, 25212, 25830, 26622, 28156, 28279, 29464, 31650, 32302,
    32470, 36865, 42863, 47425, 49870, 50254,
];

/// Whisper executor for speech-to-text.
pub struct WhisperModelExecutor {
    model: WhisperModelWrapper,
    tokenizer: tokenizers::Tokenizer,
    info: ModelInfo,
    // Special token IDs
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    no_timestamps_token: u32,
    no_speech_token: u32, // <|nocaptions|> = 50362
    sot_prev: u32,
    sot_lm: u32,
    language_tokens: HashMap<String, u32>,
    /// Precomputed suppress mask: token IDs that are always suppressed.
    suppress_token_ids: Vec<u32>,
    /// Sample length (max decode tokens per segment).
    sample_len: usize,
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
        let translate_token = token_id(&tokenizer, "<|translate|>");
        let no_timestamps_token = token_id(&tokenizer, "<|notimestamps|>");
        let no_speech_token = token_id(&tokenizer, "<|nocaptions|>");
        let sot_prev = token_id(&tokenizer, "<|startofprev|>");
        let sot_lm = token_id(&tokenizer, "<|startoflm|>");

        // Build language token map
        let mut language_tokens = HashMap::new();
        for lang in &[
            "en", "zh", "ja", "ko", "fr", "de", "es", "ru", "ar", "pt", "it", "nl", "tr", "pl",
            "sv", "da", "fi", "hu", "cs", "ro", "bg", "uk", "el", "hr", "sk", "th", "vi", "id",
            "ms", "hi", "ta", "te", "ur", "fa", "he", "ca", "gl", "eu", "la",
        ] {
            let token_str = format!("<|{lang}|>");
            if let Some(id) = tokenizer.token_to_id(&token_str) {
                language_tokens.insert(lang.to_string(), id);
            }
        }

        // Build suppress token list (matches Python _get_suppress_tokens)
        let mut suppress_ids: Vec<u32> = NON_SPEECH_TOKENS.to_vec();
        suppress_ids.extend_from_slice(&[
            transcribe_token,
            translate_token,
            sot_token,
            sot_prev,
            sot_lm,
            no_speech_token,
        ]);
        suppress_ids.sort();
        suppress_ids.dedup();

        let sample_len = model.config().max_target_positions / 2;

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
            "WhisperModelExecutor: {} (d_model={}, languages={}, suppress_tokens={})",
            model_path,
            model.config().d_model,
            language_tokens.len(),
            suppress_ids.len(),
        );

        Ok(Self {
            model,
            tokenizer,
            info,
            sot_token,
            eot_token,
            transcribe_token,
            translate_token,
            no_timestamps_token,
            no_speech_token,
            sot_prev,
            sot_lm,
            language_tokens,
            suppress_token_ids: suppress_ids,
            sample_len,
        })
    }

    /// Transcribe audio file → text.
    pub fn transcribe_file(&self, audio_path: &str, language: Option<&str>) -> Result<String> {
        let pcm = audio_processor::load_audio(audio_path)?;
        self.transcribe_pcm(&pcm, language)
    }

    /// Transcribe raw audio bytes (WAV/any) → text.
    pub fn transcribe_bytes(&self, audio_data: &[u8], language: Option<&str>) -> Result<String> {
        let pcm = audio_processor::load_audio_bytes(audio_data)?;
        self.transcribe_pcm(&pcm, language)
    }

    /// Full transcription pipeline matching Python whisper.transcribe().
    ///
    /// - Computes mel for entire audio (padded by 30s of silence)
    /// - Seek-based loop over 30s segments
    /// - For each segment: encode → decode with timestamp rules → extract segments
    /// - Temperature fallback on high compression ratio or low avg logprob
    fn transcribe_pcm(&self, pcm: &[f32], language: Option<&str>) -> Result<String> {
        let lang_token = language
            .and_then(|l| self.language_tokens.get(l).copied())
            .unwrap_or_else(|| {
                self.language_tokens
                    .get("en")
                    .copied()
                    .unwrap_or(self.sot_token + 1)
            });

        // Compute mel for full audio + 30s padding (matching Python: padding=N_SAMPLES)
        let n_samples = candle_transformers::models::whisper::N_SAMPLES;
        let n_frames = candle_transformers::models::whisper::N_FRAMES;
        let mut padded_pcm = pcm.to_vec();
        padded_pcm.resize(padded_pcm.len() + n_samples, 0.0); // 30s silence padding
        let content_frames = pcm.len() / candle_transformers::models::whisper::HOP_LENGTH;

        // Initial tokens: SOT + language + transcribe (WITHOUT no_timestamps — we use timestamps)
        let sot_sequence = vec![self.sot_token, lang_token, self.transcribe_token];
        let sample_begin = sot_sequence.len();

        // Blank token for SuppressBlank
        let blank_token = 220u32; // space token, matches Python tokenizer.encode(" ")

        // Temperatures for fallback
        let temperatures: &[f32] = &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

        // Max initial timestamp: 1.0 second → index 50 (1.0 / 0.02)
        let max_initial_timestamp_index: u32 = 50;

        let mut seek: usize = 0;
        let mut all_tokens: Vec<u32> = Vec::new();

        while seek < content_frames {
            let segment_size = n_frames.min(content_frames - seek);

            // Extract mel segment
            let mel = self.mel_segment_at(&padded_pcm, seek, n_frames)?;

            // Encode
            let encoder_out = self.model.encode(&mel)?;

            // Decode with temperature fallback
            let (tokens, avg_logprob, no_speech_prob, _temperature) = self.decode_with_fallback(
                &encoder_out,
                &sot_sequence,
                sample_begin,
                blank_token,
                max_initial_timestamp_index,
                temperatures,
            )?;

            // No speech check (matching Python transcribe.py):
            // Skip segment if no_speech_prob is high, unless logprob is also high
            let should_skip = no_speech_prob > 0.6 && avg_logprob < -1.0;
            if should_skip {
                seek += segment_size;
                continue;
            }

            // Parse timestamp tokens to determine seek advancement
            let sampled = &tokens[sample_begin..];
            let timestamp_mask: Vec<bool> = sampled.iter().map(|&t| t >= TIMESTAMP_BEGIN).collect();

            // Find consecutive timestamp pairs
            let mut consecutive_indices = Vec::new();
            for i in 0..timestamp_mask.len().saturating_sub(1) {
                if timestamp_mask[i] && timestamp_mask[i + 1] {
                    consecutive_indices.push(i + 1);
                }
            }

            // Collect text tokens (strip timestamps and special tokens)
            let text_tokens: Vec<u32> = sampled
                .iter()
                .copied()
                .filter(|&t| t < self.eot_token)
                .collect();
            all_tokens.extend_from_slice(&text_tokens);

            if !consecutive_indices.is_empty() {
                // Has consecutive timestamps → use last timestamp to advance seek
                let single_timestamp_ending = timestamp_mask.len() >= 2
                    && !timestamp_mask[timestamp_mask.len() - 2]
                    && timestamp_mask[timestamp_mask.len() - 1];

                if single_timestamp_ending {
                    seek += segment_size;
                } else {
                    let last_idx = *consecutive_indices.last().unwrap();
                    let last_ts_pos = (sampled[last_idx] - TIMESTAMP_BEGIN) as usize;
                    seek += last_ts_pos * INPUT_STRIDE;
                }
            } else {
                // No consecutive timestamps — check for any single timestamp
                let timestamps: Vec<u32> = sampled
                    .iter()
                    .copied()
                    .filter(|&t| t >= TIMESTAMP_BEGIN)
                    .collect();
                if !timestamps.is_empty() && *timestamps.last().unwrap() != TIMESTAMP_BEGIN {
                    let last_ts_pos = (*timestamps.last().unwrap() - TIMESTAMP_BEGIN) as usize;
                    seek += last_ts_pos * INPUT_STRIDE;
                } else {
                    seek += segment_size;
                }
            }
        }

        // Decode all collected text tokens
        let text = self
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| FerrumError::model(format!("decode tokens: {e}")))?;

        Ok(text.trim().to_string())
    }

    /// Extract mel segment at given seek position, pad to n_frames.
    fn mel_segment_at(&self, pcm: &[f32], seek_frames: usize, n_frames: usize) -> Result<Tensor> {
        let hop = candle_transformers::models::whisper::HOP_LENGTH;
        let start_sample = seek_frames * hop;
        let n_samples = candle_transformers::models::whisper::N_SAMPLES;
        let end_sample = (start_sample + n_samples).min(pcm.len());
        let segment = &pcm[start_sample..end_sample];
        self.model.pcm_to_mel_tensor(segment)
    }

    /// Decode one segment with temperature fallback.
    /// Returns (all_tokens, avg_logprob, no_speech_prob, temperature_used).
    fn decode_with_fallback(
        &self,
        encoder_out: &Tensor,
        sot_sequence: &[u32],
        sample_begin: usize,
        blank_token: u32,
        max_initial_timestamp_index: u32,
        temperatures: &[f32],
    ) -> Result<(Vec<u32>, f32, f32, f32)> {
        let mut last_result = None;

        for &temp in temperatures {
            let (tokens, avg_logprob, no_speech_prob) = self.decode_segment(
                encoder_out,
                sot_sequence,
                sample_begin,
                blank_token,
                max_initial_timestamp_index,
                temp,
            )?;

            let text_tokens: Vec<u32> = tokens[sample_begin..]
                .iter()
                .copied()
                .filter(|&t| t < self.eot_token)
                .collect();
            let text = self
                .tokenizer
                .decode(&text_tokens, true)
                .unwrap_or_default();

            let cr = compression_ratio(&text);

            // Matching Python: fallback if too repetitive or logprob too low,
            // but NOT if it's silence (high no_speech_prob overrides).
            let mut needs_fallback = false;
            if cr > 2.4 {
                needs_fallback = true;
            }
            if avg_logprob < -1.0 {
                needs_fallback = true;
            }
            if no_speech_prob > 0.6 {
                needs_fallback = false; // silence — accept as-is
            }

            last_result = Some((tokens, avg_logprob, no_speech_prob, temp));

            if !needs_fallback {
                break;
            }
        }

        last_result.ok_or_else(|| FerrumError::model("decode_with_fallback: no result"))
    }

    /// Decode one segment at a given temperature.
    /// Returns (full_token_sequence, avg_logprob, no_speech_prob).
    fn decode_segment(
        &self,
        encoder_out: &Tensor,
        sot_sequence: &[u32],
        sample_begin: usize,
        blank_token: u32,
        max_initial_timestamp_index: u32,
        temperature: f32,
    ) -> Result<(Vec<u32>, f32, f32)> {
        self.model.reset_decoder();

        let mut tokens: Vec<u32> = sot_sequence.to_vec();
        let mut sum_logprobs: f32 = 0.0;
        let mut no_speech_prob: f32 = 0.0;
        let mut n_text_tokens: usize = 0;

        // First forward: feed all initial tokens
        let mut logits = self.model.decode_step(&tokens, encoder_out)?;

        for step in 0..self.sample_len {
            // On first step, capture no_speech_prob
            if step == 0 {
                let sot_logits = &logits; // logits from last position of initial tokens
                let probs = softmax_vec(sot_logits);
                no_speech_prob = probs[self.no_speech_token as usize];
            }

            // ─── Apply logit filters (matching Python) ───

            let sampled_tokens = &tokens[sample_begin..];

            // 1. SuppressBlank: on first text token, suppress blank and EOT
            if sampled_tokens.is_empty() {
                logits[blank_token as usize] = f32::NEG_INFINITY;
                logits[self.eot_token as usize] = f32::NEG_INFINITY;
            }

            // 2. SuppressTokens: always suppress non-speech + special tokens
            for &t in &self.suppress_token_ids {
                if (t as usize) < logits.len() {
                    logits[t as usize] = f32::NEG_INFINITY;
                }
            }

            // 3. ApplyTimestampRules
            self.apply_timestamp_rules(
                &mut logits,
                sampled_tokens,
                sample_begin,
                max_initial_timestamp_index,
                step,
            );

            // ─── Select next token ───
            let next_token = if temperature == 0.0 {
                argmax(&logits)
            } else {
                sample_with_temperature(&logits, temperature)
            };

            // Track logprobs
            let log_probs = log_softmax_vec(&logits);
            if next_token != self.eot_token {
                sum_logprobs += log_probs[next_token as usize];
                n_text_tokens += 1;
            }

            tokens.push(next_token);

            if next_token == self.eot_token
                || tokens.len() > self.model.config().max_target_positions
            {
                break;
            }

            // Repetition detection on text tokens only (ignoring timestamps).
            // If a text token repeats > 5 times in the last 10 text tokens, stop.
            let text_tail: Vec<u32> = tokens[sample_begin..]
                .iter()
                .copied()
                .filter(|&t| t < TIMESTAMP_BEGIN && t != self.eot_token)
                .collect();
            if text_tail.len() >= 6 {
                let last = *text_tail.last().unwrap();
                let consecutive = text_tail.iter().rev().take_while(|&&t| t == last).count();
                if consecutive >= 5 {
                    // Trim: remove the repeated tokens, keep 1
                    let mut keep = tokens.len();
                    let mut removed = 0;
                    while keep > sample_begin && removed < consecutive - 1 {
                        keep -= 1;
                        if tokens[keep] == last {
                            removed += 1;
                        }
                    }
                    tokens.truncate(keep + 1);
                    break;
                }
            }

            // Next step: feed only the new token
            logits = self.model.decode_step(&[next_token], encoder_out)?;
        }

        let avg_logprob = if n_text_tokens > 0 {
            sum_logprobs / n_text_tokens as f32
        } else {
            f32::NEG_INFINITY
        };

        Ok((tokens, avg_logprob, no_speech_prob))
    }

    /// Apply timestamp rules (matching Python ApplyTimestampRules).
    fn apply_timestamp_rules(
        &self,
        logits: &mut [f32],
        sampled_tokens: &[u32],
        _sample_begin: usize,
        max_initial_timestamp_index: u32,
        _step: usize,
    ) {
        let ts_begin = TIMESTAMP_BEGIN as usize;

        // Suppress <|notimestamps|>
        logits[self.no_timestamps_token as usize] = f32::NEG_INFINITY;

        // Timestamp pairing rules
        let last_was_timestamp =
            !sampled_tokens.is_empty() && *sampled_tokens.last().unwrap() >= TIMESTAMP_BEGIN;

        let penultimate_was_timestamp =
            sampled_tokens.len() < 2 || sampled_tokens[sampled_tokens.len() - 2] >= TIMESTAMP_BEGIN;

        if last_was_timestamp {
            if penultimate_was_timestamp {
                // Two timestamps in a row → must produce non-timestamp
                for i in ts_begin..logits.len() {
                    logits[i] = f32::NEG_INFINITY;
                }
            } else {
                // Timestamp after text → must produce timestamp or EOT (suppress all text)
                for i in 0..self.eot_token as usize {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }

        // Monotonically increasing timestamps
        let timestamps: Vec<u32> = sampled_tokens
            .iter()
            .copied()
            .filter(|&t| t >= TIMESTAMP_BEGIN)
            .collect();
        if !timestamps.is_empty() {
            let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                *timestamps.last().unwrap()
            } else {
                *timestamps.last().unwrap() + 1
            };
            for i in ts_begin..timestamp_last as usize {
                if i < logits.len() {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }

        // First token: must be a timestamp, constrained by max_initial_timestamp
        if sampled_tokens.is_empty() {
            for i in 0..ts_begin {
                logits[i] = f32::NEG_INFINITY;
            }
            let last_allowed = TIMESTAMP_BEGIN + max_initial_timestamp_index;
            for i in (last_allowed as usize + 1)..logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }

        // If sum of timestamp probabilities > max text token probability, force timestamp
        let log_probs = log_softmax_vec(logits);
        let ts_logsumexp = {
            let max_ts = log_probs[ts_begin..]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if max_ts.is_finite() {
                max_ts
                    + log_probs[ts_begin..]
                        .iter()
                        .map(|&lp| (lp - max_ts).exp())
                        .sum::<f32>()
                        .ln()
            } else {
                f32::NEG_INFINITY
            }
        };
        let max_text_logprob = log_probs[..ts_begin]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if ts_logsumexp > max_text_logprob {
            for i in 0..ts_begin {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Utility functions ───────────────────────────────────────────────────

fn token_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> u32 {
    tokenizer.token_to_id(token).unwrap_or(0)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn log_softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = max + sum_exp.ln();
    logits.iter().map(|&x| x - log_sum).collect()
}

fn sample_with_temperature(logits: &[f32], temperature: f32) -> u32 {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let probs = softmax_vec(&scaled);
    // Weighted random sampling
    let r: f32 = rand_f32();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn rand_f32() -> f32 {
    // Simple xorshift-based PRNG (good enough for temperature sampling)
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x12345678_9abcdef0);
    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    STATE.store(s, Ordering::Relaxed);
    (s as f32) / (u64::MAX as f32)
}

/// Compression ratio using zlib deflate (matches Python whisper.utils.compression_ratio).
fn compression_ratio(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;
    let text_bytes = text.as_bytes();
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(text_bytes).unwrap();
    let compressed = encoder.finish().unwrap();
    text_bytes.len() as f32 / compressed.len().max(1) as f32
}

// ── Dummy KV cache + ModelExecutor trait impl ───────────────────────────

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
