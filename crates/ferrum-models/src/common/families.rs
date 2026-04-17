//! Model-family traits beyond `DecoderOnlyLLM`.
//!
//! Extension points only at this stage — no model in the current tree
//! implements them yet. Landing order in Phase D:
//!
//!   `MultimodalLLM`      Qwen-VL / LLaVA (ViT backbone + LLM decoder)
//!   `EncoderDecoderLM`   Whisper (encoder hidden + decoder loop)
//!   `EmbeddingModel`     Bert / E5 / multilingual-e5 (single forward → hidden)
//!   `Transcriber`        Whisper CLI-facing API
//!   `TtsModel`           Qwen3-TTS (talker + vocoder pipeline)
//!
//! Each trait is written so it composes with the existing
//! `DecoderOnlyLLM` where appropriate (Multimodal reuses decoder loop,
//! Transcriber wraps an EncoderDecoderLM + mel frontend, etc.).

use crate::common::llm::DecoderOnlyLLM;

/// Opaque block of visual tokens produced by a vision encoder.
/// Exact shape depends on the model; commonly `[num_patches, hidden]`.
pub type VisualTokens = Vec<f32>;

/// Opaque block of audio tokens (mel spectrogram features or encoder output).
pub type AudioTokens = Vec<f32>;

/// Opaque sample buffer in bytes (image pixel data).
pub type ImageBuffer = Vec<u8>;

/// PCM audio buffer — f32 mono samples.
pub type PcmSamples = Vec<f32>;

/// One output segment from a transcriber (start/end seconds + text).
#[derive(Clone, Debug)]
pub struct TranscriptSegment {
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
}

/// One synthesized audio chunk (stereo not supported yet).
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

/// Optional reference for voice-cloning-style TTS.
#[derive(Clone, Debug)]
pub struct SpeakerRef {
    pub ref_audio: Vec<f32>,
    pub ref_text: Option<String>,
}

// ── Multimodal LLM ──────────────────────────────────────────────────────
//
// A multimodal LLM is a decoder-only LLM that additionally accepts visual
// and/or audio inputs (Qwen-VL, LLaVA, etc.). The image/audio encoders
// typically share the Backend trait but have dedicated model code
// (separate file per model family).

pub trait MultimodalLLM: DecoderOnlyLLM {
    /// Encode an image into visual tokens that can be injected into the
    /// decoder's prefill token stream (typical LLaVA/Qwen-VL flow).
    fn encode_image(&mut self, pixels: &ImageBuffer) -> VisualTokens;

    /// Optional audio path; models that don't support audio leave the default.
    fn encode_audio(&mut self, _pcm: &PcmSamples) -> AudioTokens {
        Vec::new()
    }
}

// ── Encoder + Decoder ────────────────────────────────────────────────────
//
// Encoder-decoder models (Whisper, T5, BART) keep encoder hidden state
// around for the duration of decode. The encoder state is opaque to the
// engine — each model defines its own.

pub trait EncoderDecoderLM: Send + Sync {
    /// Encoded side output. Type-erased so different models can carry
    /// different shapes (Whisper: `[n_audio_frames, hidden]`).
    fn encode(&mut self, cache_id: &str, input: &[u32]) -> EncoderState;

    /// Advance the decoder one step, conditioned on `encoder` produced earlier.
    fn decode_step(
        &mut self,
        cache_id: &str,
        token: u32,
        pos: u32,
        encoder: &EncoderState,
    ) -> Vec<f32>;

    fn release(&mut self, cache_id: &str);
}

/// Encoder-side state handed back from `encode()` and passed into
/// `decode_step()`. Opaque to the engine.
#[derive(Clone)]
pub struct EncoderState {
    pub hidden: Vec<f32>,
    pub shape: Vec<usize>,
}

// ── Embedding Model ──────────────────────────────────────────────────────

pub trait EmbeddingModel: Send + Sync {
    /// Run a single forward pass over a token sequence and return the pooled
    /// embedding (typical CLS pooling: `[hidden]`).
    fn embed(&mut self, tokens: &[u32]) -> Vec<f32>;
}

// ── Transcriber ──────────────────────────────────────────────────────────
//
// Higher-level audio-to-text API. Wraps an internal encoder-decoder model
// plus mel-spectrogram frontend + sampler; CLI only sees this trait.

pub trait Transcriber: Send + Sync {
    fn transcribe(&mut self, pcm: &PcmSamples, language: Option<&str>) -> Vec<TranscriptSegment>;
}

// ── TTS Model ────────────────────────────────────────────────────────────

pub trait TtsModel: Send + Sync {
    fn synthesize(&mut self, text: &str, speaker: Option<&SpeakerRef>) -> AudioBuffer;
}
