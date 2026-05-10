//! Legacy architecture wrappers (candle-based, non-decoder model types).
//!
//! Decoder-only LLMs moved to `crate::models::llama_family::LlamaFamilyModel`.
//! Remaining entries here are modalities that haven't migrated yet
//! (Whisper encoder-decoder, Bert, CLIP, Qwen3-TTS pipeline) — Phase D
//! target.

pub mod bert;
pub mod clip;
pub mod qwen3_tts;
pub mod qwen3_tts_backbone;
pub mod qwen3_tts_backend;
pub mod qwen3_tts_vocoder;
pub mod speaker_encoder;
pub mod speech_tokenizer_encoder;
pub mod whisper;

pub use bert::BertModelWrapper;
pub use clip::ClipModelWrapper;
pub use whisper::WhisperModelWrapper;

/// GQA repeat_kv: repeat K/V heads to match Q heads.
/// Still used by Qwen3-TTS (Phase D target for migration).
pub(crate) fn repeat_kv(
    x: candle_core::Tensor,
    n_rep: usize,
) -> candle_core::Result<candle_core::Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, nkv, seq, hd) = x.dims4()?;
    let x = x.unsqueeze(2)?;
    let x = x.expand((b, nkv, n_rep, seq, hd))?;
    x.reshape((b, nkv * n_rep, seq, hd))
}
