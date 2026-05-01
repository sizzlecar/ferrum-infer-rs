//! Cross-model traits and helpers (Model-as-Code shared infrastructure).

pub mod families;
pub mod llm;
pub mod paged_pool;

pub use families::{
    AudioBuffer, AudioTokens, EmbeddingModel, EncoderDecoderLM, EncoderState, ImageBuffer,
    MultimodalLLM, PcmSamples, SpeakerRef, Transcriber, TranscriptSegment, TtsModel, VisualTokens,
};
pub use llm::{DecoderOnlyLLM, LlmRuntimeConfig};
