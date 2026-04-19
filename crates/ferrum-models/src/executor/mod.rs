//! Model executor implementations.
//!
//! Decoder-only LLMs go through `LlmExecutor` (wrapping any
//! `Box<dyn DecoderOnlyLLM>`). Per-modality executors (Bert / Clip / Whisper /
//! Tts) remain separate — they have different forward contracts that don't
//! fit the prefill/decode interface.

pub mod bert_executor;
pub mod clip_executor;
pub mod common;
pub mod llm_executor;
pub mod stub_executor;
pub mod tts_executor;
pub mod whisper_executor;

pub use bert_executor::BertModelExecutor;
pub use clip_executor::ClipModelExecutor;
pub use llm_executor::LlmExecutor;
pub use stub_executor::StubModelExecutor;
pub use tts_executor::TtsModelExecutor;
pub use whisper_executor::WhisperModelExecutor;
