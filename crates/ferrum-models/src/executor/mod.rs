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
#[cfg(any(test, feature = "test-support"))]
pub mod qwen35_executor;
pub mod stub_executor;
pub mod tts_executor;
mod vnext_checkpoint;
mod vnext_completion_worker;
pub mod vnext_executor;
mod vnext_timing;
pub mod whisper_executor;

pub use bert_executor::BertModelExecutor;
pub use clip_executor::ClipModelExecutor;
pub use llm_executor::LlmExecutor;
#[cfg(any(test, feature = "test-support"))]
pub use qwen35_executor::Qwen35W3Executor;
pub use stub_executor::StubModelExecutor;
pub use tts_executor::TtsModelExecutor;
pub use vnext_executor::{VNextExecutorConfig, VNextModelExecutor};
pub use whisper_executor::WhisperModelExecutor;
