//! Model executor implementations.
//!
//! Decoder-only LLMs go through `LlmExecutor` (wrapping any
//! `Box<dyn DecoderOnlyLLM>`) or the registered plan runtime. Future task
//! families must use their own execution contracts, without dummy KV caches.

pub mod common;
#[cfg(test)]
mod driver_contract_tests;
pub mod llm_executor;
pub mod stub_executor;

pub use llm_executor::LlmExecutor;
pub use stub_executor::StubModelExecutor;
