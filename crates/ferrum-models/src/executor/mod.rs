//! Model executor implementations.
//!
//! Decoder-only LLMs go through `LlmExecutor` (wrapping any
//! `Box<dyn DecoderOnlyLLM>`) or the registered plan runtime. Future task
//! families must use their own execution contracts, without dummy KV caches.

pub mod common;
pub mod llm_executor;
pub mod stub_executor;
mod vnext_checkpoint;
mod vnext_completion_worker;
pub mod vnext_executor;
mod vnext_timing;

pub use llm_executor::LlmExecutor;
pub use stub_executor::StubModelExecutor;
pub use vnext_executor::{
    VNextCompiledModel, VNextDeterminismExecutionMode, VNextDeterminismExecutionSpec,
    VNextDeterminismInitialState, VNextDeterminismParticipantSpec, VNextDeterminismPhase,
    VNextDeterminismWorkspacePoison, VNextExecutorConfig, VNextModelCompilationError,
    VNextModelExecutor, VNextRuntimeComposition, MAX_VNEXT_DETERMINISM_PARTICIPANTS,
};
