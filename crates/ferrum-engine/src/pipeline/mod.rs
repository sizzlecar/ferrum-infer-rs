//! Inference Pipeline Implementations
//!
//! This module provides optimized pipeline implementations for different
//! inference patterns:
//!
//! - Chunked Prefill: Split long prompts into chunks for better memory efficiency
//! - Prefill-Decode Separation: Dedicated handling for each phase
//! - Batch Pipeline: Efficient processing of multiple requests

pub mod chunked_prefill;
pub mod executor;

pub use chunked_prefill::{ChunkedPrefillConfig, ChunkedPrefillExecutor};
pub use executor::{PipelineExecutor, PipelineConfig, ExecutionPhase};


