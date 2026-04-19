//! Testing utilities for the ferrum inference engine.
//!
//! - **Mocks**: MockModelExecutor, MockSampler, MockKvCacheManager, MockTokenizer
//! - **Configurable**: ConfigurableModelExecutor (specific token sequences, EOS)
//! - **Bench**: BenchmarkResult, percentile calculation, JSON output
//! - **Paged**: PagedAttentionExecutor with real paged KV cache
//!
//! All components are hardware-independent (CPU-only, no GPU required).

pub mod bench;
mod configurable_executor;
mod executor;
mod kv_cache;
pub mod paged_executor;
mod sampler;
mod tensor;
mod tokenizer;

pub use configurable_executor::ConfigurableModelExecutor;
pub use executor::MockModelExecutor;
pub use kv_cache::{MockKvCacheHandle, MockKvCacheManager};
pub use paged_executor::{PagedAttentionExecutor, PagedExecutorConfig};
pub use sampler::MockSampler;
pub use tensor::{MockTensor, MockTensorFactory};
pub use tokenizer::MockTokenizer;
