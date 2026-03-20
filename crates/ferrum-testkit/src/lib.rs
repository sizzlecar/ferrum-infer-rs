//! Mock components for testing the ferrum inference engine without GPU.
//!
//! Provides MockTensor, MockModelExecutor, MockKvCacheManager, MockTokenizer,
//! and MockSampler — all hardware-independent, suitable for testing the
//! scheduling and orchestration layers on any platform.

mod tensor;
mod executor;
mod kv_cache;
mod tokenizer;
mod sampler;
pub mod paged_executor;

pub use tensor::{MockTensor, MockTensorFactory};
pub use executor::MockModelExecutor;
pub use kv_cache::{MockKvCacheHandle, MockKvCacheManager};
pub use tokenizer::MockTokenizer;
pub use sampler::MockSampler;
pub use paged_executor::{PagedAttentionExecutor, PagedExecutorConfig};
