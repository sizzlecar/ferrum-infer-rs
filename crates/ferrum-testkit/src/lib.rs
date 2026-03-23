//! Mock components for testing the ferrum inference engine without GPU.
//!
//! Provides MockTensor, MockModelExecutor, MockKvCacheManager, MockTokenizer,
//! and MockSampler — all hardware-independent, suitable for testing the
//! scheduling and orchestration layers on any platform.

mod executor;
mod kv_cache;
pub mod paged_executor;
mod sampler;
mod tensor;
mod tokenizer;

pub use executor::MockModelExecutor;
pub use kv_cache::{MockKvCacheHandle, MockKvCacheManager};
pub use paged_executor::{PagedAttentionExecutor, PagedExecutorConfig};
pub use sampler::MockSampler;
pub use tensor::{MockTensor, MockTensorFactory};
pub use tokenizer::MockTokenizer;
