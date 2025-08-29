//! # Ferrum Engine
//! 
//! Core inference engine implementation with support for:
//! - Continuous batching
//! - Dynamic scheduling
//! - KV cache management
//! - Model execution with Candle

pub mod engine;
pub mod executor;
pub mod batch_manager;
pub mod cache_manager;
pub mod memory_manager;
pub mod model_runner;
pub mod scheduler;
pub mod attention;

// Re-exports
pub use engine::{Engine, EngineConfig};
pub use executor::{CandleExecutor, ExecutorConfig};
pub use batch_manager::{ContinuousBatchManager, BatchConfig};
pub use cache_manager::{PagedKVCacheManager, CacheConfig};
pub use memory_manager::{GpuMemoryManager, MemoryConfig};
pub use scheduler::{FairScheduler, SchedulerConfig};

use ferrum_core::{Error, Result};
