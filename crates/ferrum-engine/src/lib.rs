//! # Ferrum Engine
//!
//! Core inference engine implementation with support for:
//! - Continuous batching
//! - Dynamic scheduling
//! - KV cache management
//! - Model execution with Candle

pub mod attention;
pub mod batch_manager;
pub mod cache_manager;
pub mod candle_backend;
pub mod engine;
pub mod executor;
pub mod memory_manager;
pub mod model_runner;
pub mod scheduler;

// Re-exports
pub use batch_manager::{BatchConfig, ContinuousBatchManager};
pub use cache_manager::{CacheConfig, PagedKVCacheManager};
pub use candle_backend::{CandleBackend, CandleModel};
pub use engine::{Engine, EngineConfig};
pub use executor::{ExecutorConfig, GenericExecutor};
pub use memory_manager::{GpuMemoryManager, MemoryConfig};
pub use scheduler::{FairScheduler, SchedulerConfig};
