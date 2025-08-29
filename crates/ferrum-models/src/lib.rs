//! # Ferrum Models
//! 
//! Model implementations for the Ferrum inference framework.
//! Supports various LLM architectures including Llama, Mistral, and more.

pub mod loader;
pub mod llama;
pub mod mistral;
pub mod registry;
pub mod tokenizer;
pub mod utils;
pub mod config;

// Re-exports
pub use loader::{CandleModelLoader, LoaderConfig};
pub use llama::{LlamaModel, LlamaConfig};
pub use mistral::{MistralModel, MistralConfig};
pub use registry::ModelRegistry;
pub use tokenizer::{TokenizerWrapper, TokenizerConfig};
pub use config::ModelConfigManager;

use ferrum_core::{Error, Result};
