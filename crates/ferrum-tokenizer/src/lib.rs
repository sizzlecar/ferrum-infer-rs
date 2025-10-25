//! # Ferrum Tokenizer
//!
//! MVP tokenizer implementation for Ferrum inference stack.
//!
//! This crate provides HuggingFace tokenizers integration and implements
//! the tokenizer interfaces defined in `ferrum-interfaces`.
//!
//! ## Features
//!
//! - **HuggingFace Integration**: Load tokenizers from HF Hub or local files
//! - **Incremental Decoding**: Efficient token-by-token decoding for streaming
//! - **Chat Templates**: Support for conversation formatting (basic)
//! - **Special Tokens**: Proper handling of BOS, EOS, PAD tokens

pub mod implementations;

// Re-export interface types
pub use ferrum_interfaces::{
    tokenizer::TokenizerType, IncrementalTokenizer, Tokenizer, TokenizerFactory, TokenizerInfo,
};

pub use ferrum_types::{Result, SpecialTokens, TokenId};

// Re-export implementations
pub use implementations::*;

/// Default tokenizer factory using HuggingFace backend
pub fn default_factory() -> HuggingFaceTokenizerFactory {
    HuggingFaceTokenizerFactory::new()
}

/// Load tokenizer from file
pub async fn load_from_file(path: &str) -> Result<Box<dyn Tokenizer>> {
    default_factory().load_from_file(path).await
}

/// Load tokenizer from HuggingFace Hub
pub async fn load_from_hub(repo_id: &str, revision: Option<&str>) -> Result<Box<dyn Tokenizer>> {
    default_factory().load_from_hub(repo_id, revision).await
}
