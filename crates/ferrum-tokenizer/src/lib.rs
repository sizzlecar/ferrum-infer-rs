//! # Ferrum Tokenizer
//!
//! Unified tokenizer implementations with incremental decoding and caching support.
//!
//! ## Overview
//!
//! This crate provides implementations of the tokenizer interfaces defined in
//! ferrum-interfaces, including:
//!
//! - HuggingFace tokenizers integration
//! - Incremental decoding for streaming output
//! - Tokenizer caching and sharing
//! - Chat template support
//! - Special token handling
//!
//! ## Features
//!
//! - **Incremental Decoding**: Efficiently decode tokens one by one without full re-decode
//! - **Caching**: Share tokenizers across multiple requests and models
//! - **Chat Templates**: Support for conversation formatting
//! - **Special Tokens**: Proper handling of BOS, EOS, PAD, and other special tokens

pub mod implementations;
pub mod cache;
pub mod templates;

// Re-export interfaces
pub use ferrum_interfaces::{
    Tokenizer, TokenizerFactory, TokenizerInfo, IncrementalTokenizer,
};

pub use ferrum_types::{
    TokenId, SpecialTokens, Result,
};

// Re-export implementations
pub use implementations::*;
pub use cache::*;
