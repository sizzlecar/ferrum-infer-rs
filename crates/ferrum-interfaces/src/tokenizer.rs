//! Tokenizer interface for text encoding/decoding
//!
//! This module provides tokenizer abstractions that are completely separate
//! from model implementations, supporting incremental decoding and various
//! tokenization strategies.

use ferrum_types::{Result, SpecialTokens, TokenId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core tokenizer trait for encoding/decoding operations
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>>;
    
    /// Decode token IDs to text
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String>;
    
    /// Incremental decode: given previous tokens and new token, return only the new text
    /// This is crucial for streaming applications to avoid re-decoding all tokens
    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get special tokens configuration  
    fn special_tokens(&self) -> &SpecialTokens;
    
    /// Get token ID for a specific text (if exists in vocabulary)
    fn token_id(&self, text: &str) -> Option<TokenId>;
    
    /// Get text for a specific token ID
    fn token_text(&self, token_id: TokenId) -> Option<&str>;
    
    /// Check if token is a special token
    fn is_special_token(&self, token_id: TokenId) -> bool {
        let special = self.special_tokens();
        token_id == special.bos_token.unwrap_or(TokenId::MAX)
            || token_id == special.eos_token.unwrap_or(TokenId::MAX)  
            || token_id == special.unk_token.unwrap_or(TokenId::MAX)
            || token_id == special.pad_token.unwrap_or(TokenId::MAX)
    }
    
    /// Apply chat template if supported
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        // Default implementation: just concatenate messages
        let mut result = String::new();
        for msg in messages {
            result.push_str(&format!("{}: {}\n", msg.role, msg.content));
        }
        Ok(result.trim_end().to_string())
    }
    
    /// Get tokenizer information
    fn info(&self) -> TokenizerInfo;
}

/// Asynchronous tokenizer operations for I/O-bound tokenization
#[async_trait]
pub trait AsyncTokenizer: Tokenizer {
    /// Asynchronous encoding (useful for very large texts)
    async fn encode_async(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>>;
    
    /// Asynchronous decoding
    async fn decode_async(&self, tokens: &[TokenId], skip_special: bool) -> Result<String>;
    
    /// Batch encoding for multiple texts
    async fn encode_batch(&self, texts: &[&str], add_special: bool) -> Result<Vec<Vec<TokenId>>>;
    
    /// Batch decoding for multiple token sequences
    async fn decode_batch(&self, token_sequences: &[&[TokenId]], skip_special: bool) -> Result<Vec<String>>;
}

/// Advanced tokenizer capabilities
pub trait TokenizerCapabilities: Tokenizer {
    /// Get token probability/likelihood for text
    fn token_probability(&self, text: &str, token_id: TokenId) -> Option<f32>;
    
    /// Get all possible tokens for a prefix
    fn get_prefix_tokens(&self, prefix: &str) -> Result<Vec<TokenId>>;
    
    /// Check if sequence can be extended with token
    fn can_extend(&self, tokens: &[TokenId], next_token: TokenId) -> bool;
    
    /// Get token type (word, subword, punctuation, etc.)
    fn token_type(&self, token_id: TokenId) -> TokenType;
    
    /// Normalize text before tokenization
    fn normalize_text(&self, text: &str) -> String;
    
    /// Pre-tokenize text (split into words/subwords)
    fn pre_tokenize(&self, text: &str) -> Vec<String>;
}

/// Tokenizer factory for creating tokenizer instances
#[async_trait]
pub trait TokenizerFactory: Send + Sync {
    /// Load tokenizer from file path
    async fn load_from_file(&self, path: &str) -> Result<Box<dyn Tokenizer>>;
    
    /// Load tokenizer from bytes
    async fn load_from_bytes(&self, data: &[u8]) -> Result<Box<dyn Tokenizer>>;
    
    /// Load tokenizer from Hugging Face Hub
    async fn load_from_hub(
        &self, 
        repo_id: &str, 
        revision: Option<&str>
    ) -> Result<Box<dyn Tokenizer>>;
    
    /// Create tokenizer from configuration
    async fn create_from_config(&self, config: &TokenizerConfig) -> Result<Box<dyn Tokenizer>>;
    
    /// Get supported tokenizer types
    fn supported_types(&self) -> Vec<TokenizerType>;
}

/// Tokenizer information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerInfo {
    /// Tokenizer type/algorithm
    pub tokenizer_type: TokenizerType,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Special tokens
    pub special_tokens: SpecialTokens,
    /// Whether tokenizer supports incremental decoding efficiently
    pub supports_incremental: bool,
    /// Whether tokenizer supports chat templates
    pub supports_chat_template: bool,
    /// Maximum token length
    pub max_token_length: Option<usize>,
    /// Model name or identifier this tokenizer was trained for
    pub model_name: Option<String>,
}

/// Tokenizer types/algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerType {
    /// Byte-Pair Encoding
    BPE,
    /// WordPiece (BERT-style)
    WordPiece,
    /// SentencePiece 
    SentencePiece,
    /// Tiktoken (GPT family)
    Tiktoken,
    /// Custom implementation
    Custom,
}

/// Token types for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    /// Regular word token
    Word,
    /// Subword token  
    Subword,
    /// Punctuation token
    Punctuation,
    /// Number token
    Number,
    /// Special/control token
    Special,
    /// Unknown token
    Unknown,
}

/// Chat message for template application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role (user, assistant, system, etc.)
    pub role: String,
    /// Message content
    pub content: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ChatMessage {
    /// Create user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(), 
            content: content.into(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(), 
            metadata: HashMap::new(),
        }
    }
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]  
pub struct TokenizerConfig {
    /// Tokenizer type
    pub tokenizer_type: TokenizerType,
    /// Path to tokenizer files
    pub path: String,
    /// Whether to add special tokens during encoding
    pub add_special_tokens: bool,
    /// Whether to use fast tokenization (if available)
    pub use_fast: bool,
    /// Truncation configuration
    pub truncation: Option<TruncationConfig>,
    /// Padding configuration
    pub padding: Option<PaddingConfig>,
    /// Chat template (if any)
    pub chat_template: Option<String>,
    /// Additional tokenizer-specific options
    pub extra_options: HashMap<String, serde_json::Value>,
}

/// Truncation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
    /// Stride for sliding window truncation
    pub stride: Option<usize>,
}

/// Truncation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Remove tokens from the end
    TruncateEnd,
    /// Remove tokens from the beginning  
    TruncateStart,
    /// Remove tokens from both ends equally
    TruncateBoth,
    /// Sliding window approach
    SlidingWindow,
}

/// Padding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingConfig {
    /// Padding strategy
    pub strategy: PaddingStrategy,
    /// Padding token ID
    pub token_id: TokenId,
    /// Target length (if fixed padding)
    pub length: Option<usize>,
    /// Padding direction
    pub direction: PaddingDirection,
}

/// Padding strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to longest sequence in batch
    Longest,
    /// Pad to multiple of specified value
    MultipleOf(usize),
    /// Pad to fixed length
    Fixed,
}

/// Padding direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingDirection {
    /// Pad on the right
    Right,
    /// Pad on the left
    Left,
}

/// Incremental tokenizer state for streaming
pub trait IncrementalTokenizer: Tokenizer {
    /// Tokenizer state for incremental operations
    type State: Send + Sync;
    
    /// Create initial state for incremental decoding
    fn create_state(&self) -> Self::State;
    
    /// Add token to state and get incremental text
    fn decode_incremental_with_state(
        &self, 
        state: &mut Self::State, 
        token: TokenId
    ) -> Result<String>;
    
    /// Reset state to initial condition
    fn reset_state(&self, state: &mut Self::State);
    
    /// Get all decoded text from current state
    fn get_decoded_text(&self, state: &Self::State) -> String;
}

/// Text processing utilities
pub trait TextProcessor: Send + Sync {
    /// Clean and normalize text for tokenization
    fn preprocess(&self, text: &str) -> String;
    
    /// Post-process decoded text
    fn postprocess(&self, text: &str) -> String;
    
    /// Detect language of text (if supported)
    fn detect_language(&self, text: &str) -> Option<String>;
    
    /// Split text into sentences
    fn sentence_split(&self, text: &str) -> Vec<String>;
    
    /// Count approximate tokens without full tokenization
    fn estimate_token_count(&self, text: &str) -> usize;
}

/// Tokenizer performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerStats {
    /// Total encoding operations
    pub encode_operations: u64,
    /// Total decoding operations  
    pub decode_operations: u64,
    /// Total tokens processed
    pub tokens_processed: u64,
    /// Average encoding time per character (microseconds)
    pub avg_encode_time_per_char_us: f64,
    /// Average decoding time per token (microseconds) 
    pub avg_decode_time_per_token_us: f64,
    /// Cache hit rate for incremental decoding
    pub incremental_cache_hit_rate: f32,
}

/// Tokenizer registry for managing multiple tokenizers
pub trait TokenizerRegistry: Send + Sync {
    /// Register a tokenizer with a name
    fn register(&mut self, name: &str, tokenizer: Box<dyn Tokenizer>) -> Result<()>;
    
    /// Get tokenizer by name
    fn get(&self, name: &str) -> Option<&dyn Tokenizer>;
    
    /// Remove tokenizer by name
    fn remove(&mut self, name: &str) -> Option<Box<dyn Tokenizer>>;
    
    /// List all registered tokenizer names
    fn list_names(&self) -> Vec<String>;
    
    /// Check if tokenizer exists
    fn contains(&self, name: &str) -> bool;
}
