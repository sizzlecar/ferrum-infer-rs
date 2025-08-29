//! Tokenizer wrapper for different tokenizer types

use ferrum_core::{TokenId, Result, Error};
use tokenizers::{Tokenizer, EncodeInput};
use std::path::Path;
use tracing::{info, debug};

/// Tokenizer configuration
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub model_id: String,
    pub add_special_tokens: bool,
    pub padding: bool,
    pub truncation: bool,
    pub max_length: Option<usize>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            add_special_tokens: true,
            padding: false,
            truncation: true,
            max_length: Some(2048),
        }
    }
}

/// Tokenizer wrapper
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    config: TokenizerConfig,
    special_tokens: SpecialTokens,
}

/// Special tokens
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub unk_token: Option<String>,
    pub bos_token_id: Option<TokenId>,
    pub eos_token_id: Option<TokenId>,
    pub pad_token_id: Option<TokenId>,
    pub unk_token_id: Option<TokenId>,
}

impl TokenizerWrapper {
    /// Create from file
    pub fn from_file<P: AsRef<Path>>(path: P, config: TokenizerConfig) -> Result<Self> {
        info!("Loading tokenizer from {:?}", path.as_ref());
        
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| Error::model_loading(format!("Failed to load tokenizer: {}", e)))?;
        
        // Extract special tokens
        let special_tokens = Self::extract_special_tokens(&tokenizer);
        
        Ok(Self {
            tokenizer,
            config,
            special_tokens,
        })
    }
    
    /// Create from pretrained model
    pub async fn from_pretrained(model_id: &str, config: TokenizerConfig) -> Result<Self> {
        info!("Loading tokenizer for model: {}", model_id);
        
        // Use HuggingFace hub to download tokenizer
        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(model_id.to_string());
        
        let tokenizer_file = repo.get("tokenizer.json").await
            .map_err(|e| Error::model_loading(format!("Failed to download tokenizer: {}", e)))?;
        
        Self::from_file(tokenizer_file, config)
    }
    
    /// Extract special tokens from tokenizer
    fn extract_special_tokens(tokenizer: &Tokenizer) -> SpecialTokens {
        let vocab = tokenizer.get_vocab(false);
        
        // Common special tokens
        let bos_token = vec!["<s>", "<|startoftext|>", "<|begin_of_text|>"]
            .iter()
            .find(|&&t| vocab.contains_key(t))
            .map(|&s| s.to_string());
        
        let eos_token = vec!["</s>", "<|endoftext|>", "<|end_of_text|>"]
            .iter()
            .find(|&&t| vocab.contains_key(t))
            .map(|&s| s.to_string());
        
        let pad_token = vec!["<pad>", "[PAD]"]
            .iter()
            .find(|&&t| vocab.contains_key(t))
            .map(|&s| s.to_string());
        
        let unk_token = vec!["<unk>", "[UNK]"]
            .iter()
            .find(|&&t| vocab.contains_key(t))
            .map(|&s| s.to_string());
        
        // Get token IDs
        let bos_token_id = bos_token.as_ref()
            .and_then(|t| tokenizer.token_to_id(t));
        let eos_token_id = eos_token.as_ref()
            .and_then(|t| tokenizer.token_to_id(t));
        let pad_token_id = pad_token.as_ref()
            .and_then(|t| tokenizer.token_to_id(t));
        let unk_token_id = unk_token.as_ref()
            .and_then(|t| tokenizer.token_to_id(t));
        
        SpecialTokens {
            bos_token,
            eos_token,
            pad_token,
            unk_token,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            unk_token_id,
        }
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        debug!("Encoding text of length: {}", text.len());
        
        let encoding = self.tokenizer
            .encode(text, self.config.add_special_tokens)
            .map_err(|e| Error::model_execution(format!("Tokenization failed: {}", e)))?;
        
        let mut token_ids = encoding.get_ids().to_vec();
        
        // Apply truncation if needed
        if self.config.truncation {
            if let Some(max_length) = self.config.max_length {
                if token_ids.len() > max_length {
                    token_ids.truncate(max_length);
                    // Add EOS token if truncated
                    if let Some(eos_id) = self.special_tokens.eos_token_id {
                        if let Some(last) = token_ids.last_mut() {
                            *last = eos_id;
                        }
                    }
                }
            }
        }
        
        Ok(token_ids)
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[TokenId]) -> Result<String> {
        debug!("Decoding {} tokens", token_ids.len());
        
        self.tokenizer
            .decode(token_ids, !self.config.add_special_tokens)
            .map_err(|e| Error::model_execution(format!("Detokenization failed: {}", e)))
    }
    
    /// Batch encode texts
    pub fn encode_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<TokenId>>> {
        let inputs: Vec<EncodeInput> = texts
            .into_iter()
            .map(|text| text.into())
            .collect();
        
        let encodings = self.tokenizer
            .encode_batch(inputs, self.config.add_special_tokens)
            .map_err(|e| Error::model_execution(format!("Batch tokenization failed: {}", e)))?;
        
        Ok(encodings.into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect())
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }
    
    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
    
    /// Convert token ID to string
    pub fn id_to_token(&self, id: TokenId) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
    
    /// Convert string to token ID
    pub fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.tokenizer.token_to_id(token)
    }
}
