//! Abstract tokenizer wrapper
//!
//! This module provides a framework-agnostic tokenizer interface
//! that can work with different tokenizer backends.

use crate::traits::{Tokenizer as TokenizerTrait, SpecialTokens};
use ferrum_core::{Result, Error};
use async_trait::async_trait;
use std::path::Path;

/// Tokenizer wrapper that uses the tokenizers library
pub struct TokenizerWrapper {
    /// Inner tokenizer from the tokenizers crate
    inner: tokenizers::Tokenizer,
    
    /// Special tokens configuration
    special_tokens: SpecialTokens,
}

impl TokenizerWrapper {
    /// Load tokenizer from file
    pub async fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| Error::model_loading(format!("Failed to load tokenizer: {}", e)))?;
            
        let special_tokens = Self::extract_special_tokens(&tokenizer);
        
        Ok(Self {
            inner: tokenizer,
            special_tokens,
        })
    }
    
    /// Load tokenizer from HuggingFace
    pub async fn from_pretrained(_model_id: &str, _revision: Option<&str>) -> Result<Self> {
        // This would download from HuggingFace
        // For now, return error
        Err(Error::unsupported("HuggingFace tokenizer loading not yet implemented"))
    }
    
    /// Extract special tokens from tokenizer
    fn extract_special_tokens(tokenizer: &tokenizers::Tokenizer) -> SpecialTokens {
        // Extract special tokens from the tokenizer
        // This is a simplified version
        SpecialTokens {
            bos_token_id: tokenizer.token_to_id("<s>").or(tokenizer.token_to_id("<|startoftext|>")),
            eos_token_id: tokenizer.token_to_id("</s>").or(tokenizer.token_to_id("<|endoftext|>")),
            pad_token_id: tokenizer.token_to_id("<pad>"),
            unk_token_id: tokenizer.token_to_id("<unk>"),
        }
    }
}

#[async_trait]
impl TokenizerTrait for TokenizerWrapper {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::internal(format!("Tokenization failed: {}", e)))?;
            
        Ok(encoding.get_ids().to_vec())
    }
    
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special_tokens)
            .map_err(|e| Error::internal(format!("Decoding failed: {}", e)))
    }
    
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
    
    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

/// Factory for creating tokenizers
pub struct TokenizerFactory;

impl TokenizerFactory {
    /// Create tokenizer for a specific model architecture
    pub async fn create(
        architecture: &crate::Architecture,
        tokenizer_path: Option<&Path>,
    ) -> Result<Box<dyn TokenizerTrait>> {
        match architecture {
            crate::Architecture::Llama | 
            crate::Architecture::Llama2 | 
            crate::Architecture::Llama3 => {
                // Llama uses sentencepiece tokenizer
                if let Some(path) = tokenizer_path {
                    Ok(Box::new(TokenizerWrapper::from_file(path).await?))
                } else {
                    Err(Error::configuration("Tokenizer path required"))
                }
            }
            crate::Architecture::Mistral |
            crate::Architecture::Mixtral => {
                // Mistral also uses sentencepiece
                if let Some(path) = tokenizer_path {
                    Ok(Box::new(TokenizerWrapper::from_file(path).await?))
                } else {
                    Err(Error::configuration("Tokenizer path required"))
                }
            }
            crate::Architecture::Qwen |
            crate::Architecture::Qwen2 => {
                // Qwen uses tiktoken-based tokenizer
                if let Some(path) = tokenizer_path {
                    Ok(Box::new(TokenizerWrapper::from_file(path).await?))
                } else {
                    Err(Error::configuration("Tokenizer path required"))
                }
            }
            _ => {
                // Default: try to load from file
                if let Some(path) = tokenizer_path {
                    Ok(Box::new(TokenizerWrapper::from_file(path).await?))
                } else {
                    Err(Error::unsupported(format!("Unsupported architecture: {:?}", architecture)))
                }
            }
        }
    }
}