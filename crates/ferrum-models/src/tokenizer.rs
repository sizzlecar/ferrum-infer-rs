//! Abstract tokenizer wrapper
//!
//! This module provides a framework-agnostic tokenizer interface
//! that can work with different tokenizer backends.
//! Extended with vLLM-inspired multi-mode support and LRU caching.

use crate::traits::{SpecialTokens, Tokenizer as TokenizerTrait};
use crate::source::{ModelFormat, ResolvedModelSource};
use async_trait::async_trait;
use ferrum_core::{Error, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, warn};

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

    /// Load tokenizer from HuggingFace (placeholder)
    pub async fn from_pretrained(_model_id: &str, _revision: Option<&str>) -> Result<Self> {
        // This would download from HuggingFace
        // For now, return error
        Err(Error::unsupported(
            "HuggingFace tokenizer loading not yet implemented - use load_from_source",
        ))
    }
    
    /// Load tokenizer from resolved model source
    pub async fn from_source(source: &ResolvedModelSource) -> Result<Self> {
        let tokenizer_path = match source.format {
            ModelFormat::HuggingFace => source.local_path.join("tokenizer.json"),
            ModelFormat::Mistral => source.local_path.join("tokenizer.json"), // Mistral also uses HF format
            ModelFormat::GGUF => {
                // For GGUF, tokenizer might be embedded or in same directory
                let dir = source.local_path.parent().unwrap_or(&source.local_path);
                dir.join("tokenizer.json")
            }
            ModelFormat::Auto => {
                // Try to find tokenizer.json
                let tokenizer_file = source.local_path.join("tokenizer.json");
                if tokenizer_file.exists() {
                    tokenizer_file
                } else {
                    return Err(Error::model_loading("Cannot find tokenizer.json for auto-detected model"));
                }
            }
        };
        
        debug!("Loading tokenizer from: {:?}", tokenizer_path);
        Self::from_file(&tokenizer_path).await
    }

    /// Extract special tokens from tokenizer
    fn extract_special_tokens(tokenizer: &tokenizers::Tokenizer) -> SpecialTokens {
        // Extract special tokens from the tokenizer
        // This is a simplified version
        SpecialTokens {
            bos_token_id: tokenizer
                .token_to_id("<s>")
                .or(tokenizer.token_to_id("<|startoftext|>")),
            eos_token_id: tokenizer
                .token_to_id("</s>")
                .or(tokenizer.token_to_id("<|endoftext|>")),
            pad_token_id: tokenizer.token_to_id("<pad>"),
            unk_token_id: tokenizer.token_to_id("<unk>"),
        }
    }
}

#[async_trait]
impl TokenizerTrait for TokenizerWrapper {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
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

/// Tokenizer mode (vLLM-inspired)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerMode {
    /// Auto-detect tokenizer type
    Auto,
    /// Use Mistral tokenizer
    Mistral,
    /// Use custom tokenizer
    Custom(String),
}

/// Factory for creating tokenizers with caching and multi-mode support
pub struct TokenizerFactory {
    /// Cached tokenizers (model_id -> tokenizer)
    cache: Arc<RwLock<HashMap<String, Arc<dyn TokenizerTrait>>>>,
    /// Default tokenizer mode
    default_mode: TokenizerMode,
}

impl TokenizerFactory {
    /// Create new tokenizer factory
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_mode: TokenizerMode::Auto,
        }
    }
    
    /// Create tokenizer factory with specific default mode
    pub fn with_mode(mode: TokenizerMode) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_mode: mode,
        }
    }
    
    /// Create tokenizer from resolved model source (vLLM-inspired)
    pub async fn create_from_source(
        &self,
        source: &ResolvedModelSource,
        mode: Option<TokenizerMode>,
    ) -> Result<Arc<dyn TokenizerTrait>> {
        let cache_key = format!("{}:{}", source.model_id, source.revision.as_deref().unwrap_or("main"));
        
        // Check cache first
        if let Some(cached) = self.cache.read().get(&cache_key) {
            debug!("Using cached tokenizer for {}", cache_key);
            return Ok(cached.clone());
        }
        
        let tokenizer_mode = mode.unwrap_or_else(|| self.default_mode.clone());
        let tokenizer = self.create_tokenizer_for_mode(&tokenizer_mode, source).await?;
        
        // Cache the result
        self.cache.write().insert(cache_key, tokenizer.clone());
        Ok(tokenizer)
    }
    
    /// Create tokenizer for specific mode
    async fn create_tokenizer_for_mode(
        &self,
        mode: &TokenizerMode,
        source: &ResolvedModelSource,
    ) -> Result<Arc<dyn TokenizerTrait>> {
        debug!("Creating tokenizer with mode: {:?}", mode);
        
        match mode {
            TokenizerMode::Auto => {
                // Auto-detect based on source format and architecture
                self.auto_create_tokenizer(source).await
            }
            TokenizerMode::Mistral => {
                // Use Mistral-specific tokenizer handling
                self.create_mistral_tokenizer(source).await
            }
            TokenizerMode::Custom(custom_type) => {
                // Handle custom tokenizer types
                warn!("Custom tokenizer type '{}' not implemented, falling back to auto", custom_type);
                self.auto_create_tokenizer(source).await
            }
        }
    }
    
    /// Auto-detect and create appropriate tokenizer
    async fn auto_create_tokenizer(&self, source: &ResolvedModelSource) -> Result<Arc<dyn TokenizerTrait>> {
        let tokenizer = TokenizerWrapper::from_source(source).await?;
        Ok(Arc::new(tokenizer))
    }
    
    /// Create Mistral-specific tokenizer
    async fn create_mistral_tokenizer(&self, source: &ResolvedModelSource) -> Result<Arc<dyn TokenizerTrait>> {
        // Mistral uses similar tokenizer format but may have special handling
        debug!("Creating Mistral tokenizer");
        let tokenizer = TokenizerWrapper::from_source(source).await?;
        Ok(Arc::new(tokenizer))
    }
    
    /// Clear tokenizer cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
    
    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }
    
    /// Create tokenizer for a specific model architecture (legacy method)
    pub async fn create(
        architecture: &crate::Architecture,
        tokenizer_path: Option<&Path>,
    ) -> Result<Box<dyn TokenizerTrait>> {
        match architecture {
            crate::Architecture::Llama
            | crate::Architecture::Llama2
            | crate::Architecture::Llama3 => {
                // Llama uses sentencepiece tokenizer
                if let Some(path) = tokenizer_path {
                    Ok(Box::new(TokenizerWrapper::from_file(path).await?))
                } else {
                    Err(Error::configuration("Tokenizer path required"))
                }
            }
            crate::Architecture::Mistral | crate::Architecture::Mixtral => {
                // Mistral also uses sentencepiece
                if let Some(path) = tokenizer_path {
                    Ok(Box::new(TokenizerWrapper::from_file(path).await?))
                } else {
                    Err(Error::configuration("Tokenizer path required"))
                }
            }
            crate::Architecture::Qwen | crate::Architecture::Qwen2 => {
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
                    Err(Error::unsupported(format!(
                        "Unsupported architecture: {:?}",
                        architecture
                    )))
                }
            }
        }
    }
}

impl Default for TokenizerFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_tokenizer_wrapper_from_source_missing_file_errors() {
        let tmp = TempDir::new().unwrap();
        let source = ResolvedModelSource {
            local_path: tmp.path().to_path_buf(),
            format: ModelFormat::HuggingFace,
            from_cache: true,
            revision: Some("main".to_string()),
            model_id: "org/model".to_string(),
        };

        let err = TokenizerWrapper::from_source(&source).await.err().unwrap();
        let msg = format!("{}", err);
        assert!(msg.to_lowercase().contains("tokenizer"));
    }

    #[tokio::test]
    async fn test_tokenizer_factory_cache_key() {
        let tmp = TempDir::new().unwrap();
        // write a minimal tokenizer.json that tokenizers can load
        // Use a trivial BPE with empty merges/vocab which tokenizers still parses as a structure
        // For stability, provide an actually valid json structure accepted by tokenizers
        let tokenizer_json = r#"{
            "version":"1.0",
            "truncation":null,
            "padding":null,
            "added_tokens":[],
            "normalizer": null,
            "pre_tokenizer": {"type":"Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type":"WordLevel","vocab":{"<unk>":0},"unk_token":"<unk>"}
        }"#;
        std::fs::write(tmp.path().join("tokenizer.json"), tokenizer_json).unwrap();

        let source = ResolvedModelSource {
            local_path: tmp.path().to_path_buf(),
            format: ModelFormat::HuggingFace,
            from_cache: true,
            revision: Some("r1".to_string()),
            model_id: "org/model".to_string(),
        };

        let factory = TokenizerFactory::new();
        let t1 = factory.create_from_source(&source, None).await.unwrap();
        assert_eq!(factory.cache_size(), 1);

        let t2 = factory.create_from_source(&source, None).await.unwrap();
        assert!(Arc::ptr_eq(&t1, &t2));
        assert_eq!(factory.cache_size(), 1);
    }
}

/// Cached tokenizer wrapper to avoid repeated property computation
pub struct CachedTokenizer<T: TokenizerTrait> {
    inner: T,
    vocab_size: usize,
    special_tokens: SpecialTokens,
}

impl<T: TokenizerTrait> CachedTokenizer<T> {
    pub fn new(inner: T) -> Self {
        let vocab_size = inner.vocab_size();
        let special_tokens = inner.special_tokens().clone();
        
        Self {
            inner,
            vocab_size,
            special_tokens,
        }
    }
}

#[async_trait]
impl<T: TokenizerTrait> TokenizerTrait for CachedTokenizer<T> {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner.encode(text, add_special_tokens)
    }

    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(tokens, skip_special_tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}
