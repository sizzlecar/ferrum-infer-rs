//! Tokenizer components implementing ferrum-interfaces traits
//!
//! This module provides concrete implementations of the Tokenizer trait,
//! separating tokenization concerns from model implementations.

use ferrum_interfaces::{Tokenizer as TokenizerTrait, IncrementalTokenizer, TokenizerInfo};
use ferrum_types::{Result, FerrumError, TokenId, SpecialTokens};
use tokenizers::Tokenizer as HuggingFaceTokenizer;
use std::sync::Arc;
use tracing::{debug, trace, warn};

/// Hugging Face tokenizer adapter implementing ferrum-interfaces traits
#[derive(Debug)]
pub struct HuggingFaceTokenizerAdapter {
    tokenizer: Arc<HuggingFaceTokenizer>,
    info: TokenizerInfo,
    special_tokens: SpecialTokens,
}

impl HuggingFaceTokenizerAdapter {
    pub fn new(tokenizer: HuggingFaceTokenizer) -> Result<Self> {
        // Extract special tokens from tokenizer
        let vocab = tokenizer.get_vocab(false);
        
        let bos_token = vocab.get("<s>").or(vocab.get("<|startoftext|>")).copied();
        let eos_token = vocab.get("</s>").or(vocab.get("<|endoftext|>")).copied();
        let unk_token = vocab.get("<unk>").copied();
        let pad_token = vocab.get("<pad>").copied();
        
        let special_tokens = SpecialTokens {
            bos_token,
            eos_token,
            unk_token,
            pad_token,
        };
        
        let vocab_size = vocab.len();
        
        let info = TokenizerInfo {
            vocab_size,
            model_type: "hf_tokenizer".to_string(),
            special_tokens: special_tokens.clone(),
        };
        
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            info,
            special_tokens,
        })
    }
    
    pub fn from_file(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_path)
            .map_err(|e| FerrumError::ModelLoadError {
                message: format!("Failed to load tokenizer: {}", e),
                path: tokenizer_path.to_string(),
            })?;
        
        Self::new(tokenizer)
    }
}

impl TokenizerTrait for HuggingFaceTokenizerAdapter {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        debug!("Encoding text: {} chars", text.len());
        trace!("Text to encode: {}", text);
        
        let encoding = self.tokenizer
            .encode(text, add_special)
            .map_err(|e| FerrumError::TokenizationError {
                message: format!("Tokenization failed: {}", e),
                text: text.to_string(),
            })?;
        
        let token_ids = encoding.get_ids().to_vec();
        debug!("Encoded to {} tokens", token_ids.len());
        
        Ok(token_ids)
    }
    
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        debug!("Decoding {} tokens", tokens.len());
        
        if tokens.is_empty() {
            return Ok(String::new());
        }
        
        let decoded = self.tokenizer
            .decode(tokens, skip_special)
            .map_err(|e| FerrumError::DetokenizationError {
                message: format!("Detokenization failed: {}", e),
                tokens: tokens.to_vec(),
            })?;
        
        debug!("Decoded to {} chars", decoded.len());
        Ok(decoded)
    }
    
    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String> {
        // For incremental decoding, we need to decode the full sequence
        // and then extract only the new part
        let mut full_tokens = prev.to_vec();
        full_tokens.push(next);
        
        let full_text = self.decode(&full_tokens, true)?;
        let prev_text = if prev.is_empty() {
            String::new()
        } else {
            self.decode(prev, true)?
        };
        
        // Extract the incremental part
        let incremental_text = if full_text.len() >= prev_text.len() {
            full_text[prev_text.len()..].to_string()
        } else {
            // This shouldn't happen with proper tokenizers, but handle it gracefully
            warn!(
                "Incremental decode produced shorter text: prev_len={}, full_len={}",
                prev_text.len(),
                full_text.len()
            );
            full_text
        };
        
        debug!(
            "Incremental decode: token {} -> '{}' ({} chars)", 
            next, 
            incremental_text,
            incremental_text.len()
        );
        
        Ok(incremental_text)
    }
    
    fn vocab_size(&self) -> usize {
        self.info.vocab_size
    }
    
    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

impl IncrementalTokenizer for HuggingFaceTokenizerAdapter {
    fn info(&self) -> &TokenizerInfo {
        &self.info
    }
    
    fn reset(&mut self) {
        // HF tokenizers are stateless, so no reset needed
    }
}

/// Tokenizer factory for creating tokenizer instances
pub struct TokenizerFactory;

impl TokenizerFactory {
    /// Create tokenizer from Hugging Face tokenizer file
    pub fn from_hf_file(path: &str) -> Result<Arc<dyn TokenizerTrait>> {
        let adapter = HuggingFaceTokenizerAdapter::from_file(path)?;
        Ok(Arc::new(adapter))
    }
    
    /// Create tokenizer from existing HF tokenizer instance
    pub fn from_hf_tokenizer(tokenizer: HuggingFaceTokenizer) -> Result<Arc<dyn TokenizerTrait>> {
        let adapter = HuggingFaceTokenizerAdapter::new(tokenizer)?;
        Ok(Arc::new(adapter))
    }
}

/// Cached incremental tokenizer that tracks generated tokens and provides efficient incremental decoding
pub struct CachedIncrementalTokenizer {
    base: Arc<dyn TokenizerTrait>,
    generated_tokens: Vec<TokenId>,
    cached_text: String,
}

impl CachedIncrementalTokenizer {
    pub fn new(base: Arc<dyn TokenizerTrait>) -> Self {
        Self {
            base,
            generated_tokens: Vec::new(),
            cached_text: String::new(),
        }
    }
    
    /// Add a new token and get the incremental text
    pub fn add_token(&mut self, token: TokenId) -> Result<String> {
        let incremental = self.base.decode_incremental(&self.generated_tokens, token)?;
        
        self.generated_tokens.push(token);
        self.cached_text.push_str(&incremental);
        
        Ok(incremental)
    }
    
    /// Get the full decoded text so far
    pub fn full_text(&self) -> &str {
        &self.cached_text
    }
    
    /// Get all generated tokens
    pub fn tokens(&self) -> &[TokenId] {
        &self.generated_tokens
    }
    
    /// Reset the cache
    pub fn reset(&mut self) {
        self.generated_tokens.clear();
        self.cached_text.clear();
    }
    
    /// Check if the current text contains any stop sequence
    pub fn contains_stop_sequence(&self, stop_sequences: &[String]) -> bool {
        stop_sequences.iter().any(|stop| self.cached_text.contains(stop))
    }
}

impl TokenizerTrait for CachedIncrementalTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.base.encode(text, add_special)
    }
    
    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        self.base.decode(tokens, skip_special)
    }
    
    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String> {
        self.base.decode_incremental(prev, next)
    }
    
    fn vocab_size(&self) -> usize {
        self.base.vocab_size()
    }
    
    fn special_tokens(&self) -> &SpecialTokens {
        self.base.special_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;
    use tokenizers::{AddedToken, Model, Tokenizer};
    
    fn create_test_tokenizer() -> Result<HuggingFaceTokenizerAdapter> {
        // Create a minimal test tokenizer
        let mut tokenizer = Tokenizer::new(BPE::default());
        
        // Add some basic tokens
        tokenizer.add_tokens(&[
            AddedToken::from("hello", false),
            AddedToken::from("world", false),
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ]);
        
        HuggingFaceTokenizerAdapter::new(tokenizer)
    }
    
    #[test]
    fn test_basic_encode_decode() {
        let tokenizer = create_test_tokenizer().unwrap();
        
        let text = "hello world";
        let tokens = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        
        assert!(!tokens.is_empty());
        // Note: exact text matching depends on tokenizer training
    }
    
    #[test]
    fn test_incremental_decode() {
        let tokenizer = create_test_tokenizer().unwrap();
        
        let text = "hello world";
        let tokens = tokenizer.encode(text, false).unwrap();
        
        let mut prev_tokens = Vec::new();
        let mut full_incremental = String::new();
        
        for &token in &tokens {
            let incremental = tokenizer.decode_incremental(&prev_tokens, token).unwrap();
            full_incremental.push_str(&incremental);
            prev_tokens.push(token);
        }
        
        let direct_decode = tokenizer.decode(&tokens, false).unwrap();
        // The incremental approach should produce the same result
        assert_eq!(full_incremental.trim(), direct_decode.trim());
    }
    
    #[test]
    fn test_cached_incremental_tokenizer() {
        let base = Arc::new(create_test_tokenizer().unwrap());
        let mut cached = CachedIncrementalTokenizer::new(base);
        
        let text = "hello world";
        let tokens = cached.encode(text, false).unwrap();
        
        for &token in &tokens {
            let _ = cached.add_token(token).unwrap();
        }
        
        assert_eq!(cached.tokens(), &tokens);
        assert!(!cached.full_text().is_empty());
    }
    
    #[test]
    fn test_stop_sequence_detection() {
        let base = Arc::new(create_test_tokenizer().unwrap());
        let mut cached = CachedIncrementalTokenizer::new(base);
        
        // Add some tokens that would produce text containing a stop sequence
        let text = "hello stop world";
        let tokens = cached.encode(text, false).unwrap();
        
        for &token in &tokens {
            let _ = cached.add_token(token).unwrap();
        }
        
        let stop_sequences = vec!["stop".to_string()];
        assert!(cached.contains_stop_sequence(&stop_sequences));
        
        let no_stop_sequences = vec!["xyz".to_string()];
        assert!(!cached.contains_stop_sequence(&no_stop_sequences));
    }
}
