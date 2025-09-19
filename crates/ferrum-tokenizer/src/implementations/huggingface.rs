//! HuggingFace tokenizer implementation

use crate::{IncrementalTokenizer, Tokenizer, TokenizerInfo};
use ferrum_types::{Result, SpecialTokens, TokenId};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::{collections::HashMap, sync::Arc};
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, warn};

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    /// Internal HF tokenizer
    tokenizer: Arc<HfTokenizer>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Tokenizer info
    info: TokenizerInfo,
    /// Incremental decode cache
    decode_cache: RwLock<DecodeCache>,
}

/// Cache for incremental decoding
#[derive(Debug, Default)]
struct DecodeCache {
    /// Cache of (prefix_tokens, decoded_text) pairs
    cache: HashMap<Vec<TokenId>, String>,
    /// Maximum cache size
    max_size: usize,
}

impl DecodeCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    fn get(&self, tokens: &[TokenId]) -> Option<&String> {
        self.cache.get(tokens)
    }

    fn insert(&mut self, tokens: Vec<TokenId>, text: String) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: clear half the cache
            let to_remove: Vec<_> = self.cache.keys().take(self.cache.len() / 2).cloned().collect();
            for key in to_remove {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(tokens, text);
    }

    fn clear(&mut self) {
        self.cache.clear();
    }
}

impl HuggingFaceTokenizer {
    /// Create new HuggingFace tokenizer
    pub async fn new(tokenizer: HfTokenizer) -> Result<Self> {
        let vocab_size = tokenizer.get_vocab_size(false) as usize;
        
        // Extract special tokens
        let special_tokens = extract_special_tokens(&tokenizer)?;
        
        let info = TokenizerInfo {
            name: "huggingface".to_string(),
            vocab_size,
            special_tokens: special_tokens.clone(),
            supports_incremental: true,
            model_max_length: tokenizer.get_model_max_length(),
            padding_side: "left".to_string(), // Default
        };
        
        debug!("Created HuggingFace tokenizer with vocab size {}", vocab_size);
        
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens,
            info,
            decode_cache: RwLock::new(DecodeCache::new(1000)), // Cache last 1000 entries
        })
    }

    /// Create from file
    pub async fn from_file(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(tokenizer_path)
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Failed to load tokenizer: {}", e)))?;
        Self::new(tokenizer).await
    }

    /// Create from HuggingFace Hub
    pub async fn from_pretrained(
        repo_id: &str,
        revision: Option<&str>,
        auth_token: Option<&str>,
    ) -> Result<Self> {
        let api = hf_hub::api::tokio::Api::new()
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Failed to create HF API: {}", e)))?;
        
        let mut repo = api.repo(hf_hub::Repo::model(repo_id.to_string()));
        
        if let Some(rev) = revision {
            repo = repo.revision(rev.to_string());
        }
        
        let tokenizer_file = repo.get("tokenizer.json").await
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Failed to download tokenizer: {}", e)))?;
        
        let tokenizer = HfTokenizer::from_file(&tokenizer_file)
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Failed to load tokenizer: {}", e)))?;
        
        Self::new(tokenizer).await
    }

    /// Find longest matching prefix in cache
    fn find_cached_prefix(&self, tokens: &[TokenId]) -> Option<(usize, String)> {
        let cache = self.decode_cache.read();
        
        let mut best_match = None;
        let mut best_len = 0;
        
        for (cached_tokens, cached_text) in cache.cache.iter() {
            if tokens.starts_with(cached_tokens) && cached_tokens.len() > best_len {
                best_match = Some((cached_tokens.len(), cached_text.clone()));
                best_len = cached_tokens.len();
            }
        }
        
        best_match
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        let encoding = self.tokenizer.encode(text, add_special)
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Encoding failed: {}", e)))?;
        
        Ok(encoding.get_ids().iter().map(|&id| TokenId::new(id)).collect())
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let token_ids: Vec<u32> = tokens.iter().map(|t| t.value()).collect();
        
        let text = self.tokenizer.decode(&token_ids, skip_special)
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Decoding failed: {}", e)))?;
        
        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.info.vocab_size
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn info(&self) -> &TokenizerInfo {
        &self.info
    }
}

impl IncrementalTokenizer for HuggingFaceTokenizer {
    fn decode_incremental(&self, prev_tokens: &[TokenId], new_token: TokenId) -> Result<String> {
        // Try to use cache for efficiency
        if let Some((prefix_len, prefix_text)) = self.find_cached_prefix(prev_tokens) {
            if prefix_len == prev_tokens.len() {
                // We have the full prefix cached, just decode the new token
                let new_token_text = self.decode(&[new_token], true)?;
                
                // Cache the new state
                let mut new_tokens = prev_tokens.to_vec();
                new_tokens.push(new_token);
                let full_text = format!("{}{}", prefix_text, new_token_text);
                
                {
                    let mut cache = self.decode_cache.write();
                    cache.insert(new_tokens, full_text.clone());
                }
                
                return Ok(new_token_text);
            } else if prefix_len < prev_tokens.len() {
                // We have a partial prefix cached, decode the rest
                let remaining_tokens = &prev_tokens[prefix_len..];
                let mut all_new_tokens = remaining_tokens.to_vec();
                all_new_tokens.push(new_token);
                
                let new_text = self.decode(&all_new_tokens, true)?;
                
                // Cache the full state
                let mut full_tokens = prev_tokens.to_vec();
                full_tokens.push(new_token);
                let full_text = format!("{}{}", prefix_text, new_text);
                
                {
                    let mut cache = self.decode_cache.write();
                    cache.insert(full_tokens, full_text.clone());
                }
                
                return Ok(new_text);
            }
        }
        
        // No cache hit, decode everything
        let mut all_tokens = prev_tokens.to_vec();
        all_tokens.push(new_token);
        
        let full_text = self.decode(&all_tokens, true)?;
        let prev_text = if prev_tokens.is_empty() {
            String::new()
        } else {
            self.decode(prev_tokens, true)?
        };
        
        // Cache both the previous state and new state
        {
            let mut cache = self.decode_cache.write();
            if !prev_tokens.is_empty() {
                cache.insert(prev_tokens.to_vec(), prev_text.clone());
            }
            cache.insert(all_tokens, full_text.clone());
        }
        
        // Return the incremental part
        if full_text.starts_with(&prev_text) {
            Ok(full_text[prev_text.len()..].to_string())
        } else {
            // Fallback: something went wrong, return the new token only
            warn!("Incremental decode fallback for token {:?}", new_token);
            self.decode(&[new_token], true)
        }
    }

    fn supports_incremental(&self) -> bool {
        true
    }

    fn clear_cache(&self) {
        let mut cache = self.decode_cache.write();
        cache.clear();
        debug!("Cleared tokenizer decode cache");
    }
}

/// HuggingFace tokenizer factory
pub struct HuggingFaceTokenizerFactory;

#[async_trait]
impl ferrum_interfaces::TokenizerFactory for HuggingFaceTokenizerFactory {
    async fn create_from_file(&self, path: &str) -> Result<Box<dyn Tokenizer + Send + Sync>> {
        let tokenizer = HuggingFaceTokenizer::from_file(path).await?;
        Ok(Box::new(tokenizer))
    }

    async fn create_from_pretrained(
        &self,
        repo_id: &str,
        config: Option<&ferrum_interfaces::tokenizer::TokenizerConfig>,
    ) -> Result<Box<dyn Tokenizer + Send + Sync>> {
        let revision = config.and_then(|c| c.revision.as_deref());
        let auth_token = config.and_then(|c| c.auth_token.as_deref());
        
        let tokenizer = HuggingFaceTokenizer::from_pretrained(repo_id, revision, auth_token).await?;
        Ok(Box::new(tokenizer))
    }

    async fn create_incremental_from_file(&self, path: &str) -> Result<Box<dyn IncrementalTokenizer + Send + Sync>> {
        let tokenizer = HuggingFaceTokenizer::from_file(path).await?;
        Ok(Box::new(tokenizer))
    }

    async fn create_incremental_from_pretrained(
        &self,
        repo_id: &str,
        config: Option<&ferrum_interfaces::tokenizer::TokenizerConfig>,
    ) -> Result<Box<dyn IncrementalTokenizer + Send + Sync>> {
        let revision = config.and_then(|c| c.revision.as_deref());
        let auth_token = config.and_then(|c| c.auth_token.as_deref());
        
        let tokenizer = HuggingFaceTokenizer::from_pretrained(repo_id, revision, auth_token).await?;
        Ok(Box::new(tokenizer))
    }

    fn supported_formats(&self) -> Vec<String> {
        vec!["huggingface".to_string(), "tokenizers".to_string()]
    }

    fn name(&self) -> &str {
        "huggingface"
    }
}

/// Extract special tokens from HuggingFace tokenizer
fn extract_special_tokens(tokenizer: &HfTokenizer) -> Result<SpecialTokens> {
    let vocab = tokenizer.get_vocab(false);
    
    // Common special token patterns
    let bos_token = find_token_id(&vocab, &["<s>", "[BOS]", "<bos>", "<|startoftext|>"]);
    let eos_token = find_token_id(&vocab, &["</s>", "[EOS]", "<eos>", "<|endoftext|>"]);
    let pad_token = find_token_id(&vocab, &["<pad>", "[PAD]", "<|pad|>"]);
    let unk_token = find_token_id(&vocab, &["<unk>", "[UNK]", "<|unk|>"]);
    let mask_token = find_token_id(&vocab, &["<mask>", "[MASK]", "<|mask|>"]);
    
    Ok(SpecialTokens {
        bos_token: bos_token.map(TokenId::new),
        eos_token: eos_token.map(TokenId::new),
        pad_token: pad_token.map(TokenId::new),
        unk_token: unk_token.map(TokenId::new),
        mask_token: mask_token.map(TokenId::new),
        additional_tokens: HashMap::new(), // Could be extended
    })
}

/// Find token ID for any of the given token strings
fn find_token_id(vocab: &HashMap<String, u32>, candidates: &[&str]) -> Option<u32> {
    for candidate in candidates {
        if let Some(&token_id) = vocab.get(*candidate) {
            return Some(token_id);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_basic_encode_decode() {
        // This test would require a tokenizer file, so it's mostly for demonstration
        // In practice, you'd use a fixture tokenizer
        
        // let tokenizer = HuggingFaceTokenizer::from_file("path/to/tokenizer.json").await.unwrap();
        // 
        // let text = "Hello, world!";
        // let tokens = tokenizer.encode(text, false).unwrap();
        // let decoded = tokenizer.decode(&tokens, false).unwrap();
        // 
        // assert_eq!(text, decoded);
    }

    #[test]
    fn test_decode_cache() {
        let mut cache = DecodeCache::new(3);
        
        cache.insert(vec![TokenId::new(1), TokenId::new(2)], "hello".to_string());
        cache.insert(vec![TokenId::new(1), TokenId::new(2), TokenId::new(3)], "hello world".to_string());
        
        assert_eq!(cache.get(&[TokenId::new(1), TokenId::new(2)]), Some(&"hello".to_string()));
        assert_eq!(cache.get(&[TokenId::new(1), TokenId::new(2), TokenId::new(3)]), Some(&"hello world".to_string()));
    }
}
