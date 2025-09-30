//! HuggingFace tokenizer implementation

use crate::{IncrementalTokenizer, Tokenizer, TokenizerFactory, TokenizerInfo, TokenizerType};
use async_trait::async_trait;
use ferrum_types::{Result, SpecialTokens, TokenId};
use parking_lot::RwLock;
use std::sync::Arc;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::debug;

/// HuggingFace tokenizer wrapper
pub struct HuggingFaceTokenizer {
    tokenizer: Arc<HfTokenizer>,
    special_tokens: SpecialTokens,
    info: TokenizerInfo,
    /// Incremental decode cache for efficiency
    decode_cache: RwLock<DecodeCache>,
}

/// Incremental decoding state
#[derive(Debug, Clone, Default)]
pub struct IncrementalState {
    /// Accumulated tokens
    tokens: Vec<TokenId>,
    /// Decoded text so far
    text: String,
}

/// Cache for decoded token sequences
#[derive(Debug, Default)]
struct DecodeCache {
    cache: std::collections::HashMap<Vec<TokenId>, String>,
    max_size: usize,
}

impl DecodeCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size,
        }
    }

    fn get(&self, tokens: &[TokenId]) -> Option<&String> {
        self.cache.get(tokens)
    }

    fn insert(&mut self, tokens: Vec<TokenId>, text: String) {
        if self.cache.len() >= self.max_size {
            let to_remove: Vec<_> = self
                .cache
                .keys()
                .take(self.cache.len() / 2)
                .cloned()
                .collect();
            for key in to_remove {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(tokens, text);
    }
}

impl HuggingFaceTokenizer {
    /// Create new HuggingFace tokenizer
    pub async fn new(tokenizer: HfTokenizer) -> Result<Self> {
        let vocab_size = tokenizer.get_vocab_size(false);

        // Extract special tokens
        let special_tokens = extract_special_tokens(&tokenizer)?;

        let info = TokenizerInfo {
            tokenizer_type: TokenizerType::BPE, // Most HF tokenizers use BPE
            vocab_size,
            special_tokens: special_tokens.clone(),
            supports_incremental: true,
            supports_chat_template: false, // MVP: chat template support disabled
            max_token_length: None, // HF tokenizers don't expose this directly
            model_name: None,       // Can be set externally
        };

        debug!("Created HuggingFace tokenizer with vocab size {}", vocab_size);

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens,
            info,
            decode_cache: RwLock::new(DecodeCache::new(1000)),
        })
    }

    /// Create from file path
    pub async fn from_file(path: &str) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(path).map_err(|e| {
            ferrum_types::FerrumError::tokenizer(format!("Failed to load tokenizer: {}", e))
        })?;
        Self::new(tokenizer).await
    }

    /// Create from HuggingFace Hub
    pub async fn from_pretrained(repo_id: &str, _revision: Option<&str>) -> Result<Self> {
        let api = hf_hub::api::tokio::Api::new().map_err(|e| {
            ferrum_types::FerrumError::tokenizer(format!("Failed to create HF API: {}", e))
        })?;

        let repo = api.repo(hf_hub::Repo::model(repo_id.to_string()));

        // Note: hf_hub::api::tokio::ApiRepo doesn't have set_revision in newer versions
        // Revision is handled via the Repo struct or api.model_with_revision
        let tokenizer_file = repo.get("tokenizer.json").await.map_err(|e| {
            ferrum_types::FerrumError::tokenizer(format!("Failed to download tokenizer: {}", e))
        })?;

        let tokenizer = HfTokenizer::from_file(&tokenizer_file).map_err(|e| {
            ferrum_types::FerrumError::tokenizer(format!("Failed to load tokenizer: {}", e))
        })?;

        Self::new(tokenizer).await
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special)
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Encoding failed: {}", e)))?;

        Ok(encoding
            .get_ids()
            .iter()
            .map(|&id| TokenId::new(id))
            .collect())
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let token_ids: Vec<u32> = tokens.iter().map(|t| t.get()).collect();

        let text = self
            .tokenizer
            .decode(&token_ids, skip_special)
            .map_err(|e| ferrum_types::FerrumError::tokenizer(format!("Decoding failed: {}", e)))?;

        Ok(text)
    }

    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String> {
        // Check cache first
        if let Some(cached_prev) = self.decode_cache.read().get(prev) {
            let mut all_tokens = prev.to_vec();
            all_tokens.push(next);
            let full_text = self.decode(&all_tokens, true)?;

            // Cache the new sequence
            {
                let mut cache = self.decode_cache.write();
                cache.insert(all_tokens, full_text.clone());
            }

            // Return only the delta
            return Ok(full_text[cached_prev.len()..].to_string());
        }

        // No cache hit, decode both
        let prev_text = if prev.is_empty() {
            String::new()
        } else {
            self.decode(prev, true)?
        };

        let mut all_tokens = prev.to_vec();
        all_tokens.push(next);
        let full_text = self.decode(&all_tokens, true)?;

        // Update cache
        {
            let mut cache = self.decode_cache.write();
            if !prev.is_empty() {
                cache.insert(prev.to_vec(), prev_text.clone());
            }
            cache.insert(all_tokens, full_text.clone());
        }

        Ok(full_text[prev_text.len()..].to_string())
    }

    fn vocab_size(&self) -> usize {
        self.info.vocab_size
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.tokenizer
            .token_to_id(text)
            .map(TokenId::new)
    }

    fn token_text(&self, _token_id: TokenId) -> Option<&str> {
        // HF tokenizer doesn't support this efficiently, return None
        None
    }

    fn apply_chat_template(&self, messages: &[ferrum_interfaces::tokenizer::ChatMessage]) -> Result<String> {
        // MVP: simple concatenation
        let mut result = String::new();
        for msg in messages {
            result.push_str(&format!("{}: {}\n", msg.role, msg.content));
        }
        Ok(result.trim_end().to_string())
    }

    fn info(&self) -> TokenizerInfo {
        self.info.clone()
    }
}

impl IncrementalTokenizer for HuggingFaceTokenizer {
    type State = IncrementalState;

    fn create_state(&self) -> Self::State {
        IncrementalState::default()
    }

    fn decode_incremental_with_state(
        &self,
        state: &mut Self::State,
        token: TokenId,
    ) -> Result<String> {
        state.tokens.push(token);

        // Decode all tokens
        let full_text = self.decode(&state.tokens, true)?;

        // Calculate delta
        let delta = full_text[state.text.len()..].to_string();

        // Update state
        state.text = full_text;

        Ok(delta)
    }

    fn reset_state(&self, state: &mut Self::State) {
        state.tokens.clear();
        state.text.clear();
    }

    fn get_decoded_text(&self, state: &Self::State) -> String {
        state.text.clone()
    }
}

/// HuggingFace tokenizer factory
#[derive(Debug, Clone, Default)]
pub struct HuggingFaceTokenizerFactory;

impl HuggingFaceTokenizerFactory {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TokenizerFactory for HuggingFaceTokenizerFactory {
    async fn load_from_file(&self, path: &str) -> Result<Box<dyn Tokenizer>> {
        let tokenizer = HuggingFaceTokenizer::from_file(path).await?;
        Ok(Box::new(tokenizer))
    }

    async fn load_from_bytes(&self, data: &[u8]) -> Result<Box<dyn Tokenizer>> {
        let tokenizer = HfTokenizer::from_bytes(data).map_err(|e| {
            ferrum_types::FerrumError::tokenizer(format!("Failed to load tokenizer from bytes: {}", e))
        })?;
        let tokenizer = HuggingFaceTokenizer::new(tokenizer).await?;
        Ok(Box::new(tokenizer))
    }

    async fn load_from_hub(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<Box<dyn Tokenizer>> {
        let tokenizer = HuggingFaceTokenizer::from_pretrained(repo_id, revision).await?;
        Ok(Box::new(tokenizer))
    }

    async fn create_from_config(
        &self,
        config: &ferrum_interfaces::tokenizer::TokenizerConfig,
    ) -> Result<Box<dyn Tokenizer>> {
        // Load from path specified in config
        self.load_from_file(&config.path).await
    }

    fn supported_types(&self) -> Vec<TokenizerType> {
        vec![
            TokenizerType::BPE,
            TokenizerType::WordPiece,
            TokenizerType::SentencePiece,
        ]
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract special tokens from HF tokenizer
fn extract_special_tokens(tokenizer: &HfTokenizer) -> Result<SpecialTokens> {
    let _vocab = tokenizer.get_vocab(false);

    let bos_token = tokenizer
        .token_to_id("<s>")
        .or_else(|| tokenizer.token_to_id("[BOS]"))
        .or_else(|| tokenizer.token_to_id("<bos>"))
        .map(TokenId::new);

    let eos_token = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("[EOS]"))
        .or_else(|| tokenizer.token_to_id("<eos>"))
        .map(TokenId::new);

    let unk_token = tokenizer
        .token_to_id("<unk>")
        .or_else(|| tokenizer.token_to_id("[UNK]"))
        .map(TokenId::new);

    let pad_token = tokenizer
        .token_to_id("<pad>")
        .or_else(|| tokenizer.token_to_id("[PAD]"))
        .map(TokenId::new);

    let sep_token = tokenizer
        .token_to_id("[SEP]")
        .or_else(|| tokenizer.token_to_id("<sep>"))
        .map(TokenId::new);

    let cls_token = tokenizer
        .token_to_id("[CLS]")
        .or_else(|| tokenizer.token_to_id("<cls>"))
        .map(TokenId::new);

    let mask_token = tokenizer
        .token_to_id("[MASK]")
        .or_else(|| tokenizer.token_to_id("<mask>"))
        .map(TokenId::new);

    Ok(SpecialTokens {
        bos_token,
        eos_token,
        unk_token,
        pad_token,
        sep_token,
        cls_token,
        mask_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tokenizer_creation() {
        // This test requires a tokenizer file, skip in CI
        if std::env::var("CI").is_ok() {
            return;
        }

        // Try to load a simple tokenizer if available
        // In real tests, you would provide a test tokenizer file
    }
}