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
    id_to_token: Vec<Option<String>>,
    /// Reasoning-marker token ids mapped to the canonical tag emitted in
    /// their place. Some vocabs mark think tags `special: true`
    /// (Magistral's `[THINK]`/`[/THINK]`), so a skip-special decode would
    /// silently drop them and the thinking text would leak into content.
    /// Decode preserves these ids and normalizes every dialect to
    /// `<think>`/`</think>`, which is what the serving layer splits on.
    think_markers: Vec<(u32, &'static str)>,
    /// Incremental decode cache for efficiency
    decode_cache: RwLock<DecodeCache>,
}

/// Marker-string dialects mapped to the canonical tags. Probed against the
/// vocab at construction; absent dialects cost nothing.
const THINK_MARKER_DIALECTS: [(&str, &'static str); 4] = [
    ("<think>", "<think>"),
    ("</think>", "</think>"),
    ("[THINK]", "<think>"),
    ("[/THINK]", "</think>"),
];

fn probe_think_markers(tokenizer: &HfTokenizer) -> Vec<(u32, &'static str)> {
    THINK_MARKER_DIALECTS
        .iter()
        .filter_map(|(text, canonical)| tokenizer.token_to_id(text).map(|id| (id, *canonical)))
        .collect()
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
        let id_to_token = build_id_to_token(&tokenizer);

        // Extract special tokens
        let special_tokens = extract_special_tokens(&tokenizer)?;

        let info = TokenizerInfo {
            tokenizer_type: TokenizerType::BPE, // Most HF tokenizers use BPE
            vocab_size,
            special_tokens: special_tokens.clone(),
            supports_incremental: true,
            supports_chat_template: false, // MVP: chat template support disabled
            max_token_length: None,        // HF tokenizers don't expose this directly
            model_name: None,              // Can be set externally
        };

        debug!(
            "Created HuggingFace tokenizer with vocab size {}",
            vocab_size
        );

        let think_markers = probe_think_markers(&tokenizer);

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens,
            info,
            id_to_token,
            think_markers,
            decode_cache: RwLock::new(DecodeCache::new(1000)),
        })
    }

    /// Create from file path. Special tokens are resolved from the sibling
    /// `generation_config.json` / `tokenizer_config.json` when present;
    /// vocab name probing is only the fallback for bare tokenizer.json files.
    pub async fn from_file(path: &str) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(path).map_err(|e| {
            ferrum_types::FerrumError::tokenizer(format!("Failed to load tokenizer: {}", e))
        })?;
        let overrides =
            special_token_overrides_from_configs(std::path::Path::new(path), &tokenizer);
        let mut this = Self::new(tokenizer).await?;
        this.apply_special_token_overrides(overrides);
        Ok(this)
    }

    fn apply_special_token_overrides(&mut self, overrides: SpecialTokenOverrides) {
        if overrides.bos.is_some() {
            self.special_tokens.bos_token = overrides.bos;
        }
        if overrides.eos.is_some() {
            self.special_tokens.eos_token = overrides.eos;
        }
        if !overrides.extra_eos.is_empty() {
            self.special_tokens.extra_eos_tokens = overrides.extra_eos;
        }
        self.info.special_tokens = self.special_tokens.clone();
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

        // Skip-special decode must not swallow reasoning markers: split at
        // marker ids, decode the segments, and splice the canonical tags
        // back in. The common no-marker case stays a single decode call.
        if skip_special
            && !self.think_markers.is_empty()
            && token_ids
                .iter()
                .any(|id| self.think_markers.iter().any(|(mid, _)| mid == id))
        {
            let mut out = String::new();
            let mut segment: Vec<u32> = Vec::with_capacity(token_ids.len());
            for id in &token_ids {
                if let Some((_, canonical)) =
                    self.think_markers.iter().find(|(mid, _)| mid == id)
                {
                    if !segment.is_empty() {
                        out.push_str(&self.tokenizer.decode(&segment, true).map_err(|e| {
                            ferrum_types::FerrumError::tokenizer(format!(
                                "Decoding failed: {}",
                                e
                            ))
                        })?);
                        segment.clear();
                    }
                    out.push_str(canonical);
                } else {
                    segment.push(*id);
                }
            }
            if !segment.is_empty() {
                out.push_str(&self.tokenizer.decode(&segment, true).map_err(|e| {
                    ferrum_types::FerrumError::tokenizer(format!("Decoding failed: {}", e))
                })?);
            }
            return Ok(out);
        }

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
        self.tokenizer.token_to_id(text).map(TokenId::new)
    }

    fn token_text(&self, token_id: TokenId) -> Option<&str> {
        self.id_to_token
            .get(token_id.get() as usize)
            .and_then(|value| value.as_deref())
    }

    fn apply_chat_template(
        &self,
        messages: &[ferrum_interfaces::tokenizer::ChatMessage],
    ) -> Result<String> {
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
            ferrum_types::FerrumError::tokenizer(format!(
                "Failed to load tokenizer from bytes: {}",
                e
            ))
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

fn build_id_to_token(tokenizer: &HfTokenizer) -> Vec<Option<String>> {
    let vocab = tokenizer.get_vocab(true);
    let Some(max_id) = vocab.values().copied().max() else {
        return Vec::new();
    };
    let mut id_to_token = vec![None; max_id as usize + 1];
    for (token, id) in vocab {
        let slot = &mut id_to_token[id as usize];
        if slot.is_none() {
            *slot = Some(token);
        }
    }
    id_to_token
}

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
        extra_eos_tokens: Vec::new(),
    })
}

/// EOS/BOS overrides read from the model's config files next to
/// `tokenizer.json`. The vocabulary alone cannot tell which token a model is
/// trained to emit as EOS — name probing (`</s>`-style) breaks on models that
/// rename their special-token slots (e.g. DeepSeek-R1 distills rename
/// `<|endoftext|>` to `<|end▁of▁sentence|>`), so the config files are
/// authoritative: `generation_config.json` `eos_token_id` (int or list)
/// first, then `tokenizer_config.json` `eos_token` / `bos_token` (string or
/// `{"content": ...}` AddedToken object).
#[derive(Debug, Default)]
struct SpecialTokenOverrides {
    bos: Option<TokenId>,
    eos: Option<TokenId>,
    extra_eos: Vec<TokenId>,
}

fn special_token_overrides_from_configs(
    tokenizer_json: &std::path::Path,
    tokenizer: &HfTokenizer,
) -> SpecialTokenOverrides {
    let Some(dir) = tokenizer_json.parent() else {
        return SpecialTokenOverrides::default();
    };
    let mut overrides = SpecialTokenOverrides::default();

    if let Some(gen) = read_json(&dir.join("generation_config.json")) {
        let mut eos_ids = token_id_list(gen.get("eos_token_id"));
        if !eos_ids.is_empty() {
            overrides.eos = Some(eos_ids.remove(0));
            overrides.extra_eos = eos_ids;
        }
        if let Some(bos) = token_id_list(gen.get("bos_token_id")).into_iter().next() {
            overrides.bos = Some(bos);
        }
    }

    if let Some(tok_cfg) = read_json(&dir.join("tokenizer_config.json")) {
        if overrides.eos.is_none() {
            overrides.eos = token_from_config_value(tok_cfg.get("eos_token"), tokenizer);
        }
        if overrides.bos.is_none() {
            overrides.bos = token_from_config_value(tok_cfg.get("bos_token"), tokenizer);
        }
    }

    overrides
}

fn read_json(path: &std::path::Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// `eos_token_id` / `bos_token_id` in generation_config.json is either a
/// number or a list of numbers.
fn token_id_list(value: Option<&serde_json::Value>) -> Vec<TokenId> {
    match value {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .map(|v| vec![TokenId::new(v as u32)])
            .unwrap_or_default(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|v| TokenId::new(v as u32))
            .collect(),
        _ => Vec::new(),
    }
}

/// `eos_token` / `bos_token` in tokenizer_config.json is either a plain
/// string or an AddedToken object `{"content": "...", ...}`.
fn token_from_config_value(
    value: Option<&serde_json::Value>,
    tokenizer: &HfTokenizer,
) -> Option<TokenId> {
    let text = match value? {
        serde_json::Value::String(s) => s.as_str(),
        serde_json::Value::Object(obj) => obj.get("content")?.as_str()?,
        _ => return None,
    };
    tokenizer.token_to_id(text).map(TokenId::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_cache_creation() {
        let cache = DecodeCache::new(100);
        assert_eq!(cache.max_size, 100);
        assert_eq!(cache.cache.len(), 0);
    }

    #[test]
    fn test_decode_cache_insert_and_get() {
        let mut cache = DecodeCache::new(10);
        let tokens = vec![TokenId::new(1), TokenId::new(2)];
        let text = "hello".to_string();

        cache.insert(tokens.clone(), text.clone());

        let result = cache.get(&tokens);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), &text);
    }

    #[test]
    fn test_decode_cache_eviction() {
        let mut cache = DecodeCache::new(2);

        // 填满缓存
        cache.insert(vec![TokenId::new(1)], "a".to_string());
        cache.insert(vec![TokenId::new(2)], "b".to_string());

        assert_eq!(cache.cache.len(), 2);

        // 触发驱逐
        cache.insert(vec![TokenId::new(3)], "c".to_string());

        // 应该已经清理了一些旧条目
        assert!(cache.cache.len() <= 2);
    }

    #[test]
    fn test_incremental_state_default() {
        let state = IncrementalState::default();
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("IncrementalState"));
    }

    #[test]
    fn test_incremental_state_clone() {
        let state = IncrementalState::default();
        let cloned = state.clone();

        // 验证克隆成功
        let state_str = format!("{:?}", state);
        let cloned_str = format!("{:?}", cloned);
        assert_eq!(state_str, cloned_str);
    }

    #[test]
    fn test_huggingface_tokenizer_factory_creation() {
        let factory = HuggingFaceTokenizerFactory::new();
        let debug_str = format!("{:?}", factory);
        assert!(debug_str.contains("HuggingFaceTokenizerFactory"));
    }

    #[test]
    fn test_huggingface_tokenizer_factory_default() {
        let factory = HuggingFaceTokenizerFactory;
        let debug_str = format!("{:?}", factory);
        assert!(debug_str.contains("HuggingFaceTokenizerFactory"));
    }

    #[test]
    fn test_huggingface_tokenizer_factory_clone() {
        let factory = HuggingFaceTokenizerFactory::new();
        let cloned = factory.clone();

        let factory_str = format!("{:?}", factory);
        let cloned_str = format!("{:?}", cloned);
        assert_eq!(factory_str, cloned_str);
    }

    #[test]
    fn test_huggingface_tokenizer_factory_supported_types() {
        let factory = HuggingFaceTokenizerFactory::new();
        let types = factory.supported_types();

        assert!(!types.is_empty());
        assert!(types.contains(&TokenizerType::BPE));
    }

    #[test]
    fn test_extract_special_tokens_with_mock_tokenizer() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

        // 创建一个简单的 mock tokenizer
        let vocab: Vocab = [
            ("hello".to_string(), 0),
            ("<s>".to_string(), 1),
            ("</s>".to_string(), 2),
            ("<unk>".to_string(), 3),
            ("<pad>".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();

        let mut tokenizer = HfTokenizer::new(bpe);
        tokenizer.add_special_tokens(&[
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
            AddedToken::from("<unk>", true),
            AddedToken::from("<pad>", true),
        ]);

        // 测试提取特殊 tokens
        let result = extract_special_tokens(&tokenizer);
        assert!(result.is_ok());

        let special_tokens = result.unwrap();
        assert!(special_tokens.bos_token.is_some());
        assert!(special_tokens.eos_token.is_some());
        assert!(special_tokens.unk_token.is_some());
        assert!(special_tokens.pad_token.is_some());
    }

    #[tokio::test]
    async fn test_huggingface_tokenizer_with_mock() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

        let vocab: Vocab = [
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("<s>".to_string(), 2),
            ("</s>".to_string(), 3),
            ("<unk>".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();

        let mut hf_tokenizer = HfTokenizer::new(bpe);
        hf_tokenizer.add_special_tokens(&[
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
            AddedToken::from("<unk>", true),
        ]);

        // 测试创建 HuggingFaceTokenizer
        let result = HuggingFaceTokenizer::new(hf_tokenizer).await;
        assert!(result.is_ok());

        let tokenizer = result.unwrap();
        assert_eq!(tokenizer.vocab_size(), 5);
    }

    #[tokio::test]
    async fn test_tokenizer_encode_decode() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

        let vocab: Vocab = [
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("<s>".to_string(), 2),
            ("</s>".to_string(), 3),
            ("<unk>".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();

        let mut hf_tokenizer = HfTokenizer::new(bpe);
        hf_tokenizer.add_special_tokens(&[
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
            AddedToken::from("<unk>", true),
        ]);

        let tokenizer = HuggingFaceTokenizer::new(hf_tokenizer).await.unwrap();

        // 测试 encode - 即使无法编码，也会返回 UNK token
        let result = tokenizer.encode("hello", false);
        assert!(result.is_ok());

        let _tokens = result.unwrap();
        // Tokenizer 可能返回空数组或 UNK tokens
        // 我们只验证结果是 Ok

        // 测试 decode with empty tokens
        let decoded = tokenizer.decode(&[], false);
        assert!(decoded.is_ok());
    }

    #[tokio::test]
    async fn test_tokenizer_special_tokens() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

        let vocab: Vocab = [
            ("hello".to_string(), 0),
            ("<s>".to_string(), 1),
            ("</s>".to_string(), 2),
        ]
        .into_iter()
        .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();

        let mut hf_tokenizer = HfTokenizer::new(bpe);
        hf_tokenizer.add_special_tokens(&[
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ]);

        let tokenizer = HuggingFaceTokenizer::new(hf_tokenizer).await.unwrap();
        let special_tokens = tokenizer.special_tokens();

        // 应该能找到一些特殊 tokens
        assert!(special_tokens.bos_token.is_some() || special_tokens.eos_token.is_some());
    }

    #[tokio::test]
    async fn test_tokenizer_token_id_lookup() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::Tokenizer as HfTokenizer;

        let vocab: Vocab = [("hello".to_string(), 0), ("world".to_string(), 1)]
            .into_iter()
            .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();

        let hf_tokenizer = HfTokenizer::new(bpe);
        let tokenizer = HuggingFaceTokenizer::new(hf_tokenizer).await.unwrap();

        // 测试 token_id 查找
        let token_id = tokenizer.token_id("hello");
        assert!(token_id.is_some());
        assert_eq!(token_id.unwrap().get(), 0);
    }

    #[tokio::test]
    async fn test_tokenizer_token_text_reverse_lookup() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::Tokenizer as HfTokenizer;

        let vocab: Vocab = [
            ("hello".to_string(), 0),
            ("[PAD151935]".to_string(), 1),
            ("</think>".to_string(), 2),
        ]
        .into_iter()
        .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();

        let hf_tokenizer = HfTokenizer::new(bpe);
        let tokenizer = HuggingFaceTokenizer::new(hf_tokenizer).await.unwrap();

        assert_eq!(tokenizer.token_text(TokenId::new(1)), Some("[PAD151935]"));
        assert_eq!(tokenizer.token_text(TokenId::new(2)), Some("</think>"));
        assert_eq!(tokenizer.token_text(TokenId::new(99)), None);
    }

    #[tokio::test]
    async fn test_tokenizer_info() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::Tokenizer as HfTokenizer;

        let vocab: Vocab = [("hello".to_string(), 0), ("world".to_string(), 1)]
            .into_iter()
            .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();

        let hf_tokenizer = HfTokenizer::new(bpe);
        let tokenizer = HuggingFaceTokenizer::new(hf_tokenizer).await.unwrap();

        let info = tokenizer.info();
        assert_eq!(info.vocab_size, 2);
        assert!(info.supports_incremental);
        assert_eq!(info.tokenizer_type, TokenizerType::BPE);
    }

    #[tokio::test]
    async fn test_incremental_tokenizer_interface() {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::Tokenizer as HfTokenizer;

        let vocab: Vocab = [("hello".to_string(), 0), ("world".to_string(), 1)]
            .into_iter()
            .collect();

        let merges = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();

        let hf_tokenizer = HfTokenizer::new(bpe);
        let tokenizer = HuggingFaceTokenizer::new(hf_tokenizer).await.unwrap();

        // 测试增量解码接口
        let mut state = tokenizer.create_state();

        // 添加一个 token
        let result = tokenizer.decode_incremental_with_state(&mut state, TokenId::new(0));
        assert!(result.is_ok());

        // 重置状态
        tokenizer.reset_state(&mut state);
        let text = tokenizer.get_decoded_text(&state);
        assert!(text.is_empty());
    }

    fn tiny_tokenizer_with_specials(specials: &[&str]) -> HfTokenizer {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::AddedToken;

        let vocab: Vocab = [("hello".to_string(), 0), ("world".to_string(), 1)]
            .into_iter()
            .collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("hello".to_string())
            .build()
            .unwrap();
        let mut tokenizer = HfTokenizer::new(bpe);
        tokenizer.add_special_tokens(
            &specials
                .iter()
                .map(|s| AddedToken::from(*s, true))
                .collect::<Vec<_>>(),
        );
        tokenizer
    }

    #[tokio::test]
    async fn eos_comes_from_generation_config_not_name_probing() {
        // DeepSeek-R1-distill style: special-token slots renamed, no
        // `</s>` / `<|endoftext|>`-style names anywhere in the vocab.
        let tokenizer =
            tiny_tokenizer_with_specials(&["<|end▁of▁sentence|>", "<|User|>", "<|Assistant|>"]);
        let eos_id = tokenizer.token_to_id("<|end▁of▁sentence|>").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            format!("{{\"bos_token_id\": null, \"eos_token_id\": {eos_id}}}"),
        )
        .unwrap();

        let loaded = HuggingFaceTokenizer::from_file(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            loaded.special_tokens().eos_token.map(|t| t.get()),
            Some(eos_id)
        );
        assert!(loaded.special_tokens().extra_eos_tokens.is_empty());
    }

    #[tokio::test]
    async fn multi_eos_ids_land_in_extra_eos_tokens() {
        let tokenizer = tiny_tokenizer_with_specials(&["<|eot_id|>", "<|end_of_text|>"]);
        let primary = tokenizer.token_to_id("<|end_of_text|>").unwrap();
        let extra = tokenizer.token_to_id("<|eot_id|>").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            format!("{{\"eos_token_id\": [{primary}, {extra}]}}"),
        )
        .unwrap();

        let loaded = HuggingFaceTokenizer::from_file(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            loaded.special_tokens().eos_token.map(|t| t.get()),
            Some(primary)
        );
        assert_eq!(
            loaded
                .special_tokens()
                .extra_eos_tokens
                .iter()
                .map(|t| t.get())
                .collect::<Vec<_>>(),
            vec![extra]
        );
    }

    #[tokio::test]
    async fn tokenizer_config_eos_string_is_fallback_without_generation_config() {
        let tokenizer = tiny_tokenizer_with_specials(&["<|end▁of▁sentence|>"]);
        let eos_id = tokenizer.token_to_id("<|end▁of▁sentence|>").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            "{\"eos_token\": {\"content\": \"<|end▁of▁sentence|>\"}}",
        )
        .unwrap();

        let loaded = HuggingFaceTokenizer::from_file(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            loaded.special_tokens().eos_token.map(|t| t.get()),
            Some(eos_id)
        );
    }

    #[tokio::test]
    async fn bare_tokenizer_json_still_uses_name_probing() {
        let tokenizer = tiny_tokenizer_with_specials(&["<s>", "</s>"]);
        let eos_id = tokenizer.token_to_id("</s>").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();

        let loaded = HuggingFaceTokenizer::from_file(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(
            loaded.special_tokens().eos_token.map(|t| t.get()),
            Some(eos_id)
        );
    }

    #[tokio::test]
    async fn skip_special_decode_preserves_and_normalizes_think_markers() {
        // Magistral-style: [THINK]/[/THINK] are `special: true`, so a plain
        // skip-special decode would drop them and leak thinking into content.
        let tokenizer = tiny_tokenizer_with_specials(&["[THINK]", "[/THINK]", "<eos>"]);
        let think = tokenizer.token_to_id("[THINK]").unwrap();
        let end_think = tokenizer.token_to_id("[/THINK]").unwrap();
        let eos = tokenizer.token_to_id("<eos>").unwrap();
        let hello = tokenizer.token_to_id("hello").unwrap();
        let world = tokenizer.token_to_id("world").unwrap();

        let loaded = HuggingFaceTokenizer::new(tokenizer).await.unwrap();
        let tokens: Vec<TokenId> = [think, hello, end_think, world, eos]
            .into_iter()
            .map(TokenId::new)
            .collect();
        let text = loaded.decode(&tokens, true).unwrap();

        assert_eq!(text, "<think>hello</think>world");
    }

    #[tokio::test]
    async fn skip_special_decode_without_markers_is_unchanged() {
        let tokenizer = tiny_tokenizer_with_specials(&["<eos>"]);
        let eos = tokenizer.token_to_id("<eos>").unwrap();
        let hello = tokenizer.token_to_id("hello").unwrap();

        let loaded = HuggingFaceTokenizer::new(tokenizer).await.unwrap();
        let tokens: Vec<TokenId> = [hello, eos].into_iter().map(TokenId::new).collect();

        assert_eq!(loaded.decode(&tokens, true).unwrap(), "hello");
    }
}
