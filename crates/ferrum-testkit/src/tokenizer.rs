//! Mock tokenizer that maps words to sequential token IDs.

use ferrum_interfaces::{Tokenizer, tokenizer::{TokenizerInfo, TokenizerType}};
use ferrum_types::{Result, SpecialTokens, TokenId};

/// Mock tokenizer: splits on whitespace, assigns sequential token IDs.
/// EOS token is vocab_size - 1.
pub struct MockTokenizer {
    vocab_size: usize,
    special_tokens: SpecialTokens,
}

impl MockTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        let eos = TokenId::new((vocab_size - 1) as u32);
        let bos = TokenId::new((vocab_size - 2) as u32);
        Self {
            vocab_size,
            special_tokens: SpecialTokens {
                bos_token: Some(bos),
                eos_token: Some(eos),
                unk_token: Some(TokenId::new(0)),
                pad_token: None,
                sep_token: None,
                cls_token: None,
                mask_token: None,
            },
        }
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        if add_special {
            if let Some(bos) = self.special_tokens.bos_token {
                tokens.push(bos);
            }
        }
        // Hash each word to a token ID in range [1, vocab_size - 3]
        for word in text.split_whitespace() {
            let hash = word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            let id = 1 + (hash % (self.vocab_size as u32 - 3));
            tokens.push(TokenId::new(id));
        }
        if tokens.is_empty() {
            tokens.push(TokenId::new(1)); // at least one token
        }
        Ok(tokens)
    }

    fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
        Ok(tokens
            .iter()
            .map(|t| format!("w{}", t.get()))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        Ok(format!("w{}", next.get()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_id(&self, _text: &str) -> Option<TokenId> {
        None
    }

    fn token_text(&self, _token_id: TokenId) -> Option<&str> {
        None
    }

    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::Custom,
            vocab_size: self.vocab_size,
            special_tokens: self.special_tokens.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: Some(128),
            model_name: Some("mock".into()),
        }
    }
}
