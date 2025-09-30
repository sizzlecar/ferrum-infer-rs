//! Tokenizer 占位实现

use std::sync::Arc;

use ferrum_types::Result;

#[derive(Debug, Default)]
pub struct TokenizerFactory;

impl TokenizerFactory {
    pub fn new() -> Self {
        Self
    }

    pub async fn create(&self, _source: &std::path::Path) -> Result<TokenizerHandle> {
        Ok(TokenizerHandle(Arc::new(PlaceholderTokenizer)))
    }
}

#[derive(Clone, Debug)]
pub struct TokenizerHandle(pub Arc<PlaceholderTokenizer>);

#[derive(Debug, Default)]
pub struct PlaceholderTokenizer;

impl PlaceholderTokenizer {
    pub fn vocab_size(&self) -> usize {
        0
    }
}
