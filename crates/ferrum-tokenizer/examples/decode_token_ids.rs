//! Decode saved token IDs without loading a model or generating new tokens.
//! Usage: decode_token_ids TOKENIZER_JSON TOKEN_IDS_JSON|- [--skip-special]
//! Input is a JSON array of u32 IDs or an object with a `token_ids` array.
//! Special tokens are retained by default; stdout contains only decoded text.

use std::fs::File;
use std::io::{Read, Write};

use ferrum_tokenizer::{HuggingFaceTokenizer, TokenId, Tokenizer};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(untagged)]
enum SavedTokens {
    Array(Vec<u32>),
    Dump { token_ids: Vec<u32> },
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: decode_token_ids TOKENIZER_JSON TOKEN_IDS_JSON|- [--skip-special]";
    let mut args = std::env::args_os().skip(1);
    let tokenizer_path = args.next().ok_or(USAGE)?;
    let input_path = args.next().ok_or(USAGE)?;
    let skip_special = match args.next() {
        None => false,
        Some(option) if option == "--skip-special" => true,
        Some(_) => return Err(USAGE.into()),
    };
    if args.next().is_some() {
        return Err(USAGE.into());
    }
    let mut input: Box<dyn Read> = if input_path == "-" {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(File::open(input_path)?)
    };
    let saved: SavedTokens = serde_json::from_reader(&mut input)?;
    let ids = match saved {
        SavedTokens::Array(ids) | SavedTokens::Dump { token_ids: ids } => ids,
    };
    let tokenizer = HuggingFaceTokenizer::from_file(
        tokenizer_path
            .to_str()
            .ok_or("tokenizer path must be UTF-8")?,
    )
    .await?;
    let tokens: Vec<TokenId> = ids.into_iter().map(TokenId::new).collect();
    // The upstream decoder can silently omit unknown IDs; reject corrupt dumps.
    for &id in &tokens {
        if tokenizer.token_text(id).is_none() {
            return Err(format!("token ID {} is absent from this tokenizer", id.get()).into());
        }
    }
    let text = tokenizer.decode(&tokens, skip_special)?;
    std::io::stdout().lock().write_all(text.as_bytes())?;
    Ok(())
}
