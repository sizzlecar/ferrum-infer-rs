use super::*;
use tokenizers::{
    decoders::{byte_level::ByteLevel, fuse::Fuse, sequence::Sequence},
    models::bpe::{Vocab, BPE},
    AddedToken,
};

fn hf_with_tokens(tokens: &[&str]) -> HfTokenizer {
    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| ((*token).to_owned(), index as u32))
        .collect();
    HfTokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    )
}

#[tokio::test]
async fn output_bound_covers_utf8_fragments_lossy_bytes_and_marker_segments() {
    // f0 9f 94 a5 forms 🔥; 80 and ff are invalid by themselves. 中 exercises
    // ByteLevel's whole-token fallback for non-alphabet vocabulary spellings.
    let tokens = ["ð", "Ł", "Ķ", "¥", "Ģ", "ÿ", "a", "中", "[THINK]"];
    let mut hf = hf_with_tokens(&tokens);
    hf.with_decoder(Some(ByteLevel::default()));
    hf.add_special_tokens(&[AddedToken::from("[THINK]", true)]);
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let bound = tokenizer.decoded_text_bound().unwrap();
    for skip_special in [false, true] {
        for a in 0..tokens.len() as u32 {
            for b in 0..tokens.len() as u32 {
                for c in 0..tokens.len() as u32 {
                    let ids = [TokenId::new(a), TokenId::new(b), TokenId::new(c)];
                    let text = tokenizer.decode(&ids, skip_special).unwrap();
                    assert!(text.len() <= bound.max_decoded_bytes(ids.len()).unwrap());
                }
            }
        }
        assert!(tokenizer.decode(&[], skip_special).unwrap().is_empty());
    }
    assert_eq!(
        tokenizer
            .decode(
                &[
                    TokenId::new(0),
                    TokenId::new(1),
                    TokenId::new(2),
                    TokenId::new(3)
                ],
                true
            )
            .unwrap(),
        "🔥"
    );
    assert_eq!(
        tokenizer.decode(&[TokenId::new(8)], true).unwrap(),
        "<think>"
    );
}

#[tokio::test]
async fn output_bound_accounts_for_three_byte_replacement() {
    let mut hf = hf_with_tokens(&["ÿ"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let bound = tokenizer.decoded_text_bound().unwrap();
    assert_eq!(bound.max_decoded_bytes(2), Some(6));
    assert_eq!(
        tokenizer
            .decode(&[TokenId::new(0), TokenId::new(0)], true)
            .unwrap(),
        "��"
    );
}

#[tokio::test]
async fn output_bound_includes_added_long_tokens_and_retained_text() {
    let mut hf = hf_with_tokens(&["a"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let long = "x".repeat(4096);
    hf.add_tokens(&[AddedToken::from(long.clone(), false)]);
    let long_id = TokenId::new(hf.token_to_id(&long).unwrap());
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let bound = tokenizer.decoded_text_bound().unwrap();
    assert_eq!(tokenizer.vocab_size(), 1); // Added tokens are outside this legacy count.
    assert!(bound.max_decoded_bytes(1).unwrap() >= long.len());
    let text = tokenizer
        .decode(&[long_id, long_id, TokenId::new(0)], true)
        .unwrap();
    // A stop/protocol buffer may retain the two long tokens before one new
    // token causes an event. The caller budgets all un-emitted text.
    assert!(text.len() <= bound.max_unemitted_bytes(3, 0).unwrap());
    assert_eq!(text.len(), long.len() * 2 + 1);
}

#[tokio::test]
async fn output_bound_does_not_infer_capability_from_any_byte_level_stage() {
    let absent = HuggingFaceTokenizer::new(hf_with_tokens(&["a"]))
        .await
        .unwrap();
    assert!(absent.decoded_text_bound().is_none());
    let mut hf = hf_with_tokens(&["a"]);
    hf.with_decoder(Some(Sequence::new(vec![
        ByteLevel::default().into(),
        Fuse::new().into(),
    ])));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    assert!(tokenizer.byte_level_decoder);
    assert!(tokenizer.decoded_text_bound().is_none());
}
