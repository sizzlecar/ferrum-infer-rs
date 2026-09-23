use super::*;
use crate::{implementations::HuggingFaceTokenizer, Tokenizer};

fn hf(surface: &str) -> HfTokenizer {
    let vocabulary: tokenizers::models::bpe::Vocab =
        [(surface.to_owned(), 0), ("[THINK]".to_owned(), 1)]
            .into_iter()
            .collect();
    let mut tokenizer = HfTokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    );
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
}

#[tokio::test]
async fn host_output_identity_tracks_real_content_and_decoder_not_allocations_or_cache() {
    let first = HuggingFaceTokenizer::new(hf("a")).await.unwrap();
    let same = HuggingFaceTokenizer::new(hf("a")).await.unwrap();
    let changed = HuggingFaceTokenizer::new(hf("b")).await.unwrap();
    let identity = first.host_output_policy_identity().unwrap();
    assert_eq!(Some(identity), same.host_output_policy_identity());
    assert_ne!(Some(identity), changed.host_output_policy_identity());
    first
        .decode(&[ferrum_types::TokenId::new(0)], true)
        .unwrap();
    assert_eq!(Some(identity), first.host_output_policy_identity());
    let mut alternate = hf("a");
    alternate.with_decoder(Some(tokenizers::decoders::fuse::Fuse::new()));
    let alternate = HuggingFaceTokenizer::new(alternate).await.unwrap();
    assert_ne!(Some(identity), alternate.host_output_policy_identity());
}

#[tokio::test]
async fn host_output_identity_reflects_resolved_special_source_overrides() {
    let bytes = hf("a").to_string(false).unwrap();
    let base = HuggingFaceTokenizer::from_source_bytes(bytes.as_bytes(), None, None)
        .await
        .unwrap();
    let changed = HuggingFaceTokenizer::from_source_bytes(
        bytes.as_bytes(),
        None,
        Some(br#"{"eos_token_id":0}"#),
    )
    .await
    .unwrap();
    assert_ne!(
        base.special_tokens().eos_token,
        changed.special_tokens().eos_token
    );
    assert_ne!(
        base.host_output_policy_identity().unwrap(),
        changed.host_output_policy_identity().unwrap()
    );
}

#[test]
fn host_output_identity_canonical_objects_keep_array_order_and_unknown_source() {
    let hash = |value: Value| {
        let mut digest = Sha256::new();
        hash_value(&mut digest, &value);
        <[u8; 32]>::from(digest.finalize())
    };
    assert_eq!(
        hash(serde_json::json!({"a":1,"b":2})),
        hash(serde_json::json!({"b":2,"a":1}))
    );
    assert_ne!(
        hash(serde_json::json!([1, 2])),
        hash(serde_json::json!([2, 1]))
    );
    assert_eq!(resolved(None, &SpecialTokens::default()), None);
}
