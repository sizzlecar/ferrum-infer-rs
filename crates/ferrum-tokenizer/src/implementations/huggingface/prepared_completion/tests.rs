use super::*;
use tokenizers::{
    models::bpe::{Vocab, BPE},
    AddedToken,
};

fn characters() -> HfTokenizer {
    let chars = "</>_thinktoolca|eughnd\n 中"
        .chars()
        .collect::<std::collections::BTreeSet<_>>();
    let vocabulary: Vocab = chars
        .into_iter()
        .enumerate()
        .map(|(id, ch)| (ch.to_string(), id as u32))
        .collect();
    HfTokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    )
}

#[tokio::test]
async fn prepared_completion_matches_legacy_exact_and_multitoken_encodings() {
    let mut hf = characters();
    hf.add_special_tokens(&[AddedToken::from("<tool_call>", true)]);
    let wrapper = HuggingFaceTokenizer::new(hf).await.unwrap();
    for entry in wrapper.prepared_completion.entries.iter() {
        let expected = wrapper
            .token_id(entry.text)
            .map(|id| vec![id])
            .unwrap_or_else(|| wrapper.encode(entry.text, false).unwrap());
        assert_eq!(
            wrapper.prepared_completion_tokens(entry.text),
            Some(expected.as_slice())
        );
        let tokens = wrapper.prepared_completion_tokens(entry.text).unwrap();
        let original = (tokens.as_ptr(), tokens.len());
        // Even unrelated misses never insert into the immutable finite table.
        assert!(wrapper
            .prepared_completion_tokens("request-created unknown marker")
            .is_none());
        let again = wrapper.prepared_completion_tokens(entry.text).unwrap();
        assert_eq!((again.as_ptr(), again.len()), original);
    }
    assert_eq!(
        wrapper
            .prepared_completion_tokens("<tool_call>")
            .unwrap()
            .len(),
        1
    );
    assert!(
        wrapper
            .prepared_completion_tokens("</think>")
            .unwrap()
            .len()
            > 1
    );
}

#[tokio::test]
async fn prepared_completion_preserves_actual_normalization_and_unknown_failure() {
    let mut hf = characters();
    hf.with_normalizer(Some(
        tokenizers::normalizers::replace::Replace::new("think", "中").unwrap(),
    ));
    let wrapper = HuggingFaceTokenizer::new(hf).await.unwrap();
    let expected = wrapper.encode("</think>", false).unwrap();
    assert_eq!(expected.len(), 4);
    assert_eq!(
        wrapper.prepared_completion_tokens("</think>"),
        Some(expected.as_slice())
    );

    let hf = HfTokenizer::new(
        BPE::builder()
            .vocab_and_merges(Vocab::default(), Vec::new())
            .unk_token("absent-unknown-token".into())
            .build()
            .unwrap(),
    );
    let wrapper = HuggingFaceTokenizer::new(hf).await.unwrap();
    assert!(wrapper.encode("</think>", false).is_err());
    assert!(wrapper.prepared_completion_tokens("</think>").is_none());
}
