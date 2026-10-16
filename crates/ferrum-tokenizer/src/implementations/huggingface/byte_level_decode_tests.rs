use super::*;
use tokenizers::decoders::{byte_level::ByteLevel, sequence::Sequence, strip::Strip};
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::AddedToken;

fn tokenizer_with_vocab(entries: &[(&str, u32)]) -> HfTokenizer {
    let vocab: Vocab = entries
        .iter()
        .map(|&(token, id)| (token.to_owned(), id))
        .collect();
    let model = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    let mut tokenizer = HfTokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
}

fn assert_hf_equivalent(loaded: &HuggingFaceTokenizer, hf: &HfTokenizer, ids: &[u32]) {
    let tokens: Vec<_> = ids.iter().copied().map(TokenId::new).collect();
    for skip_special in [false, true] {
        assert_eq!(
            loaded.decode(&tokens, skip_special).unwrap(),
            hf.decode(ids, skip_special).unwrap(),
            "ids={ids:?}, skip_special={skip_special}"
        );
    }
}

#[tokio::test]
async fn byte_level_decode_matches_hf_across_utf8_and_unknown_boundaries() {
    let mut hf = tokenizer_with_vocab(&[
        ("\u{00f0}\u{0141}", 0), // bytes f0 9f: first half of fire
        ("\u{0136}\u{00a5}", 1), // bytes 94 a5: second half of fire
        ("ok", 2),
        ("\u{0120}", 3),          // byte 20
        ("\u{0120}\u{1f642}", 4), // nonalphabet scalar: whole-token UTF-8 fallback
        ("\u{00ff}", 5),          // invalid UTF-8 byte ff
        ("\u{00c3}", 6),
        ("\u{00a9}", 7),                 // c3 a9 together form e-acute
        ("\u{00ef}\u{00bf}\u{00bd}", 8), // explicit replacement character
        ("<eos>", 9),
    ]);
    hf.add_special_tokens(&[AddedToken::from("<eos>", true)]);
    hf.add_tokens(&[AddedToken::from("\u{0120}added", false)]);
    let added = hf.token_to_id("\u{0120}added").unwrap();
    let loaded = HuggingFaceTokenizer::new(hf.clone()).await.unwrap();
    assert!(loaded.byte_level_decode_vocab.is_some());

    for ids in [
        vec![],
        vec![u32::MAX],
        vec![0],
        vec![1],
        vec![0, 1],
        vec![0, u32::MAX, 1],
        vec![0, 9, 1],
        vec![0, 2, 1],
        vec![0, 0, 1, 1],
        vec![6, 7],
        vec![6, 9, 7],
        vec![5, 6, 7, 5, 8],
        vec![4, 0, 1, 3, added, 9],
    ] {
        // Candidate checking sees every prefix, including incomplete UTF-8.
        for end in 0..=ids.len() {
            assert_hf_equivalent(&loaded, &hf, &ids[..end]);
        }
    }
    assert_eq!(
        loaded
            .decode(&[TokenId::new(0), TokenId::new(1)], true)
            .unwrap(),
        "\u{1f525}"
    );
    assert_eq!(
        loaded.decode(&[TokenId::new(4)], true).unwrap(),
        "\u{0120}\u{1f642}"
    );
}

#[tokio::test]
async fn byte_level_decode_preserves_added_precedence_special_strings_and_sparse_ids() {
    // A public HF added-vocabulary replacement can shadow a model ID while
    // that same special token string also remains at another model ID.
    let mut donor = tokenizer_with_vocab(&[("added", 0)]);
    donor.add_special_tokens(&[AddedToken::from("added", true)]);
    let mut hf = tokenizer_with_vocab(&[("model", 0), ("added", 3), ("tail", 5)]);
    hf.with_added_vocabulary(donor.get_added_vocabulary().clone());
    assert_eq!(hf.id_to_token(0).as_deref(), Some("added"));
    assert_eq!(hf.id_to_token(3).as_deref(), Some("added"));
    let loaded = HuggingFaceTokenizer::new(hf.clone()).await.unwrap();

    assert_hf_equivalent(&loaded, &hf, &[0, 1, 2, 3, 4, 5, u32::MAX]);
    assert_eq!(
        loaded
            .decode(&[TokenId::new(0), TokenId::new(3)], false)
            .unwrap(),
        "addedadded"
    );
    assert_eq!(
        loaded
            .decode(&[TokenId::new(0), TokenId::new(3)], true)
            .unwrap(),
        ""
    );
    assert_eq!(loaded.decode(&[TokenId::new(5)], true).unwrap(), "tail");
}

#[tokio::test]
async fn byte_level_decode_falls_back_for_a_hidden_high_model_id() {
    let mut donor = tokenizer_with_vocab(&[("added", 0)]);
    donor.add_special_tokens(&[AddedToken::from("added", true)]);
    let mut hf = tokenizer_with_vocab(&[("added", u32::MAX), ("tail", 5)]);
    hf.with_added_vocabulary(donor.get_added_vocabulary().clone());
    assert_eq!(hf.get_vocab(true).values().copied().max(), Some(5));
    assert_eq!(hf.id_to_token(u32::MAX).as_deref(), Some("added"));

    let loaded = HuggingFaceTokenizer::new(hf.clone()).await.unwrap();
    assert!(loaded.byte_level_decode_vocab.is_none());
    assert_hf_equivalent(&loaded, &hf, &[u32::MAX, 0, 1, 5]);
    assert_eq!(
        loaded.decode(&[TokenId::new(u32::MAX)], false).unwrap(),
        "added"
    );
}

#[tokio::test]
async fn byte_level_decode_preserves_semantic_marker_utf8_flushes() {
    let mut hf = tokenizer_with_vocab(&[
        ("\u{00f0}\u{0141}", 0),
        ("\u{0136}\u{00a5}", 1),
        ("ok", 2),
        ("[THINK]", 3),
        ("[/THINK]", 4),
        ("<eos>", 5),
    ]);
    hf.add_special_tokens(&[
        AddedToken::from("[THINK]", true),
        AddedToken::from("[/THINK]", true),
        AddedToken::from("<eos>", true),
    ]);
    let loaded = HuggingFaceTokenizer::new(hf.clone()).await.unwrap();
    let ids = [0, 3, 1, 4, 2, 5];
    let tokens: Vec<_> = ids.into_iter().map(TokenId::new).collect();
    let expected = format!(
        "{}<think>{}</think>{}",
        hf.decode(&[0], true).unwrap(),
        hf.decode(&[1], true).unwrap(),
        hf.decode(&[2, 5], true).unwrap(),
    );
    assert_eq!(loaded.decode(&tokens, true).unwrap(), expected);
    assert!(
        !expected.contains('\u{1f525}'),
        "marker boundaries flush incomplete bytes"
    );
    assert_eq!(
        loaded.decode(&tokens, false).unwrap(),
        hf.decode(&ids, false).unwrap()
    );
    assert_eq!(
        loaded
            .decode(&[TokenId::new(3), TokenId::new(4), TokenId::new(3)], true)
            .unwrap(),
        "<think></think><think>"
    );
}

#[tokio::test]
async fn byte_level_decode_keeps_sequences_and_other_decoders_on_hf() {
    let mut hf = tokenizer_with_vocab(&[("\u{0120}", 0), ("text", 1)]);
    hf.with_decoder(Some(Sequence::new(vec![
        ByteLevel::default().into(),
        Strip::new(' ', 1, 0).into(),
    ])));
    let loaded = HuggingFaceTokenizer::new(hf.clone()).await.unwrap();
    assert!(loaded.byte_level_decode_vocab.is_none());
    assert_hf_equivalent(&loaded, &hf, &[0, 1]);
    assert_eq!(
        loaded
            .decode(&[TokenId::new(0), TokenId::new(1)], true)
            .unwrap(),
        "text"
    );

    hf.with_decoder(Some(Strip::new('\u{0120}', 1, 0)));
    let loaded = HuggingFaceTokenizer::new(hf.clone()).await.unwrap();
    assert!(loaded.byte_level_decode_vocab.is_none());
    assert_hf_equivalent(&loaded, &hf, &[0, 1, u32::MAX]);

    hf.with_decoder(None::<DecoderWrapper>);
    let loaded = HuggingFaceTokenizer::new(hf.clone()).await.unwrap();
    assert!(loaded.byte_level_decode_vocab.is_none());
    assert_hf_equivalent(&loaded, &hf, &[0, 1, u32::MAX]);
}
