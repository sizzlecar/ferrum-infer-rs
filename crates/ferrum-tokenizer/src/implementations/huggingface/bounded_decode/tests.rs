use super::*;
use crate::Tokenizer;
use tokenizers::{
    decoders::{byte_level::ByteLevel, fuse::Fuse, sequence::Sequence},
    models::bpe::{Vocab, BPE},
    AddedToken,
};

fn hf_with_tokens(tokens: &[&str]) -> HfTokenizer {
    let vocabulary: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| ((*token).to_owned(), index as u32))
        .collect();
    HfTokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .unwrap(),
    )
}

fn ids(values: &[u32]) -> Vec<TokenId> {
    values.iter().copied().map(TokenId::new).collect()
}

fn assert_matches_ordinary(
    tokenizer: &HuggingFaceTokenizer,
    tokens: &[TokenId],
    skip_special: bool,
) -> String {
    let expected = tokenizer.decode(tokens, skip_special).unwrap();
    let required = tokenizer
        .bounded_decode_bound()
        .expect("direct ByteLevel capability")
        .requirements(tokens.len())
        .unwrap();
    let mut scratch = vec![0xa5; required.scratch_bytes];
    let mut output = String::with_capacity(required.text_bytes);
    let allocation = (output.as_ptr(), output.capacity(), scratch.as_ptr());
    tokenizer
        .decode_bounded_into(tokens, skip_special, &mut scratch, &mut output)
        .unwrap();
    assert_eq!(output, expected, "tokens={tokens:?}, skip={skip_special}");
    assert_eq!(
        (output.as_ptr(), output.capacity(), scratch.as_ptr()),
        allocation,
        "caller buffers must not be reallocated"
    );
    output
}

#[tokio::test]
async fn bounded_decode_matches_byte_alphabet_split_chinese_and_emoji() {
    let spellings: Vec<String> = (0..=u8::MAX)
        .map(|byte| {
            byte_level_char_bytes()
                .iter()
                .find_map(|(character, mapped)| (*mapped == byte).then_some(*character))
                .unwrap()
                .to_string()
        })
        .collect();
    let mut hf = hf_with_tokens(&spellings.iter().map(String::as_str).collect::<Vec<_>>());
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    for byte in 0..=u8::MAX {
        assert_matches_ordinary(&tokenizer, &ids(&[u32::from(byte)]), true);
    }
    for text in ["中文", "🔥🙂", "a 中🙂 z\n"] {
        let tokens: Vec<_> = text
            .as_bytes()
            .iter()
            .map(|byte| TokenId::new(u32::from(*byte)))
            .collect();
        assert_eq!(assert_matches_ordinary(&tokenizer, &tokens, true), text);
    }
}

#[tokio::test]
async fn bounded_decode_matches_marker_segments_fallbacks_and_specials() {
    let tokens = [
        "ð", "Ł", "Ķ", "¥", "Ģ", "ÿ", "a", "中", "Ġ中", "[THINK]", "[/THINK]", "<pad>",
    ];
    let mut hf = hf_with_tokens(&tokens);
    hf.with_decoder(Some(ByteLevel::default()));
    hf.add_special_tokens(&[
        AddedToken::from("[THINK]", true),
        AddedToken::from("[/THINK]", true),
        AddedToken::from("<pad>", true),
    ]);
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    for skip_special in [false, true] {
        for a in 0..tokens.len() as u32 {
            for b in 0..tokens.len() as u32 {
                for c in 0..tokens.len() as u32 {
                    assert_matches_ordinary(&tokenizer, &ids(&[a, b, c]), skip_special);
                }
            }
        }
        assert_matches_ordinary(&tokenizer, &[], skip_special);
    }
    // Do not combine f0 before the marker with its continuations after it.
    assert_eq!(
        assert_matches_ordinary(&tokenizer, &ids(&[0, 9, 1, 2, 3, 10]), true),
        "�<think>���</think>"
    );
    // A non-alphabet 中 forces fallback of the WHOLE token, including Ġ.
    assert_eq!(assert_matches_ordinary(&tokenizer, &ids(&[8]), true), "Ġ中");
}

#[tokio::test]
async fn bounded_decode_uses_hf_special_set_not_slots_or_latest_added_flag() {
    let mut hf = hf_with_tokens(&["<s>", "<pad>", "control", "a"]);
    hf.with_decoder(Some(ByteLevel::default()));
    hf.add_special_tokens(&[AddedToken::from("control", true)]);
    hf.add_tokens(&[AddedToken::from("control", false)]);
    assert!(!hf.get_added_tokens_decoder()[&2].special);
    assert!(hf.get_added_vocabulary().is_special_token("control"));
    let mut tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    // These product metadata overrides do not change HF's added-vocabulary set.
    tokenizer.special_tokens.eos_token = Some(TokenId::new(3));
    assert_eq!(
        assert_matches_ordinary(&tokenizer, &ids(&[0, 1, 2, 3]), true),
        "<s><pad>a"
    );
    assert_eq!(
        assert_matches_ordinary(&tokenizer, &ids(&[0, 1, 2, 3]), false),
        "<s><pad>controla"
    );
}

#[tokio::test]
async fn bounded_decode_keeps_typed_protocol_markers_and_skips_unknown_ids() {
    let mut hf = hf_with_tokens(&["a", "[THINK]", "[/THINK]", "<|im_start|>"]);
    hf.with_decoder(Some(ByteLevel::default()));
    hf.add_special_tokens(&[AddedToken::from("<|im_start|>", true)]);
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    for skip_special in [false, true] {
        assert_matches_ordinary(&tokenizer, &ids(&[u32::MAX, 0, 44, 3]), skip_special);
        assert_eq!(
            assert_matches_ordinary(&tokenizer, &ids(&[u32::MAX, 44]), skip_special),
            ""
        );
    }
    // Think canonicalization also applies when HF does not mark the IDs special.
    assert_eq!(
        assert_matches_ordinary(&tokenizer, &ids(&[1, 0, 2]), true),
        "<think>a</think>"
    );
    // Test an actual marker from each typed protocol, without assuming its name.
    for protocol in [
        ferrum_types::ModelOutputProtocol::HarmonyGptOss,
        ferrum_types::ModelOutputProtocol::GemmaThought,
    ] {
        let marker = protocol.preserved_special_token_texts()[0];
        let mut hf = hf_with_tokens(&["a", marker]);
        hf.with_decoder(Some(ByteLevel::default()));
        hf.add_special_tokens(&[AddedToken::from(marker, true)]);
        let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
        assert_eq!(
            assert_matches_ordinary(&tokenizer, &ids(&[1]), true),
            marker
        );
    }
}

#[tokio::test]
async fn bounded_decode_accounts_for_added_tokens_outside_legacy_vocab_size() {
    let mut hf = hf_with_tokens(&["a"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let long_token = "ÿ".repeat(4096);
    hf.add_tokens(&[AddedToken::from(long_token.clone(), false)]);
    let added_id = hf.token_to_id(&long_token).unwrap();
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    assert_eq!(tokenizer.vocab_size(), 1);
    let required = tokenizer
        .bounded_decode_bound()
        .unwrap()
        .requirements(2)
        .unwrap();
    assert_eq!(required.scratch_bytes, 4096 * 2);
    assert_eq!(required.text_bytes, 4096 * 2 * 3);
    assert_eq!(
        assert_matches_ordinary(&tokenizer, &ids(&[added_id, added_id]), true),
        "�".repeat(4096 * 2)
    );
}

#[tokio::test]
async fn bounded_decode_preflights_both_capacities_without_changing_buffers() {
    let mut hf = hf_with_tokens(&["ÿ"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let tokens = ids(&[0, 0]);
    let required = tokenizer
        .bounded_decode_bound()
        .unwrap()
        .requirements(2)
        .unwrap();
    let mut output = String::with_capacity(required.text_bytes);
    output.push_str("prior");
    let mut short_scratch = vec![0x5a; required.scratch_bytes - 1];
    assert_eq!(
        tokenizer.decode_bounded_into(&tokens, true, &mut short_scratch, &mut output),
        Err(BoundedDecodeError::InsufficientScratch {
            required: required.scratch_bytes,
            available: required.scratch_bytes - 1,
        })
    );
    assert_eq!(output, "prior");
    assert!(short_scratch.iter().all(|byte| *byte == 0x5a));
    let mut scratch = vec![0x5a; required.scratch_bytes];
    let mut no_capacity = String::new();
    assert_eq!(
        tokenizer.decode_bounded_into(&tokens, true, &mut scratch, &mut no_capacity),
        Err(BoundedDecodeError::InsufficientOutput {
            required: required.text_bytes,
            available: 0,
        })
    );
    assert_eq!(no_capacity, "");
    assert!(scratch.iter().all(|byte| *byte == 0x5a));
    let mut short_output = String::from("x");
    let short_capacity = short_output.capacity();
    assert!(short_capacity < required.text_bytes);
    assert_eq!(
        tokenizer.decode_bounded_into(&tokens, true, &mut scratch, &mut short_output),
        Err(BoundedDecodeError::InsufficientOutput {
            required: required.text_bytes,
            available: short_capacity,
        })
    );
    assert_eq!(short_output, "x");
    assert!(scratch.iter().all(|byte| *byte == 0x5a));
    // The old content is replaced, so available capacity is not spare capacity.
    let allocation = (output.as_ptr(), output.capacity());
    tokenizer
        .decode_bounded_into(&tokens, true, &mut scratch, &mut output)
        .unwrap();
    assert_eq!(output, "��");
    assert_eq!((output.as_ptr(), output.capacity()), allocation);
    tokenizer
        .decode_bounded_into(&[], true, &mut [], &mut output)
        .unwrap();
    assert!(output.is_empty());
    assert_eq!((output.as_ptr(), output.capacity()), allocation);
}

#[tokio::test]
async fn bounded_decode_requires_direct_byte_level_and_rejects_unproven_lookup() {
    let absent = HuggingFaceTokenizer::new(hf_with_tokens(&["a"]))
        .await
        .unwrap();
    let mut hf = hf_with_tokens(&["a"]);
    hf.with_decoder(Some(Sequence::new(vec![
        ByteLevel::default().into(),
        Fuse::new().into(),
    ])));
    let sequence = HuggingFaceTokenizer::new(hf).await.unwrap();
    for tokenizer in [absent, sequence] {
        assert!(tokenizer.bounded_decode_bound().is_none());
        let mut scratch = [17];
        let mut output = String::from("prior");
        assert_eq!(
            tokenizer.decode_bounded_into(&ids(&[0]), true, &mut scratch, &mut output),
            Err(BoundedDecodeError::Unsupported)
        );
        assert_eq!(scratch, [17]);
        assert_eq!(output, "prior");
    }
    let mut hf = hf_with_tokens(&["a", "b"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let incomplete = [Some("a".to_owned())];
    let mismatched = [Some("b".to_owned()), Some("a".to_owned())];
    assert!(ByteLevelBoundedDecoder::new(&hf, &incomplete, &[]).is_none());
    assert!(ByteLevelBoundedDecoder::new(&hf, &mismatched, &[]).is_none());
    // Deliberate ID aliases: whichever reverse spelling HF chooses is the only
    // admissible borrowed spelling. A conflicting alias cannot claim support.
    let vocabulary: Vocab = [("a".to_owned(), 0), ("alias".to_owned(), 0)]
        .into_iter()
        .collect();
    let mut aliases = HfTokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .unwrap(),
    );
    aliases.with_decoder(Some(ByteLevel::default()));
    let chosen = aliases.id_to_token(0).unwrap();
    let other = if chosen == "a" { "alias" } else { "a" };
    assert!(ByteLevelBoundedDecoder::new(&aliases, &[Some(other.into())], &[]).is_none());
    assert!(ByteLevelBoundedDecoder::new(&aliases, &[Some(chosen)], &[]).is_some());
}

#[tokio::test]
async fn bounded_decode_does_not_use_or_populate_full_history_cache() {
    let mut hf = hf_with_tokens(&["a", "b"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let tokens = ids(&[0, 1, 0]);
    tokenizer
        .decode_cache
        .write()
        .insert(tokens.clone(), "wrong cached history".into());
    let before = tokenizer.decode_cache.read().cache.clone();
    assert_eq!(assert_matches_ordinary(&tokenizer, &tokens, true), "aba");
    assert_eq!(
        assert_matches_ordinary(&tokenizer, &tokens[..2], true),
        "ab"
    );
    assert_eq!(tokenizer.decode_cache.read().cache, before);
}

#[tokio::test]
async fn bounded_decode_preserves_unknown_holes_in_the_vocabulary() {
    let vocabulary: Vocab = [("a".to_owned(), 0), ("b".to_owned(), 2)]
        .into_iter()
        .collect();
    let mut hf = HfTokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .unwrap(),
    );
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    assert_eq!(
        assert_matches_ordinary(&tokenizer, &ids(&[0, 1, 2, u32::MAX]), true),
        "ab"
    );
}

#[test]
fn bounded_decode_lossy_matches_std_for_every_two_byte_input_and_invalid_suffixes() {
    let mut output = String::with_capacity(6);
    for first in 0..=u8::MAX {
        for second in 0..=u8::MAX {
            let bytes = [first, second];
            output.clear();
            append_lossy(&bytes, &mut output);
            assert_eq!(output, String::from_utf8_lossy(&bytes), "{bytes:?}");
        }
    }
    let cases: &[&[u8]] = &[
        &[0xe2, 0x82], // One replacement for an incomplete multi-byte suffix.
        &[0xe2, 0x82, b'a'],
        &[0xe0, 0x80, 0x80],       // Overlong encoding.
        &[0xed, 0xa0, 0x80],       // Surrogate.
        &[0xf4, 0x90, 0x80, 0x80], // Above Unicode range.
        &[0xf0, 0x9f, 0x94],
        &[b'a', 0xf0, 0x9f, 0x94, b'z', 0xff],
        &[0xf0, 0x9f, 0x94, 0xa5],
    ];
    for bytes in cases {
        let mut output = String::with_capacity(bytes.len() * 3);
        let allocation = (output.as_ptr(), output.capacity());
        append_lossy(bytes, &mut output);
        assert_eq!(output, String::from_utf8_lossy(bytes), "{bytes:?}");
        assert_eq!((output.as_ptr(), output.capacity()), allocation);
    }
}

#[tokio::test]
async fn bounded_token_bytes_match_all_byte_surfaces_and_preserve_utf8_fragments() {
    let spellings: Vec<String> = (0..=u8::MAX)
        .map(|byte| {
            byte_level_char_bytes()
                .iter()
                .find_map(|(character, mapped)| (*mapped == byte).then_some(*character))
                .unwrap()
                .to_string()
        })
        .collect();
    let mut hf = hf_with_tokens(&spellings.iter().map(String::as_str).collect::<Vec<_>>());
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    assert_eq!(tokenizer.bounded_token_bytes_bound().unwrap().get(), 1);
    for byte in 0..=u8::MAX {
        let mut buffer = [0xa5, 0xa5];
        let token = TokenId::new(u32::from(byte));
        assert_eq!(
            tokenizer.token_bytes_bounded_into(token, &mut buffer),
            Ok(Some(1))
        );
        assert_eq!(buffer, [byte, 0xa5]);
        assert_eq!(tokenizer.token_bytes(token).as_deref(), Some(&buffer[..1]));
    }
    for text in ["中文", "🔥🙂"] {
        let mut raw = vec![0; text.len()];
        for (index, byte) in text.as_bytes().iter().copied().enumerate() {
            assert_eq!(
                tokenizer.token_bytes_bounded_into(
                    TokenId::new(u32::from(byte)),
                    &mut raw[index..index + 1]
                ),
                Ok(Some(1))
            );
        }
        assert_eq!(raw, text.as_bytes());
        assert_eq!(std::str::from_utf8(&raw).unwrap(), text);
    }
    let mut lead = [0];
    tokenizer
        .token_bytes_bounded_into(TokenId::new(0xf0), &mut lead)
        .unwrap();
    assert_eq!(lead, [0xf0]);
    assert!(
        std::str::from_utf8(&lead).is_err(),
        "raw fragments must not be replaced with decoded U+FFFD"
    );
    assert_eq!(tokenizer.decode(&ids(&[0xf0]), false).unwrap(), "�");
}

#[tokio::test]
async fn bounded_token_bytes_keep_special_surfaces_and_whole_token_fallback() {
    let mut hf = hf_with_tokens(&["ðŁ", "Ķ¥", "Ġ中", "[THINK]", "[/THINK]", "<pad>"]);
    hf.with_decoder(Some(ByteLevel::default()));
    hf.add_special_tokens(&[
        AddedToken::from("[THINK]", true),
        AddedToken::from("[/THINK]", true),
        AddedToken::from("<pad>", true),
    ]);
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let expected: &[&[u8]] = &[
        &[0xf0, 0x9f],
        &[0x94, 0xa5],
        "Ġ中".as_bytes(),
        b"[THINK]",
        b"[/THINK]",
        b"<pad>",
    ];
    let bound = tokenizer.bounded_token_bytes_bound().unwrap().get();
    for (id, expected) in expected.iter().enumerate() {
        let mut output = vec![0xa5; bound + 1];
        let pointer = output.as_ptr();
        assert_eq!(
            tokenizer.token_bytes_bounded_into(TokenId::new(id as u32), &mut output),
            Ok(Some(expected.len()))
        );
        assert_eq!(&output[..expected.len()], *expected);
        assert!(output[expected.len()..].iter().all(|byte| *byte == 0xa5));
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(
            tokenizer.token_bytes(TokenId::new(id as u32)).as_deref(),
            Some(*expected)
        );
    }
    assert_eq!(tokenizer.decode(&ids(&[0, 1]), false).unwrap(), "🔥");
    assert_eq!(
        tokenizer.decode(&ids(&[3, 4, 5]), true).unwrap(),
        "<think></think>"
    );
    assert_eq!(tokenizer.decode(&ids(&[2]), false).unwrap(), "Ġ中");
}

#[tokio::test]
async fn bounded_token_bytes_account_for_added_raw_surfaces_without_decode_cache() {
    let mut hf = hf_with_tokens(&["a"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let long = "ÿ".repeat(4096);
    hf.add_tokens(&[AddedToken::from(long.clone(), false)]);
    let id = TokenId::new(hf.token_to_id(&long).unwrap());
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    assert_eq!(tokenizer.vocab_size(), 1);
    assert_eq!(tokenizer.bounded_token_bytes_bound().unwrap().get(), 4096);
    tokenizer
        .decode_cache
        .write()
        .insert(vec![id], "incorrect decoded value".to_owned());
    let before = tokenizer.decode_cache.read().cache.clone();
    let mut output = vec![0; 4096];
    assert_eq!(
        tokenizer.token_bytes_bounded_into(id, &mut output),
        Ok(Some(4096))
    );
    assert!(output.iter().all(|byte| *byte == 0xff));
    assert_eq!(tokenizer.decode_cache.read().cache, before);
}

#[tokio::test]
async fn bounded_token_bytes_preflight_actual_length_and_distinguish_empty_unknown() {
    let mut hf = hf_with_tokens(&["", "a", "Ġ中"]);
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    assert_eq!(
        tokenizer.bounded_token_bytes_bound().unwrap().get(),
        "Ġ中".len()
    );
    let mut too_short = [0xa5; 4];
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(2), &mut too_short),
        Err(BoundedDecodeError::InsufficientOutput {
            required: "Ġ中".len(),
            available: 4
        })
    );
    assert_eq!(too_short, [0xa5; 4]);
    let mut one = [0xa5];
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(1), &mut one),
        Ok(Some(1))
    );
    assert_eq!(one, [b'a']);
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(0), &mut []),
        Ok(Some(0))
    );
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(u32::MAX), &mut []),
        Ok(None)
    );
    let mut unchanged = [0xa5; 6];
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(999), &mut unchanged),
        Ok(None)
    );
    assert_eq!(unchanged, [0xa5; 6]);
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(0), &mut unchanged),
        Ok(Some(0))
    );
    assert_eq!(unchanged, [0xa5; 6]);
}

#[tokio::test]
async fn bounded_token_bytes_reject_unproven_decoder_and_preserve_vocabulary_holes() {
    let mut sequence = hf_with_tokens(&["a"]);
    sequence.with_decoder(Some(Sequence::new(vec![
        ByteLevel::default().into(),
        Fuse::new().into(),
    ])));
    for hf in [hf_with_tokens(&["a"]), sequence] {
        let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
        assert!(tokenizer.bounded_token_bytes_bound().is_none());
        let mut output = [0xa5; 8];
        assert_eq!(
            tokenizer.token_bytes_bounded_into(TokenId::new(0), &mut output),
            Err(BoundedDecodeError::Unsupported)
        );
        assert_eq!(output, [0xa5; 8]);
    }
    let vocabulary: Vocab = [("a".to_owned(), 0), ("b".to_owned(), 2)]
        .into_iter()
        .collect();
    let mut hf = HfTokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .unwrap(),
    );
    hf.with_decoder(Some(ByteLevel::default()));
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let mut output = [0xa5];
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(1), &mut output),
        Ok(None)
    );
    assert_eq!(output, [0xa5]);
    assert_eq!(
        tokenizer.token_bytes_bounded_into(TokenId::new(2), &mut output),
        Ok(Some(1))
    );
    assert_eq!(output, [b'b']);
}

#[tokio::test]
async fn bounded_incremental_policy_matches_legacy_prefix_errors_and_marker_deltas() {
    let spellings = [
        "reason", "</think>", "Ċ", "Ġ", "answer", "ä", "¸", "Ń", "ÿ", "</s>",
    ];
    let mut hf = hf_with_tokens(&spellings);
    hf.with_decoder(Some(ByteLevel::default()));
    hf.add_special_tokens(&[AddedToken::from("</s>", true)]);
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    let policy = tokenizer.bounded_incremental_decode_policy().unwrap();
    let mut prefix_errors = 0;
    for a in 0..spellings.len() as u32 {
        for b in 0..spellings.len() as u32 {
            for c in 0..spellings.len() as u32 {
                let previous = ids(&[a, b]);
                let full = ids(&[a, b, c]);
                let previous_text = assert_matches_ordinary(&tokenizer, &previous, true);
                let full_text = assert_matches_ordinary(&tokenizer, &full, true);
                let bounded = policy.delta(&previous_text, &full_text);
                let legacy = tokenizer.decode_incremental(&previous, TokenId::new(c));
                match (bounded, legacy) {
                    (Some(actual), Ok(expected)) => assert_eq!(actual, expected),
                    (None, Err(_)) => prefix_errors += 1,
                    (actual, expected) => {
                        panic!("incremental mismatch {full:?}: {actual:?} vs {expected:?}")
                    }
                }
            }
        }
    }
    assert!(
        prefix_errors > 0,
        "split UTF-8 prefix replacement must remain an error"
    );
    let mut unknown = hf_with_tokens(&["a"]);
    unknown.with_decoder(Some(Fuse::new()));
    let unknown = HuggingFaceTokenizer::new(unknown).await.unwrap();
    assert!(unknown.bounded_incremental_decode_policy().is_none());
}
