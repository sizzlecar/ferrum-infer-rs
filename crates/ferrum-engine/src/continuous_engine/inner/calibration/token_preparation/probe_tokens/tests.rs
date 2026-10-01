use super::*;
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;

fn budget() -> CalibrationPrefixTokenBudgetV1 {
    CalibrationPrefixTokenBudgetV1 {
        maximum_token_ids: NonZeroUsize::new(64).unwrap(),
        maximum_token_bytes: NonZeroUsize::new(64).unwrap(),
        maximum_total_token_bytes: NonZeroUsize::new(4096).unwrap(),
        maximum_prefix_tokens: NonZeroUsize::new(4).unwrap(),
        maximum_utf8_transitions: NonZeroUsize::new(1024).unwrap(),
        maximum_search_states: NonZeroUsize::new(32).unwrap(),
    }
}
async fn tokenizer(surfaces: &[&str], eos: &[usize]) -> HuggingFaceTokenizer {
    let vocab = surfaces
        .iter()
        .enumerate()
        .map(|(i, s)| ((*s).to_owned(), i as u32))
        .collect();
    let mut raw = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    let raw = raw.to_string(false).unwrap();
    let config = serde_json::to_vec(&serde_json::json!({"eos_token_id":eos})).unwrap();
    HuggingFaceTokenizer::from_source_bytes(raw.as_bytes(), None, Some(&config))
        .await
        .unwrap()
}
fn found(value: CalibrationPrefixTokenDiscoveryV1) -> DiscoveredCalibrationPrefixTokensV1 {
    match value {
        Found(v) => v,
        other => panic!("expected validated prefixes: {other:?}"),
    }
}
fn check(tokenizer: &dyn Tokenizer, found: &DiscoveredCalibrationPrefixTokensV1) {
    let maximum = NonZeroUsize::new(8).unwrap();
    assert!(found
        .clean
        .validate_for_tokenizer(tokenizer, maximum)
        .unwrap()
        .expected_pending_bytes()
        .is_empty());
    assert!(!found
        .pending
        .validate_for_tokenizer(tokenizer, maximum)
        .unwrap()
        .expected_pending_bytes()
        .is_empty());
    assert_eq!(
        Some(found.clean.tokenizer_policy_sha256),
        tokenizer.host_output_policy_identity()
    );
    assert_eq!(
        found.clean.tokenizer_policy_sha256,
        found.pending.tokenizer_policy_sha256
    );
}

#[tokio::test]
async fn automatic_prefix_tokens_uses_actual_bytelevel_policy_and_token_id_order() {
    for surfaces in [["<unk>", "ok", "Ã", "©"], ["©", "Ã", "<unk>", "ok"]] {
        let tokenizer = tokenizer(&surfaces, &[]).await;
        let pair = found(discover_prefix_tokens(
            &tokenizer,
            NonZeroUsize::new(8).unwrap(),
            &[],
            budget(),
        ));
        check(&tokenizer, &pair);
        assert_eq!(pair.pending.token_ids, [tokenizer.token_id("Ã").unwrap()]);
        assert!(pair.audit.token_ids_examined <= surfaces.len());
        let mut wrong = pair.pending.clone();
        wrong.tokenizer_policy_sha256[0] ^= 1;
        assert!(wrong
            .validate_for_tokenizer(&tokenizer, NonZeroUsize::new(8).unwrap())
            .is_err());
    }
}

#[tokio::test]
async fn automatic_prefix_tokens_finds_multitoken_clean_trajectory_through_real_utf8_state() {
    // No non-special single token is a complete scalar: C3 + A9 yields é.
    let tokenizer = tokenizer(&["<unk>", "Ã", "©"], &[]).await;
    let pair = found(discover_prefix_tokens(
        &tokenizer,
        NonZeroUsize::new(8).unwrap(),
        &[],
        budget(),
    ));
    check(&tokenizer, &pair);
    assert_eq!(
        pair.clean.token_ids,
        [
            tokenizer.token_id("Ã").unwrap(),
            tokenizer.token_id("©").unwrap()
        ]
    );
    assert_eq!(pair.clean.release_generated, 2);
    assert_eq!(pair.pending.release_generated, 1);
    let mut limited = budget();
    limited.maximum_prefix_tokens = NonZeroUsize::MIN;
    assert!(matches!(
        discover_prefix_tokens(&tokenizer, NonZeroUsize::new(8).unwrap(), &[], limited),
        CoverageUnavailable {
            reason: U::PrefixLengthBudget,
            ..
        }
    ));
}

#[tokio::test]
async fn automatic_prefix_tokens_excludes_primary_extra_eos_stops_and_invalid_fragments() {
    // C3 and E2 are declared EOS, F0 is a caller stop token. E1 remains usable.
    let tokenizer = tokenizer(&["<unk>", "ok", "Ã", "â", "ð", "á", "ï¿½", "©"], &[2, 3]).await;
    let stop = [tokenizer.token_id("ð").unwrap()];
    let pair = found(discover_prefix_tokens(
        &tokenizer,
        NonZeroUsize::new(8).unwrap(),
        &stop,
        budget(),
    ));
    check(&tokenizer, &pair);
    assert_eq!(pair.pending.token_ids, [tokenizer.token_id("á").unwrap()]);
    for token in pair.clean.token_ids.iter().chain(&pair.pending.token_ids) {
        assert!(!tokenizer.is_special_token(*token));
        assert!(!tokenizer.special_tokens().extra_eos_tokens.contains(token));
        assert!(!stop.contains(token));
    }
    let replacement = CalibrationPrefixTokensV1 {
        tokenizer_policy_sha256: tokenizer.host_output_policy_identity().unwrap(),
        token_ids: vec![tokenizer.token_id("ï¿½").unwrap()],
        release_generated: 1,
    };
    assert!(replacement
        .validate_for_tokenizer(&tokenizer, NonZeroUsize::new(8).unwrap())
        .is_err());
}

#[tokio::test]
async fn automatic_prefix_tokens_budget_exhaustion_is_unavailable_not_unreachable() {
    let tokenizer = tokenizer(&["<unk>", "one", "two", "Ã", "©"], &[]).await;
    let mut limited = budget();
    limited.maximum_token_ids = NonZeroUsize::new(3).unwrap();
    let CoverageUnavailable { reason, audit } =
        discover_prefix_tokens(&tokenizer, NonZeroUsize::new(8).unwrap(), &[], limited)
    else {
        panic!("unscanned pending token cannot be claimed");
    };
    assert_eq!(reason, U::TokenIdBudget);
    assert_eq!(audit.token_ids_examined, 3);
    assert!(!audit.vocabulary_scan_complete);
    check(
        &tokenizer,
        &found(discover_prefix_tokens(
            &tokenizer,
            NonZeroUsize::new(8).unwrap(),
            &[],
            budget(),
        )),
    );
    let mut limited = budget();
    limited.maximum_utf8_transitions = NonZeroUsize::MIN;
    assert!(matches!(
        discover_prefix_tokens(&tokenizer, NonZeroUsize::new(8).unwrap(), &[], limited),
        CoverageUnavailable {
            reason: U::TransitionBudget,
            ..
        }
    ));
    let mut limited = budget();
    limited.maximum_total_token_bytes = NonZeroUsize::MIN;
    assert!(matches!(
        discover_prefix_tokens(&tokenizer, NonZeroUsize::new(8).unwrap(), &[], limited),
        CoverageUnavailable {
            reason: U::TokenByteBudget,
            ..
        }
    ));
    assert!(matches!(
        discover_prefix_tokens(&tokenizer, NonZeroUsize::MIN, &[], budget()),
        CoverageUnavailable {
            reason: U::NoRoomForNormalSuffix,
            ..
        }
    ));
}

#[tokio::test]
async fn automatic_prefix_tokens_search_state_cap_is_not_silently_expanded() {
    let tokenizer = tokenizer(&["<unk>", "Ã", "â", "©", "ok"], &[]).await;
    let mut limited = budget();
    limited.maximum_search_states = NonZeroUsize::MIN;
    let CoverageUnavailable { reason, audit } =
        discover_prefix_tokens(&tokenizer, NonZeroUsize::new(8).unwrap(), &[], limited)
    else {
        panic!("state cap must remain explicit");
    };
    assert_eq!(reason, U::SearchStateBudget);
    assert_eq!(audit.peak_search_states, 1);
    check(
        &tokenizer,
        &found(discover_prefix_tokens(
            &tokenizer,
            NonZeroUsize::new(8).unwrap(),
            &[],
            budget(),
        )),
    );
}

#[tokio::test]
async fn automatic_prefix_tokens_aligned_search_uses_same_original_release_depth() {
    // C3 A9 is clean after two tokens; E2 82 is pending after two. No
    // non-special one-token clean trajectory is available.
    let tokenizer = tokenizer(&["<unk>", "Ã", "©", "â", "Ĥ"], &[]).await;
    let pair = found(discover_aligned_prefix_tokens(
        &tokenizer,
        NonZeroUsize::new(8).unwrap(),
        &[],
        budget(),
    ));
    check(&tokenizer, &pair);
    assert_eq!(pair.clean.release_generated, 2);
    assert_eq!(pair.pending.release_generated, 2);
    assert!(pair.audit.utf8_transitions <= budget().maximum_utf8_transitions.get());
}

#[tokio::test]
async fn automatic_prefix_tokens_aligned_search_never_pads_incompatible_trajectories() {
    // Only even-length clean and odd-length pending trajectories exist within
    // this alphabet. The legacy unequal-release API retains its old behavior.
    let tokenizer = tokenizer(&["<unk>", "Ã", "©"], &[]).await;
    let pair = found(discover_prefix_tokens(
        &tokenizer,
        NonZeroUsize::new(8).unwrap(),
        &[],
        budget(),
    ));
    assert_ne!(pair.clean.release_generated, pair.pending.release_generated);
    assert!(matches!(
        discover_aligned_prefix_tokens(&tokenizer, NonZeroUsize::new(8).unwrap(), &[], budget()),
        CoverageUnavailable { .. }
    ));
    let mut limited = budget();
    limited.maximum_prefix_tokens = NonZeroUsize::MIN;
    assert!(matches!(
        discover_aligned_prefix_tokens(&tokenizer, NonZeroUsize::new(8).unwrap(), &[], limited),
        CoverageUnavailable { .. }
    ));
}
