use super::*;
use crate::model_executor::TokenSelectionMask;
#[test]
fn mixed_readback_roles_preserve_final_logits_and_decode_processors() {
    let greedy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1, 1])),
        repetition_penalty: None,
    };
    let intermediate = VNextParticipantOutputRole::IntermediatePrefill;
    let final_prefill = VNextParticipantOutputRole::FinalPrefill(LogitsReturnPolicy::FullLogits);
    let greedy_decode = VNextParticipantOutputRole::Decode(greedy.clone());
    let host_decode = VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits);
    assert!(intermediate.logits_policy().is_none());
    for (roles, expected) in [
        (
            vec![&intermediate, &greedy_decode],
            VNextProductOutputMode::GreedyToken,
        ),
        (
            vec![&greedy_decode, &intermediate, &greedy_decode],
            VNextProductOutputMode::GreedyToken,
        ),
        (
            vec![&intermediate, &final_prefill, &greedy_decode],
            VNextProductOutputMode::FullLogits,
        ),
        (
            vec![&intermediate, &greedy_decode, &host_decode],
            VNextProductOutputMode::FullLogits,
        ),
        (vec![&intermediate], VNextProductOutputMode::FullLogits),
    ] {
        assert_eq!(
            product_output_mode_for_roles(VNextExecutionWaveKind::Mixed, roles),
            expected,
        );
    }
    // Host sampling retains the complete row, so the caller can still apply
    // masks, repetition penalties, grammar, and structured-output processors.
    let logits = ExecutorSamplingOutput::full_logits(vec![0.1, 9.0, 0.4, 0.2]).unwrap();
    logits.validate_for_policy(&greedy, 4).unwrap();
    logits
        .validate_for_policy(&LogitsReturnPolicy::FullLogits, 4)
        .unwrap();
    assert!(ExecutorSamplingOutput::greedy_token(TokenId::new(1))
        .validate_for_policy(&LogitsReturnPolicy::FullLogits, 4)
        .is_err());
    assert_eq!(
        VNextExecutionWaveKind::Mixed.reusable_execution_class(),
        PACKED_TOKEN_REUSABLE_CLASS,
    );
}

#[test]
fn intermediate_role_cannot_discard_a_final_prefill_frontier() {
    let tokens = [0, 1, 2, 3];
    let intermediate_span = TokenSpanWork::from_token_ids(&tokens, 0..2).unwrap();
    let final_span = TokenSpanWork::from_token_ids(&tokens, 2..4).unwrap();
    let intermediate = VNextParticipantOutputRole::prefill(
        PrefillChunk::new(0, 2, 4).unwrap(),
        LogitsReturnPolicy::FullLogits,
    );
    let final_prefill = VNextParticipantOutputRole::prefill(
        PrefillChunk::new(2, 2, 4).unwrap(),
        LogitsReturnPolicy::FullLogits,
    );
    assert!(intermediate.matches_frontier(VNextExecutionWaveKind::Mixed, &intermediate_span, 4));
    assert!(!intermediate.matches_frontier(VNextExecutionWaveKind::Mixed, &final_span, 4));
    assert!(final_prefill.matches_frontier(VNextExecutionWaveKind::Mixed, &final_span, 4));
    assert!(!final_prefill.matches_frontier(VNextExecutionWaveKind::Mixed, &intermediate_span, 4));
}

#[test]
fn final_prefill_uses_its_sampling_policy_without_sampling_intermediate_chunks() {
    let policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![0, 1, 1, 0])),
        repetition_penalty: None,
    };
    let intermediate =
        VNextParticipantOutputRole::prefill(PrefillChunk::new(0, 2, 4).unwrap(), policy.clone());
    let final_prefill =
        VNextParticipantOutputRole::prefill(PrefillChunk::new(2, 2, 4).unwrap(), policy.clone());
    let host = VNextParticipantOutputRole::FinalPrefill(LogitsReturnPolicy::FullLogits);
    assert!(intermediate.logits_policy().is_none());
    let Some(LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    }) = final_prefill.logits_policy()
    else {
        panic!("final prefill must retain its exact sampling policy");
    };
    assert_eq!(mask.valid_token_mask.as_ref(), &[0, 1, 1, 0]);
    assert_eq!(
        product_output_mode_for_roles(VNextExecutionWaveKind::Prefill, [&intermediate]),
        VNextProductOutputMode::FullLogits,
    );
    assert_eq!(
        product_output_mode_for_roles(
            VNextExecutionWaveKind::Prefill,
            [&intermediate, &final_prefill],
        ),
        VNextProductOutputMode::GreedyToken,
    );
    assert_eq!(
        product_output_mode_for_roles(VNextExecutionWaveKind::Prefill, [&final_prefill, &host]),
        VNextProductOutputMode::FullLogits,
    );
    assert!(ExecutorSamplingOutput::greedy_token(TokenId::new(1))
        .validate_for_policy(&policy, 4)
        .is_ok());
    assert_eq!(
        normalized_product_token_mask(
            final_prefill.logits_policy(),
            VNextProductOutputMode::GreedyToken,
            4,
        ),
        vec![0, 1, 1, 0],
    );
}
