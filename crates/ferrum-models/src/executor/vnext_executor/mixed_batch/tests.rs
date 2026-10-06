use super::*;
use ferrum_interfaces::model_executor::TokenSelectionMask;
use ferrum_testkit::MockKvCacheHandle;

fn prefill(request_id: RequestId, chunk: PrefillChunk) -> PlanRuntimePrefillInput {
    let tokens = (0..chunk.total_prompt_tokens())
        .map(|token| TokenId::new(token as u32))
        .collect::<Vec<_>>();
    PlanRuntimePrefillInput::new(request_id, tokens, 32, chunk).unwrap()
}

fn decode(request_id: RequestId) -> PlanRuntimeDecodeInput {
    PlanRuntimeDecodeInput::new(
        request_id.clone(),
        TokenId::new(3),
        Arc::new(MockKvCacheHandle::new(request_id, 1, 4)),
    )
}

#[test]
fn mixed_frontiers_reject_the_same_request_in_both_phases_before_state_changes() {
    let id = RequestId::new();
    let prompt = prefill(id.clone(), PrefillChunk::new(0, 2, 4).unwrap());
    let continuation = decode(id);
    assert!(validate_mixed_request_ids(&[prompt], &[continuation]).is_err());
}

#[test]
fn mixed_frontiers_reject_duplicate_decodes_and_duplicate_prefills() {
    let prompt = prefill(RequestId::new(), PrefillChunk::new(0, 2, 4).unwrap());
    let continuation = decode(RequestId::new());
    assert!(validate_mixed_request_ids(
        &[prompt.clone(), prompt.clone()],
        std::slice::from_ref(&continuation),
    )
    .is_err());
    assert!(validate_mixed_request_ids(&[prompt], &[continuation.clone(), continuation]).is_err());
}

#[test]
fn mixed_readback_roles_preserve_final_logits_and_decode_processors() {
    let final_prompt = prefill(RequestId::new(), PrefillChunk::new(2, 3, 5).unwrap());
    let intermediate_prompt = prefill(RequestId::new(), PrefillChunk::new(0, 4, 9).unwrap());
    let continuation = decode(RequestId::new());
    validate_mixed_request_ids(&[final_prompt, intermediate_prompt], &[continuation]).unwrap();

    let greedy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1, 1])),
        repetition_penalty: None,
    };
    let intermediate = VNextParticipantOutputRole::IntermediatePrefill;
    let final_prefill = VNextParticipantOutputRole::FinalPrefill;
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
    let intermediate = VNextParticipantOutputRole::prefill(PrefillChunk::new(0, 2, 4).unwrap());
    let final_prefill = VNextParticipantOutputRole::prefill(PrefillChunk::new(2, 2, 4).unwrap());
    assert!(intermediate.matches_frontier(VNextExecutionWaveKind::Mixed, &intermediate_span, 4));
    assert!(!intermediate.matches_frontier(VNextExecutionWaveKind::Mixed, &final_span, 4));
    assert!(final_prefill.matches_frontier(VNextExecutionWaveKind::Mixed, &final_span, 4));
    assert!(!final_prefill.matches_frontier(VNextExecutionWaveKind::Mixed, &intermediate_span, 4));
}
