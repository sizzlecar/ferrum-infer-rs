//! Retained mixed-role guards over the real common token execution boundary.
use super::*;
use ferrum_execution::model_executor::TokenSelectionMask;

#[tokio::test]
async fn mixed_frontiers_reject_the_same_incarnation_in_both_phases_before_state_changes() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let request = RequestId::new();
    let tokens = [TokenId::new(0), TokenId::new(1)];
    let frontier = admit_tokens(&fixture, &request, &tokens).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let before = fixture.executor.cache_metrics_snapshot().unwrap();
    let work = [
        ExecutionWork::new(
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 1, 2).unwrap()),
            &policy,
        )
        .unwrap(),
        ExecutionWork::new(
            &frontier,
            TokenWork::Decode {
                token: TokenId::new(1),
                position: 0,
            },
            &policy,
        )
        .unwrap(),
    ];
    assert!(fixture.executor.execute_tokens(&work).await.is_err());
    assert_no_adapter_execution(&fixture.executor, &before);
    let completion = execute_one(
        &fixture,
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
    )
    .await;
    assert_eq!(completion.next_frontier().consumed_tokens(), 2);
    close(&fixture.executor, frontier.sequence()).await;
}

#[tokio::test]
async fn mixed_frontiers_reject_duplicate_decodes_and_duplicate_prefills() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let request = RequestId::new();
    let tokens = [TokenId::new(0)];
    let frontier = admit_tokens(&fixture, &request, &tokens).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let work = TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap());
    let before = fixture.executor.cache_metrics_snapshot().unwrap();
    assert!(fixture
        .executor
        .execute_tokens(&[
            ExecutionWork::new(&frontier, work, &policy).unwrap(),
            ExecutionWork::new(&frontier, work, &policy).unwrap(),
        ])
        .await
        .is_err());
    assert_no_adapter_execution(&fixture.executor, &before);
    let prefill = execute_one(&fixture, &frontier, work).await;
    let next = prefill.next_frontier();
    let decode = TokenWork::Decode {
        token: TokenId::new(1),
        position: 1,
    };
    let before = fixture.executor.cache_metrics_snapshot().unwrap();
    assert!(fixture
        .executor
        .execute_tokens(&[
            ExecutionWork::new(next, decode, &policy).unwrap(),
            ExecutionWork::new(next, decode, &policy).unwrap(),
        ])
        .await
        .is_err());
    assert_no_adapter_execution(&fixture.executor, &before);
    let output = execute_one(&fixture, next, decode).await;
    assert_eq!(
        sampling(&output),
        &ExecutorSamplingOutput::FullLogits(vec![3.0, 2.0, 1.0, -4.0])
    );
    close(&fixture.executor, next.sequence()).await;
}

#[test]
fn mixed_readback_roles_preserve_final_logits_and_decode_processors() {
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
