//! Real model continuation checks: an already executed child must preserve
//! its row and frontier while the next host batch changes order and policy.
use super::*;
use ferrum_interfaces::model_executor::{
    GreedyRepetitionPenalty, LogitsReturnPolicy, OneStepDecodeGrant, TokenSelectionMask,
};

fn selected_policy(token: usize) -> LogitsReturnPolicy {
    let mut mask = vec![0; 3];
    mask[token] = 1;
    LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(mask)),
        repetition_penalty: Some(GreedyRepetitionPenalty::new(1.5, vec![token as u32])),
    }
}

fn continuation(
    input: &PlanRuntimeDecodeInput,
    output: &PlanRuntimeDecodeOutput,
    token: u32,
) -> PlanRuntimeDecodeInput {
    PlanRuntimeDecodeInput::new(
        input.request_id.clone(),
        TokenId::new(token),
        Arc::clone(&output.kv_cache),
    )
}

fn authorize(input: &mut PlanRuntimeDecodeInput, token: usize) {
    input.logits_policy = selected_policy(token);
    input.lookahead = OneStepDecodeGrant::for_input(input, 4);
    assert!(input.lookahead.is_some());
}

fn release(fixture: &CpuFixture, outputs: &[PlanRuntimeDecodeOutput]) {
    for output in outputs {
        fixture.executor.release_cache(&output.kv_cache.cache_id());
    }
}

#[tokio::test]
async fn real_cpu_lookahead_pending_subset_and_fresh_rows_match_serial_after_policy_change() {
    let paired = CpuFixture::new(8).await;
    let serial = CpuFixture::new(8).await;
    let mut parents = vec![
        paired.seed_decode(&[0, 1]).await,
        paired.seed_decode(&[2, 0, 1]).await,
    ];
    let serial_parents = vec![
        serial.seed_decode(&[0, 1]).await,
        serial.seed_decode(&[2, 0, 1]).await,
    ];
    let fresh = paired.seed_decode(&[1, 2]).await;
    let serial_fresh = serial.seed_decode(&[1, 2]).await;
    authorize(&mut parents[0], 0);
    authorize(&mut parents[1], 2);
    let before = paired.submissions();
    let parents_out = split_decode(&paired, &parents).await;
    assert_eq!(
        paired.submissions() - before,
        2,
        "the test requires a submitted successor, not a serial fallback"
    );
    let serial_parents_out = split_decode(&serial, &serial_parents).await;
    for (row, token) in [0, 2].into_iter().enumerate() {
        assert_eq!(
            parents_out[row].sampling_output,
            ExecutorSamplingOutput::GreedyToken(TokenId::new(token))
        );
        assert_eq!(
            parents_out[row].kv_cache.num_tokens(),
            parents[row].kv_cache.num_tokens() + 1
        );
    }

    // An incorrect token cannot consume the valid pending row or execute a
    // replacement forward. The subsequent correct request must still work.
    let wrong = continuation(&parents[1], &parents_out[1], 1);
    let before = paired.submissions();
    assert!(paired
        .executor
        .plan_runtime_batch_decode_with_capacity(&[wrong])
        .await
        .is_err());
    assert_eq!(paired.submissions(), before);
    let pending_b = continuation(&parents[1], &parents_out[1], 2);
    let serial_b = continuation(&serial_parents[1], &serial_parents_out[1], 2);
    let b_out = split_decode(&paired, &[pending_b]).await;
    let serial_b_out = split_decode(&serial, &[serial_b]).await;
    assert_eq!(
        paired.submissions(),
        before,
        "a cached row must not execute twice"
    );
    assert_decode_matches(&b_out[0], &serial_b_out[0]);

    // Only one member of the old cohort remains. The fresh row comes first
    // in caller order; its policy differs from the cached row's new policy.
    let mut pending_a = continuation(&parents[0], &parents_out[0], 0);
    pending_a.logits_policy = selected_policy(1);
    let serial_a = continuation(&serial_parents[0], &serial_parents_out[0], 0);
    let mixed_inputs = [fresh, pending_a];
    let serial_inputs = [serial_fresh, serial_a];
    let before = paired.submissions();
    let outputs = split_decode(&paired, &mixed_inputs).await;
    let reference = split_decode(&serial, &serial_inputs).await;
    assert_eq!(
        paired.submissions() - before,
        1,
        "only the fresh row executes"
    );
    for row in 0..2 {
        assert_decode_matches(&outputs[row], &reference[row]);
        assert_eq!(
            outputs[row].kv_cache.cache_id(),
            mixed_inputs[row].kv_cache.cache_id()
        );
        outputs[row]
            .sampling_output
            .validate_for_policy(&mixed_inputs[row].logits_policy, 3)
            .unwrap();
    }
    assert!(
        matches!(
            outputs[1].sampling_output,
            ExecutorSamplingOutput::FullLogits(_)
        ),
        "the child must preserve raw logits for the new host mask and penalty"
    );

    // All rows are fresh again. This also checks that consuming a child
    // advanced both the model history and its native state exactly once.
    let next = [
        continuation(&parents[1], &b_out[0], 1),
        continuation(&mixed_inputs[1], &outputs[1], 1),
        continuation(&mixed_inputs[0], &outputs[0], 2),
    ];
    let serial_next = [
        continuation(&serial_parents[1], &serial_b_out[0], 1),
        continuation(&serial_inputs[1], &reference[1], 1),
        continuation(&serial_inputs[0], &reference[0], 2),
    ];
    let outputs = split_decode(&paired, &next).await;
    let reference = split_decode(&serial, &serial_next).await;
    for (actual, expected) in outputs.iter().zip(&reference) {
        assert_decode_matches(actual, expected);
    }
    release(&paired, &outputs);
    release(&serial, &reference);
}

#[tokio::test]
async fn real_cpu_lookahead_mixed_fallback_preserves_prefill_and_pending_decode() {
    let paired = CpuFixture::new(8).await;
    let serial = CpuFixture::new(8).await;
    let mut parent = paired.seed_decode(&[0, 1]).await;
    let serial_parent = serial.seed_decode(&[0, 1]).await;
    authorize(&mut parent, 1);
    let before = paired.submissions();
    let parent_out = split_decode(&paired, std::slice::from_ref(&parent)).await;
    assert_eq!(paired.submissions() - before, 2);
    let serial_parent_out = split_decode(&serial, std::slice::from_ref(&serial_parent)).await;
    let pending = continuation(&parent, &parent_out[0], 1);
    let serial_pending = continuation(&serial_parent, &serial_parent_out[0], 1);
    let prefill = prompt(&[1, 0, 2, 1], 2);
    let serial_prefill = prompt(&[1, 0, 2, 1], 2);
    paired.admit(&prefill);
    serial.admit(&serial_prefill);
    let before = paired.submissions();
    assert!(matches!(
        paired
            .executor
            .plan_runtime_mixed_batch_with_capacity(
                std::slice::from_ref(&prefill),
                std::slice::from_ref(&pending),
            )
            .await
            .unwrap(),
        PlanRuntimeMixedBatchOutcome::Unsupported
    ));
    assert_eq!(paired.submissions(), before);
    let prefill_out = paired.prefill(&prefill).await;
    let serial_prefill_out = serial.prefill(&serial_prefill).await;
    assert_prefill_matches(&prefill_out, &serial_prefill_out);
    let before = paired.submissions();
    let outputs = split_decode(&paired, &[pending]).await;
    let reference = split_decode(&serial, &[serial_pending]).await;
    assert_eq!(paired.submissions(), before);
    assert_decode_matches(&outputs[0], &reference[0]);
    release(&paired, &outputs);
    release(&serial, &reference);
    paired
        .executor
        .release_cache(&prefill_out.output().kv_cache().cache_id());
    serial
        .executor
        .release_cache(&serial_prefill_out.output().kv_cache().cache_id());
}
