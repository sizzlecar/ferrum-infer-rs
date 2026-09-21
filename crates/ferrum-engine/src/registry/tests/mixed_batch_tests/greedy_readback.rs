use super::*;
use ferrum_interfaces::model_executor::{
    GreedyRepetitionPenalty, LogitsReturnPolicy, TokenSelectionMask,
};

/// Independent host oracle over the split executor's unprocessed logits.
pub(super) fn expected_token(logits: &[f32], policy: &LogitsReturnPolicy) -> TokenId {
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask,
        repetition_penalty,
    } = policy
    else {
        panic!("the oracle needs the actual greedy product policy")
    };
    let mut best = None;
    for (token, &logit) in logits.iter().enumerate() {
        if token_mask
            .as_ref()
            .is_some_and(|mask| mask.valid_token_mask.get(token).copied().unwrap_or(0) == 0)
        {
            continue;
        }
        let mut score = f64::from(logit);
        if let Some(repetition) = repetition_penalty {
            if repetition.token_ids().contains(&(token as u32)) {
                let penalty = f64::from(repetition.penalty());
                score = if score > 0.0 {
                    score / penalty
                } else {
                    score * penalty
                };
            }
        }
        if score.is_finite() && best.is_none_or(|(_, previous)| score > previous) {
            best = Some((token, score));
        }
    }
    TokenId::new(best.expect("the mask leaves a finite candidate").0 as u32)
}

#[tokio::test]
async fn real_cpu_mixed_greedy_readback_preserves_masks_repetition_and_product_frontiers() {
    let mixed = CpuFixture::new(8).await;
    let split = CpuFixture::new(8).await;
    let family = &mixed
        .executor
        .resolved_model_plan()
        .unwrap()
        .parts()
        .prepared_family;
    let logits_output = family
        .program()
        .outputs()
        .iter()
        .find(|output| output.as_str() == "value.output.logits")
        .expect("the Qwen fixture declares a logits output");
    let logits_element_bytes = family.numerical_profile().boundaries[logits_output].size_bytes();
    let mut mixed_decodes = vec![
        mixed.seed_decode(&[0, 1]).await,
        mixed.seed_decode(&[2, 0]).await,
    ];
    let mut split_decodes = vec![
        split.seed_decode(&[0, 1]).await,
        split.seed_decode(&[2, 0]).await,
    ];
    // Product order differs from canonical session authority order.
    mixed_decodes.reverse();
    split_decodes.reverse();
    let policies = [
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(TokenSelectionMask::new(vec![1, 0, 0])),
            repetition_penalty: None,
        },
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(10.0, vec![2, 2])),
        },
    ];
    let mut mixed_prefill = prompt(&[1, 0, 1, 2, 0, 1], 2);
    let mut split_prefill = prompt(&[1, 0, 1, 2, 0, 1], 2);
    mixed.admit(&mixed_prefill);
    split.admit(&split_prefill);

    for (chunk_index, force_host_decode) in [(0, false), (1, true), (2, false)] {
        let chunk = PrefillChunk::new(chunk_index * 2, 2, 6).unwrap();
        mixed_prefill.chunk = chunk;
        split_prefill.chunk = chunk;
        for (index, input) in mixed_decodes.iter_mut().enumerate() {
            input.logits_policy = if force_host_decode && index == 1 {
                LogitsReturnPolicy::FullLogits
            } else {
                policies[index].clone()
            };
        }
        let expected_prefill = split.prefill(&split_prefill).await;
        let expected_decodes = split_decode(&split, &split_decodes).await;
        let before = mixed.executor.cache_metrics_snapshot().unwrap();
        let (prefills, decodes) = match mixed
            .executor
            .plan_runtime_mixed_batch_with_capacity(
                std::slice::from_ref(&mixed_prefill),
                &mixed_decodes,
            )
            .await
            .unwrap()
        {
            PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes } => (prefills, decodes),
            _ => panic!("tiny CPU mixed wave did not complete"),
        };
        assert_eq!(prefills.len(), 1);
        assert_eq!(decodes.len(), mixed_decodes.len());
        prefills[0]
            .validate_for(&mixed_prefill.request_id, chunk, 3)
            .unwrap();
        assert_prefill_matches(&prefills[0], &expected_prefill);
        assert_eq!(prefills[0].output().committed_tokens(), chunk.end());
        assert_eq!(
            matches!(
                prefills[0].output().product(),
                PlanRuntimePrefillProduct::Intermediate
            ),
            !chunk.is_final(),
            "an internal readback must not become a sampled prefill product",
        );

        let greedy_readback = !chunk.is_final() && !force_host_decode;
        let after = mixed.executor.cache_metrics_snapshot().unwrap();
        let readback_bytes = after["counters"]["readback_bytes"].as_u64().unwrap()
            - before["counters"]["readback_bytes"].as_u64().unwrap();
        let participants = (mixed_decodes.len() + 1) as u64;
        // The complete readback batch includes the internally discarded
        // prefill participant: one U32 or one vocabulary row per member. The
        // declared device logits dtype can differ from the host Vec<f32>:
        // this CPU fixture reads F16 logits before converting them to F32.
        assert_eq!(
            readback_bytes,
            participants
                * if greedy_readback {
                    4
                } else {
                    3 * logits_element_bytes
                }
        );
        for index in 0..decodes.len() {
            let ExecutorSamplingOutput::FullLogits(reference_logits) =
                &expected_decodes[index].sampling_output
            else {
                panic!("split reference must retain unprocessed logits")
            };
            let token = expected_token(reference_logits, &policies[index]);
            if greedy_readback {
                assert_eq!(
                    decodes[index].sampling_output,
                    ExecutorSamplingOutput::GreedyToken(token)
                );
                if index == 1 {
                    let plain = LogitsReturnPolicy::GreedyArgmax {
                        token_mask: None,
                        repetition_penalty: None,
                    };
                    assert_ne!(
                        token,
                        expected_token(reference_logits, &plain),
                        "the fixture must exercise a repetition penalty that changes selection"
                    );
                }
            } else {
                assert_decode_matches(&decodes[index], &expected_decodes[index]);
            }
            assert_eq!(
                decodes[index].kv_cache.cache_id(),
                mixed_decodes[index].kv_cache.cache_id()
            );
            assert_eq!(
                decodes[index].kv_cache.num_tokens(),
                mixed_decodes[index].kv_cache.num_tokens() + 1
            );
            mixed_decodes[index] = PlanRuntimeDecodeInput::new(
                mixed_decodes[index].request_id.clone(),
                token,
                Arc::clone(&decodes[index].kv_cache),
            );
            split_decodes[index] = PlanRuntimeDecodeInput::new(
                split_decodes[index].request_id.clone(),
                token,
                Arc::clone(&expected_decodes[index].kv_cache),
            );
        }
        if chunk.is_final() {
            mixed
                .executor
                .release_cache(&prefills[0].output().kv_cache().cache_id());
            split
                .executor
                .release_cache(&expected_prefill.output().kv_cache().cache_id());
        }
    }
    for input in mixed_decodes {
        mixed.executor.release_cache(&input.kv_cache.cache_id());
    }
    for input in split_decodes {
        split.executor.release_cache(&input.kv_cache.cache_id());
    }
    for fixture in [&mixed, &split] {
        let snapshot = fixture.executor.cache_metrics_snapshot().unwrap();
        assert_eq!(snapshot["active_sequences"], 0);
        assert_eq!(snapshot["pending_sequences"], 0);
    }
}
