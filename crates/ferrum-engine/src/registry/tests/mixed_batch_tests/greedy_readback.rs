use super::*;
use ferrum_execution::model_executor::{GreedyRepetitionPenalty, TokenSelectionMask};

/// Independent host oracle over the split executor's unprocessed logits.
fn expected_token(logits: &[f32], policy: &LogitsReturnPolicy) -> TokenId {
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
    let mut mixed_decodes = [
        mixed.seed_decode(&[0, 1]).await,
        mixed.seed_decode(&[2, 0]).await,
    ];
    let mut split_decodes = [
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
    let full = LogitsReturnPolicy::FullLogits;
    let mut decode_tokens = [1, 1];
    let mut mixed_prefill = mixed.admit(&[1, 0, 1, 2, 0, 1]).await;
    let mut split_prefill = split.admit(&[1, 0, 1, 2, 0, 1]).await;

    for (chunk_index, force_host_decode) in [(0, false), (1, true), (2, false)] {
        let chunk = PrefillChunk::new(chunk_index * 2, 2, 6).unwrap();
        let expected_prefill = split.prefill(&split_prefill, chunk).await;
        let expected_decodes = split
            .execute(&[
                decode_work(&split_decodes[0], decode_tokens[0], &full),
                decode_work(&split_decodes[1], decode_tokens[1], &full),
            ])
            .await;
        let before = mixed.executor.cache_metrics_snapshot().unwrap();
        let second_policy = if force_host_decode {
            &full
        } else {
            &policies[1]
        };
        let output = mixed
            .execute(&[
                ExecutionWork::new(&mixed_prefill, TokenWork::Prefill(chunk), &full).unwrap(),
                decode_work(&mixed_decodes[0], decode_tokens[0], &policies[0]),
                decode_work(&mixed_decodes[1], decode_tokens[1], second_policy),
            ])
            .await;
        assert_eq!(output.len(), 3);
        let prefill = &output[0];
        let decodes = &output[1..];
        assert_prefill_matches(prefill, &expected_prefill);
        assert_eq!(prefill.next_frontier().consumed_tokens(), chunk.end());
        assert_eq!(
            matches!(prefill.product(), ExecutionProduct::Intermediate),
            !chunk.is_final(),
            "an internal readback must not become a sampled prefill product",
        );

        let greedy_readback = !chunk.is_final() && !force_host_decode;
        let after = mixed.executor.cache_metrics_snapshot().unwrap();
        assert_eq!(
            after["counters"]["submitted_waves"].as_u64().unwrap()
                - before["counters"]["submitted_waves"].as_u64().unwrap(),
            1
        );
        let readback_bytes = after["counters"]["readback_bytes"].as_u64().unwrap()
            - before["counters"]["readback_bytes"].as_u64().unwrap();
        let participants = (mixed_decodes.len() + 1) as u64;
        // The complete readback includes the internally discarded prefill row:
        // one U32 or one vocabulary row per member. The actual device logits
        // dtype can differ from the host Vec<f32> (this fixture uses F16).
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
                sampling(&expected_decodes[index])
            else {
                panic!("split reference must retain unprocessed logits")
            };
            let token = expected_token(reference_logits, &policies[index]);
            if greedy_readback {
                assert_eq!(
                    sampling(&decodes[index]),
                    &ExecutorSamplingOutput::GreedyToken(token)
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
                decodes[index].next_frontier().sequence(),
                mixed_decodes[index].sequence()
            );
            assert_eq!(
                decodes[index].next_frontier().consumed_tokens(),
                mixed_decodes[index].consumed_tokens() + 1
            );
            mixed_decodes[index] = decodes[index].next_frontier().clone();
            split_decodes[index] = expected_decodes[index].next_frontier().clone();
            decode_tokens[index] = token.get();
        }
        mixed_prefill = prefill.next_frontier().clone();
        split_prefill = expected_prefill.next_frontier().clone();
        if chunk.is_final() {
            mixed.close(&mixed_prefill).await;
            split.close(&split_prefill).await;
        }
    }
    for frontier in &mixed_decodes {
        mixed.close(frontier).await;
    }
    for frontier in &split_decodes {
        split.close(frontier).await;
    }
    mixed.assert_no_live_sequences();
    split.assert_no_live_sequences();
}
