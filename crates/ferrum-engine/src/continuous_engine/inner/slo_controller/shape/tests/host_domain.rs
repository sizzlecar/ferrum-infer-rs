use super::*;
use domain::{row_host, FutureHostMode};
use ferrum_interfaces::execution_cost::{
    satisfied_completion_cost_signature, CostSamplingHistoryScope, HostContentDomainV1,
    HostCostFeaturesV1, HostCostPolicyV2, HostCostStateV1,
};

fn host(n: u64, maximum: u64, pending: bool) -> HostCostFeaturesV1 {
    HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
            categorical_signature: [7; 32],
            decoder_text_bytes_per_token: 8,
            decoder_scratch_bytes_per_token: 4,
            raw_token_bytes_bound: 8,
        },
        state: HostCostStateV1 {
            generated_tokens_before: n,
            maximum_output_tokens: maximum,
            sampling_history_tokens: n,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: pending,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    }
}

#[test]
fn future_host_exact_first_wave_preserves_real_pending_and_has_no_guessed_projection() {
    let request = decode(4, 1, 8);
    let mut state = frontiers(std::slice::from_ref(&request));
    let original = host(1, 8, true);
    assert_eq!(
        row_host(Some(original), state.request(0), FutureHostMode::Exact).unwrap(),
        Some(Some(original))
    );
    assert_eq!(
        row_host(None, state.request(0), FutureHostMode::Exact).unwrap(),
        Some(None)
    );
    advance(&mut state, vec![action(&request, WaveAction::Decode)]);
    assert_eq!(
        row_host(Some(original), state.request(0), FutureHostMode::Exact).unwrap(),
        None
    );
    for mode in [FutureHostMode::Greedy, FutureHostMode::FullLogits] {
        let projected = row_host(Some(original), state.request(0), mode)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(projected.state.generated_tokens_before, 2);
        assert_eq!(projected.state.sampling_history_tokens, 2);
        assert_eq!(
            projected.state.pending_decoded_utf8,
            mode == FutureHostMode::FullLogits
        );
        assert_eq!(projected.policy, original.policy);
        assert_eq!(projected.state.maximum_output_tokens, 8);
        assert_eq!(
            projected.state.completion_state_signature,
            original.state.completion_state_signature
        );
    }
    assert_eq!(original.state.generated_tokens_before, 1);
    assert!(
        original.state.pending_decoded_utf8,
        "no live state was changed to obtain either hypothetical route"
    );
}

#[test]
fn future_host_partial_final_decode_use_advanced_history_and_untouched_peer() {
    let requests = [prefill(4, 0, 0), decode(7, 2, 8)];
    let original = [host(0, 8, false), host(2, 8, true)];
    let mut state = frontiers(&requests);
    advance(&mut state, vec![action(&requests[0], chunk(0, 2))]);
    assert_eq!(
        row_host(Some(original[0]), state.request(0), FutureHostMode::Exact).unwrap(),
        Some(Some(original[0]))
    );
    advance(&mut state, vec![action(&requests[0], chunk(2, 2))]);
    for mode in [FutureHostMode::Greedy, FutureHostMode::FullLogits] {
        let next = row_host(Some(original[0]), state.request(0), mode)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                next.state.generated_tokens_before,
                next.state.sampling_history_tokens
            ),
            (1, 1)
        );
        assert_eq!(state.request(0).context_tokens, 4);
        assert_eq!(
            row_host(Some(original[1]), state.request(1), mode).unwrap(),
            Some(Some(original[1])),
            "unchanged owner keeps its actual UTF-8 state"
        );
    }
    advance(
        &mut state,
        vec![
            action(&requests[0], WaveAction::Decode),
            action(&requests[1], WaveAction::Decode),
        ],
    );
    for (index, n, kv) in [(0, 2, 5), (1, 3, 8)] {
        let next = row_host(
            Some(original[index]),
            state.request(index),
            FutureHostMode::Greedy,
        )
        .unwrap()
        .unwrap()
        .unwrap();
        assert_eq!(
            (
                next.state.generated_tokens_before,
                next.state.sampling_history_tokens
            ),
            (n, n)
        );
        assert_eq!(state.request(index).context_tokens, kv);
    }
}

#[test]
fn future_host_unsupported_completion_scope_and_policy_remain_unknown() {
    let request = decode(4, 1, 8);
    let mut state = frontiers(std::slice::from_ref(&request));
    advance(&mut state, vec![action(&request, WaveAction::Decode)]);
    let valid = host(1, 8, false);
    let mut no_domain = valid;
    no_domain.policy.empirical_content_domain = None;
    let mut completion = valid;
    completion.state.completion_state_signature = [0; 32];
    let mut history = valid;
    history.state.sampling_history_scope = CostSamplingHistoryScope::VisibleStructuredOutput;
    let mut stale = valid;
    stale.state.sampling_history_tokens = 0;
    let mut changed_maximum = valid;
    changed_maximum.state.maximum_output_tokens = 9;
    for original in [
        None,
        Some(no_domain),
        Some(completion),
        Some(history),
        Some(stale),
        Some(changed_maximum),
    ] {
        for mode in [FutureHostMode::Greedy, FutureHostMode::FullLogits] {
            assert_eq!(row_host(original, state.request(0), mode).unwrap(), None);
        }
    }
}

#[test]
fn future_host_recompute_retains_original_reference_while_advancing_output_history() {
    let request = prefill(5, 1, 5);
    let original = host(1, 8, false);
    let mut state = frontiers(std::slice::from_ref(&request));
    advance(&mut state, vec![action(&request, chunk(0, 2))]);
    assert_eq!(
        row_host(Some(original), state.request(0), FutureHostMode::Exact).unwrap(),
        Some(Some(original))
    );
    let RequestPhaseView::Prefill(progress) = &state.request(0).phase else {
        panic!("partial recompute")
    };
    assert_eq!(
        (progress.logical_high_water, progress.admitted_at_ns),
        (5, 13)
    );
    advance(&mut state, vec![action(&request, chunk(2, 3))]);
    let next = row_host(Some(original), state.request(0), FutureHostMode::FullLogits)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        (
            next.state.generated_tokens_before,
            next.state.sampling_history_tokens
        ),
        (2, 2)
    );
    assert_eq!(state.request(0).context_tokens, 5);
}
