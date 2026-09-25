use super::*;
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::{InferenceRequest, ResponseFormat, SamplingParams, TokenId};
use std::sync::Arc;

async fn tokenizer() -> Arc<HuggingFaceTokenizer> {
    let vocabulary: tokenizers::models::bpe::Vocab =
        ["a", "b", "{", "}", "\"", ":", "0", "true", "</s>"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect();
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    );
    tokenizer.add_special_tokens(&[tokenizers::AddedToken::from("</s>", true)]);
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    Arc::new(HuggingFaceTokenizer::new(tokenizer).await.unwrap())
}

fn sequence(tokenizer: Arc<HuggingFaceTokenizer>, sampling: SamplingParams) -> SequenceState {
    let mut request = InferenceRequest::new("first prompt", "policy-model");
    request.sampling_params = sampling;
    sequence_from_request(tokenizer, request)
}

fn sequence_from_request(
    tokenizer: Arc<HuggingFaceTokenizer>,
    request: InferenceRequest,
) -> SequenceState {
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    state.stream_sender = Some(tokio::sync::mpsc::channel(1).0);
    state
}

#[tokio::test]
async fn host_policy_excludes_request_prompt_seed_and_allocation_identity() {
    let tokenizer = tokenizer().await;
    let sampling = SamplingParams::greedy();
    let first = sequence(tokenizer.clone(), sampling.clone());
    let mut second = sequence(tokenizer.clone(), sampling);
    second.original_request.prompt = "different prompt with different length".to_owned();
    second.input_tokens = vec![TokenId::new(1); 7];
    second.sampling_params.seed = Some(987);
    second.original_request.sampling_params.seed = Some(987);
    assert_ne!(first.request_id, second.request_id);
    let signature = host_policy_signature(&first, tokenizer.as_ref()).unwrap();
    assert_eq!(
        Some(signature),
        host_policy_signature(&second, tokenizer.as_ref())
    );
}

#[tokio::test]
async fn host_policy_uses_actual_legacy_sender_and_unknown_sources_do_not_train() {
    let tokenizer = tokenizer().await;
    let mut state = sequence(tokenizer.clone(), SamplingParams::greedy());
    let streaming = host_policy_signature(&state, tokenizer.as_ref()).unwrap();
    state.stream_sender = None;
    assert!(host_policy_signature(&state, tokenizer.as_ref()).is_none());
    state.response_sender = Some(tokio::sync::oneshot::channel().0);
    let response = host_policy_signature(&state, tokenizer.as_ref()).unwrap();
    assert_ne!(streaming, response);
    // The request's stream bit is not authority for the installed transport.
    state.original_request.stream = !state.original_request.stream;
    assert_eq!(
        Some(response),
        host_policy_signature(&state, tokenizer.as_ref())
    );
    let unknown = ferrum_testkit::MockTokenizer::new(8);
    assert!(host_policy_signature(&state, &unknown).is_none());
}

#[tokio::test]
async fn host_policy_changes_with_real_sampling_and_resolved_constraints() {
    let tokenizer = tokenizer().await;
    let mut first = sequence(tokenizer.clone(), SamplingParams::greedy());
    let sampling = SamplingParams {
        top_k: Some(3),
        ..SamplingParams::greedy()
    };
    let mut second = sequence(tokenizer.clone(), sampling);
    assert_ne!(
        host_policy_signature(&first, tokenizer.as_ref()),
        host_policy_signature(&second, tokenizer.as_ref())
    );
    // A parameter mutation without updating the installed plan is unknown.
    second.sampling_params.top_k = None;
    assert!(host_policy_signature(&second, tokenizer.as_ref()).is_none());
    let mut second = sequence(tokenizer.clone(), SamplingParams::greedy());
    first.initial_forbidden_token_ids.extend([3, 5]);
    second.initial_forbidden_token_ids.extend([5, 3]);
    assert_eq!(
        host_policy_signature(&first, tokenizer.as_ref()),
        host_policy_signature(&second, tokenizer.as_ref())
    );
    second.initial_forbidden_token_ids.insert(6);
    assert_ne!(
        host_policy_signature(&first, tokenizer.as_ref()),
        host_policy_signature(&second, tokenizer.as_ref())
    );
}

#[tokio::test]
async fn host_policy_includes_the_compiled_structured_contract() {
    let tokenizer = tokenizer().await;
    let ordinary = sequence(tokenizer.clone(), SamplingParams::greedy());
    let mut structured = sequence(
        tokenizer.clone(),
        SamplingParams {
            response_format: ResponseFormat::JsonObject,
            ..SamplingParams::greedy()
        },
    );
    assert!(structured.structured_output_processor.is_some());
    assert_ne!(
        host_policy_signature(&ordinary, tokenizer.as_ref()).unwrap(),
        host_policy_signature(&structured, tokenizer.as_ref()).unwrap()
    );
    structured.structured_output_processor = None;
    assert!(host_policy_signature(&structured, tokenizer.as_ref()).is_none());
}

fn credited_sequence(
    tokenizer: Arc<HuggingFaceTokenizer>,
    sampling: SamplingParams,
) -> (
    SequenceState,
    ferrum_interfaces::output_flow::CreditedOutputSession,
) {
    let mut request = InferenceRequest::new("first prompt", "policy-model");
    request.sampling_params = sampling;
    credited_sequence_from_request(tokenizer, request)
}

fn credited_sequence_from_request(
    tokenizer: Arc<HuggingFaceTokenizer>,
    request: InferenceRequest,
) -> (
    SequenceState,
    ferrum_interfaces::output_flow::CreditedOutputSession,
) {
    use crate::continuous_engine::credited_output::{CreditedDecodePolicy, CreditedSequenceOutput};
    use crate::continuous_engine::output_flow_runtime::{
        spawn_output_flow_runtime, OutputFlowRuntimeOptions,
    };
    use ferrum_interfaces::output_credit::{
        OutputAccountLimits, OutputCreditPool, OutputPoolLimits,
    };
    use ferrum_interfaces::output_flow::{
        OutputProjectionContract, RequestOutputBudget, RequestOutputPlan,
    };
    use std::num::NonZeroUsize;
    let mut state = sequence_from_request(tokenizer.clone(), request);
    state.stream_sender = None;
    let plan = RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::cli_text()),
        tokenizer.as_ref(),
        &state.original_request,
        state.input_tokens.len(),
    )
    .unwrap();
    let config = ferrum_types::SloOutputConfig::default();
    let decoder = CreditedDecodePolicy::from_plan(&plan);
    let limits = OutputAccountLimits::from_slo(
        &config,
        NonZeroUsize::new(plan.terminal_credit().events).unwrap(),
    )
    .unwrap();
    let pool = OutputCreditPool::new(
        OutputPoolLimits::from_slo(&config, NonZeroUsize::new(1).unwrap()).unwrap(),
    )
    .unwrap();
    let budget = RequestOutputBudget::open(&pool, limits, plan).unwrap();
    let (port, session) = spawn_output_flow_runtime(
        budget,
        Arc::new(tokio::sync::Notify::new()),
        OutputFlowRuntimeOptions::from_config(&config),
    );
    state.credited_output = Some(CreditedSequenceOutput {
        port,
        decoder,
        grant: None,
        tokens_before_grant: 0,
        accepted_ordinal: 0,
        deferred: None,
        failure: None,
    });
    (state, session)
}

fn fixed_output_request(sampling: SamplingParams) -> InferenceRequest {
    let mut request = InferenceRequest::new("first prompt", "policy-model");
    request.sampling_params = sampling;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), serde_json::Value::Bool(true));
    request
}

#[tokio::test]
async fn real_bounded_owner_splits_lengths_from_static_policy_without_changing_v1() {
    use ferrum_interfaces::execution_cost::*;
    let tokenizer = tokenizer().await;
    let (mut first, _first_session) = credited_sequence(
        tokenizer.clone(),
        SamplingParams {
            max_tokens: 8,
            ..SamplingParams::greedy()
        },
    );
    let (second, _second_session) = credited_sequence(
        tokenizer.clone(),
        SamplingParams {
            max_tokens: 20,
            seed: Some(99),
            ..SamplingParams::greedy()
        },
    );
    assert_ne!(
        host_policy_signature(&first, tokenizer.as_ref()),
        host_policy_signature(&second, tokenizer.as_ref())
    );
    let policy = host_numeric_policy(&first, tokenizer.as_ref()).unwrap();
    assert_eq!(
        policy,
        host_numeric_policy(&second, tokenizer.as_ref()).unwrap()
    );
    first.cost_numeric_policy = Some(policy);
    for _ in 0..3 {
        first.generated_tokens.push(TokenId::new(0));
        first.sampling_history.record(TokenId::new(0));
    }
    let captured = super::super::participant_host_features(&first).unwrap();
    assert_eq!(captured.state.generated_tokens_before, 3);
    assert_eq!(captured.state.sampling_history_tokens, 3);
    let numbers = project_host_cost_features(
        captured,
        ActualRowWork::Decode { kv_tokens: 12 },
        CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 0,
            repetition_penalty_bits: 1f32.to_bits(),
        },
    )
    .unwrap();
    let required = tokenizer
        .bounded_decode_bound()
        .unwrap()
        .requirements(4)
        .unwrap();
    assert_eq!(numbers.decoded_prefix_tokens, 4);
    assert_eq!(numbers.decoded_text_bytes_bound, required.text_bytes as u64);
    assert_eq!(
        numbers.decode_scratch_bytes_bound,
        required.scratch_bytes as u64
    );
    assert!(numbers.decoded_text_bytes_bound > 0);
    first.pending_decoded_utf8_fragment = true;
    assert!(
        super::super::participant_host_features(&first)
            .unwrap()
            .state
            .pending_decoded_utf8
    );
}

#[tokio::test]
async fn empirical_content_domain_uses_installed_plain_greedy_policy_and_completion() {
    let tokenizer = tokenizer().await;
    let (mut state, _session) = credited_sequence_from_request(
        tokenizer.clone(),
        fixed_output_request(SamplingParams::greedy()),
    );
    assert!(state.model_eos_token_ids.is_empty());
    assert!(state.stop_token_ids.is_empty());
    assert_eq!(
        host_numeric_policy(&state, tokenizer.as_ref())
            .unwrap()
            .empirical_content_domain,
        Some(HostContentDomainV1::PlainTextGreedyV1)
    );
    // Unknown future bytes are a statistical disturbance in this domain.
    // They must not be confused with an unsupported installed algorithm.
    state.pending_decoded_utf8_fragment = true;
    assert!(host_numeric_policy(&state, tokenizer.as_ref())
        .unwrap()
        .empirical_content_domain
        .is_some());
    state.stop_token_ids.insert(8);
    assert!(host_numeric_policy(&state, tokenizer.as_ref())
        .unwrap()
        .empirical_content_domain
        .is_none());
    state.stop_token_ids.clear();
    state.sampling_params.response_completion_boundary =
        ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".into(),
            alternate_envelope: None,
        };
    assert!(host_numeric_policy(&state, tokenizer.as_ref())
        .unwrap()
        .empirical_content_domain
        .is_none());
    state.sampling_params.response_completion_boundary =
        ferrum_types::ResponseCompletionBoundary::Immediate;
    state.sampling_params.repetition_penalty = 1.1;
    assert!(empirical_content_domain(&state).is_none());
}

#[tokio::test]
async fn plain_text_domain_normal_constructor_resolves_length_only_without_early_eos() {
    use ferrum_types::FinishReason;
    let tokenizer = tokenizer().await;
    let sampling = SamplingParams {
        max_tokens: 3,
        ..SamplingParams::greedy()
    };
    let (mut automatic, _automatic_session) =
        credited_sequence(tokenizer.clone(), sampling.clone());
    let eos = *automatic
        .model_eos_token_ids
        .first()
        .expect("real tokenizer EOS");
    assert!(automatic
        .model_eos_token_ids
        .iter()
        .all(|id| automatic.stop_token_ids.contains(id)));
    assert!(host_numeric_policy(&automatic, tokenizer.as_ref())
        .unwrap()
        .empirical_content_domain
        .is_none());
    automatic.generated_tokens.push(TokenId::new(eos));
    assert_eq!(
        automatic.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::EOS)
    );

    let (mut fixed, _fixed_session) =
        credited_sequence_from_request(tokenizer.clone(), fixed_output_request(sampling));
    assert!(fixed.model_eos_token_ids.is_empty());
    assert!(fixed.stop_token_ids.is_empty());
    assert!(fixed.user_stop_token_ids.is_empty());
    assert!(fixed.stop_text_seqs.is_empty());
    assert_eq!(
        host_numeric_policy(&fixed, tokenizer.as_ref())
            .unwrap()
            .empirical_content_domain,
        Some(HostContentDomainV1::PlainTextGreedyV1)
    );
    // The same tokenizer terminal is ordinary generated content under the
    // effective request policy. Only the actual output budget completes it.
    for token in [eos, 0] {
        fixed.generated_tokens.push(TokenId::new(token));
        assert_eq!(fixed.stop_reason(Some(tokenizer.as_ref())), None);
    }
    fixed.generated_tokens.push(TokenId::new(1));
    assert_eq!(
        fixed.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::Length)
    );
}

#[tokio::test]
async fn plain_text_domain_ignore_eos_preserves_user_stops_and_structured_boundaries() {
    use ferrum_types::FinishReason;
    let tokenizer = tokenizer().await;
    for (stop, tokens) in [("a", vec![0]), ("ab", vec![0, 1])] {
        let (mut state, _session) = credited_sequence_from_request(
            tokenizer.clone(),
            fixed_output_request(SamplingParams {
                stop_sequences: vec![stop.into()],
                ..SamplingParams::greedy()
            }),
        );
        assert!(state.model_eos_token_ids.is_empty());
        assert_eq!(state.stop_text_seqs, vec![stop]);
        assert_eq!(state.user_stop_token_ids.contains(&0), tokens.len() == 1);
        assert!(host_numeric_policy(&state, tokenizer.as_ref())
            .unwrap()
            .empirical_content_domain
            .is_none());
        state
            .generated_tokens
            .extend(tokens.into_iter().map(TokenId::new));
        assert_eq!(
            state.stop_reason(Some(tokenizer.as_ref())),
            Some(FinishReason::Stop)
        );
    }
    let structured = sequence_from_request(
        tokenizer.clone(),
        fixed_output_request(SamplingParams {
            response_format: ResponseFormat::JsonObject,
            ..SamplingParams::greedy()
        }),
    );
    assert!(structured.model_eos_token_ids.is_empty());
    assert!(structured.stop_token_ids.is_empty());
    assert!(structured.structured_output_processor.is_some());
    // Structured output is installed by the real sequence constructor. The
    // CLI text codec does not prove a structured projection, so this negative
    // domain test must not invent a credited output owner for it.
    assert!(empirical_content_domain(&structured).is_none());
}

#[tokio::test]
async fn plain_text_domain_rejects_inconsistent_resolved_eos_state() {
    use ferrum_types::FinishReason;
    let tokenizer = tokenizer().await;
    let eos = tokenizer.token_id("</s>").unwrap().get();
    let (mut state, _session) = credited_sequence_from_request(
        tokenizer.clone(),
        fixed_output_request(SamplingParams::greedy()),
    );
    assert!(state.stop_token_ids.is_empty());
    // Deliberate negative corruption: only constructor-produced states are
    // used as positives. stop_reason reads this resolved field independently.
    state.model_eos_token_ids.push(eos);
    state.generated_tokens.push(TokenId::new(eos));
    assert_eq!(
        state.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::EOS)
    );
    assert!(host_numeric_policy(&state, tokenizer.as_ref())
        .unwrap()
        .empirical_content_domain
        .is_none());
}

#[tokio::test]
async fn numeric_policy_keeps_stop_sampling_and_actual_completion_branches() {
    use crate::continuous_engine::{DelimitedPayloadCompletionState, ResponseCompletionState};
    let tokenizer = tokenizer().await;
    let (mut first, _session) = credited_sequence(
        tokenizer.clone(),
        SamplingParams {
            max_tokens: 20,
            ..SamplingParams::greedy()
        },
    );
    let baseline = host_numeric_policy(&first, tokenizer.as_ref()).unwrap();
    first.cost_numeric_policy = Some(baseline);
    let satisfied = super::super::participant_host_features(&first).unwrap();
    first.response_completion_state = ResponseCompletionState::Pending {
        delimited_payload: DelimitedPayloadCompletionState::AwaitingPayload,
        alternate_envelope: None,
    };
    assert_ne!(
        satisfied.state.completion_state_signature,
        super::super::participant_host_features(&first)
            .unwrap()
            .state
            .completion_state_signature
    );
    first.stop_text_seqs.push("ab".into());
    assert_ne!(
        baseline.categorical_signature,
        host_numeric_policy(&first, tokenizer.as_ref())
            .unwrap()
            .categorical_signature
    );
    let (penalty, _session) = credited_sequence(
        tokenizer.clone(),
        SamplingParams {
            max_tokens: 20,
            repetition_penalty: 1.2,
            ..SamplingParams::greedy()
        },
    );
    assert_ne!(
        baseline.categorical_signature,
        host_numeric_policy(&penalty, tokenizer.as_ref())
            .unwrap()
            .categorical_signature
    );
    let legacy = sequence(tokenizer.clone(), SamplingParams::greedy());
    assert!(host_policy_signature(&legacy, tokenizer.as_ref()).is_some());
    assert!(host_numeric_policy(&legacy, tokenizer.as_ref()).is_none());
}
