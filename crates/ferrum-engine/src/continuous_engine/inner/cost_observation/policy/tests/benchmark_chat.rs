//! Installed-policy half of the proof. The CLI converter test establishes the
//! HTTP fields; this module starts from typed ordinary Chat requests and uses
//! real credited output admission, host policy and checked canonical inputs.
//! Its fixed CPU algorithm is not evidence of a CUDA route or model fit.
use super::*;
use ferrum_interfaces::vnext::DeviceCommandPhase;
use ferrum_interfaces::{execution_cost::*, output_flow::OutputProjectionContract};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2;
use ferrum_types::{ApiChatRequest, ApiRequest, ApiStreamOptions};
use std::num::{NonZeroU32, NonZeroU64};

fn typed_chat_request(usage: bool, maximum: usize, seed: u64) -> InferenceRequest {
    let mut request = fixed_output_request(SamplingParams {
        max_tokens: maximum,
        seed: Some(seed),
        ..SamplingParams::greedy()
    });
    request.stream = true;
    request.metadata.insert(
        ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY.into(),
        false.into(),
    );
    request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        tool_call_protocol: Default::default(),
        legacy_functions: Vec::new(),
        legacy_function_call: None,
        response_format: None,
        stream_options: Some(ApiStreamOptions {
            include_usage: Some(usage),
        }),
    }));
    request
}

fn chat_sequence(
    tokenizer: Arc<HuggingFaceTokenizer>,
    usage: bool,
    maximum: usize,
    seed: u64,
) -> (
    SequenceState,
    ferrum_interfaces::output_flow::CreditedOutputSession,
) {
    let request = typed_chat_request(usage, maximum, seed);
    let contract =
        OutputProjectionContract::chat_sse(request.id.to_string(), "public-alias".into(), usage);
    let (mut state, session) =
        credited_sequence_with_contract(tokenizer.clone(), request, contract);
    assert!(
        state.stop_token_ids.is_empty(),
        "ignore_eos removes the tokenizer's real EOS"
    );
    assert!(state.model_eos_token_ids.is_empty());
    let policy = host_numeric_policy(&state, tokenizer.as_ref()).unwrap();
    assert_eq!(
        policy.empirical_content_domain,
        Some(HostContentDomainV1::PlainTextGreedyV1)
    );
    state.cost_numeric_policy = Some(policy);
    for id in [0, 1] {
        state.generated_tokens.push(TokenId::new(id));
        state.sampling_history.record(TokenId::new(id));
    }
    (state, session)
}

fn checked_input(
    state: &SequenceState,
    tokenizer: &HuggingFaceTokenizer,
    rows: u32,
    kv_tokens: u32,
) -> StructuredInputV2 {
    let host = crate::continuous_engine::inner::cost_observation::participant_host_features(state)
        .unwrap();
    // One decode token per physical participant, matching the command below.
    let mut command = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
    command
        .kernel(
            SelectedAlgorithmClassV1::new("fixture.chat-policy", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: u64::from(rows) * 8,
                padded_units: u64::from(rows) * 8,
                inner_units_per_logical_unit: 2,
                grid: [1, 1, 1],
                scratch_bytes: 32,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = command.finish().unwrap();
    let mut wave =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::GreedyToken);
    wave.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.chat-policy",
        command_index: 0,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "policy-fixture",
            implementation_fingerprint: "v1",
            operation_fingerprint: "v1",
        }),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: rows,
        token_count: u64::from(rows),
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    wave.core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    for _ in 0..rows {
        wave.row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens },
            output: CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0f32.to_bits(),
            },
            host_policy_signature: host_policy_signature(state, tokenizer).unwrap(),
            mask_upload_required: false,
            host_features: Some(host),
        })
        .unwrap();
    }
    let wave = wave
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            u64::from(rows) * 32,
        )
        .unwrap();
    let domain = CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(1024).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(tokenizer.vocab_size() as u64).unwrap(),
            repetition_slot_capacity: 512,
            fixed_state_bytes_per_row: 32,
        },
    )
    .unwrap();
    let statistics = wave.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        statistics,
        statistics.structured_capture().unwrap().unwrap(),
        &domain,
    )
    .unwrap()
}

#[tokio::test]
async fn benchmark_chat_usage_selects_distinct_checked_numerical_family() {
    let tokenizer = tokenizer().await;
    let (with_usage, _usage_session) = chat_sequence(tokenizer.clone(), true, 547, 42);
    let (without_usage, _plain_session) = chat_sequence(tokenizer.clone(), false, 547, 42);
    assert_ne!(
        with_usage.cost_numeric_policy,
        without_usage.cost_numeric_policy
    );
    let enabled = checked_input(&with_usage, tokenizer.as_ref(), 1, 31);
    let disabled = checked_input(&without_usage, tokenizer.as_ref(), 1, 31);
    assert_eq!(
        enabled.owner().algorithm_domain,
        disabled.owner().algorithm_domain
    );
    assert_ne!(
        enabled.numerical_family_key().unwrap(),
        disabled.numerical_family_key().unwrap()
    );
    // The real contract must agree with the request. A caller cannot silently
    // turn off usage to borrow the other family's numerical authority.
    let request = typed_chat_request(true, 547, 42);
    assert!(ferrum_interfaces::output_flow::RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::chat_sse(
            request.id.to_string(),
            "public-alias".into(),
            false
        )),
        tokenizer.as_ref(),
        &request,
        1,
    )
    .is_err());
}

#[tokio::test]
async fn benchmark_chat_seed_and_output_limit_keep_policy_but_retain_numeric_shape() {
    let tokenizer = tokenizer().await;
    let (long, _long_session) = chat_sequence(tokenizer.clone(), true, 547, 42);
    let (short, _short_session) = chat_sequence(tokenizer.clone(), true, 3, 0);
    assert_eq!(long.cost_numeric_policy, short.cost_numeric_policy);
    assert_ne!(
        host_policy_signature(&long, tokenizer.as_ref()),
        host_policy_signature(&short, tokenizer.as_ref()),
        "old exact contract retains the output bound"
    );
    let long_input = checked_input(&long, tokenizer.as_ref(), 1, 31);
    let short_input = checked_input(&short, tokenizer.as_ref(), 1, 31);
    assert_eq!(
        long_input.numerical_family_key().unwrap(),
        short_input.numerical_family_key().unwrap()
    );
    assert_ne!(
        long_input.joint_support_coordinates(),
        short_input.joint_support_coordinates(),
        "output limit remains a checked numeric feature"
    );
    let wider = checked_input(&long, tokenizer.as_ref(), 2, 63);
    assert_eq!(
        long_input.numerical_family_key().unwrap(),
        wider.numerical_family_key().unwrap()
    );
    assert_ne!(long_input.owner().rows, wider.owner().rows);
    assert_ne!(long_input.regression_axes(), wider.regression_axes());
    assert_ne!(
        long_input.joint_support_coordinates(),
        wider.joint_support_coordinates()
    );
}
