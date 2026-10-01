use super::*;
use crate::continuous_engine::{
    credited_output::{CreditedDecodePolicy, CreditedSequenceOutput},
    inner::cost_observation::{participant_host_features, policy},
    output_flow_runtime::{spawn_output_flow_runtime, OutputFlowRuntimeOptions},
    SequenceState,
};
use ferrum_interfaces::{
    execution_cost::{CanonicalWaveCostShape, HostContentForecastV2},
    output_credit::{OutputAccountLimits, OutputCreditPool, OutputPoolLimits},
    output_flow::{OutputProjectionContract, RequestOutputBudget, RequestOutputPlan},
    tokenizer::Tokenizer,
};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::SamplingParams;

pub(super) struct Template {
    pub actual: ActualWaveShape,
    pub host: HostCostFeaturesV1,
    pub policy_signature: [u8; 32],
    exact: CanonicalWaveCostShape,
}
impl Template {
    pub fn query(&self) -> StructuredQueryV2 {
        let selected = self.actual.statistical_evidence.as_ref().unwrap();
        StructuredQueryV2::from_future(
            &self.exact,
            selected,
            selected.structured_capture().unwrap().unwrap(),
            &HostContentForecastV2::Exact,
        )
        .unwrap()
    }
}
pub(super) struct Fixture {
    pub continuing: Template,
    pub at_length: Template,
    pub nonmember: Template,
}
impl Fixture {
    pub async fn new() -> Self {
        let vocabulary: tokenizers::models::bpe::Vocab = ["a", "b", "</s>"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect();
        let mut raw = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocabulary, Vec::new())
                .build()
                .unwrap(),
        );
        raw.add_special_tokens(&[tokenizers::AddedToken::from("</s>", true)]);
        raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        let tokenizer = Arc::new(HuggingFaceTokenizer::new(raw).await.unwrap());
        let continuing = prepared(Arc::clone(&tokenizer), 8, None);
        let at_length = prepared(Arc::clone(&tokenizer), 3, None);
        let nonmember = prepared(tokenizer, 8, Some(2));
        assert_eq!(continuing.query().owner(), at_length.query().owner());
        assert_ne!(continuing.query().owner(), nonmember.query().owner());
        Self {
            continuing,
            at_length,
            nonmember,
        }
    }
}

fn prepared(
    tokenizer: Arc<HuggingFaceTokenizer>,
    maximum: usize,
    top_k: Option<usize>,
) -> Template {
    let mut request = InferenceRequest::new("aa", "installed-source6-fixture");
    request.sampling_params = SamplingParams {
        max_tokens: maximum,
        temperature: 0.7,
        top_k,
        stop_sequences: vec!["b".into()],
        ..SamplingParams::greedy()
    };
    let mut state = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(0); 5],
        Some(tokenizer.clone()),
    );
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
    let pool =
        OutputCreditPool::new(OutputPoolLimits::from_slo(&config, NonZeroUsize::MIN).unwrap())
            .unwrap();
    let budget = RequestOutputBudget::open(&pool, limits, plan).unwrap();
    let (port, _session) = spawn_output_flow_runtime(
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
    for _ in 0..2 {
        state.generated_tokens.push(TokenId::new(0));
        state.sampling_history.record(TokenId::new(0));
    }
    state.cost_numeric_policy =
        Some(policy::host_numeric_policy(&state, tokenizer.as_ref()).unwrap());
    let host = participant_host_features(&state).unwrap();
    assert!(matches!(
        host.policy.empirical_content_domain,
        Some(HostContentDomainV1::PlainTextInstalledV2(_))
    ));
    // EOS and explicit stop are both installed on this same policy. Keep the
    // exact runtime's precedence and vocabulary, rather than deleting EOS.
    assert!(!state.model_eos_token_ids.is_empty());
    assert!(state
        .user_stop_token_ids
        .contains(&tokenizer.token_id("b").unwrap().get()));
    let policy_signature = policy::host_policy_signature(&state, tokenizer.as_ref()).unwrap();
    let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    selected
        .kernel_with_replay_geometry(
            SelectedAlgorithmClassV1::new(
                "installed.source6.fixture",
                1,
                Sha256::digest(b"fixture-provider").into(),
                Sha256::digest(b"fixture-operation").into(),
            )
            .unwrap(),
            KernelNumericWorkV1 {
                logical_units: 32,
                padded_units: 32,
                inner_units_per_logical_unit: 32,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                fixed_parameters: &[32],
            },
        )
        .unwrap();
    let selected = selected.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "installed.source6.fixture",
        command_index: 0,
        node_index: None,
        command_phase: DeviceCommandPhase::Compute,
        provider: None,
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 1,
        token_count: 1,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    b.row(CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens: 7 },
        host_policy_signature: policy_signature,
        mask_upload_required: false,
        host_features: Some(host),
        output: CostRowOutput::Decode {
            requires_full_logits: true,
            repetition_tokens: 0,
            repetition_penalty_bits: 1f32.to_bits(),
        },
    })
    .unwrap();
    let built = b
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let mut actual = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
    actual.provider_signature = built.exact.provider_signature;
    actual.output_policy_signature = built.exact.output_policy_signature;
    actual.numeric_features = built.exact.numeric_features.clone();
    actual.host_content_features = built.exact.host_content_features;
    actual.row_multiset_features = built.exact.row_multiset_features.clone();
    actual.statistical_evidence = Some(built.statistical.unwrap());
    Template {
        actual,
        host,
        policy_signature,
        exact: built.exact,
    }
}
