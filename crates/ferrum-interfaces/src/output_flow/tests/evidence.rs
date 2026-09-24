use super::*;
use ferrum_types::{EngineTokenTimingEvidence, InferenceExecutionEvidence};

fn fixture() -> (RequestOutputPlan, InferenceExecutionEvidence, TokenUsage) {
    let mut request = InferenceRequest::new("prompt", "model");
    request.sampling_params.max_tokens = 2;
    request.evidence_request.capture_engine_token_timing = true;
    request.evidence_request.capture_prompt_token_ids = true;
    let plan = RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::cli_text()),
        &BoundedTokenizer::new(4),
        &request,
        3,
    )
    .unwrap();
    let evidence = InferenceExecutionEvidence {
        prompt_token_ids: vec![TokenId::new(1); 3],
        output_token_ids: vec![TokenId::new(1); 2],
        engine_token_timing: Some(EngineTokenTimingEvidence {
            clock_source: "rust_std_instant".into(),
            wall_anchor_unix_nanos: 1,
            wall_anchor_max_error_nanos: 0,
            decode_ready_nanos_since_request_start: Some(1000),
            token_commit_nanos_since_request_start: vec![1000, 3000],
            decode_stage_intervals: Vec::new(),
            decode_stage_intervals_omitted: 19,
        }),
    };
    (plan, evidence, TokenUsage::new(3, 2))
}
#[test]
fn credited_evidence_accounts_owned_capacity_and_rejects_incomplete_commits() {
    let (plan, mut evidence, usage) = fixture();
    let bounds = plan.evidence_plan();
    assert_eq!(bounds.maximum_stage_intervals(), 6);
    assert_eq!(
        bounds.retained_bytes(),
        5 * size_of::<TokenId>()
            + 2 * size_of::<u64>()
            + 6 * size_of::<ferrum_types::EngineDecodeStageInterval>()
            + "rust_std_instant".len()
    );
    bounds.validate(Some(&evidence), &usage).unwrap();
    evidence
        .engine_token_timing
        .as_mut()
        .unwrap()
        .token_commit_nanos_since_request_start
        .pop();
    assert!(bounds.validate(Some(&evidence), &usage).is_err());
    evidence
        .engine_token_timing
        .as_mut()
        .unwrap()
        .token_commit_nanos_since_request_start
        .push(3000);
    evidence.output_token_ids.reserve(100);
    assert!(
        bounds.validate(Some(&evidence), &usage).is_err(),
        "capacity matters even if length stayed legal"
    );
    assert!(bounds.validate(None, &usage).is_err());
    assert!(EngineEvidenceRetentionPlan::default()
        .validate(Some(&evidence), &usage)
        .is_err());
}
#[test]
fn credited_evidence_is_separately_reserved_and_overflow_fails_before_admission() {
    let (plan, _, _) = fixture();
    let mut request = InferenceRequest::new("prompt", "model");
    request.sampling_params.max_tokens = 2;
    let plain = RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::cli_text()),
        &BoundedTokenizer::new(4),
        &request,
        3,
    )
    .unwrap();
    assert_eq!(
        plan.retained_projection_bytes() - plain.retained_projection_bytes(),
        plan.evidence_plan().retained_bytes()
    );
    assert_eq!(plan.lifetime_wire_bytes(), plain.lifetime_wire_bytes());
    request.evidence_request.capture_engine_token_timing = true;
    request.sampling_params.max_tokens = usize::MAX;
    assert!(matches!(
        RequestOutputPlan::derive(
            Arc::new(OutputProjectionContract::cli_text()),
            &BoundedTokenizer::new(4),
            &request,
            3
        ),
        Err(OutputFlowError::Overflow)
    ));
    let too_small = limits(&plain);
    let pool = pool(&plan);
    assert!(RequestOutputBudget::open(&pool, too_small, plan).is_err());
    assert_eq!(pool.snapshot().retained_accounts, 0);
}
#[test]
fn credited_profile_borrows_complete_timing_and_discloses_partial_stages() {
    let (plan, evidence, usage) = fixture();
    let expected = ferrum_types::engine_token_timing_profile_attributes(
        evidence.engine_token_timing.as_ref().unwrap(),
    );
    let pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan).unwrap();
    // The terminal wire and retained evidence have distinct leases. Transfer
    // evidence only after the real terminal encoder has closed the account.
    let terminal = budget
        .encode_terminal(OutputTerminal::Success {
            reason: FinishReason::Length,
            usage: &usage,
            created: 0,
        })
        .unwrap();
    drop(terminal);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
    let completion = Arc::new(
        budget
            .into_retained_projection(OutputCompletion::Succeeded {
                history: None,
                reason: FinishReason::Length,
                usage,
                execution_evidence: Some(evidence),
            })
            .unwrap(),
    );
    let record = CreditedExecutionProfile::new(
        completion.clone(),
        ferrum_types::ProfileEntrypoint::Run,
        "model".into(),
        "cli_text",
        ferrum_types::ObservabilityProfileDetail::Latency,
        3,
        "sha256:fixture".into(),
        Default::default(),
    )
    .unwrap();
    let value = serde_json::to_value(&record).unwrap();
    for (key, value_expected) in expected {
        assert_eq!(value["attributes"][&key], value_expected, "{key}");
    }
    let parsed: ferrum_types::FerrumProfileEvent = serde_json::from_value(value).unwrap();
    parsed.validate().unwrap();
    drop(completion);
    assert!(pool.snapshot().data_used.projection_bytes > 0);
    drop(record);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[test]
fn credited_normal_text_evidence_fits_typed_default_projection_and_old_wire_defaults() {
    let mut request = InferenceRequest::new("prompt", "model");
    request.sampling_params.max_tokens = 2048;
    request.evidence_request.capture_engine_token_timing = true;
    request.evidence_request.capture_prompt_token_ids = true;
    let plan = RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::cli_text()),
        &BoundedTokenizer::new(7),
        &request,
        2048,
    )
    .unwrap();
    assert!(
        plan.retained_projection_bytes()
            < ferrum_types::SloOutputConfig::default()
                .max_projection_bytes_per_request
                .get()
    );
    let (_, evidence, _) = fixture();
    let mut wire = serde_json::to_value(evidence.engine_token_timing.unwrap()).unwrap();
    wire.as_object_mut()
        .unwrap()
        .remove("decode_stage_intervals_omitted");
    let old: EngineTokenTimingEvidence = serde_json::from_value(wire).unwrap();
    assert_eq!(old.decode_stage_intervals_omitted, 0);
    old.validate(2).unwrap();
}
