use ferrum_types::*;

#[test]
fn sampling_params_defaults_and_greedy() {
    let d = SamplingParams::default();
    assert_eq!(d.temperature, 1.0);
    assert!(d.top_p <= 1.0 && d.top_p > 0.0);

    let g = SamplingParams::greedy();
    assert_eq!(g.temperature, 0.0);
    assert!(g.top_k.is_none());
}

#[test]
fn sampling_params_validate_checks() {
    let mut p = SamplingParams::default();
    p.temperature = -0.1;
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.top_p = 0.0; // invalid
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.top_k = Some(0);
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.min_p = Some(1.1);
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.response_completion_boundary =
        ResponseCompletionBoundary::AfterDelimiterAndPayload(String::new());
    assert!(p.validate().is_err());
}

#[test]
fn response_completion_boundary_is_backward_compatible_and_round_trips() {
    let mut legacy = serde_json::to_value(SamplingParams::default()).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("response_completion_boundary");
    let decoded: SamplingParams = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        decoded.response_completion_boundary,
        ResponseCompletionBoundary::Immediate
    );

    let mut params = SamplingParams::default();
    params.response_completion_boundary =
        ResponseCompletionBoundary::AfterDelimiterAndPayload("</think>".to_string());
    let decoded: SamplingParams =
        serde_json::from_value(serde_json::to_value(params).unwrap()).unwrap();
    assert_eq!(
        decoded.response_completion_boundary,
        ResponseCompletionBoundary::AfterDelimiterAndPayload("</think>".to_string())
    );
}

#[test]
fn special_tokens_default() {
    let st = SpecialTokens::default();
    assert!(st.eos_token.is_none());
}
