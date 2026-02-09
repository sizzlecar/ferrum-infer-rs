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
}

#[test]
fn special_tokens_default() {
    let st = SpecialTokens::default();
    assert!(st.eos_token.is_none());
}
