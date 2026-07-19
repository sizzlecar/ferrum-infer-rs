use ferrum_interfaces::sampler::*;
use ferrum_types::{SamplingParams, TokenId};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn temperature_processor_scales_logits() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let params = SamplingParams {
        temperature: 2.0,
        ..Default::default()
    };
    let prev: Vec<TokenId> = vec![];
    let freqs = std::collections::HashMap::new();
    let vocab_size = 3;
    let mut ctx = SamplingContext::new(0, &params, &mut logits, &prev, &freqs, vocab_size);
    let p = TemperatureProcessor::new(2.0);
    p.process(&mut ctx).unwrap();
    assert!((ctx.logits[2] - 1.5).abs() < 1e-6);
}

#[test]
fn topk_masks_tail() {
    let mut logits = vec![0.0, 1.0, 2.0, 3.0];
    let params = SamplingParams::default();
    let prev = vec![];
    let freqs = std::collections::HashMap::new();
    let mut ctx = SamplingContext::new(0, &params, &mut logits, &prev, &freqs, 4);
    TopKProcessor::new(2).process(&mut ctx).unwrap();
    let masked = ctx
        .logits
        .iter()
        .filter(|v| **v == f32::NEG_INFINITY)
        .count();
    assert!(masked >= 2);
}

#[test]
fn topp_masks_beyond_p() {
    let mut logits = vec![0.0, 0.0, 10.0, 9.0];
    let params = SamplingParams {
        top_p: 0.6,
        ..Default::default()
    };
    let binding = std::collections::HashMap::new();
    let mut ctx = SamplingContext::new(0, &params, &mut logits, &[], &binding, 4);
    TopPProcessor::new(0.6).process(&mut ctx).unwrap();
    let masked = ctx
        .logits
        .iter()
        .filter(|v| **v == f32::NEG_INFINITY)
        .count();
    assert!(masked >= 1);
}

#[test]
fn repetition_penalty_applies() {
    let mut logits = vec![1.0, 1.0, -2.0];
    let params = SamplingParams::default();
    let prev = vec![TokenId::new(1), TokenId::new(1), TokenId::new(2)];
    let mut freqs = std::collections::HashMap::new();
    freqs.insert(TokenId::new(1), 2usize);
    freqs.insert(TokenId::new(2), 1usize);
    let vocab_size = 3;
    let mut ctx = SamplingContext::new(0, &params, &mut logits, &prev, &freqs, vocab_size);
    RepetitionPenaltyProcessor::new(1.1)
        .process(&mut ctx)
        .unwrap();
    assert_eq!(ctx.logits[0], 1.0);
    assert!((ctx.logits[1] - (1.0 / 1.1)).abs() < 1e-6);
    assert!((ctx.logits[2] - (-2.0 * 1.1)).abs() < 1e-6);
}

#[test]
fn presence_and_frequency_penalties_follow_generated_token_counts() {
    let mut logits = vec![1.0, 4.0, -2.0, 8.0];
    let params = SamplingParams::default();
    let previous = vec![TokenId::new(1), TokenId::new(1), TokenId::new(2)];
    let frequencies =
        std::collections::HashMap::from([(TokenId::new(1), 2usize), (TokenId::new(2), 1usize)]);
    let mut ctx = SamplingContext::new(
        previous.len(),
        &params,
        &mut logits,
        &previous,
        &frequencies,
        4,
    );

    PresenceFrequencyPenaltyProcessor::new(0.5, 0.25)
        .process(&mut ctx)
        .unwrap();

    assert_eq!(ctx.logits[0], 1.0);
    assert!((ctx.logits[1] - 3.0).abs() < 1e-6);
    assert!((ctx.logits[2] - (-2.75)).abs() < 1e-6);
    assert_eq!(ctx.logits[3], 8.0);
}

#[test]
fn negative_presence_and_frequency_penalties_promote_seen_tokens() {
    let mut logits = vec![1.0, 2.0];
    let params = SamplingParams::default();
    let frequencies = std::collections::HashMap::from([(TokenId::new(0), 3usize)]);
    let mut ctx = SamplingContext::new(3, &params, &mut logits, &[], &frequencies, 2);

    PresenceFrequencyPenaltyProcessor::new(-0.5, -0.25)
        .process(&mut ctx)
        .unwrap();

    assert!((ctx.logits[0] - 2.25).abs() < 1e-6);
    assert_eq!(ctx.logits[1], 2.0);
}

#[test]
fn sampling_config_uses_penalty_temperature_and_filter_order() {
    let params = SamplingParams {
        temperature: 2.0,
        top_p: 0.9,
        top_k: Some(8),
        min_p: Some(0.1),
        repetition_penalty: 1.1,
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        ..Default::default()
    };

    let config = SamplingConfig::from_params(&params);

    assert_eq!(
        config.processor_chain.processor_names(),
        vec![
            "repetition_penalty",
            "presence_frequency_penalty",
            "temperature",
            "min_p",
            "top_k",
            "top_p",
        ]
    );
}

#[test]
fn min_p_uses_temperature_scaled_probability_ratio() {
    let params = SamplingParams {
        temperature: 2.0,
        min_p: Some(0.5),
        ..Default::default()
    };
    let config = SamplingConfig::from_params(&params);
    let mut logits = vec![4.0, 3.0, 2.0];
    let frequencies = std::collections::HashMap::new();
    let mut ctx = SamplingContext::new(0, &params, &mut logits, &[], &frequencies, 3);

    config.processor_chain.process(&mut ctx).unwrap();

    assert_eq!(ctx.logits[0], 2.0);
    assert_eq!(ctx.logits[1], 1.5);
    assert_eq!(ctx.logits[2], f32::NEG_INFINITY);
}

#[test]
fn min_p_one_keeps_only_tokens_tied_for_maximum() {
    let params = SamplingParams {
        min_p: Some(1.0),
        ..Default::default()
    };
    let config = SamplingConfig::from_params(&params);
    let mut logits = vec![3.0, 2.0, 3.0];
    let frequencies = std::collections::HashMap::new();
    let mut ctx = SamplingContext::new(0, &params, &mut logits, &[], &frequencies, 3);

    config.processor_chain.process(&mut ctx).unwrap();

    assert_eq!(ctx.logits, &[3.0, f32::NEG_INFINITY, 3.0]);
}

#[test]
fn only_unprocessed_greedy_plan_supports_raw_speculation() {
    assert!(
        SamplingConfig::from_params(&SamplingParams::greedy()).supports_raw_greedy_speculation()
    );

    for params in [
        SamplingParams {
            temperature: 1.0,
            ..SamplingParams::greedy()
        },
        SamplingParams {
            presence_penalty: 0.1,
            ..SamplingParams::greedy()
        },
        SamplingParams {
            min_p: Some(0.9),
            ..SamplingParams::greedy()
        },
    ] {
        assert!(!SamplingConfig::from_params(&params).supports_raw_greedy_speculation());
    }
}

#[test]
fn sampling_config_from_params_builds_chain_and_samples() {
    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        top_k: Some(10),
        ..Default::default()
    };
    let cfg = SamplingConfig::from_params(&params);

    let mut rng = StdRng::seed_from_u64(42);
    let mut logits = vec![0.1, 0.2, 3.0, 0.4];
    let prev: Vec<TokenId> = vec![];
    let freqs = std::collections::HashMap::new();
    let vocab = logits.len();
    let ctx = SamplingContext::new(0, &params, &mut logits, &prev, &freqs, vocab);
    let tok = cfg.sample(ctx, &mut rng).unwrap();
    assert!((tok.get() as usize) < logits.len());
}

#[test]
fn greedy_sampler_picks_max() {
    let g = GreedySampler;
    let tok = g
        .sample(&[0.1, 10.0, 0.2], &mut StdRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(tok, TokenId::new(1));
}
