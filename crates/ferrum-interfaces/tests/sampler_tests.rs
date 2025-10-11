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
    let mut logits = vec![1.0, 1.0, 1.0];
    let params = SamplingParams::default();
    let prev = vec![TokenId::new(1)];
    let mut freqs = std::collections::HashMap::new();
    freqs.insert(TokenId::new(1), 2usize);
    let vocab_size = 3;
    let mut ctx = SamplingContext::new(0, &params, &mut logits, &prev, &freqs, vocab_size);
    RepetitionPenaltyProcessor::new(1.1)
        .process(&mut ctx)
        .unwrap();
    assert!(ctx.logits[1] != 1.0);
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
