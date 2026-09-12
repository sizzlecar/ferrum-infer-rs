//! CPU sampling microbenchmarks over deterministic synthetic dense logits.
//!
//! These are not captured model logits or end-to-end inference measurements.
//! The large vocabulary matches a measured serving workload's shape; the
//! smaller vocabulary checks that results are not specific to that shape.
//! Input reset and RNG construction are excluded from the timed region.
//! SamplingConfig covers logits processors and multinomial sampling, but not
//! engine grammar masks, tokenizer work, locks, or token/resource commits.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ferrum_interfaces::sampler::{
    LogitsProcessor, SamplingConfig, SamplingContext, SamplingRng, TopKProcessor,
};
use ferrum_types::SamplingParams;
use std::collections::HashMap;

const SEED: u64 = 20_260_912;

fn synthetic_dense_logits(vocab_size: usize) -> Vec<f32> {
    (0..vocab_size)
        .map(|index| {
            // Integer mixing gives unsorted finite values, independent of a
            // random-number library or platform transcendental functions.
            let mut bits = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            bits ^= bits >> 31;
            ((bits >> 40) as f32) * (16.0 / 16_777_216.0) - 8.0
        })
        .collect()
}

fn sampler_benchmarks(criterion: &mut Criterion) {
    let params = SamplingParams {
        temperature: 0.6,
        top_p: 0.95,
        top_k: Some(20),
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: Some(SEED),
        ..Default::default()
    };
    let config = SamplingConfig::from_params(&params);
    let top_k = TopKProcessor::new(20);
    let frequencies = HashMap::new();
    let mut group = criterion.benchmark_group("synthetic_dense_sampling");

    for vocab_size in [32_768, 248_320] {
        let template = synthetic_dense_logits(vocab_size);
        group.bench_with_input(
            BenchmarkId::new("top_k20", vocab_size),
            &template,
            |bencher, template| {
                bencher.iter_batched_ref(
                    || template.clone(),
                    |logits| {
                        let mut context = SamplingContext::new(
                            0,
                            &params,
                            black_box(logits.as_mut_slice()),
                            &[],
                            &frequencies,
                            vocab_size,
                        );
                        top_k.process(&mut context).unwrap();
                        black_box(context.logits);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("sampling_t0.6_p0.95_k20", vocab_size),
            &template,
            |bencher, template| {
                bencher.iter_batched_ref(
                    || (template.clone(), SamplingRng::seeded(SEED)),
                    |(logits, rng)| {
                        let context = SamplingContext::new(
                            0,
                            &params,
                            black_box(logits.as_mut_slice()),
                            &[],
                            &frequencies,
                            vocab_size,
                        );
                        black_box(config.sample(context, rng).unwrap());
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, sampler_benchmarks);
criterion_main!(benches);
