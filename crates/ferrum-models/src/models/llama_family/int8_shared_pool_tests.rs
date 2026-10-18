//! Real retained CUDA storage and causal prefill tests.

use super::*;
use crate::common::{llm::ModelKvReservation, DecoderOnlyLLM};
use crate::test_support::SyntheticLlamaLoader;
use ferrum_kernels::backend::cuda::CudaBackend;

type Model = LlamaFamilyModel<CudaBackend, KvInt8>;

fn model(prefix: bool) -> Model {
    model_with_heads(prefix, 1)
}

fn model_with_heads(prefix: bool, kv_heads: usize) -> Model {
    let config = LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: kv_heads,
        head_dim: 64,
        num_layers: 2,
        vocab_size: 128,
        max_seq_len: 32,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut model = Model::new(config, &loader).unwrap();
    model.runtime_env.metal_paged_kv = Some(true);
    model.runtime_env.prefix_cache = prefix;
    model.runtime_env.cuda_graph = false;
    model.runtime_env.kv_capacity = Some(32);
    model.runtime_env.kv_max_blocks = Some(3);
    model.runtime_env.paged_max_seqs = 2;
    model
}

fn fill(model: &mut Model, id: &str, tokens: &[u32]) -> Vec<f32> {
    let mut logits = Vec::new();
    for &token in tokens {
        logits = model.prefill_internal(id, &[token]);
        assert_eq!(logits.len(), 128);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
    logits
}

fn free_blocks(model: &Model) -> usize {
    model
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .free_count()
}

#[test]
#[ignore]
fn int8_shared_pool_real_cuda_prefix_alias_and_release_preserve_logits() {
    let mut actual = model(true);
    let mut reference_a = model(false);
    let mut reference_b = model(false);
    let prefix: Vec<u32> = (1..=16).collect();
    assert_eq!(
        fill(&mut actual, "a", &prefix),
        fill(&mut reference_a, "a", &prefix)
    );
    fill(&mut reference_b, "b", &prefix);
    // Register precisely the complete block whose values were computed above.
    // This invokes the same production prefix registration/acquisition path;
    // it does not manufacture a block hash or copy quantized data in the test.
    actual.register_prefix_cache("a", &prefix, 0);
    assert_eq!(actual.kv_caches["a"][0].paged_block_indices, [0]);
    assert!(matches!(
        actual.prepare_token_kv("b", 32, 17).unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    assert_eq!(actual.kv_caches["b"][0].paged_block_indices, [1, 2]);
    let mut prompt_b = prefix.clone();
    prompt_b.push(42);
    let actual_b = actual.prefill_internal("b", &prompt_b);
    let expected_b = reference_b.prefill_internal("b", &[42]);
    assert_eq!(
        actual_b, expected_b,
        "prefix view must read the producer's shared INT8 data"
    );
    assert_eq!(actual.kv_caches["b"][0].paged_block_indices, [0, 2]);
    assert_eq!(actual.prefix_cache_hits, 1);
    assert_eq!(actual.prefix_cache_saved_prefill_tokens, 16);
    for (pool, cache) in actual
        .paged_pools
        .as_ref()
        .unwrap()
        .iter()
        .zip(&actual.kv_caches["b"])
    {
        assert_eq!(pool.num_blocks, 3);
        assert_eq!(pool.k.buffer().len(), 3 * 16 * 64);
        assert_eq!(pool.v.buffer().len(), 3 * 16 * 64);
        assert_eq!(pool.k_scales.buffer().len(), 3 * 16);
        assert_eq!(pool.v_scales.buffer().len(), 3 * 16);
        assert_eq!(cache.capacity, 32);
        assert!(cache
            .paged_block_indices
            .iter()
            .any(|&block| block as usize * 16 >= cache.capacity));
    }
    assert_eq!(
        actual.decode_internal("a", 57, 16),
        reference_a.decode_internal("a", 57, 16),
        "B's private suffix must not overwrite A's KV"
    );
    assert_eq!(free_blocks(&actual), 0);
    actual.release("a");
    assert_eq!(free_blocks(&actual), 1, "shared prefix remains owned by B");
    assert_eq!(
        actual.decode_internal("b", 61, 17),
        reference_b.decode_internal("b", 61, 17)
    );
    actual.release("b");
    assert_eq!(free_blocks(&actual), 3);
    assert!(actual.kv_caches.is_empty());
    actual.release("b");
    assert_eq!(free_blocks(&actual), 3);
    reference_a.release("a");
    reference_b.release("b");
}

#[test]
#[ignore]
fn int8_shared_pool_real_cuda_serial_soft_free_rotation_preserves_logits() {
    let mut actual = model(true);
    for iteration in 0..3 {
        let tokens: Vec<u32> = (1..=16).map(|token| token + 17 * iteration).collect();
        let mut reference = model(false);
        assert_eq!(
            fill(&mut actual, "serial", &tokens),
            fill(&mut reference, "reference", &tokens)
        );
        let indices = actual.kv_caches["serial"][0].paged_block_indices.clone();
        assert_eq!(indices, [iteration]);
        actual.register_prefix_cache("serial", &tokens, 0);
        actual.release("serial");
        reference.release("reference");
        assert_eq!(free_blocks(&actual), 3);
    }
    // The third serial request wrote global block 2 even though each sequence
    // has only two logical table entries. No simultaneous requests are needed.
    assert!(actual.kv_caches.is_empty());
    assert_eq!(actual.paged_pools.as_ref().unwrap()[0].num_blocks, 3);
}

#[test]
#[ignore]
fn int8_shared_pool_real_cuda_multiquery_prefill_matches_single_query_reference() {
    let mut whole = model_with_heads(false, 2);
    let mut chunked = model_with_heads(false, 2);
    let mut reference = model_with_heads(false, 2);
    let tokens: Vec<u32> = (1..=20).collect();
    let expected = fill(&mut reference, "reference", &tokens);
    let full = whole.prefill_internal("whole", &tokens);
    chunked.prefill_internal("chunked", &tokens[..13]);
    let partial = chunked.prefill_internal("chunked", &tokens[13..]);
    let compare = |actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|value| value.is_finite()));
        let magnitude = expected.iter().fold(0.0f32, |m, value| m.max(value.abs()));
        let error = actual
            .iter()
            .zip(expected)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        // Batched F16 projections and INT8 rounding may differ from M=1.
        // The independent per-query arithmetic oracle is in int8_kv_parity.
        assert!(
            error < 0.02 * magnitude.max(1e-6),
            "full-logits error={error}, magnitude={magnitude}"
        );
    };
    compare(&full, &expected);
    compare(&partial, &expected);
    let next_reference = reference.decode_internal("reference", 71, 20);
    compare(&whole.decode_internal("whole", 71, 20), &next_reference);
    compare(&chunked.decode_internal("chunked", 71, 20), &next_reference);
    for (model, id) in [
        (&mut whole, "whole"),
        (&mut chunked, "chunked"),
        (&mut reference, "reference"),
    ] {
        assert!(model.kv_caches[id].iter().all(|cache| cache.len == 21));
        model.release(id);
        assert!(model.kv_caches.is_empty());
        assert_eq!(free_blocks(model), 3);
    }
}
