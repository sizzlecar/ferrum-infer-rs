//! Real paged prefix reuse, with the recomputed suffix kept private.

use super::*;
use crate::common::llm::ModelKvReservation;
use crate::test_support::SyntheticLlamaLoader;

fn model<B: MoeLlmBackend>(prefix: bool) -> LlamaFamilyModel<B, KvFp16> {
    let config = LlamaFamilyConfig {
        hidden_size: 256,
        intermediate_size: 512,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: 2,
        vocab_size: 128,
        max_seq_len: 64,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut model = LlamaFamilyModel::new(config, &loader).unwrap();
    model.runtime_env.metal_paged_kv = Some(true);
    model.runtime_env.prefix_cache = prefix;
    model.runtime_env.cuda_graph = false;
    model.runtime_env.kv_capacity = Some(64);
    // Keep admission unconstrained here: over-reservation before prefix lookup
    // is a separate issue from sharing the block that the suffix must rewrite.
    model.runtime_env.kv_max_blocks = Some(8);
    model.runtime_env.paged_max_seqs = 2;
    model
}

fn tokenwise<B: MoeLlmBackend>(
    model: &mut LlamaFamilyModel<B, KvFp16>,
    cache_id: &str,
    tokens: &[u32],
) -> Vec<f32> {
    let mut logits = Vec::new();
    for &token in tokens {
        logits = model.prefill_internal(cache_id, &[token]);
        assert_eq!(logits.len(), model.cfg.vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
    logits
}

fn owner_bits<B: MoeLlmBackend>(
    model: &LlamaFamilyModel<B, KvFp16>,
    cache_id: &str,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    let caches = &model.kv_caches[cache_id];
    let per_block = LLAMA_PAGED_BLOCK_SIZE * model.cfg.num_kv_heads * model.cfg.head_dim;
    model
        .paged_pools
        .as_ref()
        .unwrap()
        .iter()
        .zip(caches)
        .map(|((k, v), cache)| {
            // Producer length is exactly two complete blocks. Every captured
            // element is initialized, regardless of the backend's within-block
            // K/V layout, and only this owner's physical blocks are compared.
            assert_eq!(cache.len, 2 * LLAMA_PAGED_BLOCK_SIZE);
            assert_eq!(cache.paged_block_indices.len(), 2);
            let extent =
                (cache.paged_block_indices.iter().copied().max().unwrap() as usize + 1) * per_block;
            let k = B::to_vec(k, extent);
            let v = B::to_vec(v, extent);
            let capture = |values: &[f32]| {
                cache
                    .paged_block_indices
                    .iter()
                    .flat_map(|&block| {
                        let start = block as usize * per_block;
                        values[start..start + per_block]
                            .iter()
                            .map(|value| value.to_bits())
                    })
                    .collect()
            };
            (capture(&k), capture(&v))
        })
        .collect()
}

fn prefix_reuse_preserves_private_suffix_and_original_owner<B: MoeLlmBackend>() {
    for prompt_len in [16usize, 32, 33] {
        let mut actual = model::<B>(true);
        let mut owner_reference = model::<B>(false);
        let mut consumer_reference = model::<B>(false);
        let tokens: Vec<u32> = (1..=33).collect();
        let producer_tokens = &tokens[..32];
        assert_eq!(
            tokenwise(&mut actual, "owner", producer_tokens),
            tokenwise(&mut owner_reference, "owner", producer_tokens)
        );
        // All registered values came from the actual model's numerical core.
        actual.register_prefix_cache("owner", producer_tokens, 0);
        let original_mapping = actual.kv_caches["owner"][0].paged_block_indices.clone();
        let original_bits = owner_bits(&actual, "owner");
        let prompt = &tokens[..prompt_len];
        assert!(matches!(
            actual.prepare_token_kv("consumer", 64, prompt_len).unwrap(),
            ModelKvReservation::Reserved(_)
        ));
        assert!(matches!(
            consumer_reference
                .prepare_token_kv("consumer", 64, prompt_len)
                .unwrap(),
            ModelKvReservation::Reserved(_)
        ));

        let shared_blocks = (prompt_len - 1) / LLAMA_PAGED_BLOCK_SIZE;
        let shared_tokens = shared_blocks * LLAMA_PAGED_BLOCK_SIZE;
        // Match the actual numerical shapes independently: cached K/V was
        // produced one token at a time; the fresh suffix runs as one chunk.
        tokenwise(
            &mut consumer_reference,
            "consumer",
            &prompt[..shared_tokens],
        );
        let expected = consumer_reference.prefill_internal("consumer", &prompt[shared_tokens..]);
        let output = actual.prefill_internal("consumer", prompt);
        assert!(
            actual.kv_caches["consumer"]
                .iter()
                .all(|cache| cache.len == prompt_len),
            "prefix hit changed the physical frontier for {prompt_len} tokens"
        );
        assert_eq!(output.len(), actual.cfg.vocab_size);
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(
            output, expected,
            "full logits differ for prompt length {prompt_len}"
        );
        assert_eq!(actual.prefix_cache_hits, u64::from(shared_blocks > 0));
        assert_eq!(
            actual.prefix_cache_saved_prefill_tokens,
            shared_tokens as u64
        );
        for cache in &actual.kv_caches["consumer"] {
            assert_eq!(
                &cache.paged_block_indices[..shared_blocks],
                &original_mapping[..shared_blocks]
            );
            assert!(
                cache.paged_block_indices[shared_blocks..]
                    .iter()
                    .all(|block| !original_mapping.contains(block)),
                "the recomputed suffix must never write another owner's block"
            );
        }
        assert_eq!(
            actual.kv_caches["owner"][0].paged_block_indices,
            original_mapping
        );
        assert_eq!(
            owner_bits(&actual, "owner"),
            original_bits,
            "consumer prefill changed the original owner's complete K/V contents"
        );
        {
            let allocator = actual.paged_block_alloc.as_ref().unwrap().lock().unwrap();
            for (index, &block) in original_mapping.iter().enumerate() {
                assert_eq!(
                    allocator.ref_count(block),
                    if index < shared_blocks { 2 } else { 1 }
                );
            }
        }

        assert_eq!(
            actual.decode_internal("owner", 70, 32),
            owner_reference.decode_internal("owner", 70, 32)
        );
        assert_eq!(
            actual.decode_internal("consumer", 71, prompt_len as u32),
            consumer_reference.decode_internal("consumer", 71, prompt_len as u32)
        );
        for cache_id in ["consumer", "owner"] {
            match actual.close_token_execution(cache_id) {
                crate::common::llm::ModelTokenCloseOutcome::Ended { error: None } => (),
                _ => panic!("actual {cache_id} closure did not prove a successful end"),
            }
        }
        assert!(actual.kv_caches.is_empty());
        assert_eq!(
            actual
                .paged_block_alloc
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free_count(),
            8
        );
        for (reference, cache_id) in [
            (&mut owner_reference, "owner"),
            (&mut consumer_reference, "consumer"),
        ] {
            match reference.close_token_execution(cache_id) {
                crate::common::llm::ModelTokenCloseOutcome::Ended { error: None } => (),
                _ => panic!("reference {cache_id} closure did not prove a successful end"),
            }
        }
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal device"]
fn prefix_reuse_real_metal_exact_blocks_preserve_logits_and_shared_owner() {
    prefix_reuse_preserves_private_suffix_and_original_owner::<
        ferrum_kernels::backend::metal::MetalBackend,
    >();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an actual CUDA device"]
fn prefix_reuse_real_cuda_exact_blocks_preserve_logits_and_shared_owner() {
    prefix_reuse_preserves_private_suffix_and_original_owner::<
        ferrum_kernels::backend::cuda::CudaBackend,
    >();
}
