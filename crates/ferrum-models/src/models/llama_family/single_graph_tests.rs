//! Real CUDA single-decode graph ownership and full-output regressions.

use super::*;
use crate::test_support::SyntheticLlamaLoader;
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_types::LlamaExecutionConfig;

type Model = LlamaFamilyModel<CudaBackend>;

fn models(paged: bool) -> (Model, Model) {
    let config = LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 2,
        vocab_size: 128,
        max_seq_len: 48,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(paged);
    execution.options.cuda_graph = true;
    let actual =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.options.cuda_graph = false;
    let reference = Model::new_with_execution_config(config, &loader, execution).unwrap();
    (actual, reference)
}

fn complete_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), 128);
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && e.is_finite(), "logit {index}: {a} vs {e}");
        // Same backend/weights/route: graph capture only changes launch delivery.
        assert!(
            (a - e).abs() <= 1e-5 + 1e-5 * e.abs(),
            "logit {index}: {a} vs {e}"
        );
    }
}

fn prefill(actual: &mut Model, reference: &mut Model, id: &str, prompt: &[u32]) {
    complete_logits(
        &actual.prefill_internal(id, prompt),
        &reference.prefill_internal(id, prompt),
    );
    assert!(actual.kv_caches[id]
        .iter()
        .all(|layer| layer.len == prompt.len()));
}

fn decode(actual: &mut Model, reference: &mut Model, id: &str, token: u32, position: u32) {
    complete_logits(
        &actual.decode_internal(id, token, position),
        &reference.decode_internal(id, token, position),
    );
    assert!(actual.kv_caches[id]
        .iter()
        .all(|layer| layer.len == position as usize + 1));
}

#[test]
#[ignore = "requires CUDA device"]
fn single_graph_cuda_replay_advances_frontier_and_retires_before_capacity_growth() {
    let (mut actual, mut reference) = models(false);
    actual.prepare_kv_capacity("a", 12);
    reference.prepare_kv_capacity("a", 12);
    prefill(&mut actual, &mut reference, "a", &[1, 2, 3]);
    for position in 3..12 {
        decode(&mut actual, &mut reference, "a", position + 5, position);
        if position >= 6 {
            assert_eq!(actual.single_graph_cache_id.as_deref(), Some("a"));
            assert!(!actual.graph_capture_failed);
        }
    }
    assert_eq!(actual.kv_caches["a"][0].capacity, 12);
    decode(&mut actual, &mut reference, "a", 31, 12);
    assert!(actual.kv_caches["a"]
        .iter()
        .all(|layer| layer.capacity >= 13));
    assert!(
        actual.single_graph_cache_id.is_none(),
        "growth must retire old captured storage"
    );
    decode(&mut actual, &mut reference, "a", 32, 13);
    actual.close().unwrap();
    reference.close().unwrap();
    assert!(actual.execution_context.is_none());
}

#[test]
#[ignore = "requires CUDA device"]
fn single_graph_cuda_keeps_cache_identity_across_switch_release_and_same_id_reuse() {
    let (mut actual, mut reference) = models(false);
    for id in ["a", "b"] {
        actual.prepare_kv_capacity(id, 32);
        reference.prepare_kv_capacity(id, 32);
    }
    prefill(&mut actual, &mut reference, "a", &[1, 2, 3]);
    prefill(&mut actual, &mut reference, "b", &[10, 11, 12]);
    for position in 3..9 {
        decode(&mut actual, &mut reference, "a", position + 5, position);
    }
    assert_eq!(actual.single_graph_cache_id.as_deref(), Some("a"));
    decode(&mut actual, &mut reference, "b", 42, 3);
    assert!(actual.single_graph_cache_id.is_none());
    for position in 4..9 {
        decode(&mut actual, &mut reference, "b", position + 30, position);
    }
    assert_eq!(actual.single_graph_cache_id.as_deref(), Some("b"));
    // Returning to A must preserve its actual KV, not B's previously bound one.
    for position in 9..14 {
        decode(&mut actual, &mut reference, "a", position + 7, position);
    }
    assert_eq!(actual.single_graph_cache_id.as_deref(), Some("a"));
    actual.release("a");
    reference.release("a");
    assert!(actual.single_graph_cache_id.is_none());
    assert!(!actual.kv_caches.contains_key("a"));
    assert_eq!(actual.kv_free_pool.len(), 1);
    // Reusing both the text ID and pooled allocation is a new incarnation.
    prefill(&mut actual, &mut reference, "a", &[70, 71, 72]);
    for position in 3..9 {
        decode(&mut actual, &mut reference, "a", position + 60, position);
    }
    assert_eq!(actual.single_graph_cache_id.as_deref(), Some("a"));
    actual.truncate_kv("a", 4);
    reference.truncate_kv("a", 4);
    assert!(actual.single_graph_cache_id.is_none());
    decode(&mut actual, &mut reference, "a", 83, 4);
    decode(&mut actual, &mut reference, "b", 84, 9);
    actual.reset();
    reference.reset();
    assert!(actual.single_graph_cache_id.is_none());
    assert!(actual.kv_caches.is_empty());
    actual.close().unwrap();
    reference.close().unwrap();
}

#[test]
#[ignore = "requires CUDA device"]
fn single_graph_cuda_paged_request_uses_eager_without_capturing_stack_uploads() {
    let (mut actual, mut reference) = models(true);
    prefill(&mut actual, &mut reference, "paged", &[1, 2, 3]);
    assert!(actual.paged_pools.is_some());
    assert!(!actual.single_decode_graph_enabled());
    for position in 3..12 {
        decode(&mut actual, &mut reference, "paged", position + 8, position);
    }
    assert!(actual.single_graph_cache_id.is_none());
    assert!(
        !actual.graph_capture_failed,
        "paged eager is an explicit route, not a failed capture"
    );
    actual.close().unwrap();
    reference.close().unwrap();
}
