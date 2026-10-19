//! Actual retained LlamaFamily batch graph / eager equivalence.
//! These tests do not exercise the separate typed-runtime operator graph.

use super::{physical_batch_config, LlamaFamilyModel, SyntheticLlamaLoader};
use crate::common::DecoderOnlyLLM;
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_types::LlamaExecutionConfig;

type Model = LlamaFamilyModel<CudaBackend>;

fn models() -> (Model, Model) {
    // Preserve the original counterexamples' fixed window and capacity.
    configured_models(false, 2, 1, 32)
}

fn configured_models(
    paged: bool,
    window: usize,
    layers: usize,
    max_seq_len: usize,
) -> (Model, Model) {
    let mut config = physical_batch_config();
    config.sliding_window = window;
    config.num_layers = layers;
    config.max_seq_len = max_seq_len;
    let loader = SyntheticLlamaLoader::new(config.clone());
    let mut execution = LlamaExecutionConfig::default();
    execution.options.metal_paged_kv = Some(paged);
    execution.paged_max_seqs = 4;
    execution.kv_capacity = Some(max_seq_len);
    execution.options.cuda_graph = false;
    execution.options.unified_graph = false;
    execution.batched_graph = true;
    let actual =
        Model::new_with_execution_config(config.clone(), &loader, execution.clone()).unwrap();
    execution.batched_graph = false;
    let expected = Model::new_with_execution_config(config, &loader, execution).unwrap();
    for model in [&actual, &expected] {
        assert!(model.supports_batched_decode);
        assert!(!model.cfg.sandwich_norms);
    }
    (actual, expected)
}

fn full_logits(actual: &[f32], expected: &[f32], stage: &str) {
    assert_eq!(actual.len(), 128);
    assert_eq!(expected.len(), actual.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "{stage}: logit {index}: {a} vs {e}"
        );
        assert!(
            (a - e).abs() <= 1e-5 + 1e-5 * e.abs(),
            "{stage}: logit {index}: {a} vs {e}"
        );
    }
}

fn prefill(actual: &mut Model, expected: &mut Model, id: &str, prompt: &[u32]) {
    full_logits(
        &actual.prefill_internal(id, prompt),
        &expected.prefill_internal(id, prompt),
        &format!("prefill {id} prompt={prompt:?}"),
    );
    assert_eq!(actual.paged_pools.is_some(), expected.paged_pools.is_some());
}

fn decode(actual: &mut Model, expected: &mut Model, batch: &[(String, u32, u32)]) {
    let actual_rows = actual.decode_batch_internal_with_full_logits(batch, true);
    let expected_rows = expected.decode_batch_internal_with_full_logits(batch, true);
    assert_eq!(actual_rows.len(), batch.len());
    assert_eq!(expected_rows.len(), batch.len());
    for (row, (actual_row, expected_row)) in actual_rows.iter().zip(&expected_rows).enumerate() {
        full_logits(
            actual_row,
            expected_row,
            &format!("decode row={row} batch={batch:?}"),
        );
    }
    for (id, _, position) in batch {
        assert!(actual.kv_caches[id]
            .iter()
            .all(|cache| cache.len == *position as usize + 1));
    }
}

fn batch(ids: &[&str], position: u32, token: u32) -> Vec<(String, u32, u32)> {
    ids.iter()
        .enumerate()
        .map(|(index, id)| ((*id).to_owned(), token + index as u32, position))
        .collect()
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_same_shape_switches_original_cache_bindings() {
    let (mut actual, mut expected) = models();
    for (id, prompt) in [
        ("a0", [1, 2, 3]),
        ("a1", [11, 12, 13]),
        ("b0", [31, 32, 33]),
        ("b1", [41, 42, 43]),
    ] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..7 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1"], position, position + 50),
        );
    }
    // Set membership is published only after native end_graph_capture succeeds;
    // every capture-step output above also had to match the eager full logits.
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    decode(&mut actual, &mut expected, &batch(&["b0", "b1"], 3, 80));
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_three_then_four_rows_preserves_every_logit() {
    let (mut actual, mut expected) = models();
    // Pre-size both models before capture so m=4 does not accidentally retire
    // the m=3 graph through an unrelated scratch/logit allocation boundary.
    for model in [&mut actual, &mut expected] {
        model.ensure_scratch(4);
        model
            .run_preparation_frame(|model, frame| {
                model
                    .scratch
                    .ensure_batch_logits(&mut frame.context, &model.cfg, 4);
            })
            .into_legacy_result();
    }
    for (id, prompt) in [
        ("a0", [1, 2, 3]),
        ("a1", [11, 12, 13]),
        ("a2", [21, 22, 23]),
        ("a3", [31, 32, 33]),
    ] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..7 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1", "a2"], position, position + 50),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    let mut four = batch(&["a0", "a1", "a2"], 7, 80);
    four.push(("a3".to_owned(), 90, 3));
    decode(&mut actual, &mut expected, &four);
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_two_layers_grow_attention_score_capacity() {
    // Different synthetic layer weights and original per-layer caches exercise
    // every independent pointer-table slot. Full attention crosses two graph
    // score-capacity boundaries while physical storage remains unchanged.
    let (mut actual, mut expected) = configured_models(false, 0, 2, 64);
    for (id, prompt) in [("a0", [1, 2, 3]), ("a1", [11, 12, 13])] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..20 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1"], position, 40 + position),
        );
    }
    // Each entry is installed only after native capture succeeded. Full
    // outputs above cover capture, same-key replay, and re-capture at growth.
    assert!(actual.batched_graph_keys_seen.len() >= 3);
    assert!(!actual.batched_graph_failed);
    for id in ["a0", "a1"] {
        assert!(actual.kv_caches[id].iter().all(|cache| cache.len == 20));
    }
    actual.close().unwrap();
    expected.close().unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_batched_graph_cuda_paged_replays_new_pages_and_cache_incarnations() {
    let (mut actual, mut expected) = configured_models(true, 0, 2, 64);
    for (id, prompt) in [
        ("a0", [1, 2, 3]),
        ("a1", [11, 12, 13]),
        ("b0", [21, 22, 23]),
        ("b1", [31, 32, 33]),
    ] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    assert!(actual.paged_pools.is_some());
    for position in 3..19 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["a0", "a1"], position, 40 + position),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    for id in ["a0", "a1"] {
        assert!(actual.kv_caches[id]
            .iter()
            .all(|cache| cache.len == 19 && cache.paged_block_indices.len() >= 2));
    }
    // A different live cohort must capture its original owners, then replay.
    for position in 3..8 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["b0", "b1"], position, 70 + position),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    for id in ["b0", "b1"] {
        actual.release(id);
        expected.release(id);
        assert!(actual.batched_graph_keys_seen.is_empty());
        assert!(!actual.kv_caches.contains_key(id));
    }
    // Reuse the same public IDs with new prompts and physical block owners.
    for (id, prompt) in [("b0", [91, 92, 93]), ("b1", [101, 102, 103])] {
        prefill(&mut actual, &mut expected, id, &prompt);
    }
    for position in 3..8 {
        decode(
            &mut actual,
            &mut expected,
            &batch(&["b0", "b1"], position, 110 + position),
        );
    }
    assert!(!actual.batched_graph_keys_seen.is_empty());
    assert!(!actual.batched_graph_failed);
    actual.close().unwrap();
    expected.close().unwrap();
}
