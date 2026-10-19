//! Real adapter loading/numerics, CPU graph eligibility, and required CUDA replay.

use std::collections::HashMap;

use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tempfile::TempDir;

use crate::common::DecoderOnlyLLM;
use crate::lora::{ActiveLoraAdapter, LoraAdapterConfig};
use crate::test_support::{tiny_llama_model, TinyLlamaConfig};

fn adapter_fixture(width: usize) -> (TempDir, ActiveLoraAdapter) {
    let directory = TempDir::new().unwrap();
    let config = LoraAdapterConfig {
        r: 1,
        lora_alpha: 2,
        target_modules: vec!["o_proj".into()],
        base_model_name_or_path: "tiny-llama-test".into(),
    };
    std::fs::write(
        directory.path().join("adapter_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let a: Vec<u8> = (0..width)
        .flat_map(|index| (if index == 0 { 1.0f32 } else { 0.0 }).to_le_bytes())
        .collect();
    let b: Vec<u8> = (0..width)
        .flat_map(|index| ((index as f32 + 1.0) * 0.25).to_le_bytes())
        .collect();
    serialize_to_file(
        [
            (
                "base_model.model.layers.0.self_attn.o_proj.lora_A.weight",
                TensorView::new(Dtype::F32, vec![1, width], &a).unwrap(),
            ),
            (
                "base_model.model.layers.0.self_attn.o_proj.lora_B.weight",
                TensorView::new(Dtype::F32, vec![width, 1], &b).unwrap(),
            ),
        ],
        &None::<HashMap<String, String>>,
        &directory.path().join("adapter_model.safetensors"),
    )
    .unwrap();
    let adapter = ActiveLoraAdapter {
        name: "delta".into(),
        path: directory.path().to_path_buf(),
    };
    (directory, adapter)
}

#[test]
fn lora_single_decode_graph_eligibility_tracks_real_bindings() {
    let cfg = TinyLlamaConfig::default();
    let mut model = tiny_llama_model(&cfg);
    let (_directory, adapter) = adapter_fixture(cfg.hidden_size);
    model.runtime_env.cuda_graph = true;
    assert!(model.single_decode_graph_enabled());

    model.preload_lora_adapter(&adapter).unwrap();
    assert!(model.lora_cache_adapters.is_empty());
    assert!(model.kv_caches.is_empty());
    assert!(model.single_decode_graph_enabled());

    model
        .set_lora_adapter_for_cache("adapted", Some(adapter.clone()))
        .unwrap();
    assert!(!model.single_decode_graph_enabled());
    // An ordinary sequence does not remove another sequence's binding. Its
    // single-item fallback must also remain ineligible for the shared slot.
    model.set_lora_adapter_for_cache("base", None).unwrap();
    assert!(!model.single_decode_graph_enabled());
    model.set_lora_adapter_for_cache("adapted", None).unwrap();
    assert!(model.single_decode_graph_enabled());
    assert_eq!(model.lora_adapters.len(), 1);

    model
        .set_lora_adapter_for_cache("adapted", Some(adapter))
        .unwrap();
    model.release("adapted");
    assert!(model.single_decode_graph_enabled());
    assert!(model.runtime_env.cuda_graph);
}

#[test]
fn lora_graph_requested_decode_stays_eager_and_preserves_base_adapter_base() {
    let cfg = TinyLlamaConfig::default();
    let mut graph_requested = tiny_llama_model(&cfg);
    let mut eager = tiny_llama_model(&cfg);
    graph_requested.runtime_env.cuda_graph = true;
    let (_directory, adapter) = adapter_fixture(cfg.hidden_size);
    let mut outputs = Vec::new();
    for (cache_id, selection) in [
        ("base-before", None),
        ("adapted", Some(adapter)),
        ("base-after", None),
    ] {
        graph_requested
            .set_lora_adapter_for_cache(cache_id, selection.clone())
            .unwrap();
        eager
            .set_lora_adapter_for_cache(cache_id, selection.clone())
            .unwrap();
        let prefill = graph_requested.prefill(cache_id, &[0, 1, 2]);
        assert_eq!(prefill, eager.prefill(cache_id, &[0, 1, 2]));
        if selection.is_some() {
            // CPU's real BackendGraph rejects capture. Making capture ready
            // proves this production branch skips the attempt for an adapter;
            // it does not simulate a CUDA graph or claim replay correctness.
            graph_requested.graph_warmup = 3;
            graph_requested.graph_capture_failed = false;
        }
        let decode = graph_requested.decode(cache_id, 3, 3);
        assert_eq!(decode, eager.decode(cache_id, 3, 3));
        if selection.is_some() {
            assert!(!graph_requested.graph_capture_failed);
        }
        outputs.push((prefill, decode));
        graph_requested.release(cache_id);
        eager.release(cache_id);
    }
    assert_eq!(outputs[0], outputs[2]);
    for (base, adapted) in [
        (&outputs[0].0, &outputs[1].0),
        (&outputs[0].1, &outputs[1].1),
    ] {
        assert!(base.iter().chain(adapted).all(|value| value.is_finite()));
        assert!(base.iter().zip(adapted).any(|(a, b)| (a - b).abs() > 1e-6));
    }
    assert!(graph_requested.runtime_env.cuda_graph);
}

#[test]
fn lora_workspace_prepares_only_bound_adapter_and_current_rows() {
    let cfg = TinyLlamaConfig::default();
    let mut model = tiny_llama_model(&cfg);
    let mut reference = tiny_llama_model(&cfg);
    let (_first_directory, first) = adapter_fixture(cfg.hidden_size);
    let (_second_directory, mut second) = adapter_fixture(cfg.hidden_size);
    second.name = "unused".into();
    model.preload_lora_adapter(&first).unwrap();
    model.preload_lora_adapter(&second).unwrap();
    model.ensure_scratch(16);
    assert_eq!(
        model.prefill("base", &[0, 1, 2, 3, 4, 5]),
        reference.prefill("base", &[0, 1, 2, 3, 4, 5])
    );
    assert_eq!(
        model.lora_adapters[&first.name].workspace_rows_for_test(),
        0
    );
    assert_eq!(
        model.lora_adapters[&second.name].workspace_rows_for_test(),
        0
    );

    model
        .set_lora_adapter_for_cache("adapted", Some(first.clone()))
        .unwrap();
    reference
        .set_lora_adapter_for_cache("adapted", Some(first.clone()))
        .unwrap();
    for chunk in [&[0, 1][..], &[2, 3][..]] {
        assert_eq!(
            model.prefill("adapted", chunk),
            reference.prefill("adapted", chunk)
        );
        assert_eq!(
            model.lora_adapters[&first.name].workspace_rows_for_test(),
            2
        );
        assert_eq!(
            model.lora_adapters[&second.name].workspace_rows_for_test(),
            0
        );
    }
    assert_eq!(model.cache_len("adapted"), 4);
    assert_eq!(
        model.decode("adapted", 4, 4),
        reference.decode("adapted", 4, 4)
    );
    assert_eq!(
        model.lora_adapters[&first.name].workspace_rows_for_test(),
        2
    );
    assert_eq!(
        model.prefill("adapted", &[5, 6, 7]),
        reference.prefill("adapted", &[5, 6, 7])
    );
    assert_eq!(model.cache_len("adapted"), 8);
    assert_eq!(
        model.lora_adapters[&first.name].workspace_rows_for_test(),
        3
    );
    assert_eq!(
        model.lora_adapters[&second.name].workspace_rows_for_test(),
        0
    );
    model.release("base");
    model.release("adapted");
    model.close().unwrap();
    reference.close().unwrap();
}

#[test]
fn lora_preload_validates_identity_and_shape_without_sequence_state() {
    let cfg = TinyLlamaConfig::default();
    let mut model = tiny_llama_model(&cfg);
    let (_first_directory, first) = adapter_fixture(cfg.hidden_size);
    let (_second_directory, conflicting) = adapter_fixture(cfg.hidden_size);
    let (_bad_directory, mut bad_shape) = adapter_fixture(cfg.hidden_size + 1);
    bad_shape.name = "bad-shape".into();

    model.preload_lora_adapter(&first).unwrap();
    model.preload_lora_adapter(&first).unwrap();
    assert_eq!(model.lora_adapters.len(), 1);
    assert!(model.kv_caches.is_empty());
    assert!(model.lora_cache_adapters.is_empty());
    assert!(model.preload_lora_adapter(&conflicting).is_err());
    assert!(model.preload_lora_adapter(&bad_shape).is_err());
    assert_eq!(model.lora_adapters.len(), 1);
    assert_eq!(model.lora_adapters[&first.name].path, first.path);
    assert!(model.kv_caches.is_empty());
    assert!(model.lora_cache_adapters.is_empty());

    model
        .set_lora_adapter_for_cache("adapted", Some(first.clone()))
        .unwrap();
    assert!(model
        .set_lora_adapter_for_cache("adapted", Some(conflicting))
        .is_err());
    assert_eq!(
        model.active_lora_adapter_for_cache("adapted").unwrap().path,
        first.path
    );
    assert!(model.kv_caches.is_empty());
}

#[cfg(feature = "cuda")]
#[test]
fn lora_cuda_existing_base_graph_cannot_bypass_adapter() {
    use ferrum_kernels::backend::{cuda::CudaBackend, BackendGraph};

    use super::{LlamaFamilyConfig, LlamaFamilyModel, SINGLE_ITEM_GRAPH_KEY};
    use crate::test_support::SyntheticLlamaLoader;

    // One real model owns both the eager oracle and the captured graph on its
    // original context. A fresh context cannot replay this instance's graph.
    let cfg = LlamaFamilyConfig {
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        vocab_size: 128,
        max_seq_len: 32,
        ..Default::default()
    };
    let loader = SyntheticLlamaLoader::new(cfg.clone());
    let mut model = LlamaFamilyModel::<CudaBackend>::new(cfg.clone(), &loader).unwrap();
    // This regression exercises the legacy contiguous FP16-KV single graph.
    model.runtime_env.metal_paged_kv = Some(false);
    model.runtime_env.cuda_graph = false;
    let (_directory, adapter) = adapter_fixture(cfg.hidden_size);
    model
        .set_lora_adapter_for_cache("oracle", Some(adapter.clone()))
        .unwrap();
    let oracle_prefill = model.prefill("oracle", &[0, 1, 2]);
    let oracle_decode = model.decode("oracle", 3, 3);
    assert!(oracle_prefill
        .iter()
        .chain(&oracle_decode)
        .all(|x| x.is_finite()));
    model.release("oracle");

    model.runtime_env.cuda_graph = true;
    model.prefill("base-live", &[0, 1, 2]);
    for position in 3..8 {
        let output = model.decode("base-live", position, position);
        assert!(output.iter().all(|x| x.is_finite()));
    }
    assert!(
        !model.graph_capture_failed,
        "a real base graph must capture"
    );
    assert!(model.kv_caches.contains_key("base-live"));

    model
        .set_lora_adapter_for_cache("adapted", Some(adapter))
        .unwrap();
    let adapted_prefill = model.prefill("adapted", &[0, 1, 2]);
    assert_eq!(adapted_prefill, oracle_prefill);
    assert_eq!(model.cache_len("adapted"), 3);

    // Prove the old key is still installed and actually launches immediately
    // before the adapter call. This repeats the last base write at the same
    // device position; its still-live, separate KV is not the adapter's KV.
    // No CUDA, unsupported capture, or missing graph is a failure, not a skip.
    model
        .run_device_frame(|_, frame| {
            assert!(CudaBackend::replay_graph(&mut frame.context, SINGLE_ITEM_GRAPH_KEY).unwrap());
            frame.completed = Some(super::device_execution::checked_sync::<CudaBackend>(
                &mut frame.context,
            ));
        })
        .into_legacy_result();
    assert_eq!(model.cache_len("adapted"), 3);

    let adapted_decode = model.decode("adapted", 3, 3);
    assert_eq!(
        adapted_decode, oracle_decode,
        "adapter must run instead of the cached base graph"
    );
    assert_eq!(model.cache_len("adapted"), 4);
    assert!(model.runtime_env.cuda_graph);
    model.reset();
}
