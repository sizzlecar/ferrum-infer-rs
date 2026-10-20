//! Full model outputs across instance-owned workspaces and adapter shapes.

use super::*;
use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
use crate::test_support::SyntheticLlamaLoader;
use ferrum_kernels::backend::MoeLlmBackend;
use ferrum_types::{LlamaExecutionConfig, LlamaExecutionOptions};

fn ranked_adapter(
    name: &str,
    target: &str,
    rank: usize,
    input: usize,
    output: usize,
) -> (TempDir, ActiveLoraAdapter) {
    let directory = TempDir::new().unwrap();
    let config = LoraAdapterConfig {
        r: rank,
        lora_alpha: 2 * rank,
        target_modules: vec![target.into()],
        base_model_name_or_path: "tiny-llama-test".into(),
    };
    std::fs::write(
        directory.path().join("adapter_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let a: Vec<u8> = (0..rank * input)
        .flat_map(|index| {
            let row = index / input;
            (if index % input == row % input {
                0.25f32
            } else {
                0.0
            })
            .to_le_bytes()
        })
        .collect();
    let b: Vec<u8> = (0..output * rank)
        .flat_map(|index| (((index % 11 + 1) as f32) * 0.015625).to_le_bytes())
        .collect();
    let component = if target == "o_proj" {
        "self_attn"
    } else {
        "mlp"
    };
    let a_name = format!("base_model.model.layers.0.{component}.{target}.lora_A.weight");
    let b_name = format!("base_model.model.layers.0.{component}.{target}.lora_B.weight");
    serialize_to_file(
        [
            (
                a_name,
                TensorView::new(Dtype::F32, vec![rank, input], &a).unwrap(),
            ),
            (
                b_name,
                TensorView::new(Dtype::F32, vec![output, rank], &b).unwrap(),
            ),
        ],
        &None::<HashMap<String, String>>,
        &directory.path().join("adapter_model.safetensors"),
    )
    .unwrap();
    let adapter = ActiveLoraAdapter {
        name: name.into(),
        path: directory.path().to_path_buf(),
    };
    (directory, adapter)
}

fn model<B: MoeLlmBackend>(cfg: &LlamaFamilyConfig) -> LlamaFamilyModel<B> {
    LlamaFamilyModel::<B>::new_with_execution_config(
        cfg.clone(),
        &SyntheticLlamaLoader::new(cfg.clone()),
        LlamaExecutionConfig {
            kv_capacity: Some(32),
            options: LlamaExecutionOptions {
                metal_paged_kv: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
}

fn full_prefill<B: MoeLlmBackend>(
    actual: &mut LlamaFamilyModel<B>,
    reference: &mut LlamaFamilyModel<B>,
    cache: &str,
    tokens: &[u32],
    vocab: usize,
) {
    let logits = actual.prefill(cache, tokens);
    let expected = reference.prefill(cache, tokens);
    assert_eq!(logits.len(), vocab);
    assert!(logits
        .iter()
        .chain(&expected)
        .all(|value| value.is_finite()));
    assert_eq!(logits, expected);
    assert_eq!(actual.cache_len(cache), reference.cache_len(cache));
}

fn instance_workspaces_preserve_interleaved_outputs<B: MoeLlmBackend>() {
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
    let (_small_dir, small) =
        ranked_adapter("small", "o_proj", 1, cfg.hidden_size, cfg.hidden_size);
    let (_wide_dir, wide) = ranked_adapter(
        "wide",
        "gate_up_proj",
        3,
        cfg.hidden_size,
        2 * cfg.intermediate_size,
    );
    let mut first = model::<B>(&cfg);
    let mut peer = model::<B>(&cfg);
    let mut small_reference = model::<B>(&cfg);
    let mut wide_reference = model::<B>(&cfg);
    let mut peer_reference = model::<B>(&cfg);

    // Both independent contexts load the very same adapter file. Merely
    // loading even the larger adapter never allocates execution scratch.
    for model in [&mut first, &mut peer] {
        model.preload_lora_adapter(&small).unwrap();
        model.preload_lora_adapter(&wide).unwrap();
        assert_eq!(model.lora_workspace.capacities_for_test(), (0, 0));
        model
            .set_lora_adapter_for_cache("small", Some(small.clone()))
            .unwrap();
        model
            .set_lora_adapter_for_cache("wide", Some(wide.clone()))
            .unwrap();
    }
    // Each oracle owns only one adapter. It never follows the tested model's
    // cross-adapter scratch reuse, so a stale stride/layout cannot agree merely
    // because both sides perform the same transition.
    for (reference, cache, adapter) in [
        (&mut small_reference, "small", &small),
        (&mut wide_reference, "wide", &wide),
        (&mut peer_reference, "small", &small),
    ] {
        reference
            .set_lora_adapter_for_cache(cache, Some(adapter.clone()))
            .unwrap();
    }
    full_prefill(
        &mut first,
        &mut small_reference,
        "small",
        &[1, 2],
        cfg.vocab_size,
    );
    full_prefill(
        &mut peer,
        &mut peer_reference,
        "small",
        &[6],
        cfg.vocab_size,
    );
    assert_eq!(
        first.lora_workspace.capacities_for_test(),
        (2, 2 * cfg.hidden_size)
    );
    assert_eq!(
        peer.lora_workspace.capacities_for_test(),
        (1, cfg.hidden_size)
    );

    // Same rows, larger rank AND output: checking only a rows high-water
    // mark would reuse undersized buffers and corrupt this actual forward.
    full_prefill(
        &mut first,
        &mut wide_reference,
        "wide",
        &[3, 4],
        cfg.vocab_size,
    );
    assert_eq!(
        first.lora_workspace.capacities_for_test(),
        (6, 4 * cfg.intermediate_size)
    );
    assert_eq!(
        peer.lora_workspace.capacities_for_test(),
        (1, cfg.hidden_size)
    );

    // A smaller adapter can reuse the larger pair with more rows. Then the
    // wide adapter grows the same instance's pair for its own actual rows.
    full_prefill(
        &mut first,
        &mut small_reference,
        "small",
        &[3, 4, 5],
        cfg.vocab_size,
    );
    assert_eq!(
        first.lora_workspace.capacities_for_test(),
        (6, 4 * cfg.intermediate_size)
    );
    full_prefill(
        &mut first,
        &mut wide_reference,
        "wide",
        &[5, 6, 7],
        cfg.vocab_size,
    );
    assert_eq!(
        first.lora_workspace.capacities_for_test(),
        (9, 6 * cfg.intermediate_size)
    );
    full_prefill(
        &mut peer,
        &mut peer_reference,
        "small",
        &[7, 8],
        cfg.vocab_size,
    );
    assert_eq!(
        peer.lora_workspace.capacities_for_test(),
        (2, 2 * cfg.hidden_size)
    );
    assert_eq!(
        first.lora_workspace.capacities_for_test(),
        (9, 6 * cfg.intermediate_size)
    );

    // Retire and drop a complete peer (including its weights and workspace),
    // then continue both live caches of the first original execution owner.
    peer.close().unwrap();
    drop(peer);
    peer_reference.close().unwrap();
    drop(peer_reference);
    for (cache, reference) in [
        ("small", &mut small_reference),
        ("wide", &mut wide_reference),
    ] {
        let position = first.cache_len(cache);
        let logits = first.decode(cache, 9, position as u32);
        let expected = reference.decode(cache, 9, position as u32);
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite()));
        assert_eq!(logits, expected);
        assert_eq!(first.cache_len(cache), position + 1);
    }
    assert_eq!(
        first.lora_workspace.capacities_for_test(),
        (9, 6 * cfg.intermediate_size)
    );
    first.close().unwrap();
    small_reference.close().unwrap();
    wide_reference.close().unwrap();
}

#[test]
fn lora_instance_workspace_interleaves_adapters_and_survives_peer_close() {
    instance_workspaces_preserve_interleaved_outputs::<ferrum_kernels::backend::cpu::CpuBackend>();
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires actual Metal model execution on independent instance contexts"]
fn metal_lora_instance_workspace_interleaves_adapters_and_survives_peer_close() {
    instance_workspaces_preserve_interleaved_outputs::<ferrum_kernels::backend::metal::MetalBackend>(
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires actual CUDA model execution on independent instance contexts"]
fn cuda_lora_instance_workspace_interleaves_adapters_and_survives_peer_close() {
    instance_workspaces_preserve_interleaved_outputs::<ferrum_kernels::backend::cuda::CudaBackend>(
    );
}
