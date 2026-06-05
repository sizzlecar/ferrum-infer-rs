use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::Backend;
use ferrum_models::{load_runtime_lora_adapter, ActiveLoraAdapter, LoraAdapterConfig};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::collections::HashMap;
use tempfile::TempDir;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn write_adapter(
    dir: &std::path::Path,
    config: &LoraAdapterConfig,
    tensors: Vec<(&str, Vec<usize>, Vec<f32>)>,
) {
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_vec_pretty(config).unwrap(),
    )
    .unwrap();
    let encoded: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            let bytes = f32_bytes(values);
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            (
                (*name).to_string(),
                TensorView::new(Dtype::F32, shape.clone(), leaked).unwrap(),
            )
        })
        .collect();
    serialize_to_file(
        encoded,
        &None::<HashMap<String, String>>,
        &dir.join("adapter_model.safetensors"),
    )
    .unwrap();
}

#[test]
fn runtime_lora_loader_materializes_projection_metadata() {
    let tmp = TempDir::new().unwrap();
    let config = LoraAdapterConfig {
        r: 2,
        lora_alpha: 6,
        target_modules: vec!["qkv_proj".to_string()],
        base_model_name_or_path: "qwen3:0.6b".to_string(),
    };
    write_adapter(
        tmp.path(),
        &config,
        vec![
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_A.weight",
                vec![2, 3],
                vec![0.1; 6],
            ),
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_B.weight",
                vec![4, 2],
                vec![0.2; 8],
            ),
        ],
    );

    let adapter = load_runtime_lora_adapter::<CpuBackend>(&ActiveLoraAdapter {
        name: "sql".to_string(),
        path: tmp.path().to_path_buf(),
    })
    .unwrap();

    assert_eq!(adapter.name, "sql");
    assert_eq!(adapter.config.r, 2);
    assert_eq!(adapter.linears.len(), 1);
    let linear = &adapter.linears[0];
    assert_eq!(linear.layer_index, Some(7));
    assert_eq!(linear.target_module, "qkv_proj");
    assert_eq!(linear.in_features, 3);
    assert_eq!(linear.out_features, 4);
    assert_eq!(linear.rank, 2);
    assert_eq!(linear.scaling, 3.0);
}

#[test]
fn runtime_lora_apply_projection_adds_scaled_delta() {
    let tmp = TempDir::new().unwrap();
    let config = LoraAdapterConfig {
        r: 2,
        lora_alpha: 6,
        target_modules: vec!["qkv_proj".to_string()],
        base_model_name_or_path: "qwen3:0.6b".to_string(),
    };
    write_adapter(
        tmp.path(),
        &config,
        vec![
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_A.weight",
                vec![2, 3],
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ),
            (
                "base_model.model.layers.7.self_attn.qkv_proj.lora_B.weight",
                vec![4, 2],
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ),
        ],
    );
    let adapter = load_runtime_lora_adapter::<CpuBackend>(&ActiveLoraAdapter {
        name: "sql".to_string(),
        path: tmp.path().to_path_buf(),
    })
    .unwrap();

    let mut ctx = CpuBackend::new_context();
    let input = CpuBackend::from_slice(&[1.0, 2.0, 3.0]);
    let mut out = CpuBackend::from_slice(&[10.0, 20.0, 30.0, 40.0]);
    adapter
        .apply_projection(&mut ctx, 7, "qkv_proj", &input, &mut out, 1)
        .unwrap();

    assert_eq!(CpuBackend::to_vec(&out, 4), vec![13.0, 26.0, 39.0, 40.0]);
}
