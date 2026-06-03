use ferrum_models::{
    load_startup_lora_adapter, load_startup_lora_adapters, LoraAdapterConfig, StartupLoraSpec,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::collections::HashMap;
use std::path::Path;
use tempfile::TempDir;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn write_adapter(dir: &Path, config: &LoraAdapterConfig, tensors: Vec<(&str, Vec<usize>, Vec<f32>)>) {
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
    serialize_to_file(encoded, &None::<HashMap<String, String>>, &dir.join("adapter_model.safetensors"))
        .unwrap();
}

fn valid_config() -> LoraAdapterConfig {
    LoraAdapterConfig {
        r: 2,
        lora_alpha: 4,
        target_modules: vec!["linear".to_string()],
        base_model_name_or_path: "qwen3:0.6b".to_string(),
    }
}

#[test]
fn lora_loader_reads_config_and_safetensors() {
    let tmp = TempDir::new().unwrap();
    write_adapter(
        tmp.path(),
        &valid_config(),
        vec![
            (
                "base_model.model.layers.0.linear.lora_A.weight",
                vec![2, 3],
                vec![0.0; 6],
            ),
            (
                "base_model.model.layers.0.linear.lora_B.weight",
                vec![4, 2],
                vec![0.0; 8],
            ),
        ],
    );

    let adapter = load_startup_lora_adapter("sql", tmp.path(), "qwen3:0.6b:sql").unwrap();
    assert_eq!(adapter.name, "sql");
    assert_eq!(adapter.public_model_id, "qwen3:0.6b:sql");
    assert_eq!(adapter.config.r, 2);
    assert_eq!(adapter.tensors.len(), 1);
    assert_eq!(adapter.tensors[0].in_features, 3);
    assert_eq!(adapter.tensors[0].out_features, 4);
}

#[test]
fn lora_loader_shape_mismatch_fails() {
    let tmp = TempDir::new().unwrap();
    write_adapter(
        tmp.path(),
        &valid_config(),
        vec![
            ("linear.lora_A.weight", vec![1, 3], vec![0.0; 3]),
            ("linear.lora_B.weight", vec![4, 2], vec![0.0; 8]),
        ],
    );

    let err = load_startup_lora_adapter("bad", tmp.path(), "base:bad").unwrap_err();
    assert!(err.to_string().contains("LoRA rank mismatch"));
}

#[test]
fn lora_loader_unknown_target_fails() {
    let tmp = TempDir::new().unwrap();
    let mut config = valid_config();
    config.target_modules = vec!["not_a_projection".to_string()];
    write_adapter(tmp.path(), &config, vec![]);

    let err = load_startup_lora_adapter("bad", tmp.path(), "base:bad").unwrap_err();
    assert!(err.to_string().contains("unsupported LoRA target module"));
}

#[test]
fn lora_loader_duplicate_names_fail() {
    let tmp = TempDir::new().unwrap();
    write_adapter(
        tmp.path(),
        &valid_config(),
        vec![
            ("linear.lora_A.weight", vec![2, 3], vec![0.0; 6]),
            ("linear.lora_B.weight", vec![4, 2], vec![0.0; 8]),
        ],
    );
    let specs = vec![
        StartupLoraSpec {
            name: "sql".to_string(),
            path: tmp.path().to_path_buf(),
        },
        StartupLoraSpec {
            name: "sql".to_string(),
            path: tmp.path().to_path_buf(),
        },
    ];

    let err = load_startup_lora_adapters("qwen3:0.6b", None, &specs).unwrap_err();
    assert!(err.to_string().contains("duplicate LoRA adapter name"));
}
