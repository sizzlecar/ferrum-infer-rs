use ferrum_types::{
    CompiledKernelFeatures, ExecutionResourceAuthority, FerrumConfigBuilder, HardwareCapabilities,
    ModelCapabilities, RuntimeConfigSnapshot, WorkloadProfile,
};
use serde_json::Value;

const GIB: u64 = 1024 * 1024 * 1024;

fn llama_8b(quantization: &str) -> ModelCapabilities {
    ModelCapabilities {
        architecture: "llama".to_string(),
        quantization: Some(quantization.to_string()),
        moe: None,
        max_context_len: Some(131_072),
        num_hidden_layers: Some(32),
        head_dim: Some(128),
        kv_heads: Some(8),
        estimated_weight_bytes: Some(5 * GIB),
        recurrent_state_bytes_per_sequence: None,
        supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        graph_safe_moe: false,
    }
}

fn qwen3_moe_gguf() -> ModelCapabilities {
    let mut model = ModelCapabilities::qwen3_30b_a3b_gptq_int4();
    model.quantization = Some("q4_k_m".to_string());
    model.estimated_weight_bytes = Some(22 * GIB);
    model
}

fn metal_hardware() -> HardwareCapabilities {
    HardwareCapabilities {
        backend: "metal".to_string(),
        vram_bytes: Some(64 * GIB),
        supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        supported_kv_dtypes: vec!["fp16".to_string()],
        compiled_features: CompiledKernelFeatures::default(),
        ..HardwareCapabilities::unknown()
    }
}

fn cuda_hardware() -> HardwareCapabilities {
    HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2())
}

fn resolve(model: ModelCapabilities, hardware: HardwareCapabilities) -> Value {
    let workload = WorkloadProfile::serving_default_for_hardware(&hardware);
    FerrumConfigBuilder::new(RuntimeConfigSnapshot::default())
        .with_model_capabilities(model)
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .with_execution_resource_authority(ExecutionResourceAuthority::PlanRuntime)
        .resolve()
        .expect("runtime preset must resolve")
        .effective_config_document()
}

fn decision<'a>(document: &'a Value, selection: &str) -> &'a Value {
    document["decisions"]
        .as_array()
        .expect("decisions must be an array")
        .iter()
        .find(|item| item["selection"] == selection)
        .unwrap_or_else(|| panic!("missing decision {selection}"))
}

fn assert_common_contract(document: &Value, backend: &str, architecture: &str) {
    assert_eq!(document["hardware_capabilities"]["backend"], backend);
    assert_eq!(document["model_capabilities"]["architecture"], architecture);
    assert_eq!(document["admission"]["resource_authority"], "plan_runtime");

    for name in [
        "attention_decode_backend",
        "moe_graph_policy",
        "moe_implementation",
        "prefix_cache_policy",
    ] {
        let item = decision(document, name);
        let selected = item["selected"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} must select a value"));
        let candidates = item["candidates"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} candidates must be an array"));
        assert!(
            candidates.iter().any(|candidate| candidate == selected),
            "{name} selected value must be one of its candidates"
        );
    }

    assert!(
        decision(document, "scheduler_admission_policy")["selected"]
            .as_str()
            .is_some_and(|selected| !selected.is_empty()),
        "scheduler policy must resolve to a non-empty product strategy"
    );
}

#[test]
fn metal_dense_and_moe_presets_choose_supported_product_paths() {
    let dense = resolve(llama_8b("q4_k_m"), metal_hardware());
    assert_common_contract(&dense, "metal", "llama");
    assert_eq!(
        decision(&dense, "attention_decode_backend")["selected"],
        "portable"
    );
    assert_eq!(
        decision(&dense, "moe_implementation")["selected"],
        "legacy_moe"
    );

    let moe = resolve(qwen3_moe_gguf(), metal_hardware());
    assert_common_contract(&moe, "metal", "qwen3_moe");
    assert_eq!(
        decision(&moe, "attention_decode_backend")["selected"],
        "portable"
    );
    assert_eq!(
        decision(&moe, "moe_implementation")["selected"],
        "legacy_moe"
    );
}

#[test]
fn cuda_dense_and_moe_presets_choose_supported_product_paths() {
    let dense = resolve(llama_8b("gptq_int4"), cuda_hardware());
    assert_common_contract(&dense, "cuda", "llama");
    assert_eq!(
        decision(&dense, "attention_decode_backend")["selected"],
        "cuda_native_adaptive"
    );
    assert_eq!(
        decision(&dense, "moe_implementation")["selected"],
        "legacy_moe"
    );

    let moe = resolve(
        ModelCapabilities::qwen3_30b_a3b_gptq_int4(),
        cuda_hardware(),
    );
    assert_common_contract(&moe, "cuda", "qwen3_moe");
    assert_eq!(
        decision(&moe, "attention_decode_backend")["selected"],
        "cuda_native_adaptive"
    );
    assert_eq!(
        decision(&moe, "moe_implementation")["selected"],
        "vllm_marlin_moe_device_route_pair_ids"
    );
}
