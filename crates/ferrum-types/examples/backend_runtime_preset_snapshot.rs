use ferrum_types::{
    CompiledKernelFeatures, FerrumConfigBuilder, HardwareCapabilities, ModelCapabilities,
    RuntimeConfigSnapshot, WorkloadProfile,
};
use serde::Serialize;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Serialize)]
struct SnapshotCase {
    name: &'static str,
    description: &'static str,
    effective_config: serde_json::Value,
}

fn llama_8b_dense(quantization: Option<&str>) -> ModelCapabilities {
    ModelCapabilities {
        architecture: "llama".to_string(),
        quantization: quantization.map(str::to_string),
        moe: None,
        max_context_len: Some(131_072),
        num_hidden_layers: Some(32),
        head_dim: Some(128),
        kv_heads: Some(8),
        estimated_weight_bytes: Some(match quantization {
            Some("q4_k_m") | Some("gptq_int4") => 5 * GIB,
            _ => 16 * GIB,
        }),
        supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        graph_safe_moe: false,
    }
}

fn qwen3_30b_a3b_gguf() -> ModelCapabilities {
    let mut model = ModelCapabilities::qwen3_30b_a3b_gptq_int4();
    model.quantization = Some("q4_k_m".to_string());
    model.estimated_weight_bytes = Some(22 * GIB);
    model
}

fn metal_hardware(vram_bytes: u64) -> HardwareCapabilities {
    HardwareCapabilities {
        backend: "metal".to_string(),
        vram_bytes: Some(vram_bytes),
        supported_dtypes: vec!["fp16".to_string(), "fp32".to_string()],
        supported_kv_dtypes: vec!["fp16".to_string()],
        compiled_features: CompiledKernelFeatures::default(),
        ..HardwareCapabilities::unknown()
    }
}

fn cuda_4090_hardware() -> HardwareCapabilities {
    HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures::m3_fast_path_without_fa2())
}

fn resolve_case(
    name: &'static str,
    description: &'static str,
    model: ModelCapabilities,
    hardware: HardwareCapabilities,
    workload: WorkloadProfile,
) -> SnapshotCase {
    let resolved = FerrumConfigBuilder::new(RuntimeConfigSnapshot::default())
        .with_model_capabilities(model)
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .resolve()
        .unwrap_or_else(|err| panic!("{name}: resolve failed: {err}"));
    SnapshotCase {
        name,
        description,
        effective_config: resolved.effective_config_document(),
    }
}

fn main() {
    let metal = metal_hardware(64 * GIB);
    let cuda = cuda_4090_hardware();
    let cases = vec![
        resolve_case(
            "metal_llama_8b_dense_gguf",
            "Metal + Llama 8B-class dense GGUF",
            llama_8b_dense(Some("q4_k_m")),
            metal.clone(),
            WorkloadProfile::serving_default_for_hardware(&metal),
        ),
        resolve_case(
            "metal_qwen3_30b_a3b_moe_gguf",
            "Metal + Qwen3-30B-A3B MoE GGUF",
            qwen3_30b_a3b_gguf(),
            metal.clone(),
            WorkloadProfile::serving_default_for_hardware(&metal),
        ),
        resolve_case(
            "cuda_llama_8b_dense_gptq",
            "CUDA + Llama 8B-class dense GPTQ",
            llama_8b_dense(Some("gptq_int4")),
            cuda.clone(),
            WorkloadProfile::serving_default_for_hardware(&cuda),
        ),
        resolve_case(
            "cuda_qwen3_30b_a3b_moe_gptq",
            "CUDA + Qwen3-30B-A3B MoE/GPTQ",
            ModelCapabilities::qwen3_30b_a3b_gptq_int4(),
            cuda,
            WorkloadProfile::m3_qwen3_30b_a3b_int4(),
        ),
    ];
    println!(
        "{}",
        serde_json::to_string_pretty(&cases).expect("serialize")
    );
}
