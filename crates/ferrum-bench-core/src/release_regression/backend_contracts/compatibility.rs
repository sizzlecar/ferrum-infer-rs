//! Existing execution routes reuse shared operator evidence and their own
//! accelerator assertions. These bindings do not introduce a weight format or
//! substitute dense matrix results for routed expert execution.
use super::super::contracts::{ContractGroup, ContractTest};
use super::super::{
    Backend, Behavior, CheckDescriptor, EvidenceLayer, ExecutionTarget, Obligation, ObligationScope,
};
use ferrum_types::ModelOutputProtocol;

fn supported(target: &ExecutionTarget) -> bool {
    use Backend::{Cuda, Metal};
    use ModelOutputProtocol::{GemmaThought, HarmonyGptOss, Text};
    match (
        target.execution_path.as_str(),
        target.architecture.as_str(),
        target.backend,
        target.precision.as_str(),
        target.protocol,
    ) {
        (super::PATH, "gemma4_dense", Cuda, "compressed-tensors-w4a16", GemmaThought)
        | (super::PATH, "gpt_oss_moe", Cuda, "mxfp4", HarmonyGptOss)
        | (super::PATH, "qwen3_5_hybrid_moe", Cuda, "block-fp8-e4m3", Text)
        | (super::PATH, "qwen3_5_hybrid_moe", Metal, "gguf-q4_k_s", Text)
        | (super::PATH, "qwen3_attention_moe", Metal, "gguf-q4_k_m", Text)
        | (super::PATH, "qwen3_attention_moe", Cuda, "gptq-int4", Text) => true,
        (
            super::super::submission::LEGACY_EXECUTION_PATH,
            "llama_dense",
            Metal,
            "gguf-q4_k_m",
            Text,
        )
        | (
            super::super::submission::LEGACY_EXECUTION_PATH,
            "llama_dense",
            Cuda,
            "safetensors-bf16",
            Text,
        ) => true,
        _ => false,
    }
}

fn group_id(backend: Backend, behavior: Behavior) -> String {
    let suffix = match behavior {
        Behavior::KernelNumerics => "numerics",
        Behavior::KernelBoundaries => "boundaries",
        _ => unreachable!("compatibility groups cover operator behavior"),
    };
    format!(
        "backend-contract.{}.existing-routes-{suffix}",
        super::backend_name(backend)
    )
}

pub(super) fn groups(backend: Backend) -> Vec<ContractGroup> {
    let (module, names): (&str, &[&str]) = match backend {
        Backend::Cpu => return Vec::new(),
        Backend::Metal => (
            "backend::metal::vnext_ops::moe::tests",
            &[
                "mixed_q4k_q6k_routed_down_kernels_match_dequantized_cpu",
                "q4k_workspace_formula_covers_runtime_layout",
                "q4k_q6k_routed_workspace_formula_covers_runtime_layout",
            ],
        ),
        Backend::Cuda => (
            "backend::cuda::marlin::tests",
            &[
                "gpt_oss_mxfp4_marlin_bias_p32_two_experts_matches_logical_source",
                "gpt_oss_mxfp4_marlin_moe_bf16_ffi_matches_source_reference_for_four_cases",
                "gpt_oss_mxfp4_marlin_moe_executes_official_down_with_physical_k_padding",
                "gpt_oss_mxfp4_official_down_two_experts_four_rows_preserves_padding_and_matches_source",
                "gpt_oss_mxfp4_official_gate_up_two_experts_four_rows_matches_source",
                "qwen36_a3_fp8_marlin_moe_ffi_matches_cpu_reference_for_four_cases",
                "marlin_moe_mxfp4_bf16_args_compute_required_buffer_shapes",
                "marlin_moe_mxfp4_bf16_args_reject_invalid_pointers",
                "marlin_moe_mxfp4_bf16_args_reject_invalid_shapes_and_overflow",
                "marlin_moe_mxfp4_bf16_extern_abi_matches_locked_export",
                "marlin_moe_ffi_status_preserves_failure_stage_and_cuda_status",
                "marlin_moe_raw_args_accept_supported_modes",
                "marlin_moe_raw_args_reject_invalid_pointers",
                "marlin_moe_raw_args_reject_invalid_shapes_and_config",
                "marlin_moe_raw_args_reject_inconsistent_optional_modes",
            ],
        ),
    };
    // A backend receipt also requires all shared numerical and matrix groups.
    // Their tests exercise the affected precision, rotary, workspace and native
    // block providers. The additional tests below retain the existing routes.
    let mut tests: Vec<_> = names
        .iter()
        .map(|name| ContractTest {
            package: "ferrum-kernels".into(),
            target: "ferrum_kernels".into(),
            kind: "lib".into(),
            name: format!("{module}::{name}"),
        })
        .collect();
    tests.push(ContractTest {
        package: "ferrum-testkit".into(),
        target: "op_diff".into(),
        kind: "test".into(),
        name: format!("required_legacy_{}_operators", super::backend_name(backend)),
    });
    [Behavior::KernelNumerics, Behavior::KernelBoundaries]
        .into_iter()
        .map(|behavior| ContractGroup {
            id: group_id(backend, behavior),
            behavior,
            entrypoints: Vec::new(),
            tests: tests.clone(),
        })
        .collect()
}

fn checker_id(target: &ExecutionTarget, behavior: Behavior) -> String {
    format!(
        "{}.{}.{}",
        group_id(target.backend, behavior),
        target.architecture,
        target.precision
    )
}

pub(super) fn descriptors(targets: &[ExecutionTarget]) -> Vec<CheckDescriptor> {
    targets
        .iter()
        .filter(|target| supported(target))
        .flat_map(|target| {
            [Behavior::KernelNumerics, Behavior::KernelBoundaries]
                .into_iter()
                .map(move |behavior| CheckDescriptor {
                    id: checker_id(target, behavior),
                    behavior,
                    layer: EvidenceLayer::BackendNumerics,
                    entrypoints: Vec::new(),
                    target: Some(target.clone()),
                })
        })
        .collect()
}

pub(super) fn expected_checker(backend: Backend, obligation: &Obligation) -> Option<String> {
    let ObligationScope::Target { target } = &obligation.scope else {
        return None;
    };
    (target.backend == backend
        && supported(target)
        && matches!(
            obligation.behavior,
            Behavior::KernelNumerics | Behavior::KernelBoundaries
        ))
    .then(|| checker_id(target, obligation.behavior))
}
