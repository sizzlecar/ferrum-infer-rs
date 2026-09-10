//! Dense-hybrid operator conformance. These fixtures exercise the
//! installed launchers and independent CPU references, including state carry
//! and guarded writes. Dense Marlin targets also require the matrix groups in
//! the same verified report. MoE and legacy operators remain separate.
use super::super::contracts::{ContractGroup, ContractTest};
use super::super::{
    Backend, Behavior, CheckDescriptor, EvidenceLayer, ExecutionTarget, Obligation, ObligationScope,
};

fn supported(target: &ExecutionTarget) -> bool {
    if target.execution_path != super::PATH
        || target.architecture != "qwen3_5_dense_hybrid"
        || target.protocol != ferrum_types::ModelOutputProtocol::Text
    {
        return false;
    }
    match (target.backend, target.precision.as_str()) {
        (Backend::Cpu | Backend::Metal, "safetensors-bf16-f32" | "gguf-q4_k_m") => true,
        (
            Backend::Cuda,
            "safetensors-bf16-f32" | "gguf-q4_k_m" | "compressed-tensors-int4" | "block-fp8-e4m3",
        ) => true,
        (Backend::Cuda | Backend::Metal, "gguf-mixed-4bit") => true,
        _ => false,
    }
}

fn group_id(backend: Backend, behavior: Behavior) -> String {
    let suffix = match behavior {
        Behavior::KernelNumerics => "numerics",
        Behavior::KernelBoundaries => "boundaries",
        _ => unreachable!("only numerical operator behaviors are registered"),
    };
    format!(
        "backend-contract.{}.dense-hybrid-native-{suffix}",
        super::backend_name(backend)
    )
}

fn names(backend: Backend) -> Vec<String> {
    let modules: &[(&str, &[&str])] = match backend {
        Backend::Cpu => &[
            ("tests", &[
                "compressed_cpu_linear_and_embedding_match_decoded_weights",
                "cpu_f32_matrix_does_not_narrow_activations_or_accumulation",
                "cpu_matrix_checks_physical_ranges_before_writing",
                "cpu_elementwise_preserves_declared_float_boundaries",
                "residual_aliases_preserve_the_declared_precision_without_a_copy",
                "cpu_masked_argmax_preserves_logits_and_applies_penalty_once",
            ]),
            ("gated_delta::tests", &[
                "recurrent_steps_match_f64_with_both_declared_head_and_decay_conventions",
                "invalid_recurrent_dimensions_fail_before_memory_access",
            ]),
            ("causal_attention::tests", &[
                "causal_prefix_matches_f64_attention_and_rotary_with_gqa_and_optional_gate",
                "paged_kv_preserves_prefix_across_growth_and_cross_page_heads",
                "paged_kv_rejects_empty_odd_or_unequal_pages",
                "invalid_causal_dimensions_fail_before_memory_access",
            ]),
        ],
        Backend::Metal => &[
            ("native_blocks::tests", &["native_block_decoding_matches_cpu_on_real_metal"]),
            ("linear::native_tests", &["native_block_linears_preserve_rows_offsets_strides_and_precision_on_real_metal"]),
            ("linear::tests", &[
                "native_linear_formats_match_cpu_oracles_on_real_metal",
                "native_dense_swiglu_q4k_q6k_matches_full_cpu_oracle_on_real_metal",
                "native_last_token_q6k_f32_linear_preserves_f32_head_boundary_on_real_metal",
                "shared_k_quant_gemv_honors_batch_offsets_strides_and_tail_rows",
                "shared_quantized_tiled_gemm_matches_prefill_shape_and_preserves_output_guards",
                "raw_linear_workspace_requires_explicit_region_authorization",
            ]),
            ("primitives::tests", &[
                "q4_k_token_embedding_matches_cpu_for_f16_and_f32_on_real_metal",
                "q6_k_and_q8_token_embeddings_preserve_float_boundaries_on_real_metal",
                "f32_master_primitives_preserve_precision_and_residual_aliasing_on_real_metal",
                "native_f16_primitives_match_cpu_references_on_real_metal",
            ]),
            ("gated_delta_attention::tests", &[
                "recurrent_core_matches_cpu_and_preserves_split_decode_state_on_real_metal",
                "chunked_c64_matches_recurrent_oracle_and_non_aligned_state_continuity",
                "launch_extent_validation_rejects_msl_uint_overflow",
            ]),
            ("causal_attention::conformance_tests", &[
                "fixed_page_attention_matches_cpu_and_preserves_split_decode_state_on_real_metal",
                "grouped_decode_head256_with_gate_matches_direct_and_cpu_across_page_boundary_on_real_metal",
                "gqa_tiled_prefill_head256_with_gate_matches_general_and_cpu_across_prefix_page_and_tail_on_real_metal",
            ]),
        ],
        Backend::Cuda => &[
            ("native_blocks::tests", &[
                "native_block_decoding_matches_shared_ggml_oracle_on_cuda",
                "native_block_embeddings_preserve_f16_and_f32_activations_on_cuda",
                "native_block_linears_preserve_f16_and_f32_activations_on_cuda",
                "native_matrix_launcher_preserves_mixed_dense_and_block_partitions",
            ]),
            ("native_io::tests", &[
                "token_lookup_launcher_preserves_exact_decode_offsets_and_invalid_ids_on_cuda",
                "final_row_projection_launcher_preserves_f32_input_and_guarded_logits_on_cuda",
                "token_io_bounds_preserve_nonzero_spans_and_native_launch_capacity",
            ]),
            ("selection::tests", &["masked_selection_matches_scalar_semantics_and_preserves_logits_on_cuda"]),
            ("transformer::native_swiglu::tests", &[
                "native_swiglu_mixed_matrices_match_stage_oracles_on_cuda",
                "native_swiglu_scratch_accounts_only_for_bounded_activations",
            ]),
            ("transformer::precision::tests", &[
                "master_rms_norm_matches_f64_with_half_weights_on_cuda",
                "master_residual_preserves_f32_and_declared_inplace_alias_on_cuda",
            ]),
            ("transformer::f16_tests", &[
                "cublas_swiglu_preserves_stage_oracles_offsets_and_tail_rows_on_cuda",
                "f16_residual_preserves_rounding_alias_and_tail_guards_on_cuda",
                "f16_embedding_preserves_ids_offsets_and_output_guards_on_cuda",
                "planar_gated_activations_match_f64_and_preserve_guards_on_cuda",
            ]),
            ("transformer::marlin_tests", &[
                "marlin_runtime_resets_reused_workspace_and_preserves_matrix_guards_on_cuda",
            ]),
            ("transformer::attention::projection_stitch_tests", &[
                "segmented_projection_stitch_preserves_rows_offsets_and_unwritten_columns_on_cuda",
            ]),
            ("transformer::attention::native_projection::tests", &[
                "native_attention_projection_matches_mixed_matrix_oracle_and_chunk_boundary_on_cuda",
                "native_attention_projection_accounts_for_partitions_and_cuda_row_capacity",
            ]),
            ("transformer::attention::recurrent_tests", &[
                "recurrent_cuda_semantics_preserve_f64_oracle_state_carry_and_isolated_slots",
                "recurrent_master_provider_preserves_hidden_precision_and_residual_aliasing_on_cuda",
            ]),
            ("transformer::causal_attention::numerical_tests", &[
                "causal_kv_carry_crosses_physical_pages_and_matches_f64_attention_on_cuda",
                "causal_master_preserves_f32_hidden_and_both_residual_alias_modes_on_cuda",
            ]),
            ("transformer::causal_attention::rotary_tests", &[
                "standard_partial_rope_rotates_only_its_prefix_on_cuda",
                "proportional_partial_rope_retains_padded_head_pairs_on_cuda",
            ]),
        ],
    };
    modules
        .iter()
        .flat_map(|(module, names)| {
            names.iter().map(move |name| {
                format!(
                    "backend::{}::vnext_ops::{module}::{name}",
                    super::backend_name(backend)
                )
            })
        })
        .collect()
}

pub(super) fn groups(backend: Backend) -> Vec<ContractGroup> {
    // The same small execution verifies arithmetic and guarded state/output
    // boundaries. The harness deduplicates shared assertions across groups.
    let tests: Vec<_> = names(backend)
        .into_iter()
        .map(|name| ContractTest {
            package: "ferrum-kernels".into(),
            target: "ferrum_kernels".into(),
            kind: "lib".into(),
            name,
        })
        .collect();
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
        "{}.{}",
        group_id(target.backend, behavior),
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
                    target: Some(target.clone()),
                    entrypoints: Vec::new(),
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
