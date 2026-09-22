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
            ("hadamard::tests", &[
                "hadamard_1024_selection_requires_exact_simd_threads_and_shared_capacity",
                "hadamard_1024_production_selection_and_generic_fallback_preserve_transforms",
                "hadamard_1024_shuffle_preserves_generic_f32_order_special_values_and_signed_zero",
                "hadamard_forward_preserves_full_width_signs_blocks_and_input_precision_on_metal",
                "hadamard_grouped_permutation_precedes_signs_across_blocks_on_metal",
                "hadamard_inverse_keeps_wide_embedding_values_f32_until_final_store_on_metal",
            ]),
            ("linear::hadamard_tests", &[
                "hadamard_pq2_projection_keeps_f32_transform_through_dot_and_plain_siblings_on_metal",
            ]),
            ("linear::pq2_tests", &[
                "pq2_wide_gemv_preserves_f32_input_codes_blocks_offsets_and_output_tails",
            ]),
            ("linear::pq2_decode_tests", &[
                "pq2_float_floor_decoding_preserves_reference_tails_codes_and_f32_ranges",
            ]),
            ("linear::pq2_prefill_tests", &[
                "pq2_mixed_prefill_specialization_preserves_f32_operands_tiles_and_guards_on_metal",
                "pq2_mixed_prefill_m64_preserves_f32_operands_tiles_and_guards_on_metal",
                "pq2_mixed_prefill_m64_selection_requires_aligned_rows_wide_output_and_capability",
                "pq2_hadamard_prefill_selection_preserves_format_dtype_and_shape_boundaries",
                "pq2_mixed_prefill_production_dispatch_and_m32_fallback_preserve_output",
            ]),
            ("linear::narrow_dense_tests", &[
                "narrow_dense_gemv_preserves_arbitrary_k_output_tails_and_offsets",
                "narrow_dense_gemv_preserves_cancellation_subnormals_and_half_rounding",
                "narrow_dense_threadgroup_requires_exact_simd_and_device_capacity",
                "narrow_dense_production_dispatch_preserves_bounds_and_capability_fallback",
            ]),
            ("linear::native_tests", &[
                "native_block_linears_preserve_rows_offsets_strides_and_precision_on_real_metal",
                "pq2_0_linears_preserve_blocks_precision_and_production_dispatch_on_real_metal",
            ]),
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
                "pq2_0_token_embedding_preserves_codes_blocks_and_f32_range_on_real_metal",
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
            ("causal_attention::conformance_tests::batched_grouped", &[
                "batched_grouped_decode_matches_serial_bits_cpu_and_guards",
                "batched_grouped_decode_production_gate_accepts_n32_and_rejects_physical_aliases",
            ]),
            ("causal_attention::conformance_tests::int8", &[
                "int8_prepare_matches_reference_rounding_zero_and_subnormal_scales",
                "int8_prepare_crosses_payload_and_independent_scale_page_frontiers",
                "int8_attention_matches_cpu_for_gqa_mqa_and_complete_prefix_restore",
                "int8_prepare_marks_nonfinite_input_in_device_status",
                "int8_tiled_prefill_and_direct_decode_match_reference_across_payload_page_and_tail",
                "int8_tiled_dispatch_accounts_for_bounded_dequantization_memory",
                "int8_general_attention_accepts_a_partial_simd_head",
                "int8_optimized_attention_reads_across_an_independent_scale_page",
                "int8_gqa_prefill_reuses_kv_across_heads_and_matches_tiled_f16_and_cpu",
                "int8_gqa_prefill_reads_across_an_independent_scale_page",
            ]),
            ("causal_attention::conformance_tests::int8::packed", &[
                "packed_partial_head_f16_preserves_contiguous_rows_and_independent_histories",
                "packed_partial_head_int8_preserves_contiguous_rows_and_independent_histories",
                "packed_grouped_prepare_reordering_preserves_bits_kv_and_guards",
            ]),
            ("causal_attention::shape_tests", &[
                "int8_state_geometry_counts_independent_pages_and_omits_unused_split_decode_scratch",
            ]),
        ],
        Backend::Cuda => &[
            // Register CUDA-only boundaries and ignored device oracles explicitly;
            // paired timing microbenchmarks remain opt-in.
            ("last_token_linear_tests", &[
                "packed_last_token_rows_requires_unit_ranges_and_exact_matrix_coverage",
                "packed_last_token_rows_rejects_pointer_width_and_launch_overflow",
                "packed_f16_last_token_projection_matches_scalar_and_f64_on_cuda",
                "f16_last_token_projection_fallback_preserves_gaps_order_and_final_rows_on_cuda",
            ]),
            ("last_token_linear_tests::gather_tests", &[
                "last_token_gather_workspace_tracks_sequences_not_prefill_tokens",
                "last_token_gather_rejects_alias_without_rejecting_safe_fallback_layouts",
                "last_token_gather_does_not_hide_invalid_scalar_access",
                "last_token_gather_replay_binds_mode_copy_mapping_and_scratch_address",
                "gathered_f16_last_token_projection_matches_f64_and_preserves_fallback_on_cuda",
            ]),
            ("native_blocks::hadamard::tests", &[
                "hadamard_forward_and_inverse_match_independent_walsh_on_cuda",
                "hadamard_pq2_embedding_preserves_f32_until_inverse_on_cuda",
                "hadamard_pq2_projection_mixes_transformed_and_plain_inputs_on_cuda",
            ]),
            ("native_blocks::tests", &[
                "native_block_decoding_matches_shared_ggml_oracle_on_cuda",
                "native_block_embeddings_preserve_f16_and_f32_activations_on_cuda",
                "native_block_linears_preserve_f16_and_f32_activations_on_cuda",
                "native_matrix_launcher_preserves_mixed_dense_and_block_partitions",
            ]),
            ("native_blocks::tests::q4k", &[
                "q4k_specialization_preserves_generic_bits_and_f64_oracle_on_cuda",
            ]),
            ("native_blocks::tests::q4k_q8", &[
                "q4k_q8_prototype_pack_and_matmul_match_policy_oracle_on_cuda",
                "q56k_q8_prototype_pack_and_matmul_match_policy_oracle_on_cuda",
            ]),
            ("native_blocks::tests::prefill_gemm", &[
                "shared_prefill_gemm_preserves_f32_math_and_buffer_extents_on_cuda",
                "shared_prefill_gemm_production_dispatch_preserves_partition_offsets_on_cuda",
            ]),
            ("native_blocks::tests::shared_dispatch", &[
                "shared_gemm_midrow_production_dispatch_preserves_numeric_and_graph_boundaries_on_cuda",
            ]),
            ("native_io::tests", &[
                "token_lookup_launcher_preserves_exact_decode_offsets_and_invalid_ids_on_cuda",
                "final_row_projection_launcher_preserves_f32_input_and_guarded_logits_on_cuda",
                "token_io_bounds_preserve_nonzero_spans_and_native_launch_capacity",
            ]),
            ("native_io::tests::packed_projection", &[
                "native_projection_packing_requires_exact_physical_rows_and_nonaliasing",
                "native_projection_packing_preserves_capacity_and_transform_fallback",
                "packed_native_projection_matches_f64_and_scalar_with_guarded_fallback_on_cuda",
            ]),
            ("selection::tests", &["masked_selection_matches_scalar_semantics_and_preserves_logits_on_cuda"]),
            ("selection::tests::partitioned", &[
                "partitioned_argmax_dispatch_threshold_preserves_small_rows",
                "partitioned_argmax_matches_scalar_and_old_kernel_with_guarded_windows_on_cuda",
            ]),
            ("transformer::native_swiglu::tests", &[
                "native_swiglu_packed_rows_preserve_larger_participant_batches",
                "native_swiglu_mixed_matrices_match_stage_oracles_on_cuda",
                "native_swiglu_packed_rows_match_independent_source_slices_on_cuda",
                "native_swiglu_scratch_accounts_only_for_bounded_activations",
            ]),
            ("transformer::native_swiglu::tests::q8_tests", &[
                "native_q8_swiglu_stages_and_replay_preserve_f16_policy_on_cuda",
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
                "recurrent_cuda_mixed_chunk_boundaries_preserve_f32_state_and_slot_isolation",
                "recurrent_master_provider_preserves_hidden_precision_and_residual_aliasing_on_cuda",
            ]),
            // Quantized projection policies are checked at their actual F16
            // boundaries; the recurrent state remains F32. Fixed-QKV carry is
            // distinct from strict/Q8 equivalence or model-quality evidence.
            ("transformer::attention::recurrent_tests::q8_projection", &[
                "recurrent_q8_projections_match_policy_and_preserve_fixed_qkv_state_carry_on_cuda",
            ]),
            ("transformer::causal_attention::numerical_tests", &[
                "causal_kv_carry_crosses_physical_pages_and_matches_f64_attention_on_cuda",
                "causal_master_preserves_f32_hidden_and_both_residual_alias_modes_on_cuda",
            ]),
            ("transformer::causal_attention::rotary_tests", &[
                "standard_partial_rope_rotates_only_its_prefix_on_cuda",
                "proportional_partial_rope_retains_padded_head_pairs_on_cuda",
            ]),
            ("transformer::causal_attention::int8_tests", &[
                "int8_kv_geometry_counts_independent_scales_and_rejects_native",
                "int8_kv_prepare_and_attention_cross_independent_pages_on_cuda",
                "int8_kv_nonfinite_write_sets_step_status_without_a_device_trap",
                "int8_kv_packed_mha_gqa_mqa_preserve_independent_absolute_positions_on_cuda",
            ]),
        ],
    };
    let mut tests: Vec<_> = modules
        .iter()
        .flat_map(|(module, names)| {
            names.iter().map(move |name| {
                format!(
                    "backend::{}::vnext_ops::{module}::{name}",
                    super::backend_name(backend)
                )
            })
        })
        .collect();
    if backend == Backend::Cuda {
        // Runtime snapshot tests live outside vnext_ops. Select both range
        // boundaries and the ignored real-device parent/child fence oracle.
        tests.extend([
            "tests::staging_lease_covers_the_host_layout_including_destination_offset",
            "tests::readback_requires_whole_elements_and_bounded_source_and_destination",
            "gpu_tests::staged_u32_snapshot_reads_parent_while_same_lane_child_is_pending_on_cuda",
        ].into_iter().map(|name| {
            format!("backend::cuda::vnext_runtime::submission_readback::{name}")
        }));
    }
    tests
}

pub(super) fn groups(backend: Backend) -> Vec<ContractGroup> {
    // The same small execution verifies arithmetic and guarded state/output
    // boundaries. The harness deduplicates shared assertions across groups.
    let mut tests: Vec<_> = names(backend)
        .into_iter()
        .map(|name| ContractTest {
            package: "ferrum-kernels".into(),
            target: "ferrum_kernels".into(),
            kind: "lib".into(),
            name,
        })
        .collect();
    if backend == Backend::Cuda {
        tests.push(ContractTest {
            package: "ferrum-models".into(),
            target: "vnext_cuda_checkpoint_continuation".into(),
            kind: "test".into(),
            name: "gated_delta_q8_provider_charges_pack_and_replays_changed_inputs_with_isolated_state".into(),
        });
    }
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
