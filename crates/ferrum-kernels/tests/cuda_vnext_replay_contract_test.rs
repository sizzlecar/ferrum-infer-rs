const CAUSAL_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/causal_attention.rs");
const RECURRENT_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/attention.rs");
const TRANSFORMER_SOURCE: &str = include_str!("../src/backend/cuda/vnext_ops/transformer.rs");
const RUNTIME_SOURCE: &str = include_str!("../src/backend/cuda/vnext_runtime.rs");
const REPLAY_SOURCE: &str = include_str!("../src/backend/cuda/vnext_replay.rs");
const LINEAR_ATTENTION_KERNEL_SOURCE: &str = include_str!("../kernels/linear_attention.cu");
const GATED_DELTA_KERNEL_SOURCE: &str = include_str!("../kernels/gated_delta_rule.cu");
const MOE_PROVIDER_SOURCE: &str = include_str!("../src/backend/cuda/vnext_ops/transformer/moe.rs");
const MOE_ROUTER_KERNEL_SOURCE: &str = include_str!("../kernels/moe_router.cu");
const BUILD_SCRIPT_SOURCE: &str = include_str!("../build.rs");

#[test]
fn product_build_script_has_no_native_source_compiler_surface() {
    for forbidden_source_surface in [
        "fn compile_marlin(",
        "fn compile_vllm_marlin(",
        "fn compile_vllm_moe_marlin(",
        "fn compile_vllm_paged_attn(",
        "historical_nvcc_scheduler_signatures",
        "static_lib_cache_state",
    ] {
        assert!(
            !BUILD_SCRIPT_SOURCE.contains(forbidden_source_surface),
            "product build script retained native source surface {forbidden_source_surface}"
        );
    }
}

#[test]
fn product_cuda_build_is_artifact_only_and_fails_closed_without_a_lock() {
    let resolver = BUILD_SCRIPT_SOURCE
        .split("fn link_native_operator_artifact_set()")
        .nth(1)
        .expect("build script must define the native artifact resolver")
        .split("\nfn ")
        .next()
        .expect("native artifact resolver must have a bounded body");
    assert!(resolver.contains("if !required_build_units.is_empty()"));
    assert!(resolver.contains("are artifact-only"));
    assert!(resolver.contains("ferrum-native-ops-builder"));

    let main = BUILD_SCRIPT_SOURCE
        .split("fn main()")
        .nth(1)
        .expect("build script must define main")
        .split("\nfn detect_cuda_compute_cap(")
        .next()
        .expect("build script main must have a bounded body");
    assert!(main.contains("emit_native_artifact_build_unit(unit)"));
    for forbidden_source_build in [
        "compile_marlin(&",
        "compile_vllm_marlin(&",
        "compile_vllm_moe_marlin(&",
        "compile_vllm_paged_attn(&",
    ] {
        assert!(
            !main.contains(forbidden_source_build),
            "product build main must not call {forbidden_source_build}"
        );
    }
}

#[test]
fn causal_pages_are_fence_dependencies_not_captured_regions() {
    let retained = [
        "compute_fence_dependencies",
        ".extend(pages.iter().cloned())",
    ]
    .concat();
    let captured = ["compute_regions", ".extend(pages.iter().cloned())"].concat();

    assert!(CAUSAL_ATTENTION_SOURCE.contains(&retained));
    assert!(!CAUSAL_ATTENTION_SOURCE.contains(&captured));
    assert!(
        CAUSAL_ATTENTION_SOURCE.contains("replayable_operation_with_blas_and_fence_dependencies")
    );
    assert!(CAUSAL_ATTENTION_SOURCE
        .contains("None => CudaDeviceCommand::operation_with_blas_and_fence_dependencies"));
}

#[test]
fn executable_cache_has_no_fence_dependency_owner() {
    assert!(RUNTIME_SOURCE.contains("pub(crate) struct CudaCommandExecutable"));
    assert!(RUNTIME_SOURCE.contains("fence_dependencies: Vec<CudaBufferRegion>"));
    assert!(REPLAY_SOURCE.contains("_executables: Vec<Arc<CudaCommandExecutable>>"));
    assert!(!REPLAY_SOURCE.contains("fence_dependencies"));
}

#[test]
fn replay_identity_does_not_enable_full_profile_tool_correlation() {
    assert!(REPLAY_SOURCE.contains(
        "let profile_identity = timing_mode.physical_span_attribution_enabled().then(||"
    ));
    assert!(REPLAY_SOURCE.contains("if timing_mode.kernel_attribution_enabled()"));
    assert!(REPLAY_SOURCE
        .contains("profile_identity.map(|identity| Arc::clone(&identity.fingerprint))"));
    assert!(!REPLAY_SOURCE.contains("profile_identity.map_or_else("));
    assert!(!REPLAY_SOURCE.contains("retain_profile_identity: bool"));
    assert!(!REPLAY_SOURCE.contains("tool_correlation: bool"));
    assert!(RUNTIME_SOURCE.contains(
        "if kernel_attribution {\n            vnext_tool_correlation::prepare();\n        }"
    ));
}

#[test]
fn causal_replay_identity_uses_a_partition_capacity_envelope() {
    assert!(CAUSAL_ATTENTION_SOURCE.contains("CausalAttentionReplayTopology"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("PartitionStableDecode"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("ExactShapeEager"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("is_partition_stable"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains(".u64(replay_envelope.sequence_capacity_tokens)"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains(".i32(replay_envelope.table_capacity_entries)"));
    assert!(!CAUSAL_ATTENTION_SOURCE.contains(".u64(launch.sequence_tokens)"));
}

#[test]
fn dynamic_attention_addresses_use_one_hoistable_program_binding_boundary() {
    assert!(CAUSAL_ATTENTION_SOURCE.contains("attach_invocation_binding("));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("attach_invocation_binding("));
    assert!(TRANSFORMER_SOURCE.contains("has_compiled_program_slot"));
    assert!(TRANSFORMER_SOURCE.contains("operation.with_program_binding(binding_command)"));
    assert!(TRANSFORMER_SOURCE.contains("operation.with_dynamic_binding(binding_command)"));
    assert!(RUNTIME_SOURCE.contains("vnext_program_binding_prelude"));
    assert!(RUNTIME_SOURCE.contains("coalesced_program_bindings"));
}

#[test]
fn typed_program_binding_patches_form_one_layout_owned_upload() {
    assert!(CAUSAL_ATTENTION_SOURCE.contains("CudaDeviceCommand::program_binding_patch("));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("CudaDeviceCommand::program_binding_patch("));
    assert!(RUNTIME_SOURCE.contains("struct CudaProgramBindingPatch"));
    assert!(RUNTIME_SOURCE
        .contains("\"CUDA typed program bindings do not cover one compiled layout exactly\""));
    assert!(RUNTIME_SOURCE.contains("let mut host_patch = vec![0_u8; patch_bytes]"));
    assert!(RUNTIME_SOURCE.contains("\"aggregate program binding upload\""));
    assert!(RUNTIME_SOURCE.contains("transfer_command_count: 1"));
    assert!(RUNTIME_SOURCE.contains("fence_dependencies.extend(patch.fence_dependencies)"));
}

#[test]
fn direct_attention_bindings_do_not_rebuild_compute_commands() {
    assert!(RECURRENT_ATTENTION_SOURCE.contains("fn encode_reusable_execution_bindings("));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("encode_reusable_attention_bindings(invocation)"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("fn encode_reusable_execution_bindings("));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("encode_reusable_attention_bindings(invocation)"));

    for source in [RECURRENT_ATTENTION_SOURCE, CAUSAL_ATTENTION_SOURCE] {
        let binding_only = source
            .split("fn encode_reusable_attention_bindings(")
            .nth(1)
            .expect("CUDA attention provider must define a binding-only encoder")
            .split("\nfn ")
            .next()
            .expect("binding-only encoder must have a bounded body");
        assert!(binding_only.contains(
            "EncodedReusableExecutionBindings::empty().with_program_binding(binding_command)"
        ));
        assert!(!binding_only.contains("CudaCommandReplayKeyBuilder"));
        assert!(!binding_only.contains("replayable_operation"));
        assert!(!binding_only.contains("encode_attention("));
    }
}

#[test]
fn recurrent_state_is_indirect_and_fence_retained_not_captured() {
    assert!(RECURRENT_ATTENTION_SOURCE.contains("compute_fence_dependencies.push(conv_state"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("compute_fence_dependencies.push(delta_state"));
    assert!(RECURRENT_ATTENTION_SOURCE
        .contains("replayable_operation_with_host_storage_blas_and_fence_dependencies"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("state_binding_offset"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("launch.conv_state_region"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("launch.delta_state_region"));
    assert!(LINEAR_ATTENTION_KERNEL_SOURCE
        .contains("linear_attention_prepare_varlen_f16_params_f32_state_f16_indirect"));
    assert!(LINEAR_ATTENTION_KERNEL_SOURCE.contains("recurrent_conv_state_commit_f16_indirect"));
    assert!(LINEAR_ATTENTION_KERNEL_SOURCE.contains("state_bindings[seq * 2]"));
    assert!(GATED_DELTA_KERNEL_SOURCE.contains("recurrent_gated_delta_rule_varlen_f32_indirect"));
    assert!(GATED_DELTA_KERNEL_SOURCE
        .contains("recurrent_gated_delta_rule_varlen_tiled16_f32_indirect"));
    assert!(GATED_DELTA_KERNEL_SOURCE.contains("state_bindings[seq * 2 + 1]"));
}

#[test]
fn recurrent_attention_packs_wave_projections_and_keeps_exact_qkvzba() {
    assert!(RECURRENT_ATTENTION_SOURCE.contains("contiguous_bindings(10)"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("shared.qkvzba"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains(
        "linear_attention_prepare_varlen_packed_qkvzba_f16_params_f32_state_f16_z_f16_indirect"
    ));
    assert!(RECURRENT_ATTENTION_SOURCE
        .contains("let use_packed = participant_count_usize > 1 && input_shared && output_shared"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("super::shared_token_region("));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("DeviceBatchingForm::Packed"));
    assert!(
        RECURRENT_ATTENTION_SOURCE.contains("token_sequence_indices(&participant_token_counts)")
    );
    assert!(LINEAR_ATTENTION_KERNEL_SOURCE.contains(
        "linear_attention_prepare_varlen_packed_qkvzba_f16_params_f32_state_f16_z_f16_indirect"
    ));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.qkvz,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.ba,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.qkv,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.z,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.a,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.b,"));
}

#[test]
fn causal_attention_packs_shared_wave_projections_and_residual() {
    let token_offset = CAUSAL_ATTENTION_SOURCE
        .split("fn token_offset(")
        .nth(1)
        .expect("CUDA causal attention must define packed token offsets")
        .split("\n    }")
        .next()
        .expect("packed token offset helper must have a bounded body");

    assert!(CAUSAL_ATTENTION_SOURCE.contains("struct PackedCausalAttentionLaunch"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("fn enqueue_packed_attention("));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("super::shared_token_region("));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("DeviceBatchingForm::Packed"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("fn physical_dispatch_count("));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("validate_packed_token_ranges("));
    assert!(CAUSAL_ATTENTION_SOURCE.contains(".boolean(packed_enabled)"));
    assert!(token_offset.contains("width"));
    assert!(token_offset.contains(".checked_mul(ElementType::F16.size_bytes())"));
    assert!(!token_offset.contains("aligned_bytes("));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("\"packed causal attention Q GEMM\""));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("\"packed causal attention output GEMM\""));
}

#[test]
fn single_token_moe_router_materializes_marlin_blocks_without_generic_align() {
    assert!(MOE_PROVIDER_SOURCE.contains("MoeRoutingPlan::SingleTokenDirectMarlin"));
    assert!(MOE_PROVIDER_SOURCE.contains("launch_single_token_router"));
    assert!(MOE_PROVIDER_SOURCE.contains("MoeRoutingPlan::GenericAlign"));
    assert!(MOE_PROVIDER_SOURCE.contains("kernels.launch_align("));
    assert!(MOE_ROUTER_KERNEL_SOURCE.contains("moe_router_topk_softmax_f16_single_token_marlin"));
    assert!(MOE_ROUTER_KERNEL_SOURCE.contains("other == expert && k < tid"));
    assert!(MOE_ROUTER_KERNEL_SOURCE.contains("expert_block_ids[expert_rank] = expert"));
    assert!(MOE_ROUTER_KERNEL_SOURCE.contains("sorted_token_ids[block_start] = tid"));
    assert!(MOE_ROUTER_KERNEL_SOURCE.contains("total_tokens_post_pad[0] = padded_pair_count"));
}
