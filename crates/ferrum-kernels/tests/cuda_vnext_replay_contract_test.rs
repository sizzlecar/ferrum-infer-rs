const CAUSAL_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/causal_attention.rs");
const RECURRENT_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/attention.rs");
const RUNTIME_SOURCE: &str = include_str!("../src/backend/cuda/vnext_runtime.rs");
const REPLAY_SOURCE: &str = include_str!("../src/backend/cuda/vnext_replay.rs");
const LINEAR_ATTENTION_KERNEL_SOURCE: &str = include_str!("../kernels/linear_attention.cu");
const GATED_DELTA_KERNEL_SOURCE: &str = include_str!("../kernels/gated_delta_rule.cu");
const MOE_PROVIDER_SOURCE: &str = include_str!("../src/backend/cuda/vnext_ops/transformer/moe.rs");
const MOE_ROUTER_KERNEL_SOURCE: &str = include_str!("../kernels/moe_router.cu");

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
    assert!(CAUSAL_ATTENTION_SOURCE.contains(".with_program_binding(binding_command)"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains(".with_program_binding(binding_command)"));
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
    assert!(GATED_DELTA_KERNEL_SOURCE.contains("recurrent_gated_delta_rule_varlen_f32_indirect"));
    assert!(GATED_DELTA_KERNEL_SOURCE
        .contains("recurrent_gated_delta_rule_varlen_tiled16_f32_indirect"));
}

#[test]
fn recurrent_attention_uses_two_packed_input_projections() {
    assert!(RECURRENT_ATTENTION_SOURCE.contains("contiguous_bindings(11)"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("shared.qkvz"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("shared.ba"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains(
        "linear_attention_prepare_varlen_packed_qkvz_ba_f16_params_f32_state_f16_z_f16_indirect"
    ));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("launch.topology.compute_dispatches()"));
    assert!(LINEAR_ATTENTION_KERNEL_SOURCE.contains(
        "linear_attention_prepare_varlen_packed_qkvz_ba_f16_params_f32_state_f16_z_f16_indirect"
    ));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.qkv,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.z,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.a,"));
    assert!(!RECURRENT_ATTENTION_SOURCE.contains("shared.b,"));
}

#[test]
fn recurrent_single_token_decode_has_a_typed_packed_topology() {
    assert!(RECURRENT_ATTENTION_SOURCE.contains("enum CudaRecurrentAttentionTopology"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("SingleTokenPackedDecode"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("VarlenRecurrentScan"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("tokens == 1"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("&& shape.tiled_delta"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains(".bytes(launch.topology.as_str().as_bytes())"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("launch.topology.transfer_commands()"));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("launch_decode_conv_pack("));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("launch_decode_packed_delta("));
    assert!(LINEAR_ATTENTION_KERNEL_SOURCE.contains(
        "linear_attention_decode_prepare_packed_qkvz_to_mixed_f16_to_f32_state_f16_z_f16_indirect"
    ));
    assert!(LINEAR_ATTENTION_KERNEL_SOURCE
        .contains("conv_state_slots = reinterpret_cast<StateT*>(state_bindings[0])"));
    assert!(GATED_DELTA_KERNEL_SOURCE
        .contains("recurrent_gated_delta_rule_decode_packed_f32_ba_f16_params_f32_indirect"));
    assert!(GATED_DELTA_KERNEL_SOURCE
        .contains("state_slots = reinterpret_cast<StateT*>(state_bindings[1])"));

    let decode_branch = RECURRENT_ATTENTION_SOURCE
        .split("CudaRecurrentAttentionTopology::SingleTokenPackedDecode => {")
        .nth(1)
        .expect("CUDA recurrent provider must define a single-token packed branch")
        .split("CudaRecurrentAttentionTopology::VarlenRecurrentScan => {")
        .next()
        .expect("single-token packed branch must precede the varlen branch");
    assert!(!decode_branch.contains("launch_conv_state_commit("));
    assert!(!decode_branch.contains("launch_qk_norm("));

    let varlen_branch = RECURRENT_ATTENTION_SOURCE
        .split("CudaRecurrentAttentionTopology::VarlenRecurrentScan => {")
        .nth(1)
        .expect("CUDA recurrent provider must retain the varlen branch")
        .split("\n        }")
        .next()
        .expect("varlen branch must have a bounded body");
    assert!(varlen_branch.contains("launch_prepare("));
    assert!(varlen_branch.contains("launch_conv_state_commit("));
    assert!(varlen_branch.contains("launch_qk_norm("));
    assert!(varlen_branch.contains("launch_delta("));
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
