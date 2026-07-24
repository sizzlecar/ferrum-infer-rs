const CAUSAL_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/causal_attention.rs");
const RECURRENT_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/attention.rs");
const RUNTIME_SOURCE: &str = include_str!("../src/backend/cuda/vnext_runtime.rs");
const REPLAY_SOURCE: &str = include_str!("../src/backend/cuda/vnext_replay.rs");
const LINEAR_ATTENTION_KERNEL_SOURCE: &str = include_str!("../kernels/linear_attention.cu");
const GATED_DELTA_KERNEL_SOURCE: &str = include_str!("../kernels/gated_delta_rule.cu");

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
