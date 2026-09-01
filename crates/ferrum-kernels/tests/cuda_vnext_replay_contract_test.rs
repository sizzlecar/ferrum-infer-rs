const CAUSAL_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/causal_attention.rs");
const RECURRENT_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/attention.rs");
const TRANSFORMER_SOURCE: &str = include_str!("../src/backend/cuda/vnext_ops/transformer.rs");
const VNEXT_OPS_SOURCE: &str = include_str!("../src/backend/cuda/vnext_ops.rs");
const RUNTIME_SOURCE: &str = include_str!("../src/backend/cuda/vnext_runtime.rs");
const REPLAY_SOURCE: &str = include_str!("../src/backend/cuda/vnext_replay.rs");
const LINEAR_ATTENTION_KERNEL_SOURCE: &str = include_str!("../kernels/linear_attention.cu");
const GATED_DELTA_KERNEL_SOURCE: &str = include_str!("../kernels/gated_delta_rule.cu");
const ARGMAX_KERNEL_SOURCE: &str = include_str!("../kernels/argmax_rows.cu");
const FUSED_SILU_MUL_KERNEL_SOURCE: &str = include_str!("../kernels/fused_silu_mul.cu");
const MOE_PROVIDER_SOURCE: &str = include_str!("../src/backend/cuda/vnext_ops/transformer/moe.rs");
const MOE_ROUTER_KERNEL_SOURCE: &str = include_str!("../kernels/moe_router.cu");
const BUILD_SCRIPT_SOURCE: &str = include_str!("../build.rs");

#[test]
fn gemma_simple_ops_are_registered_replayable_and_fail_closed() {
    for capability in [
        "DENSE_GEGLU_TANH_F16_CAPABILITY_ID",
        "CONSTANT_SCALE_F16_CAPABILITY_ID",
        "LOGIT_SOFTCAP_F16_CAPABILITY_ID",
    ] {
        assert!(VNEXT_OPS_SOURCE.contains(capability));
    }
    for contract in [
        "dense_geglu_tanh_contract()",
        "constant_scale_contract()",
        "logit_softcap_contract()",
    ] {
        assert!(VNEXT_OPS_SOURCE.matches(contract).count() >= 1);
    }
    for provider in [
        "CudaDenseGeGluTanhProvider::new(runtime)",
        "CudaConstantScaleProvider::new(runtime)",
        "CudaLogitSoftcapProvider::new(runtime)",
    ] {
        assert!(VNEXT_OPS_SOURCE.contains(provider));
    }

    assert!(FUSED_SILU_MUL_KERNEL_SOURCE.contains("fused_gelu_tanh_mul_f16("));
    assert!(FUSED_SILU_MUL_KERNEL_SOURCE.contains("const __half* __restrict__ gate"));
    assert!(FUSED_SILU_MUL_KERNEL_SOURCE.contains("const __half* __restrict__ up"));
    assert!(FUSED_SILU_MUL_KERNEL_SOURCE.contains("logit_softcap_inplace_f16("));
    assert!(FUSED_SILU_MUL_KERNEL_SOURCE.contains("cap * tanhf(value / cap)"));
    assert!(
        TRANSFORMER_SOURCE.contains("dense GeGLU input {ordinal} has no physical weight layout")
    );

    let scale = TRANSFORMER_SOURCE
        .split("fn encode_constant_scale(")
        .nth(1)
        .expect("constant-scale encoder")
        .split("fn encode_logit_softcap(")
        .next()
        .expect("bounded constant-scale encoder");
    assert!(scale.contains("same_physical_region(&input, &output)"));
    assert!(scale.contains(".f32(scale)"));
    assert!(scale.contains("replayable_operation("));

    let softcap = TRANSFORMER_SOURCE
        .split("fn encode_logit_softcap(")
        .nth(1)
        .expect("logit-softcap encoder")
        .split("fn encode_residual_add(")
        .next()
        .expect("bounded logit-softcap encoder");
    assert!(softcap.contains("same_physical_region(&input_region, &output_region)"));
    assert!(softcap.contains(".f32(cap)"));
    assert!(softcap.contains("for region in regions"));
    assert!(softcap.contains("replayable_operation("));

    let geglu = TRANSFORMER_SOURCE
        .split("fn encode_dense_geglu_tanh(")
        .nth(1)
        .expect("dense GeGLU encoder")
        .split("fn encode_constant_scale(")
        .next()
        .expect("bounded dense GeGLU encoder");
    assert!(geglu.contains("dense_geglu_projection(&invocation)?"));
    assert!(geglu.contains("launch_planar_gelu_tanh_mul("));
    assert!(geglu.contains("replayable_operation_with_blas("));
    assert!(!geglu.contains("fused_gelu_tanh_mul_interleaved_f16"));
}

#[test]
fn vnext_masked_argmax_preserves_logits_and_binds_typed_scratch() {
    let provider = VNEXT_OPS_SOURCE
        .split("pub struct CudaLastTokenMaskedArgmaxProvider")
        .nth(1)
        .expect("CUDA vNext must install a masked-argmax provider")
        .split("struct LastTokenDenseLinearLaunch")
        .next()
        .expect("masked-argmax provider source must have a bounded body");
    assert!(VNEXT_OPS_SOURCE.contains("last_token_masked_argmax_preserving_logits_f16"));
    assert!(provider.contains("MASKED_ARGMAX_PRESERVING_LOGITS_FUNCTION_NAME"));
    assert!(provider.contains("transformer::shared_scratch_region("));
    assert!(provider.contains("scratch_offset_bytes"));
    assert!(provider.contains("ProviderWorkspaceSizeFormula::actual_sequences(scratch_bytes)"));
    assert!(!provider.contains("apply_repetition_penalties_sparse_f16"));

    assert!(ARGMAX_KERNEL_SOURCE.contains("const __half* __restrict__ logits"));
    assert!(ARGMAX_KERNEL_SOURCE.contains("__half* __restrict__ scratch"));
    assert!(ARGMAX_KERNEL_SOURCE.contains("selection_logits = scratch"));
    assert!(TRANSFORMER_SOURCE.contains("provider_fingerprint.as_bytes()"));
}

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
    assert!(
        resolver.contains("lock_sha256={}")
            && resolver.contains("operator_binaries={}")
            && resolver.contains("sha256_file_digest(&resolved_set.lock_path)"),
        "native artifact-set build receipt must bind lock and binary content",
    );

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
fn native_catalog_packaging_input_cannot_bypass_product_validation() {
    let composition = VNEXT_OPS_SOURCE
        .split("impl CudaVNextComposition {")
        .nth(1)
        .expect("CUDA composition impl must exist");
    let create = composition
        .split("pub fn create(")
        .nth(1)
        .expect("CUDA composition must expose product create")
        .split("\n    fn ")
        .next()
        .expect("product create must have a bounded body");
    assert!(create.contains("family: &PreparedModelFamily"));
    assert!(create.contains("Some(family)"));
    assert!(create.contains("composition.validate_compiled_native_operators()?"));
    assert!(!composition.contains("pub fn create_for_family("));

    let validated_catalog_input = VNEXT_OPS_SOURCE
        .split("pub fn cuda_validated_native_operator_catalog_input(")
        .nth(1)
        .expect("CUDA validated catalog input must exist")
        .split("/// Capture the exact provider identities needed to package")
        .next()
        .expect("CUDA validated catalog input must have a bounded body");
    assert!(validated_catalog_input.contains("composition.validate_compiled_native_operators()?"));

    let packaging_input = VNEXT_OPS_SOURCE
        .split("pub fn cuda_native_operator_catalog_input(")
        .nth(1)
        .expect("CUDA packaging catalog input must exist")
        .split("\npub struct CudaTokenEmbeddingProvider")
        .next()
        .expect("CUDA packaging catalog input must have a bounded body");
    assert!(packaging_input.contains("CudaVNextComposition::prepare("));
    assert!(!packaging_input.contains("validate_compiled_native_operators"));

    let packaging_value = VNEXT_OPS_SOURCE
        .split("pub struct CudaNativeOperatorCatalogInput {")
        .nth(1)
        .expect("CUDA packaging catalog input value must exist")
        .split("\n}")
        .next()
        .expect("CUDA packaging catalog input value must have a bounded body");
    for executable_owner in [
        "CudaDeviceRuntime",
        "OperationRuntimeRegistry",
        "WeightMaterializerRegistry",
    ] {
        assert!(
            !packaging_value.contains(executable_owner),
            "packaging catalog input retained executable owner {executable_owner}"
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
fn replay_capture_attributes_commands_from_one_bulk_graph_topology_query() {
    assert!(REPLAY_SOURCE.contains("fn capture_dependency_frontier("));
    assert!(REPLAY_SOURCE.contains("fn captured_graph_topology("));
    assert!(REPLAY_SOURCE.contains("fn command_graph_node_counts_from_topology<"));
    assert!(!REPLAY_SOURCE.contains("capture_graph_node_count"));
    assert_eq!(
        REPLAY_SOURCE.matches("sys::cuGraphGetNodes(").count(),
        2,
        "captured graph nodes must be queried only by the post-capture bulk topology read",
    );
    assert_eq!(
        REPLAY_SOURCE.matches("sys::cuGraphGetEdges_v2(").count(),
        2,
        "captured graph edges must be queried only by the post-capture bulk topology read",
    );
    assert!(REPLAY_SOURCE.contains("command_frontiers = None"));
    assert!(REPLAY_SOURCE.contains("topological_order.iter().rev()"));
    assert!(REPLAY_SOURCE.contains("reachable_command_intervals"));
    assert!(!REPLAY_SOURCE.contains("let mut closure = vec![false; nodes.len()]"));
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
    assert!(CAUSAL_ATTENTION_SOURCE.contains("partition_stable &= topology.is_partition_stable();"));
    assert!(
        CAUSAL_ATTENTION_SOURCE.contains("return Ok(ReusableExecutionTopology::EagerBoundary);")
    );
    assert!(!CAUSAL_ATTENTION_SOURCE.contains(".map(ReusableExecutionTopology::Dynamic)"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains(".u64(replay_envelope.sequence_capacity_tokens)"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains(".i32(replay_envelope.table_capacity_entries)"));
    assert!(!CAUSAL_ATTENTION_SOURCE.contains(".u64(launch.sequence_tokens)"));
}

#[test]
fn token_replay_identity_uses_typed_coordinate_ownership() {
    let topology = VNEXT_OPS_SOURCE
        .split("fn reusable_token_topology(")
        .nth(1)
        .expect("CUDA token providers must share one reusable topology helper")
        .split("\nfn ")
        .next()
        .expect("CUDA token topology helper must have a bounded body");
    assert!(topology.contains("binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)"));
    assert!(topology.contains("if bind_source_ranges"));
    assert!(VNEXT_OPS_SOURCE.contains("token-embedding.reusable-topology.v2"));
    assert!(VNEXT_OPS_SOURCE.contains("last-token-linear.reusable-topology.v2"));
    assert!(!VNEXT_OPS_SOURCE.contains("reusable-topology.v1"));
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
fn typed_program_binding_patches_form_one_layout_owned_sparse_prelude() {
    assert!(CAUSAL_ATTENTION_SOURCE.contains("CudaDeviceCommand::program_binding_patch("));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("CudaDeviceCommand::program_binding_patch("));
    assert!(RUNTIME_SOURCE.contains("struct CudaProgramBindingPatch"));
    assert!(RUNTIME_SOURCE
        .contains("\"CUDA typed program bindings do not cover one compiled layout exactly\""));
    assert!(RUNTIME_SOURCE.contains("coalesce_program_binding_transfers("));
    assert!(RUNTIME_SOURCE.contains("cuMemcpy2DAsync_v2("));
    assert!(RUNTIME_SOURCE.contains("let transfer_command_count = u64::try_from(transfers.len())"));
    assert!(RUNTIME_SOURCE.contains("executable: None"));
    assert!(!RUNTIME_SOURCE.contains("let mut host_patch = vec![0_u8; patch_bytes]"));
    assert!(!RUNTIME_SOURCE.contains("\"aggregate program binding upload\""));
    assert!(RUNTIME_SOURCE.contains("fence_dependencies.extend(patch.fence_dependencies)"));
}

#[test]
fn direct_attention_bindings_do_not_rebuild_compute_commands() {
    assert!(RECURRENT_ATTENTION_SOURCE.contains("fn encode_reusable_execution_bindings("));
    assert!(RECURRENT_ATTENTION_SOURCE.contains("encode_reusable_attention_bindings(invocation)"));
    assert!(CAUSAL_ATTENTION_SOURCE.contains("fn encode_reusable_execution_bindings("));
    assert!(CAUSAL_ATTENTION_SOURCE.contains(
        "encode_reusable_attention_bindings(\n            invocation,\n            self.semantics,\n            self.descriptor.operation_id().as_str(),\n        )"
    ));

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
        .contains("let use_packed = participant_count_usize > 1 && input_packed && output_packed"));
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
