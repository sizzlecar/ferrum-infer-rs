const CAUSAL_ATTENTION_SOURCE: &str =
    include_str!("../src/backend/cuda/vnext_ops/transformer/causal_attention.rs");
const RUNTIME_SOURCE: &str = include_str!("../src/backend/cuda/vnext_runtime.rs");
const REPLAY_SOURCE: &str = include_str!("../src/backend/cuda/vnext_replay.rs");

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
    assert!(
        CAUSAL_ATTENTION_SOURCE.contains(".u64(launch.replay_envelope.sequence_capacity_tokens)")
    );
    assert!(CAUSAL_ATTENTION_SOURCE.contains(".i32(launch.replay_envelope.table_capacity_entries)"));
    assert!(!CAUSAL_ATTENTION_SOURCE.contains(".u64(launch.sequence_tokens)"));
}
