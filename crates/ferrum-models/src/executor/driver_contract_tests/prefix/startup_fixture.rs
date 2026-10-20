//! Reusable startup with the real causal-attention provider's program binding.
//! The shared fixture retains its original CUDA Int8-KV weights and policies.
use crate::test_support::stateful_program;
use ferrum_execution::test_support::{
    assert_executor_closed, startup::assert_startup_capture_replay,
};

#[tokio::test]
#[ignore = "requires actual CUDA device and native attention kernels"]
async fn native_startup_cuda_attention_capture_replay_preserves_exact_descriptors() {
    let stateful_program::Fixture {
        directory,
        executor,
        ..
    } = stateful_program::cuda_startup();
    assert_startup_capture_replay(&executor).await;
    let _runtime = assert_executor_closed(executor);
    drop(directory);
}
