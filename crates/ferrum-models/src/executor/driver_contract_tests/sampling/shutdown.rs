use super::*;
use ferrum_execution::test_support::shutdown as assertions;

#[tokio::test]
async fn native_shutdown_cpu_retires_original_sequence_and_plan() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::program_shutdown_retires_original_cpu_sequence_and_plan(executor).await;
    assert_eq!(
        runtime.resident_bytes(),
        0,
        "all original physical owners released"
    );
    drop(_directory);
}

#[tokio::test]
async fn native_shutdown_cpu_joins_cancelled_submission_after_waiter_drop() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::program_shutdown_joins_cancelled_cpu_submission_after_waiter_drop(executor)
            .await;
    assert_eq!(
        runtime.resident_bytes(),
        0,
        "all original physical owners released"
    );
    drop(_directory);
}

#[tokio::test]
async fn native_shutdown_cpu_retains_referenced_plan_until_original_retry() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::program_shutdown_retains_referenced_cpu_plan_until_original_retry(executor)
            .await;
    assert_eq!(
        runtime.resident_bytes(),
        0,
        "same retained plan actually closed"
    );
    drop(_directory);
}

#[cfg(feature = "metal")]
#[tokio::test]
#[ignore = "requires a real Metal device and language providers"]
async fn native_shutdown_metal_joins_cancelled_submission_after_waiter_drop() {
    let Fixture {
        _directory,
        executor,
    } = metal(SamplingReadbackMode::Uniform);
    // The shared assertion checks the complete F32 row exactly. This fixture's
    // F16 source values and identity head are exactly representable, so the
    // Metal route has the same numerical contract as the CPU reference.
    let runtime =
        assertions::program_shutdown_joins_cancelled_cpu_submission_after_waiter_drop(executor)
            .await;
    let original = Arc::downgrade(&runtime);
    drop(runtime);
    assert!(
        original.upgrade().is_none(),
        "original runtime survives neither its retired plan nor its joined worker"
    );
    drop(_directory);
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires exclusive CUDA and real F32 language providers"]
async fn native_shutdown_cuda_joins_cancelled_submission_after_waiter_drop() {
    let Fixture {
        _directory,
        executor,
    } = cuda(SamplingReadbackMode::Uniform, None, ElementType::F32);
    // Reuse the admitted production composition and the original worker gate;
    // no synthetic fence, close receipt, or relaxed logits comparison is used.
    let runtime =
        assertions::program_shutdown_joins_cancelled_cpu_submission_after_waiter_drop(executor)
            .await;
    let original = Arc::downgrade(&runtime);
    drop(runtime);
    assert!(
        original.upgrade().is_none(),
        "original runtime survives neither its retired plan nor its joined worker"
    );
    drop(_directory);
}
