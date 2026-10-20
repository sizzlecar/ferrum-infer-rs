use super::*;
use ferrum_execution::test_support::lifecycle as assertions;

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_full_logits_retires_and_completes_real_cohort() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::sampling_readback_cpu_lifecycle_full_logits_retires_and_completes_real_cohort(
            executor,
        )
        .await;
    assert_eq!(runtime.resident_bytes(), 0, "all physical owners released");
    drop(_directory);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_failed_retirement_returns_exact_step_owner() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::sampling_readback_cpu_lifecycle_failed_retirement_returns_exact_step_owner(
            executor,
        )
        .await;
    assert_eq!(runtime.resident_bytes(), 0, "all physical owners released");
    drop(_directory);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_cancel_waiting_for_worker_capacity() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::sampling_readback_cpu_lifecycle_cancel_waiting_for_worker_capacity(executor)
            .await;
    assert_eq!(runtime.resident_bytes(), 0, "all physical owners released");
    drop(_directory);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_cancel_queued_shared_cohort() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::sampling_readback_cpu_lifecycle_cancel_queued_shared_cohort(executor).await;
    assert_eq!(runtime.resident_bytes(), 0, "all physical owners released");
    drop(_directory);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_cancel_delivered_unpolled_result() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::sampling_readback_cpu_lifecycle_cancel_delivered_unpolled_result(executor)
            .await;
    assert_eq!(runtime.resident_bytes(), 0, "all physical owners released");
    drop(_directory);
}
