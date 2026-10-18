use super::*;
use ferrum_execution::test_support::{assert_executor_closed, startup as assertions};

#[tokio::test]
async fn native_startup_cpu_drop_after_prerequisite_closes_exact_incarnation() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::native_startup_cpu_drop_after_prerequisite_closes_exact_incarnation(executor)
            .await;
    assert_eq!(runtime.resident_bytes(), 0);
    drop(_directory);
}

#[tokio::test]
async fn native_startup_cpu_exact_chunks_decode_and_startup_completion() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let runtime =
        assertions::native_startup_cpu_exact_chunks_decode_and_startup_completion(executor).await;
    assert_eq!(runtime.resident_bytes(), 0);
    drop(_directory);
}

#[tokio::test]
async fn native_startup_cpu_unsubmitted_prefill_cannot_enter_capture() {
    let Fixture {
        _directory,
        executor,
    } = cpu(SamplingReadbackMode::Uniform);
    let usable = assertions::startup_minimum_capacity(&executor);
    let runtime = assert_executor_closed(executor);
    assert_eq!(runtime.resident_bytes(), 0);
    drop(_directory);
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.fixture.startup-capacity").unwrap(),
        256 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let Fixture {
        _directory,
        executor,
    } = Fixture::new(
        SamplingReadbackMode::Uniform,
        Device::CPU,
        runtime,
        registry,
        materializers,
        catalog,
        FixtureOptions {
            usable_capacity_bytes: Some(usize::try_from(usable).unwrap()),
            ..FixtureOptions::default()
        },
    );
    let runtime =
        assertions::native_startup_cpu_unsubmitted_prefill_cannot_enter_capture(executor, usable)
            .await;
    assert_eq!(runtime.resident_bytes(), 0);
    drop(_directory);
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires exclusive CUDA and actual native capture/replay"]
async fn native_startup_cuda_rejects_segments_without_program_binding() {
    use ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition;
    let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
        0,
        DeviceId::new("device.fixture.startup-reuse").unwrap(),
        AttentionExecutionPolicy::Portable,
    )
    .unwrap()
    .into_parts();
    let fixture = Fixture::new(
        SamplingReadbackMode::Uniform,
        Device::CUDA(0),
        runtime,
        registry,
        materializers,
        catalog,
        FixtureOptions {
            reusable_startup: true,
            ..FixtureOptions::default()
        },
    );
    let Fixture {
        _directory,
        executor,
    } = fixture;
    let _runtime =
        assertions::native_startup_rejects_segments_without_program_binding(executor).await;
    drop(_directory);
}
