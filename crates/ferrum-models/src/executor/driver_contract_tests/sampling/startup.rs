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

#[test]
fn native_startup_cpu_late_configuration_error_precedes_static_allocation() {
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.fixture.failed-cold-activation").unwrap(),
        256 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(runtime.peak_resident_bytes(), 0);
    let directory = tempfile::tempdir().unwrap();
    let result = Fixture::try_new_with_engine_config(
        SamplingReadbackMode::Uniform,
        Device::CPU,
        Arc::clone(&runtime),
        registry,
        materializers,
        catalog,
        FixtureOptions::default(),
        |engine| {
            engine.runtime.diagnostic_output_capture =
                Some(ferrum_types::RetainedOutputCaptureConfig {
                    output_dir: directory.path().join("capture"),
                    prompt_token_count: 1,
                    prompt_token_sha256: "a".repeat(64),
                    source_span: 1..2,
                    consumed_history_sha256: "b".repeat(64),
                    // This concrete cold activation constructor used to run only
                    // after static weight allocation and successful initialization.
                    maximum_readback_bytes: 0,
                });
        },
    );
    let error = match result {
        Ok(fixture) => {
            assert_executor_closed(fixture.executor);
            panic!("invalid original capture config must reject construction");
        }
        Err(error) => error,
    };
    assert!(
        error
            .cause()
            .to_string()
            .contains("output capture requires"),
        "{error}"
    );
    assert!(error.cleanup().is_none(), "no plan ownership was acquired");
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(
        runtime.peak_resident_bytes(),
        0,
        "releasing after allocation is not prevalidation"
    );
    assert!(!directory.path().join("capture").exists());
}
