//! The collector's result boundary observes the existing close service using
//! an actual compiled CPU program and its original physical allocations.
use super::*;
use ferrum_execution::{ExecutorShutdownObserver, ExecutorShutdownPhase, ModelExecutor};
use ferrum_kernels::backend::cpu::{
    vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime,
};
use ferrum_models::test_support::native_program::{Fixture, FixtureOptions};
use ferrum_runtime::DeviceId;
use ferrum_types::{Device, SamplingReadbackMode};
use std::sync::Arc;
use std::time::Duration;

fn fixture() -> (Fixture<CpuDeviceRuntime>, Arc<CpuDeviceRuntime>) {
    let (runtime, operations, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.cpu.collector-retirement").unwrap(),
        64 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let original = Arc::clone(&runtime);
    let fixture = Fixture::new(
        SamplingReadbackMode::Uniform,
        Device::CPU,
        runtime,
        operations,
        materializers,
        catalog,
        FixtureOptions::default(),
    );
    assert!(
        original.resident_bytes() > 0,
        "actual static weights are allocated"
    );
    (fixture, original)
}

async fn retired(observer: &ExecutorShutdownObserver) {
    let mut observer = observer.clone();
    tokio::time::timeout(Duration::from_secs(5), async {
        while observer.snapshot().phase() != ExecutorShutdownPhase::Retired {
            observer.changed().await;
        }
    })
    .await
    .expect("original CPU program must retire");
    observer.snapshot().result().unwrap();
}

#[tokio::test]
async fn collector_failure_keeps_original_cause_and_checked_program_cleanup() {
    let (fixture, runtime) = fixture();
    fixture.executor.prepare_startup().await.unwrap();
    let original = ExecutorOwner::new(Box::new(fixture.executor));
    let observation = original.shutdown_observer();
    let cause = FerrumError::backend("original catalog validation failed");
    let error = finish_owned_collection::<()>(original, Err(cause.clone()))
        .await
        .unwrap_err();
    assert_eq!(error.cause().to_string(), cause.to_string());
    let cleanup = error.cleanup().expect("same original program close");
    assert_eq!(cleanup.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(
        observation.snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    cleanup.snapshot().result().unwrap();
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn collector_success_waits_for_original_peer_before_reporting_released() {
    let (fixture, runtime) = fixture();
    fixture.executor.prepare_startup().await.unwrap();
    let original = ExecutorOwner::new(Box::new(fixture.executor));
    let peer = original.scope().unwrap();
    let observation = original.shutdown_observer();
    let mut completion = Box::pin(finish_owned_collection(original, Ok(17_u32)));
    assert!(matches!(
        futures::poll!(completion.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(
        observation.snapshot().phase(),
        ExecutorShutdownPhase::Draining
    );
    assert!(runtime.resident_bytes() > 0);
    drop(peer);
    let value = tokio::time::timeout(Duration::from_secs(5), completion)
        .await
        .expect("last original peer permits retirement")
        .unwrap();
    assert_eq!(value, 17);
    assert_eq!(
        observation.snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn cancelled_collector_close_wait_preserves_original_program_service() {
    let (fixture, runtime) = fixture();
    fixture.executor.prepare_startup().await.unwrap();
    let original = ExecutorOwner::new(Box::new(fixture.executor));
    let peer = original.scope().unwrap();
    let observation = original.shutdown_observer();
    let mut completion = Box::pin(finish_owned_collection::<()>(
        original,
        Err(FerrumError::backend(
            "original collector preparation failed",
        )),
    ));
    assert!(matches!(
        futures::poll!(completion.as_mut()),
        std::task::Poll::Pending
    ));
    drop(completion);
    assert_eq!(
        observation.snapshot().phase(),
        ExecutorShutdownPhase::Draining
    );
    assert!(runtime.resident_bytes() > 0);
    drop(peer);
    retired(&observation).await;
    assert_eq!(runtime.resident_bytes(), 0);
}
