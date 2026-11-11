//! Actual native capacity with real Stored DiT, Euler and decoder execution.
//! Only the original cohort's auxiliary handoff uses the parent fixture.
use super::*;
use crate::image_generation::{ImageStage, QwenImage21T2iLayout, QwenImage21TransformerConfig};
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceVNextComposition,
};
use ferrum_runtime::{DeviceId, WeightMaterializerRegistry};
use std::sync::atomic::AtomicU64;

pub(super) fn bounded_pipeline(
    original: QwenImage21Pipeline<Runtime>,
) -> (
    QwenImage21Pipeline<Runtime>,
    native_pressure::NativePressure,
) {
    let needs = [vec![1], vec![2], vec![1, 2]]
        .into_iter()
        .map(|lengths| {
            let batch = lengths.len();
            let config = QwenImage21TransformerConfig::new(
                original.execution.sources.assets.transformer.clone(),
                QwenImage21T2iLayout::new(2, 2, 2, lengths).unwrap(),
            )
            .unwrap();
            let stage = original.prepare_denoiser(&config, batch).unwrap();
            original
                .capacity_with_prefix(&[&stage], None)
                .unwrap()
                .required_bytes
        })
        .collect::<Vec<_>>();
    let minimum = needs[0].max(needs[1]);
    let available = ((minimum + needs[2]) / 2 / 256) * 256;
    assert!(minimum <= available && available < needs[2]);
    static NEXT_DEVICE: AtomicU64 = AtomicU64::new(0);
    let instance = NEXT_DEVICE.fetch_add(1, Ordering::Relaxed);
    let composition = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new(format!("device.service.ready_capacity.{instance}")).unwrap(),
        original.execution.config.device_capacity_bytes,
    )
    .unwrap();
    let QwenImage21Pipeline {
        execution,
        profiles,
        ..
    } = original;
    let crate::image_generation::ImagePipelineExecution {
        host,
        mut config,
        sources,
        ..
    } = execution;
    config.maximum_batch_size = 3;
    let assets = Arc::try_unwrap(sources.assets)
        .unwrap_or_else(|_| panic!("fresh fixture owns its immutable assets"));
    let pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        assets,
        profiles,
        config,
    )
    .unwrap();
    let pressure = native_pressure::NativePressure::new(
        &pipeline,
        pipeline.execution.config.device_capacity_bytes - available,
    );
    assert!(pipeline.execution.runtime.snapshot().live_allocations > 0);
    (pipeline, pressure)
}

#[derive(Clone, Copy)]
enum Exit {
    Complete,
    CancelReady,
    CancelInitialization,
    CancelRunning,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ready_capacity_waiter_preserves_fifo_initialization_and_cancellation_retirement() {
    for exit in [
        Exit::Complete,
        Exit::CancelReady,
        Exit::CancelInitialization,
        Exit::CancelRunning,
    ] {
        let (service, boundary, trace, runtime, control) = start_cohort(Mode::ReadyCapacity);
        let _release = Release(control.clone());
        let _steps = ReleaseOnDrop(boundary.clone());
        let first = service.try_submit(request(1)).unwrap();
        control.wait(|s| s.metadata_entered).await;
        let ready = service.try_submit(request(2)).unwrap();
        let later = service.try_submit(request(3)).unwrap();
        control.metadata();
        control.wait(|s| s.wake.is_some()).await;
        control.finish();
        boundary.wait(1).await;
        boundary.release(1);
        boundary.wait(2).await;
        {
            let state = control.state.lock().unwrap();
            assert_eq!(state.initialized, [(1, 0), (2, 1)]);
        }
        assert_eq!(service.snapshot().active_requests, 1);
        assert_eq!(service.snapshot().preparing_requests, 2);
        match exit {
            Exit::Complete => (),
            Exit::CancelReady => ready.cancel(),
            Exit::CancelInitialization => later.cancel(),
            Exit::CancelRunning => first.cancel(),
        }
        boundary.release(2);
        let results = tokio::time::timeout(Duration::from_secs(15), async {
            [first.wait().await, ready.wait().await, later.wait().await]
        })
        .await
        .expect("admitted work and FIFO waiters must reach their terminal owners");
        let cancelled_seed = match exit {
            Exit::Complete => None,
            Exit::CancelReady => Some(2),
            Exit::CancelInitialization => Some(3),
            Exit::CancelRunning => Some(1),
        };
        for (seed, result) in (1..=3).zip(results) {
            if Some(seed) == cancelled_seed {
                assert_eq!(
                    result.err().unwrap().kind(),
                    ImageServiceErrorKind::Cancelled
                );
            } else {
                let output = result.unwrap();
                assert_eq!(output.receipt.seed, seed);
                assert_eq!(output.receipt.steps.len(), 8);
                assert!(output
                    .receipt
                    .stages
                    .iter()
                    .any(|stage| stage.execution.plan.stage == ImageStage::Decoder));
                assert_eq!(output.rgba.bytes().len(), 32 * 32 * 4 * 2);
            }
        }
        {
            let state = control.state.lock().unwrap();
            match exit {
                Exit::Complete => assert!(state.initialized.contains(&(3, 8))),
                Exit::CancelReady | Exit::CancelRunning => {
                    assert!(state.initialized.contains(&(3, 2)))
                }
                Exit::CancelInitialization => {
                    assert_eq!(state.cancelled_initializations, [(3, 2)]);
                    assert!(state.initialized.iter().all(|(seed, _)| *seed != 3));
                }
            }
        }
        assert!(trace
            .lock()
            .unwrap()
            .iter()
            .take(if matches!(exit, Exit::CancelRunning) {
                2
            } else {
                8
            })
            .all(|wave| wave.len() == 1 && wave[0].0 == 1));
        if matches!(exit, Exit::Complete) {
            assert!(trace.lock().unwrap().iter().any(|wave| {
                wave.iter().map(|(seed, _, _)| *seed).collect::<Vec<_>>() == [2, 3]
            }));
        }
        finish(&service, &runtime).await;
    }
}
