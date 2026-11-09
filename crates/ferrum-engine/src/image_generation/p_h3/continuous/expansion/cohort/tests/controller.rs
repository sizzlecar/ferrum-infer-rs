//! Real CPU causal installation through the product cohort interface. This
//! backend deliberately has no H3 F32 vision/DiT union: downstream rejection
//! is isolated only after the actual causal owner has retired.
use super::*;
use crate::image_generation::p_h3::{
    expansion::{ConfiguredTextExpansion, MergedExpansionFixture},
    PH3ImagePipelineOptions,
};
use crate::image_generation::{service::ImagePreparationResources, ImagePipelineConfig};
use ferrum_kernels::backend::cpu::{
    vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime,
};
use std::task::Poll;

type Pipeline = PH3ImagePipeline<CpuDeviceRuntime>;
fn pipeline() -> (Pipeline, MergedExpansionFixture) {
    let fixture = MergedExpansionFixture::new();
    let complete = crate::image_generation::p_h3::tests::fixture::CompleteFixture::new().unwrap();
    let mut assets = complete.assets().unwrap();
    let mut engine = ferrum_types::EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CPU;
    assets.expansion = Some(ConfiguredTextExpansion {
        assets: fixture.expansion.clone(),
        options: MergedExpansionFixture::native_options(engine),
    });
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("fixture.h3.product.cohort.cpu").unwrap(),
        16 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
        16,
    )
    .unwrap();
    let config = ImagePipelineConfig {
        device_capacity_bytes: 16 << 20,
        maximum_user_tokens: 512,
        maximum_width: 32,
        maximum_height: 32,
        maximum_batch_size: 2,
        weight_arena_target_bytes: None,
        decoder_batch_policy: Default::default(),
        edit_prefix_policy: Default::default(),
        plan_cache: Default::default(),
        residency: Default::default(),
        dynamic_residency: Default::default(),
        static_initialization: StaticInitializationPolicy::new(64 << 10, 8).unwrap(),
        executor: Default::default(),
    };
    let options = PH3ImagePipelineOptions::comfy_f32(
        NumericalProfileId::new("fixture.h3.product.noise").unwrap(),
        16,
    )
    .unwrap();
    (
        Pipeline::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            config,
            options,
        )
        .unwrap(),
        fixture,
    )
}
fn prepare(
    pipeline: &mut Pipeline,
    requests: &[ImageGenerationRequest],
) -> (ImageBatchPreparation, Vec<Uuid>) {
    let ids = (0..requests.len())
        .map(|_| Uuid::new_v4())
        .collect::<Vec<_>>();
    let members = requests
        .iter()
        .zip(&ids)
        .map(|(request, &request_id)| ImageInitializationCohortMember {
            request_id,
            preparation: pipeline
                .prepare_initialization(ImageRequestRef::Generate(request))
                .unwrap(),
            resources: ImagePreparationResources::default(),
        })
        .collect();
    let mut prepared = pipeline.prepare_initialization_cohort(members).unwrap();
    assert!(pipeline
        .try_admit_initialization_cohort_at_boundary(&mut prepared, None, &mut [])
        .unwrap());
    (prepared, ids)
}
fn views<'a>(
    requests: &'a [ImageGenerationRequest],
    ids: &[Uuid],
    cancelled: &[bool],
) -> Vec<ImageInitializationCohortRequest<'a>> {
    requests
        .iter()
        .zip(ids)
        .zip(cancelled)
        .map(
            |((request, &request_id), &cancelled)| ImageInitializationCohortRequest {
                request_id,
                request: ImageRequestRef::Generate(request),
                cancelled,
            },
        )
        .collect()
}
async fn advance(
    pipeline: &mut Pipeline,
    owner: &mut ImageRequestInitialization,
    requests: &[ImageInitializationCohortRequest<'_>],
) -> ImageInitializationCohortProgress {
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        std::future::poll_fn(|cx| {
            let result = pipeline
                .advance_initialization_cohort(owner, requests, cx.waker())
                .unwrap();
            if result.finished {
                Poll::Ready(result)
            } else {
                assert!(result.completed.is_empty());
                if result.made_progress {
                    cx.waker().wake_by_ref();
                }
                Poll::Pending
            }
        }),
    )
    .await
    .expect("actual native retirement must wake its original waiter")
}
#[tokio::test]
async fn product_cohort_cancelled_members_keep_original_ids_and_zero_one_many_native_survivors() {
    for cancelled in [[true, true], [true, false], [false, true], [false, false]] {
        let (mut pipeline, _fixture) = pipeline();
        let host = pipeline.host_account().clone();
        let runtime = pipeline.runtime().clone();
        let requests = [
            request(),
            ImageGenerationRequest {
                seed: 43,
                prompt: "dog".into(),
                ..request()
            },
        ];
        let (mut prepared, ids) = prepare(&mut pipeline, &requests);
        let views = views(&requests, &ids, &cancelled);
        let mut owner = pipeline
            .begin_initialization_cohort(&mut prepared, &views)
            .unwrap();
        if cancelled.iter().all(|c| *c) {
            assert_eq!(runtime.peak_resident_bytes(), 0);
        } else {
            assert!(runtime.resident_bytes() > 0);
        }
        let result = advance(&mut pipeline, &mut owner, &views).await;
        assert!(result.made_progress && result.finished);
        assert_eq!(result.completed.len(), 2);
        for (index, completion) in result.completed.iter().enumerate() {
            assert_eq!(completion.request_id, ids[index]);
            if cancelled[index] {
                assert!(matches!(
                    completion.outcome,
                    ImageInitializationCohortOutcome::Cancelled
                ));
            } else {
                assert!(matches!(
                    completion.outcome,
                    ImageInitializationCohortOutcome::Rejected { .. }
                        | ImageInitializationCohortOutcome::Pending(_)
                ));
            }
        }
        assert!(!pipeline.initialization_cohort_blocks_denoising(&owner));
        assert_eq!(runtime.resident_bytes(), 0);
        assert!(!pipeline.is_stopped());
        drop((result, owner, prepared));
        pipeline.shutdown().unwrap();
        drop(pipeline);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
#[tokio::test]
async fn product_cohort_live_identity_failure_returns_original_native_and_host_custody() {
    let (mut pipeline, _fixture) = pipeline();
    let host = pipeline.host_account().clone();
    let runtime = pipeline.runtime().clone();
    let requests = [
        request(),
        ImageGenerationRequest {
            seed: 43,
            ..request()
        },
    ];
    let (mut prepared, ids) = prepare(&mut pipeline, &requests);
    let views = views(&requests, &ids, &[false, false]);
    let mut owner = pipeline
        .begin_initialization_cohort(&mut prepared, &views)
        .unwrap();
    assert!(runtime.resident_bytes() > 0);
    let charged = host.usage().unwrap().committed_bytes();
    assert!(charged > 0);
    let mut changed = requests.clone();
    changed[1].seed += 1;
    let changed = views_for_changed(&changed, &ids);
    let error = pipeline
        .advance_initialization_cohort(&mut owner, &changed, &futures::task::noop_waker())
        .err()
        .expect("changed ingress after installation must retain original owner");
    assert!(matches!(error, ImageBatchError::Fatal(_)));
    assert!(pipeline.is_stopped());
    drop((owner, prepared));
    assert!(
        runtime.resident_bytes() > 0,
        "returned error owns original live installation"
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), charged);
    let ImageBatchError::Fatal(original) = &error else {
        unreachable!("fatal custody checked above")
    };
    let cleanup = original
        .downcast_ref::<CohortFailure<CpuDeviceRuntime>>()
        .expect("original typed cohort failure")
        .cleanup()
        .expect("installed native cohort has its original close observer");

    // Hold a real Host scope on that same retirement channel. Its destructor
    // starts only after native close, so device zero cannot certify Host refund.
    struct RetainedHost {
        entered: Option<tokio::sync::oneshot::Sender<()>>,
        release: std::sync::mpsc::Receiver<()>,
        _scope: ferrum_runtime::HostMemoryScope,
    }
    impl Drop for RetainedHost {
        fn drop(&mut self) {
            let _ = self.entered.take().unwrap().send(());
            let _ = self.release.recv();
        }
    }
    let (entered, retiring) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel::<()>();
    cleanup.retain_until_retired(RetainedHost {
        entered: Some(entered),
        release: released,
        _scope: host
            .reserve_scope(&[(HostMemoryClass::Output, 16)])
            .unwrap(),
    });
    drop(error);
    let retired = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        retiring.await.expect("original retention release started");
        assert_eq!(runtime.resident_bytes(), 0);
        assert_ne!(cleanup.snapshot().phase(), ExecutorShutdownPhase::Retired);
        assert!(host.usage().unwrap().committed_bytes() > 0);
        // Disconnect also releases the destructor if this test unwinds.
        drop(release);
        cleanup.wait().await
    })
    .await
    .expect("original native and Host cleanup reaches its retirement boundary");
    assert_eq!(retired.phase(), ExecutorShutdownPhase::Retired);
    retired.result().unwrap();
    assert_eq!(runtime.resident_bytes(), 0);
    pipeline.shutdown().unwrap();
    drop(pipeline);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
fn views_for_changed<'a>(
    requests: &'a [ImageGenerationRequest],
    ids: &[Uuid],
) -> Vec<ImageInitializationCohortRequest<'a>> {
    views(requests, ids, &[false, false])
}

#[tokio::test]
async fn expansion_kernel_timing_pipeline_flag_reaches_actual_b2_summary() {
    use crate::image_generation::p_h3::PH3ImagePromptExpansionDiagnosticPolicy;
    use ferrum_runtime::DeviceTimingMode;

    for mode in [DeviceTimingMode::Off, DeviceTimingMode::Kernel] {
        let (pipeline, _fixture) = pipeline();
        let mut pipeline = pipeline.with_device_timing_mode(mode).unwrap();
        let host = pipeline.host_account().clone();
        let runtime = pipeline.runtime().clone();
        let trace = pipeline
            .enable_continuous_trace(
                ImageContinuousTraceLimits {
                    maximum_bytes: 512 << 10,
                },
                &serde_json::json!({"device_timing":mode}),
            )
            .unwrap();
        pipeline
            .set_prompt_expansion_diagnostics(PH3ImagePromptExpansionDiagnosticPolicy::Summary)
            .unwrap();
        let requests = [
            request(),
            ImageGenerationRequest {
                prompt: "dog".into(),
                ..request()
            },
        ];
        let (mut prepared, ids) = prepare(&mut pipeline, &requests);
        let views = views(&requests, &ids, &[false, false]);
        let mut owner = pipeline
            .begin_initialization_cohort(&mut prepared, &views)
            .unwrap();
        let result = advance(&mut pipeline, &mut owner, &views).await;
        assert!(result.finished);
        assert_eq!(runtime.resident_bytes(), 0);
        let records = trace_records(&trace);
        let observed = records
            .iter()
            .map(|record| &record["event"])
            .filter(|event| event["kind"] == "prompt_expansion_observed")
            .collect::<Vec<_>>();
        assert_eq!(observed.len(), 2);
        assert_eq!(
            observed[0]["shared_owner"]["id"],
            observed[1]["shared_owner"]["id"]
        );
        for event in observed {
            assert_eq!(event["outcome"], "expanded");
            assert_eq!(event["token_history"]["protocol_eos_observed"], true);
            let timing = &event["executor_metrics"]["wave_timing_by_phase"]["decode"];
            let samples = timing["host_encode_submit_breakdown"]["provider_encode_submit"]
                ["samples"]
                .as_u64()
                .unwrap();
            assert_eq!(samples > 0, mode == DeviceTimingMode::Kernel);
        }
        assert!(!pipeline.is_stopped());
        drop((result, owner, prepared));
        pipeline.shutdown().unwrap();
        drop((pipeline, trace));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
