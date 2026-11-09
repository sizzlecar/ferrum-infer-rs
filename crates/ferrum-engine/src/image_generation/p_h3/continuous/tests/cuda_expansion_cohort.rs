//! Genuine tiny CUDA expansion + image stages through the cohort adapter.
//! These validate ownership, cancellation, and deterministic fixture outputs;
//! they do not qualify official weights, image quality, throughput, or SLOs.
//! Run alone with one test thread, the pinned CUDA native set, and explicit
//! FERRUM_TEST_CUDNN_LIBRARY_PATH (plus its declared dependency libraries).
use super::super::*;
use super::cuda_suspension::{finish, guard, input, pipeline};
use crate::image_generation::{
    continuous_trace::{ImageContinuousTraceHandle, ImageContinuousTraceLimits},
    p_h3::{
        expansion::MergedExpansionFixture, tests::fixture::CompleteFixture,
        PH3ImagePromptExpansionDiagnosticPolicy,
    },
    service::{
        ImageInitializationCohortCompletion, ImageInitializationCohortGenerator,
        ImageInitializationCohortMember, ImageInitializationCohortOutcome,
        ImageInitializationCohortRequest, ImagePreparationResources, ImageRequestInitialization,
        ImageRequestRef,
    },
    ImageGenerationOutput, ImageGenerationRequest,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use serde_json::Value;
use std::{future::poll_fn, task::Poll, time::Duration};
use uuid::Uuid;

type Pipeline = PH3ImagePipeline<CudaDeviceRuntime>;

fn requests() -> [ImageGenerationRequest; 2] {
    let first = input(61);
    let mut second = input(73);
    second.prompt.push_str(" cat");
    [first, second]
}

fn independent_pixels(
    fixture: &CompleteFixture,
    expansion: &MergedExpansionFixture,
    requests: &[ImageGenerationRequest],
) -> Vec<Vec<u8>> {
    let mut oracle = pipeline(fixture, expansion);
    let pixels = requests
        .iter()
        .map(|request| {
            let output = oracle
                .generate_continuous_batch(vec![request.clone()])
                .unwrap_or_else(|error| panic!("independent CUDA B1: {error}"));
            assert_eq!(output.len(), 1);
            assert_eq!(output[0].receipt.seed, request.seed);
            output[0].rgba.bytes().bytes().to_vec()
        })
        .collect();
    finish(oracle);
    pixels
}

/// Drive the real independent continuation after shared retirement, including
/// its ordinary DiT work and decoder. No new prompt expansion is performed.
fn finish_downstream(
    pipeline: &mut Pipeline,
    initialization: ImageRequestInitialization,
    request: &ImageGenerationRequest,
) -> ImageGenerationOutput {
    assert!(!pipeline.initialization_has_auxiliary_owner(&initialization));
    let mut lane = pipeline
        .continue_initialization_blocking(initialization, ImageRequestRef::Generate(request))
        .unwrap_or_else(|error| panic!("independent downstream initialization: {error}"))
        .into_inner::<ContinuousRequest<CudaDeviceRuntime>>()
        .unwrap();
    let mut batch = pipeline
        .try_prepare_continuous_batch(&mut [&mut lane])
        .unwrap_or_else(|error| panic!("real downstream batch: {error}"))
        .expect("downstream native admission");
    while !lane.is_complete() {
        batch
            .advance(pipeline, &mut [&mut lane])
            .unwrap_or_else(|error| panic!("real downstream forward: {error}"));
    }
    batch
        .pause(pipeline, &mut [&mut lane])
        .unwrap_or_else(|error| panic!("real downstream fence: {error}"));
    let output = pipeline
        .finish_continuous_request_borrowed(&mut lane, &mut batch)
        .unwrap_or_else(|error| panic!("real downstream decoder: {error}"));
    assert_eq!(output.receipt.seed, request.seed);
    drop((lane, batch));
    output
}

async fn retire(
    pipeline: &mut Pipeline,
    owner: &mut ImageRequestInitialization,
    inputs: &[ImageInitializationCohortRequest<'_>],
) -> Vec<ImageInitializationCohortCompletion> {
    tokio::time::timeout(
        Duration::from_secs(60),
        poll_fn(|cx| {
            let progress = pipeline
                .advance_initialization_cohort(owner, inputs, cx.waker())
                .unwrap_or_else(|error| panic!("real shared expansion advance: {error}"));
            if progress.finished {
                assert!(!pipeline.initialization_cohort_blocks_denoising(owner));
                Poll::Ready(progress.completed)
            } else {
                assert!(
                    progress.completed.is_empty(),
                    "no downstream ownership may escape before shared retirement"
                );
                assert!(pipeline.initialization_cohort_blocks_denoising(owner));
                if progress.made_progress {
                    cx.waker().wake_by_ref();
                }
                Poll::Pending
            }
        }),
    )
    .await
    .expect("original CUDA retirement must wake the cohort waiter")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires actual CUDA tensor+causal union, pinned native operators and cuDNN; tiny fixture ownership and cancellation only"]
async fn cuda_h3_expansion_cohort_retires_before_independent_handoff_and_preserves_survivor() {
    let _guard = guard();
    let fixture = CompleteFixture::new().unwrap();
    let expansion = MergedExpansionFixture::new();
    let requests = requests();
    let expected = independent_pixels(&fixture, &expansion, &requests);
    for cancel_first in [false, true] {
        let mut pipeline = pipeline(&fixture, &expansion);
        let host = pipeline.host_account().clone();
        // Original caller-held Output credit is separate from the auxiliary
        // grant. Derive its physical extent from this fixture's real F32 RGB.
        let bytes = host
            .backing_bytes(3 * 32 * 32 * std::mem::size_of::<f32>())
            .unwrap();
        let credits = requests
            .iter()
            .map(|_| {
                host.reserve_scope(&[(HostMemoryClass::Output, bytes)])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let retained_credit = host.usage().unwrap().committed_bytes();
        let ids = [Uuid::new_v4(), Uuid::new_v4()];
        assert_ne!(ids[0], ids[1]);
        let mut preparations = requests
            .iter()
            .map(|request| {
                pipeline
                    .prepare_initialization(ImageRequestRef::Generate(request))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let (first, rest) = preparations.split_first_mut().unwrap();
        assert!(
            pipeline
                .maximum_initialization_cohort_members(first)
                .unwrap()
                >= requests.len()
        );
        assert!(pipeline
            .initialization_cohort_compatible(first, &mut rest[0])
            .unwrap());
        let members = preparations
            .into_iter()
            .zip(ids)
            .map(
                |(preparation, request_id)| ImageInitializationCohortMember {
                    request_id,
                    preparation,
                    resources: ImagePreparationResources {
                        output_credit_bytes: bytes,
                    },
                },
            )
            .collect();
        let mut prepared = pipeline.prepare_initialization_cohort(members).unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), retained_credit);
        assert!(pipeline
            .try_admit_initialization_cohort_at_boundary(&mut prepared, None, &mut [],)
            .unwrap());
        assert!(host.usage().unwrap().committed_bytes() > retained_credit);
        let mut inputs = requests
            .iter()
            .zip(ids)
            .map(|(request, request_id)| ImageInitializationCohortRequest {
                request_id,
                request: ImageRequestRef::Generate(request),
                cancelled: false,
            })
            .collect::<Vec<_>>();
        let mut owner = pipeline
            .begin_initialization_cohort(&mut prepared, &inputs)
            .unwrap();
        assert!(pipeline.initialization_cohort_blocks_denoising(&owner));
        let progress = pipeline
            .advance_initialization_cohort(&mut owner, &inputs, &futures::task::noop_waker())
            .unwrap();
        assert!(!progress.finished && progress.completed.is_empty());
        assert!(host.usage().unwrap().committed_bytes() > retained_credit);
        inputs[0].cancelled = cancel_first;
        let completed = retire(&mut pipeline, &mut owner, &inputs).await;
        assert_eq!(
            completed
                .iter()
                .map(|result| result.request_id)
                .collect::<Vec<_>>(),
            ids
        );
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            retained_credit,
            "actual original auxiliary grant retires while caller Output credit remains owned"
        );
        drop((owner, prepared));
        for (index, completion) in completed.into_iter().enumerate() {
            if cancel_first && index == 0 {
                assert!(matches!(
                    completion.outcome,
                    ImageInitializationCohortOutcome::Cancelled
                ));
                continue;
            }
            let ImageInitializationCohortOutcome::Pending(initialization) = completion.outcome
            else {
                panic!("live survivor must transfer its independent downstream preparation");
            };
            let output = finish_downstream(&mut pipeline, initialization, &requests[index]);
            assert_eq!(
                output.rgba.bytes().bytes(),
                expected[index].as_slice(),
                "shared expansion/cancellation changed this independent CUDA B1 image"
            );
        }
        drop(credits);
        finish(pipeline);
    }
}

fn events(trace: &ImageContinuousTraceHandle) -> Vec<Value> {
    let mut bytes = Vec::new();
    let status = trace.write_jsonl(&mut bytes).unwrap();
    assert!(!status.truncated && status.dropped == 0);
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .filter_map(|mut record| record.as_object_mut()?.remove("event"))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires actual CUDA tensor+causal union, pinned native operators and cuDNN; tiny B2 run/core agreement, no official-model performance claim"]
async fn cuda_h3_synchronous_batch_uses_shared_expansion_owner_and_real_multi_request_waves() {
    let _guard = guard();
    let fixture = CompleteFixture::new().unwrap();
    let expansion = MergedExpansionFixture::new();
    let requests = requests();
    let expected = independent_pixels(&fixture, &expansion, &requests);
    let mut pipeline = pipeline(&fixture, &expansion);
    let trace = pipeline
        .enable_continuous_trace(
            ImageContinuousTraceLimits {
                maximum_bytes: 1 << 20,
            },
            &serde_json::json!({"entrypoint": "tiny-cuda-run-expansion-cohort"}),
        )
        .unwrap();
    pipeline
        .set_prompt_expansion_diagnostics(PH3ImagePromptExpansionDiagnosticPolicy::Summary)
        .unwrap();
    let outputs = pipeline
        .generate_continuous_batch(requests.to_vec())
        .unwrap_or_else(|error| panic!("real CUDA shared run batch: {error}"));
    assert_eq!(outputs.len(), requests.len());
    for (index, output) in outputs.iter().enumerate() {
        assert_eq!(output.receipt.seed, requests[index].seed);
        assert_eq!(
            output.rgba.bytes().bytes(),
            expected[index].as_slice(),
            "tiny CUDA B2 image differs from its independent B1"
        );
    }
    let events = events(&trace);
    let observations = events
        .iter()
        .filter(|event| event["kind"] == "prompt_expansion_observed")
        .collect::<Vec<_>>();
    assert_eq!(observations.len(), requests.len());
    let owner = &observations[0]["shared_owner"];
    assert!(
        !owner.is_null(),
        "run must use the actual shared native installation"
    );
    assert_ne!(observations[0]["job"], observations[1]["job"]);
    for observation in &observations {
        assert_eq!(&observation["shared_owner"], owner);
        assert_eq!(observation["executor_retired"], true);
        assert_eq!(observation["outcome"], "expanded");
    }
    assert!(
        events.iter().any(|event| {
            event["kind"] == "prompt_expansion_token_wave"
                && &event["shared_owner"] == owner
                && event["participants"].as_u64() == Some(requests.len() as u64)
        }),
        "real native waves must contain both original requests"
    );
    drop((outputs, trace));
    finish(pipeline);
}
