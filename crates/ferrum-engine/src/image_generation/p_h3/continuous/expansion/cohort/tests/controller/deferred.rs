use super::*;
use crate::image_generation::p_h3::PH3ImageMainVisionReuse;

#[test]
fn downstream_reference_preparation_preserves_backend_validation_without_an_actor() {
    let (mut pipeline, _fixture) = pipeline();
    let original_error = pipeline
        .compile_reference_encoder(32, 32)
        .err()
        .expect("CPU backend has no H3 reference encoder")
        .to_string();
    let fork = pipeline
        .execution
        .preparation_fork(pipeline.execution.config.device_capacity_bytes);
    let original = std::mem::replace(&mut pipeline.execution, fork);
    // A metadata fork must still be unavailable for device execution.
    assert!(pipeline.is_stopped());
    let prepared_error = pipeline
        .compile_reference_encoder(32, 32)
        .err()
        .expect("metadata preparation must retain backend validation")
        .to_string();
    assert_eq!(prepared_error, original_error);
    pipeline.execution.stopped = true;
    let stopped_error = pipeline
        .compile_reference_encoder(32, 32)
        .err()
        .expect("an explicitly stopped fork cannot prepare")
        .to_string();
    assert_ne!(stopped_error, original_error);
    pipeline.execution = original;
    pipeline.shutdown().unwrap();
}

#[tokio::test]
async fn retired_cohort_defers_recipes_and_keeps_cancellation_request_local() {
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
    let original_jobs = {
        let cohort = prepared
            .get_mut::<PreparedCohort<CpuDeviceRuntime>>()
            .unwrap();
        for (index, member) in cohort.members.iter_mut().enumerate() {
            member.correlation = Some(ferrum_types::ImageRequestCorrelation {
                request_id: ids[index],
                client_attempt_id: Some(Uuid::new_v4()),
            });
            member.output_credit = 64 + index;
        }
        cohort
            .members
            .iter()
            .map(|member| (member.job, member.correlation))
            .collect::<Vec<_>>()
    };
    let inputs = views(&requests, &ids, &[false, false]);
    let mut owner = pipeline
        .begin_initialization_cohort(&mut prepared, &inputs)
        .unwrap();
    let result = advance(&mut pipeline, &mut owner, &inputs).await;
    assert_eq!(runtime.resident_bytes(), 0);
    assert!(!pipeline.initialization_cohort_blocks_denoising(&owner));
    let mut pending = result
        .completed
        .into_iter()
        .enumerate()
        .map(|(index, completion)| {
            assert_eq!(completion.request_id, ids[index]);
            let ImageInitializationCohortOutcome::Pending(initialization) = completion.outcome
            else {
                panic!("retirement must not compile the unsupported downstream H3 recipes");
            };
            let state = initialization
                .get::<Initialization<CpuDeviceRuntime>>()
                .unwrap();
            assert!(matches!(
                state.state,
                InitializationState::PrepareDownstream { .. }
            ));
            assert!(state
                .identity
                .matches(ImageRequestRef::Generate(&requests[index])));
            assert!(Arc::ptr_eq(
                &state.pipeline,
                &pipeline.execution.service_preparation_identity
            ));
            assert_eq!((state.job, state.correlation), original_jobs[index]);
            assert_eq!(state.output_credit, 64 + index);
            assert!(!pipeline.initialization_has_auxiliary_owner(&initialization));
            initialization
        })
        .collect::<Vec<_>>();
    let charged = host.usage().unwrap().committed_bytes();
    // Cancel before downstream compilation: this CPU backend would reject a
    // recipe if cancellation incorrectly fell through to preparation.
    assert!(matches!(
        pipeline
            .advance_initialization(
                &mut pending[0],
                ImageRequestRef::Generate(&requests[0]),
                true,
                &futures::task::noop_waker(),
            )
            .unwrap(),
        ImageInitializationProgress::Cancelled
    ));
    assert!(matches!(
        pending[1]
            .get::<Initialization<CpuDeviceRuntime>>()
            .unwrap()
            .state,
        InitializationState::PrepareDownstream { .. }
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), charged);
    assert_eq!(runtime.resident_bytes(), 0);
    let terminal = futures::future::poll_fn(|cx| {
        match pipeline
            .advance_initialization(
                &mut pending[1],
                ImageRequestRef::Generate(&requests[1]),
                false,
                cx.waker(),
            )
            .unwrap()
        {
            ImageInitializationProgress::Pending { made_progress } => {
                if made_progress {
                    cx.waker().wake_by_ref();
                }
                std::task::Poll::Pending
            }
            terminal => std::task::Poll::Ready(terminal),
        }
    })
    .await;
    assert!(matches!(
        terminal,
        ImageInitializationProgress::Rejected { .. }
    ));
    assert!(!pipeline.is_stopped());
    assert_eq!(runtime.resident_bytes(), 0);
    drop((pending, owner, prepared));
    pipeline.shutdown().unwrap();
    drop(pipeline);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_cohort_drives_the_same_deferred_rejection_without_parking() {
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
    let inputs = requests
        .iter()
        .map(ImageRequestRef::Generate)
        .collect::<Vec<_>>();
    let preparations = inputs
        .iter()
        .map(|&input| pipeline.prepare_initialization(input).unwrap())
        .collect();
    let initializations = pipeline
        .prepare_run_initializations(preparations, &inputs)
        .unwrap();
    for (initialization, input) in initializations.into_iter().zip(inputs) {
        let PreparedRunInitialization::Expanded(initialization) = initialization else {
            panic!("compatible run requests must retain their retired cohort handoff");
        };
        assert!(matches!(
            initialization
                .get::<Initialization<CpuDeviceRuntime>>()
                .unwrap()
                .state,
            InitializationState::PrepareDownstream { .. }
        ));
        assert!(matches!(
            pipeline.continue_initialization_blocking(initialization, input),
            Err(ImageBatchError::Rejected(_))
        ));
    }
    assert!(!pipeline.is_stopped());
    assert_eq!(runtime.resident_bytes(), 0);
    pipeline.shutdown().unwrap();
    drop(pipeline);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn explicit_main_vision_reuse_preserves_eager_recipe_validation() {
    let (mut pipeline, _fixture) =
        pipeline_with_main_vision_reuse(PH3ImageMainVisionReuse::ExactImageWithinCohort);
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
    let inputs = views(&requests, &ids, &[false, false]);
    let mut owner = pipeline
        .begin_initialization_cohort(&mut prepared, &inputs)
        .unwrap();
    let result = advance(&mut pipeline, &mut owner, &inputs).await;
    // The explicit reuse path still builds every recipe before calculating its
    // shared output capacity, and rejects this backend's unsupported H3 union.
    assert!(result.completed.iter().all(|completion| matches!(
        completion.outcome,
        ImageInitializationCohortOutcome::Rejected { .. }
    )));
    assert_eq!(runtime.resident_bytes(), 0);
    assert!(!pipeline.is_stopped());
    drop((result, owner, prepared));
    pipeline.shutdown().unwrap();
    drop(pipeline);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
