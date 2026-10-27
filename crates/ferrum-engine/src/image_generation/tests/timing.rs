//! Exercise the shared stage path on a real managed Reference runtime. That
//! runtime intentionally rejects device timing instead of inventing CPU GPU time.
use super::{fixture::*, *};
use ferrum_models::qwen_image21::IMAGE_PROJECTION_INPUT_ID;

#[test]
fn tensor_device_timing_default_off_and_explicit_selection_reach_runtime() {
    assert_eq!(
        TensorExecutorConfig::default().device_timing,
        DeviceTimingMode::Off
    );
    for mode in [
        DeviceTimingMode::Off,
        DeviceTimingMode::Completion,
        DeviceTimingMode::Kernel,
    ] {
        let (mut pipeline, _assets) = super::adapter::pipeline(1);
        pipeline.config.executor.device_timing = mode;
        let directory = tempfile::tempdir().unwrap();
        let source = archive(
            directory.path(),
            vec![8, 64],
            safetensors::tensor::Dtype::BF16,
        );
        let stage = PreparedStage::compile(
            ImageStage::Denoiser,
            1,
            projection(),
            &projection_options(),
            &pipeline,
        )
        .unwrap()
        .start(&pipeline, 0, ComponentWeightSource::Stored(&source))
        .unwrap_or_else(|error| panic!("start managed stage: {error}"));
        let input = ManagedHostTensor::new(
            id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
            ResolvedTensorSpec::new(
                vec![1, 64],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            pipeline
                .host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, 128)
                .unwrap()
                .freeze(),
        )
        .unwrap();
        let before = pipeline.runtime.snapshot();
        let result =
            stage.execute_on_actor(pipeline.serial_actor.as_mut().unwrap(), 0, vec![input]);
        if mode == DeviceTimingMode::Off {
            let (stage, outputs) =
                result.unwrap_or_else(|error| panic!("execute with default timing: {error}"));
            assert!(outputs
                .iter()
                .all(|output| output.device_timing().is_none()));
            assert!(outputs
                .iter()
                .all(|output| output.bytes().bytes().iter().all(|&x| x == 0)));
            assert!(stage.receipt.device_timings.is_empty());
            assert_eq!(stage.receipt.completed_executions, 1);
            assert!(pipeline.runtime.snapshot().submissions > before.submissions);
            drop(outputs);
            stage
                .close()
                .unwrap_or_else(|error| panic!("close default stage: {error}"));
        } else {
            let Err(ImageGenerationFailure::Execution { source, active }) = result else {
                panic!("Reference must reject explicit device timing without silent fallback")
            };
            let TensorExecutionError::Dispatch {
                source,
                mut retirement,
                reaper,
            } = source
            else {
                panic!("expected the runtime's pre-submission rejection")
            };
            let OperationDispatchError::DefinitelyNotSubmitted { failures, retry } = source else {
                panic!("timing unsupported must be definitely not submitted")
            };
            assert!(!failures.is_empty());
            assert!(failures.iter().all(|failure| failure
                .failure()
                .message()
                .contains("reference runtime does not provide device timing evidence")));
            assert_eq!(pipeline.runtime.snapshot().submissions, before.submissions);
            assert!(active.receipt.device_timings.is_empty());
            drop((failures, retry, reaper));
            retirement.try_abort().unwrap();
            drop(retirement);
            active
                .close()
                .unwrap_or_else(|error| panic!("close timing-rejected stage: {error}"));
        }
        assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    }
}
