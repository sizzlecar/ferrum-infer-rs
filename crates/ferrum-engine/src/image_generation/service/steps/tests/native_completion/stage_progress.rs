//! Progress seals whole-stage retries without discarding typed native pressure
//! or the original Host scope retained by a composed pipeline failure.
use super::*;
use ferrum_runtime::HostMemoryClass;

#[derive(Debug, PartialEq, Eq)]
enum Refusal {
    Allocation(ImageStage, ferrum_runtime::FailureEnvelope),
    Capacity(ImageStage, u64, u64),
    InitializationCapacity(ImageStage, String, ferrum_runtime::DeviceCapacityPressure),
    InitializationContract(ImageStage, String, ferrum_runtime::DeviceCapacityPressure),
}
fn evidence(failure: &ImageGenerationFailure<Runtime>) -> Refusal {
    match failure {
        ImageGenerationFailure::AllocationRejected { stage, failure } => {
            Refusal::Allocation(*stage, failure.clone())
        }
        ImageGenerationFailure::CapacityRejected {
            stage,
            additional_bytes,
            available_bytes,
        } => Refusal::Capacity(*stage, *additional_bytes, *available_bytes),
        ImageGenerationFailure::InitializationCapacityRejected { plan, pressure } => {
            Refusal::InitializationCapacity(plan.stage, plan.plan_hash.clone(), pressure.clone())
        }
        ImageGenerationFailure::Initialization {
            plan,
            source:
                ferrum_execution::component::ComponentInitializationError::Contract(
                    ferrum_runtime::VNextError::DeviceCapacityUnavailable(pressure),
                ),
        } => Refusal::InitializationContract(plan.stage, plan.plan_hash.clone(), pressure.clone()),
        other => panic!("actual acknowledged native refusal required: {other}"),
    }
}

fn refused_decoder(
    pipeline: &mut QwenImage21Pipeline<Runtime>,
) -> (
    ImageGenerationFailure<Runtime>,
    super::super::native_pressure::NativePressure,
) {
    pipeline.execution.config.residency = ImageResidencyPolicy::Adaptive;
    let config = pipeline
        .execution
        .sources
        .assets
        .decoder_batch(1, 2, 2)
        .unwrap();
    let family = pipeline
        .prepare_family(
            ImageStage::Decoder,
            ImageStagePlanRole::WholeStage,
            1,
            QwenImage21VaeDecoderFamily::new().unwrap(),
            &config,
            &pipeline.profiles.decoder,
        )
        .unwrap();
    let stage = PreparedImageStage::compile(
        ImageStage::Decoder,
        1,
        family,
        &config.compile_options().unwrap(),
        pipeline,
    )
    .unwrap();
    // Leave the decoder's true dynamic floor free: the pressure plan itself
    // must fit its own small work area, but the decoder's additional Stored
    // weights cannot fit alongside it. No fake native failure is injected.
    let memory = stage.executable().execution_plan().payload().memory();
    let headroom = memory.minimum_runnable_request_bytes();
    assert!(memory.static_bytes() > 0 && headroom > 0);
    let pressure = super::super::native_pressure::NativePressure::new(
        pipeline,
        pipeline.execution.config.device_capacity_bytes - headroom,
    );
    let failure = pipeline
        .execution
        .start_stage(stage, 0, &mut Vec::new())
        .err()
        .expect("the actual external allocation leaves no decoder capacity");
    let _ = evidence(&failure);
    (failure, pressure)
}

#[test]
fn decoder_zero_execution_progress_preserves_clean_native_admission_retry() {
    let (mut pipeline, _source) = native_pipeline("device.service.decoder.zero_execution_progress");
    let (failure, pressure) = refused_decoder(&mut pipeline);
    let original = evidence(&failure);
    let failure = failure.with_stage_execution_progress(ImageStage::Decoder, 0);
    assert_eq!(evidence(&failure), original);
    assert!(clean_decoder_admission(failure).is_ok());
    assert!(!pipeline.is_stopped());
    pressure.close();
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
}

#[test]
fn decoder_completed_execution_progress_keeps_typed_pressure_and_pipeline_host_scope() {
    let (mut pipeline, _source) =
        native_pipeline("device.service.decoder.completed_execution_progress");
    let (failure, pressure) = refused_decoder(&mut pipeline);
    let original = evidence(&failure);
    let host = pipeline.execution.host.clone();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let physical = host.backing_bytes(37).unwrap();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::Output, physical)])
        .unwrap();
    let failure = ImageGenerationFailure::Pipeline {
        source: Box::new(failure.with_stage_execution_progress(ImageStage::Decoder, 1)),
        stages: Vec::new(),
        weights: Vec::new(),
        host_scopes: vec![scope],
        host_backing: Vec::new(),
    };
    let failure = clean_decoder_admission(failure)
        .err()
        .expect("a completed group cannot be stripped into a whole-decoder retry");
    assert_eq!(host.usage().unwrap().committed_bytes(), physical);
    let failure = failure
        .recover_allocation()
        .err()
        .expect("progress must also seal generic allocation recovery");
    let ImageGenerationFailure::Pipeline {
        source,
        host_scopes,
        ..
    } = failure
    else {
        panic!("the original pipeline envelope must remain intact")
    };
    assert_eq!(host_scopes.len(), 1);
    let ImageGenerationFailure::StageExecutionProgress {
        stage,
        completed_executions,
        source,
    } = *source
    else {
        panic!("the original positive progress must remain intact")
    };
    assert_eq!(stage, ImageStage::Decoder);
    assert_eq!(completed_executions, 1);
    assert_eq!(evidence(&source), original);
    assert_eq!(host.usage().unwrap().committed_bytes(), physical);
    drop(host_scopes);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    pressure.close();
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.execution.runtime.snapshot().live_allocations, 0);
}
