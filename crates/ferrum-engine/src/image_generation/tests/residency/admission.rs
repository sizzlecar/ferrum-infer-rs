use super::*;
use ferrum_execution::TensorAdmissionControl;

#[test]
fn shared_pipeline_cancel_retires_transients_and_reuses_the_same_stage_and_actor() {
    let (mut pipeline, _directory) = numeric_pipeline();
    assert!(
        pipeline.admission_control().is_some(),
        "run and serve share the explicitly waiting pipeline"
    );
    let mut stages = Vec::new();
    let stage = prepared(&pipeline, 1);
    let active = pipeline
        .start_stage(stage, 0, &mut stages)
        .unwrap_or_else(|e| panic!("{e}"));
    let control = TensorAdmissionControl::new();
    control.cancel();
    pipeline.set_admission_control(control);
    let before = pipeline.runtime.snapshot();
    let inputs = vec![input(&pipeline.host, 1, 0)];
    let failure = pipeline
        .execute_stage(active, 0, inputs)
        .err()
        .expect("cancelled execution");
    let failure = pipeline.retire_failed_execution(failure, &mut stages);
    assert!(matches!(failure, ImageGenerationFailure::Cancelled(_)));
    assert!(!pipeline.is_stopped());
    assert_eq!(pipeline.runtime.snapshot().submissions, before.submissions);
    assert_eq!(
        pipeline
            .host
            .usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output),
        0
    );
    pipeline.set_admission_control(TensorAdmissionControl::new());
    let stage = prepared(&pipeline, 1);
    let active = pipeline
        .start_stage(stage, 1, &mut stages)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(active.receipt.session_reused);
    let inputs = vec![input(&pipeline.host, 1, 0)];
    let (active, outputs) = pipeline
        .execute_stage(active, 1, inputs)
        .unwrap_or_else(|e| panic!("{e}"));
    let output = only_output(outputs, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
    for (index, part) in output.bytes().bytes().chunks_exact(2).enumerate() {
        assert_eq!(
            u16::from_le_bytes([part[0], part[1]]),
            bf16::from_f32((index + 1) as f32).to_bits()
        );
    }
    drop(output);
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|e| panic!("{e}"));
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cancellation_never_masks_a_failed_stage_retirement() {
    let (mut pipeline, _directory) = numeric_pipeline();
    let mut stages = Vec::new();
    let stage = prepared(&pipeline, 1);
    let active = pipeline
        .start_stage(stage, 0, &mut stages)
        .unwrap_or_else(|e| panic!("{e}"));
    // An occupied destination is a real retirement failure with a retained
    // session owner; a cancellation proof cannot make that owner disappear.
    let other = prepared(&pipeline, 1);
    let other = pipeline
        .start_stage(other, 1, &mut stages)
        .unwrap_or_else(|e| panic!("{e}"));
    pipeline
        .finish_stage(other, &mut stages)
        .unwrap_or_else(|e| panic!("{e}"));
    let control = TensorAdmissionControl::new();
    control.cancel();
    pipeline.set_admission_control(control);
    let inputs = vec![input(&pipeline.host, 1, 0)];
    let failure = pipeline
        .execute_stage(active, 0, inputs)
        .err()
        .expect("cancelled execution");
    let failure = pipeline.retire_failed_execution(failure, &mut stages);
    assert!(pipeline.is_stopped());
    let ImageGenerationFailure::Processing {
        active: Some(active),
        ..
    } = failure
    else {
        panic!("failed retirement must retain the actual active stage");
    };
    active.close().unwrap_or_else(|e| panic!("{e}"));
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}
