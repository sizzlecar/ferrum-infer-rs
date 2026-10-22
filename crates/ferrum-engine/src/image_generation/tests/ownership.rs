use super::fixture::*;
use super::*;

#[test]
fn stage_execution_failure_keeps_real_stored_resources_until_explicit_recovery_close() {
    let composition = composition("device.image.stage.error-owner");
    let capacity =
        DeviceCapacityOwner::open(composition.runtime().descriptor(), DEVICE_BYTES).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = archive(
        directory.path(),
        vec![8, 64],
        safetensors::tensor::Dtype::BF16,
    );
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1024,
                output_bytes: 1024,
                transfer_staging_bytes: 1024,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let component = compile(&composition, projection(), &projection_options()).unwrap();
    let plan = ImageStagePlanReceipt {
        stage: ImageStage::Denoiser,
        batch_size: 1,
        family: component.family().family_id().clone(),
        profile: component.family().numerical_profile().id.clone(),
        family_fingerprint: component.family().fingerprint().unwrap(),
        plan_hash: component
            .executable()
            .execution_plan()
            .plan_hash()
            .to_string(),
        static_bytes: component
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .static_bytes(),
        minimum_step_bytes: component
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .minimum_step_bytes(),
    };
    let session = component
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                request_id: id("request.image.owner.init").unwrap(),
                run_id: id("run.image.owner.init").unwrap(),
                transaction_id: id("transaction.image.owner.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let active = ActiveImageStage {
        receipt: ImageStageExecutionReceipt {
            plan,
            initialization: session.initialization_receipt().clone(),
            completed_executions: 0,
        },
        session,
    };
    let before = composition.runtime().snapshot();
    assert!(before.live_allocations > 0);
    let Err(ImageGenerationFailure::Execution { source, active }) = active.execute(0, Vec::new())
    else {
        panic!("missing actual input must return the original execution error and session")
    };
    assert!(matches!(source, TensorExecutionError::Runtime(_)));
    assert_eq!(active.receipt.completed_executions, 0);
    let after = composition.runtime().snapshot();
    assert_eq!(after.submissions, before.submissions);
    assert_eq!(after.dense_linear_launches, before.dense_linear_launches);
    assert!(after.live_allocations > 0);
    assert!(capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
        .is_err());
    // Retain the error and actual session until the caller chooses recovery.
    drop(source);
    match active.close() {
        Ok(receipt) => assert_eq!(receipt.close.released_static_resources(), 1),
        Err(_) => panic!("pre-submission failure must permit explicit close"),
    }
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    drop(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap(),
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
