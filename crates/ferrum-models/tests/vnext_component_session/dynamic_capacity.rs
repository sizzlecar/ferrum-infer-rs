//! Cold immutable installation and transient runnable capacity are separate.
use super::*;
use ferrum_execution::{TensorDynamicPreparationOutcome, TensorSerialActor};

#[test]
fn default_cold_initialization_refuses_dynamic_pressure_before_weight_upload() {
    let fixture = Fixture::new("device.component.cold-pool-refusal");
    let component = fixture.compile_rows(2, &fixture.family);
    let static_bytes = component
        .executable()
        .execution_plan()
        .payload()
        .memory()
        .static_bytes();
    let pressure = fixture
        .capacity
        .reserve(
            DeviceOpaqueMemoryKind::Bootstrap,
            DEVICE_BYTES - static_bytes,
        )
        .unwrap();
    let runtime = fixture.composition.runtime().clone();
    let before = runtime.snapshot();
    let result = component.initialize(
        runtime.clone(),
        fixture.host.clone(),
        ComponentWeightSource::Stored(&fixture.source),
        Fixture::init_config("cold-pool-refusal"),
    );
    let Err(ferrum_execution::component::ComponentInitializationError::Pool {
        source: VNextError::DeviceCapacityUnavailable(_),
        transaction,
        ..
    }) = result
    else {
        panic!("default cold initialization must preserve its clean pool refusal")
    };
    drop(transaction.release().unwrap());
    let after = runtime.snapshot();
    assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    assert_eq!(after.dense_linear_launches, before.dense_linear_launches);
    drop(pressure);
    fixture.assert_device_reusable();
}

#[test]
fn stored_installation_survives_dynamic_pressure_and_executes_after_release() {
    let fixture = Fixture::new("device.component.cold-pressure");
    let component = fixture.compile_rows(2, &fixture.family);
    let static_bytes = component
        .executable()
        .execution_plan()
        .payload()
        .memory()
        .static_bytes();
    // Another live owner leaves precisely the space needed for immutable
    // weights. This must not turn temporary execution pressure into a failed
    // installation with partially acquired runtime pools.
    let pressure = fixture
        .capacity
        .reserve(
            DeviceOpaqueMemoryKind::Bootstrap,
            DEVICE_BYTES - static_bytes,
        )
        .unwrap();
    let runtime = fixture.composition.runtime().clone();
    let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
    let mut session = component
        .initialize_with_pool_policy(
            runtime.clone(),
            fixture.host.clone(),
            ComponentWeightSource::Stored(&fixture.source),
            Fixture::init_config("cold-pressure"),
            ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
        )
        .unwrap();
    let installed = runtime.snapshot();
    let outcome = session
        .try_prepare_dynamic_backing_on_actor(
            &mut actor,
            id("run.cold-pressure.blocked"),
            id("request.cold-pressure.blocked"),
            vec![fixture.input(1.)],
        )
        .unwrap();
    let TensorDynamicPreparationOutcome::Capacity { refusal, .. } = outcome else {
        panic!("transient capacity must defer execution")
    };
    assert!(refusal.is_capacity_refusal());
    drop(refusal);
    let blocked = runtime.snapshot();
    assert_eq!(blocked.live_allocations, installed.live_allocations);
    assert_eq!(blocked.uploaded_bytes, installed.uploaded_bytes);
    assert_eq!(
        blocked.dense_linear_launches,
        installed.dense_linear_launches
    );
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    drop(pressure);
    assert!(matches!(
        session
            .try_prepare_dynamic_backing_on_actor(
                &mut actor,
                id("run.cold-pressure.ready"),
                id("request.cold-pressure.ready"),
                vec![fixture.input(1.)],
            )
            .unwrap(),
        TensorDynamicPreparationOutcome::Prepared(_)
    ));
    let outputs = session
        .execute_on_actor(
            &mut actor,
            id("run.cold-pressure.execute"),
            id("request.cold-pressure.execute"),
            vec![fixture.input(1.)],
        )
        .unwrap();
    assert_eq!(outputs[0].bytes().bytes(), bytes(&EXPECTED));
    assert_eq!(
        runtime.snapshot().uploaded_bytes - installed.uploaded_bytes,
        INPUT_BYTES as u64
    );
    match session.close() {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {}
        Ok(_) => panic!("completed component retained an owner"),
        Err(error) => panic!("close failed: {:?}", error.failure()),
    }
    drop(actor);
    fixture.assert_device_reusable();
    drop(outputs);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
}
