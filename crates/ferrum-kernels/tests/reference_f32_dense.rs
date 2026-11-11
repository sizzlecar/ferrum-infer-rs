//! Full managed F32 execution on the portable Reference backend.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use fixture::*;

#[test]
fn managed_f32_dense_component_retains_small_differences_and_all_output_rows() {
    managed_f32_component(false);
}
#[test]
fn managed_f32_affine_component_preserves_full_precision_bias_and_all_rows() {
    managed_f32_component(true);
}
#[test]
fn managed_f32_to_bf16_cast_executes_ieee_conversion_and_retires_all_owners() {
    let composition =
        ReferenceVNextComposition::create(id("device.reference.cast_f32_to_bf16.component"))
            .unwrap();
    let component = cast_f32_to_bf16_config().compile(
        composition.runtime().as_ref(),
        composition.registry(),
        composition.catalog(),
        composition.weight_materializers(),
    );
    let host = host();
    let mut session = component
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.cast.init"),
                request_id: id("request.cast.init"),
                transaction_id: id("transaction.cast.init"),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    for trial in 0..2 {
        drop(execute_cast_f32_to_bf16(&mut session, &host, trial));
    }
    assert!(composition.runtime().snapshot().submissions > 0);
    assert!(matches!(
        session.close().unwrap_or_else(|failure| panic!(
            "Reference cast close failed: {:?}",
            failure.failure()
        )),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
#[test]
fn managed_f32_affine_rejects_capacity_smaller_than_its_input_before_allocation() {
    let input_bytes = 3 * 3 * 4_u64;
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.affine_f32.capacity"),
        input_bytes - 1,
    )
    .unwrap();
    assert!(affine_f32_config()
        .try_compile(
            composition.runtime().as_ref(),
            composition.registry(),
            composition.catalog(),
            composition.weight_materializers()
        )
        .is_err());
    assert_eq!(composition.runtime().snapshot().allocations, 0);
    assert_eq!(composition.runtime().snapshot().submissions, 0);
}
fn managed_f32_component(affine: bool) {
    let composition =
        ReferenceVNextComposition::create(id("device.reference.dense_f32.component")).unwrap();
    let component = if affine {
        affine_f32_config()
    } else {
        f32_config()
    }
    .compile(
        composition.runtime().as_ref(),
        composition.registry(),
        composition.catalog(),
        composition.weight_materializers(),
    );
    let host = host();
    let mut session = component
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.dense_f32.init"),
                request_id: id("request.dense_f32.init"),
                transaction_id: id("transaction.dense_f32.init"),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let output = if affine {
        execute_affine_f32(&mut session, &host, 0)
    } else {
        execute_f32(&mut session, &host)
    };
    assert!(composition.runtime().snapshot().submissions > 0);
    assert!(matches!(
        session.close().unwrap_or_else(|failure| panic!(
            "Reference component close failed: {:?}",
            failure.failure()
        )),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(output[0].bytes().bytes().len(), 24);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
