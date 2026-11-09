//! Actual managed portable execution; no half-precision intermediate.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
#[path = "support/pointwise_f32_component.rs"]
mod pointwise;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use fixture::{host, id, Kind};

#[test]
fn ordinary_f32_pointwise_executes_all_rows_and_retires_its_managed_owners() {
    let composition =
        ReferenceVNextComposition::create(id("device.reference.pointwise_f32.fixture")).unwrap();
    for kind in [Kind::F32ResidualAdd, Kind::F32SiluMul] {
        let component = pointwise::config(kind).compile(
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
                    run_id: id("run.pointwise_f32.init"),
                    request_id: id("request.pointwise_f32.init"),
                    transaction_id: id("transaction.pointwise_f32.init"),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        for trial in 0..2 {
            drop(pointwise::execute(&mut session, &host, kind, trial));
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("Reference pointwise close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    }
    assert!(composition.runtime().snapshot().submissions > 0);
}
#[test]
fn f32_pointwise_rejects_a_device_budget_smaller_than_one_input_before_allocation() {
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.pointwise_f32.capacity"),
        3 * 8 * 4 - 1,
    )
    .unwrap();
    for kind in [Kind::F32ResidualAdd, Kind::F32SiluMul] {
        assert!(pointwise::config(kind)
            .try_compile(
                composition.runtime().as_ref(),
                composition.registry(),
                composition.catalog(),
                composition.weight_materializers()
            )
            .is_err());
    }
    assert_eq!(composition.runtime().snapshot().allocations, 0);
    assert_eq!(composition.runtime().snapshot().submissions, 0);
}
