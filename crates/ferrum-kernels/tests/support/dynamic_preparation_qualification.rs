use super::fixture::{self, *};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource, TensorComponentSession},
    TensorDynamicPreparationOutcome, TensorExecutorConfig, TensorSerialActor,
};
use ferrum_runtime::*;
use std::sync::Arc;

pub fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
    ceiling: u64,
    submissions: Option<&dyn Fn() -> u64>,
    native_pressure: Option<&dyn Fn() -> Box<dyn std::any::Any>>,
) {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            64 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 16 << 20,
                transfer_staging_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
    );
    let configs = [1, 2].map(|rows| Config {
        rows,
        inputs: 1024,
        outputs: 1024,
        kind: Kind::LeadingBatchMerge,
    });
    let compile = |c: Config, name: &str| {
        c.compile(runtime.as_ref(), registry, catalog, materializers)
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id(format!("run.dynamic.init.{name}")),
                    request_id: id(format!("request.dynamic.{name}")),
                    transaction_id: id(format!("transaction.dynamic.{name}")),
                    static_policy: StaticInitializationPolicy::new(4096, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap()
    };
    let mut single = compile(configs[0], "one");
    let mut two = compile(configs[1], "two");
    let inputs = |c: Config| {
        let bytes = (0..c.rows * c.inputs * c.outputs)
            .flat_map(|i| (i as u16).to_le_bytes())
            .collect::<Vec<_>>();
        vec![c.managed(&host, INPUT, &bytes)]
    };
    let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
    let run =
        |s: &mut TensorComponentSession<R>, a: &mut TensorSerialActor<R>, c: Config, name: &str| {
            let input = inputs(c);
            let expected = input[0].bytes().bytes().to_vec();
            let out = s
                .execute_on_actor(
                    a,
                    id(format!("run.dynamic.{name}")),
                    id("request.dynamic"),
                    input,
                )
                .unwrap();
            assert_eq!(out[0].bytes().bytes(), expected);
        };
    run(&mut single, &mut actor, configs[0], "before");
    // A real reservation on the original device account creates the precise
    // quota boundary. It is not an allocator-failure string or simulated OOM.
    let owner = DeviceCapacityOwner::open(runtime.descriptor(), ceiling).unwrap();
    let pressure: Box<dyn std::any::Any> = match native_pressure {
        Some(make_pressure) => make_pressure(),
        None => {
            let available = match owner.reserve(DeviceOpaqueMemoryKind::Bootstrap, ceiling) {
                Err(VNextError::DeviceCapacityUnavailable(p)) => p.available_bytes(),
                other => panic!("live first plan must occupy original quota: {other:?}"),
            };
            Box::new(
                owner
                    .reserve(DeviceOpaqueMemoryKind::Bootstrap, available)
                    .unwrap(),
            )
        }
    };
    assert!(matches!(
        single.try_release_idle_dynamic_backing().unwrap(),
        IdleDynamicReleaseOutcome::Released(_)
    ));
    let before = submissions.map(|read| read());
    let refused = two
        .try_prepare_dynamic_backing_on_actor(
            &mut actor,
            id("run.dynamic.reject"),
            id("request.dynamic.two"),
            inputs(configs[1]),
        )
        .unwrap();
    match refused {
        TensorDynamicPreparationOutcome::Capacity { rollback, refusal } => {
            if native_pressure.is_some() {
                assert!(
                    matches!(
                        refusal,
                        ferrum_execution::TensorExecutionError::Runtime(
                            VNextError::DeviceAllocationUnavailable(_)
                        )
                    ),
                    "native test must reach a real driver capacity failure: {refusal:?}"
                );
            }
            eprintln!(
                "candidate preparation rolled back {} real device bytes",
                rollback.reclaimed_bytes()
            );
        }
        _ => panic!("B2 cannot fit the backing released by this B1"),
    }
    assert!(!actor.is_failed());
    if let Some(read) = submissions {
        assert_eq!(Some(read()), before, "preparation must not submit commands");
    }
    let ready = single
        .try_prepare_dynamic_backing_on_actor(
            &mut actor,
            id("run.dynamic.resume"),
            id("request.dynamic.one"),
            inputs(configs[0]),
        )
        .unwrap();
    assert!(matches!(
        ready,
        TensorDynamicPreparationOutcome::Prepared(_)
    ));
    if let Some(read) = submissions {
        assert_eq!(Some(read()), before);
    }
    run(&mut single, &mut actor, configs[0], "after_refusal");
    drop(pressure);
    let before = submissions.map(|read| read());
    let ready = two
        .try_prepare_dynamic_backing_on_actor(
            &mut actor,
            id("run.dynamic.join"),
            id("request.dynamic.two"),
            inputs(configs[1]),
        )
        .unwrap();
    assert!(matches!(
        ready,
        TensorDynamicPreparationOutcome::Prepared(_)
    ));
    if let Some(read) = submissions {
        assert_eq!(Some(read()), before);
    }
    run(&mut two, &mut actor, configs[1], "joined");
    drop(actor);
    match single.close() {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => (),
        Ok(_retained) => panic!("B1 close retained a live original plan"),
        Err(_retained) => panic!("B1 close failed with its original owner"),
    }
    match two.close() {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => (),
        Ok(_retained) => panic!("B2 close retained a live original plan"),
        Err(_retained) => panic!("B2 close failed with its original owner"),
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
