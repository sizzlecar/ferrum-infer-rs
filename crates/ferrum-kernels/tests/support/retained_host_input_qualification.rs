//! A completed output slice remains an accounted input after its producer closes.
use super::fixture::{id, Config, Kind, INPUT};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use std::sync::Arc;

pub fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            240,
            HostMemoryClassLimits {
                input_preprocess_bytes: 96,
                output_bytes: 144,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let config = Config {
        rows: 2,
        inputs: 3,
        outputs: 8,
        kind: Kind::LeadingBatchMerge,
    };
    let initialize = |config: Config, name: &str| {
        config
            .compile(runtime.as_ref(), registry, catalog, materializers)
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id(format!("run.retained_output.init.{name}")),
                    request_id: id(format!("request.retained_output.{name}")),
                    transaction_id: id(format!("transaction.retained_output.{name}")),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap()
    };
    let mut producer = initialize(config, "producer");
    let bytes: Vec<_> = (0..48u16)
        .flat_map(|i| i.wrapping_mul(1381).to_le_bytes())
        .collect();
    let outputs = producer
        .execute(
            id("run.retained_output.produce"),
            id("request.retained_output.producer"),
            vec![config.managed(&host, INPUT, &bytes)],
        )
        .unwrap();
    assert_eq!(outputs[0].bytes().bytes(), bytes);
    let consumer_config = Config { rows: 1, ..config };
    let spec = consumer_config.input_specs().remove(&id(INPUT)).unwrap();
    let input = ManagedHostTensor::from_retained_output(
        id(INPUT),
        ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap(),
        outputs[0].bytes().slice(48..96).unwrap(),
    )
    .unwrap();
    assert_eq!(input.bytes().class(), HostMemoryClass::Output);
    assert_eq!(
        input.bytes().bytes().as_ptr(),
        outputs[0].bytes().bytes()[48..].as_ptr()
    );
    drop(outputs);
    assert!(matches!(
        producer.close(),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 96);
    // No room exists for even one duplicate InputPreprocess byte. The next
    // native upload must retain the original Output allocation and its slice.
    let pressure = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 96)
        .unwrap();
    assert!(host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .is_err());
    let mut consumer = initialize(consumer_config, "consumer");
    let result = consumer
        .execute(
            id("run.retained_output.consume"),
            id("request.retained_output.consumer"),
            vec![input],
        )
        .unwrap();
    assert_eq!(result[0].bytes().bytes(), &bytes[48..]);
    assert!(matches!(
        consumer.close(),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 144);
    drop(pressure);
    assert_eq!(host.usage().unwrap().committed_bytes(), 48);
    assert_eq!(result[0].bytes().bytes(), &bytes[48..]);
    drop(result);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
