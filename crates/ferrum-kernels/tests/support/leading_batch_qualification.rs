use super::fixture::*;
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
    for (b, s, c) in [(2, 128, 256), (3, 7, 19)] {
        let bytes: Vec<u8> = (0..b * s * c)
            .flat_map(|i| (i as u16).to_le_bytes())
            .collect();
        for kind in [
            Kind::LeadingBatchMerge,
            Kind::LeadingBatchSplit,
            Kind::LeadingBatchMergeView,
            Kind::LeadingBatchSplitView,
        ] {
            let config = Config {
                rows: b,
                inputs: s,
                outputs: c,
                kind,
            };
            let compiled = config.compile(runtime.as_ref(), registry, catalog, materializers);
            let bindings = compiled.executable().execution_plan().payload().nodes()[0].values();
            let is_view = matches!(
                kind,
                Kind::LeadingBatchMergeView | Kind::LeadingBatchSplitView
            );
            assert_eq!(bindings[0].storage() == bindings[1].storage(), is_view);
            let host = host();
            let mut session = compiled
                .initialize(
                    runtime.clone(),
                    host.clone(),
                    ComponentWeightSource::None,
                    ComponentInitializationConfig {
                        run_id: id("run.leading.init"),
                        request_id: id("request.leading.init"),
                        transaction_id: id("transaction.leading.init"),
                        static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                        executor: TensorExecutorConfig::default(),
                    },
                )
                .unwrap();
            let mut retained = None;
            for _ in 0..2 {
                let out = session
                    .execute(
                        id("run.leading.step"),
                        id("request.leading.step"),
                        vec![config.managed(&host, INPUT, &bytes)],
                    )
                    .unwrap();
                let shape = if matches!(kind, Kind::LeadingBatchMerge | Kind::LeadingBatchMergeView)
                {
                    vec![b * s, c]
                } else {
                    vec![b, s, c]
                };
                assert_eq!(out[0].tensor().dimensions(), shape);
                assert_eq!(out[0].bytes().bytes(), bytes);
                retained = Some(out);
            }
            assert!(matches!(
                session.close().unwrap_or_else(|e| {
                    let why = format!("{:?}", e.failure());
                    std::mem::forget(e);
                    panic!("close owner retained: {why}")
                }),
                PlanRuntimeCloseOutcome::Closed(_)
            ));
            assert!(host.owns(retained.as_ref().unwrap()[0].bytes()));
            drop(retained);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
}
