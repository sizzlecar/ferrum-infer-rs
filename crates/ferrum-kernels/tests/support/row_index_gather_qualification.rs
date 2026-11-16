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
    // Whole BF16 encoding space, vector-width boundaries, the production hidden
    // width, and more rows than a single bounded CUDA grid can cover.
    for (b, s, c, repeat_map) in [
        (2, 256, 256, 1),
        (3, 5, 1, 1),
        (3, 5, 7, 1),
        (3, 5, 8, 1),
        (3, 5, 16, 1),
        (2, 3, 7168, 1),
        (2, 2, 8, 5500),
    ] {
        let source: Vec<u8> = (0..b * s * c)
            .flat_map(|i| {
                (i as u16)
                    .wrapping_add((i / (s * c)) as u16 * 257)
                    .to_le_bytes()
            })
            .collect();
        let mut map: Vec<i32> = (0..s as i32).rev().collect();
        map.extend([0, -1, s as i32, i32::MIN, i32::MAX]);
        let mut map = map.repeat(repeat_map);
        let config = Config {
            rows: b,
            inputs: s,
            outputs: c,
            kind: Kind::RowIndexGather {
                output_rows: map.len() as u64,
            },
        };
        let compiled = config.compile(runtime.as_ref(), registry, catalog, materializers);
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                16 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 4 << 20,
                    transfer_staging_bytes: 4 << 20,
                    output_bytes: 4 << 20,
                    weight_backing_bytes: 4 << 20,
                },
            )
            .unwrap(),
        );
        let mut session = compiled
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.gather.init"),
                    request_id: id("request.gather.init"),
                    transaction_id: id("transaction.gather.init"),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut retained = None;
        for trial in 0..2 {
            map.rotate_left(trial);
            let indices: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
            let out = session
                .execute(
                    id(format!("run.gather.{trial}")),
                    id(format!("request.gather.{trial}")),
                    vec![
                        config.managed(&host, INPUT, &source),
                        config.managed(&host, RELATIVE, &indices),
                    ],
                )
                .unwrap();
            assert_eq!(out[0].tensor().dimensions(), [b, map.len() as u64, c]);
            let row_bytes = c as usize * 2;
            for (batch, actual_batch) in out[0]
                .bytes()
                .bytes()
                .chunks_exact(map.len() * row_bytes)
                .enumerate()
            {
                for (&index, actual) in map.iter().zip(actual_batch.chunks_exact(row_bytes)) {
                    if let Some(row) = usize::try_from(index).ok().filter(|&row| row < s as usize) {
                        let start = (batch * s as usize + row) * row_bytes;
                        assert_eq!(actual, &source[start..start + row_bytes]);
                    } else {
                        assert!(actual.iter().all(|&byte| byte == 0));
                    }
                }
            }
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
