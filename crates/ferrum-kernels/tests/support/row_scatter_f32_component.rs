//! Real managed selected-row components, including two request-sized row
//! partitions, raw NaN payloads, and the fractional-F32 addition counterexample.
use crate::fixture::{id, Config, Kind, INPUT, OUTPUT, WEIGHT};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource, TensorComponentSession},
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_runtime::*;
use std::sync::Arc;
pub fn config(kind: Kind) -> Config {
    assert!(matches!(
        kind,
        Kind::RowScatterReplaceF32 | Kind::RowScatterAddF32
    ));
    Config {
        rows: 8,
        inputs: 4,
        outputs: 4,
        kind,
    }
}
pub fn execute<R: DeviceRuntime>(
    session: &mut TensorComponentSession<R>,
    host: &HostMemoryAccount,
    kind: Kind,
    trial: usize,
) -> Vec<TensorExecutionOutput> {
    let mut base = [
        0x7f800001u32,
        0xffc12345,
        0x80000000,
        0x00000001,
        0x3f800800,
        0x3f800000,
        0x7f800000,
        0x7fc12345,
        0x3f800000,
        0x80000000,
        0x00000001,
        0xff800000,
        0x7fc98765,
        0x80000000,
        0x00000003,
        0x3f810000,
        0x7f800001,
        0xffc12345,
        0x80000000,
        0x00000001,
        0x3f800000,
        0x3f800000,
        0x3f800000,
        0x3f800000,
        0x00000000,
        0x00800000,
        0x80000001,
        0xbf800000,
        0x3f810000,
        0x3f820000,
        0x3f830000,
        0x3f840000,
    ];
    let mut updates = [
        0xbf800000u32,
        0x34400000,
        0xff800000,
        0x3f800000,
        0x7f800001,
        0x80000000,
        0x00000001,
        0x3f800000,
        0x3f800800,
        0x7fc12345,
        0x80000000,
        0x00000001,
    ];
    if trial == 1 {
        base[4] = (-3.25_f32).to_bits();
        updates[0] = 0.25_f32.to_bits();
    }
    let c = config(kind);
    let input = base
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let delta = updates
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let outputs = session
        .execute(
            id(format!("run.scatter_f32.{trial}")),
            id(format!("request.scatter_f32.{trial}")),
            vec![
                c.managed(host, INPUT, &input),
                c.managed(host, WEIGHT, &delta),
            ],
        )
        .unwrap();
    let [out] = outputs.as_slice() else {
        panic!("scatter output arity")
    };
    assert_eq!(out.value_id().as_str(), OUTPUT);
    assert_eq!(out.tensor().dimensions(), [8, 4]);
    assert_eq!(out.tensor().element_type(), ElementType::F32);
    assert!(host.owns(out.bytes()));
    let actual = out
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    for row in 0..8 {
        for col in 0..4 {
            let i = row * 4 + col;
            let update = match row {
                1 => Some(col),
                2 => Some(4 + col),
                5 => Some(8 + col),
                _ => None,
            };
            match update {
                None => assert_eq!(actual[i], base[i], "unselected raw bits row{row} col{col}"),
                Some(u) if matches!(kind, Kind::RowScatterReplaceF32) => {
                    assert_eq!(actual[i], updates[u])
                }
                Some(u) => {
                    let expected = (f64::from(f32::from_bits(base[i]))
                        + f64::from(f32::from_bits(updates[u])))
                        as f32;
                    if expected.is_nan() {
                        assert!(f32::from_bits(actual[i]).is_nan());
                    } else {
                        assert_eq!(
                            actual[i],
                            expected.to_bits(),
                            "single F32 sum row{row} col{col}"
                        );
                    }
                }
            }
        }
    }
    if trial == 0 && matches!(kind, Kind::RowScatterAddF32) {
        assert_eq!(actual[4], 0.000244140625_f32.to_bits());
    }
    outputs
}
pub fn exercise<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for kind in [Kind::RowScatterReplaceF32, Kind::RowScatterAddF32] {
        let component = config(kind).compile(runtime.as_ref(), registry, catalog, materializers);
        let host = crate::fixture::host();
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.scatter.init"),
                    request_id: id("request.scatter.init"),
                    transaction_id: id("transaction.scatter.init"),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let a = execute(&mut session, &host, kind, 0);
        let b = execute(&mut session, &host, kind, 1);
        assert_eq!(
            &a[0].bytes().bytes()[4 * 4 * 4..],
            &b[0].bytes().bytes()[4 * 4 * 4..],
            "perturbing request0 must preserve request1 exactly"
        );
        drop(a);
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("scatter close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert!(host.owns(b[0].bytes()));
        drop(b);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
