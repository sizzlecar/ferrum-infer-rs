//! Managed component operands and an independent F64 ideal-expression oracle.
//! The 1e-6 absolute bound applies only to these finite, selected unsaturated
//! cells. IEEE edge cells are exact. Neither bound claims uniform libdevice
//! qualification or equivalence to an upstream Torch executable.
use crate::fixture::{id, Config, Kind, INPUT, OUTPUT};
use ferrum_execution::{component::TensorComponentSession, TensorExecutionOutput};
use ferrum_runtime::*;
use std::sync::Arc;

pub fn exercise<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    use ferrum_execution::{
        component::{ComponentInitializationConfig, ComponentWeightSource},
        TensorExecutorConfig,
    };
    for kind in [Kind::GeluTanhF32, Kind::GeluErfF32] {
        let component = config(kind).compile(runtime.as_ref(), registry, catalog, materializers);
        let host = crate::fixture::host();
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.gelu_f32.init"),
                    request_id: id("request.gelu_f32.init"),
                    transaction_id: id("transaction.gelu_f32.init"),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        drop(execute(&mut session, &host, kind, 0));
        let retained = execute(&mut session, &host, kind, 1);
        let bytes = retained[0].bytes().bytes().to_vec();
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("GELU close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        // Readbacks retain the actual Output authorization after the native
        // session retires. Their owners refund only when the caller drops them.
        assert!(host.owns(retained[0].bytes()));
        assert_eq!(retained[0].bytes().bytes(), bytes);
        assert!(host.usage().unwrap().committed_bytes() > 0);
        drop(retained);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

pub fn config(kind: Kind) -> Config {
    assert!(matches!(kind, Kind::GeluTanhF32 | Kind::GeluErfF32));
    Config {
        rows: 4,
        inputs: 8,
        outputs: 8,
        kind,
    }
}
pub fn execute<R: DeviceRuntime>(
    session: &mut TensorComponentSession<R>,
    host: &HostMemoryAccount,
    kind: Kind,
    trial: usize,
) -> Vec<TensorExecutionOutput> {
    let values = [
        0.0_f32,
        -0.0,
        1.000244140625,
        -1.0,
        1.0,
        -2.5,
        2.5,
        -10.0,
        10.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(3),
        -f32::from_bits(3),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -5.0,
        -3.5,
        -0.25,
        0.25,
        3.5,
        5.0,
        0.000244140625,
        -0.000244140625,
        0.5,
        -0.5,
        2.0,
        -2.0,
    ];
    let values = if trial % 2 == 0 {
        values
    } else {
        let mut reordered = values;
        reordered.rotate_left(8);
        reordered
    };
    let bytes = values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let outputs = session
        .execute(
            id(format!("run.gelu_f32.{trial}")),
            id(format!("request.gelu_f32.{trial}")),
            vec![config(kind).managed(host, INPUT, &bytes)],
        )
        .unwrap();
    let [output] = outputs.as_slice() else {
        panic!("GELU produced an unexpected output count")
    };
    assert_eq!(output.value_id().as_str(), OUTPUT);
    assert_eq!(output.tensor().dimensions(), [4, 8]);
    assert_eq!(output.tensor().element_type(), ElementType::F32);
    assert!(host.owns(output.bytes()));
    let actual = output
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), values.len());
    for (i, (&x, &actual)) in values.iter().zip(&actual).enumerate() {
        let d = f64::from(x);
        let expected = (0.5
            * d
            * (1.0
                + match kind {
                    Kind::GeluTanhF32 => {
                        libm::tanh(0.7978845608028654 * (d + 0.044715 * d * d * d))
                    }
                    Kind::GeluErfF32 => libm::erf(d * std::f64::consts::FRAC_1_SQRT_2),
                    _ => unreachable!(),
                })) as f32;
        if expected.is_nan() {
            assert!(actual.is_nan(), "NaN class at {i}");
        } else if expected == 0.0
            || expected.is_infinite()
            || x.abs() == f32::MAX
            || x.abs() <= f32::MIN_POSITIVE
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "exact GELU boundary at {i}"
            );
        } else {
            assert!(
                actual.is_finite() && (f64::from(actual) - f64::from(expected)).abs() <= 1e-6,
                "selected GELU cell {i}: {actual} vs {expected}"
            );
        }
        if x.to_bits() == 1.000244140625_f32.to_bits() {
            assert_ne!(
                actual.to_bits(),
                half::bf16::from_f32(actual).to_f32().to_bits()
            );
        }
    }
    outputs
}
