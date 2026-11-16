use super::fixture::*;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use half::bf16;
use std::sync::Arc;

/// FP64 products of decoded BF16 operands are exact, including values outside
/// the F32 exponent range. One conversion to F32 then BF16 independently
/// models the final boundary. NaN payloads are not a numerical contract.
fn cases() -> Vec<(u16, u16, u16)> {
    vec![
        (0, 0x8001, 0x0001),      // negative product below F32 -> -0 under FMA
        (0xff7f, 0x7f7f, 0x4000), // F32 separate product overflows; FMA cancels
        (0x3f80, 0x3f81, 0x3f84), // BF16 intermediate would change output
        (0x0000, 0x4000, 0x4000), // raw gate greater than one: no clamp
        (0x3f80, 0x4000, 0xc000), // negative raw gate: no activation
        (0x8000, 0x8000, 0x3f80),
        (0x0001, 0x0001, 0x3f00),
        (0x7f80, 0, 0x3f80),
        (0, 0x7f80, 0),
        (0x7fc1, 0x3f80, 0x3f80),
        (0x3f80, 0xff80, 0x7f80),
        (0xbf80, 0x3fc0, 0x3f40),
    ]
}
fn expected((x, g, u): (u16, u16, u16)) -> u16 {
    let decode = |v| f64::from(bf16::from_bits(v).to_f32());
    bf16::from_f32((decode(g) * decode(u) + decode(x)) as f32).to_bits()
}
pub fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for (rows, width) in [(1, 12), (3, 259)] {
        let config = Config {
            rows,
            inputs: width,
            outputs: width,
            kind: Kind::AddcmulFmaBf16,
        };
        let compiled = config.compile(runtime.as_ref(), registry, catalog, materializers);
        let account = host();
        let mut session = compiled
            .initialize(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id(format!("run.addcmul_fma.init.{width}")),
                    request_id: id(format!("request.addcmul_fma.init.{width}")),
                    transaction_id: id(format!("transaction.addcmul_fma.init.{width}")),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut retained = None;
        for trial in 0..2 {
            let patterns = cases();
            let data: Vec<_> = (0..(rows * width) as usize)
                .map(|i| patterns[(i + trial) % patterns.len()])
                .collect();
            let bytes = |column: usize| {
                data.iter()
                    .flat_map(|&(x, g, u)| [x, g, u][column].to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let output = session
                .execute(
                    id(format!("run.addcmul_fma.{width}.{trial}")),
                    id(format!("request.addcmul_fma.{width}.{trial}")),
                    vec![
                        config.managed(&account, INPUT, &bytes(0)),
                        config.managed(&account, WEIGHT, &bytes(1)),
                        config.managed(&account, RELATIVE, &bytes(2)),
                    ],
                )
                .unwrap();
            for (actual, source) in output[0].bytes().bytes().chunks_exact(2).zip(data) {
                let actual = u16::from_le_bytes(actual.try_into().unwrap());
                let wanted = expected(source);
                if bf16::from_bits(wanted).is_nan() {
                    assert!(bf16::from_bits(actual).is_nan());
                } else {
                    assert_eq!(actual, wanted, "shape {rows}x{width}, input {source:?}");
                }
            }
            retained = Some(output);
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("{:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            (rows * width * 2) as usize
        );
        drop(retained);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    }
}
#[test]
fn independent_oracle_distinguishes_fma_and_unclamped_gate_boundaries() {
    let data = cases();
    assert_eq!(expected(data[0]), 0x8000);
    assert_eq!(expected(data[1]), 0x7f7f);
    assert_eq!(expected(data[3]), bf16::from_f32(4.).to_bits());
    assert_eq!(expected(data[4]), bf16::from_f32(-3.).to_bits());
}
