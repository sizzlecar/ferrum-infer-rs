use super::fixture::*;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use half::bf16;
use std::sync::Arc;

const PAIRS: [(u16, u16); 13] = [
    (0x3f80, 0x3b80),
    (0x3f81, 0x3b80),
    (0xbf80, 0xbb80), // BF16 midpoint ties
    (1, 1),
    (0x8001, 1),
    (0x8000, 0x8000),
    (0x8000, 0), // subnormal/cancellation/zeros
    (0x7f7f, 0x7f7f),
    (0xff7f, 0xff7f), // overflow
    (0x7f80, 0xff80),
    (0x7fc1, 0),
    (0x3f80, 0x7f80),
    (0xbf80, 0xff80),
];
fn expected(x: u16, bias: u16) -> u16 {
    let x = f64::from(bf16::from_bits(x).to_f32());
    let b = f64::from(bf16::from_bits(bias).to_f32());
    bf16::from_f32((x + b) as f32).to_bits()
}
fn compare(actual: &[u8], wanted: &[u16]) {
    assert_eq!(actual.len(), wanted.len() * 2);
    for (bytes, &wanted) in actual.chunks_exact(2).zip(wanted) {
        let value = u16::from_le_bytes(bytes.try_into().unwrap());
        if bf16::from_bits(wanted).is_nan() {
            assert!(bf16::from_bits(value).is_nan());
        } else {
            assert_eq!(value, wanted);
        }
    }
}
/// Same fixture on Reference and CUDA. All BF16 source encodings and a tail
/// channel width exercise actual admission/upload/execute/readback/retirement.
pub fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for (rows, columns) in [
        (1u64, 8u64),
        (1, 13),
        (3, 24),
        (257, 257),
        (4097, 16), // All BF16 encodings on the aligned vector path.
        (33, 2048), // Wide rows with several groups and a CTA tail.
    ] {
        let config = Config {
            rows,
            inputs: columns,
            outputs: columns,
            kind: Kind::ChannelBiasAddBf16,
        };
        let compiled = Arc::new(config.compile(runtime.as_ref(), registry, catalog, materializers));
        let account = host();
        let phase = (compiled.clone(), runtime.clone());
        let mut session = compiled
            .initialize_retained(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.channel_bias.init"),
                    request_id: id("request.channel_bias.init"),
                    transaction_id: id("transaction.channel_bias.init"),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
                phase,
            )
            .unwrap_or_else(|e| {
                std::mem::forget(e);
                panic!("channel bias initialization owner retained")
            });
        let mut retained = None;
        for trial in 0..2 {
            let bias = (0..columns as usize)
                .map(|c| PAIRS[(c + trial) % PAIRS.len()].1)
                .collect::<Vec<_>>();
            let x = (0..(rows * columns) as usize)
                .map(|i| {
                    if i < columns as usize {
                        PAIRS[(i + trial) % PAIRS.len()].0
                    } else {
                        (i + trial) as u16
                    }
                })
                .collect::<Vec<_>>();
            let wanted = x
                .iter()
                .enumerate()
                .map(|(i, &v)| expected(v, bias[i % columns as usize]))
                .collect::<Vec<_>>();
            let bytes = |v: &[u16]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            let inputs = vec![
                config.managed(&account, INPUT, &bytes(&x)),
                config.managed(&account, BIAS, &bytes(&bias)),
            ];
            let output = match session.session_mut().execute(
                id("run.channel_bias.execute"),
                id("request.channel_bias.execute"),
                inputs,
            ) {
                Ok(v) => v,
                Err(e) => {
                    std::mem::forget((session, e));
                    panic!("channel bias execution owners retained")
                }
            };
            assert_eq!(output[0].tensor().dimensions(), [rows, columns]);
            compare(output[0].bytes().bytes(), &wanted);
            retained = Some((output, wanted));
        }
        let (_, receipt) = session.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("channel bias close owner retained")
        });
        drop(receipt);
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            (rows * columns * 2) as usize
        );
        let (outputs, wanted) = retained.as_ref().unwrap();
        compare(outputs[0].bytes().bytes(), wanted);
        drop(retained);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    }
}
#[test]
fn channel_bias_oracle_exposes_ties_gradual_underflow_signed_zero_and_overflow() {
    assert_eq!(expected(PAIRS[0].0, PAIRS[0].1), 0x3f80);
    assert_eq!(expected(PAIRS[1].0, PAIRS[1].1), 0x3f82);
    assert_eq!(expected(PAIRS[2].0, PAIRS[2].1), 0xbf80);
    assert_eq!(expected(PAIRS[3].0, PAIRS[3].1), 2);
    assert_eq!(expected(PAIRS[4].0, PAIRS[4].1), 0);
    assert_eq!(expected(PAIRS[5].0, PAIRS[5].1), 0x8000);
    assert_eq!(expected(PAIRS[6].0, PAIRS[6].1), 0);
    assert_eq!(expected(PAIRS[7].0, PAIRS[7].1), 0x7f80);
    assert_eq!(expected(PAIRS[8].0, PAIRS[8].1), 0xff80);
    assert!(bf16::from_bits(expected(PAIRS[9].0, PAIRS[9].1)).is_nan());
}
