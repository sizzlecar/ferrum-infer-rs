use super::fixture::{id, Config, COSINE, GAMMA, INPUT, OUTPUT, SINE};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_runtime::*;
use half::bf16;
use std::{collections::BTreeMap, sync::Arc};

/// Independent F64 sum/sqrt, followed by the declared F32 products/FMA and
/// final BF16. This deliberately does not emulate provider lane reductions
/// or its rsqrt approximation. Predeclared bound: two BF16 ULP on this finite
/// dyadic domain; this does not establish Torch/Triton bitwise equality.
pub fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    c: Config,
) {
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.qk.test"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let component = c.compile(runtime.as_ref(), registry, &catalog, &materializers);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4 << 20,
                transfer_staging_bytes: 4 << 20,
                output_bytes: 4 << 20,
                weight_backing_bytes: 4 << 20,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.qk.init"),
                request_id: id("request.qk.init"),
                transaction_id: id("transaction.qk.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("QK initialization failed; owner retained")
        });
    let (inputs, expected) = data(c);
    let mut retained = None;
    let mut worst = 0;
    for trial in 0..2 {
        let outputs = session
            .execute(
                id(format!("run.qk.{trial}")),
                id(format!("request.qk.{trial}")),
                inputs
                    .iter()
                    .map(|(name, bytes)| c.managed(&host, name, bytes))
                    .collect(),
            )
            .unwrap_or_else(|e| {
                std::mem::forget(e);
                panic!("QK execution failed; owner retained")
            });
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].value_id().as_str(), OUTPUT);
        let actual = outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert!(bf16::from_bits(a).is_finite());
            let error = ordered(a).abs_diff(ordered(e));
            worst = worst.max(error);
            assert!(
                error <= 2,
                "{c:?} index {i}: {a:04x} != {e:04x}; ULP={error}"
            );
        }
        if c.head == 4 {
            assert_eq!(
                actual[0], 0,
                "first-half partner FMA(+0) must retain the declared zero sign"
            );
        }
        if c.head == 2 {
            assert!(
                bf16::from_bits(actual[0]).to_f32() < -0.007,
                "must not round normalized heads before rotation"
            );
        }
        if let Some(previous) = retained.take() {
            let previous: Vec<TensorExecutionOutput> = previous;
            assert_eq!(previous[0].bytes().bytes(), outputs[0].bytes().bytes());
        }
        retained = Some(outputs);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("QK close failed; owner retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    let retained = retained.unwrap();
    assert!(host.owns(retained[0].bytes()));
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    eprintln!("single-round RMS/RoPE {c:?}: max_bf16_ulp={worst}; repeated=2; owner_closed=true");
}
fn ordered(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}
fn data(c: Config) -> (BTreeMap<&'static str, Vec<u8>>, Vec<u16>) {
    let (rows, width, head, rot) = (
        c.rows as usize,
        c.width as usize,
        c.head as usize,
        c.rot as usize,
    );
    let x = (0..rows * width)
        .map(|i| {
            if head == 4 {
                0.0
            } else if head == 2 {
                1.0 + (i % 2) as f32 / 128.
            } else {
                0.5 + (i % 17) as f32 / 8.
            }
        })
        .collect::<Vec<_>>();
    let gamma = (0..head)
        .map(|i| {
            if head == 2 || head == 4 {
                1.0
            } else {
                0.75 + (i % 3) as f32 / 8.
            }
        })
        .collect::<Vec<_>>();
    let cosine = (0..rows * rot)
        .map(|i| {
            if head == 4 {
                -1.0
            } else if head == 2 {
                1.0
            } else {
                0.75 + (i / rot % 2) as f32 / 8.
            }
        })
        .collect::<Vec<_>>();
    let sine = (0..rows * rot)
        .map(|i| {
            if head == 2 || head == 4 {
                1.0
            } else {
                0.125 + (i / rot % 3) as f32 / 16.
            }
        })
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for row in 0..rows {
        for h in 0..width / head {
            let start = row * width + h * head;
            let sum = x[start..start + head]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>();
            let mean = (sum / head as f64) as f32;
            let inverse = (1.0 / ((mean + 1e-5f32) as f64).sqrt()) as f32;
            for j in 0..head {
                let norm = (inverse * x[start + j]) * gamma[j];
                let value = if j < rot {
                    let first = j < rot / 2;
                    let p = if first { j + rot / 2 } else { j - rot / 2 };
                    let v = inverse * x[start + p];
                    let rotated = if first {
                        (-v).mul_add(gamma[p], 0.0)
                    } else {
                        v * gamma[p]
                    };
                    norm.mul_add(cosine[row * rot + j], rotated * sine[row * rot + j])
                } else {
                    norm
                };
                expected.push(bf16::from_f32(value).to_bits());
            }
        }
    }
    (
        BTreeMap::from([
            (INPUT, bytes(&x)),
            (GAMMA, bytes(&gamma)),
            (COSINE, bytes(&cosine)),
            (SINE, bytes(&sine)),
        ]),
        expected,
    )
}
