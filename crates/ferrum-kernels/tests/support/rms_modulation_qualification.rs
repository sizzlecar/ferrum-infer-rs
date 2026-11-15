use super::fixture::{id, Config, BRANCH, GAMMA, HIDDEN, INPUT, OUTPUT, TABLE};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use half::bf16;
use std::{collections::BTreeMap, sync::Arc};

/// Independent FP64 sum/sqrt oracle on a bounded dyadic domain. Unlike the
/// provider this neither emulates its lane reduction nor calls its libm sqrt.
/// Explicit F32 stores/products/FMA and final BF16 are retained. At most two
/// BF16 ULP are allowed for the different reduction and rsqrt approximation;
/// rounded hidden is exact. This is numerical qualification, not model parity.
pub fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    c: Config,
) {
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.rms_modulation.test"),
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
            2 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 2 << 20,
                transfer_staging_bytes: 2 << 20,
                output_bytes: 2 << 20,
                weight_backing_bytes: 2 << 20,
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
                run_id: id("run.rms_modulation.init"),
                request_id: id("request.rms_modulation.init"),
                transaction_id: id("transaction.rms_modulation.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|failure| {
            std::mem::forget(failure);
            panic!("RMS modulation initialization failed; original owner retained");
        });
    let (inputs, expected) = data(c);
    let mut retained = None;
    let mut worst = 0;
    for trial in 0..2 {
        let outputs = session
            .execute(
                id(format!("run.rms_modulation.{trial}")),
                id(format!("request.rms_modulation.{trial}")),
                inputs
                    .iter()
                    .map(|(name, bytes)| c.managed(&host, name, bytes))
                    .collect(),
            )
            .unwrap_or_else(|failure| {
                std::mem::forget(failure);
                panic!("RMS modulation execution failed; original owner retained");
            });
        for out in &outputs {
            let expected = &expected[out.value_id().as_str()];
            let actual = out
                .bytes()
                .bytes()
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), expected.len());
            for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                assert!(bf16::from_bits(a).is_finite());
                if out.value_id().as_str() == HIDDEN {
                    assert_eq!(a, e, "hidden {index}");
                } else {
                    let error = ordered(a).abs_diff(ordered(e));
                    worst = worst.max(error);
                    assert!(
                        error <= 2,
                        "{:?} output {index}: {a:04x} != {e:04x}, ULP={error}",
                        c
                    );
                }
            }
            if c.residual && c.width == 2 && out.value_id().as_str() == OUTPUT {
                assert!(
                    bf16::from_bits(actual[0]).to_f32() > 0.001,
                    "normalization must use the unrounded residual"
                );
            }
            if !c.residual && c.width == 2 && out.value_id().as_str() == OUTPUT {
                assert!(
                    bf16::from_bits(actual[0]).to_f32() > 0.003,
                    "the modulation factor must retain F32 1+scale"
                );
            }
        }
        retained = Some(outputs);
    }
    let closed = session.close().unwrap_or_else(|failure| {
        std::mem::forget(failure);
        panic!("RMS modulation close failed; original owner retained");
    });
    assert!(matches!(closed, PlanRuntimeCloseOutcome::Closed(_)));
    let retained = retained.unwrap();
    assert!(retained.iter().all(|o| host.owns(o.bytes())));
    assert!(host.usage().unwrap().committed_bytes() > 0);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    eprintln!("fused RMS modulation {c:?}: max_bf16_ulp={worst}; repeated=2; owner_closed=true");
}
fn ordered(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}
fn data(
    c: Config,
) -> (
    BTreeMap<&'static str, Vec<u8>>,
    BTreeMap<&'static str, Vec<u16>>,
) {
    let h = c.width as usize;
    let rows = c.rows as usize;
    let x = (0..rows * h)
        .map(|i| {
            if h == 2 {
                1.0
            } else {
                1.0 + (i % 7) as f32 / 8.
            }
        })
        .collect::<Vec<_>>();
    let branch = (0..rows * h)
        .map(|i| {
            if h == 2 {
                1.0
            } else {
                ((i % 5) as f32 - 2.) / 8.
            }
        })
        .collect::<Vec<_>>();
    let gamma = (0..h)
        .map(|i| {
            if h == 2 {
                1.0
            } else {
                1.0 + (i % 3) as f32 / 8.
            }
        })
        .collect::<Vec<_>>();
    let mut table = vec![0f32; 3 * 6 * h];
    for p in 0..3 {
        for j in 0..h {
            table[p * 6 * h + j] = if h == 2 {
                -1.0
            } else {
                ((p + j) % 3) as f32 / 16.
            };
            table[p * 6 * h + h + j] = if h == 2 {
                if !c.residual && j == 0 {
                    1. / 256.
                } else {
                    0.0
                }
            } else {
                ((p + j) % 5) as f32 / 16. - 0.125
            };
            table[p * 6 * h + 2 * h + j] = if h == 2 {
                if j == 0 {
                    1. / 256.
                } else {
                    0.0
                }
            } else {
                0.125
            };
        }
    }
    let mut output = Vec::new();
    let mut hidden = Vec::new();
    for r in 0..rows {
        let p = if r == 0 { 2 } else { r % 3 };
        let z = (0..h)
            .map(|j| {
                if c.residual {
                    ((table[p * 6 * h + 2 * h + j] as f64) * (branch[r * h + j] as f64)
                        + x[r * h + j] as f64) as f32
                } else {
                    x[r * h + j]
                }
            })
            .collect::<Vec<_>>();
        let mean = z.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / (h as f64);
        let inverse = (1.0 / (mean + (1e-5f32 as f64)).sqrt()) as f32;
        for j in 0..h {
            hidden.push(bf16::from_f32(z[j]).to_bits());
            let normalized = (z[j] as f64 * inverse as f64) as f32;
            let weighted = (normalized as f64 * gamma[j] as f64) as f32;
            let factor = (1.0f64 + table[p * 6 * h + h + j] as f64) as f32;
            let y = (weighted as f64 * factor as f64 + table[p * 6 * h + j] as f64) as f32;
            output.push(bf16::from_f32(y).to_bits());
        }
    }
    let bytes = |v: Vec<f32>| {
        v.into_iter()
            .flat_map(|x| bf16::from_f32(x).to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut inputs = BTreeMap::from([
        (INPUT, bytes(x)),
        (TABLE, bytes(table)),
        (GAMMA, bytes(gamma)),
    ]);
    let mut expected = BTreeMap::from([(OUTPUT, output)]);
    if c.residual {
        inputs.insert(BRANCH, bytes(branch));
        expected.insert(HIDDEN, hidden);
    }
    (inputs, expected)
}
