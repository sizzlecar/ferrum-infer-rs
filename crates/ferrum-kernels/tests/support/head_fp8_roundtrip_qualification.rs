use super::fixture::{id, Config, INDICES, INPUT};
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use half::bf16;
use std::sync::Arc;
fn decode(code: usize) -> f32 {
    let e = code >> 3;
    let m = code & 7;
    if e == 0 {
        m as f32 / 512.
    } else {
        (1. + m as f32 / 8.) * 2f32.powi(e as i32 - 7)
    }
}
fn data(c: Config) -> (Vec<u8>, Vec<u16>) {
    let mut x = vec![0u16; (c.batches * c.heads * c.rows * c.width) as usize];
    for (group, row) in x.chunks_exact_mut(c.width as usize).enumerate() {
        for (i, v) in row.iter_mut().enumerate() {
            *v = match group {
                0 => {
                    if i % 2 == 0 {
                        0
                    } else {
                        0x8000
                    }
                }
                1 => {
                    if i == 0 {
                        1
                    } else {
                        0
                    }
                }
                2 => {
                    if i == 0 {
                        0x7fc1
                    } else {
                        0x3f80
                    }
                }
                3 => {
                    if i == 0 {
                        0x7f80
                    } else {
                        0x3f80
                    }
                }
                _ => {
                    let sf = 2f32.powi((group % 5) as i32 - 2);
                    let value = if i == 0 {
                        448.
                    } else {
                        let code = (group * 13 + i) % 126;
                        if group == 4 {
                            (decode(code) + decode(code + 1)) / 2.
                        } else {
                            decode(code)
                        }
                    };
                    bf16::from_f32(value * sf * if i % 2 == 0 { 1. } else { -1. }).to_bits()
                }
            };
        }
    }
    // Independent exhaustive-code lookup, RN divisions only. The dyadic
    // scale rows avoid division ambiguity; class checks cover exceptional rows.
    let expected: Vec<u16> = x
        .chunks_exact(c.width as usize)
        .flat_map(|row| {
            let mut amax = 0f32;
            for &bits in row {
                let a = bf16::from_bits(bits).to_f32().abs();
                amax = if a.is_nan() || amax.is_nan() {
                    f32::NAN
                } else {
                    amax.max(a)
                };
            }
            let sf = amax / 448.;
            let inv = if sf == 0. { 0. } else { 1. / sf };
            row.iter()
                .map(|&b| {
                    let v = (bf16::from_bits(b).to_f32() * inv).clamp(-448., 448.);
                    if v.is_nan() {
                        return bf16::NAN.to_bits();
                    }
                    let mut best = 0;
                    let mut dist = f64::INFINITY;
                    for code in 0..127 {
                        let delta = (f64::from(v.abs()) - f64::from(decode(code))).abs();
                        if delta < dist || (delta == dist && code % 2 == 0) {
                            best = code;
                            dist = delta;
                        }
                    }
                    bf16::from_f32(decode(best).copysign(v) * sf).to_bits()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let expected = if c.selected_rows.is_some() {
        let columns = (c.width * c.heads) as usize;
        let mut selected = Vec::new();
        for batch in 0..c.batches as usize {
            for row in c.indices() {
                if row < 0 || row as u64 >= c.rows {
                    selected.extend(std::iter::repeat_n(0, columns));
                } else {
                    let start = (batch * c.rows as usize + row as usize) * columns;
                    selected.extend_from_slice(&expected[start..start + columns]);
                }
            }
        }
        selected
    } else {
        expected
    };
    (x.iter().flat_map(|v| v.to_le_bytes()).collect(), expected)
}
/// Exact expected bits on the declared dyadic-scale/code/midpoint domain,
/// signed-zero equality, and exceptional-value classes. This does not certify
/// div.full against RN division for arbitrary dynamic scales.
pub fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    c: Config,
) {
    let m = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let e = EngineProviderDescriptor::new(
        id("provider.engine.head_fp8.test"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = m
        .augment_catalog(registry.capability_catalog(d.clone(), vec![e]).unwrap())
        .unwrap();
    let component = c.compile(runtime.as_ref(), registry, &catalog, &m);
    let host_bytes: usize =
        (c.batches * c.heads * c.rows.max(c.selected_rows.unwrap_or(0)) * c.width * 2 * 8
            + c.selected_rows.unwrap_or(0) * 8
            + 4096)
            .max(1 << 20)
            .try_into()
            .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: host_bytes,
                transfer_staging_bytes: host_bytes,
                output_bytes: host_bytes,
                weight_backing_bytes: host_bytes,
            },
        )
        .unwrap(),
    );
    let mut s = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.head_fp8.init"),
                request_id: id("request.head_fp8.init"),
                transaction_id: id("transaction.head_fp8.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("head FP8 initialization owners retained")
        });
    let (x, expected) = data(c);
    let mut retained = None;
    let mut first = None;
    for _ in 0..2 {
        let mut inputs = vec![c.managed(&host, INPUT, &x)];
        if c.selected_rows.is_some() {
            inputs.push(
                c.managed(
                    &host,
                    INDICES,
                    &c.indices()
                        .iter()
                        .flat_map(|i| i.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            );
        }
        let y = match s.execute(
            id("run.head_fp8.execute"),
            id("request.head_fp8.execute"),
            inputs,
        ) {
            Ok(y) => y,
            Err(e) => {
                let reason = format!("{e:?}");
                std::mem::forget((s, e));
                panic!("head FP8 execution owners retained: {reason}");
            }
        };
        let actual = y[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            if bf16::from_bits(e).is_nan() {
                assert!(bf16::from_bits(a).is_nan(), "NaN at {i}");
            } else {
                assert_eq!(a, e, "head FP8 dyadic/code oracle at {i}");
            }
        }
        if let Some(first) = &first {
            assert_eq!(first, &actual);
        } else {
            first = Some(actual);
        }
        retained = Some(y);
    }
    assert!(matches!(
        s.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("head FP8 close owner retained")
        }),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(host.owns(retained.as_ref().unwrap()[0].bytes()));
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
