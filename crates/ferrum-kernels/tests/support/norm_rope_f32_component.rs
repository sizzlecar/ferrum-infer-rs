//! Shared execution/oracles using the one typed numeric component family.
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
    kind: Kind,
) {
    let widths: &[u64] = if matches!(kind, Kind::UnitHeadRmsPartialRopeF32) {
        &[16, 64, 128, 2560]
    } else if matches!(kind, Kind::HeadRmsF32) {
        &[4, 128, 2560]
    } else {
        &[4, 128]
    };
    for &width in widths {
        let c = Config {
            rows: 2,
            inputs: width,
            outputs: 2 * width,
            kind,
        };
        let component = c.compile(runtime.as_ref(), registry, catalog, materializers);
        let account = host();
        let mut session = component
            .initialize(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id(format!("run.norm_rope.init.{width}")),
                    request_id: id(format!("request.norm_rope.init.{width}")),
                    transaction_id: id(format!("transaction.norm_rope.init.{width}")),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        for trial in 0..2 {
            let (x, parameter, sine, expected) = if matches!(kind, Kind::HeadRmsF32) {
                rms_data(width as usize, trial)
            } else if matches!(kind, Kind::UnitHeadRmsPartialRopeF32) {
                unit_partial_data(width as usize, trial)
            } else {
                rope_data(width as usize, trial)
            };
            let bytes = |v: Vec<f32>| v.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
            let x = bytes(x);
            let parameter = bytes(parameter);
            let mut inputs = vec![
                c.managed(&account, INPUT, &x),
                c.managed(
                    &account,
                    if matches!(kind, Kind::HeadRmsF32) {
                        WEIGHT
                    } else {
                        COSINE
                    },
                    &parameter,
                ),
            ];
            if let Some(sine) = sine {
                inputs.push(c.managed(&account, SINE, &bytes(sine)));
            }
            let output = session
                .execute(
                    id(format!("run.norm_rope.{width}.{trial}")),
                    id(format!("request.norm_rope.{width}.{trial}")),
                    inputs,
                )
                .unwrap();
            assert_eq!(output[0].tensor().dimensions(), &[2, 2 * width]);
            assert_eq!(output[0].tensor().element_type(), ElementType::F32);
            let actual: Vec<u32> = output[0]
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(
                actual,
                expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
            );
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|f| panic!("F32 norm/RoPE close failed: {:?}", f.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
fn unit_partial_data(d: usize, trial: usize) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
    let rot = 3 * d / 4;
    let mut x = vec![0.0_f32; 4 * d];
    for head in 0..4 {
        for c in 0..d {
            // Fifteen of each sixteen entries have square 4: mean 3.75,
            // epsilon .25 gives an independently exact inverse of .5.
            x[head * d + c] = if c % 16 == 0 {
                0.0
            } else if (c + head + trial) % 2 == 0 {
                2.0
            } else {
                -2.0
            };
        }
    }
    let cosine: Vec<f32> = (0..2 * rot)
        .map(|i| {
            if i / rot == 0 {
                1.000244140625
            } else {
                0.5001220703125
            }
        })
        .collect();
    let sine: Vec<f32> = (0..2 * rot)
        .map(|i| {
            if i % rot < rot / 2 {
                0.0001220703125
            } else {
                -0.125
            }
        })
        .collect();
    let normalized: Vec<f32> = x
        .chunks_exact(d)
        .flat_map(|head| {
            let mean = head.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / d as f64;
            let inverse = 1.0_f64 / (mean + 0.25).sqrt();
            assert_eq!(inverse, 0.5);
            head.iter()
                .map(|v| (f64::from(*v) * inverse) as f32)
                .collect::<Vec<_>>()
        })
        .collect();
    let expected: Vec<f32> = (0..x.len())
        .map(|i| {
            let c = i % d;
            if c >= rot {
                return normalized[i];
            }
            let frequency = i / (2 * d) * rot + c;
            let partner = if c < rot / 2 {
                -normalized[i + rot / 2]
            } else {
                normalized[i - rot / 2]
            };
            let p = (f64::from(normalized[i]) * f64::from(cosine[frequency])) as f32;
            let q = (f64::from(partner) * f64::from(sine[frequency])) as f32;
            (f64::from(p) + f64::from(q)) as f32
        })
        .collect();
    assert_ne!(expected[d - 1], x[d - 1], "tail must be normalized");
    assert!(expected[..rot]
        .iter()
        .any(|v| *v != half::bf16::from_f32(*v).to_f32()));
    (x, cosine, Some(sine), expected)
}
fn rms_data(d: usize, trial: usize) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
    let mut x = vec![0.0_f32; 4 * d];
    for head in 0..4 {
        for column in 0..3 * d / 4 {
            x[head * d + column] = if trial == 0 {
                if head % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
        }
    }
    let mut gamma = vec![1.000244140625_f32; d];
    gamma[1] = 1.0001220703125;
    // Independent F64 oracle: exactly 3D/4 unit squares gives .75, and the
    // fixture's exact rational epsilon .25 makes sqrt(mean+eps)=1. No
    // reduction-order or root-library tolerance can mask a half conversion.
    let expected: Vec<f32> = x
        .chunks_exact(d)
        .flat_map(|head| {
            let sum = head.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
            let inv = (1.0_f64 / (sum / d as f64 + 0.25).sqrt()) as f32;
            head.iter()
                .zip(&gamma)
                .map(move |(x, g)| ((*x * inv) * g))
                .collect::<Vec<_>>()
        })
        .collect();
    if trial == 0 {
        assert_eq!(expected[0], 1.000244140625_f32);
        assert_ne!(expected[0], half::bf16::from_f32(expected[0]).to_f32());
    }
    (x, gamma, None, expected)
}
fn rope_data(d: usize, trial: usize) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
    let mut x: Vec<f32> = (0..4 * d)
        .map(|i| {
            if trial == 0 {
                (i % 7) as f32 * 0.125 - 0.25
            } else {
                0.0
            }
        })
        .collect();
    let mut cosine = vec![1.0_f32; 2 * d];
    let mut sine = vec![0.0_f32; 2 * d];
    for row in 0..2 {
        for col in 0..d {
            cosine[row * d + col] = if row == 0 { 0.5 } else { 1.5 };
            sine[row * d + col] = if col < d / 2 { 0.25 } else { -0.125 };
        }
    }
    // First output: the noncontracted product rounds away 2^-24 before
    // cancellation. FMA would retain it; neither contract permits that.
    if trial == 0 {
        x[0] = 1.000244140625;
        x[d / 2] = 1.0;
    }
    cosine[0] = 1.000244140625;
    sine[0] = 1.00048828125;
    if trial == 0 {
        x[d] = 1.000244140625;
        x[d + d / 2] = 1.0;
    }
    // Nonrepeating halves and row frequencies also exercise indexing.
    let expected = (0..4 * d)
        .map(|i| {
            let local = i % (2 * d) % d;
            let frequency = i / (2 * d) * d + local;
            let rotated = if local < d / 2 {
                -x[i + d / 2]
            } else {
                x[i - d / 2]
            };
            let p = (f64::from(x[i]) * f64::from(cosine[frequency])) as f32;
            let q = (f64::from(rotated) * f64::from(sine[frequency])) as f32;
            (f64::from(p) + f64::from(q)) as f32
        })
        .collect::<Vec<_>>();
    assert_eq!(expected[0], 0.0);
    if trial == 0 {
        assert_eq!(x[0].mul_add(cosine[0], -sine[0]), 2.0_f32.powi(-24));
        assert!(expected
            .iter()
            .any(|v| *v != half::bf16::from_f32(*v).to_f32()));
    }
    (x, cosine, Some(sine), expected)
}
