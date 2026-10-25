//! Rectangular Head128 dispatch boundaries and exact IEEE accumulation cases.
use super::*;

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorComponentPolicy,
    host: &HostMemoryAccount,
) {
    let run = |case: &CrossCase, values: &[Vec<f32>]| {
        run_case(
            runtime,
            registry,
            catalog,
            policy,
            host,
            &case.fixture(),
            values,
            |actual| oracle::check(case, values, actual),
        );
    };
    // Cross uses its actual Q and KV lengths independently. These four cases
    // straddle the Q32 dispatch boundary and the K16 tile boundary with B2/H2;
    // the existing independent FP64 oracle checks every query/head/channel.
    for query_length in [31, 32] {
        for key_length in [16, 17] {
            let case = case(query_length, key_length, vec![vec![(key_length, true)]; 2]);
            run(&case, &case.values());
        }
    }
    // Batch zero has a leading hole and a poisoned final key in the second
    // tile. Batch one is empty, including poisoned Q, and must remain +0.
    let masked = case(
        32,
        17,
        vec![vec![(1, false), (15, true), (1, false)], vec![(17, false)]],
    );
    let mut values = masked.values();
    values[0][masked.query_length * masked.heads * masked.width..].fill(f32::NAN);
    run(&masked, &values);

    for finite_cancellation in [false, true] {
        let key_length = if finite_cancellation { 32 } else { 17 };
        let mut case = case(33, key_length, vec![vec![(key_length, true)]]);
        let mut values = [case.query_length, case.key_length, case.key_length]
            .into_iter()
            .map(|rows| vec![0.0_f32; rows * case.heads * case.width])
            .collect::<Vec<_>>();
        if finite_cancellation {
            for head in 0..case.heads {
                for (key, sign) in [(0, 1.0_f32), (16, 1.0), (24, -1.0)] {
                    for channel in [0, 127] {
                        values[2][(key * case.heads + head) * case.width + channel] =
                            sign * 2_f32.powi(127);
                    }
                }
            }
        } else {
            for query in 0..case.query_length {
                for head in 0..case.heads {
                    values[0][(query * case.heads + head) * case.width] = 1.0;
                }
            }
            for head in 0..case.heads {
                let key = (16 * case.heads + head) * case.width;
                values[1][key] = -2048.0;
                values[2][key] = f32::INFINITY;
                values[2][key + 127] = f32::NEG_INFINITY;
            }
        }
        for masked in [false, true] {
            if masked {
                case.keys = vec![if finite_cancellation {
                    vec![(16, true), (1, false), (15, true)]
                } else {
                    vec![(16, true), (1, false)]
                }];
            }
            run_case(
                runtime,
                registry,
                catalog,
                policy,
                host,
                &case.fixture(),
                &values,
                |actual| {
                    if finite_cancellation {
                        // All scores are zero. The three finite V terms sum
                        // to 2^127, so 32 unnormalized unit weights give the
                        // exact output 2^122. Masking key16 leaves numerator0.
                        // In-place O can overflow before key24 cancels key16;
                        // the guarded separate-PV branch must preserve it.
                        oracle::check(&case, &values, actual);
                    }
                    assert_eq!(actual.len(), case.query_length * case.heads * case.width);
                    for query in 0..case.query_length {
                        for head in 0..case.heads {
                            for channel in 0..case.width {
                                let got =
                                    actual[(query * case.heads + head) * case.width + channel];
                                let affected = !masked && matches!(channel, 0 | 127);
                                if affected && !finite_cancellation {
                                    // Key16's score is exactly -256. Its FP32
                                    // exp is zero (<2^-150); visible 0*Inf is
                                    // NaN for every query, since Cross is not
                                    // causal. This is outside the moderate-
                                    // score FP64 oracle's declared domain.
                                    assert!(got.is_nan(), "cross visible underflow*Inf q={query} h={head} c={channel}");
                                } else {
                                    let expected = if affected { 2_f32.powi(122) } else { 0.0 };
                                    assert_eq!(got.to_bits(), expected.to_bits(),
                                        "cross exact boundary q={query} h={head} c={channel}, finite_cancellation={finite_cancellation}, masked={masked}");
                                }
                            }
                        }
                    }
                },
            );
        }
    }

    // Different-head rejection is covered by the provider's existing
    // rectangular_geometry_preserves_mask_strides_and_exact_threadgroup_capacity
    // unit test, independently of these real execution cases.
    eprintln!("Metal cross Head128: Q31/32 K16/17, rectangular underflow*Inf/cancellation/masks, repeat and retirement passed");
}

fn case(query_length: usize, key_length: usize, keys: Vec<Vec<(usize, bool)>>) -> CrossCase {
    CrossCase {
        query_length,
        key_length,
        heads: 2,
        width: 128,
        keys,
        scale: CanonicalRational::new(1, 8).unwrap(),
    }
}
