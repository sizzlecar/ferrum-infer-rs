//! Finite V can overflow an in-place accumulator before a later cancellation.
use super::*;

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorComponentPolicy,
    host: &HostMemoryAccount,
) {
    for kind in [
        MetalTensorOperation::StructuredAttention,
        MetalTensorOperation::StructuredGqaAttention,
    ] {
        let mut case = Case {
            sequence: 32,
            heads: 2,
            kv_heads: if kind == MetalTensorOperation::StructuredAttention {
                2
            } else {
                1
            },
            width: 128,
            segments: vec![(32, true)],
            keys: vec![vec![(32, true)]],
            scale: CanonicalRational::new(1, 8).unwrap(),
        };
        let mut values = [case.heads, case.kv_heads, case.kv_heads]
            .into_iter()
            .map(|heads| vec![0.0_f32; case.sequence * heads * case.width])
            .collect::<Vec<_>>();
        for head in 0..case.kv_heads {
            for (key, sign) in [(0, 1.0_f32), (16, 1.0), (24, -1.0)] {
                for channel in [0, 127] {
                    values[2][(key * case.kv_heads + head) * case.width + channel] =
                        sign * 2_f32.powi(127);
                }
            }
        }

        // Q=K=0 gives scores=0 and exact unit unnormalized weights. All three
        // nonzero V values are finite BF16. The mathematical numerator is
        // 2^127 and the output is exactly 2^122 for 32 visible keys.
        // A separate second-tile PV cancels +2^127 and -2^127 before adding
        // the previous accumulator. Starting that tile's first MMA at O
        // instead can overflow on 2^127+2^127, before the negative term.
        for masked in [false, true] {
            if masked {
                // Remove only the second positive term, keeping the same
                // payload and an exact zero numerator over 31 visible keys.
                case.keys = vec![vec![(16, true), (1, false), (15, true)]];
            }
            run_case(
                runtime,
                registry,
                catalog,
                policy,
                host,
                &case.fixture(kind),
                &values,
                |actual| {
                    case.check(&values, actual);
                    for query in 0..case.sequence {
                        for head in 0..case.heads {
                            for channel in 0..case.width {
                                let expected = if !masked && matches!(channel, 0 | 127) {
                                    2_f32.powi(122)
                                } else {
                                    0.0
                                };
                                let got =
                                    actual[(query * case.heads + head) * case.width + channel];
                                assert_eq!(
                                    got.to_bits(),
                                    expected.to_bits(),
                                    "{kind:?} finite V cancellation q={query} h={head} c={channel}, masked={masked}"
                                );
                            }
                        }
                    }
                },
            );
        }
    }
}
