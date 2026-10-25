//! IEEE classification outside the moderate-score FP64 error-bound fixtures.
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
            sequence: 33,
            heads: 2,
            kv_heads: if kind == MetalTensorOperation::StructuredAttention {
                2
            } else {
                1
            },
            width: 128,
            segments: vec![(33, false)],
            keys: vec![vec![(33, true)]],
            scale: CanonicalRational::new(1, 8).unwrap(),
        };
        let mut values = [case.heads, case.kv_heads, case.kv_heads]
            .into_iter()
            .map(|heads| vec![0.0_f32; case.sequence * heads * case.width])
            .collect::<Vec<_>>();
        for query in 0..case.sequence {
            for head in 0..case.heads {
                values[0][(query * case.heads + head) * case.width] = 1.0;
            }
        }
        for head in 0..case.kv_heads {
            let key_one = (case.kv_heads + head) * case.width;
            values[1][key_one] = -2048.0;
            values[2][key_one] = f32::INFINITY;
            values[2][key_one + 127] = f32::NEG_INFINITY;
        }

        // All Q/K values and the scale are exact BF16/F32 powers of two.
        // The scores are zero except key 1, whose score is -256. Thus its
        // FP32 exp underflows to zero (exp(-256) < 2^-150). Visible 0*Inf
        // remains NaN in its own output channel; it must not be skipped.
        // The same input payloads are reused with key 1 explicitly masked.
        for masked in [false, true] {
            if masked {
                case.keys = vec![vec![(1, true), (1, false), (31, true)]];
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
                    assert_eq!(actual.len(), case.sequence * case.heads * case.width);
                    for query in 0..case.sequence {
                        for head in 0..case.heads {
                            for channel in 0..case.width {
                                let got =
                                    actual[(query * case.heads + head) * case.width + channel];
                                if !masked && query >= 1 && matches!(channel, 0 | 127) {
                                    assert!(got.is_nan(), "{kind:?} visible underflow*Inf q={query} h={head} c={channel}");
                                } else {
                                    assert_eq!(got.to_bits(), 0, "{kind:?} masked/future Inf must not contaminate q={query} h={head} c={channel}, masked={masked}");
                                }
                            }
                        }
                    }
                },
            );
        }
    }
}
