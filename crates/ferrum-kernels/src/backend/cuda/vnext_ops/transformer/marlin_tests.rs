//! Exercise the vNext wrapper as well as the shared native matrix entrypoint.
use super::test_support::Guarded;
use super::*;
use crate::marlin_repack::{
    fixtures::{build_host_fixture, cpu_reference, FIXTURES, GROUP_SIZE},
    repack_compressed_tensors_zero_points_to_marlin, repack_gptq_to_marlin,
    repack_scales_to_marlin,
};
use cudarc::driver::CudaContext;
use half::f16;

#[path = "../native_blocks/tests/captured.rs"]
mod captured;

#[test]
#[ignore = "requires an actual CUDA device and the Marlin operator artifact"]
fn marlin_runtime_resets_reused_workspace_and_preserves_matrix_guards_on_cuda() {
    let context = CudaContext::new(0).expect("Marlin runtime conformance requires CUDA");
    let stream = context.default_stream();
    let runtime = MarlinProjectionRuntime {
        multiprocessor_count: context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap(),
        device_ordinal: 0,
    };
    let workspace_bytes = runtime.workspace_bytes().unwrap();
    // A valid workspace need not arrive zeroed. The production wrapper owns
    // initialization before every projection that reuses this allocation.
    let workspace = Guarded::new(
        &stream,
        &vec![73_i32; workspace_bytes as usize / 4],
        -117_i32,
    );
    for fixture in FIXTURES {
        // SAFETY: Only the authorized interior is dirtied; the next projection
        // must order its own reset after this queued write on the same stream.
        unsafe {
            cudarc::driver::result::memset_d8_async(
                workspace.pointer(&stream),
                73,
                workspace_bytes as usize,
                stream.cu_stream(),
            )
        }
        .unwrap();
        let host = build_host_fixture(fixture);
        let reference = cpu_reference(fixture, &host);
        let k = fixture.input_features;
        let n = fixture.output_features;
        let packed = repack_gptq_to_marlin(&host.qweight_gptq, k, n);
        let scales = repack_scales_to_marlin(&host.scales_grouped, k, n, GROUP_SIZE);
        let zeros = repack_compressed_tensors_zero_points_to_marlin(
            &host.zero_points_compressed_tensors,
            k / GROUP_SIZE,
            n,
        );
        let x = Guarded::new(&stream, &host.input, f16::from_f32(-117.0));
        let w = Guarded::new(&stream, &packed, 0x5a5a1234_i32);
        let s = Guarded::new(&stream, &scales, f16::from_f32(-117.0));
        let z = Guarded::new(&stream, &zeros, 0x5a5a1234_i32);
        let y = Guarded::new(
            &stream,
            &vec![f16::NAN; fixture.rows * n],
            f16::from_f32(-117.0),
        );
        let invoke = |length| {
            runtime.launch(
                MarlinF16WeightType::U4,
                &stream,
                x.pointer(&stream),
                w.pointer(&stream),
                s.pointer(&stream),
                Some(z.pointer(&stream)),
                y.pointer(&stream),
                workspace.pointer(&stream),
                length,
                fixture.rows as i32,
                n as i32,
                k as i32,
                GROUP_SIZE as i32,
                "Marlin workspace conformance",
            )
        };
        // The undersized range must fail before memset or matrix submission.
        let before = workspace.read(&stream);
        assert!(invoke(workspace_bytes - 1).is_err());
        assert_eq!(workspace.read(&stream), before);
        assert!(y.read(&stream).iter().all(|value| value.is_nan()));
        invoke(workspace_bytes).unwrap();
        let actual = y.read(&stream);
        let mut reference_squared = 0.0_f64;
        let mut error_squared = 0.0_f64;
        for (a, b) in actual.iter().zip(&reference) {
            assert!(
                a.is_finite(),
                "{} produced a non-finite output",
                fixture.name
            );
            reference_squared += b.to_f64().powi(2);
            error_squared += (a.to_f64() - b.to_f64()).powi(2);
        }
        // Identical fixture bytes/reference/bound to the existing raw launcher test.
        let relative_l2 = (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt();
        assert!(
            relative_l2 < 0.05,
            "{}: relative_l2={relative_l2}",
            fixture.name
        );
        workspace.read(&stream);
        x.assert_unchanged(&stream);
        w.assert_unchanged(&stream);
        s.assert_unchanged(&stream);
        z.assert_unchanged(&stream);
    }
}

#[test]
#[ignore = "requires an actual CUDA device and the Marlin operator artifact"]
fn marlin_invocation_workspace_dirty_reuse_and_graph_replay_are_bit_exact_on_cuda() {
    use crate::marlin_repack::fixtures::Fixture;
    use std::sync::Arc;

    struct Matrix {
        shape: Fixture,
        input: Guarded<f16>,
        weight: Guarded<i32>,
        scales: Guarded<f16>,
        zeros: Guarded<i32>,
        output: Guarded<f16>,
    }

    impl Matrix {
        fn new(stream: &Arc<CudaStream>, shape: Fixture) -> Self {
            let host = build_host_fixture(shape);
            let k = shape.input_features;
            let n = shape.output_features;
            Self {
                shape,
                input: Guarded::new(stream, &host.input, f16::from_f32(-117.0)),
                weight: Guarded::new(
                    stream,
                    &repack_gptq_to_marlin(&host.qweight_gptq, k, n),
                    0x5a5a1234,
                ),
                scales: Guarded::new(
                    stream,
                    &repack_scales_to_marlin(&host.scales_grouped, k, n, GROUP_SIZE),
                    f16::from_f32(-117.0),
                ),
                zeros: Guarded::new(
                    stream,
                    &repack_compressed_tensors_zero_points_to_marlin(
                        &host.zero_points_compressed_tensors,
                        k / GROUP_SIZE,
                        n,
                    ),
                    0x5a5a1234,
                ),
                output: Guarded::new(
                    stream,
                    &vec![f16::NAN; shape.rows * n],
                    f16::from_f32(-117.0),
                ),
            }
        }

        fn enqueue(
            &self,
            workspace: &MarlinInvocationWorkspace<'_>,
            stream: &Arc<CudaStream>,
            weight_type: MarlinF16WeightType,
        ) {
            workspace
                .launch(
                    weight_type,
                    self.input.pointer(stream),
                    self.weight.pointer(stream),
                    self.scales.pointer(stream),
                    if weight_type == MarlinF16WeightType::U4 {
                        Some(self.zeros.pointer(stream))
                    } else {
                        None
                    },
                    self.output.pointer(stream),
                    self.shape.rows as i32,
                    self.shape.output_features as i32,
                    self.shape.input_features as i32,
                    GROUP_SIZE as i32,
                    "Marlin invocation workspace conformance",
                )
                .unwrap();
        }

        fn bits(&self, stream: &Arc<CudaStream>) -> Vec<u16> {
            self.input.assert_unchanged(stream);
            self.weight.assert_unchanged(stream);
            self.scales.assert_unchanged(stream);
            self.zeros.assert_unchanged(stream);
            self.output
                .read(stream)
                .into_iter()
                .map(|value| {
                    assert!(value.is_finite());
                    value.to_bits()
                })
                .collect()
        }
    }

    let context = CudaContext::new(0).expect("Marlin workspace conformance requires CUDA");
    let stream = context.new_stream().unwrap();
    let runtime = MarlinProjectionRuntime {
        multiprocessor_count: context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap(),
        device_ordinal: 0,
    };
    let bytes = runtime.workspace_bytes().unwrap();
    // Different M/N/K shapes exercise untouched locks and split-K reductions.
    let matrices = FIXTURES
        .into_iter()
        .chain([
            Fixture {
                name: "decode-split-k",
                rows: 32,
                input_features: 4096,
                output_features: 256,
            },
            Fixture {
                name: "prefill-m-split",
                rows: 65,
                input_features: 512,
                output_features: 128,
            },
        ])
        .map(|shape| Matrix::new(&stream, shape))
        .collect::<Vec<_>>();
    let dirty = |workspace: &Guarded<i32>| {
        unsafe {
            cudarc::driver::result::memset_d8_async(
                workspace.pointer(&stream),
                73,
                bytes as usize,
                stream.cu_stream(),
            )
        }
        .unwrap();
    };
    let check = |workspace: &Guarded<i32>, expected: &[Vec<u16>]| {
        assert!(workspace.read(&stream).iter().all(|lock| *lock == 0));
        for (matrix, bits) in matrices.iter().zip(expected) {
            assert_eq!(matrix.bits(&stream), *bits, "{}", matrix.shape.name);
        }
    };

    for weight_type in [MarlinF16WeightType::U4, MarlinF16WeightType::U4B8] {
        let baseline = Guarded::new(&stream, &vec![73_i32; bytes as usize / 4], -117);
        let expected = matrices
            .iter()
            .map(|matrix| {
                dirty(&baseline);
                // Keep the original public wrapper as the exact-bit oracle.
                runtime
                    .launch(
                        weight_type,
                        &stream,
                        matrix.input.pointer(&stream),
                        matrix.weight.pointer(&stream),
                        matrix.scales.pointer(&stream),
                        if weight_type == MarlinF16WeightType::U4 {
                            Some(matrix.zeros.pointer(&stream))
                        } else {
                            None
                        },
                        matrix.output.pointer(&stream),
                        baseline.pointer(&stream),
                        bytes,
                        matrix.shape.rows as i32,
                        matrix.shape.output_features as i32,
                        matrix.shape.input_features as i32,
                        GROUP_SIZE as i32,
                        "Marlin original per-projection reset oracle",
                    )
                    .unwrap();
                assert!(baseline.read(&stream).iter().all(|lock| *lock == 0));
                matrix.bits(&stream)
            })
            .collect::<Vec<_>>();

        // A new allocation and a newly captured graph must never inherit the
        // previous invocation's host-side initialized flag or scratch contents.
        for _fresh_allocation in 0..2 {
            let workspace = Guarded::new(&stream, &vec![73_i32; bytes as usize / 4], -117);
            let enqueue = || {
                let scope = MarlinInvocationWorkspace {
                    runtime,
                    stream: &stream,
                    pointer: workspace.pointer(&stream),
                    length_bytes: bytes,
                    initialized: std::cell::Cell::new(false),
                    _region: None,
                };
                for matrix in &matrices {
                    matrix.enqueue(&scope, &stream, weight_type);
                }
            };
            for _invocation in 0..2 {
                dirty(&workspace);
                enqueue();
                check(&workspace, &expected);
            }
            let graph = captured::Captured::new(&stream, enqueue);
            for _replay in 0..3 {
                // This write is outside the graph. Every replay must execute
                // the first captured reset before consuming the dirty locks.
                dirty(&workspace);
                graph.launch();
                check(&workspace, &expected);
            }
        }
    }
}
