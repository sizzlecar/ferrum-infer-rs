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
