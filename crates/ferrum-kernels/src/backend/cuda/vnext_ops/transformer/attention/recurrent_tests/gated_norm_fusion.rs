//! Compare the selected epilogue with the unchanged F32-output kernel followed
//! by its original cast. Raw integer carriers preserve NaNs and signed zeros.
use super::*;

#[test]
fn gated_norm_fusion_scope_and_physical_work_follow_each_participant_span() {
    let dense = ProjectionWeightSupport::DenseF16;
    for precision in [AttentionPrecision::F16, AttentionPrecision::CandidateF16] {
        for counts in [&[1][..], &[1, 1], &[2], &[1, 2], &[2, 2]] {
            assert!(fused_gated_norm_for(precision, dense, counts));
        }
        for counts in [&[][..], &[0], &[1, 3], &[65, 1], &[67]] {
            assert!(!fused_gated_norm_for(precision, dense, counts));
        }
    }
    assert!(!fused_gated_norm_for(
        AttentionPrecision::F32Master,
        dense,
        &[1],
    ));
    assert!(!fused_gated_norm_for(
        AttentionPrecision::CandidateF32Master,
        dense,
        &[2],
    ));
    assert!(!fused_gated_norm_for(
        AttentionPrecision::F16,
        ProjectionWeightSupport::LegacyFormats,
        &[1],
    ));
    let weight = SharedProjectionWeight::F16 { region: 0 };
    // Includes the two original dense projections and all recurrent work.
    assert_eq!(
        attention_dispatches_per_launch(&weight, &weight, 4, true).unwrap(),
        9
    );
    assert_eq!(
        attention_dispatches_per_launch(&weight, &weight, 66, false).unwrap(),
        10
    );
}

fn runtime() -> CudaDeviceRuntime {
    CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.gated-norm-fusion").unwrap(),
            AttentionExecutionPolicy::default(),
        )
        .unwrap(),
    )
    .expect("gated norm conformance requires CUDA")
}

fn compare(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    counts: &[u64],
    dim: u64,
    special: bool,
) {
    let tokens = counts.iter().sum::<u64>();
    let heads = 3;
    let n = (tokens * heads * dim) as usize;
    let mut core = (0..n)
        .map(|i| sample(i, 3, 0.03125).to_bits())
        .collect::<Vec<_>>();
    let mut z = (0..n)
        .map(|i| f16::from_f32(sample(i, 5, 0.0625)).to_bits())
        .collect::<Vec<_>>();
    let mut weights = (0..dim as usize)
        .map(|i| (0.5 + sample(i, 7, 0.03125)).to_bits())
        .collect::<Vec<_>>();
    if special {
        let core_cases = [
            0_u32,
            0x8000_0000,
            1,
            0x007f_ffff,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0123,
            0x7f80_0123,
        ];
        let z_cases = [
            0_u16, 0x8000, 1, 0x03ff, 0x7bff, 0x7c00, 0xfc00, 0x7e23, 0x7c01,
        ];
        // Separate head rows exercise finite, overflow and nonfinite reductions.
        for (row, bits) in core_cases.into_iter().enumerate() {
            core[row * dim as usize] = bits;
            z[row * dim as usize] = z_cases[row];
        }
        weights[1] = 0x8000_0000;
        weights[2] = 0x7fc0_0567;
        weights[3] = 0x7f80_0000;
    }
    let core_gpu = Guarded::new(stream, &core, 0x4242_4242_u32);
    let weight_gpu = Guarded::new(stream, &weights, 0x4343_4343_u32);
    let old_z = Guarded::new(stream, &z, 0x5151_u16);
    let selected_z = Guarded::new(stream, &z, 0x5252_u16);
    let old_f32 = Guarded::new(stream, &vec![0x4545_4545_u32; n], 0x4646_4646_u32);
    // Same capacity as the original F32 gated scratch. Its unused F16 half
    // must stay untouched when the fused kernel writes the first n elements.
    const UNUSED: u16 = 0x5656;
    let selected_scratch = Guarded::new(stream, &vec![UNUSED; 2 * n], 0x5757_u16);
    let mut logical = super::super::tests::test_shape();
    logical.value_heads = heads;
    logical.value_head_dim = dim;
    logical.value_features = heads * dim;
    let shape = logical.cuda_shape().unwrap();

    // Independent old path: do not route this reference through the selector.
    launch_gated_norm(
        stream,
        &functions.gated_norm,
        core_gpu.pointer(stream),
        old_z.pointer(stream),
        weight_gpu.pointer(stream),
        old_f32.pointer(stream),
        tokens,
        shape,
    )
    .unwrap();
    launch_cast(
        stream,
        &functions.f32_to_f16,
        old_f32.pointer(stream),
        old_z.pointer(stream),
        n as u64,
    )
    .unwrap();

    let fused = fused_gated_norm_for(
        AttentionPrecision::F16,
        ProjectionWeightSupport::DenseF16,
        counts,
    );
    let output = launch_gated_norm_epilogue(
        stream,
        functions,
        core_gpu.pointer(stream),
        selected_z.pointer(stream),
        weight_gpu.pointer(stream),
        selected_scratch.pointer(stream),
        tokens,
        shape,
        fused,
    )
    .unwrap();
    let expected = old_z.read(stream);
    let scratch = selected_scratch.read(stream);
    let actual = if fused {
        assert_eq!(output, selected_scratch.pointer(stream));
        assert!(scratch[n..].iter().all(|&v| v == UNUSED));
        selected_z.assert_unchanged(stream);
        scratch[..n].to_vec()
    } else {
        assert_eq!(output, selected_z.pointer(stream));
        selected_z.read(stream)
    };
    assert_eq!(
        actual, expected,
        "counts={counts:?}, dim={dim}, special={special}"
    );
    core_gpu.assert_unchanged(stream);
    weight_gpu.assert_unchanged(stream);
    // Also check both guard regions on the original F32 intermediate.
    old_f32.read(stream);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn gated_norm_fusion_cuda_matches_original_bits_for_small_mixed_and_long_spans() {
    let runtime = runtime();
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new_dense_f16(&runtime).unwrap();
    let stream = runtime.context().default_stream();
    for counts in [&[1][..], &[1, 1], &[2], &[1, 2], &[2, 2], &[1, 65], &[67]] {
        for dim in [1, 17, 128, 257] {
            compare(&stream, &provider.functions, counts, dim, false);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn gated_norm_fusion_cuda_preserves_original_nonfinite_and_signed_zero_bits() {
    let runtime = runtime();
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new_dense_f16(&runtime).unwrap();
    let stream = runtime.context().default_stream();
    for counts in [&[1, 2][..], &[1, 3]] {
        compare(&stream, &provider.functions, counts, 128, true);
    }
}

/// A diagnostic, not a latency assertion. This intentionally measures eager
/// launches, with identical persistent owners and no allocation/readback in
/// the timed interval. CUDA event elapsed time may include device idle gaps
/// while the host enqueues; it is not a per-kernel profiler measurement.
#[test]
#[ignore = "requires CUDA; diagnostic timing, no performance pass threshold"]
fn gated_norm_fusion_cuda_alternating_microbenchmark() {
    use cudarc::driver::sys::CUevent_flags;
    use std::time::Instant;
    let runtime = runtime();
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new_dense_f16(&runtime).unwrap();
    let stream = runtime.context().new_stream().unwrap();
    const ITERATIONS: usize = 128;
    let mut shape = super::super::tests::test_shape();
    shape.value_heads = 32;
    shape.value_head_dim = 128;
    shape.value_features = 4096;
    let shape = shape.cuda_shape().unwrap();
    for counts in [&[1][..], &[2], &[1, 1], &[1, 2], &[2, 2]] {
        let tokens = counts.iter().sum::<u64>();
        let n = tokens as usize * 4096;
        let core = Guarded::new(
            &stream,
            &(0..n).map(|i| sample(i, 3, 0.03125)).collect::<Vec<_>>(),
            81.0_f32,
        );
        let input = (0..n)
            .map(|i| f16::from_f32(sample(i, 5, 0.0625)))
            .collect::<Vec<_>>();
        let z = Guarded::new(&stream, &input, f16::from_f32(82.0));
        let weights = Guarded::new(
            &stream,
            &(0..128)
                .map(|i| 0.5 + sample(i, 7, 0.03125))
                .collect::<Vec<_>>(),
            83.0_f32,
        );
        let scratch = Guarded::new(&stream, &vec![0_u16; n * 2], 0x5151_u16);
        let old_f32 = Guarded::new(&stream, &vec![0.0_f32; n], 84.0_f32);
        let old_output = Guarded::new(&stream, &vec![0_u16; n], 0x5252_u16);
        // Resolve pointer guards before timing, like the production command's
        // already-retained regions. All owners outlive every measured event.
        let core_ptr = core.pointer(&stream);
        let z_ptr = z.pointer(&stream);
        let weight_ptr = weights.pointer(&stream);
        let scratch_ptr = scratch.pointer(&stream);
        let old_f32_ptr = old_f32.pointer(&stream);
        let old_output_ptr = old_output.pointer(&stream);
        // The old cast uses a separate destination so every repetition reads
        // the same z, just like successive model waves supply fresh inputs.
        let launch = |fused: bool| {
            if fused {
                launch_gated_norm_epilogue(
                    &stream,
                    &provider.functions,
                    core_ptr,
                    z_ptr,
                    weight_ptr,
                    scratch_ptr,
                    tokens,
                    shape,
                    true,
                )
                .unwrap();
            } else {
                launch_gated_norm(
                    &stream,
                    &provider.functions.gated_norm,
                    core_ptr,
                    z_ptr,
                    weight_ptr,
                    old_f32_ptr,
                    tokens,
                    shape,
                )
                .unwrap();
                launch_cast(
                    &stream,
                    &provider.functions.f32_to_f16,
                    old_f32_ptr,
                    old_output_ptr,
                    n as u64,
                )
                .unwrap();
            }
        };
        for _ in 0..32 {
            launch(false);
            launch(true);
        }
        stream.synchronize().unwrap();
        assert_eq!(&scratch.read(&stream)[..n], &old_output.read(&stream));
        assert!(scratch.read(&stream)[n..].iter().all(|&v| v == 0));
        let mut timing = [[0.0_f64; 2]; 2];
        for pair in 0..10 {
            let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
            for arm in order {
                stream.synchronize().unwrap();
                let start = stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                let enqueue_start = Instant::now();
                for _ in 0..ITERATIONS {
                    launch(arm == 1);
                }
                let host_enqueue_ms = enqueue_start.elapsed().as_secs_f64() * 1000.0;
                let end = stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                end.synchronize().unwrap();
                let device_ms = start.elapsed_ms(&end).unwrap() as f64;
                timing[arm] = [device_ms, host_enqueue_ms];
            }
            println!(
                "{}",
                serde_json::json!({
                    "diagnostic":"gated_norm_f16_epilogue",
                    "counts":counts,"value_heads":32,"head_dimension":128,
                    "pair":pair,"order":if pair % 2 == 0 {"old_fused"} else {"fused_old"},
                    "iterations":ITERATIONS,"eager":true,
                    "old":{"cuda_event_ms":timing[0][0],"host_enqueue_ms":timing[0][1],"kernel_launches_per_iteration":2},
                    "fused":{"cuda_event_ms":timing[1][0],"host_enqueue_ms":timing[1][1],"kernel_launches_per_iteration":1},
                    "timing_scope":"epilogue only; CUDA events include any host-fed device gaps; excludes projections, scan, allocation, validation and readback",
                    "performance_threshold":null
                })
            );
        }
        assert_eq!(&scratch.read(&stream)[..n], &old_output.read(&stream));
        core.assert_unchanged(&stream);
        z.assert_unchanged(&stream);
        weights.assert_unchanged(&stream);
        old_f32.read(&stream);
    }
}
