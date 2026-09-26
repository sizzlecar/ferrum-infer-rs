use super::*;
use crate::backend::cuda::vllm_paged_attn::{
    dispatch_vnext_addressed_paged_attention_batch_raw,
    dispatch_vnext_addressed_paged_attention_raw,
};
use crate::backend::cuda::vnext_ops::{
    cuda_vnext_runtime_config, transformer::test_support::Guarded,
};
use cudarc::driver::{
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
    DevicePtr,
};
use ferrum_interfaces::vnext::DeviceId;
use half::f16;

#[test]
#[ignore = "requires actual CUDA and the pinned addressed-vLLM operator set"]
fn batched_v1_v2_mixed_match_each_owner_and_replay_reads_current_aos_lengths() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.causal-batch").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    // The production lane owns a non-default stream. CUDA forbids graph
    // capture on the legacy default stream, so exercise that same contract.
    let stream = runtime.context().new_stream().unwrap();
    for head_dim in [128_u64, 256] {
        for owners in [4_usize, 8] {
            for case in 0..3 {
                let initial_lengths = (0..owners)
                    .map(|owner| match case {
                        0 => [32_i32, 129, 257, 499][owner % 4],
                        1 => [513_i32, 767, 1025, 1537][owner % 4],
                        _ => [32_i32, 129, 513, 767, 200, 1025, 1537, 300][owner % 8],
                    })
                    .collect::<Vec<_>>();
                let paths = initial_lengths
                    .iter()
                    .map(|&length| {
                        CausalAttentionKernelPath::select(
                            AttentionExecutionPolicy::NativeAdaptive,
                            shape(head_dim),
                            1,
                            length as u64,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let shape = shape(head_dim);
                let binding_layout = BindingLayout::new(shape, owners).unwrap();
                let slot_words = binding_layout.slot_bytes as usize / 8;
                let blocks = 2048_usize / 16;
                let block_elements = 2 * shape.key_value_heads as usize * head_dim as usize * 16;
                // Separate owner allocations, plus a shared read-only prefix block.
                // The native ABI cannot assume a common KV base or consecutive pages.
                let pages = (0..owners)
                    .map(|owner| {
                        let values = (0..blocks * block_elements)
                            .map(|i| {
                                f16::from_f32((((i * 7 + owner * 19) % 97) as f32 - 48.0) / 512.0)
                            })
                            .collect::<Vec<_>>();
                        Guarded::new(&stream, &values, f16::from_f32(77.0))
                    })
                    .collect::<Vec<_>>();
                let queries = (0..owners * shape.query_features as usize)
                    .map(|i| f16::from_f32(((i * 11 % 31) as f32 - 15.0) / 64.0))
                    .collect::<Vec<_>>();
                let query = Guarded::new(&stream, &queries, f16::from_f32(79.0));
                let output = Guarded::new(
                    &stream,
                    &vec![f16::ZERO; queries.len()],
                    f16::from_f32(80.0),
                );
                let expected = Guarded::new(
                    &stream,
                    &vec![f16::ZERO; queries.len()],
                    f16::from_f32(81.0),
                );
                let lengths = Guarded::new(&stream, &vec![0_i32; owners], -777);
                let rows = owners * shape.query_heads as usize * 4;
                let sums = Guarded::new(&stream, &vec![0_f32; rows], 82.0);
                let maxima = Guarded::new(&stream, &vec![0_f32; rows], 83.0);
                let temporary = Guarded::new(
                    &stream,
                    &vec![f16::ZERO; rows * head_dim as usize],
                    f16::from_f32(84.0),
                );
                let serial_rows = shape.query_heads as usize * 4;
                let serial_sums = Guarded::new(&stream, &vec![0_f32; serial_rows], 85.0);
                let serial_maxima = Guarded::new(&stream, &vec![0_f32; serial_rows], 86.0);
                let serial_temporary = Guarded::new(
                    &stream,
                    &vec![f16::ZERO; serial_rows * head_dim as usize],
                    f16::from_f32(87.0),
                );
                let mut binding_words = vec![u64::MAX; owners * slot_words + 16];
                for owner in 0..owners {
                    for block in 0..blocks {
                        let base = if block == 0 {
                            pages[0].pointer(&stream)
                        } else {
                            pages[owner].pointer(&stream)
                        };
                        binding_words[8 + owner * slot_words + 3 + block] =
                            base + (block * block_elements * 2) as u64;
                    }
                }
                let mut binding = stream.clone_htod(&binding_words).unwrap();
                let binding_pointer = binding.device_ptr(&stream).0 + 64;
                let enqueue = || {
                    gather_lengths(
                        &stream,
                        &provider.functions.gather_decode_lengths,
                        binding_pointer,
                        binding_layout.slot_bytes,
                        lengths.pointer(&stream),
                        owners as i32,
                    )
                    .unwrap();
                    for range in group_ranges(&paths) {
                        let row_bytes = range.start as u64 * shape.query_features * 2;
                        let maximum_sequence = if paths[range.start]
                            == CausalAttentionKernelPath::VllmAddressedDecodeV1
                        {
                            512
                        } else {
                            2048
                        };
                        unsafe {
                            dispatch_vnext_addressed_paged_attention_batch_raw(
                                &stream,
                                output.pointer(&stream) + row_bytes,
                                query.pointer(&stream) + row_bytes,
                                binding_pointer
                                    + range.start as u64 * binding_layout.slot_bytes
                                    + 24,
                                lengths.pointer(&stream) + range.start as u64 * 4,
                                maximum_sequence,
                                Some(sums.pointer(&stream)),
                                Some(maxima.pointer(&stream)),
                                Some(temporary.pointer(&stream)),
                                range.len() as i32,
                                shape.query_heads as i32,
                                shape.key_value_heads as i32,
                                head_dim as i32,
                                slot_words as i32,
                            )
                            .unwrap();
                        }
                    }
                };
                // Capture the exact gather + native batch. Only the binding contents
                // change between replays; graph pointers and scratch remain fixed.
                stream
                    .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .unwrap();
                enqueue();
                let graph = stream
                    .end_capture(
                        CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                    )
                    .unwrap()
                    .expect("nonempty batched attention graph");
                for generation in 0..2 {
                    let actual_lengths = initial_lengths
                        .iter()
                        .map(|length| length + generation * 11)
                        .collect::<Vec<_>>();
                    for (owner, length) in actual_lengths.iter().copied().enumerate() {
                        // Control word 3 occupies the upper i32 half of u64 word 1.
                        binding_words[8 + owner * slot_words + 1] = (length as u64) << 32;
                    }
                    stream.memcpy_htod(&binding_words, &mut binding).unwrap();
                    graph.launch().unwrap();
                    for (owner, &length) in actual_lengths.iter().enumerate() {
                        let row_bytes = (owner as u64) * shape.query_features * 2;
                        unsafe {
                            dispatch_vnext_addressed_paged_attention_raw(
                                &stream,
                                expected.pointer(&stream) + row_bytes,
                                query.pointer(&stream) + row_bytes,
                                binding_pointer + owner as u64 * binding_layout.slot_bytes + 24,
                                binding_pointer + owner as u64 * binding_layout.slot_bytes + 12,
                                (length as u64).div_ceil(512) * 512,
                                Some(serial_sums.pointer(&stream)),
                                Some(serial_maxima.pointer(&stream)),
                                Some(serial_temporary.pointer(&stream)),
                                shape.query_heads as i32,
                                shape.key_value_heads as i32,
                                head_dim as i32,
                                (length as u64).div_ceil(512) as i32 * 32,
                            )
                            .unwrap();
                        }
                    }
                    assert_eq!(
                        lengths.read(&stream),
                        actual_lengths,
                        "replay cached old lengths"
                    );
                    let actual = output.read(&stream);
                    let reference = expected.read(&stream);
                    assert!(actual.iter().all(|x| x.is_finite()));
                    assert_eq!(actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "batch differs from per-owner selected math: heads={head_dim}, owners={owners}, generation={generation}");
                    assert_eq!(stream.clone_dtoh(&binding).unwrap(), binding_words);
                    query.assert_unchanged(&stream);
                    for page in &pages {
                        page.assert_unchanged(&stream);
                    }
                    // Scratch need not preserve its contents, but must preserve guards.
                    let _ = (
                        sums.read(&stream),
                        maxima.read(&stream),
                        temporary.read(&stream),
                        serial_sums.read(&stream),
                        serial_maxima.read(&stream),
                        serial_temporary.read(&stream),
                    );
                }
            }
        }
    }
}
