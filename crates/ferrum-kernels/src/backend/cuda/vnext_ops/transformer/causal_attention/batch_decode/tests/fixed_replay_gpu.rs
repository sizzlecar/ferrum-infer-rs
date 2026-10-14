//! Complete raw-backend replay oracle; physical owners stay live on one stream.
//! Timings cover all sixteen real layer command bundles, not product throughput.
use super::*;
use crate::backend::cuda::vnext_ops::transformer::causal_attention::batch_decode::fixed::FixedBatchDecode;
use cudarc::driver::{sys::CUevent_flags, CudaGraph};

#[cfg(feature = "cuda-causal-gqa2-tests")]
#[path = "fixed_replay_gpu/gqa2.rs"]
mod gqa2;

enum ReplayBatch {
    Indexed(BatchDecode),
    Fixed(FixedBatchDecode),
    #[cfg(feature = "cuda-causal-gqa2-tests")]
    FixedGqa2(FixedBatchDecode),
}

impl ReplayBatch {
    fn test_candidate(self, gqa2: bool) -> Self {
        #[cfg(feature = "cuda-causal-gqa2-tests")]
        if gqa2 {
            return match self {
                Self::Fixed(batch) => Self::FixedGqa2(batch),
                _ => panic!("GQA2 requires the validated fixed batch"),
            };
        }
        assert!(!gqa2, "GQA2 requires its test-only feature");
        self
    }

    fn new(
        fixed: bool,
        launches: &[CausalAttentionLaunch],
        binding: BindingLayout,
        shape: CausalAttentionShape,
        layout: ScratchLayout,
    ) -> Self {
        if fixed {
            Self::Fixed(
                FixedBatchDecode::for_launches(launches, true, binding, shape, layout, true, true)
                    .unwrap()
                    .expect("fixture has packed independent decode rows"),
            )
        } else {
            Self::Indexed(
                BatchDecode::for_launches(
                    launches,
                    true,
                    binding,
                    shape,
                    layout,
                    true,
                    true,
                    CausalDecodeAttentionGrouping::Indexed,
                )
                .unwrap()
                .expect("fixture has indexed decode rows"),
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        functions: &CausalAttentionFunctions,
        launches: &[CausalAttentionLaunch],
        arm: &Arm,
        binding: BindingLayout,
        shape: CausalAttentionShape,
        layout: ScratchLayout,
        query_norm: &Guarded<f16>,
        key_norm: &Guarded<f16>,
    ) {
        let arguments = (
            arm.binding_pointer(stream),
            shape.cuda_shape().unwrap(),
            arm.scratch_pointer(stream),
            query_norm.pointer(stream),
            key_norm.pointer(stream),
        );
        match self {
            Self::Indexed(batch) => batch.enqueue(
                stream,
                functions,
                launches,
                arguments.0,
                binding,
                arguments.1,
                layout,
                arguments.2,
                arguments.3,
                arguments.4,
                shape.output_gate,
            ),
            Self::Fixed(batch) => batch.enqueue(
                stream,
                functions,
                launches,
                arguments.0,
                binding,
                arguments.1,
                layout,
                arguments.2,
                arguments.3,
                arguments.4,
                shape.output_gate,
            ),
            #[cfg(feature = "cuda-causal-gqa2-tests")]
            Self::FixedGqa2(batch) => gqa2::enqueue(
                batch,
                stream,
                functions,
                launches,
                arguments.0,
                binding,
                arguments.1,
                layout,
                arguments.2,
                arguments.3,
                arguments.4,
                shape.output_gate,
            ),
        }
        .unwrap();
    }
}

struct Layer {
    arms: [Arm; 2],
    query_norm: Guarded<f16>,
    key_norm: Guarded<f16>,
}

fn layer(
    stream: &Arc<CudaStream>,
    shape: CausalAttentionShape,
    binding: BindingLayout,
    layout: ScratchLayout,
    lengths: &[u64],
    layer_seed: usize,
) -> Layer {
    let requests = (0..lengths.len()).collect::<Vec<_>>();
    let input = poisoned_input(shape, layout, &requests, layer_seed);
    let arms = [false, true].map(|_| {
        let mut arm = Arm {
            scratch: stream.clone_htod(&input).unwrap(),
            binding: stream
                .clone_htod(&vec![
                    u64::MAX;
                    binding.required_bytes as usize / 8 + 2 * GUARD_WORDS
                ])
                .unwrap(),
            requests: Vec::new(),
        };
        for (request_index, &length) in lengths.iter().enumerate() {
            let mut request = RequestPages {
                pages: Vec::new(),
                previous: Vec::new(),
                seed: layer_seed * 67 + request_index,
            };
            request.grow_to(stream, shape, length);
            arm.requests.push(request);
        }
        arm.share_allocation_with_disjoint_first_pages(stream, 0);
        let words = arm.binding_words(stream, shape, binding, &requests, lengths);
        stream.memcpy_htod(&words, &mut arm.binding).unwrap();
        arm
    });
    Layer {
        arms,
        query_norm: Guarded::new(
            stream,
            &(0..shape.head_dim)
                .map(|i| f16::from_f32(0.75 + (i % 11) as f32 / 32.0))
                .collect::<Vec<_>>(),
            f16::from_f32(74.0),
        ),
        key_norm: Guarded::new(
            stream,
            &vec![f16::ONE; shape.head_dim as usize],
            f16::from_f32(75.0),
        ),
    }
}

fn poisoned_input(
    shape: CausalAttentionShape,
    layout: ScratchLayout,
    requests: &[usize],
    wave: usize,
) -> Vec<f16> {
    let mut input = scratch_words(shape, layout, requests, wave);
    let partition = layout.vllm.unwrap();
    let begin = GUARD_WORDS + partition.sequence_lengths as usize / 2;
    let end = GUARD_WORDS + (partition.row_indices as usize + requests.len() * 8) / 2;
    // Every length and both complete masks must be overwritten, including the
    // inactive rows. NaN poison cannot accidentally act as an original row.
    input[begin..end].fill(f16::from_bits(0x7e35));
    input
}

#[allow(clippy::too_many_arguments)]
fn capture(
    stream: &Arc<CudaStream>,
    functions: &CausalAttentionFunctions,
    layers: &[Layer],
    batch: &ReplayBatch,
    launches: &[CausalAttentionLaunch],
    arm: usize,
    binding: BindingLayout,
    shape: CausalAttentionShape,
    layout: ScratchLayout,
) -> CudaGraph {
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    for layer in layers {
        batch.enqueue(
            stream,
            functions,
            launches,
            &layer.arms[arm],
            binding,
            shape,
            layout,
            &layer.query_norm,
            &layer.key_norm,
        );
    }
    stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .expect("complete nonempty layer bundle")
}

fn read_i32(words: &[f16], base: u64, count: usize) -> Vec<i32> {
    (0..count)
        .map(|i| {
            let offset = GUARD_WORDS + base as usize / 2 + i * 2;
            (u32::from(words[offset].to_bits()) | (u32::from(words[offset + 1].to_bits()) << 16))
                as i32
        })
        .collect()
}

fn check_scratch(
    stream: &Arc<CudaStream>,
    layer: &Layer,
    shape: CausalAttentionShape,
    layout: ScratchLayout,
    input: &[f16],
    lengths: &[u64],
) {
    let actual = layer
        .arms
        .iter()
        .map(|arm| stream.clone_dtoh(&arm.scratch).unwrap())
        .collect::<Vec<_>>();
    for (base, width) in [
        (layout.query, shape.query_features),
        (layout.context, shape.query_features),
    ] {
        let begin = GUARD_WORDS + base as usize / 2;
        let end = begin + lengths.len() * width as usize;
        assert_eq!(
            bits(&actual[0][begin..end]),
            bits(&actual[1][begin..end]),
            "complete current-row query/context bits"
        );
        assert!(actual[0][begin..end].iter().all(|value| value.is_finite()));
        assert!(
            actual[0][begin..end]
                .iter()
                .any(|value| *value != f16::ZERO),
            "attention/preparation left the complete output empty"
        );
    }
    let partition = layout.vllm.unwrap();
    assert_eq!(
        read_i32(&actual[1], partition.sequence_lengths, lengths.len()),
        lengths
            .iter()
            .map(|&length| length as i32)
            .collect::<Vec<_>>()
    );
    for (base, v1) in [
        (partition.row_indices, true),
        (partition.row_indices + lengths.len() as u64 * 4, false),
    ] {
        let expected = lengths
            .iter()
            .enumerate()
            .map(|(row, &length)| {
                if (length <= 512) == v1 {
                    row as i32
                } else {
                    -1
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            read_i32(&actual[1], base, lengths.len()),
            expected,
            "complete fresh fixed mask including inactive rows"
        );
    }
    let padding = GUARD_WORDS + (partition.row_indices as usize + lengths.len() * 8) / 2;
    for words in &actual {
        assert_eq!(&words[..GUARD_WORDS], &input[..GUARD_WORDS]);
        assert_eq!(
            &words[words.len() - GUARD_WORDS..],
            &input[input.len() - GUARD_WORDS..]
        );
        assert_eq!(
            bits(&words[padding..words.len() - GUARD_WORDS]),
            bits(&input[padding..input.len() - GUARD_WORDS]),
            "gather escaped both admitted masks"
        );
        for (base, width) in [
            (layout.query_raw, shape.query_projection_features),
            (layout.key_raw, shape.kv_features),
            (layout.value_raw, shape.kv_features),
        ] {
            let begin = GUARD_WORDS + base as usize / 2;
            let end = begin + lengths.len() * width as usize;
            assert_eq!(
                bits(&words[begin..end]),
                bits(&input[begin..end]),
                "fresh raw projection input modified"
            );
        }
    }
}

fn check_current_pages(
    arms: &mut [Arm; 2],
    stream: &Arc<CudaStream>,
    shape: CausalAttentionShape,
    lengths: &[u64],
) {
    let [reference, candidate] = arms;
    for (row, &length) in lengths.iter().enumerate() {
        let (page, mask) = writable_elements(shape, length - 1);
        let left = &mut reference.requests[row];
        let right = &mut candidate.requests[row];
        let actual = bits(&left.pages[page].read(stream));
        assert_eq!(
            actual,
            bits(&right.pages[page].read(stream)),
            "timed current KV page differs"
        );
        let mut changed = false;
        for (i, (&now, &before)) in actual.iter().zip(&left.previous[page]).enumerate() {
            if mask[i] {
                changed |= now != before;
            } else {
                assert_eq!(
                    now, before,
                    "timed write escaped current token on its physical page"
                );
            }
        }
        assert!(changed, "timed prepare ignored current KV owner");
        left.previous[page] = actual.clone();
        right.previous[page] = actual;
        // These cases never write token zero. Checking its shared allocation
        // also verifies the common allocation's two outer canaries.
        if page != 0 {
            assert_eq!(bits(&left.pages[0].read(stream)), left.previous[0]);
            assert_eq!(bits(&right.pages[0].read(stream)), right.previous[0]);
        }
    }
}

#[test]
#[ignore = "requires CUDA and pinned indexed addressed-vLLM operators"]
fn fixed_capacity_captured_replay_matches_current_indexed_bits_and_guards_on_cuda() {
    captured_capacity_replay(&[
        (128, 8, false, 2048, None),
        (256, 32, true, 2048, None),
        (128, 8, false, 512, None),
    ]);
}

#[test]
#[ignore = "requires CUDA and pinned indexed addressed-vLLM operators"]
fn fixed_execution_capacity_captured_replay_matches_indexed_bits_and_guards_on_cuda() {
    captured_capacity_replay(&[
        (256, 32, true, 262_144, Some(2_048)),
        (128, 8, false, 32_768, Some(4_096)),
    ]);
}

fn captured_capacity_replay(cases: &[(u64, usize, bool, u64, Option<usize>)]) {
    captured_capacity_replay_with_gqa2(cases, false);
}

fn captured_capacity_replay_with_gqa2(
    cases: &[(u64, usize, bool, u64, Option<usize>)],
    gqa2: bool,
) {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.fixed-causal-replay").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    let stream = runtime.context().new_stream().unwrap();
    for &(head_dim, participants, output_gate, maximum_context, execution_limit) in cases {
        let mut shape = shape(head_dim);
        if gqa2 {
            assert_eq!(head_dim, 256);
            shape.hidden_size = 5120;
            shape.query_heads = 24;
            shape.key_value_heads = 4;
            shape.query_features = 24 * 256;
            shape.kv_features = 4 * 256;
        }
        shape.maximum_context_tokens = maximum_context;
        shape.output_gate = output_gate;
        shape.query_projection_features = shape.query_features * if output_gate { 2 } else { 1 };
        shape = shape
            .with_execution_context_limit(
                CausalDecodeAttentionGrouping::FixedExecutionCapacity,
                execution_limit,
            )
            .unwrap();
        let binding = BindingLayout::new(shape, participants).unwrap();
        let layout = ScratchLayout::for_participants(
            shape,
            participants as u64,
            participants,
            CausalProjection::F16,
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap();
        let initial = vec![129; participants];
        let launches = build_launches(shape, binding, layout, &initial);
        let fixed = ReplayBatch::new(true, &launches, binding, shape, layout).test_candidate(gqa2);
        let replay_key = |batch: &ReplayBatch| match batch {
            ReplayBatch::Fixed(batch) => batch
                .bind_replay(CudaCommandReplayKeyBuilder::new("fixture", "fixed-causal"))
                .finish()
                .bind_runtime_payload("causal", std::iter::empty(), &[]),
            ReplayBatch::Indexed(_) => unreachable!("fixed replay-key oracle"),
            #[cfg(feature = "cuda-causal-gqa2-tests")]
            ReplayBatch::FixedGqa2(batch) => batch
                .bind_replay(CudaCommandReplayKeyBuilder::new("fixture", "fixed-causal"))
                .finish()
                .bind_runtime_payload("causal", std::iter::empty(), &[]),
        };
        let initial_key = replay_key(&fixed);
        let mut layers = vec![layer(&stream, shape, binding, layout, &initial, 0)];
        let fixed_graph = capture(
            &stream,
            &provider.functions,
            &layers,
            &fixed,
            &launches,
            1,
            binding,
            shape,
            layout,
        );
        let mut previous_query = None;
        for wave in 0..6 {
            let mut requests = (0..participants).collect::<Vec<_>>();
            if wave >= 3 {
                if wave == 3 {
                    for arm in &mut layers[0].arms {
                        for row in 0..participants {
                            let mut request = RequestPages {
                                pages: Vec::new(),
                                previous: Vec::new(),
                                seed: participants + row,
                            };
                            request.grow_to(&stream, shape, shape.maximum_context_tokens);
                            arm.requests.push(request);
                        }
                        arm.share_allocation_with_disjoint_first_pages(&stream, participants);
                    }
                }
                requests = (0..participants)
                    .map(|row| participants + (row ^ 1))
                    .collect();
            } else if wave == 2 {
                requests.reverse();
            }
            let lengths = (0..participants)
                .map(|row| match wave {
                    0 => 129 + row as u64,
                    1 => 512 + (row % 2) as u64,
                    2 => 1023 + (row % 2) as u64,
                    3 => shape.maximum_context_tokens - (row % 2) as u64,
                    4 if execution_limit.is_some() => {
                        [511, 512, 513, 2047, 2048, 2049, 4095, 4096][row % 8]
                    }
                    4 => [511, 512, 513, 1023, 1024, 1025, 2047, 2048][row % 8],
                    _ => 257 + row as u64,
                })
                .map(|length| length.min(shape.maximum_context_tokens))
                .collect::<Vec<_>>();
            let current = build_launches(shape, binding, layout, &lengths);
            let fresh_fixed = ReplayBatch::new(true, &current, binding, shape, layout);
            assert_eq!(
                initial_key,
                replay_key(&fresh_fixed),
                "fixed identity included current paths, frontier or row permutation"
            );
            let input = poisoned_input(shape, layout, &requests, wave + 1);
            let mut expected = Vec::new();
            for arm in &mut layers[0].arms {
                for (&request, &length) in requests.iter().zip(&lengths) {
                    arm.requests[request].grow_to(&stream, shape, length);
                }
                let words = arm.binding_words(&stream, shape, binding, &requests, &lengths);
                stream.memcpy_htod(&words, &mut arm.binding).unwrap();
                stream.memcpy_htod(&input, &mut arm.scratch).unwrap();
                expected.push(words);
            }
            // Reference recaptures its current host-selected groups. The fixed
            // graph is captured exactly once and sees only fresh device data.
            let reference = ReplayBatch::new(gqa2, &current, binding, shape, layout);
            let reference_graph = capture(
                &stream,
                &provider.functions,
                &layers,
                &reference,
                &current,
                0,
                binding,
                shape,
                layout,
            );
            reference_graph.launch().unwrap();
            fixed_graph.launch().unwrap();
            stream.synchronize().unwrap();
            check_scratch(&stream, &layers[0], shape, layout, &input, &lengths);
            if gqa2 {
                assert_eq!(
                    bits(&stream.clone_dtoh(&layers[0].arms[0].scratch).unwrap()),
                    bits(&stream.clone_dtoh(&layers[0].arms[1].scratch).unwrap()),
                    "GQA2 complete scratch after fresh binding/reorder/growth"
                );
            }
            for (arm, expected) in layers[0].arms.iter().zip(expected) {
                assert_eq!(
                    stream.clone_dtoh(&arm.binding).unwrap(),
                    expected,
                    "current binding and slot canaries modified"
                );
            }
            check_pages(&mut layers[0].arms, &stream, shape, &requests, &lengths);
            layers[0].query_norm.assert_unchanged(&stream);
            layers[0].key_norm.assert_unchanged(&stream);
            let actual = stream.clone_dtoh(&layers[0].arms[1].scratch).unwrap();
            let begin = GUARD_WORDS + layout.query as usize / 2;
            let query = bits(&actual[begin..begin + participants * shape.query_features as usize]);
            if let Some(previous) = previous_query.replace(query.clone()) {
                assert_ne!(previous, query, "captured replay ignored fresh query input");
            }
            drop(reference_graph);
        }
        drop(fixed_graph);
        println!("fixed_capacity_complete_replay head_dim={head_dim} M={participants} model_maximum_context={maximum_context} effective_maximum_context={} execution_limit={execution_limit:?} fixed_captures=1 v1_only_capacity={} current_requests_pages_and_maximum_context=true full_query_context_kv_bits=true masks_complete=true", shape.maximum_context_tokens, shape.maximum_context_tokens <= 512);
    }
}

#[test]
#[ignore = "requires CUDA and pinned indexed addressed-vLLM operators"]
fn fixed_capacity_m8_m32_sixteen_layer_inclusive_paired_on_cuda() {
    sixteen_layer_inclusive_paired(false);
}

fn sixteen_layer_inclusive_paired(gqa2: bool) {
    #[cfg(not(feature = "cuda-causal-gqa2-tests"))]
    assert!(
        !gqa2,
        "GQA2 requires its separately linked test-only native export"
    );
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.fixed-causal-bundle").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    let stream = runtime.context().new_stream().unwrap();
    for participants in [8, 32] {
        let mut shape = shape(256);
        shape.hidden_size = 5120;
        shape.query_heads = 24;
        shape.key_value_heads = 4;
        shape.query_features = 24 * 256;
        shape.query_projection_features = shape.query_features * 2;
        shape.kv_features = 4 * 256;
        shape.output_gate = true;
        let binding = BindingLayout::new(shape, participants).unwrap();
        let layout = ScratchLayout::for_participants(
            shape,
            participants as u64,
            participants,
            CausalProjection::F16,
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap();
        let capacity_lengths = vec![shape.maximum_context_tokens; participants];
        let capacity_launches = build_launches(shape, binding, layout, &capacity_lengths);
        let fixed =
            ReplayBatch::new(true, &capacity_launches, binding, shape, layout).test_candidate(gqa2);
        // Distinct allocations and seed data for sixteen actual causal layers.
        // Max-context pages are allocated once, outside every timing window.
        let mut layers = (0..16)
            .map(|seed| layer(&stream, shape, binding, layout, &capacity_lengths, seed))
            .collect::<Vec<_>>();
        let fixed_graph = capture(
            &stream,
            &provider.functions,
            &layers,
            &fixed,
            &capacity_launches,
            1,
            binding,
            shape,
            layout,
        );
        let requests = (0..participants).collect::<Vec<_>>();
        for case in ["all_v1", "mixed", "all_v2"] {
            let lengths = (0..participants)
                .map(|row| match case {
                    "all_v1" => [129, 257, 511, 512][row % 4],
                    "mixed" => {
                        if row < participants - if participants == 8 { 1 } else { 3 } {
                            [129, 257, 511, 512][row % 4]
                        } else {
                            [767, 1024, 1537][row % 3]
                        }
                    }
                    _ => [513, 767, 1024, 1537, 2048][row % 5],
                })
                .collect::<Vec<_>>();
            let launches = build_launches(shape, binding, layout, &lengths);
            // The GQA2 arm compares against the current fixed implementation,
            // not the older indexed CPU regrouping used by the original test.
            let reference = ReplayBatch::new(gqa2, &launches, binding, shape, layout);
            let wave = match case {
                "all_v1" => 11,
                "mixed" => 23,
                _ => 37,
            };
            for (seed, layer) in layers.iter_mut().enumerate() {
                let input = poisoned_input(shape, layout, &requests, seed + wave);
                for arm in &mut layer.arms {
                    let words = arm.binding_words(&stream, shape, binding, &requests, &lengths);
                    stream.memcpy_htod(&words, &mut arm.binding).unwrap();
                    stream.memcpy_htod(&input, &mut arm.scratch).unwrap();
                }
            }
            let reference_graph = capture(
                &stream,
                &provider.functions,
                &layers,
                &reference,
                &launches,
                0,
                binding,
                shape,
                layout,
            );
            let graphs = [&reference_graph, &fixed_graph];
            for graph in &graphs {
                for _ in 0..8 {
                    graph.launch().unwrap();
                }
            }
            stream.synchronize().unwrap();
            for pair in 0..8 {
                let mut gpu_ms = [0.0; 2];
                let mut host_ns = [0_u128; 2];
                for position in 0..2 {
                    let arm = if pair % 2 == 0 {
                        position
                    } else {
                        1 - position
                    };
                    let started = std::time::Instant::now();
                    let begin = stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    for _ in 0..4 {
                        graphs[arm].launch().unwrap();
                    }
                    let end = stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    end.synchronize().unwrap();
                    host_ns[arm] = started.elapsed().as_nanos() / 4;
                    gpu_ms[arm] = f64::from(begin.elapsed_ms(&end).unwrap()) / 4.0;
                }
                println!("{{\"diagnostic\":\"{}\",\"case\":\"{case}\",\"M\":{participants},\"pair\":{pair},\"first_arm\":\"{}\",\"original_gpu_ms\":{},\"fixed_gpu_ms\":{},\"original_submit_wait_ns\":{},\"fixed_submit_wait_ns\":{},\"layers\":16,\"query_heads\":24,\"kv_heads\":4,\"head_dim\":256,\"maximum_context\":2048,\"gqa2_v2_only\":{gqa2},\"includes_prepare_gather_attention_reduce_gate\":true,\"excludes_cold_capture_and_binding_upload\":true}}", if gqa2 { "gqa2_fixed_sixteen_layer_bundle" } else { "fixed_capacity_sixteen_layer_bundle" }, if pair % 2 == 0 { "original" } else { "fixed" }, gpu_ms[0], gpu_ms[1], host_ns[0], host_ns[1]);
            }
            // Full output/state checks are outside the timed CUDA intervals.
            for (seed, layer) in layers.iter_mut().enumerate() {
                let input = poisoned_input(shape, layout, &requests, seed + wave);
                check_scratch(&stream, layer, shape, layout, &input, &lengths);
                check_current_pages(&mut layer.arms, &stream, shape, &lengths);
                if gqa2 {
                    // Identical fixed masks/scratch layout permit an exact
                    // whole-allocation check, including every active V2
                    // max/exp/tmp output and untouched inactive slot.
                    assert_eq!(
                        bits(&stream.clone_dtoh(&layer.arms[0].scratch).unwrap()),
                        bits(&stream.clone_dtoh(&layer.arms[1].scratch).unwrap()),
                        "GQA2 complete fixed scratch/state bits"
                    );
                }
                for arm in &layer.arms {
                    let expected = arm.binding_words(&stream, shape, binding, &requests, &lengths);
                    assert_eq!(
                        stream.clone_dtoh(&arm.binding).unwrap(),
                        expected,
                        "timed binding and slot canaries modified"
                    );
                }
                layer.query_norm.assert_unchanged(&stream);
                layer.key_norm.assert_unchanged(&stream);
            }
            drop(reference_graph);
        }
        drop(fixed_graph);
    }
}
