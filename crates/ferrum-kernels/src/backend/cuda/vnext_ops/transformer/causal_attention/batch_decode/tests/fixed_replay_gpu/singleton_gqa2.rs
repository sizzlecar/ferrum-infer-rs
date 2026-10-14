//! TEST-only singleton adapter: the ordinary prepare/selected-attention/gate
//! bundle versus paired V2, retaining its original per-partition envelope.
//! No FixedBatchDecode admission, fixed mask gather, or projection is included.
use super::*;
use crate::backend::cuda::vllm_paged_attn::dispatch_vnext_addressed_paged_attention_gqa2_fixed_masked_batch_raw;
use cudarc::driver::sys;

fn singleton_shape(head_dim: u64, maximum_context: u64) -> CausalAttentionShape {
    let mut logical = shape(head_dim);
    logical.hidden_size = 5120;
    logical.query_heads = 24;
    logical.key_value_heads = 4;
    logical.query_features = 24 * head_dim;
    logical.query_projection_features = logical.query_features * 2;
    logical.kv_features = 4 * head_dim;
    logical.output_gate = true;
    logical.maximum_context_tokens = maximum_context;
    logical
}

fn paired_eligible(shape: CausalAttentionShape, launch: CausalAttentionLaunch) -> bool {
    let envelope = launch.replay_topology.envelope();
    launch.tokens == 1
        && launch.path == CausalAttentionKernelPath::VllmAddressedDecodeV2
        && shape.head_dim == 256
        && !shape.int8_kv
        && shape.query_heads % shape.key_value_heads == 0
        && (shape.query_heads / shape.key_value_heads) % 2 == 0
        && envelope.sequence_capacity_tokens > 512
        && envelope.sequence_capacity_tokens <= 2048
}

#[allow(clippy::too_many_arguments)]
fn enqueue_singleton(
    stream: &Arc<CudaStream>,
    functions: &CausalAttentionFunctions,
    layer: &Layer,
    arm: usize,
    launch: CausalAttentionLaunch,
    shape: CausalAttentionShape,
    layout: ScratchLayout,
    row_map: &Guarded<i32>,
) {
    let state = &layer.arms[arm];
    let scratch = state.scratch_pointer(stream);
    let control = scratch_pointer(state.binding_pointer(stream), launch.binding_offset).unwrap();
    let table = scratch_pointer(control, BINDING_CONTROL_BYTES).unwrap();
    let query_raw = scratch_pointer(scratch, launch.packed_query_raw).unwrap();
    let query = scratch_pointer(scratch, launch.packed_query).unwrap();
    let context = scratch_pointer(scratch, launch.packed_context).unwrap();
    let cuda = shape.cuda_shape().unwrap();
    // Same singleton branch as enqueue_packed_causal_attention: no gather and
    // no second inactive attention group is introduced by this adapter.
    launch_prepare(
        stream,
        &functions.prepare,
        query_raw,
        scratch_pointer(scratch, launch.packed_key_raw).unwrap(),
        scratch_pointer(scratch, launch.packed_value_raw).unwrap(),
        layer.query_norm.pointer(stream),
        layer.key_norm.pointer(stream),
        query,
        control,
        table,
        launch,
        cuda,
        i32::from(launch.path.uses_vllm_layout()),
        None,
    )
    .unwrap();
    if arm == 1 && paired_eligible(shape, launch) {
        let partition = layout.vllm.unwrap();
        let envelope = launch.replay_topology.envelope();
        // One original row, mapped to itself. This immutable allocation and
        // the original per-owner table/length remain live through every replay.
        let selected = unsafe {
            dispatch_vnext_addressed_paged_attention_gqa2_fixed_masked_batch_raw(
                stream,
                context,
                query,
                table,
                scratch_pointer(control, BINDING_SEQUENCE_LENGTH_OFFSET).unwrap(),
                row_map.pointer(stream),
                envelope.sequence_capacity_tokens,
                Some(scratch_pointer(scratch, partition.exp_sums).unwrap()),
                Some(scratch_pointer(scratch, partition.max_logits).unwrap()),
                Some(scratch_pointer(scratch, partition.temporary_output).unwrap()),
                1,
                1,
                cuda.query_heads,
                cuda.key_value_heads,
                cuda.head_dim,
                envelope.table_capacity_entries,
                VnextAddressedPagedAttentionKernel::V2,
            )
        }
        .unwrap();
        assert_eq!(selected, VnextAddressedPagedAttentionKernel::V2);
    } else {
        launch_selected_attention(
            stream, functions, query, query_raw, control, table, context, launch, cuda, layout,
            scratch,
        )
        .unwrap();
    }
    if shape.output_gate {
        launch_attention_gate(
            stream,
            &functions.attention_gate,
            context,
            query_raw,
            launch.tokens,
            cuda,
        )
        .unwrap();
    }
}

fn capture_singleton(
    stream: &Arc<CudaStream>,
    functions: &CausalAttentionFunctions,
    layers: &[Layer],
    arm: usize,
    launch: CausalAttentionLaunch,
    shape: CausalAttentionShape,
    layout: ScratchLayout,
    row_map: &Guarded<i32>,
) -> (CudaGraph, usize) {
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    for layer in layers {
        enqueue_singleton(
            stream, functions, layer, arm, launch, shape, layout, row_map,
        );
    }
    let mut state = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id = 0;
    let mut graph = std::ptr::null_mut();
    let mut dependencies = std::ptr::null();
    let mut edge_data = std::ptr::null();
    let mut dependency_count = 0;
    let status = unsafe {
        sys::cuStreamGetCaptureInfo_v3(
            stream.cu_stream(),
            &mut state,
            &mut id,
            &mut graph,
            &mut dependencies,
            &mut edge_data,
            &mut dependency_count,
        )
    };
    assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
    assert_eq!(
        state,
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    );
    let mut count = 0;
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) },
        sys::CUresult::CUDA_SUCCESS
    );
    let mut nodes = vec![std::ptr::null_mut(); count];
    assert_eq!(
        unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) },
        sys::CUresult::CUDA_SUCCESS
    );
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        assert_eq!(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            kind,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL,
            "singleton bundle gained a transfer or other node"
        );
    }
    assert_eq!(
        count as u64,
        layers.len() as u64 * (2 + launch.path.attention_dispatch_count())
    );
    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .expect("nonempty singleton layer bundle");
    (graph, count)
}

fn update_layer(
    stream: &Arc<CudaStream>,
    layer: &mut Layer,
    shape: CausalAttentionShape,
    binding: BindingLayout,
    layout: ScratchLayout,
    request: usize,
    length: u64,
    wave: usize,
) -> Vec<f16> {
    let input = scratch_words(shape, layout, &[request], wave);
    for arm in &mut layer.arms {
        arm.requests[request].grow_to(stream, shape, length);
        let words = arm.binding_words(stream, shape, binding, &[request], &[length]);
        stream.memcpy_htod(&words, &mut arm.binding).unwrap();
        stream.memcpy_htod(&input, &mut arm.scratch).unwrap();
    }
    input
}

fn check_singleton(
    stream: &Arc<CudaStream>,
    layer: &mut Layer,
    shape: CausalAttentionShape,
    binding: BindingLayout,
    layout: ScratchLayout,
    launch: CausalAttentionLaunch,
    request: usize,
    length: u64,
    input: &[f16],
    row_map: &Guarded<i32>,
) -> Vec<u16> {
    let actual = layer
        .arms
        .iter()
        .map(|arm| stream.clone_dtoh(&arm.scratch).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        bits(&actual[0]),
        bits(&actual[1]),
        "singleton complete scratch bits, including every V2 partition"
    );
    let mut writable = vec![false; input.len()];
    let mut permit = |base: u64, bytes: u64| {
        let begin = GUARD_WORDS + base as usize / 2;
        writable[begin..begin + bytes as usize / 2].fill(true);
    };
    for base in [layout.query, layout.context] {
        permit(base, shape.query_features * 2);
        let begin = GUARD_WORDS + base as usize / 2;
        let values = &actual[0][begin..begin + shape.query_features as usize];
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != f16::ZERO));
    }
    if launch.path == CausalAttentionKernelPath::VllmAddressedDecodeV2 {
        let partition = layout.vllm.unwrap();
        let rows = shape.query_heads
            * launch
                .replay_topology
                .envelope()
                .sequence_capacity_tokens
                .div_ceil(512);
        permit(partition.exp_sums, rows * 4);
        permit(partition.max_logits, rows * 4);
        permit(partition.temporary_output, rows * shape.head_dim * 2);
    }
    for words in &actual {
        for (index, (&now, &before)) in words.iter().zip(input).enumerate() {
            if !writable[index] {
                assert_eq!(
                    now.to_bits(),
                    before.to_bits(),
                    "singleton changed raw input, unused scratch or canary at {index}"
                );
            }
        }
    }
    for arm in &layer.arms {
        assert_eq!(
            stream.clone_dtoh(&arm.binding).unwrap(),
            arm.binding_words(stream, shape, binding, &[request], &[length]),
            "binding length/table/guards changed"
        );
    }
    check_pages(&mut layer.arms, stream, shape, &[request], &[length]);
    layer.query_norm.assert_unchanged(stream);
    layer.key_norm.assert_unchanged(stream);
    row_map.assert_unchanged(stream);
    let begin = GUARD_WORDS + layout.query as usize / 2;
    bits(&actual[1][begin..begin + shape.query_features as usize])
}

fn runtime() -> (
    CudaDeviceRuntime,
    CudaCausalPagedAttentionProvider,
    Arc<CudaStream>,
) {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.singleton-gqa2").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    let stream = runtime.context().new_stream().unwrap();
    (runtime, provider, stream)
}

#[test]
#[ignore = "requires CUDA and the production GQA2 archive; singleton bits/KV/replay oracle"]
fn singleton_gqa2_v2_preserves_bits_pages_and_partition_replay_on_cuda() {
    let (_runtime, provider, stream) = runtime();
    let row_map = Guarded::new(&stream, &[0_i32], -73);
    for (head_dim, maximum, first) in [
        (256, 2048, 511),
        (256, 2048, 512),
        (256, 2048, 513),
        (256, 2048, 1024),
        (256, 2048, 1025),
        (256, 2048, 2047),
        (256, 2048, 2048),
        (128, 2048, 513),
        (256, 4096, 2049),
    ] {
        let shape = singleton_shape(head_dim, maximum);
        let binding = BindingLayout::new(shape, 1).unwrap();
        let layout = ScratchLayout::for_participants(
            shape,
            1,
            1,
            CausalProjection::F16,
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap();
        let launch = build_launches(shape, binding, layout, &[first])[0];
        assert_eq!(
            paired_eligible(shape, launch),
            head_dim == 256 && first > 512 && first <= 2048
        );
        let mut layers = vec![layer(&stream, shape, binding, layout, &[first], 0)];
        let graphs = [0, 1].map(|arm| {
            capture_singleton(
                &stream,
                &provider.functions,
                &layers,
                arm,
                launch,
                shape,
                layout,
                &row_map,
            )
        });
        let envelope = launch.replay_topology.envelope();
        let physical_growth = head_dim == 256 && first == 513;
        let grown = if physical_growth {
            let CausalKvLayout::VllmBlocks16 {
                blocks_per_page, ..
            } = shape.kv_layout().unwrap()
            else {
                panic!("singleton fixture requires VLLM physical pages");
            };
            let page_tokens = blocks_per_page * VLLM_BLOCK_TOKENS;
            let grown = first.div_ceil(page_tokens) * page_tokens + 1;
            assert!(grown <= envelope.sequence_capacity_tokens);
            assert!(
                shape.physical_state_bytes(grown).unwrap()
                    > shape.physical_state_bytes(first).unwrap()
            );
            grown
        } else {
            (first + 1).min(envelope.sequence_capacity_tokens)
        };
        let mut previous_query = None;
        for wave in 0..4 {
            let request = usize::from(wave >= 2);
            if wave == 2 {
                for arm in &mut layers[0].arms {
                    let mut pages = RequestPages {
                        pages: Vec::new(),
                        previous: Vec::new(),
                        seed: 97,
                    };
                    pages.grow_to(&stream, shape, first);
                    assert_ne!(
                        pages.pages[0].pointer(&stream),
                        arm.requests[0].pages[0].pointer(&stream)
                    );
                    arm.requests.push(pages);
                }
            }
            let length = if wave % 2 == 0 { first } else { grown };
            let current = build_launches(shape, binding, layout, &[length])[0];
            assert_eq!(current.path, launch.path);
            assert_eq!(
                current.replay_topology, launch.replay_topology,
                "cross-partition changes require a new capture"
            );
            let page_counts = layers[0]
                .arms
                .each_ref()
                .map(|arm| arm.requests[request].pages.len());
            let input = update_layer(
                &stream,
                &mut layers[0],
                shape,
                binding,
                layout,
                request,
                length,
                wave + 1,
            );
            if physical_growth && wave % 2 == 1 {
                for (arm, before) in layers[0].arms.iter().zip(page_counts) {
                    assert!(
                        arm.requests[request].pages.len() > before,
                        "replay did not acquire a new physical KV page"
                    );
                }
            }
            for arm in 0..2 {
                if wave == 0 {
                    enqueue_singleton(
                        &stream,
                        &provider.functions,
                        &layers[0],
                        arm,
                        launch,
                        shape,
                        layout,
                        &row_map,
                    );
                } else {
                    graphs[arm].0.launch().unwrap();
                }
            }
            stream.synchronize().unwrap();
            let query = check_singleton(
                &stream,
                &mut layers[0],
                shape,
                binding,
                layout,
                launch,
                request,
                length,
                &input,
                &row_map,
            );
            if let Some(previous) = previous_query.replace(query.clone()) {
                assert_ne!(query, previous, "replay ignored fresh input");
            }
        }
        drop(graphs);
        println!(
            "{}",
            serde_json::json!({"diagnostic":"singleton_gqa2_correctness","M":1,"head_dim":head_dim,"first_length":first,"replayed_length":grown,"maximum_context":maximum,"envelope_capacity":envelope.sequence_capacity_tokens,"table_capacity":envelope.table_capacity_entries,"paired_v2":paired_eligible(shape,launch),"full_bits_kv_guards":true,"fresh_input_and_new_physical_request":true,"physical_kv_growth":physical_growth,"eager_and_capture":true})
        );
    }
}

#[test]
#[ignore = "requires CUDA and the production GQA2 archive; full singleton sixteen-layer diagnostic"]
fn singleton_gqa2_v2_sixteen_layer_inclusive_pairs_on_cuda() {
    let (_runtime, provider, stream) = runtime();
    let row_map = Guarded::new(&stream, &[0_i32], -73);
    let shape = singleton_shape(256, 2048);
    let binding = BindingLayout::new(shape, 1).unwrap();
    let layout = ScratchLayout::for_participants(
        shape,
        1,
        1,
        CausalProjection::F16,
        AttentionExecutionPolicy::NativeAdaptive,
    )
    .unwrap();
    for length in [512, 513, 1025, 2048] {
        let launch = build_launches(shape, binding, layout, &[length])[0];
        let mut layers = (0..16)
            .map(|seed| layer(&stream, shape, binding, layout, &[length], seed))
            .collect::<Vec<_>>();
        let inputs = layers
            .iter_mut()
            .enumerate()
            .map(|(seed, layer)| {
                update_layer(&stream, layer, shape, binding, layout, 0, length, seed + 41)
            })
            .collect::<Vec<_>>();
        let graphs = [0, 1].map(|arm| {
            capture_singleton(
                &stream,
                &provider.functions,
                &layers,
                arm,
                launch,
                shape,
                layout,
                &row_map,
            )
        });
        for (graph, _) in &graphs {
            for _ in 0..8 {
                graph.launch().unwrap();
            }
        }
        stream.synchronize().unwrap();
        for pair in 0..8 {
            let mut gpu_ms = [0.0; 2];
            let mut submit_wait_ns = [0_u128; 2];
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
                    graphs[arm].0.launch().unwrap();
                }
                let end = stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                end.synchronize().unwrap();
                submit_wait_ns[arm] = started.elapsed().as_nanos() / 4;
                gpu_ms[arm] = f64::from(begin.elapsed_ms(&end).unwrap()) / 4.0;
            }
            println!(
                "{}",
                serde_json::json!({"diagnostic":"singleton_gqa2_sixteen_layer_bundle","M":1,"length":length,"pair":pair,"first_arm":if pair%2==0 {"ordinary"} else {"paired"},"ordinary_gpu_ms":gpu_ms[0],"paired_gpu_ms":gpu_ms[1],"ordinary_submit_wait_ns":submit_wait_ns[0],"paired_submit_wait_ns":submit_wait_ns[1],"layers":16,"paired_v2":paired_eligible(shape,launch),"envelope_capacity":launch.replay_topology.envelope().sequence_capacity_tokens,"graph_nodes":[graphs[0].1,graphs[1].1],"includes_prepare_attention_reduce_gate":true,"excludes_projections_cold_capture_and_binding_upload":true})
            );
        }
        for (layer, input) in layers.iter_mut().zip(inputs) {
            check_singleton(
                &stream, layer, shape, binding, layout, launch, 0, length, &input, &row_map,
            );
        }
        drop(graphs);
    }
}
