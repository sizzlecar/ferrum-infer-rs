//! Same captured M1 work, changing only cu_seqlens host allocation ownership.
//! Projections are excluded; this is not a whole-model performance benchmark.
use super::*;
use cudarc::driver::PinnedHostSlice;

fn enqueue_singleton(
    layer: &Layer,
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    arm: usize,
    lengths: &[u32],
) {
    assert_eq!(layer.counts, [1]);
    assert_eq!(lengths, layer.lengths);
    let buffers = &layer.arms[arm];
    // Match production's participant-loop M1: one HtoD and one zero memset,
    // unlike the older generic implicit-control fixture's two HtoD baseline.
    put(stream, buffers.lengths.pointer(stream), lengths);
    unsafe {
        cudarc::driver::result::memset_d8_async(
            buffers.indices.pointer(stream),
            0,
            layer.indices.len() * std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
    }
    .unwrap();
    layer.enqueue_math(
        stream,
        functions,
        arm,
        buffers.lengths.pointer(stream),
        buffers.indices.pointer(stream),
    );
}

fn check(layers: &[Layer], stream: &Arc<CudaStream>) {
    for layer in layers {
        layer.check_controls(stream, false);
        for arm in &layer.arms {
            assert_eq!(arm.lengths.read(stream), layer.lengths);
            assert_eq!(arm.indices.read(stream), layer.indices);
        }
    }
}

fn inventory(graph: &Graph) -> serde_json::Value {
    serde_json::json!({
        "nodes":graph.nodes, "kernels":graph.kernels, "copies":graph.copies,
        "memsets":graph.memsets, "edges":graph.edges, "roots":graph.roots,
        "capture_wall_ns":graph.capture_ns,
        "instantiate_wall_ns":graph.instantiate_ns,
        "upload_and_sync_wall_ns":graph.upload_ns,
        "instantiate_flags":0
    })
}

#[test]
#[ignore = "requires CUDA; actual M1 48-layer pageable versus pinned control graph"]
fn captured_singleton_control_pinned_host_graph_pairs() {
    let cold_start = Instant::now();
    let runtime = runtime();
    let stream = runtime.context().new_stream().unwrap();
    let functions = CudaGatedDeltaRecurrentAttentionProvider::new(&runtime)
        .unwrap()
        .functions;
    let layers = (0..48)
        .map(|index| Layer::new(&stream, shape(true), vec![1], index + 1))
        .collect::<Vec<_>>();
    stream.synchronize().unwrap();
    let fixture_setup_ns = cold_start.elapsed().as_nanos();

    let pin_start = Instant::now();
    let pinned = layers
        .iter()
        .map(|layer| {
            let mut allocation: PinnedHostSlice<u32> =
                unsafe { runtime.context().alloc_pinned(layer.lengths.len()) }.unwrap();
            allocation
                .as_mut_slice()
                .unwrap()
                .copy_from_slice(&layer.lengths);
            allocation
        })
        .collect::<Vec<_>>();
    let pin_allocate_and_fill_ns = pin_start.elapsed().as_nanos();
    // Borrow before capture: PinnedHostSlice access may synchronize its event.
    // These immutable owners and slices outlive both executable graphs.
    let pinned_slices = pinned
        .iter()
        .map(|allocation| allocation.as_slice().unwrap())
        .collect::<Vec<_>>();
    for (layer, data) in layers.iter().zip(&pinned_slices) {
        assert_eq!(*data, layer.lengths);
        let mut flags = 0;
        unsafe {
            ok(sys::cuMemHostGetFlags(
                &mut flags,
                data.as_ptr().cast_mut().cast(),
            ));
        }
    }

    // Declared last, dropped first (also during unwind). Graph::drop waits for
    // the stream before destroying graphs; host pins and device state stay live.
    let graph_setup_start = Instant::now();
    let graphs = [
        Graph::new(&stream, || {
            for layer in &layers {
                enqueue_singleton(layer, &stream, &functions, 0, &layer.lengths);
            }
        }),
        Graph::new(&stream, || {
            for (layer, lengths) in layers.iter().zip(&pinned_slices) {
                enqueue_singleton(layer, &stream, &functions, 1, lengths);
            }
        }),
    ];
    let capture_inventory_instantiate_upload_ns = graph_setup_start.elapsed().as_nanos();
    for graph in &graphs {
        assert_eq!(graph.copies, layers.len());
        assert_eq!(graph.memsets, layers.len());
        assert_eq!(graph.roots, 1);
    }
    assert_eq!(graphs[0].nodes, graphs[1].nodes);
    assert_eq!(graphs[0].kernels, graphs[1].kernels);
    assert_eq!(graphs[0].edges, graphs[1].edges);
    println!(
        "{}",
        serde_json::json!({
            "diagnostic":"gdn_singleton_pinned_cold", "rows":1, "layers":layers.len(),
            "key_heads":layers[0].shape.key_heads, "value_heads":layers[0].shape.value_heads,
            "head_dim":layers[0].shape.key_head_dim, "conv_kernel":layers[0].shape.conv_kernel,
            "pinned_bytes":pinned_slices.iter().map(|s| std::mem::size_of_val(*s)).sum::<usize>(),
            "fixture_setup_wall_ns":fixture_setup_ns,
            "pin_allocate_and_fill_wall_ns":pin_allocate_and_fill_ns,
            "capture_inventory_instantiate_upload_wall_ns":capture_inventory_instantiate_upload_ns,
            "total_cold_wall_ns":cold_start.elapsed().as_nanos(),
            "pageable":inventory(&graphs[0]), "pinned":inventory(&graphs[1]),
            "all_pinned_pointers_cuda_host_flags_query_succeeded":true,
            "scope":"prepare/conv-commit/QK-norm/delta/gated-norm/cast; projections and whole model excluded"
        })
    );

    let mut advances = [0usize; 2];
    for arm in 0..2 {
        graphs[arm].launch();
        advances[arm] += 1;
    }
    check(&layers, &stream);
    // One entire actual-shape layer has the existing complete F64 oracle;
    // every layer has full bitwise output/state comparisons and guard checks.
    layers[0].oracle_first(&stream);
    let before_fresh_core = layers[0].arms[0].core.read(&stream);
    let before_fresh_state = layers[0].arms[0].state.read(&stream);
    for layer in &layers {
        assert!(layer.arms[0]
            .core
            .read(&stream)
            .iter()
            .all(|v| v.is_finite()));
        layer.fresh(&stream, false);
    }
    for arm in 0..2 {
        graphs[arm].launch();
        advances[arm] += 1;
    }
    check(&layers, &stream);
    assert_ne!(before_fresh_core, layers[0].arms[0].core.read(&stream));
    assert_ne!(before_fresh_state, layers[0].arms[0].state.read(&stream));

    for pair in 0..8 {
        let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
        let mut measured = [serde_json::Value::Null, serde_json::Value::Null];
        for arm in order {
            measured[arm] = measure(&graphs[arm]);
            advances[arm] += 1;
        }
        assert_eq!(advances[0], advances[1]);
        println!(
            "{}",
            serde_json::json!({
                "diagnostic":"gdn_singleton_pinned_pair", "pair":pair,
                "order":if pair%2==0 {"AB"} else {"BA"},
                "rows":1, "layers":layers.len(), "pageable":measured[0], "pinned":measured[1],
                "state_advances":advances,
                "cold_allocation_capture_upload_input_updates_validation_outside_pair_timing":true,
                "same_memcpy_memset_kernel_nodes":true,
                "launch_wall_includes_cpu_clock_reads":true
            })
        );
    }
    check(&layers, &stream);
    // Nonfinite fresh inputs run after timing, so they cannot alter its work.
    for layer in &layers {
        layer.fresh(&stream, true);
    }
    for arm in 0..2 {
        graphs[arm].launch();
        advances[arm] += 1;
    }
    check(&layers, &stream);
    assert_eq!(advances[0], advances[1]);
    for (layer, data) in layers.iter().zip(&pinned_slices) {
        assert_eq!(*data, layer.lengths);
    }
    stream.synchronize().unwrap();
    println!(
        "{}",
        serde_json::json!({
            "diagnostic":"gdn_singleton_pinned_correctness",
            "all_48_complete_outputs_states_guards_fresh_bits_equal":true,
            "one_actual_layer_complete_f64_first_forward":true,
            "fresh_nonfinite_bits_equal":true, "state_advances":advances,
            "pinned_host_contents_unchanged":true, "projections_excluded":true
        })
    );
}
