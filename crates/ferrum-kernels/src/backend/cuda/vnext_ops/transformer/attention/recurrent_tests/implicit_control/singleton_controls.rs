//! Actual M1 control-node removal, preserving all six recurrent kernels.
//! This diagnostic excludes projections and the rest of the model.
use super::*;

fn enqueue_explicit(layer: &Layer, stream: &Arc<CudaStream>, functions: &AttentionFunctions) {
    assert_eq!(layer.counts, [1]);
    let arm = &layer.arms[0];
    // The production participant-loop singleton uploads lengths and zeroes
    // indices. Do not use the generic fixture's two-HtoD baseline here.
    put(stream, arm.lengths.pointer(stream), &layer.lengths);
    unsafe {
        cudarc::driver::result::memset_d8_async(
            arm.indices.pointer(stream),
            0,
            layer.indices.len() * std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
    }
    .unwrap();
    layer.enqueue_math(
        stream,
        functions,
        0,
        arm.lengths.pointer(stream),
        arm.indices.pointer(stream),
    );
}

fn check(layers: &[Layer], stream: &Arc<CudaStream>) {
    for layer in layers {
        // Includes complete output/state bits and every guard, plus candidate
        // controls remaining at their original poison values after each replay.
        layer.check(stream);
        assert_eq!(layer.arms[0].lengths.read(stream), layer.lengths);
        assert_eq!(layer.arms[0].indices.read(stream), layer.indices);
    }
}

fn inventory(graph: &Graph) -> serde_json::Value {
    serde_json::json!({
        "nodes":graph.nodes, "kernels":graph.kernels, "copies":graph.copies,
        "memsets":graph.memsets, "edges":graph.edges, "roots":graph.roots,
        "capture_wall_ns":graph.capture_ns,
        "instantiate_wall_ns":graph.instantiate_ns,
        "upload_and_sync_wall_ns":graph.upload_ns, "instantiate_flags":0
    })
}

#[test]
#[ignore = "requires CUDA; actual M1 48-layer explicit versus implicit control graph"]
fn captured_singleton_implicit_controls_graph_pairs() {
    let cold_start = Instant::now();
    let runtime = runtime();
    let stream = runtime.context().new_stream().unwrap();
    let functions = functions(&runtime);
    let layers = (0..48)
        .map(|index| Layer::new(&stream, shape(true), vec![1], index + 1))
        .collect::<Vec<_>>();
    stream.synchronize().unwrap();
    let fixture_setup_ns = cold_start.elapsed().as_nanos();

    // Graphs drop before every captured host allocation, kernel and device
    // region. Graph::drop synchronizes the stream before destroying each graph.
    let graph_setup_start = Instant::now();
    let graphs = [
        Graph::new(&stream, || {
            for layer in &layers {
                enqueue_explicit(layer, &stream, &functions.0);
            }
        }),
        Graph::new(&stream, || {
            for layer in &layers {
                // Existing TEST entrypoints infer seq/start/end and receive
                // null control pointers. Their arithmetic body is unchanged.
                assert_eq!(layer.counts, [1]);
                layer.enqueue(&stream, &functions, 1);
            }
        }),
    ];
    let capture_inventory_instantiate_upload_ns = graph_setup_start.elapsed().as_nanos();
    assert_eq!(graphs[0].copies, layers.len());
    assert_eq!(graphs[0].memsets, layers.len());
    assert_eq!(graphs[1].copies, 0);
    assert_eq!(graphs[1].memsets, 0);
    for graph in &graphs {
        assert_eq!(graph.kernels, layers.len() * 6);
        assert_eq!(graph.roots, 1);
    }
    assert_eq!(graphs[0].nodes - graphs[1].nodes, layers.len() * 2);
    println!(
        "{}",
        serde_json::json!({
            "diagnostic":"gdn_singleton_implicit_cold", "rows":1, "layers":layers.len(),
            "key_heads":layers[0].shape.key_heads, "value_heads":layers[0].shape.value_heads,
            "head_dim":layers[0].shape.key_head_dim, "conv_kernel":layers[0].shape.conv_kernel,
            "fixture_setup_wall_ns":fixture_setup_ns,
            "capture_inventory_instantiate_upload_wall_ns":capture_inventory_instantiate_upload_ns,
            "total_cold_wall_ns":cold_start.elapsed().as_nanos(),
            "explicit":inventory(&graphs[0]), "implicit":inventory(&graphs[1]),
            "scope":"prepare/conv-commit/QK-norm/delta/gated-norm/cast; projections and whole model excluded"
        })
    );

    let mut advances = [0usize; 2];
    for arm in 0..2 {
        graphs[arm].launch();
        advances[arm] += 1;
    }
    check(&layers, &stream);
    // Existing F64 oracle covers the first layer's entire recurrent core and
    // carried state; gated/cast outputs receive full bitwise comparisons.
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
                "diagnostic":"gdn_singleton_implicit_pair", "pair":pair,
                "order":if pair%2==0 {"AB"} else {"BA"},
                "rows":1, "layers":layers.len(), "explicit":measured[0], "implicit":measured[1],
                "state_advances":advances,
                "cold_allocation_capture_upload_input_updates_validation_outside_pair_timing":true,
                "baseline_memcpy_and_memset_inside_graph":true,
                "launch_wall_includes_cpu_clock_reads":true
            })
        );
    }
    check(&layers, &stream);
    for layer in &layers {
        layer.fresh(&stream, true);
    }
    for arm in 0..2 {
        graphs[arm].launch();
        advances[arm] += 1;
    }
    check(&layers, &stream);
    assert_eq!(advances[0], advances[1]);
    stream.synchronize().unwrap();
    println!(
        "{}",
        serde_json::json!({
            "diagnostic":"gdn_singleton_implicit_correctness",
            "all_48_complete_outputs_states_guards_fresh_bits_equal":true,
            "one_actual_layer_complete_f64_core_and_state_first_forward":true,
            "fresh_nonfinite_bits_equal":true, "state_advances":advances,
            "candidate_control_poison_unchanged":true, "projections_excluded":true
        })
    );
}
