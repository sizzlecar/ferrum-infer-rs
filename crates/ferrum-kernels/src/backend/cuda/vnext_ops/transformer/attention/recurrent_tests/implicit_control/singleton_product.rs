//! Normal product symbols and the whole-invocation control selector.
use super::*;

fn enqueue(
    layer: &Layer,
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    arm_index: usize,
    control: AttentionControl,
) {
    let arm = &layer.arms[arm_index];
    let (lengths, indices) = match control {
        AttentionControl::Explicit => {
            put(stream, arm.lengths.pointer(stream), &layer.lengths);
            if layer.counts.len() == 1 {
                // Match the product participant-loop path, including M1's
                // lengths HtoD and zero memset instead of a second HtoD.
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        arm.indices.pointer(stream),
                        0,
                        layer.indices.len() * std::mem::size_of::<u32>(),
                        stream.cu_stream(),
                    )
                }
                .unwrap();
            } else {
                put(stream, arm.indices.pointer(stream), &layer.indices);
            }
            (arm.lengths.pointer(stream), arm.indices.pointer(stream))
        }
        AttentionControl::ImplicitSingleton => (0, 0),
    };
    // Uses normal product functions and launch_delta's actual typed selector;
    // none of the ferrum_test_* exports are loaded or substituted here.
    layer.enqueue_math_with_control(stream, functions, arm_index, lengths, indices, control);
}

#[test]
#[ignore = "requires CUDA; normal singleton symbols, fresh state, and explicit fallbacks"]
fn singleton_product_controls_preserve_bits_and_fallback() {
    let runtime = runtime();
    let stream = runtime.context().new_stream().unwrap();
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new(&runtime).unwrap();
    let functions = &provider.functions;
    // Keep all owners alive: the third invocation has independently allocated
    // physical state while exercising M1 -> multi-token -> M1 selection.
    let layers = [vec![1], vec![2], vec![1], vec![1, 1]]
        .into_iter()
        .enumerate()
        .map(|(i, counts)| Layer::new(&stream, shape(true), counts, i + 1))
        .collect::<Vec<_>>();
    for arm in 0..2 {
        assert_ne!(
            layers[0].arms[arm].state.pointer(&stream),
            layers[2].arms[arm].state.pointer(&stream)
        );
        assert_ne!(
            layers[0].arms[arm].conv.pointer(&stream),
            layers[2].arms[arm].conv.pointer(&stream)
        );
        assert_ne!(
            layers[0].arms[arm].binding.pointer(&stream),
            layers[2].arms[arm].binding.pointer(&stream)
        );
    }
    for (case, layer) in layers.iter().enumerate() {
        let selected = AttentionControl::for_invocation(
            layer.shape,
            layer.counts.len(),
            layer.indices.len() as u64,
        );
        let implicit = matches!(case, 0 | 2);
        assert_eq!(selected == AttentionControl::ImplicitSingleton, implicit);
        let initial_state = layer.arms[0].state.read(&stream);
        let graphs = [
            Graph::new(&stream, || {
                enqueue(layer, &stream, functions, 0, AttentionControl::Explicit)
            }),
            Graph::new(&stream, || enqueue(layer, &stream, functions, 1, selected)),
        ];
        for graph in &graphs {
            assert_eq!(graph.kernels, 6);
            assert_eq!(graph.roots, 1);
        }
        assert_eq!(graphs[0].copies + graphs[0].memsets, 2);
        assert_eq!(
            graphs[0].copies,
            if layer.counts.len() == 1 { 1 } else { 2 }
        );
        assert_eq!(
            graphs[1].copies + graphs[1].memsets,
            selected.transfers() as usize
        );
        assert_eq!(graphs[1].nodes, 6 + selected.transfers() as usize);
        for graph in &graphs {
            graph.launch();
        }
        layer.check_controls(&stream, implicit);
        layer.oracle_first(&stream);
        assert_ne!(initial_state, layer.arms[0].state.read(&stream));
        let before_fresh_core = layer.arms[0].core.read(&stream);
        let before_fresh_state = layer.arms[0].state.read(&stream);
        layer.fresh(&stream, false);
        for graph in &graphs {
            graph.launch();
        }
        layer.check_controls(&stream, implicit);
        assert_ne!(before_fresh_core, layer.arms[0].core.read(&stream));
        assert_ne!(before_fresh_state, layer.arms[0].state.read(&stream));
        layer.fresh(&stream, true);
        for graph in &graphs {
            graph.launch();
        }
        layer.check_controls(&stream, implicit);
        for arm in if implicit { 0..1 } else { 0..2 } {
            assert_eq!(layer.arms[arm].lengths.read(&stream), layer.lengths);
            assert_eq!(layer.arms[arm].indices.read(&stream), layer.indices);
        }
        println!(
            "{}",
            serde_json::json!({
                "diagnostic":"gdn_singleton_product_correctness", "case":case,
                "participants":layer.counts.len(), "tokens":layer.indices.len(),
                "selected":format!("{selected:?}"), "normal_symbols":true,
                "complete_bits_states_guards_fresh_nonfinite_equal":true,
                "first_forward_complete_f64_core_and_state":true,
                "implicit_control_poison_unchanged":implicit,
                "state_advances_per_arm":3,
                "baseline_control_nodes":graphs[0].copies+graphs[0].memsets,
                "selected_control_nodes":graphs[1].copies+graphs[1].memsets,
                "scope":"recurrent core; model executor run/serve coverage is separate"
            })
        );
    }
    stream.synchronize().unwrap();
}
