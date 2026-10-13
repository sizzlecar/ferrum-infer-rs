//! Same-node-count control: only row flag memset becomes a standalone kernel.
//! Production and the original direct-F16 comparison remain unchanged.
use super::*;
use std::collections::HashMap;

unsafe extern "C" {
    fn ferrum_test_upstream_mmq_pack_f16_direct_kernel_zero_v1(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn cudaGraphNodeGetType(node: *mut c_void, kind: *mut i32) -> i32;
    fn cudaGraphGetEdges_v2(
        graph: *mut c_void,
        from: *mut *mut c_void,
        to: *mut *mut c_void,
        edge_data: *mut c_void,
        count: *mut usize,
    ) -> i32;
}

struct Topology {
    nodes: usize,
    edges: usize,
    // Reconstructed from the actual dependencies, not GetNodes array order.
    chain_types: Vec<i32>,
}

impl Topology {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "nodes":self.nodes,"edges":self.edges,"single_dependency_chain":true,
            "kernel_nodes":self.chain_types.iter().filter(|&&t|t==0).count(),
            "memset_nodes":self.chain_types.iter().filter(|&&t|t==2).count(),
            "other_nodes":self.chain_types.iter().filter(|&&t|t!=0 && t!=2).count(),
            "inspection":"cudaGraphGetNodes, cudaGraphNodeGetType, cudaGraphGetEdges_v2 with null edgeData; default dependencies required"})
    }
}

fn capture_topology(stream: &Stream, enqueue: impl FnOnce()) -> (Graph, Topology) {
    unsafe {
        ok(cudaStreamBeginCapture(stream.0, 0));
        enqueue();
        let mut graph = ptr::null_mut();
        ok(cudaStreamEndCapture(stream.0, &mut graph));
        let mut count = 0;
        ok(cudaGraphGetNodes(graph, ptr::null_mut(), &mut count));
        assert!(count > 0);
        let mut nodes = vec![ptr::null_mut(); count];
        ok(cudaGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count));
        assert_eq!(nodes.len(), count);
        let index: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, p)| (*p as usize, i))
            .collect();
        assert_eq!(
            index.len(),
            count,
            "graph returned duplicate node identities"
        );
        let mut types = vec![0; count];
        for (&node, kind) in nodes.iter().zip(&mut types) {
            ok(cudaGraphNodeGetType(node, kind));
        }
        let mut edges = 0;
        // Bind the versioned five-argument ABI explicitly. A null edgeData
        // requires default dependencies; a lossy-query error fails this check.
        ok(cudaGraphGetEdges_v2(
            graph,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut edges,
        ));
        let mut from = vec![ptr::null_mut(); edges];
        let mut to = vec![ptr::null_mut(); edges];
        ok(cudaGraphGetEdges_v2(
            graph,
            from.as_mut_ptr(),
            to.as_mut_ptr(),
            ptr::null_mut(),
            &mut edges,
        ));
        assert_eq!(from.len(), edges);
        assert_eq!(
            edges + 1,
            count,
            "captured route must be one dependency chain"
        );
        let mut predecessors = vec![0; count];
        let mut next = vec![None; count];
        for (source, target) in from.into_iter().zip(to) {
            let source = index[&(source as usize)];
            let target = index[&(target as usize)];
            assert_ne!(source, target);
            assert!(
                next[source].replace(target).is_none(),
                "branching dependency"
            );
            predecessors[target] += 1;
            assert_eq!(predecessors[target], 1, "multiple predecessors");
        }
        let roots: Vec<_> = (0..count).filter(|&i| predecessors[i] == 0).collect();
        assert_eq!(roots.len(), 1);
        let mut visited = vec![false; count];
        let mut chain_types = Vec::with_capacity(count);
        let mut current = Some(roots[0]);
        while let Some(node) = current {
            assert!(!visited[node], "cycle in captured dependencies");
            visited[node] = true;
            chain_types.push(types[node]);
            current = next[node];
        }
        assert!(visited.iter().all(|&v| v), "disconnected captured nodes");
        let mut exec = ptr::null_mut();
        ok(cudaGraphInstantiateWithFlags(&mut exec, graph, 0));
        ok(cudaGraphDestroy(graph));
        (
            Graph(exec),
            Topology {
                nodes: count,
                edges,
                chain_types,
            },
        )
    }
}

fn assert_direct_equivalent(fixture: &Fixture, host: &[u16], finite: bool) -> Vec<u16> {
    let result = fixture.assert_equal(host, finite);
    let baseline_converted = &fixture.routes[0].converted;
    assert_eq!(
        baseline_converted.read::<u8>(baseline_converted.bytes),
        vec![0x35; baseline_converted.bytes],
        "both arms must use direct pack"
    );
    result
}

#[test]
#[ignore = "exclusive CUDA same-node-count row flag clear control; no product change"]
fn direct_f16_row_flag_clear_kernel_preserves_bits_and_pairs() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    let shape = Shape {
        algorithm: UpstreamLinearAlgorithm::Mmq,
        format: UpstreamLinearFormat::Iq4Xs,
        m: 8,
        k: 5120,
        n: 17408,
    };
    let fixture = Fixture::new(shape, 3 * l2 as usize);
    let stream = Stream::new();
    fixture.scan(&stream);
    let packs: [Pack; 2] = [
        ferrum_test_upstream_mmq_pack_f16_direct_v1,
        ferrum_test_upstream_mmq_pack_f16_direct_kernel_zero_v1,
    ];
    let enqueue = |arm, index| fixture.enqueue_with_pack(&stream, arm, index, packs[arm]);
    let host = fixture.write_input(0, false);
    for arm in 0..2 {
        for index in 0..fixture.count {
            enqueue(arm, index);
        }
    }
    stream.sync();
    let stable = assert_direct_equivalent(&fixture, &host, true);
    let laps = 320_usize.div_ceil(fixture.count);
    let projections = laps * fixture.count;
    let graphs = [0, 1].map(|arm| {
        capture_topology(&stream, || {
            for _ in 0..laps {
                for index in 0..fixture.count {
                    enqueue(arm, index);
                }
            }
        })
    });
    let (baseline, candidate) = (&graphs[0].1, &graphs[1].1);
    assert_eq!(baseline.nodes, candidate.nodes);
    assert_eq!(baseline.edges, candidate.edges);
    // Every position is an unchanged compute kernel or the one intentional
    // memset-to-kernel replacement. Structural checks above prove equal chain
    // dependencies rather than merely asserting a hard-coded total node count.
    let mut replacements = 0;
    for (&before, &after) in baseline.chain_types.iter().zip(&candidate.chain_types) {
        match (before, after) {
            (0, 0) => {}
            (2, 0) => replacements += 1,
            _ => panic!("unexpected graph node type change {before}->{after}"),
        }
    }
    assert_eq!(
        replacements, projections,
        "each pack replaces exactly its row flag memset"
    );
    let edge_host = fixture.write_input(0, true);
    for (graph, _) in &graphs {
        graph.launch(&stream);
    }
    stream.sync();
    assert_direct_equivalent(&fixture, &edge_host, false);
    let host = fixture.write_input(0, false);
    for (graph, _) in &graphs {
        graph.launch(&stream);
    }
    stream.sync();
    assert_eq!(assert_direct_equivalent(&fixture, &host, true), stable);
    const WARM: usize = 2;
    const PAIRS: usize = 8;
    let samples = paired_graph_samples(&stream, [&graphs[0].0, &graphs[1].0], WARM, PAIRS);
    stream.sync();
    assert_eq!(assert_direct_equivalent(&fixture, &host, true), stable);
    let fresh = fixture.write_input(1, false);
    for (graph, _) in &graphs {
        graph.launch(&stream);
    }
    stream.sync();
    assert_ne!(assert_direct_equivalent(&fixture, &fresh, true), stable);
    for (launches, pair, order, arm, wall, cpu, block_wall_ns, gpu_ns) in samples {
        let mut record = serde_json::json!({
            "experiment":"direct_f16_row_flag_clear_kernel_control",
            "algorithm":"Mmq","format":"Iq4Xs","M":8,"K":shape.k,"N":shape.n,
            "matrix_count":fixture.count,"weight_bytes":fixture.host_weights.len(),
            "working_set_bytes":fixture.count*fixture.host_weights.len(),"device_l2_bytes":l2,
            "same_physical_weights_between_arms":true,"input_stride":fixture.input_stride,
            "input_view_offset_bytes":2,"output_stride":shape.n+5,"output_view_offset_elements":1,
            "laps":laps,"projections_per_graph":projections,
            "correctness":"full packed/guard/flags/F32/fixup/F16/padding/canaries/fresh replay, special values then finite reset; both converted buffers untouched"});
        record["graph"] = serde_json::json!({
            "baseline":baseline.describe(),"candidate":candidate.describe(),
            "same_node_count_and_dependency_chain":true,"memset_to_kernel_replacements":replacements,
            "mode":if launches==1 {"idle_single"} else {"idle_then_queued_two"},
            "graph_launches_per_block":launches,"projections_per_block":launches*projections,
            "warm_pairs":WARM,"pairs":PAIRS});
        record["sample"] = serde_json::json!({
            "pair":pair,"order":order,"arm":if arm==0 {"direct_f16_memset"} else {"direct_f16_clear_kernel"},
            "gpu_ns":gpu_ns,"block_wall_ns":block_wall_ns,
            "graph_launch_wall_ns":&wall[..launches],"graph_launch_thread_cpu_ns":&cpu[..launches]});
        record["scope"] = serde_json::json!({
            "only_change":"same direct F16 pack; row flag cudaMemsetAsync becomes standalone clear kernel",
            "timing":"raw runtime cudaGraphLaunch with adjacent CPU-clock reads; optional Linux thread CPU; status check outside clocks; GPU events and full block through final fence",
            "excluded":"pre-block idle fence/allocation/upload/cold scan/capture/topology queries/assertions/readback",
            "limits":"runtime API proxy, not product driver API; event span not sum of kernels; CPU may include busy-spin; no model throughput claim"});
        println!("{record}");
    }
}
