//! Check the allocation and numerical effects of an owned cuBLAS workspace.
//! Preparation timings include the native calls; they are not model speedups.
use super::*;
use cudarc::{cublas::sys as blas_sys, driver::sys};
use std::collections::BTreeMap;

struct Graph {
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            if !self.executable.is_null() {
                sys::cuGraphExecDestroy(self.executable);
            }
            if !self.graph.is_null() {
                sys::cuGraphDestroy(self.graph);
            }
        }
    }
}

fn check(result: sys::CUresult) {
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);
}

fn node_types(graph: sys::CUgraph) -> BTreeMap<String, usize> {
    let mut count = 0;
    check(unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) });
    let mut nodes = vec![std::ptr::null_mut(); count];
    check(unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) });
    let mut types = BTreeMap::new();
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        check(unsafe { sys::cuGraphNodeGetType(node, &mut kind) });
        *types.entry(format!("{kind:?}")).or_default() += 1;
    }
    types
}

#[test]
#[ignore = "requires exclusive CUDA; native workspace allocation/numerical diagnostic, not model performance"]
fn cuda_dense_f16_owned_workspace_capture_preserves_outputs_and_bounds_graph_memory() {
    const WORKSPACE_BYTES: usize = 32 << 20;
    const ALIGNMENT: usize = 256;
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let blas = CudaBlas::new(Arc::clone(&stream)).unwrap();
    let workspace = stream
        .clone_htod(&vec![0xac_u8; WORKSPACE_BYTES + 2 * ALIGNMENT])
        .unwrap();
    let workspace_pointer = workspace.device_ptr(&stream).0 + ALIGNMENT as u64;
    assert_eq!(workspace_pointer % ALIGNMENT as u64, 0);

    for (name, outputs, inputs) in [("ffn_down", 4096, 12288), ("gate_up", 24576, 4096)] {
        let largest = Shape::new(name, 2, outputs, inputs);
        let x = Buffer::new(&stream, largest.rows * inputs, |i| {
            value(i, 37, 1.0 / 1024.0)
        });
        let w = Buffer::new(&stream, outputs * inputs, |i| value(i, 731, 1.0 / 65536.0));
        let reference = f64_reference(largest, x.values(), w.values());
        for rows in [1, 2] {
            let mut arm_outputs = Vec::new();
            for owned in [false, true] {
                stream.synchronize().unwrap();
                // SetStream restores the vendor pool before selecting this arm.
                unsafe {
                    blas_sys::cublasSetStream_v2(*blas.handle(), stream.cu_stream() as _)
                        .result()
                        .unwrap();
                    if owned {
                        blas_sys::cublasSetWorkspace_v2(
                            *blas.handle(),
                            workspace_pointer as *mut _,
                            WORKSPACE_BYTES,
                        )
                        .result()
                        .unwrap();
                    }
                }
                let output = Buffer::new(&stream, rows * outputs, |_| GUARD);
                let pointers = [
                    x.pointer(&stream),
                    w.pointer(&stream),
                    output.pointer(&stream),
                ];
                let launch = || {
                    launch_gemm_f16(
                        &blas,
                        pointers[0],
                        pointers[1],
                        pointers[2],
                        rows as i32,
                        outputs as i32,
                        inputs as i32,
                        "owned workspace diagnostic",
                    )
                    .unwrap();
                };
                launch();
                stream.synchronize().unwrap();
                let eager = output.read(&stream);
                let errors = validate(&eager, &reference[..rows * outputs]);

                let started = Instant::now();
                check(unsafe {
                    sys::cuStreamBeginCapture_v2(
                        stream.cu_stream(),
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                    )
                });
                launch();
                let mut graph = Graph {
                    graph: std::ptr::null_mut(),
                    executable: std::ptr::null_mut(),
                };
                check(unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph.graph) });
                let capture_us = started.elapsed().as_micros();
                let types = node_types(graph.graph);
                if owned {
                    assert_eq!(
                        types
                            .get("CU_GRAPH_NODE_TYPE_MEM_ALLOC")
                            .copied()
                            .unwrap_or(0),
                        0,
                        "owned workspace must not allocate another workspace in the graph"
                    );
                }
                let started = Instant::now();
                check(unsafe {
                    sys::cuGraphInstantiateWithFlags(&mut graph.executable, graph.graph, 0)
                });
                let instantiate_us = started.elapsed().as_micros();
                let started = Instant::now();
                check(unsafe { sys::cuGraphUpload(graph.executable, stream.cu_stream()) });
                let upload_host_us = started.elapsed().as_micros();
                stream.synchronize().unwrap();
                let upload_complete_us = started.elapsed().as_micros();
                for _ in 0..2 {
                    check(unsafe { sys::cuGraphLaunch(graph.executable, stream.cu_stream()) });
                    stream.synchronize().unwrap();
                    assert_eq!(
                        output.read(&stream),
                        eager,
                        "capture/replay changed native GEMM output"
                    );
                }
                println!(
                    "{}",
                    serde_json::json!({
                        "diagnostic":"dense_f16_workspace_graph", "shape":name, "m":rows,
                        "n":outputs, "k":inputs, "owned_workspace_bytes":if owned { WORKSPACE_BYTES } else { 0 },
                        "graph_nodes":types, "capture_us":capture_us, "instantiate_us":instantiate_us,
                        "upload_host_us":upload_host_us, "upload_complete_us":upload_complete_us,
                        "errors":errors, "model_performance_claim":false
                    })
                );
                arm_outputs.push(eager);
            }
            let mismatches = arm_outputs[0]
                .iter()
                .zip(&arm_outputs[1])
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "{}",
                serde_json::json!({"diagnostic":"dense_f16_workspace_output_comparison",
                "shape":name,"m":rows,"all_outputs_checked":rows*outputs,"bit_mismatches":mismatches})
            );
            assert_eq!(
                arm_outputs[0], arm_outputs[1],
                "workspace selection changed fixture output"
            );
        }
        x.assert_read_only(&stream);
        w.assert_read_only(&stream);
    }
    assert_eq!(
        stream.clone_dtoh(&workspace.slice(..ALIGNMENT)).unwrap(),
        vec![0xac_u8; ALIGNMENT]
    );
    assert_eq!(
        stream
            .clone_dtoh(&workspace.slice(ALIGNMENT + WORKSPACE_BYTES..))
            .unwrap(),
        vec![0xac_u8; ALIGNMENT]
    );
    // Detach the caller-owned pointer before its allocation is dropped.
    unsafe {
        blas_sys::cublasSetStream_v2(*blas.handle(), stream.cu_stream() as _)
            .result()
            .unwrap();
    }
}
