//! TEST-only control-node experiment. Projections are deliberately outside this
//! 48-layer recurrent subgraph; no claim about full-model launch cost is made.
use super::*;
use cudarc::driver::sys::{self, CUevent_flags};
use ferrum_runtime::{DeviceEnqueueStage, DeviceSubmissionStage, DeviceSubmissionTimingSink};
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

const PREPARE: &str = "ferrum_test_gdn_prepare_implicit_single_token_f16";
const DELTA: &str = "ferrum_test_gdn_delta_implicit_single_token_tiled16_interleaved_f32";

fn shape(actual: bool) -> AttentionShape {
    let mut s = super::super::tests::test_shape();
    s.hidden_size = if actual { 5120 } else { 16 };
    s.key_heads = if actual { 16 } else { 1 };
    s.value_heads = if actual { 48 } else { 2 };
    s.key_head_dim = 128;
    s.value_head_dim = 128;
    s.value_features = s.value_heads * 128;
    s.qkv_features = 2 * s.key_heads * 128 + s.value_features;
    s.qkvz_features = s.qkv_features + s.value_features;
    s.ba_features = 2 * s.value_heads;
    s.qkvzba_features = s.qkvz_features + s.ba_features;
    s.decay_parameterization = GatedDeltaDecayParameterization::NegativeRate;
    s.value_head_mapping = GatedDeltaValueHeadMapping::InterleavedByKeyHead;
    s
}

fn functions(runtime: &CudaDeviceRuntime) -> (AttentionFunctions, AttentionFunctions) {
    let original = CudaGatedDeltaRecurrentAttentionProvider::new(runtime)
        .unwrap()
        .functions;
    let mut implicit = original.clone();
    let linear = runtime
        .context()
        .load_module(Ptx::from_src(crate::ptx::LINEAR_ATTENTION.to_owned()))
        .unwrap();
    let delta = runtime
        .context()
        .load_module(Ptx::from_src(crate::ptx::GATED_DELTA_RULE.to_owned()))
        .unwrap();
    implicit.prepare_negative_rate = linear.load_function(PREPARE).unwrap();
    implicit.delta_tiled_interleaved = delta.load_function(DELTA).unwrap();
    (original, implicit)
}

fn f32s(n: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| sample(i, salt, scale)).collect()
}
fn halves(n: usize, salt: usize, scale: f32) -> Vec<f16> {
    f32s(n, salt, scale)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}
fn put<T: cudarc::driver::DeviceRepr>(stream: &Arc<CudaStream>, pointer: u64, data: &[T]) {
    unsafe { cudarc::driver::result::memcpy_htod_async(pointer, data, stream.cu_stream()) }
        .unwrap();
}
fn bits(stream: &Arc<CudaStream>, a: &Guarded<f32>, b: &Guarded<f32>) {
    assert_eq!(
        a.read(stream)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        b.read(stream)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );
}

struct Arm {
    conv: Guarded<f16>,
    state: Guarded<f32>,
    binding: Guarded<u64>,
    query: Guarded<f32>,
    key: Guarded<f32>,
    value: Guarded<f32>,
    g: Guarded<f32>,
    beta: Guarded<f32>,
    core: Guarded<f32>,
    gated: Guarded<f32>,
    z: Guarded<f16>,
    final_conv: Guarded<f16>,
    cast: Guarded<f16>,
    lengths: Guarded<u32>,
    indices: Guarded<u32>,
}
struct Layer {
    arms: [Arm; 2],
    raw: Guarded<f16>,
    weight: Guarded<f16>,
    decay: Guarded<f32>,
    bias: Guarded<f32>,
    norm: Guarded<f32>,
    counts: Vec<usize>,
    lengths: Vec<u32>,
    indices: Vec<u32>,
    shape: AttentionShape,
    salt: usize,
}
impl Layer {
    fn new(
        stream: &Arc<CudaStream>,
        shape: AttentionShape,
        counts: Vec<usize>,
        salt: usize,
    ) -> Self {
        assert!(counts.iter().all(|&n| n > 0));
        let m = counts.len();
        let tokens = counts.iter().sum::<usize>();
        let qk = (shape.key_heads * shape.key_head_dim) as usize;
        let v = shape.value_features as usize;
        let conv = shape.conv_state_elements().unwrap() as usize;
        let state = (shape.value_heads * shape.value_head_dim * shape.key_head_dim) as usize;
        let mut lengths = vec![0u32];
        let mut indices = Vec::new();
        for (i, &n) in counts.iter().enumerate() {
            lengths.push(lengths.last().unwrap() + n as u32);
            indices.extend(std::iter::repeat_n(i as u32, n));
        }
        let arms = std::array::from_fn(|_| {
            let conv_buffer = Guarded::new(
                stream,
                &halves(m * conv, salt + 1, 0.0078125),
                f16::from_f32(81.0),
            );
            let state_buffer = Guarded::new(stream, &f32s(m * state, salt + 3, 0.0009765625), 91.0);
            // Reverse the logical-to-physical sequence mapping in both arms.
            let pointers = (0..m)
                .rev()
                .flat_map(|slot| {
                    [
                        conv_buffer.pointer(stream) + (slot * conv * 2) as u64,
                        state_buffer.pointer(stream) + (slot * state * 4) as u64,
                    ]
                })
                .collect::<Vec<_>>();
            let f32buf = |n| Guarded::new(stream, &vec![0.0f32; n], 101.0);
            let halfbuf = |n| Guarded::new(stream, &vec![f16::ZERO; n], f16::from_f32(111.0));
            Arm {
                conv: conv_buffer,
                state: state_buffer,
                binding: Guarded::new(stream, &pointers, u64::MAX),
                query: f32buf(tokens * qk),
                key: f32buf(tokens * qk),
                value: f32buf(tokens * v),
                g: f32buf(tokens * shape.value_heads as usize),
                beta: f32buf(tokens * shape.value_heads as usize),
                core: f32buf(tokens * v),
                gated: f32buf(tokens * v),
                z: halfbuf(tokens * v),
                final_conv: halfbuf(m * conv),
                cast: halfbuf(tokens * v),
                lengths: Guarded::new(stream, &vec![u32::MAX; m + 1], u32::MAX - 1),
                indices: Guarded::new(stream, &vec![u32::MAX; tokens], u32::MAX - 1),
            }
        });
        Self {
            arms,
            raw: Guarded::new(
                stream,
                &halves(tokens * shape.qkvzba_features as usize, salt + 7, 0.03125),
                f16::from_f32(41.0),
            ),
            weight: Guarded::new(
                stream,
                &halves(
                    shape.qkv_features as usize * shape.conv_kernel as usize,
                    salt + 9,
                    0.015625,
                ),
                f16::from_f32(71.0),
            ),
            decay: Guarded::new(stream, &vec![-0.5f32; shape.value_heads as usize], 61.0),
            bias: Guarded::new(
                stream,
                &f32s(shape.value_heads as usize, salt + 5, 0.03125),
                51.0,
            ),
            norm: Guarded::new(stream, &vec![1.0f32; shape.value_head_dim as usize], 31.0),
            counts,
            lengths,
            indices,
            shape,
            salt,
        }
    }
    fn implicit(&self) -> bool {
        self.counts.iter().all(|&n| n == 1)
    }
    fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        funcs: &(AttentionFunctions, AttentionFunctions),
        arm: usize,
    ) {
        let a = &self.arms[arm];
        let implicit = arm == 1 && self.implicit();
        let f = if implicit { &funcs.1 } else { &funcs.0 };
        let s = self.shape;
        let cuda = s.cuda_shape().unwrap();
        let tokens = self.indices.len();
        let m = self.counts.len();
        if !implicit {
            put(stream, a.lengths.pointer(stream), &self.lengths);
            put(stream, a.indices.pointer(stream), &self.indices);
        }
        // Null controls make any accidental read by the candidate observable.
        let lengths = if implicit {
            0
        } else {
            a.lengths.pointer(stream)
        };
        let indices = if implicit {
            0
        } else {
            a.indices.pointer(stream)
        };
        launch_prepare(AttentionPrepareRequest {
            stream,
            function: &f.prepare_negative_rate,
            buffers: AttentionPrepareBuffers {
                qkvzba: self.raw.pointer(stream),
                conv_weight: self.weight.pointer(stream),
                state_binding: a.binding.pointer(stream),
                a_log: self.decay.pointer(stream),
                dt_bias: self.bias.pointer(stream),
                cu_seqlens: lengths,
                token_seq_indices: indices,
                query: a.query.pointer(stream),
                key: a.key.pointer(stream),
                value: a.value.pointer(stream),
                z: a.z.pointer(stream),
                g: a.g.pointer(stream),
                beta: a.beta.pointer(stream),
                final_conv_state: a.final_conv.pointer(stream),
            },
            context: AttentionPrepareContext {
                tokens: tokens as u64,
                tokens_i32: tokens as i32,
                batch: m as i32,
                physical: cuda,
                logical: s,
            },
        })
        .unwrap();
        launch_conv_state_commit(
            stream,
            &f.conv_state_commit,
            a.final_conv.pointer(stream),
            a.binding.pointer(stream),
            m as i32,
            s.conv_state_elements().unwrap() as i32,
        )
        .unwrap();
        launch_qk_norm(
            stream,
            &f.qk_norm,
            a.query.pointer(stream),
            a.key.pointer(stream),
            tokens as u64,
            tokens as i32,
            cuda,
        )
        .unwrap();
        launch_delta(
            stream,
            f,
            a.query.pointer(stream),
            a.key.pointer(stream),
            a.value.pointer(stream),
            a.g.pointer(stream),
            a.beta.pointer(stream),
            a.binding.pointer(stream),
            lengths,
            a.core.pointer(stream),
            m as i32,
            tokens as i32,
            cuda,
        )
        .unwrap();
        launch_gated_norm(
            stream,
            &f.gated_norm,
            a.core.pointer(stream),
            a.z.pointer(stream),
            self.norm.pointer(stream),
            a.gated.pointer(stream),
            tokens as u64,
            cuda,
        )
        .unwrap();
        launch_cast(
            stream,
            &f.f32_to_f16,
            a.gated.pointer(stream),
            a.cast.pointer(stream),
            tokens as u64 * s.value_features,
        )
        .unwrap();
    }
    fn check(&self, stream: &Arc<CudaStream>) {
        let [a, b] = &self.arms;
        for (x, y) in [
            (&a.state, &b.state),
            (&a.query, &b.query),
            (&a.key, &b.key),
            (&a.value, &b.value),
            (&a.g, &b.g),
            (&a.beta, &b.beta),
            (&a.core, &b.core),
            (&a.gated, &b.gated),
        ] {
            bits(stream, x, y);
        }
        for (x, y) in [
            (&a.conv, &b.conv),
            (&a.z, &b.z),
            (&a.final_conv, &b.final_conv),
            (&a.cast, &b.cast),
        ] {
            assert_eq!(
                x.read(stream)
                    .into_iter()
                    .map(f16::to_bits)
                    .collect::<Vec<_>>(),
                y.read(stream)
                    .into_iter()
                    .map(f16::to_bits)
                    .collect::<Vec<_>>()
            );
        }
        for a in &self.arms {
            a.binding.assert_unchanged(stream);
            a.lengths.read(stream);
            a.indices.read(stream);
        }
        if self.implicit() {
            b.lengths.assert_unchanged(stream);
            b.indices.assert_unchanged(stream);
        }
        self.raw.read(stream);
        self.weight.assert_unchanged(stream);
        self.decay.assert_unchanged(stream);
        self.bias.assert_unchanged(stream);
        self.norm.assert_unchanged(stream);
    }
    fn fresh(&self, stream: &Arc<CudaStream>, special: bool) {
        let mut raw = halves(
            self.indices.len() * self.shape.qkvzba_features as usize,
            self.salt + 13,
            0.03125,
        );
        raw[0] = f16::from_bits(0x8000);
        raw[1] = f16::from_bits(1);
        if special {
            raw[2] = f16::NAN;
            raw[3] = f16::INFINITY;
        }
        put(stream, self.raw.pointer(stream), &raw);
        // The raw async copy borrows this local host Vec. Keep it alive until
        // the transfer completes; every fresh-input update is outside timing.
        stream.synchronize().unwrap();
    }
    fn oracle_first(&self, stream: &Arc<CudaStream>) {
        let s = self.shape;
        let m = self.counts.len();
        let c = s.conv_state_elements().unwrap() as usize;
        let n = (s.value_heads * s.value_head_dim * s.key_head_dim) as usize;
        let cv = halves(m * c, self.salt + 1, 0.0078125);
        let st = f32s(m * n, self.salt + 3, 0.0009765625);
        let mut reference = Reference {
            conv: (0..m)
                .rev()
                .map(|i| cv[i * c..(i + 1) * c].iter().map(|v| v.to_f64()).collect())
                .collect(),
            delta: (0..m)
                .rev()
                .map(|i| {
                    st[i * n..(i + 1) * n]
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect()
                })
                .collect(),
        };
        let expected = reference.forward(
            s,
            &self.counts,
            &self.raw.read(stream),
            &self.weight.read(stream),
            &self.decay.read(stream),
            &self.bias.read(stream),
        );
        assert_close(
            &self.arms[0].core.read(stream),
            &expected.output,
            "explicit complete recurrent F64 output",
        );
        let actual = self.arms[0].state.read(stream);
        for (logical, physical) in (0..m).rev().enumerate() {
            assert_close(
                &actual[physical * n..(physical + 1) * n],
                &reference.delta[logical],
                "F64 carried state",
            );
        }
    }
}

// Raw graph lifecycle follows the existing captured.rs test helper, with node
// types inspected here because removing only control copies is this experiment.
struct Graph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    nodes: usize,
    kernels: usize,
    copies: usize,
    edges: usize,
}
fn ok(r: sys::CUresult) {
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS);
}
impl Graph {
    fn new(stream: &Arc<CudaStream>, enqueue: impl FnOnce()) -> Self {
        stream.synchronize().unwrap();
        unsafe {
            ok(sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            ));
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(enqueue));
        let mut graph = std::ptr::null_mut();
        unsafe {
            ok(sys::cuStreamEndCapture(stream.cu_stream(), &mut graph));
        }
        if let Err(p) = result {
            if !graph.is_null() {
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            std::panic::resume_unwind(p);
        }
        let mut count = 0;
        unsafe {
            ok(sys::cuGraphGetNodes(
                graph,
                std::ptr::null_mut(),
                &mut count,
            ));
        }
        let mut nodes = vec![std::ptr::null_mut(); count];
        unsafe {
            ok(sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count));
        }
        let (mut kernels, mut copies) = (0, 0);
        for node in nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            unsafe {
                ok(sys::cuGraphNodeGetType(node, &mut kind));
            }
            match kind {
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => kernels += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => copies += 1,
                _ => panic!("unexpected node {kind:?}"),
            }
        }
        let mut edges = 0;
        unsafe {
            ok(sys::cuGraphGetEdges_v2(
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut edges,
            ));
        }
        assert_eq!(
            edges,
            count - 1,
            "same-stream complete subgraph must remain one chain"
        );
        let mut from = vec![std::ptr::null_mut(); edges];
        let mut to = vec![std::ptr::null_mut(); edges];
        // Null edgeData requests only default dependencies; CUDA rejects a lossy
        // query if a non-default dependency was captured.
        unsafe {
            ok(sys::cuGraphGetEdges_v2(
                graph,
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut edges,
            ));
        }
        let mut indegree = std::collections::HashMap::new();
        let mut outdegree = std::collections::HashMap::new();
        for (a, b) in from.into_iter().zip(to) {
            assert_ne!(a, b);
            *outdegree.entry(a).or_insert(0usize) += 1;
            *indegree.entry(b).or_insert(0usize) += 1;
        }
        assert!(
            indegree.values().chain(outdegree.values()).all(|&n| n == 1),
            "captured layer dependencies must remain a single chain"
        );
        let mut exec = std::ptr::null_mut();
        unsafe {
            ok(sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0));
            ok(sys::cuGraphUpload(exec, stream.cu_stream()));
        }
        stream.synchronize().unwrap();
        Self {
            graph,
            exec,
            stream: stream.clone(),
            nodes: count,
            kernels,
            copies,
            edges,
        }
    }
    fn launch(&self) {
        unsafe {
            ok(sys::cuGraphLaunch(self.exec, self.stream.cu_stream()));
        }
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}
#[derive(Default)]
struct Timing(Mutex<(Duration, Option<Duration>)>);
impl DeviceSubmissionTimingSink for Timing {
    const ENABLED: bool = true;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {}
    fn record_device_enqueue(&self, _: DeviceEnqueueStage, wall: Duration, cpu: Option<Duration>) {
        *self.0.lock().unwrap() = (wall, cpu);
    }
}
fn measure(g: &Graph) -> serde_json::Value {
    g.stream.synchronize().unwrap();
    let wall = Instant::now();
    let start = g
        .stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let sink = Timing::default();
    let status = crate::backend::enqueue_timing::measure(
        &sink,
        DeviceEnqueueStage::GraphLaunch,
        true,
        || unsafe { sys::cuGraphLaunch(g.exec, g.stream.cu_stream()) },
    );
    ok(status);
    let end = g
        .stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    let block = wall.elapsed();
    let gpu = start.elapsed_ms(&end).unwrap();
    let (api, cpu) = *sink.0.lock().unwrap();
    serde_json::json!({"gpu_ms":gpu,"block_wall_ns":block.as_nanos(),"launch_wall_ns":api.as_nanos(),"launch_thread_cpu_ns":cpu.map(|n|n.as_nanos())})
}
fn runtime() -> CudaDeviceRuntime {
    CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.gdn-implicit-controls").unwrap(),
            AttentionExecutionPolicy::default(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
#[ignore = "requires CUDA; test-only implicit control symbols"]
fn implicit_single_token_controls_preserve_bits_state_fresh_and_fallback() {
    let r = runtime();
    // CUDA graph capture cannot begin on the legacy default stream.
    let stream = r.context().new_stream().unwrap();
    let f = functions(&r);
    for counts in [vec![1], vec![1; 8], vec![1; 32], vec![1, 2, 1]] {
        let layer = Layer::new(&stream, shape(false), counts, 4);
        let graphs = [
            Graph::new(&stream, || layer.enqueue(&stream, &f, 0)),
            Graph::new(&stream, || layer.enqueue(&stream, &f, 1)),
        ];
        for g in &graphs {
            g.launch();
        }
        layer.check(&stream);
        layer.oracle_first(&stream);
        for special in [false, true] {
            layer.fresh(&stream, special);
            for g in &graphs {
                g.launch();
            }
            layer.check(&stream);
        }
        println!(
            "{}",
            serde_json::json!({"diagnostic":"gdn_implicit_control_correctness","counts":layer.counts,"implicit":layer.implicit(),"full_bits_state_guards_fresh":true,"f64_finite_first":true,"nonfinite_fresh_bit_comparison":true,"old_nodes":graphs[0].nodes,"new_nodes":graphs[1].nodes})
        );
    }
}

#[test]
#[ignore = "requires CUDA; 48 independent actual-shape recurrent layers"]
fn implicit_single_token_controls_forty_eight_layer_graph_pairs() {
    let r = runtime();
    // CUDA graph capture cannot begin on the legacy default stream.
    let stream = r.context().new_stream().unwrap();
    let f = functions(&r);
    let s = shape(true);
    let layers = (0..48)
        .map(|i| Layer::new(&stream, s, vec![1; 8], i + 1))
        .collect::<Vec<_>>();
    // Graphs are declared last and drop first; every captured pointer and boxed
    // host control array remains alive until both executable graphs are gone.
    let graphs = [
        Graph::new(&stream, || {
            for l in &layers {
                l.enqueue(&stream, &f, 0)
            }
        }),
        Graph::new(&stream, || {
            for l in &layers {
                l.enqueue(&stream, &f, 1)
            }
        }),
    ];
    assert_eq!(graphs[0].kernels, graphs[1].kernels);
    assert_eq!(graphs[0].copies, 96);
    assert_eq!(graphs[1].copies, 0);
    assert_eq!(graphs[0].nodes - graphs[1].nodes, 96);
    for _ in 0..2 {
        for g in &graphs {
            g.launch();
        }
    }
    stream.synchronize().unwrap();
    for l in &layers {
        l.check(&stream);
        l.fresh(&stream, false);
    }
    stream.synchronize().unwrap();
    for pair in 0..8 {
        let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
        let mut measured = [serde_json::Value::Null, serde_json::Value::Null];
        for arm in order {
            measured[arm] = measure(&graphs[arm]);
        }
        println!(
            "{}",
            serde_json::json!({"diagnostic":"gdn_implicit_control_48_layer_pair","pair":pair,"order":if pair%2==0{"AB"}else{"BA"},"rows":8,"layers":48,"key_heads":16,"value_heads":48,"head_dim":128,"conv_kernel":4,"baseline":measured[0],"implicit":measured[1],"nodes":[graphs[0].nodes,graphs[1].nodes],"kernels":[graphs[0].kernels,graphs[1].kernels],"copies":[graphs[0].copies,graphs[1].copies],"edges":[graphs[0].edges,graphs[1].edges],"scope":"complete prepare/conv-commit/QK-norm/delta/gated-norm/cast subgraph; projections and whole model excluded","capture_graph_upload_alloc_input_upload_validation_outside_timing":true,"baseline_control_uploads_inside_timed_graph":true,"state_advance_count_equal":true,"launch_wall_includes_cpu_clock_reads":true})
        );
    }
    for l in &layers {
        l.check(&stream);
    }
    println!(
        "{}",
        serde_json::json!({"diagnostic":"gdn_implicit_control_48_layer_correctness","all_48_outputs_states_guards_fresh_equal":true,"projections_excluded":true})
    );
}
