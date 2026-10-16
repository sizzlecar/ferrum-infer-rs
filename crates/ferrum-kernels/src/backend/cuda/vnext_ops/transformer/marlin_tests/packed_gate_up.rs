//! Complete production-wrapper gate/up comparison; no new kernel or product route.
use super::*;
use crate::marlin_repack::fixtures::{Fixture, HostFixture};
use cudarc::driver::{sys, CudaFunction, CudaStream};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc, time::Instant};

const PAIRS: usize = 8;
const PHYSICAL_SETS: usize = 3;
const CPU_RELATIVE_L2_LIMIT: f64 = 0.05;

mod complete_ffn_reference;

struct Repacked {
    weights: Vec<i32>,
    scales: Vec<f16>,
    zeros: Vec<i32>,
}

impl Repacked {
    fn new(q: &[i32], s: &[f16], z: &[i32], k: usize, n: usize) -> Self {
        Self {
            weights: repack_gptq_to_marlin(q, k, n),
            scales: repack_scales_to_marlin(s, k, n, GROUP_SIZE),
            zeros: repack_compressed_tensors_zero_points_to_marlin(z, k / GROUP_SIZE, n),
        }
    }

    fn bytes(&self) -> usize {
        self.weights.len() * 4 + self.scales.len() * 2 + self.zeros.len() * 4
    }
}

fn source(f: Fixture) -> HostFixture {
    let mut host = build_host_fixture(f);
    let n = f.output_features / 2;
    // Distinct gate/up scales and zeros make a half swap observable even when
    // the original deterministic fixture happens to repeat after N columns.
    for row in host.scales_grouped.chunks_exact_mut(2 * n) {
        for value in &mut row[n..] {
            *value = f16::from_f32(value.to_f32() * 1.125);
        }
    }
    let half_zeros = (n / 8) * (f.input_features / GROUP_SIZE);
    for value in &mut host.zero_points_compressed_tensors[half_zeros..] {
        *value ^= 0x2222_2222;
    }
    host
}

fn repack_source(f: Fixture, host: &HostFixture) -> [Repacked; 3] {
    let k = f.input_features;
    let n = f.output_features / 2;
    let half_rows = |rows: &[i32], half: usize| -> Vec<i32> {
        rows.chunks_exact(2 * n)
            .flat_map(|row| row[half * n..(half + 1) * n].iter().copied())
            .collect()
    };
    let half_scales = |half: usize| -> Vec<f16> {
        host.scales_grouped
            .chunks_exact(2 * n)
            .flat_map(|row| row[half * n..(half + 1) * n].iter().copied())
            .collect()
    };
    let z_half = (n / 8) * (k / GROUP_SIZE);
    let gate_q = half_rows(&host.qweight_gptq, 0);
    let up_q = half_rows(&host.qweight_gptq, 1);
    let gate_s = half_scales(0);
    let up_s = half_scales(1);
    assert!(gate_q != up_q, "gate/up quantized values must differ");
    assert!(gate_s != up_s, "gate/up scales must differ");
    assert!(
        host.zero_points_compressed_tensors[..z_half]
            != host.zero_points_compressed_tensors[z_half..]
    );
    [
        Repacked::new(
            &gate_q,
            &gate_s,
            &host.zero_points_compressed_tensors[..z_half],
            k,
            n,
        ),
        Repacked::new(
            &up_q,
            &up_s,
            &host.zero_points_compressed_tensors[z_half..],
            k,
            n,
        ),
        Repacked::new(
            &host.qweight_gptq,
            &host.scales_grouped,
            &host.zero_points_compressed_tensors,
            k,
            2 * n,
        ),
    ]
}

struct Projection {
    weights: Guarded<i32>,
    scales: Guarded<f16>,
    zeros: Guarded<i32>,
    output: Guarded<f16>,
    pointers: [u64; 5],
    columns: usize,
}

impl Projection {
    fn new(
        stream: &Arc<CudaStream>,
        host: &Repacked,
        rows: usize,
        columns: usize,
        workspace_pointer: u64,
    ) -> Self {
        let weights = Guarded::new(stream, &host.weights, 0x5a5a1234_i32);
        let scales = Guarded::new(stream, &host.scales, f16::from_f32(-117.0));
        let zeros = Guarded::new(stream, &host.zeros, 0x5a5a1234_i32);
        let output = Guarded::new(
            stream,
            &vec![f16::NAN; rows * columns],
            f16::from_f32(-117.0),
        );
        let pointers = [
            weights.pointer(stream),
            scales.pointer(stream),
            zeros.pointer(stream),
            output.pointer(stream),
            workspace_pointer,
        ];
        Self {
            weights,
            scales,
            zeros,
            output,
            pointers,
            columns,
        }
    }

    fn launch(
        &self,
        runtime: MarlinProjectionRuntime,
        stream: &Arc<CudaStream>,
        input: u64,
        f: Fixture,
    ) {
        runtime
            .launch(
                MarlinF16WeightType::U4,
                stream,
                input,
                self.pointers[0],
                self.pointers[1],
                Some(self.pointers[2]),
                self.pointers[3],
                self.pointers[4],
                runtime.workspace_bytes().unwrap(),
                f.rows as i32,
                self.columns as i32,
                f.input_features as i32,
                GROUP_SIZE as i32,
                "packed gate/up complete transaction",
            )
            .unwrap();
    }

    fn immutable_and_output_guards(&self, stream: &Arc<CudaStream>) {
        self.weights.assert_unchanged(stream);
        self.scales.assert_unchanged(stream);
        self.zeros.assert_unchanged(stream);
        assert!(self.output.read(stream).iter().all(|v| v.is_finite()));
    }
}

struct Arm {
    input: Guarded<f16>,
    workspace: Guarded<i32>,
    workspace_bytes: usize,
    projections: Vec<Projection>,
    activated: Guarded<f16>,
    input_pointer: u64,
    activation_pointer: u64,
}

impl Arm {
    fn new(
        stream: &Arc<CudaStream>,
        runtime: MarlinProjectionRuntime,
        f: Fixture,
        input: &[f16],
        packed: &[Repacked],
        merged: bool,
    ) -> Self {
        let input = Guarded::new(stream, input, f16::from_f32(-117.0));
        // Production gate/up uses one shared workspace and resets it before
        // each Marlin invocation. Preserve that aliasing in the two-call arm.
        let workspace_bytes = runtime.workspace_bytes().unwrap() as usize;
        let workspace = Guarded::new(stream, &vec![73_i32; workspace_bytes / 4], -117_i32);
        let workspace_pointer = workspace.pointer(stream);
        let n = f.output_features / 2;
        let projections = if merged {
            vec![Projection::new(
                stream,
                &packed[2],
                f.rows,
                2 * n,
                workspace_pointer,
            )]
        } else {
            vec![
                Projection::new(stream, &packed[0], f.rows, n, workspace_pointer),
                Projection::new(stream, &packed[1], f.rows, n, workspace_pointer),
            ]
        };
        let activated = Guarded::new(stream, &vec![f16::NAN; f.rows * n], f16::from_f32(-117.0));
        let input_pointer = input.pointer(stream);
        let activation_pointer = activated.pointer(stream);
        Self {
            input,
            workspace,
            workspace_bytes,
            projections,
            activated,
            input_pointer,
            activation_pointer,
        }
    }

    fn enqueue(
        &self,
        runtime: MarlinProjectionRuntime,
        stream: &Arc<CudaStream>,
        f: Fixture,
        planar: &CudaFunction,
        merged: &CudaFunction,
    ) {
        for projection in &self.projections {
            projection.launch(runtime, stream, self.input_pointer, f);
        }
        let n = f.output_features / 2;
        if self.projections.len() == 2 {
            launch_planar_silu_mul(
                stream,
                planar,
                self.projections[0].pointers[3],
                self.projections[1].pointers[3],
                self.activation_pointer,
                (f.rows * n) as u64,
            )
            .unwrap();
        } else {
            launch_silu_mul(
                stream,
                merged,
                self.projections[0].pointers[3],
                self.activation_pointer,
                n as i32,
                (f.rows * n) as u64,
            )
            .unwrap();
        }
    }

    fn projection_values(&self, stream: &Arc<CudaStream>, f: Fixture) -> Vec<f16> {
        if self.projections.len() == 1 {
            return self.projections[0].output.read(stream);
        }
        let gate = self.projections[0].output.read(stream);
        let up = self.projections[1].output.read(stream);
        let n = f.output_features / 2;
        let mut out = Vec::with_capacity(f.rows * 2 * n);
        for row in 0..f.rows {
            out.extend_from_slice(&gate[row * n..(row + 1) * n]);
            out.extend_from_slice(&up[row * n..(row + 1) * n]);
        }
        out
    }

    fn prepare_replay_input(&self, stream: &Arc<CudaStream>, input: &[f16], f: Fixture) {
        unsafe {
            ok(sys::cuMemcpyHtoDAsync_v2(
                self.input_pointer,
                input.as_ptr().cast(),
                std::mem::size_of_val(input),
                stream.cu_stream(),
            ));
            for projection in &self.projections {
                cudarc::driver::result::memset_d8_async(
                    projection.pointers[3],
                    0xff,
                    f.rows * projection.columns * 2,
                    stream.cu_stream(),
                )
                .unwrap();
            }
            cudarc::driver::result::memset_d8_async(
                self.projections[0].pointers[4],
                73,
                self.workspace_bytes,
                stream.cu_stream(),
            )
            .unwrap();
            cudarc::driver::result::memset_d8_async(
                self.activation_pointer,
                0xff,
                f.rows * f.output_features,
                stream.cu_stream(),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        assert!(self
            .input
            .read(stream)
            .iter()
            .zip(input)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

fn ok(result: sys::CUresult) {
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);
}

struct Graph {
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    inventory: BTreeMap<String, usize>,
}

impl Graph {
    fn new(stream: &Arc<CudaStream>, enqueue: impl FnOnce()) -> Self {
        stream.synchronize().unwrap();
        let mut graph = Self {
            graph: std::ptr::null_mut(),
            executable: std::ptr::null_mut(),
            stream: Arc::clone(stream),
            inventory: BTreeMap::new(),
        };
        unsafe {
            ok(sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            ));
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(enqueue));
        let end = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph.graph) };
        if let Err(error) = result {
            std::panic::resume_unwind(error);
        }
        ok(end);
        unsafe {
            let mut count = 0;
            ok(sys::cuGraphGetNodes(
                graph.graph,
                std::ptr::null_mut(),
                &mut count,
            ));
            let mut nodes = vec![std::ptr::null_mut(); count];
            ok(sys::cuGraphGetNodes(
                graph.graph,
                nodes.as_mut_ptr(),
                &mut count,
            ));
            for node in nodes {
                let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                ok(sys::cuGraphNodeGetType(node, &mut kind));
                *graph.inventory.entry(format!("{kind:?}")).or_default() += 1;
            }
            ok(sys::cuGraphInstantiateWithFlags(
                &mut graph.executable,
                graph.graph,
                0,
            ));
            ok(sys::cuGraphUpload(graph.executable, stream.cu_stream()));
        }
        stream.synchronize().unwrap();
        graph
    }

    fn launch(&self) {
        unsafe {
            ok(sys::cuGraphLaunch(self.executable, self.stream.cu_stream()));
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
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

struct Drain(Arc<CudaStream>);
impl Drop for Drain {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}

fn difference(a: &[f16], b: &[f16]) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "non-finite numerical result"
    );
    let mut bits = 0usize;
    let mut maximum = 0.0f64;
    let mut error = 0.0f64;
    let mut reference = 0.0f64;
    for (a, b) in a.iter().zip(b) {
        bits += usize::from(a.to_bits() != b.to_bits());
        let delta = a.to_f64() - b.to_f64();
        maximum = maximum.max(delta.abs());
        error += delta * delta;
        reference += a.to_f64().powi(2);
    }
    json!({"elements":a.len(),"different_bits":bits,"exact_bits":bits==0,"max_absolute":maximum,"relative_l2":(error/reference.max(f64::MIN_POSITIVE)).sqrt()})
}

fn cpu_projection(f: Fixture, host: &HostFixture, input: &[f16], row: usize, col: usize) -> f64 {
    let k = f.input_features;
    let n = f.output_features;
    let groups = k / GROUP_SIZE;
    (0..k)
        .map(|i| {
            let q = ((host.qweight_gptq[(i / 8) * n + col] as u32 >> (4 * (i % 8))) & 15) as i32;
            let z = ((host.zero_points_compressed_tensors[(col / 8) * groups + i / GROUP_SIZE]
                as u32
                >> (4 * (col % 8)))
                & 15) as i32;
            input[row * k + i].to_f64()
                * f64::from(q - z)
                * host.scales_grouped[(i / GROUP_SIZE) * n + col].to_f64()
        })
        .sum()
}

fn oracle(
    f: Fixture,
    host: &HostFixture,
    input: &[f16],
    projection: &[f16],
    activated: &[f16],
    full: bool,
) -> serde_json::Value {
    let n = f.output_features;
    let indices: Vec<usize> = if full {
        (0..f.rows * n).collect()
    } else {
        [0, f.rows / 2, f.rows - 1]
            .into_iter()
            .flat_map(|row| [0, 1, n / 2 - 1, n / 2, n / 2 + 1, n - 1].map(|col| row * n + col))
            .collect()
    };
    let mut squared = 0.0f64;
    let mut error = 0.0f64;
    let mut maximum = 0.0f64;
    for &index in &indices {
        let expected = cpu_projection(f, host, input, index / n, index % n);
        let delta = projection[index].to_f64() - expected;
        squared += expected * expected;
        error += delta * delta;
        maximum = maximum.max(delta.abs());
    }
    let relative_l2 = (error / squared.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        relative_l2 < CPU_RELATIVE_L2_LIMIT,
        "independent projection reference: {relative_l2}"
    );
    // Activation oracle consumes each arm's actual half-rounded projection;
    // it diagnoses layout/math without hiding projection differences.
    let mut activation_max = 0.0f64;
    for row in 0..f.rows {
        for col in 0..n / 2 {
            let g = projection[row * n + col].to_f64();
            let u = projection[row * n + n / 2 + col].to_f64();
            let expected = g / (1.0 + (-g).exp()) * u;
            let delta = (activated[row * (n / 2) + col].to_f64() - expected).abs();
            activation_max = activation_max.max(delta);
            assert!(
                delta <= 0.002 + 0.002 * expected.abs(),
                "activation oracle at row {row} col {col}: {delta}"
            );
        }
    }
    json!({"projection_f64_points":indices.len(),"full_projection_oracle":full,"projection_relative_l2":relative_l2,"projection_max_absolute":maximum,"activation_full_f64_from_actual_half_projection_max_absolute":activation_max})
}

fn measure(stream: &Arc<CudaStream>, graph: &Graph) -> serde_json::Value {
    let wall = Instant::now();
    let start = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let launch = Instant::now();
    graph.launch();
    let launch_ns = launch.elapsed().as_nanos();
    let end = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    json!({"block_wall_ns":wall.elapsed().as_nanos(),"launch_wall_ns":launch_ns,"gpu_ns":f64::from(start.elapsed_ms(&end).unwrap())*1e6})
}

fn run_case(
    context: &Arc<CudaContext>,
    f: Fixture,
    physical_sets: usize,
    full_oracle: bool,
    timed: bool,
) {
    let setup = Instant::now();
    let stream = context.new_stream().unwrap();
    let runtime = MarlinProjectionRuntime {
        multiprocessor_count: context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap(),
        device_ordinal: 0,
    };
    let l2 = context
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap();
    let host = source(f);
    let repack_start = Instant::now();
    let packed = repack_source(f, &host);
    let repack_ns = repack_start.elapsed().as_nanos();
    let bytes_per_set = packed[2].bytes();
    assert_eq!(packed[0].bytes() + packed[1].bytes(), bytes_per_set);
    let arms: [Vec<Arm>; 2] = std::array::from_fn(|arm| {
        (0..physical_sets)
            .map(|_| Arm::new(&stream, runtime, f, &host.input, &packed, arm == 1))
            .collect()
    });
    let _drain = Drain(Arc::clone(&stream));
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let planar = module.load_function(PLANAR_SILU_MUL_FUNCTION_NAME).unwrap();
    let merged = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    for arm in &arms {
        for group in arm {
            group.enqueue(runtime, &stream, f, &planar, &merged);
        }
    }
    stream.synchronize().unwrap();
    let allocation_repack_eager_setup_ns = setup.elapsed().as_nanos();
    let capture_start = Instant::now();
    let graphs: [Graph; 2] = std::array::from_fn(|index| {
        Graph::new(&stream, || {
            for group in &arms[index] {
                group.enqueue(runtime, &stream, f, &planar, &merged);
            }
        })
    });
    let capture_upload_ns = capture_start.elapsed().as_nanos();
    println!(
        "{}",
        json!({"kind":"packed_gate_up_setup","name":f.name,"m":f.rows,"k":f.input_features,"n_per_projection":f.output_features/2,"physical_sets_per_arm":physical_sets,"weight_bytes_per_arm":bytes_per_set*physical_sets,"l2_bytes":l2,"weight_sets":"distinct allocations, same deterministic contents across sets; independent arms; not a cache-eviction proof or full-model stream","repack_ns":repack_ns,"allocation_repack_eager_setup_ns":allocation_repack_eager_setup_ns,"capture_upload_ns":capture_upload_ns,"graph_node_types_A":graphs[0].inventory,"graph_node_types_B":graphs[1].inventory,"atomic_add":false,"fp32_reduce":false})
    );
    let mut original: [Option<Vec<f16>>; 2] = [None, None];
    for generation in 0..2 {
        let input: Vec<_> = host
            .input
            .iter()
            .map(|x| f16::from_f32(x.to_f32() * if generation == 0 { 1.0 } else { 0.5 }))
            .collect();
        for arm in &arms {
            for group in arm {
                group.prepare_replay_input(&stream, &input, f);
            }
        }
        // Fresh matched eager references use each arm's own unchanged weights.
        let eager: [Vec<(Vec<f16>, Vec<f16>)>; 2] = std::array::from_fn(|arm| {
            arms[arm]
                .iter()
                .map(|group| {
                    group.enqueue(runtime, &stream, f, &planar, &merged);
                    (
                        group.projection_values(&stream, f),
                        group.activated.read(&stream),
                    )
                })
                .collect()
        });
        // Poison both intermediate and final outputs after saving the eager
        // references. A graph that fails to write cannot pass with stale eager data.
        for arm in &arms {
            for group in arm {
                group.prepare_replay_input(&stream, &input, f);
            }
        }
        graphs[0].launch();
        graphs[1].launch();
        stream.synchronize().unwrap();
        for set in 0..physical_sets {
            let projection: [Vec<f16>; 2] =
                std::array::from_fn(|arm| arms[arm][set].projection_values(&stream, f));
            let activation: [Vec<f16>; 2] =
                std::array::from_fn(|arm| arms[arm][set].activated.read(&stream));
            for arm in 0..2 {
                assert!(
                    projection[arm]
                        .iter()
                        .zip(&eager[arm][set].0)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "captured projection differs from matched eager"
                );
                assert!(
                    activation[arm]
                        .iter()
                        .zip(&eager[arm][set].1)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "captured activation differs from matched eager"
                );
            }
            if set == 0 {
                for arm in 0..2 {
                    if generation == 0 {
                        original[arm] = Some(activation[arm].clone());
                    } else {
                        assert!(
                            activation[arm]
                                .iter()
                                .zip(original[arm].as_ref().unwrap())
                                .any(|(a, b)| a.to_bits() != b.to_bits()),
                            "fresh input had no observable effect for arm {arm}"
                        );
                    }
                }
            }
            let references: Vec<_> = (0..2)
                .map(|arm| {
                    oracle(
                        f,
                        &host,
                        &input,
                        &projection[arm],
                        &activation[arm],
                        full_oracle,
                    )
                })
                .collect();
            println!(
                "{}",
                json!({"kind":"packed_gate_up_correctness","name":f.name,"generation":generation,"physical_set":set,"projection":difference(&projection[0],&projection[1]),"activation":difference(&activation[0],&activation[1]),"independent_oracles":references,"matched_eager_full_bits":true,"fresh_observable":generation==1})
            );
        }
    }
    // Both arms retain the same last validated input; all uploads, checks and
    // repacking are outside the complete graph replay measurement.
    for _ in 0..2 {
        graphs[0].launch();
        graphs[1].launch();
    }
    stream.synchronize().unwrap();
    if timed {
        for pair in 0..PAIRS {
            let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut result = [serde_json::Value::Null, serde_json::Value::Null];
            for arm in order {
                result[arm] = measure(&stream, &graphs[arm]);
            }
            println!(
                "{}",
                json!({"kind":"packed_gate_up_pair","name":f.name,"pair":pair,"order":if order[0]==0 {"AB"} else {"BA"},"groups_per_graph":physical_sets,"A":result[0],"B":result[1],"scope":"workspace resets + production Marlin projection(s) + SiLU for every group; event record/wait included in block wall"})
            );
        }
    }
    for arm in &arms {
        for group in arm {
            for projection in &group.projections {
                projection.immutable_and_output_guards(&stream);
            }
            group.workspace.read(&stream);
            group.activated.read(&stream);
            group.input.read(&stream);
        }
    }
}

#[test]
#[ignore = "requires actual CUDA and pinned Marlin operator artifact; complete transaction diagnostic"]
fn marlin_ct_packed_gate_up_complete_transaction_pairs() {
    println!(
        "{}",
        json!({"kind":"packed_gate_up_preregistered_contract","product_equivalence":"full intermediate projection and activation half bits must match; differences are reported and timing still runs, not product validation","independent_projection_cpu_relative_l2_limit":CPU_RELATIVE_L2_LIMIT,"independent_projection_scope":"full small case, bounded F64 sample at real geometry; original actual GPTQ/scale/zero bytes","activation_bound":"absolute <= 0.002 + 0.002*abs(reference), full F64 SiLU from each actual half projection; not an A/B equivalence allowance","primary_shape":[32,5120,17408],"pairs":PAIRS,"cold_costs_separate":true})
    );
    let context = CudaContext::new(0).expect("packed CT gate/up probe requires CUDA");
    run_case(
        &context,
        Fixture {
            name: "small-full-oracle",
            rows: 3,
            input_features: 128,
            output_features: 128,
        },
        1,
        true,
        false,
    );
    run_case(
        &context,
        Fixture {
            name: "ct-m32-gate-up",
            rows: 32,
            input_features: 5120,
            output_features: 34816,
        },
        PHYSICAL_SETS,
        false,
        true,
    );
}
