//! Test-only launch fusion of two original strict Q8_0 F16 projections. No
//! activation quantization, concatenated weights or production route is added.
use super::*;
use cudarc::driver::{sys, CudaSlice, DevicePtr};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

const CANARY: u8 = 0x35;
const WEIGHT_PREFIX: usize = 5;

struct Buffer {
    gpu: CudaSlice<u8>,
    bytes: usize,
}
impl Buffer {
    fn new(s: &Arc<CudaStream>, bytes: usize) -> Self {
        Self {
            gpu: s.clone_htod(&vec![CANARY; bytes + 64]).unwrap(),
            bytes,
        }
    }
    fn pointer(&self, s: &Arc<CudaStream>, offset: usize) -> u64 {
        assert!(offset <= self.bytes);
        let (p, lease) = self.gpu.device_ptr(s);
        drop(lease);
        p + 32 + offset as u64
    }
    fn write(&mut self, s: &Arc<CudaStream>, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.bytes);
        s.memcpy_htod(bytes, &mut self.gpu.slice_mut(32..32 + self.bytes))
            .unwrap();
    }
    fn poison(&mut self, s: &Arc<CudaStream>) {
        self.write(s, &vec![CANARY; self.bytes]);
    }
    fn read(&self, s: &Arc<CudaStream>) -> Vec<u8> {
        s.clone_dtoh(&self.gpu.slice(32..32 + self.bytes)).unwrap()
    }
    fn full(&self, s: &Arc<CudaStream>) -> Vec<u8> {
        s.clone_dtoh(&self.gpu).unwrap()
    }
    fn guards(&self, s: &Arc<CudaStream>) {
        let b = self.full(s);
        assert!(b[..32]
            .iter()
            .chain(&b[32 + self.bytes..])
            .all(|v| *v == CANARY));
    }
}
fn half_bytes(x: &[f16]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}
fn half_read(x: &[u8]) -> Vec<u16> {
    x.chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
struct Weight {
    bytes: Vec<u8>,
    gpu: Buffer,
}
struct Case {
    m: usize,
    k: usize,
    n: [usize; 2],
    stride: usize,
    offsets: [usize; 2],
    x: Vec<f16>,
    input: Buffer,
    weights: Vec<Weight>,
    old_output: Buffer,
    pair_output: Buffer,
}
impl Case {
    fn new(s: &Arc<CudaStream>, m: usize, k: usize, n: [usize; 2], layers: usize) -> Self {
        assert!(k % 32 == 0 && n.iter().all(|v| *v > 0));
        let (stride, offsets) = if (k, n) == (5120, [48, 48]) {
            (16480, [16384, 16432])
        } else {
            (n[0] + n[1] + 9, [2, n[0] + 5])
        };
        let x = (0..m * k)
            .map(|i| f16::from_f32((((i / k) * 13 + (i % k) * 19) % 127) as f32 / 128.0 - 0.5))
            .collect::<Vec<_>>();
        let mut input = Buffer::new(s, x.len() * 2);
        input.write(s, &half_bytes(&x));
        let weights = (0..2 * layers)
            .map(|salt| {
                let mut bytes = vec![CANARY; WEIGHT_PREFIX];
                for block in 0..n[salt % 2] * (k / 32) {
                    let scale = match (block + salt) % 11 {
                        0 => f16::from_bits(0x8000),
                        1 => f16::from_bits(1),
                        _ => f16::from_f32(
                            (1 + (block + salt) % 5) as f32 / 4096.0
                                * if (block + salt) % 2 == 0 { 1.0 } else { -1.0 },
                        ),
                    };
                    bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
                    bytes.extend((0..32).map(|i| ((block * 7 + i * 29 + salt * 11) % 256) as u8));
                }
                let mut gpu = Buffer::new(s, bytes.len());
                gpu.write(s, &bytes);
                Weight { bytes, gpu }
            })
            .collect();
        Self {
            m,
            k,
            n,
            stride,
            offsets,
            x,
            input,
            weights,
            old_output: Buffer::new(s, m * stride * 2),
            pair_output: Buffer::new(s, m * stride * 2),
        }
    }
    fn upload(&mut self, s: &Arc<CudaStream>) {
        self.input.write(s, &half_bytes(&self.x));
        for w in &mut self.weights {
            w.gpu.write(s, &w.bytes);
        }
        s.synchronize().unwrap();
    }
    fn reset(&mut self, s: &Arc<CudaStream>) {
        self.old_output.poison(s);
        self.pair_output.poison(s);
        s.synchronize().unwrap();
    }
    fn old(&self, s: &Arc<CudaStream>, kernels: &CudaNativeBlockKernels, index: usize) {
        let leaf = index % 2;
        let x = self.input.pointer(s, 0);
        let w = self.weights[index].gpu.pointer(s, WEIGHT_PREFIX);
        let y = self.old_output.pointer(s, 0);
        let args = [
            self.m as u32,
            self.k as u32,
            self.n[leaf] as u32,
            self.stride as u32,
            self.offsets[leaf] as u32,
            8,
            32,
            34,
        ];
        let mut l = s.launch_builder(&kernels.linear_q8_0_f16);
        l.arg(&x).arg(&w).arg(&y);
        for a in &args {
            l.arg(a);
        }
        // SAFETY: old T1/128-thread geometry, complete immutable weight/input
        // views and independently bounded output interval for this leaf.
        unsafe {
            l.launch(LaunchConfig {
                grid_dim: ((self.n[leaf] as u32).div_ceil(4), self.m as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    fn pair(&self, s: &Arc<CudaStream>, kernels: &CudaNativeBlockKernels, layer: usize) {
        let x = self.input.pointer(s, 0);
        let w0 = self.weights[2 * layer].gpu.pointer(s, WEIGHT_PREFIX);
        let w1 = self.weights[2 * layer + 1].gpu.pointer(s, WEIGHT_PREFIX);
        let y = self.pair_output.pointer(s, 0);
        let args = [
            self.m as u32,
            self.k as u32,
            self.n[0] as u32,
            self.n[1] as u32,
            self.stride as u32,
            self.offsets[0] as u32,
            self.offsets[1] as u32,
        ];
        let mut l = s.launch_builder(&kernels.linear_q8_0_pair_f16);
        l.arg(&x).arg(&w0).arg(&w1).arg(&y);
        for a in &args {
            l.arg(a);
        }
        // SAFETY: z selects one independently retained matrix/interval. Each
        // CTA preserves the original T1 body, K order, warp and block geometry.
        unsafe {
            l.launch(LaunchConfig {
                grid_dim: (
                    (self.n[0].max(self.n[1]) as u32).div_ceil(4),
                    self.m as u32,
                    2,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    fn guards(&self, s: &Arc<CudaStream>) {
        for b in [&self.input, &self.old_output, &self.pair_output] {
            b.guards(s);
        }
        assert_eq!(self.input.read(s), half_bytes(&self.x));
        for w in &self.weights {
            w.gpu.guards(s);
            assert_eq!(w.gpu.read(s), w.bytes);
            assert!(w.bytes[..WEIGHT_PREFIX].iter().all(|v| *v == CANARY));
        }
        for output in [&self.old_output, &self.pair_output] {
            let y = half_read(&output.read(s));
            for row in 0..self.m {
                for col in 0..self.stride {
                    if !(0..2).any(|leaf| {
                        (self.offsets[leaf]..self.offsets[leaf] + self.n[leaf]).contains(&col)
                    }) {
                        assert_eq!(
                            y[row * self.stride + col],
                            0x3535,
                            "untouched output [{row},{col}]"
                        );
                    }
                }
            }
        }
    }
    fn equal(&self, s: &Arc<CudaStream>) {
        s.synchronize().unwrap();
        assert_eq!(
            self.pair_output.full(s),
            self.old_output.full(s),
            "fused strict changed any output or guard bits"
        );
    }
    fn oracle(&self, s: &Arc<CudaStream>, layer: usize) -> serde_json::Value {
        let y = half_read(&self.old_output.read(s));
        let mut max_error = 0.0_f64;
        let mut checked = 0;
        for row in 0..self.m {
            for leaf in 0..2 {
                for col in 0..self.n[leaf] {
                    let w = &self.weights[2 * layer + leaf].bytes;
                    let mut target = 0.0;
                    let mut magnitude = 0.0;
                    for i in 0..self.k {
                        // Literal Q8_0 wire oracle independent of shared/native decode.
                        let b = WEIGHT_PREFIX + (col * (self.k / 32) + i / 32) * 34;
                        let scale = f16::from_bits(u16::from_le_bytes([w[b], w[b + 1]])).to_f32();
                        let weight = f64::from(scale) * f64::from(w[b + 2 + i % 32] as i8);
                        let term = weight * f64::from(self.x[row * self.k + i].to_f32());
                        target += term;
                        magnitude += term.abs();
                    }
                    let actual = f64::from(
                        f16::from_bits(y[row * self.stride + self.offsets[leaf] + col]).to_f32(),
                    );
                    let epsilon = f64::from(f32::EPSILON);
                    let operations = (2 * self.k + 64) as f64;
                    let gamma = operations * epsilon / (1.0 - operations * epsilon);
                    let bound = gamma * magnitude + target.abs() / 2048.0 + 2.0_f64.powi(-24);
                    assert!(
                        actual.is_finite() && (actual - target).abs() <= bound,
                        "strict F64 [{row},{leaf},{col}] {actual} != {target}, bound{bound}"
                    );
                    max_error = max_error.max((actual - target).abs());
                    checked += 1;
                }
            }
        }
        serde_json::json!({"oracle":"independent_q8_weight_original_f16_input_f64","checked_outputs":checked,"max_absolute_error":max_error})
    }
}

struct Captured {
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    nodes: usize,
}
impl Captured {
    fn new(s: &Arc<CudaStream>, enqueue: impl FnOnce()) -> Self {
        s.context().bind_to_thread().unwrap();
        s.synchronize().unwrap();
        assert_eq!(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    s.cu_stream(),
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
            },
            sys::CUresult::CUDA_SUCCESS
        );
        let encoded = catch_unwind(AssertUnwindSafe(enqueue));
        let mut c = Self {
            graph: std::ptr::null_mut(),
            executable: std::ptr::null_mut(),
            stream: s.clone(),
            nodes: 0,
        };
        let end = unsafe { sys::cuStreamEndCapture(s.cu_stream(), &mut c.graph) };
        let mut state = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED;
        assert_eq!(
            unsafe { sys::cuStreamIsCapturing(s.cu_stream(), &mut state) },
            sys::CUresult::CUDA_SUCCESS
        );
        if state != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            let mut abandoned = std::ptr::null_mut();
            unsafe {
                sys::cuStreamEndCapture(s.cu_stream(), &mut abandoned);
                if !abandoned.is_null() {
                    sys::cuGraphDestroy(abandoned);
                }
            }
            assert_eq!(
                unsafe { sys::cuStreamIsCapturing(s.cu_stream(), &mut state) },
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                state,
                sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
            );
        }
        if let Err(panic) = encoded {
            resume_unwind(panic);
        }
        assert_eq!(end, sys::CUresult::CUDA_SUCCESS);
        assert!(!c.graph.is_null());
        assert_eq!(
            unsafe { sys::cuGraphGetNodes(c.graph, std::ptr::null_mut(), &mut c.nodes) },
            sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            unsafe { sys::cuGraphInstantiateWithFlags(&mut c.executable, c.graph, 0) },
            sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            unsafe { sys::cuGraphUpload(c.executable, s.cu_stream()) },
            sys::CUresult::CUDA_SUCCESS
        );
        s.synchronize().unwrap();
        c
    }
    fn launch(&self) {
        assert_eq!(
            unsafe { sys::cuGraphLaunch(self.executable, self.stream.cu_stream()) },
            sys::CUresult::CUDA_SUCCESS
        );
    }
}
impl Drop for Captured {
    fn drop(&mut self) {
        if self.stream.context().bind_to_thread().is_err() {
            return;
        }
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
fn context() -> (Arc<CudaContext>, Arc<CudaStream>, CudaNativeBlockKernels) {
    let c = CudaContext::new(0).expect("strict Q8 pair pilot requires an actual CUDA device");
    // The fixture owns allocations until every graph drops, with explicit fences.
    unsafe { c.disable_event_tracking() };
    let s = c.new_stream().unwrap();
    let k = CudaNativeBlockKernels::load(&c).unwrap();
    (c, s, k)
}
fn measure(s: &Arc<CudaStream>, enqueue: impl FnOnce()) -> (f64, f64) {
    let wall = std::time::Instant::now();
    let start = s
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    enqueue();
    let end = s
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    (
        wall.elapsed().as_secs_f64() * 1e9,
        f64::from(start.elapsed_ms(&end).unwrap()) * 1e6,
    )
}

#[test]
#[ignore = "strict Q8 pair direct/graph bit oracle requires actual CUDA"]
fn q8_strict_pair_preserves_independent_leaf_bits_and_all_guards_on_cuda() {
    let (_, s, kernels) = context();
    for (m, k, n) in [(3, 96, [7, 17]), (8, 5120, [48, 48]), (32, 5120, [48, 48])] {
        let mut c = Case::new(&s, m, k, n, 1);
        c.old(&s, &kernels, 0);
        c.old(&s, &kernels, 1);
        c.pair(&s, &kernels, 0);
        c.equal(&s);
        c.guards(&s);
        let oracle = c.oracle(&s, 0);
        let old = Captured::new(&s, || {
            c.old(&s, &kernels, 0);
            c.old(&s, &kernels, 1);
        });
        let pair = Captured::new(&s, || c.pair(&s, &kernels, 0));
        let ordinary = c.x.clone();
        let good = c
            .weights
            .iter()
            .map(|w| w.bytes.clone())
            .collect::<Vec<_>>();
        for pattern in 0..9 {
            c.x = ordinary.clone();
            for (w, b) in c.weights.iter_mut().zip(&good) {
                w.bytes = b.clone();
            }
            match pattern {
                0 => c.x.fill(f16::from_bits(0x8000)),
                1 => {
                    for (i, v) in c.x.iter_mut().enumerate() {
                        *v = f16::from_bits(1 + (i % 1023) as u16);
                    }
                }
                2 => c.x[(m - 1) * k] = f16::NAN,
                3 => c.x[(m - 1) * k] = f16::INFINITY,
                4 => c.x[(m - 1) * k] = f16::NEG_INFINITY,
                5 => c.weights[0].bytes[WEIGHT_PREFIX..WEIGHT_PREFIX + 2]
                    .copy_from_slice(&0x7e00_u16.to_le_bytes()),
                6 => c.weights[0].bytes[WEIGHT_PREFIX..WEIGHT_PREFIX + 2]
                    .copy_from_slice(&0x7c00_u16.to_le_bytes()),
                7 => c.weights[1].bytes[WEIGHT_PREFIX..WEIGHT_PREFIX + 2]
                    .copy_from_slice(&0xfc00_u16.to_le_bytes()),
                _ => {} // recover ordinary inputs/weights on the same captured addresses
            }
            c.upload(&s);
            c.reset(&s);
            c.old(&s, &kernels, 0);
            c.old(&s, &kernels, 1);
            c.pair(&s, &kernels, 0);
            c.equal(&s);
            c.guards(&s);
            let reference = c.old_output.full(&s);
            old.launch();
            pair.launch();
            old.launch();
            pair.launch();
            c.equal(&s);
            c.guards(&s);
            assert_eq!(
                reference,
                c.pair_output.full(&s),
                "direct/captured strict pair bits"
            );
        }
        drop(pair);
        drop(old);
        c.x = ordinary;
        for (w, b) in c.weights.iter_mut().zip(good) {
            w.bytes = b;
        }
        // Swapped, noncontiguous output windows ensure matrix identity is not
        // inferred from adjacency, a physical owner, or a flattened column.
        c.offsets = if n[0] == n[1] {
            [c.offsets[1], c.offsets[0]]
        } else {
            [n[1] + 5, 2]
        };
        c.upload(&s);
        c.reset(&s);
        c.old(&s, &kernels, 0);
        c.old(&s, &kernels, 1);
        c.pair(&s, &kernels, 0);
        c.equal(&s);
        c.guards(&s);
        c.oracle(&s, 0);
        println!(
            "{}",
            serde_json::json!({"kind":"q8_strict_pair_correctness","m":m,"k":k,"n":n,"oracle":oracle,
            "full_allocation_bit_equality":true,"direct_and_graph":true,"nonfinite_independent_leaf_bits":true,
            "odd_weight_byte_views":true,"swapped_output_windows":true,"activation_quantization":false})
        );
    }
}

#[test]
#[ignore = "48-layer strict Q8 pair inclusive comparison requires exclusive CUDA"]
fn q8_strict_pair_48_layer_inclusive_paired_on_cuda() {
    let (_, s, kernels) = context();
    const LAYERS: usize = 48;
    for m in [8, 32] {
        let mut c = Case::new(&s, m, 5120, [48, 48], LAYERS);
        c.reset(&s);
        for layer in 0..LAYERS {
            c.old(&s, &kernels, 2 * layer);
            c.old(&s, &kernels, 2 * layer + 1);
            c.pair(&s, &kernels, layer);
        }
        c.equal(&s);
        c.guards(&s);
        let oracle = c.oracle(&s, LAYERS - 1);
        let initial = c.old_output.full(&s);
        let old = Captured::new(&s, || {
            for i in 0..2 * LAYERS {
                c.old(&s, &kernels, i);
            }
        });
        let pair = Captured::new(&s, || {
            for layer in 0..LAYERS {
                c.pair(&s, &kernels, layer);
            }
        });
        for _ in 0..2 {
            old.launch();
            pair.launch();
        }
        s.synchronize().unwrap();
        for repetition in 0..8 {
            for position in 0..2 {
                let fused = (repetition + position) % 2 == 1;
                let (wall, gpu) = measure(&s, || {
                    if fused {
                        pair.launch();
                    } else {
                        old.launch();
                    }
                });
                let actual = if fused {
                    c.pair_output.full(&s)
                } else {
                    c.old_output.full(&s)
                };
                assert_eq!(
                    actual, initial,
                    "timed replay changed strict output or guards"
                );
                println!(
                    "{}",
                    serde_json::json!({"kind":"q8_strict_pair_inclusive_paired","synthetic":true,"m":m,"k":5120,"n_per_leaf":48,
                "layers":LAYERS,"leaves_per_layer":2,"distinct_weight_matrices":c.weights.len(),
                "logical_weight_bytes":c.weights.iter().map(|w|w.bytes.len()-WEIGHT_PREFIX).sum::<usize>(),
                "route":if fused {"strict_q8_t1_two_leaf_single_launch"} else {"original_strict_q8_t1_two_launches"},
                "mode":"graph_replay","pair":repetition,"position":position,"graph_nodes":if fused {pair.nodes} else {old.nodes},
                "kernel_launches":if fused {LAYERS} else {2*LAYERS},"row_tile":1,"threads":128,
                "wall_ns":wall,"gpu_ns":gpu,"wall_ns_per_original_leaf_pair":wall/LAYERS as f64,"gpu_ns_per_original_leaf_pair":gpu/LAYERS as f64,
                "output_stride":c.stride,"output_offsets":c.offsets,"activation_quantization":false,"weight_concat":false,
                "same_original_arithmetic":true,"inclusive_original_projection_stages":true,"performance_assertion":null,"semantic_quality":"not evaluated"})
                );
            }
        }
        c.equal(&s);
        c.guards(&s);
        println!(
            "{}",
            serde_json::json!({"kind":"q8_strict_pair_paired_summary","m":m,"oracle":oracle,"full_allocation_bit_equality":true,
            "baseline_row_tile":1,"candidate_row_tile":1,"same_weights_and_input":true,"same_route_repeat_bits":true,"activation_quantization":false})
        );
    }
}
