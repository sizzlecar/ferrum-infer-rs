//! Diagnostic only: complete GDN input production, with unequal QKV/Z widths.
//! A successful test does not imply that merging Marlin projections is bit equivalent.
use super::super::test_support::Guarded;
use super::*;
use crate::marlin_repack::{
    fixtures::{build_host_fixture, Fixture, HostFixture, GROUP_SIZE},
    repack_compressed_tensors_zero_points_to_marlin, repack_gptq_to_marlin,
    repack_scales_to_marlin,
};
use cudarc::driver::{sys, CudaContext, CudaSlice, DevicePtr};
use half::f16;
use serde_json::{json, Value};
use std::time::Instant;

const PAIRS: usize = 8;
const PHYSICAL_SETS: usize = 3;
const EPSILON: f32 = 1.0e-6;
// Existing independent CT launcher sanity bound, from transformer/marlin_tests.rs.
// It is not an A/B equivalence allowance: that requires every output half bit.
const EXISTING_CT_REFERENCE_RELATIVE_L2_LIMIT: f64 = 0.05;
const CUBLAS_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
const CUBLAS_WORKSPACE_ALIGNMENT: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Comparison {
    MergeQkvZ,
    ExplicitBlasWorkspace,
}

impl Comparison {
    fn name(self) -> &'static str {
        match self {
            Self::MergeQkvZ => "split_vs_merged_qkv_z",
            Self::ExplicitBlasWorkspace => "split_default_vs_explicit_cublas_workspace",
        }
    }
}

struct BlasWorkspace {
    storage: CudaSlice<u8>,
    pointer: u64,
}

impl BlasWorkspace {
    fn new(stream: &Arc<CudaStream>) -> Self {
        let values = vec![0xa5_u8; CUBLAS_WORKSPACE_BYTES + 2 * CUBLAS_WORKSPACE_ALIGNMENT];
        let storage = stream.clone_htod(&values).unwrap();
        let pointer = storage.device_ptr(stream).0 + CUBLAS_WORKSPACE_ALIGNMENT as u64;
        assert_eq!(pointer % CUBLAS_WORKSPACE_ALIGNMENT as u64, 0);
        // The user workspace is initialized and ready before any capture.
        stream.synchronize().unwrap();
        Self { storage, pointer }
    }

    fn check_guards(&self, stream: &Arc<CudaStream>) {
        stream.synchronize().unwrap();
        let values = stream.clone_dtoh(&self.storage).unwrap();
        assert!(values[..CUBLAS_WORKSPACE_ALIGNMENT]
            .iter()
            .all(|&x| x == 0xa5));
        assert!(
            values[CUBLAS_WORKSPACE_ALIGNMENT + CUBLAS_WORKSPACE_BYTES..]
                .iter()
                .all(|&x| x == 0xa5)
        );
    }
}

struct BlasArms {
    // Handles are destroyed before the workspace. Graphs and the drain below
    // are declared after this owner and therefore retire before either field.
    handles: [Arc<CudaBlas>; 2],
    workspace: Option<BlasWorkspace>,
}

impl BlasArms {
    fn new(stream: &Arc<CudaStream>, comparison: Comparison) -> Self {
        let baseline = Arc::new(CudaBlas::new(Arc::clone(stream)).unwrap());
        if comparison == Comparison::MergeQkvZ {
            return Self {
                handles: [Arc::clone(&baseline), baseline],
                workspace: None,
            };
        }
        let explicit = Arc::new(CudaBlas::new(Arc::clone(stream)).unwrap());
        let workspace = BlasWorkspace::new(stream);
        unsafe {
            use cudarc::cublas::sys as blas_sys;
            // SetStream resets a user workspace. Bind it first, then set the
            // workspace once; neither handle's stream changes during this test.
            assert_eq!(
                blas_sys::cublasSetStream_v2(*explicit.handle(), stream.cu_stream().cast()),
                blas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
            );
            assert_eq!(
                blas_sys::cublasSetWorkspace_v2(
                    *explicit.handle(),
                    workspace.pointer as *mut std::ffi::c_void,
                    CUBLAS_WORKSPACE_BYTES,
                ),
                blas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
            );
        }
        Self {
            handles: [baseline, explicit],
            workspace: Some(workspace),
        }
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    name: &'static str,
    rows: usize,
    k: usize,
    qkv: usize,
    z: usize,
    ba: usize,
}

impl Geometry {
    fn quant_width(self) -> usize {
        self.qkv + self.z
    }

    fn output_width(self) -> usize {
        self.quant_width() + 2 * self.ba
    }

    fn fixture(self) -> Fixture {
        Fixture {
            name: self.name,
            rows: self.rows,
            input_features: self.k,
            output_features: self.quant_width(),
        }
    }
}

struct Source {
    quant: HostFixture,
    norm: Vec<f16>,
    ba: [Vec<f16>; 2],
}

impl Source {
    fn new(g: Geometry, set: usize) -> Self {
        let mut quant = build_host_fixture(g.fixture());
        let n = g.quant_width();
        // Make the unequal segments and physical sets observably distinct.
        for row in quant.scales_grouped.chunks_exact_mut(n) {
            for (column, value) in row.iter_mut().enumerate() {
                let factor = 1.0 + set as f32 * 0.0625 + if column >= g.qkv { 0.125 } else { 0.0 };
                *value = f16::from_f32(value.to_f32() * factor);
            }
        }
        let zero_boundary = g.qkv / 8 * (g.k / GROUP_SIZE);
        for (index, value) in quant.zero_points_compressed_tensors.iter_mut().enumerate() {
            let nibble = ((set + 1 + 2 * usize::from(index >= zero_boundary)) & 7) as u32;
            *value ^= (nibble * 0x1111_1111) as i32;
        }
        let norm = (0..g.k)
            .map(|i| f16::from_f32(0.875 + ((i * 7 + set) % 17) as f32 / 64.0))
            .collect();
        let ba = std::array::from_fn(|part| {
            (0..g.ba * g.k)
                .map(|i| {
                    f16::from_f32((((i * 13 + part * 17 + set * 11) % 47) as f32 - 23.0) / 256.0)
                })
                .collect()
        });
        Self { quant, norm, ba }
    }

    fn input(&self, generation: usize) -> Vec<f16> {
        self.quant
            .input
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if generation == 0 {
                    *x
                } else {
                    // A uniform rescaling could disappear under RMS normalization.
                    f16::from_f32(x.to_f32() * 0.75 + (i % 5) as f32 * 0.03125 - 0.0625)
                }
            })
            .collect()
    }
}

struct Repacked {
    weights: Vec<i32>,
    scales: Vec<f16>,
    zeros: Vec<i32>,
}

impl Repacked {
    fn range(g: Geometry, source: &Source, start: usize, width: usize) -> Self {
        assert!(start + width <= g.quant_width());
        assert!(start.is_multiple_of(64) && width.is_multiple_of(64));
        let q: Vec<_> = source
            .quant
            .qweight_gptq
            .chunks_exact(g.quant_width())
            .flat_map(|row| row[start..start + width].iter().copied())
            .collect();
        let scales: Vec<_> = source
            .quant
            .scales_grouped
            .chunks_exact(g.quant_width())
            .flat_map(|row| row[start..start + width].iter().copied())
            .collect();
        let groups = g.k / GROUP_SIZE;
        let zeros = &source.quant.zero_points_compressed_tensors
            [start / 8 * groups..(start + width) / 8 * groups];
        Self {
            weights: repack_gptq_to_marlin(&q, g.k, width),
            scales: repack_scales_to_marlin(&scales, g.k, width, GROUP_SIZE),
            zeros: repack_compressed_tensors_zero_points_to_marlin(zeros, groups, width),
        }
    }

    fn bytes(&self) -> usize {
        self.weights.len() * 4 + self.scales.len() * 2 + self.zeros.len() * 4
    }
}

struct HalfBuffer {
    storage: Guarded<f16>,
    pointer: u64,
    len: usize,
}

impl HalfBuffer {
    fn new(stream: &Arc<CudaStream>, values: &[f16]) -> Self {
        let storage = Guarded::new(stream, values, f16::from_f32(-117.0));
        let pointer = storage.pointer(stream);
        Self {
            storage,
            pointer,
            len: values.len(),
        }
    }

    fn output(stream: &Arc<CudaStream>, len: usize) -> Self {
        Self::new(stream, &vec![f16::NAN; len])
    }

    fn poison(&self, stream: &Arc<CudaStream>) {
        unsafe {
            cudarc::driver::result::memset_d8_async(
                self.pointer,
                0xff,
                self.len * 2,
                stream.cu_stream(),
            )
        }
        .unwrap();
    }

    fn read(&self, stream: &Arc<CudaStream>) -> Vec<f16> {
        let values = self.storage.read(stream);
        assert!(values.iter().all(|v| v.is_finite()));
        values
    }
}

struct QuantProjection {
    weights: Guarded<i32>,
    scales: Guarded<f16>,
    zeros: Guarded<i32>,
    pointers: [u64; 3],
    output: HalfBuffer,
    width: usize,
    offset: usize,
}

struct DenseProjection {
    weights: HalfBuffer,
    output: HalfBuffer,
}

struct Arm {
    input: HalfBuffer,
    norm_weight: HalfBuffer,
    normalized: HalfBuffer,
    workspace: Guarded<i32>,
    workspace_pointer: u64,
    workspace_bytes: usize,
    quant: Vec<QuantProjection>,
    dense: [DenseProjection; 2],
    output: HalfBuffer,
}

impl Arm {
    fn new(
        stream: &Arc<CudaStream>,
        runtime: MarlinProjectionRuntime,
        g: Geometry,
        source: &Source,
        packed: &[Repacked; 3],
        merged: bool,
    ) -> Self {
        let workspace_bytes = runtime.workspace_bytes().unwrap() as usize;
        let workspace = Guarded::new(stream, &vec![73_i32; workspace_bytes / 4], -117_i32);
        let workspace_pointer = workspace.pointer(stream);
        let parts = if merged {
            vec![(2, 0, g.quant_width())]
        } else {
            vec![(0, 0, g.qkv), (1, g.qkv, g.z)]
        };
        let quant = parts
            .into_iter()
            .map(|(index, offset, width)| {
                let p = &packed[index];
                let weights = Guarded::new(stream, &p.weights, 0x5a5a1234_i32);
                let scales = Guarded::new(stream, &p.scales, f16::from_f32(-117.0));
                let zeros = Guarded::new(stream, &p.zeros, 0x5a5a1234_i32);
                let pointers = [
                    weights.pointer(stream),
                    scales.pointer(stream),
                    zeros.pointer(stream),
                ];
                QuantProjection {
                    weights,
                    scales,
                    zeros,
                    pointers,
                    output: HalfBuffer::output(stream, g.rows * width),
                    width,
                    offset,
                }
            })
            .collect();
        Self {
            input: HalfBuffer::new(stream, &source.quant.input),
            norm_weight: HalfBuffer::new(stream, &source.norm),
            normalized: HalfBuffer::output(stream, g.rows * g.k),
            workspace,
            workspace_pointer,
            workspace_bytes,
            quant,
            dense: std::array::from_fn(|i| DenseProjection {
                weights: HalfBuffer::new(stream, &source.ba[i]),
                output: HalfBuffer::output(stream, g.rows * g.ba),
            }),
            output: HalfBuffer::output(stream, g.rows * g.output_width()),
        }
    }

    fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
        runtime: MarlinProjectionRuntime,
        functions: &Functions,
        g: Geometry,
    ) {
        launch_rms_norm(
            stream,
            &functions.norm,
            self.input.pointer,
            self.norm_weight.pointer,
            self.normalized.pointer,
            g.rows as u64,
            g.k as i32,
            EPSILON,
        )
        .unwrap();
        for part in &self.quant {
            runtime
                .launch(
                    MarlinF16WeightType::U4,
                    stream,
                    self.normalized.pointer,
                    part.pointers[0],
                    part.pointers[1],
                    Some(part.pointers[2]),
                    part.output.pointer,
                    self.workspace_pointer,
                    self.workspace_bytes as u64,
                    g.rows as i32,
                    part.width as i32,
                    g.k as i32,
                    GROUP_SIZE as i32,
                    "GDN input Marlin diagnostic",
                )
                .unwrap();
            launch_projection_stitch(
                stream,
                &functions.stitch,
                part.output.pointer,
                self.output.pointer,
                g.rows as i32,
                part.width as i32,
                g.output_width() as i32,
                part.offset as i32,
                "GDN quantized input stitch",
            )
            .unwrap();
        }
        for (index, part) in self.dense.iter().enumerate() {
            launch_gemm_f16(
                blas,
                self.normalized.pointer,
                part.weights.pointer,
                part.output.pointer,
                g.rows as i32,
                g.ba as i32,
                g.k as i32,
                "GDN floating input projection",
            )
            .unwrap();
            launch_projection_stitch(
                stream,
                &functions.stitch,
                part.output.pointer,
                self.output.pointer,
                g.rows as i32,
                g.ba as i32,
                g.output_width() as i32,
                (g.quant_width() + index * g.ba) as i32,
                "GDN floating input stitch",
            )
            .unwrap();
        }
    }

    fn refill(&self, stream: &Arc<CudaStream>, input: &[f16]) {
        assert_eq!(input.len(), self.input.len);
        unsafe {
            ok(sys::cuMemcpyHtoDAsync_v2(
                self.input.pointer,
                input.as_ptr().cast(),
                std::mem::size_of_val(input),
                stream.cu_stream(),
            ));
            cudarc::driver::result::memset_d8_async(
                self.workspace_pointer,
                73,
                self.workspace_bytes,
                stream.cu_stream(),
            )
            .unwrap();
        }
        self.normalized.poison(stream);
        self.output.poison(stream);
        for p in &self.quant {
            p.output.poison(stream);
        }
        for p in &self.dense {
            p.output.poison(stream);
        }
        // Keep the borrowed pageable source alive through the H2D completion.
        stream.synchronize().unwrap();
        assert_bits(&self.input.read(stream), input, "fresh input upload");
    }

    fn observe(&self, stream: &Arc<CudaStream>, g: Geometry) -> Observation {
        let normalized = self.normalized.read(stream);
        let mut assembled = vec![f16::NAN; g.rows * g.output_width()];
        let mut projection = vec![f16::NAN; g.rows * g.quant_width()];
        for part in &self.quant {
            let values = part.output.read(stream);
            for row in 0..g.rows {
                let values = &values[row * part.width..][..part.width];
                projection[row * g.quant_width() + part.offset..][..part.width]
                    .copy_from_slice(values);
                assembled[row * g.output_width() + part.offset..][..part.width]
                    .copy_from_slice(values);
            }
            part.weights.assert_unchanged(stream);
            part.scales.assert_unchanged(stream);
            part.zeros.assert_unchanged(stream);
        }
        let dense: [Vec<f16>; 2] = std::array::from_fn(|index| {
            self.dense[index].weights.storage.assert_unchanged(stream);
            self.dense[index].output.read(stream)
        });
        for row in 0..g.rows {
            for (index, values) in dense.iter().enumerate() {
                assembled[row * g.output_width() + g.quant_width() + index * g.ba..][..g.ba]
                    .copy_from_slice(&values[row * g.ba..][..g.ba]);
            }
        }
        let output = self.output.read(stream);
        assert_bits(&output, &assembled, "full production stitch layout");
        self.norm_weight.storage.assert_unchanged(stream);
        self.workspace.read(stream);
        Observation {
            normalized,
            projection,
            dense,
            output,
        }
    }
}

struct Functions {
    norm: CudaFunction,
    stitch: CudaFunction,
}
struct Observation {
    normalized: Vec<f16>,
    projection: Vec<f16>,
    dense: [Vec<f16>; 2],
    output: Vec<f16>,
}

fn assert_bits(a: &[f16], b: &[f16], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}");
    assert!(
        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}"
    );
}

fn difference(a: &[f16], b: &[f16]) -> Value {
    assert_eq!(a.len(), b.len());
    let mut bits = 0usize;
    let mut maximum = 0.0f64;
    let mut error = 0.0f64;
    let mut reference = 0.0f64;
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        bits += usize::from(a.to_bits() != b.to_bits());
        maximum = maximum.max((a.to_f64() - b.to_f64()).abs());
        error += (a.to_f64() - b.to_f64()).powi(2);
        reference += a.to_f64().powi(2);
    }
    json!({"elements":a.len(),"different_bits":bits,"exact_bits":bits==0,
        "max_absolute":maximum,"relative_l2":(error/reference.max(f64::MIN_POSITIVE)).sqrt()})
}

fn quant_reference(
    g: Geometry,
    source: &Source,
    normalized: &[f16],
    row: usize,
    col: usize,
) -> f64 {
    let n = g.quant_width();
    let groups = g.k / GROUP_SIZE;
    (0..g.k)
        .map(|i| {
            let q = ((source.quant.qweight_gptq[(i / 8) * n + col] as u32 >> (4 * (i % 8))) & 15)
                as i32;
            let z = ((source.quant.zero_points_compressed_tensors
                [(col / 8) * groups + i / GROUP_SIZE] as u32
                >> (4 * (col % 8)))
                & 15) as i32;
            normalized[row * g.k + i].to_f64()
                * f64::from(q - z)
                * source.quant.scales_grouped[(i / GROUP_SIZE) * n + col].to_f64()
        })
        .sum()
}

fn oracle(g: Geometry, source: &Source, input: &[f16], actual: &Observation, full: bool) -> Value {
    let mut norm_max = 0.0f64;
    for row in 0..g.rows {
        let x = &input[row * g.k..][..g.k];
        let inverse = (x.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() / g.k as f64
            + f64::from(EPSILON))
        .sqrt()
        .recip();
        for col in 0..g.k {
            let reference = x[col].to_f64() * inverse * source.norm[col].to_f64();
            norm_max =
                norm_max.max((actual.normalized[row * g.k + col].to_f64() - reference).abs());
        }
    }
    let points: Vec<_> = if full {
        (0..g.rows)
            .flat_map(|row| (0..g.quant_width()).map(move |col| (row, col)))
            .collect()
    } else {
        [0, g.rows / 2, g.rows - 1]
            .into_iter()
            .flat_map(|row| {
                [0, 1, g.qkv - 1, g.qkv, g.qkv + 1, g.quant_width() - 1].map(|col| (row, col))
            })
            .collect()
    };
    let mut reference_sq = 0.0f64;
    let mut error_sq = 0.0f64;
    let mut quant_max = 0.0f64;
    for &(row, col) in &points {
        let reference = quant_reference(g, source, &actual.normalized, row, col);
        let error = actual.projection[row * g.quant_width() + col].to_f64() - reference;
        reference_sq += reference * reference;
        error_sq += error * error;
        quant_max = quant_max.max(error.abs());
    }
    let relative_l2 = (error_sq / reference_sq.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        relative_l2 < EXISTING_CT_REFERENCE_RELATIVE_L2_LIMIT,
        "independent CT projection sanity: relative_l2={relative_l2}"
    );
    let dense_points: Vec<_> = if full {
        (0..g.rows)
            .flat_map(|row| (0..g.ba).map(move |col| (row, col)))
            .collect()
    } else {
        [0, g.rows / 2, g.rows - 1]
            .into_iter()
            .flat_map(|row| [0, g.ba - 1].map(|col| (row, col)))
            .collect()
    };
    let mut dense_max = 0.0f64;
    for part in 0..2 {
        for &(row, col) in &dense_points {
            let products = (0..g.k).map(|i| {
                actual.normalized[row * g.k + i].to_f64() * source.ba[part][col * g.k + i].to_f64()
            });
            let reference = products.clone().sum::<f64>();
            // Same F32 reduction + F16 output bound as transformer/f16_tests.rs.
            let bound = (g.k as f64 * f64::from(f32::EPSILON) + 0.0009765625)
                * products.map(f64::abs).sum::<f64>()
                + f16::from_bits(1).to_f64();
            let error = (actual.dense[part][row * g.ba + col].to_f64() - reference).abs();
            assert!(
                error <= bound,
                "dense b/a oracle error={error}, bound={bound}"
            );
            dense_max = dense_max.max(error);
        }
    }
    json!({"full_projection_oracle":full,"quant_f64_points":points.len(),
        "quant_reference_relative_l2":relative_l2,"quant_reference_max_absolute":quant_max,
        "dense_f64_points":2*dense_points.len(),"dense_reference_max_absolute":dense_max,
        "norm_full_f64_max_absolute":norm_max,"norm_reference_scope":"observational; unchanged production RMSNorm, independently matched between arms",
        "oracle_input":"each arm actual half-rounded normalized input","product_equivalence_bound":"none; requires complete A/B bit equality"})
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

fn measure(stream: &Arc<CudaStream>, graph: &Graph) -> Value {
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
    json!({"block_wall_ns":wall.elapsed().as_nanos(),"launch_wall_ns":launch_ns,
        "gpu_ns":f64::from(start.elapsed_ms(&end).unwrap())*1e6})
}

fn run_case(
    context: &Arc<CudaContext>,
    g: Geometry,
    sets: usize,
    timed: bool,
    comparison: Comparison,
) -> bool {
    let setup = Instant::now();
    let stream = context.new_stream().unwrap();
    let blas_setup = Instant::now();
    let blas = BlasArms::new(&stream, comparison);
    let blas_setup_ns = blas_setup.elapsed().as_nanos();
    let runtime = MarlinProjectionRuntime {
        multiprocessor_count: context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap(),
        device_ordinal: 0,
    };
    let norm_module = context
        .load_module(Ptx::from_src(crate::ptx::RMS_NORM))
        .unwrap();
    let stitch_module = context
        .load_module(Ptx::from_src(crate::ptx::LINEAR_ATTENTION))
        .unwrap();
    let functions = Functions {
        norm: norm_module.load_function("rms_norm_f16").unwrap(),
        stitch: stitch_module
            .load_function(PROJECTION_STITCH_FUNCTION)
            .unwrap(),
    };
    let sources: Vec<_> = (0..sets).map(|set| Source::new(g, set)).collect();
    let repack = Instant::now();
    let packed: Vec<[Repacked; 3]> = sources
        .iter()
        .map(|source| {
            [
                Repacked::range(g, source, 0, g.qkv),
                Repacked::range(g, source, g.qkv, g.z),
                Repacked::range(g, source, 0, g.quant_width()),
            ]
        })
        .collect();
    let repack_ns = repack.elapsed().as_nanos();
    for p in &packed {
        assert_eq!(p[0].bytes() + p[1].bytes(), p[2].bytes());
    }
    let arms: [Vec<Arm>; 2] = std::array::from_fn(|arm| {
        (0..sets)
            .map(|set| {
                Arm::new(
                    &stream,
                    runtime,
                    g,
                    &sources[set],
                    &packed[set],
                    comparison == Comparison::MergeQkvZ && arm == 1,
                )
            })
            .collect()
    });
    // Declared after all raw-pointer owners so an unwind drains before freeing them.
    let _drain = Drain(Arc::clone(&stream));
    for (index, arm) in arms.iter().enumerate() {
        for set in arm {
            set.enqueue(&stream, &blas.handles[index], runtime, &functions, g);
        }
    }
    stream.synchronize().unwrap();
    let allocation_repack_eager_setup_ns = setup.elapsed().as_nanos();
    let capture = Instant::now();
    let graphs: [Graph; 2] = std::array::from_fn(|arm| {
        Graph::new(&stream, || {
            for set in &arms[arm] {
                set.enqueue(&stream, &blas.handles[arm], runtime, &functions, g);
            }
        })
    });
    println!(
        "{}",
        json!({"kind":"gdn_input_setup","name":g.name,"comparison":comparison.name(),"m":g.rows,"k":g.k,
        "quant_widths":[g.qkv,g.z],"dense_widths":[g.ba,g.ba],"output_width":g.output_width(),
        "physical_sets_per_arm":sets,"quant_weight_bytes_per_arm":packed.iter().map(|p|p[2].bytes()).sum::<usize>(),
        "dense_weight_bytes_per_arm":sets*2*g.ba*g.k*2,"l2_bytes":context.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE).unwrap(),
        "working_set_scope":"independent physical arms; distinct scale/zero/dense contents per rotating set; not proof of DRAM coldness",
        "repack_ns":repack_ns,"allocation_repack_eager_setup_ns":allocation_repack_eager_setup_ns,
        "capture_upload_ns":capture.elapsed().as_nanos(),"cublas_setup_ns":blas_setup_ns,
        "explicit_workspace_bytes_B":if blas.workspace.is_some() {CUBLAS_WORKSPACE_BYTES} else {0},
        "explicit_workspace_alignment_B":if blas.workspace.is_some() {CUBLAS_WORKSPACE_ALIGNMENT} else {0},
        "independent_cublas_handles":comparison==Comparison::ExplicitBlasWorkspace,
        "graph_node_types_A":graphs[0].inventory,
        "graph_node_types_B":graphs[1].inventory,"atomic_add":false,"fp32_reduce":false})
    );
    if comparison == Comparison::ExplicitBlasWorkspace {
        for kind in [
            "CU_GRAPH_NODE_TYPE_MEM_ALLOC",
            "CU_GRAPH_NODE_TYPE_MEM_FREE",
        ] {
            assert_eq!(
                graphs[1].inventory.get(kind).copied().unwrap_or(0),
                0,
                "explicit workspace must remove captured cuBLAS memory nodes"
            );
            if timed {
                assert!(
                    graphs[0].inventory.get(kind).copied().unwrap_or(0) > 0,
                    "the M32 baseline must exercise captured cuBLAS memory nodes"
                );
            }
        }
    }
    let mut exact = true;
    let mut previous: [Vec<Observation>; 2] = std::array::from_fn(|_| Vec::new());
    for generation in 0..2 {
        let inputs: Vec<_> = sources.iter().map(|s| s.input(generation)).collect();
        let refill = || {
            for arm in &arms {
                for (set, input) in arm.iter().zip(&inputs) {
                    set.refill(&stream, input);
                }
            }
        };
        refill();
        let eager: [Vec<Observation>; 2] = std::array::from_fn(|arm| {
            arms[arm]
                .iter()
                .map(|set| {
                    set.enqueue(&stream, &blas.handles[arm], runtime, &functions, g);
                    set.observe(&stream, g)
                })
                .collect()
        });
        refill();
        graphs[0].launch();
        graphs[1].launch();
        let observed: [Vec<Observation>; 2] = std::array::from_fn(|arm| {
            arms[arm]
                .iter()
                .map(|set| set.observe(&stream, g))
                .collect()
        });
        for set in 0..sets {
            for arm in 0..2 {
                assert_bits(
                    &observed[arm][set].output,
                    &eager[arm][set].output,
                    "graph vs eager full output",
                );
                assert_bits(
                    &observed[arm][set].normalized,
                    &eager[arm][set].normalized,
                    "graph vs eager normalized",
                );
                if comparison == Comparison::ExplicitBlasWorkspace {
                    assert_bits(
                        &observed[arm][set].projection,
                        &eager[arm][set].projection,
                        "graph vs eager complete quantized intermediates",
                    );
                    for part in 0..2 {
                        assert_bits(
                            &observed[arm][set].dense[part],
                            &eager[arm][set].dense[part],
                            "graph vs eager complete dense intermediates",
                        );
                    }
                }
                if generation > 0 {
                    assert!(
                        observed[arm][set]
                            .output
                            .iter()
                            .zip(&previous[arm][set].output)
                            .any(|(a, b)| a.to_bits() != b.to_bits()),
                        "fresh input must be observable"
                    );
                }
            }
            assert_bits(
                &observed[0][set].normalized,
                &observed[1][set].normalized,
                "unchanged RMS between arms",
            );
            if comparison == Comparison::MergeQkvZ {
                for part in 0..2 {
                    assert_bits(
                        &observed[0][set].dense[part],
                        &observed[1][set].dense[part],
                        "unchanged b/a between arms",
                    );
                }
            }
            let dense: Vec<_> = (0..2)
                .map(|part| {
                    difference(&observed[0][set].dense[part], &observed[1][set].dense[part])
                })
                .collect();
            let projection = difference(&observed[0][set].projection, &observed[1][set].projection);
            let output = difference(&observed[0][set].output, &observed[1][set].output);
            let equivalent = output["exact_bits"].as_bool().unwrap()
                && projection["exact_bits"].as_bool().unwrap()
                && dense
                    .iter()
                    .all(|part| part["exact_bits"].as_bool().unwrap());
            exact &= equivalent;
            let oracles: Vec<_> = (0..2)
                .map(|arm| oracle(g, &sources[set], &inputs[set], &observed[arm][set], !timed))
                .collect();
            println!(
                "{}",
                json!({"kind":"gdn_input_correctness","name":g.name,"comparison":comparison.name(),"generation":generation,
                "physical_set":set,"projection":projection,"dense":dense,"packed_output":output,"independent_oracles":oracles,
                "product_equivalent":equivalent,"matched_eager_full_bits":true,
                "full_stitch_bits":true,"fresh_observable":generation>0})
            );
        }
        previous = observed;
    }
    for _ in 0..2 {
        graphs[0].launch();
        graphs[1].launch();
    }
    stream.synchronize().unwrap();
    if timed {
        for pair in 0..PAIRS {
            let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut result = [Value::Null, Value::Null];
            for arm in order {
                result[arm] = measure(&stream, &graphs[arm]);
            }
            println!(
                "{}",
                json!({"kind":"gdn_input_pair","name":g.name,"comparison":comparison.name(),"pair":pair,
            "order":if order[0]==0 {"AB"} else {"BA"},"groups_per_graph":sets,
            "A":result[0],"B":result[1],"product_equivalent":exact,
            "scope":"RMSNorm + all Marlin resets/projections + two F16 b/a GEMMs + all stitches; block wall includes event record/wait; uploads/oracles/cold setup excluded"})
            );
        }
    }
    for arm in 0..2 {
        for set in 0..sets {
            let final_state = arms[arm][set].observe(&stream, g);
            assert_bits(
                &final_state.output,
                &previous[arm][set].output,
                "post-timing repeat full bits",
            );
            if comparison == Comparison::ExplicitBlasWorkspace {
                assert_bits(
                    &final_state.normalized,
                    &previous[arm][set].normalized,
                    "post-timing normalized bits",
                );
                assert_bits(
                    &final_state.projection,
                    &previous[arm][set].projection,
                    "post-timing quantized intermediate bits",
                );
                for part in 0..2 {
                    assert_bits(
                        &final_state.dense[part],
                        &previous[arm][set].dense[part],
                        "post-timing dense intermediate bits",
                    );
                }
            }
            assert_bits(
                &arms[arm][set].input.read(&stream),
                &sources[set].input(1),
                "input unchanged during replay",
            );
        }
    }
    if let Some(workspace) = &blas.workspace {
        workspace.check_guards(&stream);
    }
    println!(
        "{}",
        json!({"kind":"gdn_input_result","name":g.name,"comparison":comparison.name(),"product_equivalent":exact,
        "scope":"synthetic complete input-projection diagnostic, not model quality or serving performance"})
    );
    exact
}

#[test]
#[ignore = "requires actual CUDA and pinned Marlin; diagnostic, reports non-equivalence without adopting it"]
fn marlin_ct_gdn_input_projection_complete_transaction_pairs() {
    println!(
        "{}",
        json!({"kind":"gdn_input_contract",
        "product_equivalence":"complete projection/output half bits; differences remain product_equivalent=false even when independent numerical sanity and timing complete",
        "ct_reference_bound_source":"transformer/marlin_tests.rs existing independent CT launcher relative-L2 sanity only",
        "dense_reference_bound_source":"transformer/f16_tests.rs F32 accumulation plus F16 rounding",
        "pairs":PAIRS,"cold_costs_separate":true,"hidden_and_output":"F16",
        "native":"unchanged U4 asymmetric group32, atomic=false, fp32_reduce=false, automatic tiles",
        "excluded":"Core dispatch, prepare/conv/recurrent/state, serving, checkpoint quality"})
    );
    let context = CudaContext::new(0).expect("GDN input projection probe requires CUDA");
    let small = run_case(
        &context,
        Geometry {
            name: "small-unequal-full-oracle",
            rows: 3,
            k: 128,
            qkv: 128,
            z: 64,
            ba: 7,
        },
        1,
        false,
        Comparison::MergeQkvZ,
    );
    let real = run_case(
        &context,
        Geometry {
            name: "ct-m32-gdn-input",
            rows: 32,
            k: 5120,
            qkv: 10240,
            z: 6144,
            ba: 48,
        },
        PHYSICAL_SETS,
        true,
        Comparison::MergeQkvZ,
    );
    println!(
        "{}",
        json!({"kind":"gdn_input_diagnostic_complete","product_equivalent":small&&real,
        "not_product_qualification":true})
    );
}

#[test]
#[ignore = "requires actual CUDA and pinned Marlin; diagnostic explicit cuBLAS workspace comparison"]
fn marlin_ct_gdn_input_projection_explicit_cublas_workspace_pairs() {
    println!(
        "{}",
        json!({"kind":"gdn_input_workspace_contract",
        "comparison":Comparison::ExplicitBlasWorkspace.name(),
        "A":"split QKV/Z with default cuBLAS workspace",
        "B":"same split QKV/Z with independent handle and owned 4MiB aligned workspace",
        "workspace_bytes":CUBLAS_WORKSPACE_BYTES,"workspace_alignment":CUBLAS_WORKSPACE_ALIGNMENT,
        "stream_order":"set stream before binding workspace; no subsequent SetStream",
        "product_equivalence":"all normalized, quantized, dense and stitched F16 bits; no numerical allowance",
        "pairs":PAIRS,"cold_costs_separate":true,"hidden_and_output":"F16",
        "native":"unchanged U4 asymmetric group32, atomic=false, fp32_reduce=false, automatic tiles",
        "limitation":"production graph memory-node inventory is not yet established; observed production GraphLaunch wall largely overlaps GPU execution and is not a recoverable throughput budget",
        "excluded":"Core dispatch, prepare/conv/recurrent/state, serving, checkpoint quality"})
    );
    let context = CudaContext::new(0).expect("cuBLAS workspace probe requires CUDA");
    let small = run_case(
        &context,
        Geometry {
            name: "small-unequal-full-oracle",
            rows: 3,
            k: 128,
            qkv: 128,
            z: 64,
            ba: 7,
        },
        1,
        false,
        Comparison::ExplicitBlasWorkspace,
    );
    let real = run_case(
        &context,
        Geometry {
            name: "ct-m32-gdn-input",
            rows: 32,
            k: 5120,
            qkv: 10240,
            z: 6144,
            ba: 48,
        },
        PHYSICAL_SETS,
        true,
        Comparison::ExplicitBlasWorkspace,
    );
    println!(
        "{}",
        json!({"kind":"gdn_input_workspace_diagnostic_complete","product_equivalent":small&&real,
        "not_product_qualification":true})
    );
    assert!(
        small && real,
        "explicit cuBLAS workspace changed complete transaction bits"
    );
}
