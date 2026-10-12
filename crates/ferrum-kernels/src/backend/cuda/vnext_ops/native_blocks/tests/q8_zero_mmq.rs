//! Test-only Q8_0 MMQ pilot: complete F16 boundary pipeline versus the retained
//! strict dispatch. Synthetic projection evidence cannot establish model quality.
use super::*;
use crate::native_ops::upstream_linear::{
    Algorithm, Arithmetic, Device, DeviceSpan, Format, Layout, PreparedUpstreamLinear,
};
use cudarc::driver::{sys, CudaSlice, DevicePtr};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

const CANARY: u8 = 0x35;

struct Buffer {
    allocation: CudaSlice<u8>,
    bytes: usize,
}
impl Buffer {
    fn new(s: &Arc<CudaStream>, bytes: usize) -> Self {
        Self {
            allocation: s.clone_htod(&vec![CANARY; bytes + 64]).unwrap(),
            bytes,
        }
    }
    fn span_at(&self, s: &Arc<CudaStream>, offset: usize) -> DeviceSpan {
        assert!(offset <= self.bytes);
        if self.bytes == 0 {
            return DeviceSpan {
                address: 0,
                bytes: 0,
            };
        }
        let (pointer, lease) = self.allocation.device_ptr(s);
        drop(lease);
        DeviceSpan {
            address: pointer + 32 + offset as u64,
            bytes: (self.bytes - offset) as u64,
        }
    }
    fn span(&self, s: &Arc<CudaStream>) -> DeviceSpan {
        self.span_at(s, 0)
    }
    fn write(&mut self, s: &Arc<CudaStream>, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.bytes);
        if !bytes.is_empty() {
            s.memcpy_htod(bytes, &mut self.allocation.slice_mut(32..32 + self.bytes))
                .unwrap();
        }
    }
    fn poison(&mut self, s: &Arc<CudaStream>) {
        self.write(s, &vec![CANARY; self.bytes]);
    }
    fn read(&self, s: &Arc<CudaStream>) -> Vec<u8> {
        s.clone_dtoh(&self.allocation.slice(32..32 + self.bytes))
            .unwrap()
    }
    fn guards(&self, s: &Arc<CudaStream>) {
        let b = s.clone_dtoh(&self.allocation).unwrap();
        assert!(b[..32]
            .iter()
            .chain(&b[32 + self.bytes..])
            .all(|v| *v == CANARY));
    }
}
fn half_bytes(x: &[f16]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}
fn halves(x: &[u8]) -> Vec<u16> {
    x.chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn floats(x: &[u8]) -> Vec<f32> {
    x.chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn device(c: &CudaContext) -> Device {
    use sys::CUdevice_attribute::*;
    let a = |key| u32::try_from(c.attribute(key).unwrap()).unwrap();
    Device {
        architecture: a(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 100
            + a(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) * 10,
        multiprocessors: a(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        maximum_dynamic_shared_bytes: u64::from(a(
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )),
    }
}
fn context() -> (Arc<CudaContext>, Arc<CudaStream>, CudaNativeBlockKernels) {
    let c = CudaContext::new(0).expect("Q8_0 MMQ pilot requires an actual CUDA device");
    // One stream owns all buffers through graph destruction and explicit fences.
    unsafe { c.disable_event_tracking() };
    let s = c.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&c).unwrap();
    (c, s, kernels)
}

struct Weight {
    bytes: Vec<u8>,
    gpu: Buffer,
    flag: Buffer,
}
fn weight_value(w: &[u8], k: usize, col: usize, index: usize) -> f64 {
    // Independent literal Q8_0 decode: half scale then 32 signed bytes. This
    // does not call Ferrum's decoder or the native quantizer under test.
    let block = (col * (k / 32) + index / 32) * 34;
    let scale = f16::from_bits(u16::from_le_bytes([w[block], w[block + 1]])).to_f32();
    f64::from(scale) * f64::from(w[block + 2 + index % 32] as i8)
}
struct Case {
    m: usize,
    k: usize,
    n: usize,
    stride: usize,
    offsets: [usize; 2],
    x: Vec<f16>,
    input: Buffer,
    weights: Vec<Weight>,
    native: PreparedUpstreamLinear,
    native_preparation_wall_ns: f64,
    converted: Buffer,
    packed: Buffer,
    raw: Buffer,
    fixup: Buffer,
    row_flags: Buffer,
    output: Buffer,
    strict_output: Buffer,
}
impl Case {
    fn new(s: &Arc<CudaStream>, m: usize, k: usize, n: usize, layers: usize) -> Self {
        let preparation = std::time::Instant::now();
        let native = PreparedUpstreamLinear::new(
            Algorithm::Mmq,
            Arithmetic::MarkerV2,
            Format::Q8_0,
            Layout::Columns,
            m as u32,
            k as u32,
            n as u32,
            device(s.context()),
        )
        .unwrap();
        let native_preparation_wall_ns = preparation.elapsed().as_secs_f64() * 1e9;
        let p = *native.geometry();
        assert_eq!(p.pack_abi, 1, "Q8_0 requires actual D4 F32-scale packing");
        assert_eq!(p.weight_bytes, (n * (k / 32) * 34) as u64);
        assert_eq!(p.output_bytes, (m * n * 4) as u64);
        let (stride, offsets) = if (k, n) == (5120, 48) {
            (16480, [16384, 16432])
        } else {
            (2 * n + 9, [2, n + 5])
        };
        let x = (0..m * k)
            .map(|i| f16::from_f32((((i / k) * 13 + (i % k) * 19) % 127) as f32 / 128.0 - 0.5))
            .collect::<Vec<_>>();
        let mut input = Buffer::new(s, x.len() * 2);
        input.write(s, &half_bytes(&x));
        let weights = (0..2 * layers)
            .map(|salt| {
                let mut bytes = Vec::with_capacity(p.weight_bytes as usize);
                for block in 0..n * (k / 32) {
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
                Weight {
                    bytes,
                    gpu,
                    flag: Buffer::new(s, 4),
                }
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
            native,
            native_preparation_wall_ns,
            converted: Buffer::new(s, p.converted_bytes as usize),
            packed: Buffer::new(s, p.packed_bytes as usize),
            raw: Buffer::new(s, p.output_bytes as usize),
            fixup: Buffer::new(s, p.fixup_bytes as usize),
            row_flags: Buffer::new(s, m * 4),
            output: Buffer::new(s, m * stride * 2),
            strict_output: Buffer::new(s, m * stride * 2),
        }
    }
    fn scan(&self, s: &Arc<CudaStream>) {
        for w in &self.weights {
            // SAFETY: independently guarded, aligned, live same-device buffers;
            // the exact weight/flag association remains retained through replay.
            unsafe {
                self.native
                    .check_weights(w.gpu.span(s), w.flag.span(s), s.cu_stream().cast())
            }
            .unwrap();
        }
    }
    fn upload(&mut self, s: &Arc<CudaStream>) {
        self.input.write(s, &half_bytes(&self.x));
        for w in &mut self.weights {
            w.gpu.write(s, &w.bytes);
        }
        self.scan(s);
        s.synchronize().unwrap();
    }
    fn reset(&mut self, s: &Arc<CudaStream>) {
        for b in [
            &mut self.converted,
            &mut self.packed,
            &mut self.raw,
            &mut self.fixup,
            &mut self.row_flags,
            &mut self.output,
            &mut self.strict_output,
        ] {
            b.poison(s);
        }
        s.synchronize().unwrap();
    }
    fn inclusive(&self, s: &Arc<CudaStream>, index: usize) {
        let w = &self.weights[index];
        // SAFETY: sequential same-stream use retains every allocation; native
        // pack includes row-flag reset, conversion, K padding and packed guard.
        unsafe {
            self.native
                .pack(
                    self.input.span(s),
                    self.k as u32,
                    self.converted.span(s),
                    self.packed.span(s),
                    self.row_flags.span(s),
                    s.cu_stream().cast(),
                )
                .unwrap();
            self.native
                .dot(
                    w.gpu.span(s),
                    self.packed.span(s),
                    self.raw.span(s),
                    self.fixup.span(s),
                    s.cu_stream().cast(),
                )
                .unwrap();
            self.native
                .cast(
                    self.raw.span(s),
                    self.output.span_at(s, self.offsets[index % 2] * 2),
                    self.stride as u32,
                    self.row_flags.span(s),
                    w.flag.span(s),
                    s.cu_stream().cast(),
                )
                .unwrap();
        }
    }
    fn strict(&self, s: &Arc<CudaStream>, kernels: &CudaNativeBlockKernels, index: usize) {
        let tile = q8_0_f16_row_tile(
            self.m as u32,
            self.k as u32,
            self.n as u32,
            kernels.multiprocessors,
        );
        // Match the retained production dispatcher, including its Q8_0 T1
        // selection for narrow large-K output and generic T8 alternative.
        let kernel = if tile == 1 {
            &kernels.linear_q8_0_f16
        } else {
            &kernels.linear_tiled_f16
        };
        let x = self.input.span(s).address;
        let w = self.weights[index].gpu.span(s).address;
        let y = self.strict_output.span(s).address;
        let args = [
            self.m as u32,
            self.k as u32,
            self.n as u32,
            self.stride as u32,
            self.offsets[index % 2] as u32,
            8,
            32,
            34,
        ];
        let mut launch = s.launch_builder(kernel);
        launch.arg(&x).arg(&w).arg(&y);
        for arg in &args {
            launch.arg(arg);
        }
        // SAFETY: all matrices and offsets have full guarded extents; kernels
        // guard their partial column/row tiles and consume Q8_0 block bytes.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    (self.n as u32).div_ceil(4),
                    (self.m as u32).div_ceil(tile),
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    fn guards(&self, s: &Arc<CudaStream>) {
        for b in [
            &self.input,
            &self.converted,
            &self.packed,
            &self.raw,
            &self.fixup,
            &self.row_flags,
            &self.output,
            &self.strict_output,
        ] {
            b.guards(s);
        }
        assert_eq!(self.input.read(s), half_bytes(&self.x));
        for w in &self.weights {
            w.gpu.guards(s);
            w.flag.guards(s);
            assert_eq!(w.gpu.read(s), w.bytes);
            let bad = w
                .bytes
                .chunks_exact(34)
                .any(|b| !f16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite());
            assert_eq!(
                u32::from_le_bytes(w.flag.read(s).try_into().unwrap()) != 0,
                bad
            );
        }
    }
    fn activation(&self, bytes: &[u8], row: usize, index: usize) -> (i8, f32) {
        let g = index / 32;
        let base = (g / 4 * self.m + row) * 144;
        let d = f32::from_le_bytes(
            bytes[base + g % 4 * 4..base + g % 4 * 4 + 4]
                .try_into()
                .unwrap(),
        );
        (bytes[base + 16 + g % 4 * 32 + index % 32] as i8, d)
    }
    fn validate_pack(&self, s: &Arc<CudaStream>) -> Vec<u8> {
        let packed = self.packed.read(s);
        let converted = floats(&self.converted.read(s));
        let flags = self.row_flags.read(s);
        let padded = self.native.geometry().padded_inputs as usize;
        for row in 0..self.m {
            let bad = self.x[row * self.k..(row + 1) * self.k]
                .iter()
                .any(|x| !x.is_finite());
            assert_eq!(
                u32::from_le_bytes(flags[row * 4..row * 4 + 4].try_into().unwrap()) != 0,
                bad
            );
            for i in 0..padded {
                let expected = if i < self.k {
                    self.x[row * self.k + i].to_f32()
                } else {
                    0.0
                };
                let actual = converted[row * padded + i];
                assert!(
                    actual.to_bits() == expected.to_bits()
                        || (actual.is_nan() && expected.is_nan())
                );
            }
            for g in 0..padded / 32 {
                let (first, d) = self.activation(&packed, row, g * 32);
                let group = (g * 32 < self.k)
                    .then(|| &self.x[row * self.k + g * 32..row * self.k + (g + 1) * 32]);
                let invalid = group.is_some_and(|x| x.iter().any(|v| !v.is_finite()));
                let amax = group.map_or(0.0, |x| {
                    x.iter().map(|v| v.to_f32().abs()).fold(0.0_f32, f32::max)
                });
                if invalid || amax == 0.0 {
                    assert_eq!(d.to_bits(), if invalid { 0x7fc00000 } else { 0 });
                    assert_eq!(first, 0);
                } else {
                    let ideal = amax / 127.0;
                    assert!(d.is_finite() && (d - ideal).abs() <= ideal * 8.0 * f32::EPSILON);
                }
                for i in 0..32 {
                    let (q, scale) = self.activation(&packed, row, g * 32 + i);
                    assert_eq!(scale.to_bits(), d.to_bits());
                    if invalid || amax == 0.0 {
                        assert_eq!(q, 0);
                    } else {
                        assert_ne!(q, i8::MIN);
                        let x = f64::from(group.unwrap()[i].to_f32());
                        // Compiled fast-math reciprocal need not match ideal
                        // IEEE division at half-cell ties; bound actual cells.
                        assert!(
                            (x - f64::from(q) * f64::from(d)).abs()
                                <= f64::from(d) * 0.501
                                    + f64::from(amax) * 8.0 * f64::from(f32::EPSILON)
                        );
                    }
                }
            }
        }
        assert!(packed[self.m * padded * 9 / 8..].iter().all(|v| *v == 0));
        packed
    }
    fn oracle(&self, s: &Arc<CudaStream>, indices: &[usize], strict: bool) -> serde_json::Value {
        s.synchronize().unwrap();
        let packed = (!strict).then(|| self.validate_pack(s));
        let y = halves(&if strict {
            self.strict_output.read(s)
        } else {
            self.output.read(s)
        });
        let mut quant_error = 0.0_f64;
        let mut strict_error = 0.0_f64;
        let mut checked = 0;
        for row in 0..self.m {
            for column in 0..self.stride {
                let leaf = indices.iter().copied().find(|i| {
                    (self.offsets[i % 2]..self.offsets[i % 2] + self.n).contains(&column)
                });
                let Some(index) = leaf else {
                    assert_eq!(y[row * self.stride + column], 0x3535);
                    continue;
                };
                let col = column - self.offsets[index % 2];
                let weights = &self.weights[index].bytes;
                let weight_bad = weights
                    .chunks_exact(34)
                    .any(|b| !f16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite());
                let row_bad = self.x[row * self.k..(row + 1) * self.k]
                    .iter()
                    .any(|v| !v.is_finite());
                if row_bad || weight_bad {
                    if !strict {
                        assert_eq!(y[row * self.stride + column], 0x7e00);
                    }
                    continue;
                }
                let mut target = 0.0;
                let mut magnitude = 0.0;
                let mut original = 0.0;
                for i in 0..self.k {
                    let w = weight_value(weights, self.k, col, i);
                    let original_x = f64::from(self.x[row * self.k + i].to_f32());
                    let x = packed.as_ref().map_or(original_x, |p| {
                        let (q, d) = self.activation(p, row, i);
                        f64::from(q) * f64::from(d)
                    });
                    original += w * original_x;
                    target += w * x;
                    magnitude += (w * x).abs();
                }
                let actual = f64::from(f16::from_bits(y[row * self.stride + column]).to_f32());
                let epsilon = f64::from(f32::EPSILON);
                let operations = (2 * self.k + 64) as f64;
                let gamma = operations * epsilon / (1.0 - operations * epsilon);
                let bound = gamma * magnitude + target.abs() / 2048.0 + 2.0_f64.powi(-24);
                assert!(actual.is_finite() && (actual - target).abs() <= bound,
                    "Q8_0 M{} K{} N{} strict{strict} [{row},{column}] actual{actual} F64{target} bound{bound}", self.m,self.k,self.n);
                quant_error = quant_error.max((target - original).abs());
                strict_error = strict_error.max((actual - original).abs());
                checked += 1;
            }
        }
        serde_json::json!({"checked_outputs":checked,"oracle":"independent_decoded_q8_weight_actual_d4_activation_f64",
            "max_input_quantization_output_error":quant_error,"max_published_error_vs_original_input_f64":strict_error,
            "strict_bit_equality_required":false,"semantic_quality":"not evaluated"})
    }
    fn strict_difference(&self, s: &Arc<CudaStream>, indices: &[usize]) -> serde_json::Value {
        let candidate = halves(&self.output.read(s));
        let strict = halves(&self.strict_output.read(s));
        let mut maximum = 0.0_f64;
        let mut changed = 0;
        let mut count = 0;
        for row in 0..self.m {
            for &index in indices {
                for col in 0..self.n {
                    let offset = row * self.stride + self.offsets[index % 2] + col;
                    let a = f16::from_bits(candidate[offset]).to_f32();
                    let b = f16::from_bits(strict[offset]).to_f32();
                    assert!(a.is_finite() && b.is_finite());
                    maximum = maximum.max((f64::from(a) - f64::from(b)).abs());
                    changed += usize::from(candidate[offset] != strict[offset]);
                    count += 1;
                }
            }
        }
        serde_json::json!({"checked_outputs":count,"changed_output_bits":changed,
            "max_absolute_difference_vs_retained_strict_f16":maximum})
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
#[ignore = "requires the locked Q8_0 pilot artifact and actual CUDA"]
fn q8_zero_mmq_actual_pack_f64_and_shared_output_boundaries_on_cuda() {
    let (_, s, kernels) = context();
    for (m, k, n) in [(3, 768, 17), (8, 5120, 48), (32, 5120, 48)] {
        let mut c = Case::new(&s, m, k, n, 1);
        c.scan(&s);
        s.synchronize().unwrap();
        for indices in [&[0][..], &[1][..], &[0, 1][..]] {
            c.reset(&s);
            for &i in indices {
                c.inclusive(&s, i);
                c.strict(&s, &kernels, i);
            }
            let oracle = c.oracle(&s, indices, false);
            let strict_oracle = c.oracle(&s, indices, true);
            let strict_difference = c.strict_difference(&s, indices);
            c.guards(&s);
            let before = c.output.read(&s);
            let graph = Captured::new(&s, || {
                for &i in indices {
                    c.inclusive(&s, i);
                }
            });
            graph.launch();
            graph.launch();
            s.synchronize().unwrap();
            assert_eq!(before, c.output.read(&s), "same route direct/graph bits");
            c.guards(&s);
            println!(
                "{}",
                serde_json::json!({"kind":"q8_zero_mmq_correctness","m":m,"k":k,"n":n,
                "indices":indices,"output_stride":c.stride,"output_offsets":c.offsets,
                "j":c.native.geometry().j,"fixup":c.native.geometry().fixup,"oracle":oracle,"strict_oracle":strict_oracle,
                "strict_difference":strict_difference})
            );
        }
        let ordinary = c.x.clone();
        for pattern in 0..6 {
            c.x = ordinary.clone();
            match pattern {
                0 => c.x.fill(f16::from_bits(0x8000)),
                1 => {
                    for (i, x) in c.x.iter_mut().enumerate() {
                        *x = f16::from_bits(1 + (i % 1023) as u16);
                    }
                }
                2 => {
                    for row in 0..m {
                        for g in 0..k / 32 {
                            c.x[row * k + g * 32] = f16::from_f32(1.0);
                            for i in 1..32 {
                                c.x[row * k + g * 32 + i] = f16::from_f32((i as f32 + 0.5) / 127.0);
                            }
                        }
                    }
                }
                3 => c.x[(m - 1) * k] = f16::NAN,
                4 => c.x[(m - 1) * k] = f16::INFINITY,
                _ => c.x[(m - 1) * k] = f16::NEG_INFINITY,
            }
            c.upload(&s);
            c.reset(&s);
            c.inclusive(&s, 0);
            c.inclusive(&s, 1);
            let oracle = c.oracle(&s, &[0, 1], false);
            c.guards(&s);
            println!(
                "{}",
                serde_json::json!({"kind":"q8_zero_mmq_marker_boundary","m":m,"k":k,"n":n,"pattern":pattern,"oracle":oracle})
            );
        }
        c.x = ordinary;
        let good = c.weights[0].bytes.clone();
        for invalid in [0x7c00_u16, 0xfc00, 0x7e00] {
            c.weights[0].bytes[..2].copy_from_slice(&invalid.to_le_bytes());
            c.upload(&s);
            c.reset(&s);
            c.inclusive(&s, 0);
            c.inclusive(&s, 1);
            c.oracle(&s, &[0, 1], false);
            c.guards(&s);
        }
        c.weights[0].bytes = good;
        c.upload(&s);
        c.reset(&s);
        c.inclusive(&s, 0);
        c.inclusive(&s, 1);
        c.oracle(&s, &[0, 1], false);
        c.guards(&s); // finite flag/row recovery
    }
    // Independently exercise MarkerV2's exact publish boundary. Dot fixtures
    // above stay finite; this covers nonfinite raw results, F16 overflow and
    // ties on both sides of an even/odd half significand without manufacturing
    // a particular matrix reduction order.
    let mut c = Case::new(&s, 3, 768, 17, 1);
    c.scan(&s);
    c.reset(&s);
    c.inclusive(&s, 0);
    s.synchronize().unwrap();
    let values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0,
        65520.0,
        1.0 + 1.0 / 2048.0,
        1.0 + 3.0 / 2048.0,
        2.0_f32.powi(-25),
        3.0 * 2.0_f32.powi(-25),
    ];
    let raw = (0..c.m * c.n)
        .map(|i| values[i % values.len()])
        .collect::<Vec<_>>();
    c.raw.write(
        &s,
        &raw.iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    );
    c.output.poison(&s);
    unsafe {
        c.native.cast(
            c.raw.span(&s),
            c.output.span_at(&s, c.offsets[0] * 2),
            c.stride as u32,
            c.row_flags.span(&s),
            c.weights[0].flag.span(&s),
            s.cu_stream().cast(),
        )
    }
    .unwrap();
    s.synchronize().unwrap();
    let output = halves(&c.output.read(&s));
    for row in 0..c.m {
        for col in 0..c.stride {
            let expected = if (c.offsets[0]..c.offsets[0] + c.n).contains(&col) {
                let v = raw[row * c.n + col - c.offsets[0]];
                let half = f16::from_f32(v);
                if !v.is_finite() || !half.is_finite() {
                    0x7e00
                } else {
                    half.to_bits()
                }
            } else {
                0x3535
            };
            assert_eq!(
                output[row * c.stride + col],
                expected,
                "MarkerV2 F16 RNE publish [{row},{col}]"
            );
        }
    }
    c.guards(&s);
}

#[test]
#[ignore = "synthetic paired complete Q8_0 pipeline; coordinate exclusive CUDA access"]
fn q8_zero_mmq_48_layer_two_leaf_inclusive_paired_on_cuda() {
    let (_, s, kernels) = context();
    const LAYERS: usize = 48;
    for m in [8, 32] {
        let plan_start = std::time::Instant::now();
        let mut c = Case::new(&s, m, 5120, 48, LAYERS);
        let preparation_wall_ns = plan_start.elapsed().as_secs_f64() * 1e9;
        let (scan_wall_ns, scan_gpu_ns) = measure(&s, || c.scan(&s));
        c.reset(&s);
        for i in 0..2 * LAYERS {
            c.inclusive(&s, i);
            c.strict(&s, &kernels, i);
        }
        let oracle = c.oracle(&s, &[2 * LAYERS - 2, 2 * LAYERS - 1], false);
        let strict_oracle = c.oracle(&s, &[2 * LAYERS - 2, 2 * LAYERS - 1], true);
        let strict_difference = c.strict_difference(&s, &[2 * LAYERS - 2, 2 * LAYERS - 1]);
        c.guards(&s);
        let candidate_initial = c.output.read(&s);
        let strict_initial = c.strict_output.read(&s);
        let strict = Captured::new(&s, || {
            for i in 0..2 * LAYERS {
                c.strict(&s, &kernels, i);
            }
        });
        let inclusive = Captured::new(&s, || {
            for i in 0..2 * LAYERS {
                c.inclusive(&s, i);
            }
        });
        for _ in 0..2 {
            strict.launch();
            inclusive.launch();
        }
        s.synchronize().unwrap();
        for pair in 0..8 {
            for position in 0..2 {
                let candidate = (pair + position) % 2 == 1;
                let (wall_ns, gpu_ns) = measure(&s, || {
                    if candidate {
                        inclusive.launch();
                    } else {
                        strict.launch();
                    }
                });
                let actual = if candidate {
                    c.output.read(&s)
                } else {
                    c.strict_output.read(&s)
                };
                assert_eq!(
                    &actual,
                    if candidate {
                        &candidate_initial
                    } else {
                        &strict_initial
                    },
                    "paired replay changed same-route output"
                );
                println!(
                    "{}",
                    serde_json::json!({"kind":"q8_zero_mmq_inclusive_paired","synthetic":true,
                    "m":m,"k":c.k,"n":c.n,"layers":LAYERS,"leaves_per_layer":2,
                    "distinct_weight_matrices":c.weights.len(),"total_weight_bytes":c.weights.iter().map(|w|w.bytes.len()).sum::<usize>(),
                    "input_shared_across_leaves":true,"output_stride":c.stride,"output_offsets":c.offsets,
                    "route":if candidate {"f16_convert_pad_d4_pack_mmq_fixup_marker_publish"} else {"retained_strict_q8_f16_dispatch"},
                    "mode":"graph_replay","pair":pair,"position":position,"graph_nodes":if candidate {inclusive.nodes} else {strict.nodes},
                    "wall_ns":wall_ns,"gpu_ns":gpu_ns,"wall_ns_per_projection":wall_ns/(2*LAYERS) as f64,
                    "gpu_ns_per_projection":gpu_ns/(2*LAYERS) as f64,"inclusive_pipeline":candidate,
                    "static_weight_validation_included":false,"row_flag_reset_included":candidate,
                    "converted_k_padding_included":candidate,"packed_guard_reset_included":candidate,
                    "performance_assertion":null,"semantic_quality":"not evaluated"})
                );
            }
        }
        c.guards(&s);
        println!(
            "{}",
            serde_json::json!({"kind":"q8_zero_mmq_inclusive_paired_summary","m":m,
            "preparation_and_fixture_allocation_wall_ns":preparation_wall_ns,"static_weight_scan_wall_ns":scan_wall_ns,
            "native_plan_preparation_wall_ns":c.native_preparation_wall_ns,
            "static_weight_scan_gpu_ns":scan_gpu_ns,"j":c.native.geometry().j,"fixup":c.native.geometry().fixup,
            "pack_abi":c.native.geometry().pack_abi,"converted_bytes":c.native.geometry().converted_bytes,
            "packed_bytes":c.native.geometry().packed_bytes,"fixup_bytes":c.native.geometry().fixup_bytes,
            "oracle":oracle,"strict_oracle":strict_oracle,"strict_difference":strict_difference,"same_route_repeat_bits":true})
        );
    }
}
