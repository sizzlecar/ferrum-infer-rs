//! Experimental native mechanism oracle. Raw backend owners are real guarded
//! allocations; this fixture does not mint a Core admission/independence proof.
//! The candidate keeps F32 queries/output. The retained control explicitly
//! casts queries to F16 and uses the existing F16 V1/V2 indexed kernels.
use super::*;
use crate::backend::cuda::vllm_paged_attn::{
    dispatch_vnext_addressed_paged_attention_indexed_batch_raw, VnextAddressedPagedAttentionKernel,
};
use crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config;
use cudarc::driver::{
    sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode},
    CudaFunction, CudaSlice, DevicePtr, DeviceRepr,
};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::DeviceId;
use half::f16;
use std::fmt::Debug;
use std::time::Instant;

const HEADS: usize = 24;
const KV_HEADS: usize = 4;
const DIM: usize = 256;
const MAX_LENGTH: usize = 2048;
const PARTITIONS: usize = 16;
const BLOCK_HALF: usize = 2 * KV_HEADS * DIM * 16;
const GUARDS: usize = 8;
const LAYERS: usize = 16;
const LENGTH_CASES: [usize; 8] = [16, 17, 511, 512, 513, 1024, 1025, 2048];
// Declared before running either arm. All output elements are checked, using
// absolute tolerance near zero and relative tolerance elsewhere.
const CANDIDATE_ATOL: f64 = 5e-5;
const CANDIDATE_RTOL: f64 = 5e-4;
const CONTROL_ATOL: f64 = 5e-4;
const CONTROL_RTOL: f64 = 1e-2;

struct GuardedBuffer<T> {
    device: CudaSlice<T>,
    host: Vec<T>,
}

impl<T: DeviceRepr + Clone + PartialEq + Debug> GuardedBuffer<T> {
    fn new(stream: &Arc<CudaStream>, values: &[T], guard: T) -> Self {
        let mut host = vec![guard.clone(); GUARDS];
        host.extend_from_slice(values);
        host.extend(vec![guard; GUARDS]);
        Self {
            device: stream.clone_htod(&host).unwrap(),
            host,
        }
    }

    fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.device.device_ptr(stream).0 + (GUARDS * std::mem::size_of::<T>()) as u64
    }

    fn values(&self) -> &[T] {
        &self.host[GUARDS..self.host.len() - GUARDS]
    }

    fn replace(&mut self, stream: &Arc<CudaStream>, values: &[T]) {
        assert_eq!(values.len(), self.values().len());
        let end = self.host.len() - GUARDS;
        self.host[GUARDS..end].clone_from_slice(values);
        stream.memcpy_htod(&self.host, &mut self.device).unwrap();
    }

    fn read(&self, stream: &Arc<CudaStream>) -> Vec<T> {
        stream.synchronize().unwrap();
        let result = stream.clone_dtoh(&self.device).unwrap();
        let end = result.len() - GUARDS;
        assert_eq!(
            &result[..GUARDS],
            &self.host[..GUARDS],
            "leading guard modified"
        );
        assert_eq!(&result[end..], &self.host[end..], "trailing guard modified");
        result[GUARDS..end].to_vec()
    }

    fn assert_unchanged(&self, stream: &Arc<CudaStream>) {
        assert_eq!(self.read(stream), self.values(), "read-only input modified");
    }
}

struct Page {
    allocation: Arc<GuardedBuffer<f16>>,
    offset: usize,
}

impl Page {
    fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.allocation.pointer(stream) + (self.offset * 2) as u64
    }
    fn values(&self) -> &[f16] {
        &self.allocation.values()[self.offset..self.offset + BLOCK_HALF]
    }
}

struct ProbeFunctions {
    partial: CudaFunction,
    reduce: CudaFunction,
    cast: CudaFunction,
    gate: CudaFunction,
    control_gate: CudaFunction,
}

impl ProbeFunctions {
    fn load(runtime: &CudaDeviceRuntime) -> Self {
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::INDEXED_GQA_DECODE_PROBE))
            .unwrap();
        Self {
            partial: module
                .load_function("indexed_gqa_decode_f32_query_f16_kv_h256_gqa6_partial")
                .unwrap(),
            reduce: module
                .load_function("indexed_gqa_decode_f32_query_f16_kv_h256_gqa6_reduce")
                .unwrap(),
            cast: module
                .load_function("indexed_gqa_probe_cast_f32_to_f16")
                .unwrap(),
            gate: module.load_function("indexed_gqa_probe_gate_f32").unwrap(),
            control_gate: module
                .load_function("indexed_gqa_probe_gate_f16_to_f32")
                .unwrap(),
        }
    }
}

struct Node {
    pages: Vec<Vec<Page>>,
    retired: Vec<Arc<GuardedBuffer<f16>>>,
    requests: Vec<usize>,
    current_lengths: Vec<usize>,
    layout: BindingLayout,
    binding: GuardedBuffer<u64>,
    query: GuardedBuffer<f32>,
    gate: GuardedBuffer<f32>,
    control_query: GuardedBuffer<f16>,
    control_context: GuardedBuffer<f16>,
    context: GuardedBuffer<f32>,
    control_output: GuardedBuffer<f32>,
    output: GuardedBuffer<f32>,
    lengths: GuardedBuffer<i32>,
    row_map: GuardedBuffer<i32>,
    partial: GuardedBuffer<f32>,
    sums: GuardedBuffer<f32>,
    maxima: GuardedBuffer<f32>,
    temporary: GuardedBuffer<f16>,
}

fn probe_shape() -> CausalAttentionShape {
    let mut s = shape(DIM as u64);
    s.query_heads = HEADS as u64;
    s.key_value_heads = KV_HEADS as u64;
    s.query_features = (HEADS * DIM) as u64;
    s.query_projection_features = s.query_features;
    s.hidden_size = s.query_features;
    s.kv_features = (KV_HEADS * DIM) as u64;
    s
}

fn page_values(request: usize, layer: usize, block: usize) -> Vec<f16> {
    // Nonzero unused tails are deliberately much larger than live KV. A
    // missing tail mask must fail the full output oracle, not just a canary.
    (0..BLOCK_HALF)
        .map(|i| {
            f16::from_f32(
                (((i * 7 + request * 29 + layer * 17 + block * 31) % 257) as f32 - 128.0) / 128.0,
            )
        })
        .collect()
}

impl Node {
    fn new(stream: &Arc<CudaStream>, owners: usize, layer: usize) -> Self {
        let shape = probe_shape();
        let layout = BindingLayout::new(shape, owners).unwrap();
        let current_lengths = (0..owners).map(|i| LENGTH_CASES[i % 8]).collect::<Vec<_>>();
        let mut pages = current_lengths
            .iter()
            .enumerate()
            .map(|(request, &length)| {
                (0..length.div_ceil(16))
                    .map(|block| Page {
                        allocation: Arc::new(GuardedBuffer::new(
                            stream,
                            &page_values(request, layer, block),
                            f16::from_f32(71.0),
                        )),
                        offset: 0,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Different exact page ranges can share one current allocation.
        let shared = pages[0][0]
            .values()
            .iter()
            .chain(pages[1][0].values())
            .copied()
            .collect::<Vec<_>>();
        let allocation = Arc::new(GuardedBuffer::new(stream, &shared, f16::from_f32(72.0)));
        pages[0][0] = Page {
            allocation: Arc::clone(&allocation),
            offset: 0,
        };
        pages[1][0] = Page {
            allocation,
            offset: BLOCK_HALF,
        };
        // A genuinely shared, read-only prefix block with separate live tails.
        pages[6][0] = Page {
            allocation: Arc::clone(&pages[5][0].allocation),
            offset: pages[5][0].offset,
        };
        let elements = owners * HEADS * DIM;
        let mut node = Self {
            pages,
            retired: Vec::new(),
            requests: (0..owners).collect(),
            current_lengths,
            layout,
            binding: GuardedBuffer::new(
                stream,
                &vec![u64::MAX; layout.required_bytes as usize / 8],
                u64::MAX - 7,
            ),
            query: GuardedBuffer::new(stream, &vec![0.0; elements], 73.0),
            gate: GuardedBuffer::new(stream, &vec![0.0; elements], 74.0),
            control_query: GuardedBuffer::new(
                stream,
                &vec![f16::ZERO; elements],
                f16::from_f32(75.0),
            ),
            control_context: GuardedBuffer::new(
                stream,
                &vec![f16::ZERO; elements],
                f16::from_f32(76.0),
            ),
            context: GuardedBuffer::new(stream, &vec![0.0; elements], 77.0),
            control_output: GuardedBuffer::new(stream, &vec![0.0; elements], 78.0),
            output: GuardedBuffer::new(stream, &vec![0.0; elements], 79.0),
            lengths: GuardedBuffer::new(stream, &vec![-1; owners], -711),
            row_map: GuardedBuffer::new(stream, &vec![-1; owners], -712),
            partial: GuardedBuffer::new(
                stream,
                &vec![0.0; owners * HEADS * PARTITIONS * (DIM + 2)],
                80.0,
            ),
            sums: GuardedBuffer::new(stream, &vec![0.0; owners * HEADS * 4], 81.0),
            maxima: GuardedBuffer::new(stream, &vec![0.0; owners * HEADS * 4], 82.0),
            temporary: GuardedBuffer::new(
                stream,
                &vec![f16::ZERO; owners * HEADS * 4 * DIM],
                f16::from_f32(83.0),
            ),
        };
        node.update(stream, layer, 0);
        node
    }

    fn update(&mut self, stream: &Arc<CudaStream>, layer: usize, generation: usize) {
        let owners = self.requests.len();
        if generation != 0 {
            // Preserve original allocations as real retired owners. Every
            // replay sees current pointers/lengths and a shuffled request slot.
            self.requests = (0..owners)
                .map(|slot| (slot * 5 + 3 + generation * 2) % owners)
                .collect();
        }
        for request in 0..owners {
            let initial = LENGTH_CASES[request % 8];
            let length = (initial + generation.min(1)).min(MAX_LENGTH);
            self.current_lengths[request] = length;
            let block = (length - 1) / 16;
            let token = (length - 1) % 16;
            if block == self.pages[request].len() {
                self.pages[request].push(Page {
                    allocation: Arc::new(GuardedBuffer::new(
                        stream,
                        &page_values(request, layer, block),
                        f16::from_f32(84.0),
                    )),
                    offset: 0,
                });
            }
            let mut values = self.pages[request][block].values().to_vec();
            for head in 0..KV_HEADS {
                for d in 0..DIM {
                    values[head * DIM * 16 + (d / 8) * 16 * 8 + token * 8 + d % 8] = f16::from_f32(
                        (((d * 3 + head * 7 + request * 11 + generation * 13) % 101) as f32 - 50.0)
                            / 96.0,
                    );
                    values[KV_HEADS * DIM * 16 + head * DIM * 16 + d * 16 + token] = f16::from_f32(
                        (((d * 5 + head * 11 + request * 3 + generation * 17) % 103) as f32 - 51.0)
                            / 104.0,
                    );
                }
            }
            // Fresh page replacement contains the complete previous history
            // plus precisely the current append. Old pointed-to bytes remain.
            let replacement = Page {
                allocation: Arc::new(GuardedBuffer::new(stream, &values, f16::from_f32(85.0))),
                offset: 0,
            };
            let before = std::mem::replace(&mut self.pages[request][block], replacement);
            self.retired.push(before.allocation);
        }
        // Keep disjoint current ranges of one allocation in the live input,
        // even after the current-token replacement above. They are not merged
        // owners, and neither range can supply the other request's history.
        let shared = self.pages[0][0]
            .values()
            .iter()
            .chain(self.pages[1][0].values())
            .copied()
            .collect::<Vec<_>>();
        let allocation = Arc::new(GuardedBuffer::new(stream, &shared, f16::from_f32(86.0)));
        for request in 0..2 {
            let page = Page {
                allocation: Arc::clone(&allocation),
                offset: request * BLOCK_HALF,
            };
            self.retired
                .push(std::mem::replace(&mut self.pages[request][0], page).allocation);
        }
        let slot_words = self.layout.slot_bytes as usize / 8;
        let mut words = vec![u64::MAX; self.layout.required_bytes as usize / 8];
        for (slot, &request) in self.requests.iter().enumerate() {
            let length = self.current_lengths[request];
            let base = slot * slot_words;
            words[base] = length.div_ceil(16) as u64 | (((length - 1) as u64) << 32);
            words[base + 1] = 1 | ((length as u64) << 32);
            words[base + 2] = slot as u64;
            for (block, page) in self.pages[request].iter().enumerate() {
                words[base + 3 + block] = page.pointer(stream);
            }
        }
        self.binding.replace(stream, &words);
        let query = self
            .requests
            .iter()
            .flat_map(|&request| {
                (0..HEADS * DIM).map(move |i| {
                    (((i * 11 + request * 19 + layer * 7 + generation * 23) % 127) as f32 - 63.0)
                        / 91.0
                })
            })
            .collect::<Vec<_>>();
        let gate = self
            .requests
            .iter()
            .flat_map(|&request| {
                (0..HEADS * DIM).map(move |i| {
                    (((i * 13 + request * 17 + layer * 3 + generation * 29) % 89) as f32 - 44.0)
                        / 32.0
                })
            })
            .collect::<Vec<_>>();
        self.query.replace(stream, &query);
        self.gate.replace(stream, &gate);
    }

    fn v1_count(&self) -> usize {
        self.requests
            .iter()
            .filter(|&&r| self.current_lengths[r] <= 512)
            .count()
    }

    fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        provider: &CudaCausalPagedAttentionProvider,
        functions: &ProbeFunctions,
        candidate: bool,
    ) {
        let owners = self.requests.len() as i32;
        let v1_count = self.v1_count() as i32;
        gather_lengths_indexed(
            stream,
            &provider.functions.gather_decode_lengths_indexed,
            self.binding.pointer(stream),
            self.layout.slot_bytes,
            self.lengths.pointer(stream),
            self.row_map.pointer(stream),
            owners,
            v1_count,
        )
        .unwrap();
        if candidate {
            let q = self.query.pointer(stream);
            let table = self.binding.pointer(stream) + 24;
            let lengths = self.lengths.pointer(stream);
            let map = self.row_map.pointer(stream);
            let partial = self.partial.pointer(stream);
            let query_heads = HEADS as i32;
            let kv_heads = KV_HEADS as i32;
            let stride = (self.layout.slot_bytes / 8) as i32;
            let query_stride = (HEADS * DIM) as i32;
            let maximum = MAX_LENGTH as i32;
            let partitions = PARTITIONS as i32;
            let scale = 1.0_f32 / (DIM as f32).sqrt();
            let mut launch = stream.launch_builder(&functions.partial);
            launch
                .arg(&q)
                .arg(&table)
                .arg(&lengths)
                .arg(&map)
                .arg(&partial)
                .arg(&owners)
                .arg(&owners)
                .arg(&query_heads)
                .arg(&kv_heads)
                .arg(&stride)
                .arg(&query_stride)
                .arg(&maximum)
                .arg(&partitions)
                .arg(&scale);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (KV_HEADS as u32, owners as u32, PARTITIONS as u32),
                    block_dim: (192, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            let context = self.context.pointer(stream);
            let mut launch = stream.launch_builder(&functions.reduce);
            launch
                .arg(&partial)
                .arg(&lengths)
                .arg(&map)
                .arg(&context)
                .arg(&owners)
                .arg(&owners)
                .arg(&query_heads)
                .arg(&maximum)
                .arg(&partitions);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (HEADS as u32, owners as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        } else {
            let input = self.query.pointer(stream);
            let output = self.control_query.pointer(stream);
            let elements = owners * (HEADS * DIM) as i32;
            let mut cast = stream.launch_builder(&functions.cast);
            cast.arg(&input).arg(&output).arg(&elements);
            unsafe { cast.launch(LaunchConfig::for_num_elems(elements as u32)) }.unwrap();
            for (start, count, maximum, kernel) in [
                (0, v1_count, 512, VnextAddressedPagedAttentionKernel::V1),
                (
                    v1_count,
                    owners - v1_count,
                    MAX_LENGTH,
                    VnextAddressedPagedAttentionKernel::V2,
                ),
            ] {
                if count == 0 {
                    continue;
                }
                unsafe {
                    dispatch_vnext_addressed_paged_attention_indexed_batch_raw(
                        stream,
                        self.control_context.pointer(stream),
                        self.control_query.pointer(stream),
                        self.binding.pointer(stream) + 24,
                        self.lengths.pointer(stream),
                        self.row_map.pointer(stream) + (start as u64 * 4),
                        maximum as u64,
                        Some(self.sums.pointer(stream)),
                        Some(self.maxima.pointer(stream)),
                        Some(self.temporary.pointer(stream)),
                        count,
                        owners,
                        HEADS as i32,
                        KV_HEADS as i32,
                        DIM as i32,
                        (self.layout.slot_bytes / 8) as i32,
                        kernel,
                    )
                    .unwrap();
                }
            }
        }
        let context = if candidate {
            self.context.pointer(stream)
        } else {
            self.control_context.pointer(stream)
        };
        let gate = self.gate.pointer(stream);
        let output = if candidate {
            self.output.pointer(stream)
        } else {
            self.control_output.pointer(stream)
        };
        let elements = owners * (HEADS * DIM) as i32;
        let function = if candidate {
            &functions.gate
        } else {
            &functions.control_gate
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&context).arg(&gate).arg(&output).arg(&elements);
        unsafe { launch.launch(LaunchConfig::for_num_elems(elements as u32)) }.unwrap();
    }

    fn dense_oracle(&self, cast_query: bool) -> Vec<f64> {
        let mut output = vec![0.0; self.requests.len() * HEADS * DIM];
        for (slot, &request) in self.requests.iter().enumerate() {
            let length = self.current_lengths[request];
            for head in 0..HEADS {
                let kv_head = head / 6;
                let qbase = (slot * HEADS + head) * DIM;
                let query = self.query.values()[qbase..qbase + DIM]
                    .iter()
                    .map(|&q| {
                        f64::from(if cast_query {
                            f16::from_f32(q).to_f32()
                        } else {
                            q
                        })
                    })
                    .collect::<Vec<_>>();
                let mut scores = vec![0.0_f64; length];
                for (token, score) in scores.iter_mut().enumerate() {
                    let page = self.pages[request][token / 16].values();
                    let within = token % 16;
                    *score = query
                        .iter()
                        .enumerate()
                        .map(|(d, &q)| {
                            let k = page
                                [kv_head * DIM * 16 + (d / 8) * 16 * 8 + within * 8 + d % 8]
                                .to_f32();
                            q * f64::from(k)
                        })
                        .sum::<f64>()
                        / (DIM as f64).sqrt();
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denominator = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f64>();
                for (token, score) in scores.iter().enumerate() {
                    let page = self.pages[request][token / 16].values();
                    let within = token % 16;
                    let weight = score / denominator;
                    for d in 0..DIM {
                        output[qbase + d] += weight
                            * f64::from(
                                page[KV_HEADS * DIM * 16 + kv_head * DIM * 16 + d * 16 + within]
                                    .to_f32(),
                            );
                    }
                }
                for d in 0..DIM {
                    // The old control stores context in half before the F32
                    // output gate; the candidate retains F32 through the gate.
                    if cast_query {
                        output[qbase + d] = f64::from(f16::from_f64(output[qbase + d]).to_f32());
                    }
                    let gate = f64::from(self.gate.values()[qbase + d]);
                    output[qbase + d] /= 1.0 + (-gate).exp();
                }
            }
        }
        output
    }

    fn check_outputs(&self, stream: &Arc<CudaStream>, generation: usize) {
        let expected = self.dense_oracle(false);
        let control_expected = self.dense_oracle(true);
        check_close(
            &self.output.read(stream),
            &expected,
            CANDIDATE_ATOL,
            CANDIDATE_RTOL,
            "F32 candidate",
            generation,
        );
        check_close(
            &self.control_output.read(stream),
            &control_expected,
            CONTROL_ATOL,
            CONTROL_RTOL,
            "F16 control",
            generation,
        );
        // Cross-control tolerance explicitly accommodates the old Q/output
        // casts. It never replaces either independent full-output oracle.
        check_close(
            &self.output.read(stream),
            &self
                .control_output
                .read(stream)
                .iter()
                .map(|&x| f64::from(x))
                .collect::<Vec<_>>(),
            2e-3,
            2e-2,
            "candidate vs old precision",
            generation,
        );
        self.check_buffers(stream);
        let expected_lengths = self
            .requests
            .iter()
            .map(|&r| self.current_lengths[r] as i32)
            .collect::<Vec<_>>();
        assert_eq!(self.lengths.read(stream), expected_lengths);
        let expected_map = (0..self.requests.len())
            .filter(|&i| expected_lengths[i] <= 512)
            .chain((0..self.requests.len()).filter(|&i| expected_lengths[i] > 512))
            .map(|i| i as i32)
            .collect::<Vec<_>>();
        assert_eq!(
            self.row_map.read(stream),
            expected_map,
            "cached slot/path map on replay"
        );
    }

    fn check_buffers(&self, stream: &Arc<CudaStream>) {
        self.query.assert_unchanged(stream);
        self.gate.assert_unchanged(stream);
        self.binding.assert_unchanged(stream);
        for page in self.pages.iter().flatten() {
            page.allocation.assert_unchanged(stream);
        }
        for allocation in &self.retired {
            allocation.assert_unchanged(stream);
        }
        let _ = (
            self.control_query.read(stream),
            self.control_context.read(stream),
            self.context.read(stream),
            self.output.read(stream),
            self.control_output.read(stream),
            self.lengths.read(stream),
            self.row_map.read(stream),
            self.partial.read(stream),
            self.sums.read(stream),
            self.maxima.read(stream),
            self.temporary.read(stream),
        );
    }
}

fn check_close(
    actual: &[f32],
    expected: &[f64],
    atol: f64,
    rtol: f64,
    label: &str,
    generation: usize,
) {
    assert_eq!(actual.len(), expected.len());
    let mut max_absolute = 0.0_f64;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label} nonfinite element{i}"
        );
        let error = (f64::from(a) - b).abs();
        max_absolute = max_absolute.max(error);
        assert!(
            error <= atol + rtol * b.abs(),
            "{label} generation{generation} element{i}: {a} vs {b}, abs={error}, tolerance={}",
            atol + rtol * b.abs()
        );
    }
    eprintln!("indexed_gqa_oracle label={label:?} generation={generation} elements={} max_abs={max_absolute:.9} atol={atol} rtol={rtol}", actual.len());
}

fn capture_bundle(
    stream: &Arc<CudaStream>,
    nodes: &[Node],
    provider: &CudaCausalPagedAttentionProvider,
    functions: &ProbeFunctions,
    candidate: bool,
) -> cudarc::driver::CudaGraph {
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    for node in nodes {
        node.enqueue(stream, provider, functions, candidate);
    }
    stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .expect("nonempty attention bundle")
}

#[test]
#[ignore = "requires actual CUDA, the pinned indexed addressed operator set, and experimental GQA PTX"]
fn indexed_gqa_full_output_oracle_and_complete_bundle_gate() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.indexed-gqa-probe").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    let functions = ProbeFunctions::load(&runtime);
    let stream = runtime.context().new_stream().unwrap();
    let mut gate_results = Vec::new();
    for owners in [8, 32] {
        // Full oracle runs on a real node before allocating the16-node timed
        // bundle. Both replays retain the old pages while current ptrs change.
        let mut oracle = Node::new(&stream, owners, 0);
        let mut graphs = None;
        for generation in 0..3 {
            if generation != 0 {
                oracle.update(&stream, 0, generation);
            }
            // Crossing512 changes the retained V1/V2 group topology at wave1.
            // Wave2 keeps topology but replaces pointers, inputs, and slots:
            // reuse the exact graph, proving it reads current device bindings.
            if generation < 2 {
                graphs = Some((
                    capture_bundle(
                        &stream,
                        std::slice::from_ref(&oracle),
                        &provider,
                        &functions,
                        false,
                    ),
                    capture_bundle(
                        &stream,
                        std::slice::from_ref(&oracle),
                        &provider,
                        &functions,
                        true,
                    ),
                ));
            }
            let (control, candidate) = graphs.as_ref().unwrap();
            for _ in 0..2 {
                control.launch().unwrap();
                candidate.launch().unwrap();
            }
            oracle.check_outputs(&stream, generation);
        }
        drop(graphs);
        drop(oracle);
        let nodes = (0..LAYERS)
            .map(|layer| Node::new(&stream, owners, layer))
            .collect::<Vec<_>>();
        let control = capture_bundle(&stream, &nodes, &provider, &functions, false);
        let candidate = capture_bundle(&stream, &nodes, &provider, &functions, true);
        for _ in 0..8 {
            control.launch().unwrap();
            candidate.launch().unwrap();
        }
        stream.synchronize().unwrap();
        let mut control_ms = Vec::new();
        let mut candidate_ms = Vec::new();
        for repeat in 0..3 {
            for candidate_arm in if repeat % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let graph = if candidate_arm { &candidate } else { &control };
                let started = stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                let wall = Instant::now();
                for _ in 0..64 {
                    graph.launch().unwrap();
                }
                let ended = stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                ended.synchronize().unwrap();
                let gpu_ms = f64::from(started.elapsed_ms(&ended).unwrap()) / 64.0;
                let host_ms = wall.elapsed().as_secs_f64() * 1000.0 / 64.0;
                eprintln!("indexed_gqa_bundle owners={owners} layers={LAYERS} arm={} repeat={repeat} iterations=64 gpu_ms={gpu_ms:.9} host_ms={host_ms:.9} includes=gather,attention_partitions,reduce,output_gate,control_query_cast precision_candidate=F32Q_F16KV_F32out precision_control=F16Q_F16KV_F16context_F32gate",
                    if candidate_arm { "candidate" } else { "control" });
                if candidate_arm {
                    candidate_ms.push(gpu_ms);
                } else {
                    control_ms.push(gpu_ms);
                }
            }
        }
        for (layer, node) in nodes.iter().enumerate() {
            if layer == 0 || layer == LAYERS - 1 {
                node.check_outputs(&stream, 0);
            } else {
                check_close(
                    &node.output.read(&stream),
                    &node
                        .control_output
                        .read(&stream)
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect::<Vec<_>>(),
                    2e-3,
                    2e-2,
                    "timed distinct-layer candidate vs old precision",
                    layer,
                );
                node.check_buffers(&stream);
            }
        }
        control_ms.sort_by(f64::total_cmp);
        candidate_ms.sort_by(f64::total_cmp);
        let faster = candidate_ms[1] < control_ms[1];
        let budget = owners != 32 || candidate_ms[1] <= 1.05;
        eprintln!("indexed_gqa_native_gate owners={owners} layers={LAYERS} control_median_ms={:.9} candidate_median_ms={:.9} faster={faster} m32_budget_ms=1.05 budget_pass={budget} whole_model_tps_claim=false", control_ms[1], candidate_ms[1]);
        gate_results.push((owners, faster, budget));
    }
    // A measured timing outcome is a benchmark fact, not a unit-test contract.
    // Root evaluates the predeclared stop gate before any product integration.
    let gate_failed = gate_results
        .iter()
        .any(|(_, faster, budget)| !*faster || !*budget);
    eprintln!("indexed_gqa_native_result gate_failed={gate_failed} observations={gate_results:?} whole_model_tps_claim=false");
}
