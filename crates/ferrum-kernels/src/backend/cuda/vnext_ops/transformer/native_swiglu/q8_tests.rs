use super::*;
use crate::gguf_blocks::q4k_q8_reference::{dot_reference, fixture_block, pack_rows, DotReference};
use crate::gguf_blocks::q56k_q8_reference::{dot_q5, dot_q6, fixture_q5, fixture_q6};
use cudarc::driver::{sys, CudaGraph, CudaSlice};
use std::ffi::CStr;
use std::sync::Arc;

fn part(format: MatrixFormat, rows: usize, columns: usize, offset: usize) -> MatrixPart {
    MatrixPart {
        component_id: WeightId::new("component.q8-swiglu-test").unwrap(),
        format,
        rows: rows as u32,
        columns: columns as u32,
        output_offset: offset as u32,
        transform: None,
        signs_region: None,
    }
}

#[test]
fn native_q8_swiglu_workspace_reuses_packs_and_rejects_incomplete_groups() {
    use GgufBlockFormat::*;
    let gate_up = [
        part(MatrixFormat::Block(Q4K), 512, 256, 0),
        part(MatrixFormat::Block(Q5K), 512, 256, 512),
    ];
    let down = [part(MatrixFormat::Block(Q6K), 256, 512, 0)];
    let gate_bytes = q8_part_workspace_per_token(&gate_up).unwrap();
    let down_bytes = q8_part_workspace_per_token(&down).unwrap();
    // Gate and up share a pack, and the larger down pack reuses its lifetime.
    assert_eq!(
        gate_bytes,
        q8_part_workspace_per_token(&gate_up[..1]).unwrap()
    );
    assert!(down_bytes > gate_bytes);
    let per_token = ScratchLayout::new(1, 512).unwrap().total_bytes + gate_bytes.max(down_bytes);
    for rows in [1, 3, 32] {
        let declared = per_token * rows;
        let stages = ScratchLayout::new(rows, 512).unwrap();
        let packed_input = PackLayout::new(rows, 256).unwrap();
        let packed_down = PackLayout::new(rows, 512).unwrap();
        assert_eq!(declared, stages.total_bytes + packed_down.total_bytes);
        assert!(packed_input.total_bytes <= declared - stages.total_bytes);
        assert!(packed_down.total_bytes > declared - stages.total_bytes - 1);
    }
    assert_eq!(
        q8_part_workspace_per_token(&[part(MatrixFormat::DenseF16, 17, 33, 0)]).unwrap(),
        0
    );
    let mut invalid = part(MatrixFormat::Block(Q5K), 17, 255, 0);
    assert!(q8_part_workspace_per_token(&[invalid.clone()]).is_err());
    invalid.columns = 256;
    invalid.signs_region = Some(0);
    assert!(q8_part_workspace_per_token(&[invalid]).is_err());
}

struct HostMatrix {
    format: MatrixFormat,
    columns: usize,
    salt: usize,
    bytes: Vec<u8>,
    dense: Vec<f32>,
}

impl HostMatrix {
    fn new(format: MatrixFormat, rows: usize, columns: usize, salt: usize) -> Self {
        let (bytes, dense) = match format {
            MatrixFormat::DenseF16 => matrix(format, rows, columns, salt),
            MatrixFormat::Block(kind) => {
                let mut bytes = Vec::new();
                for column in 0..rows {
                    for block in 0..columns / 256 {
                        match kind {
                            GgufBlockFormat::Q4K => {
                                bytes.extend(fixture_block(column + salt, block).encode())
                            }
                            GgufBlockFormat::Q5K => {
                                bytes.extend(fixture_q5(column + salt, block).encode())
                            }
                            GgufBlockFormat::Q6K => {
                                bytes.extend(fixture_q6(column + salt, block).encode())
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                (bytes, Vec::new())
            }
        };
        Self {
            format,
            columns,
            salt,
            bytes,
            dense,
        }
    }

    fn policy(&self, input: &[f16], column: usize, sum_policy: Q8SumPolicy) -> (f64, f64) {
        if sum_policy == Q8SumPolicy::Input {
            use crate::gguf_blocks::q8_input_sum_reference::{dot_q4, dot_q5};
            let corrected = match self.format {
                MatrixFormat::Block(GgufBlockFormat::Q4K) => Some(dot_q4(
                    input,
                    &(0..self.columns / 256)
                        .map(|b| fixture_block(column + self.salt, b))
                        .collect::<Vec<_>>(),
                )),
                MatrixFormat::Block(GgufBlockFormat::Q5K) => Some(dot_q5(
                    input,
                    &(0..self.columns / 256)
                        .map(|b| fixture_q5(column + self.salt, b))
                        .collect::<Vec<_>>(),
                )),
                _ => None,
            };
            if let Some(reference) = corrected {
                let nu = ((self.columns / 32).div_ceil(4) + 9) as f64 * f64::from(f32::EPSILON);
                return (
                    reference.policy,
                    nu / (1.0 - nu) * reference.expanded_abs_terms,
                );
            }
        }
        let reference: DotReference = match self.format {
            MatrixFormat::DenseF16 => {
                let products = input
                    .iter()
                    .zip(&self.dense[column * self.columns..][..self.columns])
                    .map(|(x, w)| x.to_f64() * f64::from(*w));
                let sum = products.clone().sum::<f64>();
                return (
                    sum,
                    self.columns as f64
                        * f64::from(f32::EPSILON)
                        * products.map(f64::abs).sum::<f64>(),
                );
            }
            MatrixFormat::Block(GgufBlockFormat::Q4K) => dot_reference(
                input,
                &(0..self.columns / 256)
                    .map(|b| fixture_block(column + self.salt, b))
                    .collect::<Vec<_>>(),
            ),
            MatrixFormat::Block(GgufBlockFormat::Q5K) => dot_q5(
                input,
                &(0..self.columns / 256)
                    .map(|b| fixture_q5(column + self.salt, b))
                    .collect::<Vec<_>>(),
            ),
            MatrixFormat::Block(GgufBlockFormat::Q6K) => dot_q6(
                input,
                &(0..self.columns / 256)
                    .map(|b| fixture_q6(column + self.salt, b))
                    .collect::<Vec<_>>(),
            ),
            _ => unreachable!(),
        };
        let nu = ((self.columns / 32).div_ceil(4) + 9) as f64 * f64::from(f32::EPSILON);
        (
            reference.policy,
            nu / (1.0 - nu) * reference.expanded_abs_terms,
        )
    }
}

// Mutable, aligned allocations are needed to replay with different data at the
// same pointers. Every reset also restores both guards; writes never reallocate.
struct Bytes {
    gpu: CudaSlice<u8>,
    len: usize,
}

impl Bytes {
    fn new(stream: &Arc<CudaStream>, payload: &[u8]) -> Self {
        let mut padded = vec![0xab; 16];
        padded.extend_from_slice(payload);
        padded.extend([0xab; 16]);
        Self {
            gpu: stream.clone_htod(&padded).unwrap(),
            len: payload.len(),
        }
    }

    fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.gpu.device_ptr(stream).0 + 16
    }

    fn write(&mut self, stream: &Arc<CudaStream>, payload: &[u8]) {
        assert_eq!(payload.len(), self.len);
        let mut padded = vec![0xab; 16];
        padded.extend_from_slice(payload);
        padded.extend([0xab; 16]);
        stream.memcpy_htod(&padded, &mut self.gpu).unwrap();
    }

    fn read(&self, stream: &Arc<CudaStream>) -> Vec<u8> {
        stream.synchronize().unwrap();
        let bytes = stream.clone_dtoh(&self.gpu).unwrap();
        assert!(bytes[..16]
            .iter()
            .chain(&bytes[16 + self.len..])
            .all(|b| *b == 0xab));
        bytes[16..16 + self.len].to_vec()
    }
}

fn half_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect()
}

fn halves(bytes: &[u8]) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn assert_rounded(actual: f16, policy: f64, accumulation: f64, stage: &str) {
    let expected = f16::from_f64(policy);
    let bound =
        accumulation + 0.0009765625 * (policy.abs() + accumulation) + f16::from_bits(1).to_f64();
    assert!(
        actual.is_finite() && (actual.to_f64() - expected.to_f64()).abs() <= bound,
        "{stage}: actual={actual} policy={policy} rounded={expected} bound={bound}"
    );
}

fn check_stages(
    input: &[f16],
    matrices: &[HostMatrix; 3],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    scratch: &[u8],
    output: &[u8],
    packed_bytes: usize,
    sum_policy: Q8SumPolicy,
) {
    let gates = halves(&scratch[packed_bytes..packed_bytes + rows * intermediate * 4]);
    let activation = halves(&scratch[packed_bytes + rows * intermediate * 4..]);
    let output = halves(output);
    for row in 0..rows {
        let x = &input[row * hidden..][..hidden];
        for col in 0..intermediate {
            for (ordinal, matrix) in matrices[..2].iter().enumerate() {
                let (policy, error) = matrix.policy(x, col, sum_policy);
                assert_rounded(
                    gates[row * 2 * intermediate + ordinal * intermediate + col],
                    policy,
                    error,
                    "gate/up",
                );
            }
            // Each downstream oracle consumes the actual, independently checked
            // F16 storage boundary, never an unrounded F64 projection result.
            let g = gates[row * 2 * intermediate + col].to_f32();
            let u = gates[row * 2 * intermediate + intermediate + col].to_f32();
            let silu = g / (1.0_f32 + (-g).exp()) * u;
            assert_rounded(
                activation[row * intermediate + col],
                f64::from(silu),
                8.0 * f64::from(f32::EPSILON) * f64::from(silu.abs()),
                "SiLU F32 to F16",
            );
        }
        let x = &activation[row * intermediate..][..intermediate];
        for col in 0..hidden {
            let (policy, error) = matrices[2].policy(x, col, sum_policy);
            assert_rounded(output[row * hidden + col], policy, error, "down");
        }
    }
    let packed = pack_rows(&activation, rows, intermediate);
    let scales = packed
        .scales
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let words = packed.quants.iter().map(|q| *q as u8).collect::<Vec<_>>();
    assert_eq!(
        &scratch[..scales.len()],
        scales,
        "down must repack the F16 SiLU result"
    );
    let words_offset = if sum_policy == Q8SumPolicy::Input {
        let sums =
            crate::gguf_blocks::q8_input_sum_reference::pack_rows(&activation, rows, intermediate)
                .input_sums;
        let sum_bytes: Vec<_> = sums
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect();
        assert_eq!(
            &scratch[scales.len()..scales.len() + sum_bytes.len()],
            sum_bytes
        );
        scales.len() + sum_bytes.len()
    } else {
        scales.len()
    };
    assert_eq!(&scratch[words_offset..words_offset + words.len()], words);
}

fn assert_captured_packs(
    graph: &CudaGraph,
    input: u64,
    activation: u64,
    workspace: u64,
    rows: u32,
    hidden: u32,
    intermediate: u32,
    sum_policy: Q8SumPolicy,
) {
    // SAFETY: graph remains alive and no thread accesses it concurrently. CUDA
    // owns the returned node parameter storage. Only the known pack ABI is read.
    unsafe {
        let mut count = 0;
        assert_eq!(
            sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count),
            sys::CUresult::CUDA_SUCCESS
        );
        let mut nodes = vec![std::ptr::null_mut(); count];
        assert_eq!(
            sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count),
            sys::CUresult::CUDA_SUCCESS
        );
        let mut packs = Vec::new();
        for node in nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            assert_eq!(
                sys::cuGraphNodeGetType(node, &mut kind),
                sys::CUresult::CUDA_SUCCESS
            );
            if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                continue;
            }
            let mut params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
            assert_eq!(
                sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()),
                sys::CUresult::CUDA_SUCCESS
            );
            let params = params.assume_init();
            let mut name = std::ptr::null();
            assert_eq!(
                sys::cuFuncGetName(&mut name, params.func),
                sys::CUresult::CUDA_SUCCESS
            );
            assert!(!name.is_null());
            let pack_name: &[u8] = match sum_policy {
                Q8SumPolicy::Quantized => b"vnext_gguf_q8_f32scale_pack_f16_prototype",
                Q8SumPolicy::Input => b"vnext_gguf_q8_f32scale_input_sum_pack_f16_prototype",
            };
            if CStr::from_ptr(name).to_bytes() != pack_name {
                continue;
            }
            assert!(!params.kernelParams.is_null());
            let pointer = |i| (*params.kernelParams.add(i)).cast::<u64>().read_unaligned();
            let n = (*params.kernelParams.add(3)).cast::<u32>().read_unaligned();
            let k = (*params.kernelParams.add(4)).cast::<u32>().read_unaligned();
            assert_eq!(pointer(1), workspace);
            let layout = PackLayout::with_policy(n.into(), k.into(), sum_policy).unwrap();
            assert_eq!(pointer(2), workspace + layout.words_offset);
            if sum_policy == Q8SumPolicy::Input {
                assert_eq!(pointer(5), workspace + layout.scales_bytes);
            }
            packs.push((pointer(0), n, k));
        }
        packs.sort_unstable();
        let mut expected = [(input, rows, hidden), (activation, rows, intermediate)];
        expected.sort_unstable();
        assert_eq!(
            packs, expected,
            "gate/up must share one pack and down must repack"
        );
    }
}

#[test]
#[ignore = "requires an actual SM80+ CUDA device; Q8 SwiGLU stage and graph conformance"]
fn native_q8_swiglu_stages_and_replay_preserve_f16_policy_on_cuda() {
    stages_and_replay(Q8SumPolicy::Quantized, &[1, 3, 32]);
}

#[test]
#[ignore = "requires an actual SM80+ CUDA device; explicit input-sum SwiGLU stage and graph conformance"]
fn native_q8_input_sum_swiglu_stages_and_replay_on_cuda() {
    stages_and_replay(Q8SumPolicy::Input, &[1, 4, 8, 33]);
}

fn stages_and_replay(sum_policy: Q8SumPolicy, row_counts: &[usize]) {
    use GgufBlockFormat::*;
    use MatrixFormat::{Block as Q, DenseF16 as D};
    let context = CudaContext::new(0).expect("Q8 SwiGLU requires CUDA");
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let q8 = Q8F32ScaleKernels::load_with_policy(&context, sum_policy).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    for (formats, hidden, intermediate) in [
        ([Q(Q4K), Q(Q5K), Q(Q6K)], 256, 512),
        ([D, Q(Q6K), Q(Q5K)], 256, 256),
    ] {
        let matrices = [
            HostMatrix::new(formats[0], intermediate, hidden, 0),
            HostMatrix::new(formats[1], intermediate, hidden, 1),
            HostMatrix::new(formats[2], hidden, intermediate, 2),
        ];
        let device_weights = matrices.each_ref().map(|m| Bytes::new(&stream, &m.bytes));
        let pointers = device_weights.each_ref().map(|m| m.pointer(&stream));
        let gate_up = [
            part(formats[0], intermediate, hidden, 0),
            part(formats[1], intermediate, hidden, intermediate),
        ];
        let down = [part(formats[2], hidden, intermediate, 0)];
        for &rows in row_counts {
            let layout = ScratchLayout::new(rows as u64, intermediate as u64).unwrap();
            let packed = q8_part_workspace_per_token_with_policy(&gate_up, sum_policy)
                .unwrap()
                .max(q8_part_workspace_per_token_with_policy(&down, sum_policy).unwrap())
                * rows as u64;
            let mut input = Bytes::new(&stream, &vec![0; rows * hidden * 2]);
            let mut scratch =
                Bytes::new(&stream, &vec![0x7e; (packed + layout.total_bytes) as usize]);
            let mut output = Bytes::new(&stream, &vec![0x7e; rows * hidden * 2]);
            let xp = input.pointer(&stream);
            let sp = scratch.pointer(&stream);
            let yp = output.pointer(&stream);
            let gp = sp + packed;
            let ap = gp + layout.gate_up_bytes;
            let launch = || {
                launch_with_q8(
                    &kernels,
                    &silu,
                    &stream,
                    &gate_up,
                    &down,
                    &pointers,
                    xp,
                    yp,
                    gp,
                    ap,
                    rows as u32,
                    hidden as u32,
                    intermediate as u32,
                    0,
                    Some(&q8),
                    sp,
                )
                .unwrap()
            };
            // Resolve any lazy kernel loading before capture; this is not timed.
            launch();
            stream.synchronize().unwrap();
            scratch.read(&stream);
            output.read(&stream);
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .unwrap();
            launch();
            let graph = stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .expect("nonempty Q8 SwiGLU graph");
            assert_captured_packs(
                &graph,
                xp,
                ap,
                sp,
                rows as u32,
                hidden as u32,
                intermediate as u32,
                sum_policy,
            );
            let mut previous = None;
            for generation in [0, 1] {
                let source = (0..rows * hidden)
                    .map(|i| {
                        f16::from_f32((((i * 13 + generation * 17) % 67) as f32 - 29.0) / 127.0)
                    })
                    .collect::<Vec<_>>();
                let source_bytes = half_bytes(&source);
                input.write(&stream, &source_bytes);
                scratch.write(&stream, &vec![0x7e; scratch.len]);
                output.write(&stream, &vec![0x7e; output.len]);
                launch();
                let eager_scratch = scratch.read(&stream);
                let eager_output = output.read(&stream);
                check_stages(
                    &source,
                    &matrices,
                    rows,
                    hidden,
                    intermediate,
                    &eager_scratch,
                    &eager_output,
                    packed as usize,
                    sum_policy,
                );
                scratch.write(&stream, &vec![0x7e; scratch.len]);
                output.write(&stream, &vec![0x7e; output.len]);
                graph.launch().unwrap();
                assert_eq!(
                    scratch.read(&stream),
                    eager_scratch,
                    "eager/replay intermediate or pack drift"
                );
                assert_eq!(
                    output.read(&stream),
                    eager_output,
                    "eager/replay output drift"
                );
                assert_eq!(input.read(&stream), source_bytes);
                if let Some(previous) = previous {
                    assert_ne!(eager_output, previous, "new input reused a stale result");
                }
                previous = Some(eager_output);
            }
            drop(graph);
        }
        for (device, matrix) in device_weights.iter().zip(&matrices) {
            assert_eq!(device.read(&stream), matrix.bytes);
        }
    }
}
