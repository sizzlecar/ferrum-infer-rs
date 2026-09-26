//! Production strict GDN projection selection and its downstream state carry.
//! Synthetic weights at the reached physical geometry; this is not a model teacher.
use super::super::super::native_matrix;
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::weights::{MatrixFormat, MatrixPart};
use crate::gguf_blocks::{
    q4k_q8_reference::fixture_block, q56k_q8_reference::fixture_q5, GgufBlockFormat,
};
use cudarc::driver::{sys, CudaGraph};
use ferrum_interfaces::vnext::WeightId;
use std::sync::Arc;

const HIDDEN: usize = 4096;
const WIDTH: usize = 12352;
const BATCHES: [[usize; 3]; 4] = [[0, 1, 3], [0, 2, 2], [0, 3, 5], [0, 5, 3]];

fn part(format: GgufBlockFormat, rows: u32, offset: u32, salt: usize) -> MatrixPart {
    MatrixPart {
        component_id: WeightId::new(format!("pair.recurrent.{salt}")).unwrap(),
        format: MatrixFormat::Block(format),
        rows,
        columns: HIDDEN as u32,
        output_offset: offset,
        transform: None,
        signs_region: None,
    }
}
fn parts() -> Vec<MatrixPart> {
    use GgufBlockFormat::*;
    vec![
        part(Q5K, 8192, 0, 0),
        part(Q4K, 4096, 8192, 1),
        part(Q8_0, 32, 12288, 2),
        part(Q8_0, 32, 12320, 3),
    ]
}
#[test]
fn strict_pair_attribution_and_future_projection_count_preserve_other_routes() {
    let parts = parts();
    let qkv = SharedProjectionWeight::Native {
        first_region: 0,
        parts: parts.clone().into(),
    };
    let output_parts = vec![part(GgufBlockFormat::Q5K, HIDDEN as u32, 0, 4)];
    let output = SharedProjectionWeight::Native {
        first_region: 4,
        parts: output_parts.into(),
    };
    for rows in [4, 8] {
        assert_eq!(
            cost_route::native_projection_dispatches(&parts, rows, false).unwrap(),
            3
        );
        assert_eq!(qkv.dispatch_count(rows).unwrap(), 3);
        assert_eq!(
            attention_dispatches_per_launch(&qkv, &output, rows, false).unwrap(),
            12
        );
        // NativeQ8 keeps its original pack+four leaves, including unquantized BA.
        assert_eq!(
            cost_route::native_projection_dispatches(&parts, rows, true).unwrap(),
            5
        );
        assert_eq!(
            attention_dispatches_per_launch(&qkv, &output, rows, true).unwrap(),
            15
        );
    }
    for rows in [1, 3, 5, 9, 16] {
        assert_eq!(qkv.dispatch_count(rows).unwrap(), 4);
        assert_eq!(
            cost_route::native_projection_dispatches(&parts, rows, false).unwrap(),
            4
        );
    }
    assert_eq!(qkv.dispatch_count(native_matrix::MAX_ROWS + 4).unwrap(), 7);
    assert_eq!(qkv.dispatch_count(native_matrix::MAX_ROWS + 8).unwrap(), 7);
    assert_eq!(qkv.dispatch_count(native_matrix::MAX_ROWS).unwrap(), 4);
    assert_eq!(
        qkv.dispatch_count(2 * native_matrix::MAX_ROWS + 8).unwrap(),
        11
    );
    let mut incompatible = parts.clone();
    incompatible[3].format = MatrixFormat::DenseF16;
    assert_eq!(
        cost_route::native_projection_dispatches(&incompatible, 8, false).unwrap(),
        4
    );
    // Existing matrix key retains order, physical shape, dtype and offsets.
    let old = qkv
        .bind_replay_topology(CudaCommandReplayKeyBuilder::new("pair", "gdn"))
        .finish();
    let changed = SharedProjectionWeight::Native {
        first_region: 0,
        parts: incompatible.into(),
    }
    .bind_replay_topology(CudaCommandReplayKeyBuilder::new("pair", "gdn"))
    .finish();
    assert_ne!(old, changed);
    assert!(native_projection::strict_dispatch_count(&[], 4).is_err());
}

struct Matrix {
    part: MatrixPart,
    bytes: Guarded<u8>,
}
impl Matrix {
    fn new(stream: &Arc<CudaStream>, part: MatrixPart, salt: usize) -> Self {
        let MatrixFormat::Block(format) = part.format else {
            unreachable!()
        };
        let mut bytes = Vec::new();
        for col in 0..part.rows as usize {
            for block in 0..HIDDEN / 256 {
                match format {
                    GgufBlockFormat::Q4K => {
                        let mut w = fixture_block(col + salt, block);
                        w.d = f16::from_f32(w.d.to_f32() / 1024.0);
                        w.dmin = f16::from_f32(w.dmin.to_f32() / 1024.0);
                        bytes.extend(w.encode());
                    }
                    GgufBlockFormat::Q5K => {
                        let mut w = fixture_q5(col + salt, block);
                        w.low.d = f16::from_f32(w.low.d.to_f32() / 1024.0);
                        w.low.dmin = f16::from_f32(w.low.dmin.to_f32() / 1024.0);
                        bytes.extend(w.encode());
                    }
                    GgufBlockFormat::Q8_0 => {
                        for sub in 0..8 {
                            // Unequal B/A payloads, no all-zero/identical leaf shortcut.
                            bytes.extend(f16::from_f32((1 + salt) as f32 / 4096.0).to_le_bytes());
                            bytes.extend((0..32).map(|i| {
                                (((i * 7 + block * 8 + sub + col + salt) % 23) as i8 - 11) as u8
                            }));
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        Self {
            part,
            bytes: Guarded::new(stream, &bytes, 0xab),
        }
    }
}
fn shape() -> AttentionShape {
    AttentionShape {
        hidden_size: HIDDEN as u64,
        key_heads: 16,
        value_heads: 32,
        key_head_dim: 128,
        value_head_dim: 128,
        qkv_features: 8192,
        value_features: 4096,
        qkvz_features: 12288,
        ba_features: 64,
        qkvzba_features: WIDTH as u64,
        conv_kernel: 4,
        conv_state_width: 3,
        epsilon: 1.0e-6,
        layer_index: 0,
        decay_parameterization: GatedDeltaDecayParameterization::NegativeRate,
        value_head_mapping: GatedDeltaValueHeadMapping::InterleavedByKeyHead,
    }
}
fn half_bits(values: &[f16]) -> Vec<u16> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn graph_count(graph: &CudaGraph) -> usize {
    let mut count = 0;
    unsafe {
        assert_eq!(
            sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count),
            sys::CUresult::CUDA_SUCCESS
        );
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    unsafe {
        assert_eq!(
            sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count),
            sys::CUresult::CUDA_SUCCESS
        );
    }
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        unsafe {
            assert_eq!(
                sys::cuGraphNodeGetType(node, &mut kind),
                sys::CUresult::CUDA_SUCCESS
            );
        }
        assert_eq!(kind, sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL);
    }
    count
}
struct Frame {
    hidden: Guarded<f32>,
    raw: Guarded<f16>,
}

fn frames(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    matrices: &[Matrix],
) -> [Vec<Frame>; 2] {
    let mut control = functions.clone();
    control.native = control.native.with_unpaired_q8_control();
    let mut result: [Vec<Frame>; 2] = std::array::from_fn(|_| Vec::new());
    for rows in [4, 8] {
        let hidden = Guarded::new(stream, &vec![0.0_f32; rows * HIDDEN], 29.0);
        let norm = Guarded::new(
            stream,
            &vec![f16::from_f32(1.0); HIDDEN],
            f16::from_f32(19.0),
        );
        let normalized = Guarded::new(stream, &vec![f16::ZERO; rows * HIDDEN], f16::from_f32(23.0));
        let outputs = std::array::from_fn::<_, 2, _>(|_| {
            Guarded::new(stream, &vec![f16::ZERO; rows * WIDTH], f16::from_f32(41.0))
        });
        // All cudarc access-event bookkeeping happens before stream capture.
        let x = hidden.pointer(stream);
        let norm_ptr = norm.pointer(stream);
        let normalized_ptr = normalized.pointer(stream);
        let ys = outputs.each_ref().map(|v| v.pointer(stream));
        let weights = matrices
            .iter()
            .map(|m| m.bytes.pointer(stream))
            .collect::<Vec<_>>();
        let graphs = [&control, functions].map(|f| {
            stream.synchronize().unwrap();
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .unwrap();
            launch_rms_norm(
                stream,
                &f.rms_norm,
                x,
                norm_ptr,
                normalized_ptr,
                rows as u64,
                HIDDEN as i32,
                shape().epsilon,
            )
            .unwrap();
            native_projection::launch_parts(
                stream,
                &f.native,
                matrices.iter().zip(&weights).map(|(m, &w)| (&m.part, w, 0)),
                normalized_ptr,
                ys[usize::from(f.native.q8_pair_enabled())],
                rows as i32,
                WIDTH as i32,
                HIDDEN as i32,
                0,
            )
            .unwrap();
            stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .unwrap()
        });
        let metadata = matrices.iter().map(|m| m.part.clone()).collect::<Vec<_>>();
        assert_eq!(graph_count(&graphs[0]), 5); // actual RMS + original four leaves
        assert_eq!(
            graph_count(&graphs[1]),
            1 + cost_route::native_projection_dispatches(&metadata, rows as u64, false).unwrap()
                as usize
        );
        let mut previous = None;
        for generation in [0, 1] {
            let values = (0..rows * HIDDEN)
                .map(|i| 1.0 + sample(i, generation + 7, 0.03125))
                .collect::<Vec<_>>();
            unsafe {
                cudarc::driver::result::memcpy_htod_async(x, &values, stream.cu_stream()).unwrap();
            }
            for graph in &graphs {
                graph.launch().unwrap();
            }
            let old = outputs[0].read(stream);
            let new = outputs[1].read(stream);
            assert!(old.iter().chain(&new).all(|x| x.is_finite()));
            assert_eq!(
                half_bits(&old),
                half_bits(&new),
                "all QKV/Z/B/A outputs; B={rows}"
            );
            if let Some(previous) = previous {
                assert_ne!(
                    previous,
                    half_bits(&new),
                    "changed input must change actual projections"
                );
            }
            previous = Some(half_bits(&new));
            assert_eq!(hidden.read(stream), values);
            normalized.read(stream);
            norm.assert_unchanged(stream);
            for arm in 0..2 {
                // Retain actual GPU projection output for downstream state carry.
                // No host-generated QKV is substituted into the recurrence.
                let raw = Guarded::new(stream, &vec![f16::ZERO; rows * WIDTH], f16::from_f32(51.0));
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        raw.pointer(stream),
                        ys[arm],
                        rows * WIDTH * 2,
                        stream.cu_stream(),
                    )
                    .unwrap();
                }
                stream.synchronize().unwrap();
                result[arm].push(Frame {
                    hidden: Guarded::new(stream, &values, 31.0_f32),
                    raw,
                });
            }
        }
        for m in matrices {
            m.bytes.assert_unchanged(stream);
        }
        // Graphs drop before hidden/norm/outputs; each raw copy owns its storage.
    }
    result
}

fn compose(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    frames: &[Frame],
    output: &Matrix,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let raw = frames.iter().map(|f| &f.raw).collect::<Vec<_>>();
    let norm = Guarded::new(stream, &vec![1.0_f32; 128], 37.0);
    let mut final_output = Vec::new();
    let (core, state) = exercise_with_raw(
        stream,
        functions,
        shape(),
        &BATCHES,
        Some(&raw),
        |index, counts, core, z| {
            let rows = counts.iter().sum::<usize>();
            let gated = Guarded::new(stream, &vec![0.0_f32; rows * HIDDEN], 31.0);
            launch_gated_norm(
                stream,
                &functions.gated_norm,
                core.pointer(stream),
                z.pointer(stream),
                norm.pointer(stream),
                gated.pointer(stream),
                rows as u64,
                shape().cuda_shape().unwrap(),
            )
            .unwrap();
            assert!(gated.read(stream).iter().all(|x| x.is_finite()));
            launch_cast(
                stream,
                &functions.f32_to_f16,
                gated.pointer(stream),
                z.pointer(stream),
                (rows * HIDDEN) as u64,
            )
            .unwrap();
            let projected =
                Guarded::new(stream, &vec![f16::ZERO; rows * HIDDEN], f16::from_f32(47.0));
            native_projection::launch_parts(
                stream,
                &functions.native,
                std::iter::once((&output.part, output.bytes.pointer(stream), 0)),
                z.pointer(stream),
                projected.pointer(stream),
                rows as i32,
                HIDDEN as i32,
                HIDDEN as i32,
                0,
            )
            .unwrap();
            let value = Guarded::new(stream, &vec![0.0_f32; rows * HIDDEN], 53.0);
            launch_residual(
                stream,
                &functions.residual_add,
                frames[index].hidden.pointer(stream),
                projected.pointer(stream),
                value.pointer(stream),
                (rows * HIDDEN) as u64,
            )
            .unwrap();
            let values = value.read(stream);
            assert!(values.iter().all(|v| v.is_finite()));
            let expected = frames[index]
                .hidden
                .read(stream)
                .iter()
                .zip(projected.read(stream))
                .map(|(x, y)| *x + y.to_f32())
                .collect::<Vec<_>>();
            assert_eq!(
                f32_bits(&values),
                f32_bits(&expected),
                "actual F32 residual boundary"
            );
            final_output.extend(values);
            frames[index].hidden.assert_unchanged(stream);
            norm.assert_unchanged(stream);
            output.bytes.assert_unchanged(stream);
        },
    );
    (core, state, final_output)
}

#[test]
#[ignore = "actual CUDA: strict production pair topology, graph input changes and GDN state/output bits"]
fn recurrent_strict_q8_pair_preserves_actual_projection_and_carried_state() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.strict-q8-pair").unwrap(),
            AttentionExecutionPolicy::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new_f32_master(&runtime).unwrap();
    let stream = runtime.context().new_stream().unwrap();
    let matrices = parts()
        .into_iter()
        .enumerate()
        .map(|(salt, p)| Matrix::new(&stream, p, salt))
        .collect::<Vec<_>>();
    let output = Matrix::new(&stream, part(GgufBlockFormat::Q5K, HIDDEN as u32, 0, 4), 4);
    let frames = frames(&stream, &provider.functions, &matrices);
    let mut control = provider.functions.clone();
    control.native = control.native.with_unpaired_q8_control();
    let old = compose(&stream, &control, &frames[0], &output);
    let new = compose(&stream, &provider.functions, &frames[1], &output);
    let bits = |v: &[Vec<f32>]| v.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&old.0),
        bits(&new.0),
        "production recurrent core outputs"
    );
    assert_eq!(
        bits(&old.1),
        bits(&new.1),
        "all carried F32 states including idle slot"
    );
    assert_eq!(
        f32_bits(&old.2),
        f32_bits(&new.2),
        "gated norm, Q5 output projection and F32 residual"
    );
    println!("strict_q8_pair_composed: actual projection widths4/8, changed-input graphs, 3 logical owners with permuted slots; model teacher remains separate");
}
