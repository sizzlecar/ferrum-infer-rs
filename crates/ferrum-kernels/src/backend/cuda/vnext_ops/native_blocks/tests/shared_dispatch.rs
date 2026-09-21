//! Production dispatch, captured graph identity, and full output/extent oracles.
use super::*;
use cudarc::driver::{sys, CudaGraph, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{ElementType, WeightId};
use std::ffi::CStr;

fn captured_kernel(graph: &CudaGraph) -> String {
    // SAFETY: The graph is alive and not modified concurrently. CUDA owns the
    // node parameter/name storage; only the kernel name is copied.
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
        let mut names = Vec::new();
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
            let mut name = std::ptr::null();
            assert_eq!(
                sys::cuFuncGetName(&mut name, params.assume_init().func),
                sys::CUresult::CUDA_SUCCESS
            );
            assert!(!name.is_null());
            names.push(CStr::from_ptr(name).to_str().unwrap().to_owned());
        }
        assert_eq!(names.len(), 1, "one physical matrix part is one launch");
        names.pop().unwrap()
    }
}

#[allow(clippy::too_many_arguments)]
fn check<I: Scalar, O: Scalar>(
    context: &Arc<CudaContext>,
    format: GgufBlockFormat,
    rows: usize,
    inputs: usize,
    outputs: usize,
    input_type: ElementType,
    output_type: ElementType,
    expected_kernel: &str,
) {
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(context).unwrap();
    let (mut encoded, _) = matrix(format, outputs, inputs / 256);
    let scale_offset = if format == GgufBlockFormat::Q6K {
        208
    } else {
        0
    };
    for (index, block) in encoded.chunks_exact_mut(format.block_bytes()).enumerate() {
        let col = index / (inputs / 256);
        // Distinct finite half scales for every column, including large parts.
        // Byte-offset weights below also exercise unaligned native decoding.
        let scale = (16 + col as u16).to_le_bytes();
        block[scale_offset..scale_offset + 2].copy_from_slice(&scale);
    }
    let mut decoded = vec![0.0_f32; outputs * inputs];
    format.decode(&encoded, &mut decoded).unwrap();
    let sign = |k: usize| ((k * 11 % 37) % 3) as f32 - 1.0;
    let factors = (0..rows)
        .map(|row| I::from_f32((row + 1) as f32 / 127.31).as_f32())
        .collect::<Vec<_>>();
    // x[row,k] = exactly stored factor[row] * {-1,0,1}. Compute the
    // independent F64 column sums once; this checks every output without an
    // O(M*N*K) CPU oracle or weakening the established F32/half error bound.
    let reference = decoded
        .chunks_exact(inputs)
        .map(|weight| {
            let sum = weight
                .iter()
                .enumerate()
                .map(|(k, &w)| f64::from(w) * f64::from(sign(k)))
                .sum::<f64>();
            let absolute = weight
                .iter()
                .enumerate()
                .map(|(k, &w)| (f64::from(w) * f64::from(sign(k))).abs())
                .sum::<f64>();
            (sum, absolute)
        })
        .collect::<Vec<_>>();
    let mut input = vec![I::from_f32(-12345.0); 3 + rows * inputs + 5];
    for row in 0..rows {
        for k in 0..inputs {
            input[3 + row * inputs + k] = I::from_f32(factors[row] * sign(k));
        }
    }
    let mut weights = vec![0xcc_u8; 5];
    weights.extend_from_slice(&encoded);
    weights.extend_from_slice(&[0xcc; 7]);
    // A large logical output stride must not qualify a small physical part.
    let stride = (outputs + 7).max(8192);
    let initial = vec![O::from_f32(-12345.0); 8 + rows * stride + 7];
    let input_gpu = stream.clone_htod(&input).unwrap();
    let weights_gpu = stream.clone_htod(&weights).unwrap();
    let mut output_gpu = stream.clone_htod(&initial).unwrap();
    let part = weights::MatrixPart {
        component_id: WeightId::new("component.midrow-dispatch").unwrap(),
        format: weights::MatrixFormat::Block(format),
        rows: outputs as u32,
        columns: inputs as u32,
        output_offset: 3,
        transform: None,
        signs_region: None,
    };
    for captured in [false, true] {
        stream.memcpy_htod(&initial, &mut output_gpu).unwrap();
        stream.synchronize().unwrap();
        {
            let input_view = input_gpu.slice(3..3 + rows * inputs);
            let weight_view = weights_gpu.slice(5..5 + encoded.len());
            let mut output_view = output_gpu.slice_mut(8..8 + rows * stride);
            let (ip, _ig) = input_view.device_ptr(&stream);
            let (wp, _wg) = weight_view.device_ptr(&stream);
            let (op, _og) = output_view.device_ptr_mut(&stream);
            let launch = || {
                kernels
                    .linear_with_precision(
                        &stream,
                        ip,
                        wp,
                        op,
                        &part,
                        rows as u32,
                        stride as u32,
                        input_type,
                        output_type,
                    )
                    .unwrap()
            };
            if captured {
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .unwrap();
                launch();
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                ).unwrap().expect("nonempty production linear graph");
                assert_eq!(captured_kernel(&graph), expected_kernel);
                graph.launch().unwrap();
                stream.synchronize().unwrap();
            } else {
                launch();
            }
        }
        let actual = stream.clone_dtoh(&output_gpu).unwrap();
        for (index, value) in actual.iter().enumerate() {
            let position = index
                .checked_sub(8)
                .filter(|&i| i < rows * stride && (3..3 + outputs).contains(&(i % stride)));
            if let Some(position) = position {
                let row = position / stride;
                let col = position % stride - 3;
                let (base, absolute) = reference[col];
                let expected = base * f64::from(factors[row]);
                let sum_abs = absolute * f64::from(factors[row]).abs();
                let bound =
                    (inputs as f64 * f64::from(f32::EPSILON) + O::ROUNDING) * sum_abs + 1e-6;
                let observed = value.as_f32();
                assert!(
                    observed.is_finite() && (f64::from(observed) - expected).abs() <= bound,
                    "{format:?} {rows}x{inputs}x{outputs}/{input_type:?}->{output_type:?} captured={captured} at {row},{col}: {observed} vs {expected}, bound {bound}"
                );
            } else {
                assert_eq!(
                    value.as_f32().to_bits(),
                    initial[index].as_f32().to_bits(),
                    "partition/outer guard {index}"
                );
            }
        }
    }
    assert!(stream
        .clone_dtoh(&input_gpu)
        .unwrap()
        .iter()
        .zip(&input)
        .all(|(a, b)| a.as_f32().to_bits() == b.as_f32().to_bits()));
    assert_eq!(stream.clone_dtoh(&weights_gpu).unwrap(), weights);
}

#[test]
#[ignore = "requires an actual CUDA device; production matrix partition/graph conformance"]
fn shared_gemm_midrow_production_dispatch_preserves_numeric_and_graph_boundaries_on_cuda() {
    use ElementType::{F16, F32};
    use GgufBlockFormat::{Q4K, Q5K, Q6K};
    let context = CudaContext::new(0).expect("native matrix dispatch requires CUDA");
    for rows in [31, 32, 33, 63, 64, 65] {
        for (format, outputs, shared) in [
            (Q5K, 2560, "vnext_gguf_gemm_q5k_f16"),
            (Q6K, 4096, "vnext_gguf_gemm_q6k_f16"),
        ] {
            check::<f16, f16>(
                &context,
                format,
                rows,
                4096,
                outputs,
                F16,
                F16,
                if rows == 31 {
                    "vnext_gguf_linear_tiled_f16"
                } else {
                    shared
                },
            );
        }
    }
    // Additional wide/short-K part, the real wider Q5 projection, and small
    // physical parts inside a much larger logical output stride.
    for (format, rows, inputs, outputs, name) in [
        (Q5K, 32, 2560, 8192, "vnext_gguf_gemm_q5k_f16"),
        (Q5K, 33, 4096, 8192, "vnext_gguf_gemm_q5k_f16"),
        (Q5K, 32, 256, 8192, "vnext_gguf_linear_tiled_f16"),
        (Q5K, 32, 4096, 127, "vnext_gguf_linear_tiled_f16"),
        (Q6K, 32, 4096, 127, "vnext_gguf_linear_tiled_f16"),
        (Q4K, 64, 4096, 4096, "vnext_gguf_linear_q4k_tiled_f16"),
        (Q5K, 128, 256, 129, "vnext_gguf_gemm_q5k_f16"),
    ] {
        check::<f16, f16>(&context, format, rows, inputs, outputs, F16, F16, name);
    }
    // F32 activations include non-half-representable values. These exercise
    // the transformed-input interface and F32 output exclusion numerically.
    check::<f32, f16>(
        &context,
        Q5K,
        32,
        4096,
        2560,
        F32,
        F16,
        "vnext_gguf_linear_tiled_f32_f16",
    );
    check::<f32, f32>(
        &context,
        Q6K,
        32,
        4096,
        4096,
        F32,
        F32,
        "vnext_gguf_linear_tiled_f32",
    );
}
