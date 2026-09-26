//! Hardware conformance of the production selector/launchers and cold packer.
//! This is correctness evidence; no timing or model-quality claim is emitted.
use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr};
use half::f16;
use std::{fmt::Debug, sync::Arc};

struct Live<T> {
    device: CudaSlice<T>,
    host: Vec<T>,
    prefix: usize,
    count: usize,
}
impl<T: DeviceRepr + Clone + PartialEq + Debug> Live<T> {
    fn new(stream: &Arc<CudaStream>, values: &[T], guard: T, prefix: usize) -> Self {
        let mut host = vec![guard.clone(); prefix];
        host.extend_from_slice(values);
        host.extend(vec![guard; 8]);
        Self {
            device: stream.clone_htod(&host).unwrap(),
            host,
            prefix,
            count: values.len(),
        }
    }
    fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        self.device.device_ptr(stream).0 + (self.prefix * std::mem::size_of::<T>()) as u64
    }
    fn read(&self, stream: &Arc<CudaStream>) -> Vec<T> {
        stream.synchronize().unwrap();
        let all = stream.clone_dtoh(&self.device).unwrap();
        let end = self.prefix + self.count;
        assert_eq!(&all[..self.prefix], &self.host[..self.prefix]);
        assert_eq!(&all[end..], &self.host[end..]);
        all[self.prefix..end].to_vec()
    }
    fn replace(&mut self, stream: &Arc<CudaStream>, values: &[T]) {
        assert_eq!(values.len(), self.count);
        self.host[self.prefix..self.prefix + self.count].clone_from_slice(values);
        stream.memcpy_htod(&self.host, &mut self.device).unwrap();
    }
}
fn source(plan: RnF16FragmentPlanV1, salt: usize) -> Vec<u8> {
    let spec = plan.source_block_spec();
    let bytes = spec.bytes_per_block as usize;
    let mut data = (0..plan.source_bytes() as usize)
        .map(|i| ((i * 37 + salt * 13) % 251) as u8)
        .collect::<Vec<_>>();
    for block in data.chunks_exact_mut(bytes) {
        let offset = if plan.source_format() == RnF16FragmentSourceFormatV1::Q6K {
            208
        } else {
            0
        };
        block[offset..offset + 2].copy_from_slice(&f16::from_f32(0.0008).to_bits().to_le_bytes());
        if offset == 0 {
            block[2..4].copy_from_slice(&f16::from_f32(0.0004).to_bits().to_le_bytes());
        }
    }
    data
}
fn matrix(
    stream: &Arc<CudaStream>,
    coeff: &CudaFunction,
    plan: RnF16FragmentPlanV1,
    salt: usize,
) -> (Vec<f16>, Live<f16>, Live<u8>) {
    let source = source(plan, salt);
    let format = match plan.source_format() {
        RnF16FragmentSourceFormatV1::Q4K => GgufBlockFormat::Q4K,
        RnF16FragmentSourceFormatV1::Q5K => GgufBlockFormat::Q5K,
        RnF16FragmentSourceFormatV1::Q6K => GgufBlockFormat::Q6K,
    };
    let dense = crate::gguf_f16_projection_materializer::convert_rn_f16_diagnostic(format, &source)
        .unwrap()
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect::<Vec<_>>();
    // Partition a gate/up source at a complete row; the packet remains one N16
    // layout, never independently padded concatenated packet streams.
    let split = (plan.n() / 2 * plan.source_row_bytes()) as usize;
    let packet = crate::gguf_rn_fragment::pack_rn_f16_fragments(
        &plan,
        &[&source[..split], &source[split..]],
    )
    .unwrap();
    let dense_device = Live::new(stream, &dense, f16::from_f32(-117.0), 3);
    let packet_device = Live::new(stream, &packet, 0xa5, 5);
    let reconstructed = Live::new(
        stream,
        &vec![f16::NAN; dense.len()],
        f16::from_f32(-119.0),
        3,
    );
    let (p, bytes, y, k, n, format, abi) = (
        packet_device.ptr(stream),
        plan.packed_bytes(),
        reconstructed.ptr(stream),
        plan.k() as u32,
        plan.n() as u32,
        plan::format_code(plan.source_format()),
        plan.packing_abi(),
    );
    let mut launch = stream.launch_builder(coeff);
    launch
        .arg(&p)
        .arg(&bytes)
        .arg(&y)
        .arg(&k)
        .arg(&n)
        .arg(&format)
        .arg(&abi);
    // SAFETY: exactly N*K outputs and the typed packet span remain live.
    unsafe { launch.launch(LaunchConfig::for_num_elems(dense.len() as u32)) }.unwrap();
    assert_eq!(
        reconstructed
            .read(stream)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        dense.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    (dense, dense_device, packet_device)
}
fn check_projection(input: &[f16], weight: &[f16], actual: &[f16], k: usize, n: usize) {
    for (r, x) in input.chunks_exact(k).enumerate() {
        for (column, w) in weight.chunks_exact(k).enumerate() {
            let products = x.iter().zip(w).map(|(x, w)| x.to_f64() * w.to_f64());
            let expected = products.clone().sum::<f64>();
            let bound = (k as f64 * f64::from(f32::EPSILON) + 0.0009765625)
                * products.map(f64::abs).sum::<f64>()
                + f16::from_bits(1).to_f64();
            let value = actual[r * n + column].to_f64();
            assert!(
                value.is_finite() && (value - expected).abs() <= bound,
                "[{r},{column}] {value} vs {expected}, bound {bound}"
            );
        }
    }
}
#[test]
#[ignore = "requires actual SM80+ CUDA; exercises production pack and physical M boundary"]
fn rn_fragment_product_launchers_keep_f16_stages_and_changed_input_graphs() {
    assert!(
        compiled_mma_target(crate::ptx::VNEXT_GGUF),
        "production PTX must contain MMA, not an unsupported-target trap"
    );
    let context = CudaContext::new(0).unwrap();
    assert!(context.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap() >= 8);
    let stream = context.new_stream().unwrap();
    let blas = CudaBlas::new(stream.clone()).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF))
        .unwrap();
    let mma = module.load_function(ENTRY).unwrap();
    let coeff = module
        .load_function("vnext_rn_fragment_coefficients")
        .unwrap();
    let silu = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap()
        .load_function(SILU_MUL_FUNCTION_NAME)
        .unwrap();
    for down_format in [
        RnF16FragmentSourceFormatV1::Q4K,
        RnF16FragmentSourceFormatV1::Q5K,
        RnF16FragmentSourceFormatV1::Q6K,
    ] {
        let gate_plan =
            RnF16FragmentPlanV1::new(RnF16FragmentSourceFormatV1::Q4K, 512, 256).unwrap();
        let down_plan = RnF16FragmentPlanV1::new(down_format, 256, 256).unwrap();
        let (gate_host, gate_dense, gate_packet) = matrix(&stream, &coeff, gate_plan, 1);
        let (down_host, down_dense, down_packet) = matrix(&stream, &coeff, down_plan, 2);
        for m in [1u64, 7, 8, 9] {
            let shape =
                Shape::new(m, 256, 256, RnF16FragmentSourceFormatV1::Q4K, down_format).unwrap();
            let elements = m as usize * 256;
            let mut input = Live::new(
                &stream,
                &vec![f16::ZERO; elements],
                f16::from_f32(-117.0),
                3,
            );
            let gate = Live::new(
                &stream,
                &vec![f16::NAN; elements * 2],
                f16::from_f32(-117.0),
                3,
            );
            let activation =
                Live::new(&stream, &vec![f16::NAN; elements], f16::from_f32(-117.0), 3);
            let output = Live::new(&stream, &vec![f16::NAN; elements], f16::from_f32(-117.0), 3);
            // DevicePtr::device_ptr waits on allocation write events and records
            // a read event on drop. Resolve every address before capture; the
            // closure must only enqueue kernels/library calls on this stream.
            // All allocations and handles outlive the graph, and host access
            // below synchronizes this same stream before observing raw writes.
            let (x, g, a, y, gate_packet_ptr, gate_dense_ptr, down_packet_ptr, down_dense_ptr) = (
                input.ptr(&stream),
                gate.ptr(&stream),
                activation.ptr(&stream),
                output.ptr(&stream),
                gate_packet.ptr(&stream),
                gate_dense.ptr(&stream),
                down_packet.ptr(&stream),
                down_dense.ptr(&stream),
            );
            let enqueue = || {
                if shape.fragment() {
                    plan::launch(&stream, &mma, shape, gate_plan, x, gate_packet_ptr, g).unwrap();
                } else {
                    shape
                        .gate
                        .launch(&blas, x, gate_dense_ptr, g, "test dense fallback gate")
                        .unwrap();
                }
                launch_silu_mul(
                    &stream,
                    &silu,
                    g,
                    a,
                    shape.intermediate,
                    shape.activation_elements,
                )
                .unwrap();
                if shape.fragment() {
                    plan::launch(&stream, &mma, shape, down_plan, a, down_packet_ptr, y).unwrap();
                } else {
                    shape
                        .down
                        .launch(&blas, a, down_dense_ptr, y, "test dense fallback down")
                        .unwrap();
                }
            };
            enqueue();
            stream.synchronize().unwrap();
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .unwrap();
            enqueue();
            let graph = stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .unwrap();
            for generation in 0..2 {
                let host = (0..elements)
                    .map(|i| f16::from_f32(((i * 13 + generation * 7) % 41) as f32 / 64.0 - 0.3125))
                    .collect::<Vec<_>>();
                input.replace(&stream, &host);
                graph.launch().unwrap();
                let gh = gate.read(&stream);
                let ah = activation.read(&stream);
                let yh = output.read(&stream);
                check_projection(&host, &gate_host, &gh, 256, 512);
                for row in 0..m as usize {
                    for col in 0..256 {
                        let v = gh[row * 512 + col].to_f64();
                        let expected = v / (1.0 + (-v).exp()) * gh[row * 512 + 256 + col].to_f64();
                        assert!(
                            ah[row * 256 + col].is_finite()
                                && (ah[row * 256 + col].to_f64() - expected).abs()
                                    <= expected.abs().max(1.0) * 0.001 + 1e-5
                        );
                    }
                }
                check_projection(&ah, &down_host, &yh, 256, 256);
                enqueue();
                for (actual, expected) in [
                    (gate.read(&stream), gh),
                    (activation.read(&stream), ah),
                    (output.read(&stream), yh),
                ] {
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                assert_eq!(input.read(&stream), host);
            }
        }
        assert_eq!(gate_dense.read(&stream), gate_host);
        assert_eq!(down_dense.read(&stream), down_host);
        assert_eq!(
            gate_packet.read(&stream),
            gate_packet.host[gate_packet.prefix..gate_packet.prefix + gate_packet.count]
        );
        assert_eq!(
            down_packet.read(&stream),
            down_packet.host[down_packet.prefix..down_packet.prefix + down_packet.count]
        );
    }
}
