//! Complete prepared-adapter qualification; raw experiment timing lives in
//! ferrum-native-ops. Every allocation is held through stream completion.
use super::*;
use cudarc::driver::{sys::CUdevice_attribute::*, CudaContext, CudaSlice, CudaStream, DevicePtr};
use half::f16;
use std::sync::Arc;

#[cfg(all(
    feature = "cuda-q4-shared-weight-decode",
    feature = "cuda-q5-shared-weight-decode"
))]
#[path = "pdl_tests.rs"]
mod pdl_tests;

struct Buffer {
    data: CudaSlice<u8>,
    bytes: usize,
}
impl Buffer {
    fn new(stream: &Arc<CudaStream>, bytes: u64) -> Self {
        let bytes = usize::try_from(bytes).unwrap();
        Self {
            data: stream.clone_htod(&vec![0x35; bytes + 64]).unwrap(),
            bytes,
        }
    }
    fn span(&self, stream: &Arc<CudaStream>) -> DeviceSpan {
        let (address, lease) = self.data.device_ptr(stream);
        drop(lease);
        DeviceSpan {
            address: if self.bytes == 0 { 0 } else { address + 32 },
            bytes: self.bytes as u64,
        }
    }
    fn write(&mut self, stream: &Arc<CudaStream>, bytes: &[u8]) {
        assert!(bytes.len() <= self.bytes);
        stream
            .memcpy_htod(bytes, &mut self.data.slice_mut(32..32 + bytes.len()))
            .unwrap();
    }
    fn read(&self, stream: &Arc<CudaStream>) -> Vec<u8> {
        let bytes = stream.clone_dtoh(&self.data).unwrap();
        assert!(bytes[..32]
            .iter()
            .chain(&bytes[32 + self.bytes..])
            .all(|b| *b == 0x35));
        bytes[32..32 + self.bytes].to_vec()
    }
}

#[test]
#[cfg(feature = "cuda-q4-shared-weight-decode")]
#[ignore = "requires CUDA and the locked additive Q4 MMVQ shared-weight native export"]
fn prepared_shared_q4_mmvq_preserves_marker_buffers_and_rejects_ineligible_spans() {
    let context = CudaContext::new(0).unwrap();
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let attr = |key| u32::try_from(context.attribute(key).unwrap()).unwrap();
    let device = Device {
        architecture: attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 100
            + attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) * 10,
        multiprocessors: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        maximum_dynamic_shared_bytes: u64::from(attr(
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )),
    };
    for (inputs, outputs) in [(512, 17), (5120, 10240), (6144, 5120)] {
        for (rows, extended) in [(8, false), (4, true), (8, true)] {
            verify_prepared_shared_mmvq(
                &stream,
                device,
                Format::Q4K,
                inputs,
                outputs,
                rows,
                extended,
            );
        }
    }
}

#[test]
#[cfg(feature = "cuda-q5-shared-weight-decode")]
#[ignore = "requires CUDA and the locked additive Q5 MMVQ shared-weight native export"]
fn prepared_shared_q5_mmvq_preserves_marker_buffers_and_rejects_ineligible_spans() {
    let context = CudaContext::new(0).unwrap();
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let attr = |key| u32::try_from(context.attribute(key).unwrap()).unwrap();
    let device = Device {
        architecture: attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 100
            + attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) * 10,
        multiprocessors: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        maximum_dynamic_shared_bytes: u64::from(attr(
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )),
    };
    for (inputs, outputs) in [(512, 17), (5120, 10240), (6144, 5120)] {
        for (rows, extended) in [(8, false), (4, true), (8, true)] {
            verify_prepared_shared_mmvq(
                &stream,
                device,
                Format::Q5K,
                inputs,
                outputs,
                rows,
                extended,
            );
        }
    }
}

fn verify_prepared_shared_mmvq(
    stream: &Arc<CudaStream>,
    device: Device,
    format: Format,
    inputs: u32,
    outputs: u32,
    row_count: u32,
    extended: bool,
) {
    let baseline = PreparedUpstreamLinear::new(
        Algorithm::Mmvq,
        Arithmetic::MarkerV2,
        format,
        Layout::Columns,
        row_count,
        inputs,
        outputs,
        device,
    )
    .unwrap();
    let q4_mode = if extended {
        ferrum_types::Q4MmvqWeightDecode::SharedM4M8
    } else {
        ferrum_types::Q4MmvqWeightDecode::SharedM8
    };
    let q5_mode = if extended {
        ferrum_types::Q5MmvqWeightDecode::SharedM4M8
    } else {
        ferrum_types::Q5MmvqWeightDecode::SharedM8
    };
    let select = |plan: PreparedUpstreamLinear| match format {
        Format::Q4K => plan.with_q4_mmvq_weight_decode(q4_mode),
        Format::Q5K => plan.with_q5_mmvq_weight_decode(q5_mode),
        _ => unreachable!("shared MMVQ format"),
    };
    let shared = select(baseline.clone()).unwrap();
    assert_eq!(shared.geometry(), baseline.geometry());
    assert_eq!(
        shared.q4_mmvq_weight_decode(),
        if format == Format::Q4K {
            q4_mode
        } else {
            ferrum_types::Q4MmvqWeightDecode::Baseline
        }
    );
    assert_eq!(
        shared.q5_mmvq_weight_decode(),
        if format == Format::Q5K {
            q5_mode
        } else {
            ferrum_types::Q5MmvqWeightDecode::Baseline
        }
    );
    let other_format = if format == Format::Q4K {
        Format::Q5K
    } else {
        Format::Q4K
    };
    for (algorithm, format, layout, rows) in [
        // SharedM8 must keep rejecting M4; the new mode must reject M1.
        (
            Algorithm::Mmvq,
            format,
            Layout::Columns,
            if extended { 1 } else { 4 },
        ),
        (Algorithm::Mmvq, other_format, Layout::Columns, 8),
        (Algorithm::Mmvq, format, Layout::Channels, 8),
        (Algorithm::Mmq, format, Layout::Columns, 8),
    ] {
        let other = PreparedUpstreamLinear::new(
            algorithm,
            Arithmetic::MarkerV2,
            format,
            layout,
            rows,
            inputs,
            outputs,
            device,
        )
        .unwrap();
        assert!(matches!(select(other), Err(Error::UnsupportedGeometry)));
    }
    let geometry = baseline.geometry();
    let block_bytes = if format == Format::Q5K { 176 } else { 144 };
    let input_stride = u64::from(inputs) + 3;
    let output_stride = u64::from(outputs) + 3;
    for nonfinite_input in [false, true] {
        for bad_weight in [false, true] {
            let mut input = Buffer::new(&stream, input_stride * u64::from(row_count) * 2);
            let mut weights = Buffer::new(&stream, geometry.weight_bytes);
            let mut xs = (0..input_stride * u64::from(row_count))
                .map(|i| f16::from_f32(((i % 29) as f32 - 14.0) * 0.03125).to_bits())
                .collect::<Vec<_>>();
            if nonfinite_input {
                xs[0] = 0x7e00;
            }
            let input_bytes = xs
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>();
            input.write(&stream, &input_bytes);
            let mut ws = vec![0_u8; usize::try_from(geometry.weight_bytes).unwrap()];
            let real_weight_bytes =
                usize::try_from(u64::from(outputs) * u64::from(inputs / 256) * block_bytes)
                    .unwrap();
            assert!(real_weight_bytes <= ws.len());
            // The native plan owns padded output rows; keep that tail initialized.
            for (block, data) in ws[..real_weight_bytes]
                .chunks_exact_mut(block_bytes as usize)
                .enumerate()
            {
                data[..2].copy_from_slice(&f16::from_f32(0.001953125).to_bits().to_le_bytes());
                data[2..4].copy_from_slice(&f16::from_f32(0.0009765625).to_bits().to_le_bytes());
                data[4..16].fill(3);
                for (i, byte) in data[16..].iter_mut().enumerate() {
                    *byte = (block * 17 + i * 13) as u8;
                }
            }
            if bad_weight {
                ws[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
            }
            weights.write(&stream, &ws);
            let mut observations = Vec::new();
            for plan in [&baseline, &shared, &shared] {
                let converted = Buffer::new(&stream, geometry.converted_bytes);
                let packed = Buffer::new(&stream, geometry.packed_bytes);
                let raw = Buffer::new(&stream, geometry.output_bytes);
                let output = Buffer::new(&stream, output_stride * u64::from(row_count) * 2);
                let rows = Buffer::new(&stream, plan.row_poison_bytes());
                let weight_flag = Buffer::new(&stream, 4);
                let fixup = DeviceSpan {
                    address: 0,
                    bytes: 0,
                };
                let s = stream.cu_stream().cast();
                unsafe {
                    plan.check_weights(weights.span(&stream), weight_flag.span(&stream), s)
                        .unwrap();
                    plan.pack(
                        input.span(&stream),
                        input_stride as u32,
                        converted.span(&stream),
                        packed.span(&stream),
                        rows.span(&stream),
                        s,
                    )
                    .unwrap();
                    plan.dot(
                        weights.span(&stream),
                        packed.span(&stream),
                        raw.span(&stream),
                        fixup,
                        s,
                    )
                    .unwrap();
                    plan.cast(
                        raw.span(&stream),
                        output.span(&stream),
                        output_stride as u32,
                        rows.span(&stream),
                        weight_flag.span(&stream),
                        s,
                    )
                    .unwrap();
                }
                stream.synchronize().unwrap();
                let raw_before = raw.read(&stream);
                let short = DeviceSpan {
                    bytes: packed.span(&stream).bytes - 1,
                    ..packed.span(&stream)
                };
                assert!(matches!(
                    unsafe { plan.dot(weights.span(&stream), short, raw.span(&stream), fixup, s) },
                    Err(Error::Span)
                ));
                assert!(matches!(
                    unsafe {
                        plan.dot(
                            weights.span(&stream),
                            packed.span(&stream),
                            weights.span(&stream),
                            fixup,
                            s,
                        )
                    },
                    Err(Error::Span)
                ));
                assert!(matches!(
                    unsafe {
                        plan.pack(
                            input.span(&stream),
                            inputs - 1,
                            converted.span(&stream),
                            packed.span(&stream),
                            rows.span(&stream),
                            s,
                        )
                    },
                    Err(Error::Span)
                ));
                stream.synchronize().unwrap();
                assert_eq!(raw.read(&stream), raw_before);
                let output_bytes = output.read(&stream);
                for row in output_bytes.chunks_exact(output_stride as usize * 2) {
                    assert!(row[outputs as usize * 2..].iter().all(|byte| *byte == 0x35));
                }
                observations.push([
                    converted.read(&stream),
                    packed.read(&stream),
                    raw_before,
                    output_bytes,
                    rows.read(&stream),
                    weight_flag.read(&stream),
                ]);
            }
            assert_eq!(observations[1], observations[0]);
            assert_eq!(observations[2], observations[0]);
            assert_eq!(input.read(&stream), input_bytes);
            assert_eq!(weights.read(&stream), ws);
        }
    }
}
