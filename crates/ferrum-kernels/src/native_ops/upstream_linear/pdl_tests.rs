//! Product prepared-adapter coverage; timing and dependency-edge inspection
//! remain in the independently linked native experiment.
use super::*;
use cudarc::driver::sys;

struct Graph {
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
}
impl Graph {
    fn capture(stream: &Arc<CudaStream>, enqueue: impl FnOnce()) -> Self {
        unsafe {
            assert_eq!(
                sys::cuStreamBeginCapture_v2(
                    stream.cu_stream(),
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                ),
                sys::CUresult::CUDA_SUCCESS
            );
            enqueue();
            let mut graph = std::ptr::null_mut();
            assert_eq!(
                sys::cuStreamEndCapture(stream.cu_stream(), &mut graph),
                sys::CUresult::CUDA_SUCCESS
            );
            let mut executable = std::ptr::null_mut();
            assert_eq!(
                sys::cuGraphInstantiateWithFlags(&mut executable, graph, 0),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuGraphUpload(executable, stream.cu_stream()),
                sys::CUresult::CUDA_SUCCESS
            );
            stream.synchronize().unwrap();
            Self { graph, executable }
        }
    }
    fn launch(&self, stream: &Arc<CudaStream>) {
        assert_eq!(
            unsafe { sys::cuGraphLaunch(self.executable, stream.cu_stream()) },
            sys::CUresult::CUDA_SUCCESS
        );
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuGraphExecDestroy(self.executable);
            let _ = sys::cuGraphDestroy(self.graph);
        }
    }
}

// Query the linked native cold capability independently of the adapter's
// automatic selection. Missing capability and unsupported GPU/loaded PTX are
// valid ordered fallbacks, while a positive result must activate PDL.
fn native_support(plan: &PreparedUpstreamLinear) -> Option<bool> {
    #[cfg(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_pdl))]
    {
        let mut supported = 0;
        Error::from_status(unsafe {
            ffi::ferrum_upstream_mmvq_pdl_support_v1(
                &plan.raw,
                pdl::dot_kind(plan) as u32,
                &mut supported,
            )
        })
        .unwrap();
        Some(supported != 0)
    }
    #[cfg(not(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_pdl)))]
    {
        let _ = plan;
        None
    }
}

fn verify(
    stream: &Arc<CudaStream>,
    device: Device,
    format: Format,
    rows: u32,
    inputs: u32,
    outputs: u32,
    shared: bool,
) {
    let ordered = PreparedUpstreamLinear::new(
        Algorithm::Mmvq,
        Arithmetic::MarkerV2,
        format,
        Layout::Columns,
        rows,
        inputs,
        outputs,
        device,
    )
    .unwrap();
    let ordered = if shared {
        match format {
            Format::Q4K => {
                ordered.with_q4_mmvq_weight_decode(ferrum_types::Q4MmvqWeightDecode::SharedM4M8)
            }
            Format::Q5K => {
                ordered.with_q5_mmvq_weight_decode(ferrum_types::Q5MmvqWeightDecode::SharedM4M8)
            }
            _ => unreachable!(),
        }
        .unwrap()
    } else {
        ordered
    };
    assert_eq!(ordered.mmvq_launch_kind(), MmvqLaunchKind::Ordered);
    let native_supported = native_support(&ordered);
    let candidate = ordered.clone().with_automatic_mmvq_launch().unwrap();
    let expected_launch = if native_supported == Some(true) {
        MmvqLaunchKind::Programmatic
    } else {
        MmvqLaunchKind::Ordered
    };
    assert_eq!(
        candidate.mmvq_launch_kind(),
        expected_launch,
        "automatic launch must agree with the linked native GPU/PTX support result"
    );
    assert_eq!(ordered.geometry(), candidate.geometry());
    assert_eq!(ordered.pack_work_counts(), candidate.pack_work_counts());
    assert_eq!(
        ordered.q4_mmvq_weight_decode(),
        candidate.q4_mmvq_weight_decode()
    );
    assert_eq!(
        ordered.q5_mmvq_weight_decode(),
        candidate.q5_mmvq_weight_decode()
    );
    let geometry = ordered.geometry();
    let (input_stride, output_stride, output_offset) = (inputs + 3, outputs + 5, 2);
    let mut input = Buffer::new(stream, u64::from(rows * input_stride) * 2);
    let mut weights = Buffer::new(stream, geometry.weight_bytes);
    // Six independently owned spans per arm; native scratch must not alias.
    let buffers = [&ordered, &candidate].map(|plan| {
        [
            Buffer::new(stream, geometry.converted_bytes),
            Buffer::new(stream, geometry.packed_bytes),
            Buffer::new(stream, geometry.output_bytes),
            Buffer::new(stream, u64::from(rows * output_stride) * 2),
            Buffer::new(stream, plan.row_poison_bytes()),
            Buffer::new(stream, 4),
        ]
    });
    let plans = [&ordered, &candidate];
    let block_bytes = match format {
        Format::Q4K => 144,
        Format::Q5K => 176,
        Format::Iq4Xs => 136,
    };
    let real_bytes = outputs as usize * (inputs / 256) as usize * block_bytes;
    let mut finite_weights = vec![0; geometry.weight_bytes as usize];
    for (block, bytes) in finite_weights[..real_bytes]
        .chunks_exact_mut(block_bytes)
        .enumerate()
    {
        bytes[..2].copy_from_slice(&f16::from_f32(0.00137).to_bits().to_le_bytes());
        let codes = if format == Format::Iq4Xs {
            bytes[2..8].fill(0x53);
            8
        } else {
            bytes[2..4].copy_from_slice(&f16::from_f32(0.00071).to_bits().to_le_bytes());
            bytes[4..16].fill(3);
            16
        };
        for (i, byte) in bytes[codes..].iter_mut().enumerate() {
            *byte = (block * 17 + i * 13) as u8;
        }
    }
    let enqueue = |arm: usize, input: &Buffer, weights: &Buffer| {
        let p = plans[arm];
        let b = &buffers[arm];
        let out = b[3].span(stream);
        let output = DeviceSpan {
            address: out.address + u64::from(output_offset * 2),
            bytes: out.bytes - u64::from(output_offset * 2),
        };
        let s = stream.cu_stream().cast();
        unsafe {
            p.pack(
                input.span(stream),
                input_stride,
                b[0].span(stream),
                b[1].span(stream),
                b[4].span(stream),
                s,
            )
            .unwrap();
            p.dot(
                weights.span(stream),
                b[1].span(stream),
                b[2].span(stream),
                DeviceSpan {
                    address: 0,
                    bytes: 0,
                },
                s,
            )
            .unwrap();
            p.cast(
                b[2].span(stream),
                output,
                output_stride,
                b[4].span(stream),
                b[5].span(stream),
                s,
            )
            .unwrap();
        }
    };
    // Capture owns no host data; all device spans outlive graphs and completion.
    let graphs = [0, 1].map(|arm| Graph::capture(stream, || enqueue(arm, &input, &weights)));
    let mut first_finite = None;
    for generation in 0..4 {
        let mut words = (0..rows * input_stride)
            .map(|i| {
                f16::from_f32(((i * 17 + generation * 23) % 101) as f32 / 256.0 - 0.2).to_bits()
            })
            .collect::<Vec<_>>();
        if generation == 1 {
            words[0] = 0x7e00;
        }
        let input_bytes = words
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let mut weight_bytes = finite_weights.clone();
        if generation == 2 {
            weight_bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        }
        input.write(stream, &input_bytes);
        weights.write(stream, &weight_bytes);
        for arm in 0..2 {
            unsafe {
                plans[arm]
                    .check_weights(
                        weights.span(stream),
                        buffers[arm][5].span(stream),
                        stream.cu_stream().cast(),
                    )
                    .unwrap();
            }
        }
        stream.synchronize().unwrap();
        // First execute eagerly; subsequent inputs exercise captured fresh replay.
        for arm in 0..2 {
            if generation == 0 {
                enqueue(arm, &input, &weights);
            } else {
                graphs[arm].launch(stream);
            }
        }
        stream.synchronize().unwrap();
        let observed = buffers
            .each_ref()
            .map(|b| b.each_ref().map(|v| v.read(stream)));
        assert_eq!(observed[0], observed[1], "format={format:?} M={rows} K={inputs} N={outputs} shared={shared} generation={generation}");
        let output = &observed[1][3];
        for (row_index, row) in output.chunks_exact(output_stride as usize * 2).enumerate() {
            assert!(row[..output_offset as usize * 2]
                .iter()
                .chain(&row[(output_offset + outputs) as usize * 2..])
                .all(|v| *v == 0x35));
            for word in row[output_offset as usize * 2..(output_offset + outputs) as usize * 2]
                .chunks_exact(2)
            {
                let bits = u16::from_le_bytes([word[0], word[1]]);
                if generation == 2 || (generation == 1 && row_index == 0) {
                    assert_eq!(bits, 0x7e00);
                } else {
                    assert!(f16::from_bits(bits).is_finite());
                }
            }
        }
        let row_flags = observed[1][4]
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(row_flags[0] != 0, generation == 1);
        assert!(row_flags[1..rows as usize].iter().all(|v| *v == 0));
        assert_eq!(
            u32::from_le_bytes(observed[1][5][..4].try_into().unwrap()) != 0,
            generation == 2
        );
        if generation == 0 {
            first_finite = Some(output.clone());
        }
        if generation == 3 {
            assert_ne!(first_finite.as_ref().unwrap(), output);
        }
        assert_eq!(input.read(stream), input_bytes);
        assert_eq!(weights.read(stream), weight_bytes);
    }
    let b = &buffers[1];
    let s = stream.cu_stream().cast();
    let raw_before = b[2].read(stream);
    let packed = b[1].span(stream);
    let fixup = DeviceSpan {
        address: 0,
        bytes: 0,
    };
    assert!(matches!(
        unsafe {
            candidate.dot(
                weights.span(stream),
                DeviceSpan {
                    bytes: packed.bytes - 1,
                    ..packed
                },
                b[2].span(stream),
                fixup,
                s,
            )
        },
        Err(Error::Span)
    ));
    assert!(matches!(
        unsafe { candidate.dot(weights.span(stream), packed, weights.span(stream), fixup, s) },
        Err(Error::Span)
    ));
    stream.synchronize().unwrap();
    assert_eq!(raw_before, b[2].read(stream));
    println!(
        "{}",
        serde_json::json!({"case":"prepared_mmvq_programmatic", "format":format!("{format:?}"),"M":rows,"K":inputs,"N":outputs,"shared":shared,"ordered":format!("{:?}",ordered.mmvq_launch_kind()),"candidate":format!("{:?}",candidate.mmvq_launch_kind()),"native_support":native_supported,"fallback":match native_supported { None => Some("native_capability_not_compiled"), Some(false) => Some("native_gpu_or_loaded_ptx_unsupported"), Some(true) => None },"full_buffers_equal":true,"captured_fresh_poison_recovery":true,"guards_padding_and_rejected_spans":true})
    );
}

#[test]
#[ignore = "requires CUDA and locked native MMVQ exports; qualifies PDL or an explicit ordered fallback"]
fn prepared_mmvq_programmatic_launch_matches_ordered_across_actual_templates() {
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
    for format in [Format::Q4K, Format::Q5K, Format::Iq4Xs] {
        for (rows, inputs) in [(1, 256), (1, 5120), (4, 768), (8, 768)] {
            verify(&stream, device, format, rows, inputs, 17, false);
        }
    }
    for format in [Format::Q4K, Format::Q5K] {
        verify(&stream, device, format, 4, 512, 17, true);
    }
    verify(&stream, device, Format::Q4K, 8, 5120, 10240, true);
    verify(&stream, device, Format::Q5K, 8, 6144, 5120, true);
}
