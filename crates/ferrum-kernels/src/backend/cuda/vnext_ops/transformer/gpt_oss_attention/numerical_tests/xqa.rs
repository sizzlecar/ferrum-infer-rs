//! Native F16 XQA correctness and paired, resident single-layer diagnostics.
use super::*;
use crate::native_ops::flashinfer_xqa::{
    Bindings, DeviceSpan, PreparedXqa, Request, FLASHINFER_XQA_ABI,
};
use cudarc::driver::sys::CUdevice_attribute::*;
use cudarc::driver::sys::CUevent_flags;
use std::time::Instant;

// Reuse the same explicit capture/upload/drop lifecycle as other native tests.
#[path = "../../../native_blocks/tests/captured.rs"]
mod captured;

struct Fixture {
    query: Guarded<f16>,
    sinks: Guarded<f16>,
    sinks_f32: Guarded<f32>,
    kv: KvFixture,
    output: Guarded<f16>,
    semaphores: Guarded<u32>,
    scratch: Guarded<u8>,
    scratch_offset: usize,
    prepared: PreparedXqa,
    bindings: Bindings,
}

impl Fixture {
    fn new(stream: &Arc<CudaStream>, positions: &[usize], window: u64) -> Self {
        let shape = attention_shape(window);
        let context = stream.context();
        let get = |attr| context.attribute(attr).unwrap() as u32;
        let kv = KvFixture::new(stream, positions, shape);
        let prepared = PreparedXqa::new(Request {
            abi: FLASHINFER_XQA_ABI,
            size: std::mem::size_of::<Request>() as u32,
            batch: positions.len() as u32,
            max_pages_per_sequence: shape.maximum_pages().unwrap() as u32,
            binding_stride_bytes: kv.stride as u32,
            sliding_window: window as u32,
            device_ordinal: 0,
            architecture: get(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 100
                + get(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) * 10,
            multiprocessors: get(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            reserved: 0,
            maximum_dynamic_shared_bytes: get(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
                as u64,
        })
        .unwrap();
        let elements = positions.len() * shape.query_features as usize;
        let query = Guarded::new(
            stream,
            &(0..elements).map(|i| sample(i, 11)).collect::<Vec<_>>(),
            f16::from_f32(81.0),
        );
        let sink_values = (0..64)
            .map(|i| f16::from_f32(if i % 2 == 0 { 6.0 } else { -3.0 }))
            .collect::<Vec<_>>();
        let sinks = Guarded::new(stream, &sink_values, f16::from_f32(82.0));
        let sinks_f32 = Guarded::new(
            stream,
            &sink_values.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            83.0,
        );
        let output = Guarded::new(stream, &vec![f16::ZERO; elements], f16::from_f32(84.0));
        // Nonzero initial semaphores make a missing multi-split reset observable;
        // single-split execution must leave them untouched.
        let semaphores = Guarded::new(
            stream,
            &vec![0xBAD0_u32; prepared.plan().semaphore_bytes as usize / 4],
            0xDEAD_BEEF,
        );
        let scratch = Guarded::new(
            stream,
            &vec![0xCD_u8; prepared.plan().scratch_bytes as usize + 1023],
            0xEF,
        );
        let scratch_address = (scratch.pointer(stream) + 1023) & !1023;
        let scratch_offset = (scratch_address - scratch.pointer(stream)) as usize;
        let span = |address, bytes| DeviceSpan { address, bytes };
        let bindings = Bindings {
            query: span(query.pointer(stream), prepared.plan().request.query_bytes()),
            sinks_f32: span(sinks_f32.pointer(stream), 256),
            packed_metadata: span(
                kv.bindings.pointer(stream),
                prepared.plan().request.metadata_bytes().unwrap(),
            ),
            output: span(
                output.pointer(stream),
                prepared.plan().request.query_bytes(),
            ),
            semaphores: span(semaphores.pointer(stream), prepared.plan().semaphore_bytes),
            scratch: if prepared.plan().scratch_bytes == 0 {
                DeviceSpan::default()
            } else {
                span(scratch_address, prepared.plan().scratch_bytes)
            },
        };
        prepared.validate(&bindings).unwrap();
        Self {
            query,
            sinks,
            sinks_f32,
            kv,
            output,
            semaphores,
            scratch,
            scratch_offset,
            prepared,
            bindings,
        }
    }

    fn launch(&self, stream: &Arc<CudaStream>) {
        // All checked buffers/pages remain owned by this fixture through synchronization.
        unsafe { self.prepared.run(&self.bindings, stream.cu_stream().cast()) }.unwrap();
    }

    fn check(&self, stream: &Arc<CudaStream>, positions: &[usize], window: u64) {
        let shape = attention_shape(window);
        let query = self.query.read(stream);
        let sinks = self.sinks.read(stream);
        let kv = self.kv.read(stream);
        let output = self.output.read(stream);
        for (p, position) in positions.iter().enumerate() {
            let start = p * shape.query_features as usize;
            let end = start + shape.query_features as usize;
            let expected = reference(&query[start..end], &kv[p], &sinks, *position, shape);
            for (i, (actual, reference)) in output[start..end].iter().zip(expected).enumerate() {
                assert!(
                    actual.is_finite(),
                    "nonfinite XQA output at participant={p} element={i}"
                );
                assert!((actual.to_f64() - reference).abs() <= 0.001 * reference.abs().max(1.0),
                    "XQA differs from F64: participant={p} position={position} window={window} element={i} actual={actual} reference={reference}");
            }
        }
        self.query.assert_unchanged(stream);
        self.sinks.assert_unchanged(stream);
        self.sinks_f32.assert_unchanged(stream);
        let _ = self.semaphores.read(stream);
        let scratch = self.scratch.read(stream);
        let end = self.scratch_offset + self.prepared.plan().scratch_bytes as usize;
        assert!(scratch[..self.scratch_offset].iter().all(|&v| v == 0xCD));
        assert!(scratch[end..].iter().all(|&v| v == 0xCD));
    }
}

#[test]
#[ignore = "requires SM120 and its locked XQA artifact"]
fn xqa_gpt_oss_decode_preserves_sinks_pages_windows_and_graph_replay_on_cuda() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    for window in [0, 128] {
        for batch in [1, 3, 17] {
            let contexts = [4095, 0, 128, 31, 32, 126, 127, 257, 593, 1023, 1, 2, 3];
            let positions = (0..batch)
                .map(|i| contexts[i % contexts.len()])
                .collect::<Vec<_>>();
            let fixture = Fixture::new(&stream, &positions, window);
            fixture.launch(&stream);
            fixture.check(&stream, &positions, window);
            if fixture.prepared.plan().subsequences == 1 {
                fixture.semaphores.assert_unchanged(&stream);
            }
            let graph = captured::Captured::new(&stream, || fixture.launch(&stream));
            for _ in 0..3 {
                graph.launch();
                fixture.check(&stream, &positions, window);
                if fixture.prepared.plan().subsequences == 1 {
                    fixture.semaphores.assert_unchanged(&stream);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires SM120 and its locked XQA artifact; isolates single-split graph reset cost"]
fn xqa_gpt_oss_single_split_graph_reset_microbench() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::SANDWICH_NORM.to_owned()))
        .unwrap();
    let cast_sinks = module.load_function("cast_f16_to_f32_slice").unwrap();
    const BATCH: usize = 16;
    const LAYERS: usize = 24;
    const ITERATIONS: usize = 32;
    let positions = vec![593; BATCH];
    let fixtures = [
        Fixture::new(&stream, &positions, 128),
        Fixture::new(&stream, &positions, 0),
    ];
    for fixture in &fixtures {
        assert_eq!(
            fixture.prepared.plan().subsequences,
            1,
            "paired reset diagnostic requires single-split geometry"
        );
    }
    let sources = fixtures
        .each_ref()
        .map(|fixture| fixture.sinks.pointer(&stream));
    // Both arms include identical sink casts and native attention. The second
    // reinstates precisely the old single-split semaphore reset per layer.
    let graphs = [false, true].map(|extra_reset| {
        captured::Captured::new(&stream, || {
            for layer in 0..LAYERS {
                let index = layer % fixtures.len();
                let fixture = &fixtures[index];
                let source = sources[index];
                let destination = fixture.bindings.sinks_f32.address;
                let zero = 0_i32;
                let count = 64_i32;
                let mut launch = stream.launch_builder(&cast_sinks);
                launch
                    .arg(&source)
                    .arg(&destination)
                    .arg(&zero)
                    .arg(&zero)
                    .arg(&count);
                // All buffers remain retained through both graph lifetimes.
                unsafe { launch.launch(LaunchConfig::for_num_elems(64)) }.unwrap();
                if extra_reset {
                    unsafe {
                        cudarc::driver::result::memset_d8_async(
                            fixture.bindings.semaphores.address,
                            0,
                            fixture.prepared.plan().semaphore_bytes as usize,
                            stream.cu_stream(),
                        )
                    }
                    .unwrap();
                }
                fixture.launch(&stream);
            }
        })
    });
    // Upload happened during Captured::new; warm both paths before timing.
    for _ in 0..3 {
        for graph in &graphs {
            graph.launch();
            stream.synchronize().unwrap();
        }
    }
    for graph in &graphs {
        graph.launch();
        for (fixture, window) in fixtures.iter().zip([128, 0]) {
            fixture.check(&stream, &positions, window);
        }
    }
    let start = context
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = context
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    for round in 0..5 {
        for (order, index) in [0, 1, 1, 0].into_iter().enumerate() {
            let mut host_ns = 0_u128;
            let mut gpu_ms = 0.0_f64;
            for _ in 0..ITERATIONS {
                // Avoid enqueue backpressure. Events, synchronization and
                // numerical inspection are outside cuGraphLaunch wall timing.
                stream.synchronize().unwrap();
                start.record(&stream).unwrap();
                let begin = Instant::now();
                graphs[index].launch();
                host_ns += begin.elapsed().as_nanos();
                end.record(&stream).unwrap();
                end.synchronize().unwrap();
                gpu_ms += f64::from(start.elapsed_ms(&end).unwrap());
            }
            println!(
                "{}",
                serde_json::json!({
                    "benchmark": "gpt_oss_xqa_single_split_graph_reset",
                    "arm": if index == 0 { "native_no_reset" } else { "external_reset_then_native" },
                    "participants": BATCH, "context_tokens": 594,
                    "alternating_full_local_layers": LAYERS, "subsequences": 1,
                    "graph_nodes": graphs[index].node_count,
                    "round": round, "abba_index": order, "iterations": ITERATIONS,
                    "graph_launch_host_us": host_ns as f64 / 1000.0 / ITERATIONS as f64,
                    "cuda_event_interval_us": gpu_ms * 1000.0 / ITERATIONS as f64,
                    "scope": "paired resident attention graph; host timing wraps only graph launch; CUDA event interval may include host enqueue gaps, not device busy time; not serving performance",
                })
            );
        }
    }
    for (fixture, window) in fixtures.iter().zip([128, 0]) {
        fixture.check(&stream, &positions, window);
    }
}

#[test]
#[ignore = "requires SM120 and its locked XQA artifact; emits paired CUDA-event diagnostics"]
fn xqa_gpt_oss_attention_microbench() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::GPT_OSS_ATTENTION.to_owned()))
        .unwrap();
    let paged = module
        .load_function("gpt_oss_paged_attention_sink_packed_decode_multiwarp_paged_f16")
        .unwrap();
    const ITERATIONS: usize = 64;
    for batch in [8, 16] {
        for context_tokens in [594, 1024] {
            for window in [0, 128] {
                let positions = vec![context_tokens - 1; batch];
                let fixture = Fixture::new(&stream, &positions, window);
                let shape = attention_shape(window);
                let paged_output =
                    Guarded::new(&stream, &vec![f16::ZERO; batch * 4096], f16::from_f32(85.0));
                for round in 0..6 {
                    for arm in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                        stream.synchronize().unwrap();
                        let (wall_ns, gpu_ns) = captured::measure(&stream, || {
                            for _ in 0..ITERATIONS {
                                if arm == 0 {
                                    attention(
                                        &stream,
                                        &paged,
                                        true,
                                        128,
                                        1,
                                        shape.cuda_shape().unwrap(),
                                        &fixture.query,
                                        &fixture.sinks,
                                        &paged_output,
                                        &fixture.kv,
                                        0,
                                        batch,
                                    );
                                } else {
                                    fixture.launch(&stream);
                                }
                            }
                        });
                        if round > 0 {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "benchmark": "gpt_oss_xqa_attention", "kernel": if arm == 0 { "paged" } else { "xqa" },
                                    "participants": batch, "context_tokens": context_tokens, "sliding_window_tokens": window,
                                    "round": round - 1, "iterations": ITERATIONS,
                                    "gpu_us": gpu_ns / 1000.0 / ITERATIONS as f64, "wall_us": wall_ns / 1000.0 / ITERATIONS as f64,
                                    "scope": "resident single-layer KV; native includes semaphore reset; not serving performance",
                                })
                            );
                        }
                    }
                }
                fixture.check(&stream, &positions, window);
                for (vendor, scalar) in fixture
                    .output
                    .read(&stream)
                    .iter()
                    .zip(paged_output.read(&stream))
                {
                    assert!(scalar.is_finite());
                    assert!((vendor.to_f32() - scalar.to_f32()).abs() <= 0.002);
                }
            }
        }
    }
}
