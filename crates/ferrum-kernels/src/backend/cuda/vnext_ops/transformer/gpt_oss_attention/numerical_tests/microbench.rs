//! Diagnostic single-layer timing; resident KV can fit in cache. Serving
//! throughput and latency must still be measured over the complete model.
use super::*;
use cudarc::driver::sys::CUevent_flags;
use std::time::Instant;

#[test]
#[ignore = "requires an actual CUDA device; emits CUDA-event attention microbenchmark JSON"]
fn packed_gpt_oss_attention_microbench() {
    let context = CudaContext::new(0).expect("attention microbenchmark requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::GPT_OSS_ATTENTION.to_owned()))
        .unwrap();
    let functions = [
        (
            "multiwarp",
            "gpt_oss_paged_attention_sink_packed_decode_multiwarp_f16",
        ),
        (
            "paged",
            "gpt_oss_paged_attention_sink_packed_decode_multiwarp_paged_f16",
        ),
    ]
    .map(|(name, symbol)| (name, module.load_function(symbol).unwrap()));
    const ITERATIONS: usize = 64;
    for participants in [8, 16] {
        for context_tokens in [128, 594, 1024] {
            for window in [0, 128] {
                let shape = attention_shape(window);
                let cuda = shape.cuda_shape().unwrap();
                let positions = vec![context_tokens - 1; participants];
                let kv = KvFixture::new(&stream, &positions, shape);
                let queries = (0..participants * shape.query_features as usize)
                    .map(|i| sample(i, 11))
                    .collect::<Vec<_>>();
                let query = Guarded::new(&stream, &queries, f16::from_f32(81.0));
                let sinks = Guarded::new(
                    &stream,
                    &(0..shape.query_heads as usize)
                        .map(|i| f16::from_f32(if i % 2 == 0 { 6.0 } else { -3.0 }))
                        .collect::<Vec<_>>(),
                    f16::from_f32(82.0),
                );
                let blank = vec![f16::ZERO; queries.len()];
                let outputs = [
                    Guarded::new(&stream, &blank, f16::from_f32(83.0)),
                    Guarded::new(&stream, &blank, f16::from_f32(84.0)),
                ];
                // Alternate order; the first round warms both paths and is omitted.
                // No speed threshold is a correctness gate.
                for round in 0..6 {
                    for index in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                        stream.synchronize().unwrap();
                        let wall = Instant::now();
                        let start = stream
                            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap();
                        for _ in 0..ITERATIONS {
                            attention(
                                &stream,
                                &functions[index].1,
                                true,
                                128,
                                1,
                                cuda,
                                &query,
                                &sinks,
                                &outputs[index],
                                &kv,
                                0,
                                participants,
                            );
                        }
                        let end = stream
                            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap();
                        end.synchronize().unwrap();
                        let wall_us = wall.elapsed().as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
                        let gpu_us =
                            f64::from(start.elapsed_ms(&end).unwrap()) * 1000.0 / ITERATIONS as f64;
                        if round > 0 {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "benchmark": "gpt_oss_attention",
                                    "kernel": functions[index].0,
                                    "participants": participants,
                                    "context_tokens": context_tokens,
                                    "sliding_window_tokens": window,
                                    "round": round - 1,
                                    "iterations": ITERATIONS,
                                    "gpu_us": gpu_us,
                                    "wall_us": wall_us,
                                    "scope": "resident single-layer KV; not serving performance",
                                })
                            );
                        }
                    }
                }
                for (old, new) in outputs[0]
                    .read(&stream)
                    .iter()
                    .zip(outputs[1].read(&stream))
                {
                    assert!(old.is_finite() && new.is_finite());
                    assert_eq!(old.to_bits(), new.to_bits());
                }
                query.assert_unchanged(&stream);
                sinks.assert_unchanged(&stream);
                // Also checks binding and allocation canaries after repeated launches.
                let _ = kv.read(&stream);
            }
        }
    }
}
