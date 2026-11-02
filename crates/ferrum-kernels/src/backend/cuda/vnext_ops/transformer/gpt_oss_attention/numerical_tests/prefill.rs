use super::*;
use cudarc::driver::sys::CUevent_flags;
use std::time::Instant;

#[test]
#[ignore = "requires an actual CUDA device"]
fn paged_gpt_oss_prefill_preserves_causality_and_windows_on_cuda() {
    let context = CudaContext::new(0).expect("prefill numerical conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::GPT_OSS_ATTENTION.to_owned()))
        .unwrap();
    let original = module
        .load_function("gpt_oss_paged_attention_sink_f16")
        .unwrap();
    let paged = module
        .load_function("gpt_oss_paged_attention_sink_paged_f16")
        .unwrap();
    let multiwarp = module
        .load_function("gpt_oss_paged_attention_sink_multiwarp_paged_f16")
        .unwrap();
    for window in [0, 128] {
        let shape = attention_shape(window);
        for range in [0..3, 29..34, 123..134, 589..598, 4093..4096] {
            let kv = KvFixture::with_ranges(&stream, std::slice::from_ref(&range), shape);
            let before = kv.read(&stream);
            let values = (0..range.len() * shape.query_features as usize)
                .map(|i| sample(i, 13))
                .collect::<Vec<_>>();
            let query = Guarded::new(&stream, &values, f16::from_f32(81.0));
            let sink_values = (0..shape.query_heads as usize)
                .map(|i| f16::from_f32(if i % 2 == 0 { 6.0 } else { -3.0 }))
                .collect::<Vec<_>>();
            let sinks = Guarded::new(&stream, &sink_values, f16::from_f32(82.0));
            let blank = vec![f16::ZERO; values.len()];
            let old_output = Guarded::new(&stream, &blank, f16::from_f32(83.0));
            let new_output = Guarded::new(&stream, &blank, f16::from_f32(84.0));
            let multiwarp_output = Guarded::new(&stream, &blank, f16::from_f32(85.0));
            for (function, output, threads) in [
                (&original, &old_output, 32),
                (&paged, &new_output, 32),
                (&multiwarp, &multiwarp_output, 128),
            ] {
                attention(
                    &stream,
                    function,
                    false,
                    threads,
                    range.len() as u32,
                    shape.cuda_shape().unwrap(),
                    &query,
                    &sinks,
                    output,
                    &kv,
                    0,
                    1,
                );
            }
            let old = old_output.read(&stream);
            let new = new_output.read(&stream);
            let multiwarp = multiwarp_output.read(&stream);
            assert!(old
                .iter()
                .zip(&new)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            for (token, position) in range.clone().enumerate() {
                let start = token * shape.query_features as usize;
                let end = start + shape.query_features as usize;
                let expected = reference(
                    &values[start..end],
                    &before[0],
                    &sink_values,
                    position,
                    shape,
                );
                for (kernel, output) in [("paged", &new), ("multiwarp", &multiwarp)] {
                    for (dim, (actual, expected)) in
                        output[start..end].iter().zip(&expected).enumerate()
                    {
                        assert!(
                            actual.is_finite() && (actual.to_f64() - expected).abs()
                                <= 0.001 * expected.abs().max(1.0),
                            "{kernel} prefill range={range:?}, window={window}, token={token}, dim={dim}: {actual:?} vs {expected}",
                        );
                    }
                }
            }
            query.assert_unchanged(&stream);
            sinks.assert_unchanged(&stream);
            assert_eq!(kv.read(&stream), before, "attention modified KV");
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device; emits paired CUDA-event prefill diagnostics"]
fn gpt_oss_prefill_attention_microbench() {
    let context = CudaContext::new(0).expect("prefill microbenchmark requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::GPT_OSS_ATTENTION.to_owned()))
        .unwrap();
    let functions = [
        ("scalar", "gpt_oss_paged_attention_sink_paged_f16", 32),
        (
            "multiwarp",
            "gpt_oss_paged_attention_sink_multiwarp_paged_f16",
            128,
        ),
    ]
    .map(|(name, symbol, threads)| (name, module.load_function(symbol).unwrap(), threads));
    const ITERATIONS: usize = 16;
    for window in [0, 128] {
        let shape = attention_shape(window);
        for range in [0..128, 0..593, 269..593, 0..1024] {
            let kv = KvFixture::with_ranges(&stream, std::slice::from_ref(&range), shape);
            let before = kv.read(&stream);
            let values = (0..range.len() * shape.query_features as usize)
                .map(|i| sample(i, 13))
                .collect::<Vec<_>>();
            let query = Guarded::new(&stream, &values, f16::from_f32(81.0));
            let sinks = Guarded::new(
                &stream,
                &(0..shape.query_heads as usize)
                    .map(|i| f16::from_f32(if i % 2 == 0 { 6.0 } else { -3.0 }))
                    .collect::<Vec<_>>(),
                f16::from_f32(82.0),
            );
            let blank = vec![f16::ZERO; values.len()];
            let outputs = [
                Guarded::new(&stream, &blank, f16::from_f32(83.0)),
                Guarded::new(&stream, &blank, f16::from_f32(84.0)),
            ];
            // Alternate order; omit a complete warm-up of both paths.
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
                            false,
                            functions[index].2,
                            range.len() as u32,
                            shape.cuda_shape().unwrap(),
                            &query,
                            &sinks,
                            &outputs[index],
                            &kv,
                            0,
                            1,
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
                                "benchmark": "gpt_oss_prefill_attention",
                                "kernel": functions[index].0,
                                "source_start": range.start, "source_end": range.end,
                                "query_tokens": range.len(), "sliding_window_tokens": window,
                                "round": round - 1, "iterations": ITERATIONS,
                                "gpu_us": gpu_us, "wall_us": wall_us,
                                "scope": "resident single-layer KV; not serving performance",
                            })
                        );
                    }
                }
            }
            for (scalar, multiwarp) in outputs[0]
                .read(&stream)
                .iter()
                .zip(outputs[1].read(&stream))
            {
                assert!(scalar.is_finite() && multiwarp.is_finite());
                assert!((scalar.to_f32() - multiwarp.to_f32()).abs() <= 0.002);
            }
            query.assert_unchanged(&stream);
            sinks.assert_unchanged(&stream);
            assert_eq!(kv.read(&stream), before, "attention modified KV");
        }
    }
}
