use super::*;

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
    for window in [0, 128] {
        let shape = attention_shape(window);
        for range in [29..34, 123..134] {
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
            for (function, output) in [(&original, &old_output), (&paged, &new_output)] {
                attention(
                    &stream,
                    function,
                    false,
                    32,
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
                for (dim, (actual, expected)) in new[start..end].iter().zip(expected).enumerate() {
                    assert!(
                        actual.is_finite() && (actual.to_f64() - expected).abs()
                            <= 0.001 * expected.abs().max(1.0),
                        "prefill range={range:?}, window={window}, token={token}, dim={dim}: {actual:?} vs {expected}",
                    );
                }
            }
            query.assert_unchanged(&stream);
            sinks.assert_unchanged(&stream);
            assert_eq!(kv.read(&stream), before, "attention modified KV");
        }
    }
}
