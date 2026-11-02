//! Real CUDA checks for packed decode over independent, non-contiguous KV pages.
use super::super::test_support::Guarded;
use super::*;
use cudarc::driver::CudaContext;
use half::f16;

mod microbench;
mod prefill;
#[cfg(ferrum_native_flashinfer_xqa)]
mod xqa;

fn attention_shape(sliding_window_tokens: u64) -> GptOssAttentionShape {
    GptOssAttentionShape {
        hidden_size: 2880,
        query_heads: 64,
        kv_heads: 8,
        head_dim: 64,
        query_features: 4096,
        kv_features: 512,
        rope_dim: 64,
        maximum_context_tokens: 4096,
        rope_theta: 150_000.0,
        yarn_factor: 32.0,
        yarn_original_context_tokens: 4096,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        sliding_window_tokens,
        epsilon: 1.0e-5,
        layer_index: if sliding_window_tokens == 0 { 1 } else { 0 },
    }
}

fn sample(index: usize, salt: usize) -> f16 {
    f16::from_f32((((index * 13 + salt * 7) % 41) as f32 - 20.0) * 0.03125)
}

struct KvFixture {
    pages: Vec<Vec<Guarded<f16>>>,
    bindings: Guarded<u64>,
    stride: u64,
}

impl KvFixture {
    fn new(stream: &Arc<CudaStream>, positions: &[usize], shape: GptOssAttentionShape) -> Self {
        let ranges = positions.iter().map(|&p| p..p + 1).collect::<Vec<_>>();
        Self::with_ranges(stream, &ranges, shape)
    }

    fn with_ranges(
        stream: &Arc<CudaStream>,
        ranges: &[std::ops::Range<usize>],
        shape: GptOssAttentionShape,
    ) -> Self {
        let page_elements = VNEXT_KV_PAGE_BYTES as usize / 2;
        let layout = BindingLayout::new(shape, ranges.len()).unwrap();
        let mut words = vec![0_u64; layout.required_bytes as usize / 8];
        let pages = ranges
            .iter()
            .enumerate()
            .map(|(participant, range)| {
                let page_count =
                    (range.end * 2 * shape.kv_features as usize).div_ceil(page_elements);
                let pages: Vec<_> = (0..page_count)
                    .map(|page| {
                        let values = (0..page_elements)
                            .map(|i| sample(page * page_elements + i, participant + 3))
                            .collect::<Vec<_>>();
                        Guarded::new(stream, &values, f16::from_f32(81.0))
                    })
                    .collect();
                let start = layout.binding_offset(participant).unwrap() as usize / 8;
                // Native ABI has four i32 control words followed by u64 pointers.
                words[start] = page_count as u64 | ((range.start as u64) << 32);
                words[start + 1] = range.len() as u64 | ((range.end as u64) << 32);
                // Reverse physical page order to exercise actual page-table addressing.
                for (i, page) in pages.iter().rev().enumerate() {
                    words[start + 2 + i] = page.pointer(stream);
                }
                pages
            })
            .collect();
        Self {
            pages,
            bindings: Guarded::new(stream, &words, 0xDEAD_BEEF_u64),
            stride: layout.slot_bytes,
        }
    }

    fn read(&self, stream: &Arc<CudaStream>) -> Vec<Vec<f16>> {
        self.bindings.assert_unchanged(stream);
        self.pages
            .iter()
            .map(|pages| {
                pages
                    .iter()
                    .rev()
                    .flat_map(|page| page.read(stream))
                    .collect()
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    packed: bool,
    shape: CudaGptOssAttentionShape,
    raw: &[Guarded<f16>; 3],
    bias: &[Guarded<f16>; 3],
    query: &Guarded<f16>,
    kv: &KvFixture,
    participant: usize,
    participants: usize,
) {
    let offset = if packed { 0 } else { participant as u64 };
    let control = kv.bindings.pointer(stream) + offset * kv.stride;
    let pointers = [
        raw[0].pointer(stream) + offset * shape.query_features as u64 * 2,
        raw[1].pointer(stream) + offset * shape.kv_features as u64 * 2,
        raw[2].pointer(stream) + offset * shape.kv_features as u64 * 2,
        bias[0].pointer(stream),
        bias[1].pointer(stream),
        bias[2].pointer(stream),
        query.pointer(stream) + offset * shape.query_features as u64 * 2,
        control,
    ];
    let table = control + BINDING_CONTROL_BYTES;
    let count = participants as i32;
    let mut launch = stream.launch_builder(function);
    for pointer in &pointers {
        launch.arg(pointer);
    }
    if packed {
        launch.arg(&kv.stride).arg(&count);
    } else {
        launch.arg(&table);
    }
    let dims = [
        (VNEXT_KV_PAGE_BYTES / 2) as i32,
        shape.query_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.rope_dim,
        shape.query_features,
        shape.kv_features,
    ];
    for dim in &dims {
        launch.arg(dim);
    }
    let original_context = shape.yarn_original_context_tokens as f32;
    launch
        .arg(&shape.rope_theta)
        .arg(&shape.yarn_factor)
        .arg(&original_context)
        .arg(&shape.yarn_beta_fast)
        .arg(&shape.yarn_beta_slow);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (
                if packed { participants as u32 } else { 1 },
                (shape.query_heads + 2 * shape.kv_heads) as u32,
                1,
            ),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
}

#[allow(clippy::too_many_arguments)]
fn attention(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    packed: bool,
    block_threads: u32,
    query_tokens: u32,
    shape: CudaGptOssAttentionShape,
    query: &Guarded<f16>,
    sinks: &Guarded<f16>,
    output: &Guarded<f16>,
    kv: &KvFixture,
    participant: usize,
    participants: usize,
) {
    let offset = if packed { 0 } else { participant as u64 };
    let control = kv.bindings.pointer(stream) + offset * kv.stride;
    let pointers = [
        query.pointer(stream) + offset * shape.query_features as u64 * 2,
        sinks.pointer(stream),
        control,
    ];
    let table = control + BINDING_CONTROL_BYTES;
    let destination = output.pointer(stream) + offset * shape.query_features as u64 * 2;
    let count = participants as i32;
    let mut launch = stream.launch_builder(function);
    for pointer in &pointers {
        launch.arg(pointer);
    }
    if packed {
        launch.arg(&kv.stride).arg(&count);
    } else {
        launch.arg(&table);
    }
    launch.arg(&destination);
    let dims = [
        (VNEXT_KV_PAGE_BYTES / 2) as i32,
        shape.query_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.sliding_window_tokens,
    ];
    for dim in &dims {
        launch.arg(dim);
    }
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (
                if packed {
                    participants as u32
                } else {
                    query_tokens
                },
                shape.query_heads as u32,
                1,
            ),
            block_dim: (block_threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
}

fn reference(
    query: &[f16],
    kv: &[f16],
    sinks: &[f16],
    position: usize,
    shape: GptOssAttentionShape,
) -> Vec<f64> {
    let width = shape.head_dim as usize;
    let kv_features = shape.kv_features as usize;
    let first = if shape.sliding_window_tokens == 0 {
        0
    } else {
        (position + 1).saturating_sub(shape.sliding_window_tokens as usize)
    };
    let mut output = vec![0.0; shape.query_features as usize];
    for head in 0..shape.query_heads as usize {
        let kv_head = head / (shape.query_heads / shape.kv_heads) as usize;
        let scores = (first..=position)
            .map(|token| {
                (0..width)
                    .map(|dim| {
                        query[head * width + dim].to_f64()
                            * kv[token * 2 * kv_features + kv_head * width + dim].to_f64()
                    })
                    .sum::<f64>()
                    / (width as f64).sqrt()
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(sinks[head].to_f64(), f64::max);
        let denominator = (sinks[head].to_f64() - maximum).exp()
            + scores
                .iter()
                .map(|score| (score - maximum).exp())
                .sum::<f64>();
        for (token, score) in (first..=position).zip(scores) {
            for dim in 0..width {
                output[head * width + dim] += (score - maximum).exp() / denominator
                    * kv[(token * 2 + 1) * kv_features + kv_head * width + dim].to_f64();
            }
        }
    }
    output
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn packed_gpt_oss_decode_preserves_yarn_sink_windows_and_independent_pages_on_cuda() {
    let context = CudaContext::new(0).expect("GPT-OSS numerical conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::GPT_OSS_ATTENTION.to_owned()))
        .unwrap();
    let scalar_prepare = module.load_function(PREPARE_FUNCTION).unwrap();
    let scalar_attention = module
        .load_function("gpt_oss_paged_attention_sink_f16")
        .unwrap();
    let packed_prepare = module
        .load_function("gpt_oss_prepare_qkv_yarn_packed_decode_f16")
        .unwrap();
    let packed_attention = module
        .load_function("gpt_oss_paged_attention_sink_packed_decode_f16")
        .unwrap();
    let multiwarp_attention = module
        .load_function("gpt_oss_paged_attention_sink_packed_decode_multiwarp_f16")
        .unwrap();
    let paged_attention = module
        .load_function("gpt_oss_paged_attention_sink_packed_decode_multiwarp_paged_f16")
        .unwrap();
    for sliding_window_tokens in [0, 128] {
        let shape = attention_shape(sliding_window_tokens);
        let cuda = shape.cuda_shape().unwrap();
        for participants in [1, 3, 17] {
            let context_positions = [0, 1, 2, 3, 30, 31, 32, 126, 127, 128, 257, 593, 1023, 4095];
            let positions = (0..participants)
                .map(|i| context_positions[i % context_positions.len()])
                .collect::<Vec<_>>();
            let scalar_kv = KvFixture::new(&stream, &positions, shape);
            let packed_kv = KvFixture::new(&stream, &positions, shape);
            let before = scalar_kv.read(&stream);
            let raw = std::array::from_fn(|i| {
                let width = if i == 0 {
                    shape.query_features
                } else {
                    shape.kv_features
                } as usize;
                Guarded::new(
                    &stream,
                    &(0..participants * width)
                        .map(|j| sample(j, 11 + i))
                        .collect::<Vec<_>>(),
                    f16::from_f32(82.0),
                )
            });
            let bias = std::array::from_fn(|i| {
                let width = if i == 0 {
                    shape.query_features
                } else {
                    shape.kv_features
                } as usize;
                Guarded::new(
                    &stream,
                    &(0..width).map(|j| sample(j, 17 + i)).collect::<Vec<_>>(),
                    f16::from_f32(83.0),
                )
            });
            let sinks_host = (0..shape.query_heads as usize)
                .map(|i| f16::from_f32(if i % 2 == 0 { 6.0 } else { -3.0 }))
                .collect::<Vec<_>>();
            let sinks = Guarded::new(&stream, &sinks_host, f16::from_f32(84.0));
            let blank = vec![f16::ZERO; participants * shape.query_features as usize];
            let scalar_q = Guarded::new(&stream, &blank, f16::from_f32(85.0));
            let packed_q = Guarded::new(&stream, &blank, f16::from_f32(86.0));
            let scalar_output = Guarded::new(&stream, &blank, f16::from_f32(87.0));
            let packed_output = Guarded::new(&stream, &blank, f16::from_f32(88.0));
            let multiwarp_output = Guarded::new(&stream, &blank, f16::from_f32(89.0));
            let paged_output = Guarded::new(&stream, &blank, f16::from_f32(90.0));
            for p in 0..participants {
                prepare(
                    &stream,
                    &scalar_prepare,
                    false,
                    cuda,
                    &raw,
                    &bias,
                    &scalar_q,
                    &scalar_kv,
                    p,
                    participants,
                );
                attention(
                    &stream,
                    &scalar_attention,
                    false,
                    32,
                    1,
                    cuda,
                    &scalar_q,
                    &sinks,
                    &scalar_output,
                    &scalar_kv,
                    p,
                    participants,
                );
            }
            prepare(
                &stream,
                &packed_prepare,
                true,
                cuda,
                &raw,
                &bias,
                &packed_q,
                &packed_kv,
                0,
                participants,
            );
            attention(
                &stream,
                &packed_attention,
                true,
                32,
                1,
                cuda,
                &packed_q,
                &sinks,
                &packed_output,
                &packed_kv,
                0,
                participants,
            );
            attention(
                &stream,
                &multiwarp_attention,
                true,
                128,
                1,
                cuda,
                &packed_q,
                &sinks,
                &multiwarp_output,
                &packed_kv,
                0,
                participants,
            );
            attention(
                &stream,
                &paged_attention,
                true,
                128,
                1,
                cuda,
                &packed_q,
                &sinks,
                &paged_output,
                &packed_kv,
                0,
                participants,
            );
            let queries = packed_q.read(&stream);
            let values = packed_output.read(&stream);
            let multiwarp_values = multiwarp_output.read(&stream);
            let paged_values = paged_output.read(&stream);
            assert!(
                multiwarp_values
                    .iter()
                    .zip(&paged_values)
                    .all(|(old, new)| old.to_bits() == new.to_bits()),
                "page addressing changed values, batch={participants}, window={sliding_window_tokens}"
            );
            let kv = packed_kv.read(&stream);
            assert_eq!(
                queries,
                scalar_q.read(&stream),
                "YaRN batch={participants}, window={sliding_window_tokens}"
            );
            assert_eq!(kv, scalar_kv.read(&stream), "KV writes differ");
            assert_eq!(
                values,
                scalar_output.read(&stream),
                "attention batch={participants}, window={sliding_window_tokens}"
            );
            for (p, &position) in positions.iter().enumerate() {
                let written = position * 2 * shape.kv_features as usize
                    ..(position + 1) * 2 * shape.kv_features as usize;
                for (i, (actual, old)) in kv[p].iter().zip(&before[p]).enumerate() {
                    if !written.contains(&i) {
                        assert_eq!(actual, old, "unowned KV p={p} element={i}");
                    }
                }
                let start = p * shape.query_features as usize;
                let end = start + shape.query_features as usize;
                let expected =
                    reference(&queries[start..end], &kv[p], &sinks_host, position, shape);
                // Parallel reduction changes addition order. Both kernels must satisfy
                // the independent reference; only the original packed path is bitwise.
                for (kernel, actual_values) in [
                    ("packed", &values),
                    ("multiwarp", &multiwarp_values),
                    ("paged", &paged_values),
                ] {
                    for (i, (actual, expected)) in
                        actual_values[start..end].iter().zip(&expected).enumerate()
                    {
                        let bound = 0.001 * expected.abs().max(1.0);
                        assert!(
                            actual.is_finite() && (actual.to_f64() - expected).abs() <= bound,
                            "F64 {kernel} attention p={p} context={} window={sliding_window_tokens} element={i}: {actual:?} vs {expected}",
                            position + 1,
                        );
                    }
                }
            }
            for input in raw.iter().chain(&bias).chain(std::iter::once(&sinks)) {
                input.assert_unchanged(&stream);
            }
        }
    }
}
