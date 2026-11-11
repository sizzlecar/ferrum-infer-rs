//! Addressed F16 KV and explicit positions through the production fallback
//! launchers. Executor cancellation remains covered by the existing cohort tests.
use super::*;

trait AttentionScalar: cudarc::driver::DeviceRepr + Copy + PartialEq + std::fmt::Debug {
    const BF16: bool;
    const TOLERANCE: f64;
    const NAN: Self;
    fn from_f32(x: f32) -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
}
macro_rules! scalar {
    ($t:ty, $bf16:expr, $tol:expr) => {
        impl AttentionScalar for $t {
            const BF16: bool = $bf16;
            const TOLERANCE: f64 = $tol;
            const NAN: Self = Self::NAN;
            fn from_f32(x: f32) -> Self {
                Self::from_f32(x)
            }
            fn from_f64(x: f64) -> Self {
                Self::from_f64(x)
            }
            fn to_f64(self) -> f64 {
                self.to_f64()
            }
        }
    };
}
scalar!(f16, false, 0.001);
scalar!(half::bf16, true, 0.008);
fn sample_scalar<T: AttentionScalar>(index: usize, salt: usize) -> T {
    T::from_f32(sample(index, salt).to_f32())
}
fn close_scalar<T: AttentionScalar>(actual: T, expected: f64, stage: &str, index: usize) {
    let value = actual.to_f64();
    let bound = expected.abs().max(1.0) * T::TOLERANCE + 1.0e-5;
    assert!(
        value.is_finite() && (value - expected).abs() <= bound,
        "{stage}[{index}]: actual={value}, F64 reference={expected}, bound={bound}"
    );
}

fn addressed_index(
    shape: CausalAttentionShape,
    token: usize,
    value: bool,
    head: usize,
    dim: usize,
) -> usize {
    let width = shape.head_dim as usize;
    let heads = shape.key_value_heads as usize;
    let block = token / 16 * 2 * heads * width * 16;
    if value {
        block + heads * width * 16 + head * width * 16 + dim * 16 + token % 16
    } else {
        block + head * width * 16 + dim / 8 * 16 * 8 + token % 16 * 8 + dim % 8
    }
}

fn explicit_reference<T: AttentionScalar>(
    raw: &[T],
    norm: &[T],
    cos: &[T],
    sin: &[T],
    epsilon: f32,
) -> Vec<f64> {
    let mean = raw.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() / raw.len() as f64;
    let mut result = raw
        .iter()
        .zip(norm)
        .map(|(v, w)| v.to_f64() * w.to_f64() / (mean + f64::from(epsilon)).sqrt())
        .collect::<Vec<_>>();
    for pair in 0..cos.len() {
        let (low, high) = (result[pair], result[pair + cos.len()]);
        result[pair] = low * cos[pair].to_f64() - high * sin[pair].to_f64();
        result[pair + cos.len()] = high * cos[pair].to_f64() + low * sin[pair].to_f64();
    }
    result
}

fn exercise_explicit<T: AttentionScalar>(
    stream: &Arc<CudaStream>,
    functions: &CausalAttentionFunctions,
    width: u64,
    position_variant: usize,
    batches: &[usize],
) -> (Vec<T>, Vec<T>) {
    let mut shape = shape(width, false);
    shape.query_heads = 4;
    shape.query_features = 4 * width;
    shape.query_projection_features = 4 * width;
    shape.maximum_context_tokens = 2048;
    assert!(matches!(
        shape.kv_layout().unwrap(),
        CausalKvLayout::VllmBlocks16 { .. }
    ));
    let width = width as usize;
    let qheads = shape.query_heads as usize;
    let kvheads = shape.key_value_heads as usize;
    let half_rope = shape.rope_dim as usize / 2;
    // The next row crosses both a 16-token block and a physical page boundary
    // for every tested head size. Three rows also exercise unequal KV partitions.
    let prefix = 1023;
    let end = prefix + batches.iter().sum::<usize>();
    let page_elements = VNEXT_KV_PAGE_BYTES as usize / 2;
    let page_count =
        (shape.physical_state_bytes(end as u64).unwrap() / VNEXT_KV_PAGE_BYTES) as usize;
    let unused = T::from_f32(-73.0);
    let mut previous = vec![unused; page_count * page_elements];
    for token in 0..prefix {
        for head in 0..kvheads {
            for dim in 0..width {
                for (value, salt) in [(false, 11), (true, 17)] {
                    let index = addressed_index(shape, token, value, head, dim);
                    previous[index] = sample_scalar::<T>(index, salt);
                }
            }
        }
    }
    let pages = previous
        .chunks_exact(page_elements)
        .rev()
        .map(|values| Guarded::new(stream, values, T::from_f32(81.0)))
        .collect::<Vec<_>>();
    let idle = Guarded::new(stream, &vec![unused; page_elements], T::from_f32(82.0));
    let addresses = pages
        .iter()
        .rev()
        .map(|page| page.pointer(stream))
        .collect::<Vec<_>>();
    let block_addresses = binding_addresses(
        shape.kv_layout().unwrap(),
        i32::try_from(shape.table_entries(end as u64).unwrap()).unwrap(),
        &addresses,
    )
    .unwrap();
    let table = Guarded::new(stream, &block_addresses, 0xDEAD_u64);
    let norm = (0..width)
        .map(|i| T::from_f32(0.75 + sample_scalar::<T>(i, 5).to_f32() * 0.25))
        .collect::<Vec<_>>();
    let norm_gpu = Guarded::new(stream, &norm, T::from_f32(84.0));
    let cuda = shape.cuda_shape().unwrap();
    let mut position = prefix;
    let mut outputs = Vec::new();
    for &tokens in batches {
        let raw = |stride, salt| {
            (position * stride..(position + tokens) * stride)
                .map(|i| sample_scalar::<T>(i, salt))
                .collect::<Vec<_>>()
        };
        let q = raw(qheads * width, 3);
        let k = raw(kvheads * width, 7);
        let v = raw(kvheads * width, 13);
        let q_gpu = Guarded::new(stream, &q, T::from_f32(85.0));
        let k_gpu = Guarded::new(stream, &k, T::from_f32(86.0));
        let v_gpu = Guarded::new(stream, &v, T::from_f32(87.0));
        // These positions are deliberately independent of the physical KV
        // cursor and differ across otherwise identical request fixtures.
        let angles = (0..tokens * half_rope)
            .map(|i| {
                (position + i / half_rope) as f64 * (0.13 + position_variant as f64 * 0.07)
                    + (i % half_rope) as f64 * 0.03
            })
            .collect::<Vec<_>>();
        let cos = angles
            .iter()
            .map(|x| T::from_f64(x.cos()))
            .collect::<Vec<_>>();
        let sin = angles
            .iter()
            .map(|x| T::from_f64(x.sin()))
            .collect::<Vec<_>>();
        let cos_gpu = Guarded::new(stream, &cos, T::from_f32(90.0));
        let sin_gpu = Guarded::new(stream, &sin, T::from_f32(91.0));
        let length = position + tokens;
        let entries = shape.table_entries(length as u64).unwrap() as i32;
        let control = Guarded::new(
            stream,
            &[entries, position as i32, tokens as i32, length as i32, 0, 0],
            -111_i32,
        );
        let query = Guarded::new(stream, &vec![T::NAN; q.len()], T::from_f32(88.0));
        let output = Guarded::new(stream, &vec![T::NAN; q.len()], T::from_f32(89.0));
        let mut active = launch(shape, position, tokens);
        active.path = if T::BF16 {
            CausalAttentionKernelPath::select(
                AttentionExecutionPolicy::Portable,
                shape,
                tokens as u64,
                length as u64,
            )
            .unwrap()
        } else {
            CausalAttentionKernelPath::VllmAddressedFallback
        };
        active.table_entries_i32 = entries;
        active.replay_topology =
            CausalAttentionReplayTopology::new(shape, active.path, length as u64).unwrap();
        launch_prepare_with_rope(
            stream,
            &functions.prepare,
            q_gpu.pointer(stream),
            k_gpu.pointer(stream),
            v_gpu.pointer(stream),
            norm_gpu.pointer(stream),
            norm_gpu.pointer(stream),
            query.pointer(stream),
            control.pointer(stream),
            table.pointer(stream),
            active,
            cuda,
            1,
            None,
            Some([cos_gpu.pointer(stream), sin_gpu.pointer(stream)]),
        )
        .unwrap();
        let prepared_query = query.read(stream);
        let current = pages
            .iter()
            .rev()
            .flat_map(|page| page.read(stream))
            .collect::<Vec<_>>();
        let mut written = vec![false; current.len()];
        for token in 0..tokens {
            let cos = &cos[token * half_rope..][..half_rope];
            let sin = &sin[token * half_rope..][..half_rope];
            for head in 0..qheads {
                let base = (token * qheads + head) * width;
                for (dim, reference) in
                    explicit_reference(&q[base..base + width], &norm, cos, sin, shape.epsilon)
                        .into_iter()
                        .enumerate()
                {
                    close_scalar(
                        prepared_query[base + dim],
                        reference,
                        "explicit query",
                        base + dim,
                    );
                }
            }
            for head in 0..kvheads {
                let base = (token * kvheads + head) * width;
                let expected =
                    explicit_reference(&k[base..base + width], &norm, cos, sin, shape.epsilon);
                for dim in 0..width {
                    let ki = addressed_index(shape, position + token, false, head, dim);
                    let vi = addressed_index(shape, position + token, true, head, dim);
                    close_scalar(current[ki], expected[dim], "explicit key", ki);
                    assert_eq!(current[vi], v[base + dim]);
                    written[ki] = true;
                    written[vi] = true;
                }
            }
        }
        for (index, (&actual, &before)) in current.iter().zip(&previous).enumerate() {
            if !written[index] {
                assert_eq!(actual, before, "unowned KV element {index}");
            }
        }
        match active.path {
            CausalAttentionKernelPath::VllmAddressedVarlen
            | CausalAttentionKernelPath::VllmAddressedVarlenTiled => {
                let tiled = active.path == CausalAttentionKernelPath::VllmAddressedVarlenTiled;
                launch_addressed_varlen_attention(
                    stream,
                    if tiled {
                        &functions.varlen_tiled_addressed
                    } else {
                        &functions.varlen_addressed
                    },
                    query.pointer(stream),
                    control.pointer(stream),
                    table.pointer(stream),
                    output.pointer(stream),
                    active,
                    cuda,
                    tiled,
                )
                .unwrap();
            }
            _ => launch_fallback_attention(
                stream,
                functions,
                query.pointer(stream),
                q_gpu.pointer(stream),
                control.pointer(stream),
                table.pointer(stream),
                output.pointer(stream),
                active,
                cuda,
                1,
                None,
            )
            .unwrap(),
        }
        let actual = output.read(stream);
        for token in 0..tokens {
            for head in 0..qheads {
                let kvhead = head / (qheads / kvheads);
                let query = &prepared_query[(token * qheads + head) * width..][..width];
                let logits = (0..=position + token)
                    .map(|key| {
                        query
                            .iter()
                            .enumerate()
                            .map(|(dim, q)| {
                                q.to_f64()
                                    * current[addressed_index(shape, key, false, kvhead, dim)]
                                        .to_f64()
                            })
                            .sum::<f64>()
                            * f64::from(cuda.attention_scale)
                    })
                    .collect::<Vec<_>>();
                let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let scores = logits
                    .iter()
                    .map(|value| (value - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = scores.iter().sum::<f64>();
                for dim in 0..width {
                    let reference = scores
                        .iter()
                        .enumerate()
                        .map(|(key, score)| {
                            score * current[addressed_index(shape, key, true, kvhead, dim)].to_f64()
                        })
                        .sum::<f64>()
                        / denominator;
                    let index = (token * qheads + head) * width + dim;
                    close_scalar(
                        actual[index],
                        reference,
                        "addressed explicit attention",
                        index,
                    );
                }
            }
        }
        assert_eq!(
            pages
                .iter()
                .rev()
                .flat_map(|page| page.read(stream))
                .collect::<Vec<_>>(),
            current,
            "attention must not mutate KV"
        );
        outputs.extend(actual);
        for immutable in [&q_gpu, &k_gpu, &v_gpu, &cos_gpu, &sin_gpu, &norm_gpu] {
            immutable.assert_unchanged(stream);
        }
        control.assert_unchanged(stream);
        table.assert_unchanged(stream);
        idle.assert_unchanged(stream);
        previous = current;
        position = length;
    }
    (outputs, previous)
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn causal_parallel_explicit_rope_addressed_decode_matches_f64_and_serial_on_cuda() {
    let runtime = runtime();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::Portable)
            .unwrap();
    let stream = runtime.context().new_stream().unwrap();
    for width in [64, 128, 256] {
        let mut variants = Vec::new();
        for position_variant in [0, 1] {
            let parallel = exercise_explicit::<f16>(
                &stream,
                &provider.functions,
                width,
                position_variant,
                &[1, 1, 1],
            );
            let serial = exercise_explicit::<f16>(
                &stream,
                &provider.functions,
                width,
                position_variant,
                &[3],
            );
            assert_eq!(
                parallel.1, serial.1,
                "explicit preparation must preserve identical KV stores"
            );
            assert_eq!(parallel.0.len(), serial.0.len());
            for (index, (&actual, expected)) in parallel.0.iter().zip(&serial.0).enumerate() {
                close(
                    actual,
                    expected.to_f64(),
                    "explicit parallel/serial attention",
                    index,
                );
            }
            variants.push(parallel.0);
        }
        assert_ne!(
            variants[0], variants[1],
            "explicit positions must affect actual attention outputs"
        );
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn causal_bf16_explicit_rope_kv_and_prefill_match_f64_on_cuda() {
    let mut config = cuda_vnext_runtime_config(
        0,
        DeviceId::new("device.cuda.0").unwrap(),
        AttentionExecutionPolicy::Portable,
    )
    .unwrap();
    config.capabilities.insert(
        ferrum_runtime::CapabilityId::new(
            ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_BF16_CAPABILITY_ID,
        )
        .unwrap(),
    );
    let runtime = CudaDeviceRuntime::new(config).unwrap();
    let provider = CudaCausalPagedAttentionProvider::new_explicit_rope_bf16(&runtime).unwrap();
    let stream = runtime.context().default_stream();
    // Both addressed varlen paths share the same F64 oracle and page guards.
    for count in [3, 4] {
        let decoded =
            exercise_explicit::<half::bf16>(&stream, &provider.functions, 128, 1, &vec![1; count]);
        let prefilled =
            exercise_explicit::<half::bf16>(&stream, &provider.functions, 128, 1, &[count]);
        assert_eq!(
            decoded.1, prefilled.1,
            "BF16 KV must match across prefill/decode"
        );
        for (i, (&actual, expected)) in decoded.0.iter().zip(&prefilled.0).enumerate() {
            close_scalar(actual, expected.to_f64(), "BF16 prefill/decode", i);
        }
    }
}
