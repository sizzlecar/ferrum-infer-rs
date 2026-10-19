//! Samples the existing decode paths and replays their original attention
//! kernels on the captured inputs; it does not implement a second forward.

use super::{
    assert_cuda_physical_logits_qualification, physical_batch_and_single_logits,
    physical_batch_config, physical_batch_full_logits, physical_batch_model_with_loader,
    DecoderOnlyLLM, LlamaFamilyConfig, LlamaFamilyModel, SyntheticLlamaLoader,
};
use ferrum_interfaces::kv_dtype::KvFp16;
use ferrum_kernels::backend::{cuda::CudaBackend, AttnConfig, Backend};
use ferrum_kernels::linear::LinearMetadata;
use ferrum_quantization::{Linear, QuantConfig, WeightLoader};
use ferrum_types::Result;
use half::f16;
use parking_lot::Mutex;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

type Context = <CudaBackend as Backend>::Context;
type Buffer = <CudaBackend as Backend>::Buffer;

#[derive(Debug)]
struct ProjectionSample {
    name: String,
    rows: usize,
    input_width: usize,
    output_width: usize,
    input: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Default)]
struct ProjectionTrace {
    enabled: AtomicBool,
    samples: Mutex<Vec<ProjectionSample>>,
}

struct TappedLinear {
    inner: Box<dyn Linear<CudaBackend>>,
    name: String,
    trace: Arc<ProjectionTrace>,
}

impl Linear<CudaBackend> for TappedLinear {
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    fn metadata(&self) -> LinearMetadata {
        self.inner.metadata()
    }

    fn cuda_marlin_touch_ref(
        &self,
    ) -> Option<ferrum_kernels::quant_linear::cuda_marlin::CudaMarlinTouchRef<'_>> {
        self.inner.cuda_marlin_touch_ref()
    }

    fn forward(&self, ctx: &mut Context, input: &Buffer, out: &mut Buffer, m: usize) {
        // Prefill follows the original path without diagnostic readbacks.
        // Each decode projection still runs exactly once on its original ctx.
        let enabled = self.trace.enabled.load(Ordering::Acquire);
        self.inner.forward(ctx, input, out, m);
        if enabled {
            // Linear's input is immutable and distinct from its output. Both
            // readbacks use the original stream and synchronize before the
            // following operation can overwrite its scratch. Retain host
            // values only; no buffer, stream, or model escapes to the trace.
            let input_width = self.in_features();
            let output_width = self.out_features();
            let input = CudaBackend::to_vec_in_context(
                ctx,
                input,
                m.checked_mul(input_width).expect("projection input extent"),
            );
            let output = CudaBackend::to_vec_in_context(
                ctx,
                out,
                m.checked_mul(output_width)
                    .expect("projection output extent"),
            );
            self.trace.samples.lock().push(ProjectionSample {
                name: self.name.clone(),
                rows: m,
                input_width,
                output_width,
                input,
                output,
            });
        }
    }
}

struct TappedLoader {
    inner: SyntheticLlamaLoader,
    trace: Arc<ProjectionTrace>,
}

impl WeightLoader<CudaBackend> for TappedLoader {
    fn load_tensor(&self, ctx: &mut Context, name: &str) -> Result<Buffer> {
        <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::load_tensor(&self.inner, ctx, name)
    }

    fn load_linear(&self, ctx: &mut Context, name: &str) -> Result<Box<dyn Linear<CudaBackend>>> {
        let inner = <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::load_linear(
            &self.inner,
            ctx,
            name,
        )?;
        Ok(Box::new(TappedLinear {
            inner,
            name: name.to_owned(),
            trace: Arc::clone(&self.trace),
        }))
    }

    fn has_tensor(&self, name: &str) -> bool {
        <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::has_tensor(&self.inner, name)
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::quant_config(&self.inner)
    }
}

struct SampledDecode {
    batch_logits: Vec<Vec<f32>>,
    single_logits: Vec<Vec<f32>>,
    batch_samples: Vec<ProjectionSample>,
    single_samples: Vec<ProjectionSample>,
    batch_attention: Vec<AttentionSample>,
    single_attention: Vec<AttentionSample>,
}

struct AttentionSample {
    q: Vec<f32>,
    // Compact [kv_head, valid_position, dimension], excluding capacity padding.
    k: Vec<f32>,
    v: Vec<f32>,
    output: Vec<f32>,
    kv_len: usize,
}

// Rebuild only the physical head-major storage. Unused capacity is initialized
// independently of the model, so the replay cannot inherit its scratch or KV
// allocation history. Every valid element remains the sampled FP16 value.
fn replay_cache(config: &LlamaFamilyConfig, sample: &AttentionSample, values: &[f32]) -> Vec<f32> {
    const CAPACITY: usize = 32;
    let hd = config.head_dim;
    assert!(sample.kv_len > 0 && sample.kv_len <= CAPACITY);
    assert_eq!(values.len(), config.num_kv_heads * sample.kv_len * hd);
    let mut padded = vec![0.0; config.num_kv_heads * CAPACITY * hd];
    for head in 0..config.num_kv_heads {
        let source = head * sample.kv_len * hd;
        let destination = head * CAPACITY * hd;
        padded[destination..destination + sample.kv_len * hd]
            .copy_from_slice(&values[source..source + sample.kv_len * hd]);
    }
    padded
}

fn replay_single_attention(config: &LlamaFamilyConfig, sample: &AttentionSample) -> Vec<f32> {
    let mut ctx = CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
    CudaBackend::begin_execution(&mut ctx).unwrap();
    let width = config.num_heads * config.head_dim;
    assert_eq!(sample.q.len(), width);
    let q = CudaBackend::from_slice_in_context(&mut ctx, &sample.q);
    let k = CudaBackend::from_slice_in_context(&mut ctx, &replay_cache(config, sample, &sample.k));
    let v = CudaBackend::from_slice_in_context(&mut ctx, &replay_cache(config, sample, &sample.v));
    let mut output = CudaBackend::alloc_in_context(&mut ctx, width);
    CudaBackend::flash_attention(
        &mut ctx,
        &q,
        &k,
        &v,
        &mut output,
        1,
        1,
        sample.kv_len,
        sample.kv_len - 1,
        &AttnConfig {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            causal: true,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            kv_seq_stride: 32,
            sliding_window: 0,
        },
    );
    CudaBackend::observe_completion(&mut ctx)
        .into_result()
        .unwrap();
    let values = CudaBackend::to_vec_in_context(&mut ctx, &output, width);
    CudaBackend::retire_context(&mut ctx).unwrap();
    values
}

fn replay_batched_attention(
    config: &LlamaFamilyConfig,
    samples: &[AttentionSample],
) -> Vec<Vec<f32>> {
    assert_eq!(samples.len(), 2);
    let mut ctx = CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
    CudaBackend::begin_execution(&mut ctx).unwrap();
    let width = config.num_heads * config.head_dim;
    let q_values: Vec<_> = samples
        .iter()
        .flat_map(|sample| {
            assert_eq!(sample.q.len(), width);
            sample.q.iter().copied()
        })
        .collect();
    let q = CudaBackend::from_slice_in_context(&mut ctx, &q_values);
    let keys: Vec<_> = samples
        .iter()
        .map(|sample| {
            CudaBackend::from_slice_in_context(&mut ctx, &replay_cache(config, sample, &sample.k))
        })
        .collect();
    let values: Vec<_> = samples
        .iter()
        .map(|sample| {
            CudaBackend::from_slice_in_context(&mut ctx, &replay_cache(config, sample, &sample.v))
        })
        .collect();
    let lengths: Vec<_> = samples
        .iter()
        .map(|sample| u32::try_from(sample.kv_len).unwrap())
        .collect();
    let lengths = CudaBackend::from_slice_typed_in_context(&mut ctx, &lengths);
    let mut output = CudaBackend::alloc_in_context(&mut ctx, samples.len() * width);
    CudaBackend::flash_attention_batched_per_cache(
        &mut ctx,
        &q,
        &keys.iter().collect::<Vec<_>>(),
        &values.iter().collect::<Vec<_>>(),
        &lengths,
        &mut output,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        1.0 / (config.head_dim as f32).sqrt(),
        samples.iter().map(|sample| sample.kv_len).max().unwrap(),
        32,
        0,
        0,
    )
    .unwrap();
    CudaBackend::observe_completion(&mut ctx)
        .into_result()
        .unwrap();
    let result = CudaBackend::to_vec_in_context(&mut ctx, &output, samples.len() * width);
    CudaBackend::retire_context(&mut ctx).unwrap();
    result.chunks_exact(width).map(<[f32]>::to_vec).collect()
}

fn assert_attention_kernel_replay(config: &LlamaFamilyConfig, sampled: &SampledDecode) {
    assert_eq!(config.sliding_window, 0);
    let replayed_batch = replay_batched_attention(config, &sampled.batch_attention);
    let replayed_single: Vec<_> = sampled
        .single_attention
        .iter()
        .map(|sample| replay_single_attention(config, sample))
        .collect();
    let model_batch: Vec<_> = sampled
        .batch_attention
        .iter()
        .map(|sample| sample.output.clone())
        .collect();
    let model_single: Vec<_> = sampled
        .single_attention
        .iter()
        .map(|sample| sample.output.clone())
        .collect();
    assert_unchanged_logits(
        "M2 original attention kernel replay",
        &replayed_batch,
        &model_batch,
    );
    assert_unchanged_logits(
        "M1 original attention kernel replay",
        &replayed_single,
        &model_single,
    );
    for row in 0..sampled.batch_attention.len() {
        assert_attention_reference(
            row,
            "M2_kernel_replay_to_f64",
            &replayed_batch[row],
            &attention_reference(config, &sampled.batch_attention[row]),
        );
        assert_attention_reference(
            row,
            "M1_kernel_replay_to_f64",
            &replayed_single[row],
            &attention_reference(config, &sampled.single_attention[row]),
        );
        eprintln!("cuda_attention_kernel_replay row={row} elements={} M1_model_bitwise=true M2_model_bitwise=true", replayed_batch[row].len());
    }
}

fn sample_attention(
    model: &mut LlamaFamilyModel<CudaBackend, KvFp16>,
    cache_ids: &[&str],
    batched: bool,
) -> Vec<AttentionSample> {
    model
        .run_preparation_frame(|model, frame| {
            assert_eq!(
                model.cfg.num_layers, 1,
                "scratch belongs to this sole layer"
            );
            assert_eq!(model.cfg.sliding_window, 0);
            assert!(batched || cache_ids.len() == 1);
            let ctx = &mut frame.context;
            let width = model.cfg.num_heads * model.cfg.head_dim;
            let (q, output) = if batched {
                (&model.scratch.q_normed_batched, &model.scratch.attn_flat)
            } else {
                (
                    &model.scratch.q_head_major,
                    &model.scratch.attn_head_major_out,
                )
            };
            let q = CudaBackend::to_vec_in_context(ctx, q, cache_ids.len() * width);
            let output = CudaBackend::to_vec_in_context(ctx, output, cache_ids.len() * width);
            cache_ids
                .iter()
                .enumerate()
                .map(|(row, cache_id)| {
                    let layers = &model.kv_caches[*cache_id];
                    assert_eq!(layers.len(), 1);
                    let cache = &layers[0];
                    assert_eq!(
                        cache.block_size, 0,
                        "this fixture uses contiguous head-major KV"
                    );
                    assert_eq!(cache.capacity, 32);
                    assert_eq!(cache.len, 4);
                    assert_eq!(cache.num_kv_heads, model.cfg.num_kv_heads);
                    assert_eq!(cache.head_dim, model.cfg.head_dim);
                    // Read through the last valid element, then exclude gaps between
                    // heads. In particular, capacity 32 is not the attention length 4.
                    let extent =
                        ((cache.num_kv_heads - 1) * cache.capacity + cache.len) * cache.head_dim;
                    let mut read_valid = |buffer: &Buffer| {
                        let raw = CudaBackend::to_vec_in_context(ctx, buffer, extent);
                        (0..cache.num_kv_heads)
                            .flat_map(|head| {
                                let start = head * cache.capacity * cache.head_dim;
                                raw[start..start + cache.len * cache.head_dim]
                                    .iter()
                                    .copied()
                            })
                            .collect()
                    };
                    AttentionSample {
                        q: q[row * width..(row + 1) * width].to_vec(),
                        k: read_valid(&cache.k),
                        v: read_valid(&cache.v),
                        output: output[row * width..(row + 1) * width].to_vec(),
                        kv_len: cache.len,
                    }
                })
                .collect()
        })
        .into_legacy_result()
}

fn sampled_decode() -> SampledDecode {
    let config = physical_batch_config();
    let batch_trace = Arc::new(ProjectionTrace::default());
    let single_trace = Arc::new(ProjectionTrace::default());
    let batch_loader = TappedLoader {
        inner: SyntheticLlamaLoader::new(config.clone()),
        trace: Arc::clone(&batch_trace),
    };
    let single_loader = TappedLoader {
        inner: SyntheticLlamaLoader::new(config.clone()),
        trace: Arc::clone(&single_trace),
    };
    let mut batched =
        physical_batch_model_with_loader::<CudaBackend>(config.clone(), &batch_loader);
    let mut single = physical_batch_model_with_loader::<CudaBackend>(config, &single_loader);
    // Host readback is intentionally incompatible with capture. Keep the
    // original numerical fixture's eager policy, including both graph flags.
    for model in [&batched, &single] {
        assert!(!model.runtime_env.cuda_graph);
        assert!(!model.batched_cfg.batched_graph);
    }
    assert!(batch_trace.samples.lock().is_empty());
    assert!(single_trace.samples.lock().is_empty());
    batch_trace.enabled.store(true, Ordering::Release);
    single_trace.enabled.store(true, Ordering::Release);

    let batch = [("a".to_string(), 4, 3), ("b".to_string(), 5, 3)];
    let mut single_logits = Vec::new();
    let mut single_attention = Vec::new();
    for (id, token, position) in &batch {
        single_logits.push(single.decode_internal(id, *token, *position));
        single_attention.extend(sample_attention(&mut single, &[id.as_str()], false));
    }
    let batch_logits = physical_batch_full_logits(&mut batched, &batch);
    let batch_attention = sample_attention(&mut batched, &["a", "b"], true);
    batch_trace.enabled.store(false, Ordering::Release);
    single_trace.enabled.store(false, Ordering::Release);
    for model in [&mut batched, &mut single] {
        for id in ["a", "b"] {
            assert!(model.kv_caches[id].iter().all(|cache| cache.len == 4));
            model.release(id);
        }
        assert!(model.kv_caches.is_empty());
        model.close().unwrap();
    }
    let batch_samples = std::mem::take(&mut *batch_trace.samples.lock());
    let single_samples = std::mem::take(&mut *single_trace.samples.lock());
    SampledDecode {
        batch_logits,
        single_logits,
        batch_samples,
        single_samples,
        batch_attention,
        single_attention,
    }
}

fn assert_unchanged_logits(label: &str, sampled: &[Vec<f32>], original: &[Vec<f32>]) {
    assert_eq!(sampled.len(), original.len(), "{label} rows");
    for (row, (sampled, original)) in sampled.iter().zip(original).enumerate() {
        assert_eq!(sampled.len(), original.len(), "{label} row={row} width");
        for (index, (&sampled, &original)) in sampled.iter().zip(original).enumerate() {
            assert!(sampled.is_finite() && original.is_finite());
            assert_eq!(
                sampled.to_bits(),
                original.to_bits(),
                "sampling changed {label} row={row} logit={index}: {sampled} vs {original}",
            );
        }
    }
}

#[derive(Debug)]
struct Difference {
    index: usize,
    batched: f32,
    single: f32,
}

// Compare every element, even after the first difference. The returned bitwise
// difference localizes divergence; it does not declare that rounding is valid.
fn compare_values(
    row: usize,
    stage: &str,
    side: &str,
    batched: &[f32],
    single: &[f32],
) -> Option<Difference> {
    assert_eq!(batched.len(), single.len(), "{stage} {side} width");
    let mut first = None;
    let mut bit_differences = 0;
    let (mut max_abs, mut error_sq, mut reference_sq) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (index, (&batched, &single)) in batched.iter().zip(single).enumerate() {
        assert!(
            batched.is_finite() && single.is_finite(),
            "non-finite row={row} stage={stage} side={side} index={index}: {batched}/{single}",
        );
        if batched.to_bits() != single.to_bits() {
            bit_differences += 1;
            first.get_or_insert(Difference {
                index,
                batched,
                single,
            });
        }
        let error = f64::from(batched) - f64::from(single);
        max_abs = max_abs.max(error.abs());
        error_sq += error * error;
        reference_sq += f64::from(single).powi(2);
    }
    let relative_l2 = if error_sq == 0.0 {
        0.0
    } else {
        (error_sq / reference_sq).sqrt()
    };
    eprintln!(
        "cuda_projection row={row} stage={stage} side={side} elements={} bit_differences={bit_differences} first={first:?} max_abs={max_abs:.9e} relative_l2={relative_l2:.9e}",
        batched.len(),
    );
    first
}

// Independent scalar attention over the actual quantized Q/K/V. This neither
// repeats a model forward nor models either CUDA reduction/rounding algorithm.
fn attention_reference(config: &LlamaFamilyConfig, sample: &AttentionSample) -> Vec<f64> {
    let hd = config.head_dim;
    let heads_per_kv = config.num_heads / config.num_kv_heads;
    assert_eq!(config.num_heads % config.num_kv_heads, 0);
    assert_eq!(sample.q.len(), config.num_heads * hd);
    assert_eq!(sample.k.len(), config.num_kv_heads * sample.kv_len * hd);
    assert_eq!(sample.k.len(), sample.v.len());
    for values in [&sample.q, &sample.k, &sample.v, &sample.output] {
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(
            values
                .iter()
                .all(|&value| f16::from_f32(value).to_f32().to_bits() == value.to_bits()),
            "reference input/output must be the actual FP16 storage values"
        );
    }
    // Match the scalar passed to both GPU kernels; all subsequent arithmetic
    // is f64. The final query at position 3 can attend to all four valid tokens.
    let scale = f64::from(1.0_f32 / (hd as f32).sqrt());
    let mut output = Vec::with_capacity(config.num_heads * hd);
    for head in 0..config.num_heads {
        let kv_head = head / heads_per_kv;
        let scores = (0..sample.kv_len)
            .map(|position| {
                let k_start = (kv_head * sample.kv_len + position) * hd;
                (0..hd)
                    .map(|dimension| {
                        f64::from(sample.q[head * hd + dimension])
                            * f64::from(sample.k[k_start + dimension])
                    })
                    .sum::<f64>()
                    * scale
            })
            .collect::<Vec<_>>();
        assert!(scores.iter().all(|score| score.is_finite()));
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator: f64 = weights.iter().sum();
        assert!(denominator.is_finite() && denominator > 0.0);
        for dimension in 0..hd {
            output.push(
                weights
                    .iter()
                    .enumerate()
                    .map(|(position, weight)| {
                        let v_index = (kv_head * sample.kv_len + position) * hd + dimension;
                        weight * f64::from(sample.v[v_index])
                    })
                    .sum::<f64>()
                    / denominator,
            );
        }
    }
    assert!(output.iter().all(|value| value.is_finite()));
    output
}

fn reference_error(row: usize, label: &str, actual: &[f32], reference: &[f64]) {
    assert_eq!(actual.len(), reference.len());
    let (mut max_abs, mut error_sq, mut reference_sq) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut nearest_half_differences = 0;
    for (&actual, &reference) in actual.iter().zip(reference) {
        assert!(actual.is_finite() && reference.is_finite());
        let error = f64::from(actual) - reference;
        max_abs = max_abs.max(error.abs());
        error_sq += error * error;
        reference_sq += reference * reference;
        nearest_half_differences +=
            usize::from(actual.to_bits() != f16::from_f64(reference).to_f32().to_bits());
    }
    let relative_l2 = if error_sq == 0.0 {
        0.0
    } else {
        (error_sq / reference_sq).sqrt()
    };
    eprintln!("cuda_attention_reference row={row} comparison={label} elements={} finite=true max_abs={max_abs:.12e} relative_l2={relative_l2:.12e} nearest_half_differences={nearest_half_differences}", actual.len());
}

fn assert_attention_reference(row: usize, label: &str, actual: &[f32], reference: &[f64]) {
    reference_error(row, label, actual, reference);
    // Reuse the existing F16-store engineering envelope from kernels'
    // causal_attention/numerical_tests.rs::close. The oracle evaluates the
    // actual stored Q/K/V in F64; neither GPU shape is the other's truth.
    // This is not a claim that the attention expression is correctly rounded.
    for (index, (&actual, &expected)) in actual.iter().zip(reference).enumerate() {
        let bound = expected.abs().max(1.0) * 0.001 + 1.0e-5;
        assert!(
            actual.is_finite() && expected.is_finite()
                && (f64::from(actual) - expected).abs() <= bound,
            "{label} row={row} element={index}: actual={actual}, F64 reference={expected}, bound={bound}"
        );
    }
}

fn adjacent_half(value: f16, upward: bool) -> f16 {
    let bits = value.to_bits();
    assert!(value.is_finite());
    let next = if bits & 0x7fff == 0 {
        if upward {
            1
        } else {
            0x8001
        }
    } else if (bits & 0x8000 == 0) == upward {
        bits + 1
    } else {
        bits - 1
    };
    f16::from_bits(next)
}

fn compare_attention(
    row: usize,
    config: &LlamaFamilyConfig,
    batched: &AttentionSample,
    single: &AttentionSample,
) {
    assert_eq!(batched.kv_len, single.kv_len);
    let mut equal_inputs = true;
    for (side, batch_values, single_values) in [
        ("Q", &batched.q, &single.q),
        ("K", &batched.k, &single.k),
        ("V", &batched.v, &single.v),
    ] {
        equal_inputs &=
            compare_values(row, "attention_boundary", side, batch_values, single_values).is_none();
    }
    let reference = attention_reference(config, single);
    assert_attention_reference(row, "M1_to_M1_inputs_f64", &single.output, &reference);
    let batch_reference = attention_reference(config, batched);
    assert_attention_reference(
        row,
        "M2_to_M2_inputs_f64",
        &batched.output,
        &batch_reference,
    );
    if equal_inputs {
        assert_eq!(reference, batch_reference);
    } else {
        // Cross-input comparison is diagnostic only; each shape was qualified
        // against its own actual Q/K/V above.
        reference_error(row, "M2_to_M1_inputs_f64", &batched.output, &reference);
    }
    let input_effect = reference
        .iter()
        .zip(&batch_reference)
        .map(|(single, batched)| (single - batched).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("cuda_attention_boundary row={row} qkv_bitwise_equal={equal_inputs} reference_input_max_abs={input_effect:.12e}");
    // Bound diagnostic volume, but statistics above include every element.
    for (index, (&batch_value, &single_value)) in batched
        .output
        .iter()
        .zip(&single.output)
        .enumerate()
        .filter(|(_, (batch, single))| batch.to_bits() != single.to_bits())
        .take(8)
    {
        for (source, value) in [
            ("M1_inputs", reference[index]),
            ("M2_inputs", batch_reference[index]),
        ] {
            let nearest = f16::from_f64(value);
            let (lower, upper) = if nearest.to_f64() <= value {
                (nearest.to_f64(), adjacent_half(nearest, true).to_f64())
            } else {
                (adjacent_half(nearest, false).to_f64(), nearest.to_f64())
            };
            let midpoint = (lower + upper) / 2.0;
            eprintln!("cuda_attention_half_boundary row={row} index={index} reference_source={source} reference={value:.17e} M2={batch_value:.17e} M1={single_value:.17e} nearest={:.17e} lower={lower:.17e} upper={upper:.17e} midpoint={midpoint:.17e} reference_minus_midpoint={:.17e}", nearest.to_f64(), value - midpoint);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device; checks kernel replay and independent numerical oracles"]
fn physical_batch_cuda_projection_inputs_locate_first_difference() {
    let (original_batch, original_single) = physical_batch_and_single_logits::<CudaBackend>();
    let sampled = sampled_decode();
    // Without this check, added synchronization could conceal the original
    // issue. Do not use a perturbed run to attribute a numerical difference.
    assert_unchanged_logits("M2", &sampled.batch_logits, &original_batch);
    assert_unchanged_logits("M1", &sampled.single_logits, &original_single);

    let config = physical_batch_config();
    let qkv_width = (config.num_heads + 2 * config.num_kv_heads) * config.head_dim;
    let stages = [
        (
            "model.layers.0.self_attn.qkv_proj",
            config.hidden_size,
            qkv_width,
        ),
        (
            "model.layers.0.self_attn.o_proj",
            config.hidden_size,
            config.hidden_size,
        ),
        (
            "model.layers.0.mlp.gate_up_proj",
            config.hidden_size,
            2 * config.intermediate_size,
        ),
        (
            "model.layers.0.mlp.down_proj",
            config.intermediate_size,
            config.hidden_size,
        ),
        ("lm_head", config.hidden_size, config.vocab_size),
    ];
    assert_eq!(sampled.batch_samples.len(), stages.len());
    assert_eq!(sampled.single_samples.len(), 2 * stages.len());
    assert_eq!(sampled.batch_attention.len(), 2);
    assert_eq!(sampled.single_attention.len(), 2);
    assert_attention_kernel_replay(&config, &sampled);
    for row in 0..2 {
        let batch_attention = &sampled.batch_attention[row];
        let single_attention = &sampled.single_attention[row];
        let width = config.num_heads * config.head_dim;
        // Validate that the post-decode scratch is exactly the original
        // attention output consumed by o_proj, not a later scratch reuse.
        assert_unchanged_logits(
            "M2 attention scratch",
            &[batch_attention.output.clone()],
            &[sampled.batch_samples[1].input[row * width..(row + 1) * width].to_vec()],
        );
        assert_unchanged_logits(
            "M1 attention scratch",
            &[single_attention.output.clone()],
            &[sampled.single_samples[row * stages.len() + 1].input.clone()],
        );
        compare_attention(row, &config, batch_attention, single_attention);
        let mut first_difference = None;
        for (stage_index, &(name, input_width, output_width)) in stages.iter().enumerate() {
            let batched = &sampled.batch_samples[stage_index];
            let single = &sampled.single_samples[row * stages.len() + stage_index];
            for (sample, rows) in [(batched, 2), (single, 1)] {
                assert_eq!(sample.name, name);
                assert_eq!(sample.rows, rows, "{name} actual projection M");
                assert_eq!(sample.input_width, input_width, "{name} K");
                assert_eq!(sample.output_width, output_width, "{name} N");
                assert_eq!(sample.input.len(), rows * input_width);
                assert_eq!(sample.output.len(), rows * output_width);
            }
            for (side, width, batch_values, single_values) in [
                ("input", input_width, &batched.input, &single.input),
                ("output", output_width, &batched.output, &single.output),
            ] {
                let difference = compare_values(
                    row,
                    name,
                    side,
                    &batch_values[row * width..(row + 1) * width],
                    single_values,
                );
                if let Some(difference) = difference {
                    first_difference.get_or_insert((name, side, difference));
                }
            }
        }
        match first_difference {
            Some((
                stage,
                side,
                Difference {
                    index,
                    batched,
                    single,
                },
            )) => eprintln!(
                "cuda_projection_first_difference row={row} stage={stage} side={side} index={index} M2={batched:?} M1={single:?}",
            ),
            None => eprintln!("cuda_projection_first_difference row={row} none"),
        }
    }
    // Use the same existing CPU-oracle qualification as the mandatory CUDA
    // numerical test. The former cross-shape 1e-5 failure remains diagnostic
    // evidence; it is not retroactively claimed to have passed.
    assert_cuda_physical_logits_qualification(&sampled.batch_logits, &sampled.single_logits);
}
