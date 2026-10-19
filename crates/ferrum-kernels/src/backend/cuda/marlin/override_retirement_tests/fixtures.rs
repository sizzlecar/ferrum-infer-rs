use super::super::{marlin_gemm, marlin_gemm_moe, MarlinWeight};
use crate::marlin_repack::{repack_gptq_to_marlin, repack_scales_to_marlin};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use std::sync::Arc;

const K: usize = 256;
const GROUP_SIZE: usize = 128;

/// Original GPTQ words and scales stay independent of the Marlin repack.
/// Symmetric AutoGPTQ zero code 7 means the arithmetic zero point is 8.
struct Int4Source {
    n: usize,
    words: Vec<i32>,
    scales: Vec<f16>,
}

impl Int4Source {
    fn new(n: usize, seed: u64) -> Self {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32
        };
        let words = (0..K / 8 * n).map(|_| next() as i32).collect();
        // Exact F16 scales vary across both groups and output channels.
        let scales = (0..K / GROUP_SIZE * n)
            .map(|_| f16::from_f32((1 + next() % 7) as f32 / 128.0))
            .collect();
        Self { n, words, scales }
    }

    fn packed(&self) -> (Vec<i32>, Vec<f16>) {
        (
            repack_gptq_to_marlin(&self.words, K, self.n),
            repack_scales_to_marlin(&self.scales, K, self.n, GROUP_SIZE),
        )
    }

    fn reference(&self, input: &[f16]) -> Vec<f32> {
        assert_eq!(input.len() % K, 0);
        let mut output = vec![0.0_f32; input.len() / K * self.n];
        for row in 0..input.len() / K {
            for column in 0..self.n {
                let mut sum = 0.0_f32;
                for feature in 0..K {
                    let word = self.words[(feature / 8) * self.n + column] as u32;
                    let code = ((word >> (4 * (feature % 8))) & 15) as i32;
                    let scale = self.scales[(feature / GROUP_SIZE) * self.n + column].to_f32();
                    sum += input[row * K + feature].to_f32() * (code - 8) as f32 * scale;
                }
                output[row * self.n + column] = sum;
            }
        }
        output
    }
}

fn input(rows: usize) -> Vec<f16> {
    (0..rows * K)
        .map(|index| f16::from_f32(((index * 17 + index / K * 11) % 127) as f32 / 64.0 - 1.0))
        .collect()
}

fn assert_reference(label: &str, actual: &[f16], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f32;
    let mut reference_max = 0.0_f32;
    let (mut error_sq, mut reference_sq) = (0.0_f64, 0.0_f64);
    for (index, (actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = actual.to_f32();
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label} non-finite index={index}: actual={actual} expected={expected}",
        );
        let error = actual - expected;
        max_abs = max_abs.max(error.abs());
        reference_max = reference_max.max(expected.abs());
        error_sq += f64::from(error).powi(2);
        reference_sq += f64::from(expected).powi(2);
    }
    assert!(
        reference_max > 0.0,
        "fixture must exercise nonzero arithmetic"
    );
    let relative_max = max_abs / reference_max.max(1.0e-6);
    let relative_l2 = error_sq.sqrt() / reference_sq.sqrt().max(1.0e-6);
    eprintln!(
        "native INT4 reference entry={label} elements={} max_abs={max_abs:.8} relative_max={relative_max:.8} relative_l2={relative_l2:.8} all_finite=true",
        actual.len(),
    );
    // Preserve the existing GPTQ CUDA-vs-CPU numerical contract. Every
    // element contributes to the maximum; non-finite values cannot vanish
    // through f32::max. No tolerance is widened for retired overrides.
    assert!(relative_max < 0.05, "{label} relative_max={relative_max}");
}

fn classic_weight(stream: &Arc<CudaStream>, sources: &[Int4Source]) -> MarlinWeight {
    let n = sources[0].n;
    assert!(sources.iter().all(|source| source.n == n));
    let (mut words, mut scales) = (Vec::new(), Vec::new());
    for source in sources {
        let (packed_words, packed_scales) = source.packed();
        words.extend(packed_words);
        scales.extend(packed_scales);
    }
    MarlinWeight {
        qweight: stream.clone_htod(&words).expect("upload Marlin INT4"),
        scales: stream.clone_htod(&scales).expect("upload Marlin scales"),
        qzeros: None,
        workspace: stream
            .alloc_zeros(sources.len() * (n / 128) * 16)
            .expect("Marlin locks"),
        k: K,
        n: sources.len() * n,
        group_size: GROUP_SIZE as i32,
        vllm_moe: false,
        perm: None,
    }
}

pub(super) fn classic_dense() {
    const N: usize = 128;
    const M: usize = 2;
    let context = CudaContext::new(0).expect("actual CUDA device");
    let stream = context.new_stream().expect("CUDA test stream");
    let source = Int4Source::new(N, 0xC0FFEE);
    let input = input(M);
    let reference = source.reference(&input);
    let weight = classic_weight(&stream, &[source]);
    let input_device = stream.clone_htod(&input).expect("upload dense input");
    let mut output: CudaSlice<f16> = stream.alloc_zeros(M * N).expect("dense output");
    // Original wrapper passes (-1,-1) for native automatic tile choice.
    marlin_gemm(
        &stream,
        &input_device,
        &weight,
        &mut output,
        M as i32,
        false,
    )
    .expect("classic dense native FFI status");
    stream
        .synchronize()
        .expect("classic dense actual terminal completion");
    let actual = stream
        .clone_dtoh(&output)
        .expect("read classic dense output");
    assert_reference("marlin_cuda N128 M2", &actual, &reference);
}

pub(super) fn classic_fused() {
    const N: usize = 128;
    let context = CudaContext::new(0).expect("actual CUDA device");
    let stream = context.new_stream().expect("CUDA test stream");
    let sources = [
        Int4Source::new(N, 0xCAFE0000),
        Int4Source::new(N, 0xCAFE0001),
    ];
    let input = input(5);
    let mut reference = sources[0].reference(&input[..2 * K]);
    reference.extend(sources[1].reference(&input[2 * K..]));
    let weight = classic_weight(&stream, &sources);
    let input_device = stream.clone_htod(&input).expect("upload fused input");
    let offsets = stream.clone_htod(&[0_i32, 2]).expect("expert offsets");
    let rows = stream.clone_htod(&[2_i32, 3]).expect("expert row counts");
    let active = stream
        .clone_htod(&[1_i32, 0])
        .expect("nonidentity active experts");
    let mut output: CudaSlice<f16> = stream.alloc_zeros(5 * N).expect("fused output");
    marlin_gemm_moe(
        &stream,
        &input_device,
        &weight,
        &mut output,
        &offsets,
        &rows,
        Some(&active),
        2,
        16,
        N as i32,
        2,
    )
    .expect("classic fused native FFI status");
    stream
        .synchronize()
        .expect("classic fused actual terminal completion");
    let actual = stream
        .clone_dtoh(&output)
        .expect("read classic fused output");
    assert_reference("marlin_cuda_moe N128 E2", &actual, &reference);
}

#[cfg(feature = "vllm-moe-marlin")]
pub(super) fn vllm_f16() {
    use super::super::{
        launch_marlin_moe_vllm_raw, test_guards::Guarded, MarlinMoeF16WeightType,
        MarlinMoeRawLaunchArgs,
    };
    use cudarc::driver::{
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, DevicePtr, DevicePtrMut,
    };

    const N: usize = 256;
    const M: usize = 4;
    const TOP_K: usize = 2;
    const BLOCK: usize = 16;
    let context = CudaContext::new(0).expect("actual CUDA device");
    let stream = context.new_stream().expect("CUDA test stream");
    let sms = usize::try_from(
        context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap(),
    )
    .unwrap();
    let sources = [
        Int4Source::new(N, 0xCAFE0000),
        Int4Source::new(N, 0xCAFE0001),
    ];
    let input = input(M);
    let references: Vec<_> = sources
        .iter()
        .map(|source| source.reference(&input))
        .collect();
    let mut reference = vec![0.0_f32; M * TOP_K * N];
    let mut sorted = vec![(M * TOP_K) as i32; 2 * BLOCK];
    let mut counts = [0_usize; 2];
    for row in 0..M {
        for choice in 0..TOP_K {
            let expert = (row + choice) % 2;
            let pair = row * TOP_K + choice;
            reference[pair * N..(pair + 1) * N]
                .copy_from_slice(&references[expert][row * N..(row + 1) * N]);
            sorted[expert * BLOCK + counts[expert]] = pair as i32;
            counts[expert] += 1;
        }
    }
    let (mut words, mut scales) = (Vec::new(), Vec::new());
    for source in &sources {
        let (packed_words, packed_scales) = source.packed();
        words.extend(packed_words);
        scales.extend(packed_scales);
    }
    let a = stream.clone_htod(&input).expect("U4B8 input");
    let b = stream.clone_htod(&words).expect("U4B8 packed experts");
    let s = stream.clone_htod(&scales).expect("U4B8 scales");
    let sorted = stream.clone_htod(&sorted).expect("U4B8 sorted token pairs");
    let experts = stream.clone_htod(&[0_i32, 1]).expect("U4B8 expert blocks");
    let padded = stream
        .clone_htod(&[(2 * BLOCK) as i32])
        .expect("U4B8 padded count");
    let mut output = Guarded::new(
        &stream,
        &vec![f16::ZERO; M * TOP_K * N],
        f16::from_f32(-117.0),
    );
    let mut workspace = Guarded::new(&stream, &vec![0_i32; N.div_ceil(128) * sms * 4], -117_i32);
    {
        let mut output_view = output.view_mut();
        let mut workspace_view = workspace.view_mut();
        let (a_ptr, _a) = a.device_ptr(&stream);
        let (b_ptr, _b) = b.device_ptr(&stream);
        let (s_ptr, _s) = s.device_ptr(&stream);
        let (out_ptr, _out) = output_view.device_ptr_mut(&stream);
        let (ws_ptr, _ws) = workspace_view.device_ptr_mut(&stream);
        let (sorted_ptr, _sorted) = sorted.device_ptr(&stream);
        let (expert_ptr, _experts) = experts.device_ptr(&stream);
        let (padded_ptr, _padded) = padded.device_ptr(&stream);
        launch_marlin_moe_vllm_raw(
            &stream,
            MarlinMoeRawLaunchArgs {
                weight_type: MarlinMoeF16WeightType::U4B8,
                a: a_ptr,
                b: b_ptr,
                c: out_ptr,
                c_tmp: None,
                scales: s_ptr,
                zero_points: None,
                workspace: ws_ptr,
                sorted_token_ids: sorted_ptr,
                expert_ids: expert_ptr,
                num_tokens_past_padded: padded_ptr,
                topk_weights: None,
                moe_block_size: BLOCK as i32,
                top_k: TOP_K as i32,
                mul_topk_weights: false,
                is_ep: false,
                prob_m: M as i32,
                prob_n: N as i32,
                prob_k: K as i32,
                group_size: GROUP_SIZE as i32,
                has_zero_points: false,
                device_ordinal: context.ordinal() as i32,
                use_atomic_add: true,
                use_fp32_reduce: false,
            },
        )
        .expect("ferrum_vllm_marlin_moe_f16 native FFI status");
        stream
            .synchronize()
            .expect("U4B8/F16 actual terminal completion");
    }
    let actual = output.read(&stream);
    workspace.read(&stream);
    assert_reference("ferrum_vllm_marlin_moe_f16 E2 topK2", &actual, &reference);
}
