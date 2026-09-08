//! Small native vLLM paged-attention parity test; no model weights are needed.

#![cfg(all(feature = "cuda", feature = "vllm-paged-attn-v2"))]

use cudarc::driver::CudaContext;
use ferrum_kernels::backend::cuda::vllm_paged_attn::dispatch_paged_attention_v2;
use half::f16;

const BLOCK_SIZE: usize = 16;
const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const PARTITION_SIZE: usize = 512;

struct Inputs {
    head_dim: usize,
    sequence_lengths: [usize; 2],
    queries: Vec<f16>,
    // Logical token-major storage, independent of the native cache layout.
    keys: Vec<Vec<f16>>,
    values: Vec<Vec<f16>>,
}

impl Inputs {
    fn new(head_dim: usize) -> Self {
        // The short row has a partial block; the long row needs two partitions
        // and a partial last block. A mixed batch also tests inactive partitions.
        let sequence_lengths = [BLOCK_SIZE + 3, PARTITION_SIZE + BLOCK_SIZE + 1];
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let mut sample = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 32) % 129) as f32 / 128.0 - 0.5
        };
        let queries = (0..sequence_lengths.len() * QUERY_HEADS * head_dim)
            .map(|_| f16::from_f32(2.0 * sample()))
            .collect();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (sequence, &length) in sequence_lengths.iter().enumerate() {
            let mut sequence_keys = Vec::with_capacity(length * KV_HEADS * head_dim);
            let mut sequence_values = Vec::with_capacity(length * KV_HEADS * head_dim);
            for token in 0..length {
                for kv_head in 0..KV_HEADS {
                    for _ in 0..head_dim {
                        sequence_keys.push(f16::from_f32(2.0 * sample()));
                        // Distinct KV heads expose GQA indexing errors. The
                        // second partition changes the answer enough that
                        // omitting its contribution cannot hide in FP16 error.
                        let offset = sequence as f32 * 0.25
                            + kv_head as f32
                            + (token / PARTITION_SIZE) as f32 * 3.0;
                        sequence_values.push(f16::from_f32(sample() + offset));
                    }
                }
            }
            keys.push(sequence_keys);
            values.push(sequence_values);
        }
        Self {
            head_dim,
            sequence_lengths,
            queries,
            keys,
            values,
        }
    }

    fn max_blocks(&self) -> usize {
        self.sequence_lengths
            .iter()
            .copied()
            .max()
            .unwrap()
            .div_ceil(BLOCK_SIZE)
    }

    fn native_cache(&self) -> (Vec<f16>, Vec<f16>, Vec<u32>) {
        let used_blocks: usize = self
            .sequence_lengths
            .iter()
            .map(|length| length.div_ceil(BLOCK_SIZE))
            .sum();
        // Block zero is a valid but unused guard. Reverse and rotate the other
        // blocks so neither sequence can use logical indices as physical ids.
        let mut physical_blocks: Vec<usize> = (1..=used_blocks).rev().collect();
        physical_blocks.rotate_left(used_blocks / 3);
        let elements = (used_blocks + 1) * KV_HEADS * self.head_dim * BLOCK_SIZE;
        // Finite padding deliberately differs from valid values. Reading past
        // seq_len or unused block-table entries changes the numerical result.
        let mut keys = vec![f16::from_f32(16.0); elements];
        let mut values = keys.clone();
        let mut table = vec![0; self.sequence_lengths.len() * self.max_blocks()];
        let mut next_block = 0;
        const KEY_PACK: usize = 8; // Native K layout uses 16-byte FP16 vectors.
        for (sequence, &length) in self.sequence_lengths.iter().enumerate() {
            for logical_block in 0..length.div_ceil(BLOCK_SIZE) {
                table[sequence * self.max_blocks() + logical_block] =
                    physical_blocks[next_block] as u32;
                next_block += 1;
            }
            for token in 0..length {
                let block = table[sequence * self.max_blocks() + token / BLOCK_SIZE] as usize;
                let slot = token % BLOCK_SIZE;
                for kv_head in 0..KV_HEADS {
                    let base = (block * KV_HEADS + kv_head) * self.head_dim * BLOCK_SIZE;
                    for dimension in 0..self.head_dim {
                        let logical = (token * KV_HEADS + kv_head) * self.head_dim + dimension;
                        // K: [block, kv_head, dim/8, token_in_block, dim%8].
                        let key_index = base
                            + (dimension / KEY_PACK) * BLOCK_SIZE * KEY_PACK
                            + slot * KEY_PACK
                            + dimension % KEY_PACK;
                        // V: [block, kv_head, dim, token_in_block].
                        let value_index = base + dimension * BLOCK_SIZE + slot;
                        keys[key_index] = self.keys[sequence][logical];
                        values[value_index] = self.values[sequence][logical];
                    }
                }
            }
        }
        (keys, values, table)
    }

    fn cpu_reference(&self) -> Vec<f64> {
        // One global FP64 softmax over logical tokens: no block-table lookup,
        // packed indexing, or native partition/reduction algorithm is reused.
        let mut output = vec![0.0; self.queries.len()];
        let scale = f64::from(1.0_f32 / (self.head_dim as f32).sqrt());
        for (sequence, &length) in self.sequence_lengths.iter().enumerate() {
            for head in 0..QUERY_HEADS {
                let kv_head = head / (QUERY_HEADS / KV_HEADS);
                let query_start = (sequence * QUERY_HEADS + head) * self.head_dim;
                let mut logits = Vec::with_capacity(length);
                for token in 0..length {
                    let key_start = (token * KV_HEADS + kv_head) * self.head_dim;
                    let dot: f64 = (0..self.head_dim)
                        .map(|dimension| {
                            f64::from(self.queries[query_start + dimension].to_f32())
                                * f64::from(self.keys[sequence][key_start + dimension].to_f32())
                        })
                        .sum();
                    logits.push(dot * scale);
                }
                let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> =
                    logits.iter().map(|logit| (logit - maximum).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                for (token, weight) in weights.iter().enumerate() {
                    let value_start = (token * KV_HEADS + kv_head) * self.head_dim;
                    for dimension in 0..self.head_dim {
                        output[query_start + dimension] += weight / denominator
                            * f64::from(self.values[sequence][value_start + dimension].to_f32());
                    }
                }
            }
        }
        output
    }
}

#[test]
#[ignore = "requires a CUDA GPU and the vLLM paged-attention native artifact"]
fn native_paged_attention_matches_cpu_with_permuted_blocks_and_partition_tail() {
    let context = CudaContext::new(0).expect("CUDA device 0");
    let stream = context.default_stream();
    // These are the two separately compiled native head-size instantiations.
    // Reuse one context and stream because the dispatcher owns shared scratch.
    for head_dim in [128, 256] {
        let inputs = Inputs::new(head_dim);
        let expected = inputs.cpu_reference();
        let (keys, values, table) = inputs.native_cache();
        let lengths: Vec<u32> = inputs.sequence_lengths.iter().map(|&n| n as u32).collect();
        let query_device = stream.clone_htod(&inputs.queries).expect("upload queries");
        let key_device = stream.clone_htod(&keys).expect("upload packed keys");
        let value_device = stream.clone_htod(&values).expect("upload packed values");
        let table_device = stream.clone_htod(&table).expect("upload block table");
        let length_device = stream
            .clone_htod(&lengths)
            .expect("upload sequence lengths");
        // A missing/partial output write must fail the finite check below.
        let mut output_device = stream
            .clone_htod(&vec![f16::NAN; expected.len()])
            .expect("initialize output sentinel");
        dispatch_paged_attention_v2(
            &stream,
            0,
            &mut output_device,
            &query_device,
            &key_device,
            &value_device,
            &table_device,
            &length_device,
            inputs.sequence_lengths.len(),
            QUERY_HEADS,
            KV_HEADS,
            head_dim,
            BLOCK_SIZE,
            inputs.max_blocks(),
            *inputs.sequence_lengths.iter().max().unwrap(),
        )
        .expect("dispatch native vLLM attention");
        // max_seq_len > 512 necessarily selects the native V2 kernel + reduce,
        // regardless of the optional short-sequence V1 environment setting.
        stream
            .synchronize()
            .expect("native attention CUDA completion");
        let actual = stream
            .clone_dtoh(&output_device)
            .expect("read native output");
        let mut max_absolute_error = 0.0_f64;
        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let actual = f64::from(actual.to_f32());
            assert!(
                actual.is_finite() && expected.is_finite(),
                "H{head_dim} output {index}: actual={actual}, expected={expected}"
            );
            let error = (actual - expected).abs();
            // Allows FP16 probability/tmp-output rounding and FP32 reductions,
            // while testing every element, including references near zero.
            let tolerance = 0.002 + 0.003 * expected.abs();
            assert!(error <= tolerance, "H{head_dim} output {index}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}");
            max_absolute_error = max_absolute_error.max(error);
            squared_error += error * error;
            squared_reference += expected * expected;
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "operation": "native_vllm_paged_attention_v2",
                "head_dim": head_dim,
                "query_heads": QUERY_HEADS,
                "kv_heads": KV_HEADS,
                "sequence_lengths": inputs.sequence_lengths,
                "compared_elements": expected.len(),
                "non_finite": 0,
                "max_absolute_error": max_absolute_error,
                "relative_l2": (squared_error / squared_reference).sqrt(),
            })
        );
    }
}
