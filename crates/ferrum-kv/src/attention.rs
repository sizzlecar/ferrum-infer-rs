//! CPU reference implementation of paged attention.
//!
//! Computes scaled dot-product attention where K/V are gathered from
//! non-contiguous physical blocks via a PagedKvCacheManager.
//!
//! This is **not** a high-performance kernel — it exists for correctness
//! testing and as a reference for GPU kernel implementations.

use crate::managers::paged::{PagedKvCacheHandle, PagedKvCacheManager};
use ferrum_types::{FerrumError, Result};

/// Compute scaled dot-product attention for a single layer, reading
/// cached K/V through the paged block table.
///
/// # Arguments
/// * `query` — query vectors, shape `[q_tokens, num_heads, head_dim]` flattened
///   row-major.  For decode, `q_tokens = 1`.
/// * `q_tokens` — number of query tokens.
/// * `num_heads` — number of attention heads (query heads; may differ from KV
///   heads in GQA).
/// * `num_kv_heads` — number of KV heads (for GQA repeat).
/// * `head_dim` — dimension per head.
/// * `manager` — the paged cache manager holding K/V data.
/// * `handle` — the per-request cache handle with block table.
/// * `layer` — transformer layer index.
/// * `kv_len` — number of cached K/V tokens to attend over (0..kv_len).
///
/// # Returns
/// Attention output flattened as `[q_tokens, num_heads, head_dim]`.
pub fn paged_attention(
    query: &[f32],
    q_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    manager: &PagedKvCacheManager,
    handle: &PagedKvCacheHandle,
    layer: usize,
    kv_len: usize,
) -> Result<Vec<f32>> {
    if query.len() != q_tokens * num_heads * head_dim {
        return Err(FerrumError::invalid_parameter(format!(
            "Query length mismatch: expected {}, got {}",
            q_tokens * num_heads * head_dim,
            query.len()
        )));
    }
    if kv_len == 0 {
        return Err(FerrumError::invalid_parameter(
            "kv_len must be positive",
        ));
    }

    // GQA: number of query heads per KV head
    let heads_per_kv = num_heads / num_kv_heads;

    // Gather all cached K/V for this layer: shapes [kv_len, num_kv_heads, head_dim]
    let (all_keys, all_values) = manager.read_kv(handle, layer, 0, kv_len)?;
    let kv_head_stride = head_dim;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; q_tokens * num_heads * head_dim];

    for qt in 0..q_tokens {
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv; // GQA: which KV head this query head uses

            // Q vector for this (qt, h): query[qt * num_heads * head_dim + h * head_dim ..]
            let q_offset = (qt * num_heads + h) * head_dim;
            let q = &query[q_offset..q_offset + head_dim];

            // Compute attention scores: Q·K^T / sqrt(d)
            let mut scores = Vec::with_capacity(kv_len);
            for kv_pos in 0..kv_len {
                let k_offset = (kv_pos * num_kv_heads + kv_h) * kv_head_stride;
                let k = &all_keys[k_offset..k_offset + head_dim];
                let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }

            // Causal masking: for prefill, query position qt can only attend to
            // positions 0..=(kv_len - q_tokens + qt).  For decode (q_tokens=1),
            // all positions are visible.
            let max_visible = kv_len - q_tokens + qt;
            for kv_pos in (max_visible + 1)..kv_len {
                scores[kv_pos] = f32::NEG_INFINITY;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in &mut scores {
                    *s /= sum;
                }
            }

            // Weighted sum of V
            let out_offset = (qt * num_heads + h) * head_dim;
            for kv_pos in 0..kv_len {
                let v_offset = (kv_pos * num_kv_heads + kv_h) * kv_head_stride;
                let v = &all_values[v_offset..v_offset + head_dim];
                let w = scores[kv_pos];
                for d in 0..head_dim {
                    output[out_offset + d] += w * v[d];
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::managers::paged::PagedKvCacheConfig;
    use ferrum_interfaces::kv_cache::AllocationRequest;
    use ferrum_interfaces::KvCacheManager;
    use ferrum_types::{DataType, Device, RequestId};

    /// Helper: create a small manager and handle for testing.
    async fn setup(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        initial_tokens: usize,
    ) -> (PagedKvCacheManager, RequestId) {
        let config = PagedKvCacheConfig {
            block_size,
            max_gpu_blocks: 64,
            max_cpu_blocks: 0,
            enable_cow: false,
            enable_swapping: false,
            num_layers,
            num_heads,
            head_dim,
            enable_prefix_cache: false,
            ..Default::default()
        };
        let manager = PagedKvCacheManager::new(Device::CPU, config).unwrap();

        let request = AllocationRequest {
            request_id: RequestId::new(),
            initial_tokens,
            max_sequence_length: 1024,
            num_layers,
            num_heads,
            head_dim,
            device: Device::CPU,
            dtype: DataType::FP16,
            priority: ferrum_types::Priority::Normal,
        };
        let rid = request.request_id.clone();
        let _ = manager.allocate(&request).await.unwrap();
        (manager, rid)
    }

    #[tokio::test]
    async fn single_token_decode_attention() {
        let num_heads = 2;
        let head_dim = 4;
        let (manager, rid) = setup(1, num_heads, head_dim, 16, 3).await;
        let handle_dyn = manager.get_handle(rid.clone()).unwrap();
        let handle = handle_dyn.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();

        // Store K/V for 3 cached tokens in layer 0
        let kv_size = num_heads * head_dim; // 8
        for pos in 0..3 {
            let key = vec![1.0f32; kv_size];
            let val = vec![(pos + 1) as f32; kv_size];
            manager.write_kv(handle, 0, pos, &key, &val).unwrap();
        }

        // Query: all ones → dot product with each key = head_dim = 4
        // All scores equal → uniform attention → output = mean of values
        let query = vec![1.0f32; num_heads * head_dim];
        let output = paged_attention(
            &query, 1, num_heads, num_heads, head_dim, &manager, handle, 0, 3,
        )
        .unwrap();

        assert_eq!(output.len(), num_heads * head_dim);

        // Mean of values: (1 + 2 + 3) / 3 = 2.0
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-5, "Expected ~2.0, got {}", v);
        }
    }

    #[tokio::test]
    async fn prefill_causal_masking() {
        let num_heads = 1;
        let head_dim = 2;
        let (manager, rid) = setup(1, num_heads, head_dim, 16, 3).await;
        let handle_dyn = manager.get_handle(rid.clone()).unwrap();
        let handle = handle_dyn.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();

        let kv_size = num_heads * head_dim; // 2

        // Token 0: K=[1,0], V=[1,0]
        // Token 1: K=[0,1], V=[0,1]
        // Token 2: K=[1,1], V=[1,1]
        let keys_data = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let vals_data = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        for pos in 0..3 {
            manager
                .write_kv(handle, 0, pos, &keys_data[pos], &vals_data[pos])
                .unwrap();
        }

        // Prefill with 3 query tokens: Q=K (self-attention style)
        let mut query = Vec::with_capacity(3 * kv_size);
        for pos in 0..3 {
            query.extend_from_slice(&keys_data[pos]);
        }

        let output = paged_attention(
            &query, 3, num_heads, num_heads, head_dim, &manager, handle, 0, 3,
        )
        .unwrap();

        assert_eq!(output.len(), 3 * kv_size);

        // Token 0 (q_pos=0) can only see token 0
        // Q=[1,0], K0=[1,0] → dot=1, only one token → softmax=1 → output=V0=[1,0]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5);

        // Token 1 (q_pos=1) can see tokens 0,1
        // Q=[0,1], K0=[1,0]→0, K1=[0,1]→1
        // softmax([0/sqrt(2), 1/sqrt(2)]) → token 1 gets higher weight
        // Output should lean toward V1=[0,1]
        assert!(output[2] < 0.5, "Expected output[2] < 0.5, got {}", output[2]);
        assert!(output[3] > 0.5, "Expected output[3] > 0.5, got {}", output[3]);
    }

    #[tokio::test]
    async fn attention_across_blocks() {
        // block_size=2, 4 tokens → 2 blocks
        let num_heads = 1;
        let head_dim = 2;
        let (manager, rid) = setup(1, num_heads, head_dim, 2, 4).await;
        let handle_dyn = manager.get_handle(rid.clone()).unwrap();
        let handle = handle_dyn.as_any().downcast_ref::<PagedKvCacheHandle>().unwrap();

        // Write 4 tokens across 2 blocks
        for pos in 0..4 {
            let key = vec![(pos + 1) as f32; head_dim];
            let val = vec![(pos + 1) as f32 * 10.0; head_dim];
            manager.write_kv(handle, 0, pos, &key, &val).unwrap();
        }

        // Decode query attending to all 4 tokens
        let query = vec![1.0f32; head_dim];
        let output = paged_attention(
            &query, 1, num_heads, num_heads, head_dim, &manager, handle, 0, 4,
        )
        .unwrap();

        // All K are proportional to [1,1], so Q·K scores differ by magnitude
        // but softmax weights favor later tokens (higher dot products).
        // Output should be a weighted average of V values.
        assert_eq!(output.len(), head_dim);
        // Sum of V is (10+20+30+40)*w_i — should be between 10 and 40
        assert!(output[0] > 10.0 && output[0] < 40.0);
    }
}
