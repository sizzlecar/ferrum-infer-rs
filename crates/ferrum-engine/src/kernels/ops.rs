//! Executable attention operations.
//!
//! Unlike `AttentionKernel` (which is metadata/info), `AttentionOp` actually
//! computes attention.  Implementations exist for CPU (reference), Metal, and
//! CUDA (via FlashAttention FFI).
//!
//! # Architecture
//!
//! ```text
//!        AttentionOp (trait)
//!        ┌──────┬──────────┐
//!        │      │          │
//!   CpuAttn  MetalAttn  CudaAttn
//!   (ref)    (.metal)   (FlashAttn FFI)
//! ```
//!
//! Model executors receive an `Arc<dyn AttentionOp>` and call `prefill()` or
//! `decode()` without knowing which backend is active.

use async_trait::async_trait;
use ferrum_types::{Device, Result};

/// Input for a prefill attention operation.
///
/// Prefill processes the entire prompt in one pass.
#[derive(Debug, Clone)]
pub struct PrefillAttentionInput {
    /// Query tensor, flattened `[seq_len, num_heads, head_dim]`.
    pub query: Vec<f32>,
    /// Key tensor, flattened `[seq_len, num_kv_heads, head_dim]`.
    pub key: Vec<f32>,
    /// Value tensor, flattened `[seq_len, num_kv_heads, head_dim]`.
    pub value: Vec<f32>,
    /// Number of query tokens (seq_len).
    pub seq_len: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of KV heads (may differ for GQA).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Whether to apply causal (autoregressive) masking.
    pub causal: bool,
}

/// Input for a decode attention operation.
///
/// Decode processes one new token against cached KV.
/// K/V are pre-gathered by the caller (from paged cache or contiguous buffer).
#[derive(Debug, Clone)]
pub struct DecodeAttentionInput {
    /// Query tensor for the new token, flattened `[1, num_heads, head_dim]`.
    pub query: Vec<f32>,
    /// Pre-gathered cached keys, flattened `[kv_len, num_kv_heads, head_dim]`.
    pub cached_keys: Vec<f32>,
    /// Pre-gathered cached values, flattened `[kv_len, num_kv_heads, head_dim]`.
    pub cached_values: Vec<f32>,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of cached KV tokens to attend over.
    pub kv_len: usize,
}

/// Output from an attention operation.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Output tensor, flattened `[seq_len, num_heads, head_dim]`.
    pub output: Vec<f32>,
}

/// Executable attention operation.
///
/// This is the core kernel interface that model executors use for attention
/// computation.  Backends implement this trait to provide hardware-specific
/// attention (CPU reference, Metal shaders, CUDA FlashAttention).
#[async_trait]
pub trait AttentionOp: Send + Sync + std::fmt::Debug {
    /// Backend name for diagnostics.
    fn name(&self) -> &str;

    /// Which device this backend targets.
    fn device(&self) -> Device;

    /// Compute attention for prefill (full prompt).
    ///
    /// Standard scaled dot-product attention with optional causal masking.
    /// Q, K, V are dense tensors — no paged indirection.
    async fn prefill(&self, input: &PrefillAttentionInput) -> Result<AttentionOutput>;

    /// Compute attention for decode (single new token against paged KV cache).
    ///
    /// K/V are read from the paged block table via the KV cache manager.
    /// Implementors that don't support paged attention should return
    /// `Err(FerrumError::unsupported(...))`.
    async fn decode(&self, input: &DecodeAttentionInput) -> Result<AttentionOutput>;
}

// ────────────────────────────────────────────────────────────────────────────
// CPU reference implementation
// ────────────────────────────────────────────────────────────────────────────

/// CPU reference attention — scalar loops, no SIMD.
///
/// Uses the existing `paged_attention()` kernel from `ferrum-kv` for decode,
/// and inline scaled dot-product for prefill.  Correct but slow.
#[derive(Debug)]
pub struct CpuAttentionOp;

impl CpuAttentionOp {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuAttentionOp {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AttentionOp for CpuAttentionOp {
    fn name(&self) -> &str {
        "cpu-reference"
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    async fn prefill(&self, input: &PrefillAttentionInput) -> Result<AttentionOutput> {
        cpu_prefill_attention(input)
    }

    async fn decode(&self, input: &DecodeAttentionInput) -> Result<AttentionOutput> {
        cpu_decode_attention(input)
    }
}

/// CPU prefill: scaled dot-product attention with causal masking.
fn cpu_prefill_attention(input: &PrefillAttentionInput) -> Result<AttentionOutput> {
    let PrefillAttentionInput {
        query,
        key,
        value,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        causal,
    } = input;

    let seq_len = *seq_len;
    let num_heads = *num_heads;
    let num_kv_heads = *num_kv_heads;
    let head_dim = *head_dim;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let expected_q = seq_len * num_heads * head_dim;
    let expected_kv = seq_len * num_kv_heads * head_dim;
    if query.len() != expected_q {
        return Err(ferrum_types::FerrumError::invalid_parameter(format!(
            "Query length mismatch: expected {expected_q}, got {}",
            query.len()
        )));
    }
    if key.len() != expected_kv || value.len() != expected_kv {
        return Err(ferrum_types::FerrumError::invalid_parameter(
            "K/V length mismatch",
        ));
    }

    let mut output = vec![0.0f32; seq_len * num_heads * head_dim];

    for qt in 0..seq_len {
        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;

            let q_off = (qt * num_heads + h) * head_dim;
            let q = &query[q_off..q_off + head_dim];

            // Compute scores
            let mut scores = Vec::with_capacity(seq_len);
            for kv_pos in 0..seq_len {
                let k_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
                let k = &key[k_off..k_off + head_dim];
                let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }

            // Causal masking
            if *causal {
                for kv_pos in (qt + 1)..seq_len {
                    scores[kv_pos] = f32::NEG_INFINITY;
                }
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
            let out_off = (qt * num_heads + h) * head_dim;
            for kv_pos in 0..seq_len {
                let v_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
                let v = &value[v_off..v_off + head_dim];
                let w = scores[kv_pos];
                for d in 0..head_dim {
                    output[out_off + d] += w * v[d];
                }
            }
        }
    }

    Ok(AttentionOutput { output })
}

/// CPU decode: single-token attention against pre-gathered K/V.
fn cpu_decode_attention(input: &DecodeAttentionInput) -> Result<AttentionOutput> {
    let DecodeAttentionInput {
        query,
        cached_keys,
        cached_values,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
    } = input;

    let num_heads = *num_heads;
    let num_kv_heads = *num_kv_heads;
    let head_dim = *head_dim;
    let kv_len = *kv_len;
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    if kv_len == 0 {
        return Err(ferrum_types::FerrumError::invalid_parameter(
            "kv_len must be positive",
        ));
    }

    let mut output = vec![0.0f32; num_heads * head_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_off = h * head_dim;
        let q = &query[q_off..q_off + head_dim];

        // Compute scores
        let mut scores = Vec::with_capacity(kv_len);
        for kv_pos in 0..kv_len {
            let k_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
            let k = &cached_keys[k_off..k_off + head_dim];
            let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot * scale);
        }

        // Softmax (no causal mask needed — decode attends to all cached tokens)
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
        let out_off = h * head_dim;
        for kv_pos in 0..kv_len {
            let v_off = (kv_pos * num_kv_heads + kv_h) * head_dim;
            let v = &cached_values[v_off..v_off + head_dim];
            let w = scores[kv_pos];
            for d in 0..head_dim {
                output[out_off + d] += w * v[d];
            }
        }
    }

    Ok(AttentionOutput { output })
}

// ────────────────────────────────────────────────────────────────────────────
// CUDA FlashAttention backend (feature-gated)
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub mod cuda_attention {
    use super::*;

    // ── FlashAttention-2 FFI declarations ───────────────────────────────
    //
    // These match the C API exported by the flash-attn library
    // (https://github.com/Dao-AILab/flash-attention).
    //
    // The actual linking happens at build time via:
    //   cargo:rustc-link-lib=flash_attn
    //   cargo:rustc-link-search=<path-to-flash-attn-build>
    //
    // Parameters follow the FlashAttention-2 paper conventions:
    //   - All tensors are device pointers (CUDA memory)
    //   - Strides are in elements, not bytes
    //   - softmax_scale is typically 1/sqrt(head_dim)

    #[allow(non_snake_case)]
    extern "C" {
        /// FlashAttention-2 forward pass (variable-length sequences).
        ///
        /// Computes scaled dot-product attention with online softmax,
        /// supporting causal masking and GQA.
        ///
        /// # Safety
        /// All pointer arguments must be valid CUDA device pointers with
        /// the described shapes.  The CUDA stream must be valid and the
        /// device must be set correctly before calling.
        fn flash_attn_varlen_fwd(
            // Output: [total_q, num_heads, head_dim]
            out: *mut std::ffi::c_void,
            // Query: [total_q, num_heads, head_dim]
            q: *const std::ffi::c_void,
            // Key: [total_k, num_kv_heads, head_dim]
            k: *const std::ffi::c_void,
            // Value: [total_k, num_kv_heads, head_dim]
            v: *const std::ffi::c_void,
            // Cumulative sequence lengths for Q: [batch_size + 1]
            cu_seqlens_q: *const i32,
            // Cumulative sequence lengths for K: [batch_size + 1]
            cu_seqlens_k: *const i32,
            // Maximum sequence length in Q batch
            max_seqlen_q: i32,
            // Maximum sequence length in K batch
            max_seqlen_k: i32,
            // Softmax scale (1/sqrt(head_dim))
            softmax_scale: f32,
            // Whether to apply causal masking
            is_causal: bool,
            // Number of query heads
            num_heads: i32,
            // Number of KV heads (for GQA; == num_heads for MHA)
            num_heads_k: i32,
            // Head dimension
            head_dim: i32,
            // Batch size
            batch_size: i32,
            // CUDA stream
            stream: *mut std::ffi::c_void,
        ) -> i32;

        /// FlashAttention-2 paged attention for decode.
        ///
        /// Reads K/V from a block table (paged KV cache) and computes
        /// single-token attention against cached KV.
        ///
        /// This is the kernel vLLM uses for decode-phase attention.
        fn flash_attn_paged_fwd(
            // Output: [batch_size, num_heads, head_dim]
            out: *mut std::ffi::c_void,
            // Query: [batch_size, num_heads, head_dim]
            q: *const std::ffi::c_void,
            // KV cache: [num_blocks, 2, num_kv_heads, block_size, head_dim]
            kv_cache: *const std::ffi::c_void,
            // Block table: [batch_size, max_blocks_per_seq]
            block_table: *const i32,
            // Sequence lengths: [batch_size]
            seq_lens: *const i32,
            // Softmax scale
            softmax_scale: f32,
            // Number of query heads
            num_heads: i32,
            // Number of KV heads
            num_heads_k: i32,
            // Head dimension
            head_dim: i32,
            // Block size
            block_size: i32,
            // Maximum sequence length across batch
            max_seq_len: i32,
            // Batch size
            batch_size: i32,
            // CUDA stream
            stream: *mut std::ffi::c_void,
        ) -> i32;
    }

    /// CUDA attention backend using FlashAttention-2.
    ///
    /// Requires the `flash-attn` shared library to be available at link time.
    /// The library is typically built from the FlashAttention repository and
    /// installed system-wide or pointed to via `FLASH_ATTN_LIB_DIR`.
    #[derive(Debug)]
    pub struct CudaAttentionOp {
        device_index: usize,
    }

    impl CudaAttentionOp {
        pub fn new(device_index: usize) -> Self {
            Self { device_index }
        }
    }

    #[async_trait]
    impl AttentionOp for CudaAttentionOp {
        fn name(&self) -> &str {
            "cuda-flash-attention-2"
        }

        fn device(&self) -> Device {
            Device::CUDA(self.device_index)
        }

        async fn prefill(&self, input: &PrefillAttentionInput) -> Result<AttentionOutput> {
            // In a real implementation:
            // 1. Copy Q/K/V to GPU memory (or receive device pointers)
            // 2. Build cu_seqlens arrays
            // 3. Call flash_attn_varlen_fwd()
            // 4. Copy output back (or return device pointer)
            //
            // For now this is a compile-time placeholder that proves the
            // FFI linkage compiles.  Real GPU tensor management will be
            // added when CUDA hardware is available for testing.
            let _ = input;
            Err(ferrum_types::FerrumError::unsupported(
                "CudaAttentionOp::prefill not yet wired to GPU tensor management",
            ))
        }

        async fn decode(&self, input: &DecodeAttentionInput) -> Result<AttentionOutput> {
            let _ = input;
            Err(ferrum_types::FerrumError::unsupported(
                "CudaAttentionOp::decode not yet wired to GPU tensor management",
            ))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Attention dispatch — select backend by device
// ────────────────────────────────────────────────────────────────────────────

/// Creates the appropriate `AttentionOp` for the given device.
///
/// - `Device::CPU` → `CpuAttentionOp`
/// - `Device::CUDA(n)` → `CudaAttentionOp` (requires `cuda` feature)
/// - Others → error
pub fn create_attention_op(device: &Device) -> Result<Box<dyn AttentionOp>> {
    match device {
        Device::CPU => Ok(Box::new(CpuAttentionOp::new())),

        #[cfg(feature = "cuda")]
        Device::CUDA(idx) => Ok(Box::new(cuda_attention::CudaAttentionOp::new(*idx))),

        #[cfg(not(feature = "cuda"))]
        Device::CUDA(_) => Err(ferrum_types::FerrumError::unsupported(
            "CUDA attention requires the 'cuda' feature flag",
        )),

        _ => Err(ferrum_types::FerrumError::unsupported(format!(
            "No attention backend for device {:?}",
            device
        ))),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cpu_prefill_self_attention() {
        // 2 tokens, 1 head, head_dim=2, causal
        // Token 0: Q=K=V=[1,0]
        // Token 1: Q=K=V=[0,1]
        let input = PrefillAttentionInput {
            query: vec![1.0, 0.0, 0.0, 1.0],
            key: vec![1.0, 0.0, 0.0, 1.0],
            value: vec![1.0, 0.0, 0.0, 1.0],
            seq_len: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            causal: true,
        };

        let op = CpuAttentionOp::new();
        let out = op.prefill(&input).await.unwrap();

        // Token 0 can only see itself → output = V0 = [1, 0]
        assert!((out.output[0] - 1.0).abs() < 1e-5);
        assert!((out.output[1] - 0.0).abs() < 1e-5);

        // Token 1 sees tokens 0 and 1
        // Q1=[0,1], K0=[1,0]→dot=0, K1=[0,1]→dot=1
        // After softmax (scale=1/sqrt(2)): token 1 gets more weight
        // Output leans toward V1=[0,1]
        assert!(out.output[2] < 0.5, "Expected <0.5, got {}", out.output[2]);
        assert!(out.output[3] > 0.5, "Expected >0.5, got {}", out.output[3]);
    }

    #[tokio::test]
    async fn cpu_prefill_non_causal() {
        // 2 tokens, 1 head, head_dim=2, non-causal
        // Both tokens attend to everything
        let input = PrefillAttentionInput {
            query: vec![1.0, 1.0, 1.0, 1.0],
            key: vec![1.0, 0.0, 0.0, 1.0],
            value: vec![1.0, 0.0, 0.0, 1.0],
            seq_len: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            causal: false,
        };

        let op = CpuAttentionOp::new();
        let out = op.prefill(&input).await.unwrap();

        // Q=[1,1] for both tokens, K0=[1,0]→dot=1, K1=[0,1]→dot=1
        // Equal attention → output = mean(V0, V1) = [0.5, 0.5]
        for &v in &out.output {
            assert!((v - 0.5).abs() < 1e-5, "Expected 0.5, got {}", v);
        }
    }

    #[tokio::test]
    async fn cpu_prefill_gqa() {
        // 4 query heads, 2 KV heads → heads_per_kv = 2
        // 1 token, head_dim=2
        let input = PrefillAttentionInput {
            query: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
            key: vec![1.0, 0.0, 0.0, 1.0],
            value: vec![2.0, 3.0, 4.0, 5.0],
            seq_len: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
            causal: true,
        };

        let op = CpuAttentionOp::new();
        let out = op.prefill(&input).await.unwrap();
        assert_eq!(out.output.len(), 4 * 2);

        // With 1 token, softmax is trivially 1.0
        // Head 0 (kv_head 0): output = V0 = [2, 3]
        assert!((out.output[0] - 2.0).abs() < 1e-5);
        assert!((out.output[1] - 3.0).abs() < 1e-5);
        // Head 1 (kv_head 0): output = V0 = [2, 3]
        assert!((out.output[2] - 2.0).abs() < 1e-5);
        assert!((out.output[3] - 3.0).abs() < 1e-5);
        // Head 2 (kv_head 1): output = V1 = [4, 5]
        assert!((out.output[4] - 4.0).abs() < 1e-5);
        assert!((out.output[5] - 5.0).abs() < 1e-5);
        // Head 3 (kv_head 1): output = V1 = [4, 5]
        assert!((out.output[6] - 4.0).abs() < 1e-5);
        assert!((out.output[7] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn cpu_decode_single_token() {
        // Query = [1,1], cached KV = 3 tokens with K=V=[1,0], [0,1], [1,1]
        let input = DecodeAttentionInput {
            query: vec![1.0, 1.0],
            cached_keys: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            cached_values: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            kv_len: 3,
        };

        let op = CpuAttentionOp::new();
        let out = op.decode(&input).await.unwrap();

        // Q=[1,1]: K0=[1,0]→1, K1=[0,1]→1, K2=[1,1]→2
        // After scale (1/sqrt(2)) and softmax, K2 gets highest weight
        // Output should lean toward V2=[1,1]
        assert!(out.output[0] > 0.3, "Expected >0.3, got {}", out.output[0]);
        assert!(out.output[1] > 0.3, "Expected >0.3, got {}", out.output[1]);
    }

    #[test]
    fn create_cpu_attention_op() {
        let op = create_attention_op(&Device::CPU).unwrap();
        assert_eq!(op.name(), "cpu-reference");
        assert_eq!(op.device(), Device::CPU);
    }

    #[test]
    fn create_cuda_attention_op_without_feature() {
        // On a non-CUDA build, requesting CUDA should fail gracefully
        #[cfg(not(feature = "cuda"))]
        {
            let result = create_attention_op(&Device::CUDA(0));
            assert!(result.is_err());
        }
    }
}
