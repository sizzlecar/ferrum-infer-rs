//! CPU fused attention using Accelerate cblas_sgemm + single-pass softmax.

use crate::AttentionParams;

// ── BLAS binding ────────────────────────────────────────────────────��───

#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

/// C[m,n] = A[m,k] @ B[n,k]^T  (row-major)
#[cfg(target_os = "macos")]
fn gemm_at_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        cblas_sgemm(
            101, 111, 112, // RowMajor, NoTrans, Trans
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), k as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn gemm_at_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Portable fallback with f64 accumulation
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * b[j * k + p] as f64;
            }
            c[i * n + j] = sum as f32;
        }
    }
}

/// C[m,n] = A[m,k] @ B[k,n]  (row-major, B not transposed)
#[cfg(target_os = "macos")]
fn gemm_at_b(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        cblas_sgemm(
            101, 111, 111, // RowMajor, NoTrans, NoTrans
            m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0, c.as_mut_ptr(), n as i32,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn gemm_at_b(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = sum as f32;
        }
    }
}

// ── Fused softmax (single pass, f64 accumulator) ────────────────────────

/// In-place softmax on rows of scores[m, n], with causal mask.
/// For causal: row i can attend to columns 0..=pos_offset+i.
fn softmax_inplace(scores: &mut [f32], m: usize, n: usize, causal: bool, pos_offset: usize) {
    for i in 0..m {
        let row = &mut scores[i * n..(i + 1) * n];
        let attend_len = if causal { (pos_offset + i + 1).min(n) } else { n };

        // Mask future positions
        for j in attend_len..n {
            row[j] = f32::NEG_INFINITY;
        }

        // Max
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..attend_len {
            if row[j] > max_val { max_val = row[j]; }
        }

        // Exp + sum (f64 accumulator)
        let mut sum = 0.0f64;
        for j in 0..n {
            let e = (row[j] - max_val).exp();
            row[j] = e;
            sum += e as f64;
        }

        // Normalize
        let inv = (1.0 / sum) as f32;
        for j in 0..n {
            row[j] *= inv;
        }
    }
}

// ── Main entry point ────────────────────────────────────────────────────

pub fn fused_attention(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    p: &AttentionParams,
) {
    let nh = p.num_heads;
    let nkv = p.num_kv_heads;
    let n_rep = nh / nkv;
    let sq = p.q_len;
    let sk = p.kv_len;
    let d = p.head_dim;
    let scale = 1.0f32 / (d as f32).sqrt();

    for b in 0..p.batch {
        for h in 0..nh {
            let kv_h = h / n_rep; // GQA: which KV head

            let q_off = (b * nh + h) * sq * d;
            let k_off = (b * nkv + kv_h) * sk * d;
            let v_off = (b * nkv + kv_h) * sk * d;
            let o_off = (b * nh + h) * sq * d;

            let q_slice = &q[q_off..q_off + sq * d];
            let k_slice = &k[k_off..k_off + sk * d];
            let v_slice = &v[v_off..v_off + sk * d];

            // scores[sq, sk] = Q[sq, d] @ K[sk, d]^T * scale
            let mut scores = vec![0.0f32; sq * sk];
            gemm_at_bt(q_slice, k_slice, &mut scores, sq, sk, d);
            for s in scores.iter_mut() { *s *= scale; }

            // Fused softmax with causal mask
            softmax_inplace(&mut scores, sq, sk, p.causal, p.pos_offset);

            // out[sq, d] = scores[sq, sk] @ V[sk, d]
            let o_slice = &mut out[o_off..o_off + sq * d];
            gemm_at_b(&scores, v_slice, o_slice, sq, d, sk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_attention_causal() {
        // 1 batch, 1 head, 2 tokens, dim=4
        let q = vec![1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 8];

        let params = AttentionParams {
            batch: 1, num_heads: 1, num_kv_heads: 1,
            q_len: 2, kv_len: 2, head_dim: 4,
            causal: true, pos_offset: 0,
        };

        fused_attention(&q, &k, &v, &mut out, &params);

        // Token 0 only attends to token 0
        assert!((out[0] - 1.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-4);

        // Token 1 attends to both tokens
        // scores: [q1·k0, q1·k1] * scale = [0, 1] * 0.5 = [0, 0.5]
        // softmax([0, 0.5]) ≈ [0.3775, 0.6225]
        let e0 = (-0.5f32).exp();
        let w0 = e0 / (e0 + 1.0);
        let w1 = 1.0 / (e0 + 1.0);
        let expected = w0 * 1.0 + w1 * 5.0;
        assert!((out[4] - expected).abs() < 1e-3, "out[4]={} expected={}", out[4], expected);
    }

    #[test]
    fn test_cpu_attention_gqa() {
        // 1 batch, 2 Q heads, 1 KV head, 1 token, dim=2
        let q = vec![1.0, 0.0,  0.0, 1.0]; // 2 heads
        let k = vec![1.0, 0.0]; // 1 head
        let v = vec![3.0, 7.0]; // 1 head
        let mut out = vec![0.0f32; 4];

        let params = AttentionParams {
            batch: 1, num_heads: 2, num_kv_heads: 1,
            q_len: 1, kv_len: 1, head_dim: 2,
            causal: false, pos_offset: 0,
        };

        fused_attention(&q, &k, &v, &mut out, &params);

        // Both heads attend to the single KV, softmax of single element = 1.0
        // out = V for both heads
        assert!((out[0] - 3.0).abs() < 1e-4);
        assert!((out[1] - 7.0).abs() < 1e-4);
        assert!((out[2] - 3.0).abs() < 1e-4);
        assert!((out[3] - 7.0).abs() < 1e-4);
    }
}
