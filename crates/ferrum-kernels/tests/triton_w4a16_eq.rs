//! Equivalence test: triton-rs `launch_w4a16_gptq_triton` matches a CPU
//! reference implementation of the GPTQ INT4×FP16 GEMM on a small tile.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_w4a16_eq -- --nocapture
//!
//! Skipped (no tests compiled) without the feature combo.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use cudarc::driver::{CudaContext, CudaSlice};
use ferrum_kernels::triton_w4a16::{launch_w4a16_gptq_triton, TritonGptqWeight};

// Two-test layout: small (K=256, fast iteration) and "real" (K=2048, N=2560,
// matching qkv_proj shapes from Qwen2.5-3B). Both share the same code path —
// only constants differ.
const M: usize = 32;
const K: usize = 256;
const N: usize = 64;
const G: usize = 64; // group_size — divides K; ≥ BK=32 (the kernel's K-tile)

// Real Qwen2.5-3B qkv_proj shape: K=2048, N=2560 (q+k+v fused), group_size=128.
const M2: usize = 10; // simulated prefill batch
const K2: usize = 2048;
const N2: usize = 2560;
const G2: usize = 128;

fn build_gptq_tensors() -> (Vec<half::f16>, Vec<i32>, Vec<half::f16>, Vec<i32>) {
    use half::f16;

    // ── A: [M, K] f16 ──
    let a: Vec<f16> = (0..M * K)
        .map(|i| f16::from_f32(((i as f32) * 0.0017).sin()))
        .collect();

    // ── INT4 weights, range 0..15. Pack 8 K-rows per int32. ──
    // Generate a [K, N] grid of int4 values, then pack.
    let w_int: Vec<u8> = (0..K * N).map(|i| ((i * 7 + 3) % 16) as u8).collect();
    let mut qw: Vec<i32> = vec![0; (K / 8) * N];
    for pk in 0..K / 8 {
        for n in 0..N {
            let mut packed: u32 = 0;
            for i in 0..8 {
                let v = w_int[(pk * 8 + i) * N + n] as u32;
                packed |= v << (i * 4);
            }
            qw[pk * N + n] = packed as i32;
        }
    }

    // ── Scales: [K/G, N] f16, small positive values ──
    let scales: Vec<f16> = (0..(K / G) * N)
        .map(|i| f16::from_f32(0.01 + ((i as f32) * 0.0007).cos() * 0.005))
        .collect();

    // ── Zeros: int4 in [K/G, N], packed [K/G, N/8] along N ──
    let z_int: Vec<u8> = (0..(K / G) * N)
        .map(|i| ((i * 11 + 5) % 16) as u8)
        .collect();
    let mut qz: Vec<i32> = vec![0; (K / G) * (N / 8)];
    for kg in 0..K / G {
        for pn in 0..N / 8 {
            let mut packed: u32 = 0;
            for j in 0..8 {
                let v = z_int[kg * N + pn * 8 + j] as u32;
                packed |= v << (j * 4);
            }
            qz[kg * (N / 8) + pn] = packed as i32;
        }
    }

    (a, qw, scales, qz)
}

/// Mirror the kernel formula on host: dequant + GEMM in f32 → truncate to f16.
fn cpu_reference_gemm(
    a: &[half::f16],
    qw: &[i32],
    scales: &[half::f16],
    qz: &[i32],
) -> Vec<half::f16> {
    use half::f16;

    let mut deq = vec![0f32; K * N];
    for k in 0..K {
        for n in 0..N {
            let pk = k / 8;
            let shift = (k % 8) * 4;
            let qw_v = qw[pk * N + n] as u32;
            let nibble = ((qw_v >> shift) & 0xF) as i32;

            let kg = k / G;
            let pn = n / 8;
            let z_shift = (n % 8) * 4;
            let qz_v = qz[kg * (N / 8) + pn] as u32;
            // AutoGPTQ stores qzero = actual_zero - 1; recover with +1.
            let zero = (((qz_v >> z_shift) & 0xF) as i32) + 1;

            let scale = scales[kg * N + n].to_f32();
            deq[k * N + n] = (nibble - zero) as f32 * scale;
        }
    }

    let mut c = vec![f16::from_f32(0.0); M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += a[m * K + k].to_f32() * deq[k * N + n];
            }
            c[m * N + n] = f16::from_f32(acc);
        }
    }
    c
}

#[test]
fn triton_w4a16_matches_cpu_reference() {
    use half::f16;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    let (a, qw, scales, qz) = build_gptq_tensors();
    let c_ref = cpu_reference_gemm(&a, &qw, &scales, &qz);

    // Upload inputs.
    let a_dev: CudaSlice<f16> = stream.clone_htod(&a).unwrap();
    let qw_dev: CudaSlice<i32> = stream.clone_htod(&qw).unwrap();
    let sc_dev: CudaSlice<f16> = stream.clone_htod(&scales).unwrap();
    let qz_dev: CudaSlice<i32> = stream.clone_htod(&qz).unwrap();
    let mut c_dev: CudaSlice<f16> = stream.alloc_zeros::<f16>(M * N).unwrap();

    let weight = TritonGptqWeight {
        qweight: qw_dev,
        scales: sc_dev,
        qzeros: qz_dev,
        k: K,
        n: N,
        group_size: G as i32,
    };

    // Manually load the function (mirrors the dispatcher).
    let func = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(
            ferrum_kernels::triton_w4a16::W4A16_PTX.to_string(),
        ))
        .unwrap()
        .load_function(ferrum_kernels::triton_w4a16::fn_name())
        .unwrap();

    launch_w4a16_gptq_triton(&stream, &func, &a_dev, &weight, &mut c_dev, M as i32)
        .expect("triton w4a16 launch failed");
    stream.synchronize().expect("sync");

    let c_gpu: Vec<f16> = stream.memcpy_dtov(&c_dev).unwrap();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut argmax = 0usize;
    for (i, (g, r)) in c_gpu.iter().zip(c_ref.iter()).enumerate() {
        let g32 = g.to_f32();
        let r32 = r.to_f32();
        let diff = (g32 - r32).abs();
        if diff > max_abs {
            max_abs = diff;
            argmax = i;
        }
        let denom = g32.abs().max(r32.abs()).max(1e-6);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }

    println!(
        "w4a16 eq: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
         (worst @ idx {argmax}: gpu={:.6} ref={:.6})",
        c_gpu[argmax].to_f32(),
        c_ref[argmax].to_f32()
    );

    // Tolerance: f16 GEMM with f32 accumulation; ~K=256 fp16 mults per
    // output. Worst-case ULPs accumulate but should stay well under 1e-1.
    assert!(
        max_abs < 5e-2,
        "max_abs {max_abs:.3e} exceeds tolerance 5e-2 — kernel diverges"
    );
}

/// Same equivalence check at a "production" tile shape: M=10 prefill rows,
/// K=2048, N=2560, G=128 — i.e. Qwen2.5-3B-Instruct-GPTQ-Int4's `qkv_proj`.
/// This is the exact configuration that broke prefill in the live `bench`
/// run on Blackwell, so passing here means the kernel itself is fine and
/// any remaining divergence is in the dispatch / load wiring.
fn build_real_gptq_tensors() -> (Vec<half::f16>, Vec<i32>, Vec<half::f16>, Vec<i32>) {
    use half::f16;
    let a: Vec<f16> = (0..M2 * K2)
        .map(|i| f16::from_f32(((i as f32) * 0.0017).sin() * 0.1))
        .collect();

    let w_int: Vec<u8> = (0..K2 * N2).map(|i| ((i * 7 + 3) % 16) as u8).collect();
    let mut qw: Vec<i32> = vec![0; (K2 / 8) * N2];
    for pk in 0..K2 / 8 {
        for n in 0..N2 {
            let mut packed: u32 = 0;
            for i in 0..8 {
                packed |= (w_int[(pk * 8 + i) * N2 + n] as u32) << (i * 4);
            }
            qw[pk * N2 + n] = packed as i32;
        }
    }

    let scales: Vec<f16> = (0..(K2 / G2) * N2)
        .map(|i| f16::from_f32(0.005 + ((i as f32) * 0.0007).cos() * 0.002))
        .collect();

    let z_int: Vec<u8> = (0..(K2 / G2) * N2)
        .map(|i| ((i * 11 + 5) % 16) as u8)
        .collect();
    let mut qz: Vec<i32> = vec![0; (K2 / G2) * (N2 / 8)];
    for kg in 0..K2 / G2 {
        for pn in 0..N2 / 8 {
            let mut packed: u32 = 0;
            for j in 0..8 {
                packed |= (z_int[kg * N2 + pn * 8 + j] as u32) << (j * 4);
            }
            qz[kg * (N2 / 8) + pn] = packed as i32;
        }
    }

    (a, qw, scales, qz)
}

fn cpu_reference_gemm_real(
    a: &[half::f16],
    qw: &[i32],
    scales: &[half::f16],
    qz: &[i32],
) -> Vec<half::f16> {
    use half::f16;
    let mut deq = vec![0f32; K2 * N2];
    for k in 0..K2 {
        for n in 0..N2 {
            let pk = k / 8;
            let shift = (k % 8) * 4;
            let qw_v = qw[pk * N2 + n] as u32;
            let nibble = ((qw_v >> shift) & 0xF) as i32;

            let kg = k / G2;
            let pn = n / 8;
            let z_shift = (n % 8) * 4;
            let qz_v = qz[kg * (N2 / 8) + pn] as u32;
            // AutoGPTQ stores qzero = actual_zero - 1; recover with +1.
            let zero = (((qz_v >> z_shift) & 0xF) as i32) + 1;

            let scale = scales[kg * N2 + n].to_f32();
            deq[k * N2 + n] = (nibble - zero) as f32 * scale;
        }
    }

    let mut c = vec![f16::from_f32(0.0); M2 * N2];
    for m in 0..M2 {
        for n in 0..N2 {
            let mut acc = 0f32;
            for k in 0..K2 {
                acc += a[m * K2 + k].to_f32() * deq[k * N2 + n];
            }
            c[m * N2 + n] = f16::from_f32(acc);
        }
    }
    c
}

#[test]
fn triton_w4a16_matches_cpu_reference_qkv_shape() {
    use half::f16;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    let (a, qw, scales, qz) = build_real_gptq_tensors();
    let c_ref = cpu_reference_gemm_real(&a, &qw, &scales, &qz);

    let a_dev: CudaSlice<f16> = stream.clone_htod(&a).unwrap();
    let qw_dev: CudaSlice<i32> = stream.clone_htod(&qw).unwrap();
    let sc_dev: CudaSlice<f16> = stream.clone_htod(&scales).unwrap();
    let qz_dev: CudaSlice<i32> = stream.clone_htod(&qz).unwrap();
    let mut c_dev: CudaSlice<f16> = stream.alloc_zeros::<f16>(M2 * N2).unwrap();

    let weight = TritonGptqWeight {
        qweight: qw_dev,
        scales: sc_dev,
        qzeros: qz_dev,
        k: K2,
        n: N2,
        group_size: G2 as i32,
    };

    let func = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(
            ferrum_kernels::triton_w4a16::W4A16_PTX.to_string(),
        ))
        .unwrap()
        .load_function(ferrum_kernels::triton_w4a16::fn_name())
        .unwrap();

    launch_w4a16_gptq_triton(&stream, &func, &a_dev, &weight, &mut c_dev, M2 as i32)
        .expect("launch failed");
    stream.synchronize().expect("sync");

    let c_gpu: Vec<f16> = stream.memcpy_dtov(&c_dev).unwrap();

    let mut max_abs = 0f32;
    let mut argmax = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for (i, (g, _)) in c_gpu.iter().zip(c_ref.iter()).enumerate() {
        let g32 = g.to_f32();
        if g32.is_nan() {
            nan_count += 1;
            continue;
        }
        if !g32.is_finite() {
            inf_count += 1;
            continue;
        }
        let r32 = c_ref[i].to_f32();
        let diff = (g32 - r32).abs();
        if diff > max_abs {
            max_abs = diff;
            argmax = i;
        }
    }

    println!(
        "qkv shape eq: max_abs={max_abs:.3e}  nan_count={nan_count}  inf_count={inf_count}  \
         (worst @ idx {argmax}: gpu={:.3} ref={:.3})",
        c_gpu[argmax].to_f32(),
        c_ref[argmax].to_f32()
    );

    assert_eq!(
        nan_count, 0,
        "kernel produced NaN — bug in dispatch/wiring at scale"
    );
    assert_eq!(
        inf_count, 0,
        "kernel produced Inf — bug in dispatch/wiring at scale"
    );

    // K=2048 f16 mults — accumulated f16 truncation error scales with sqrt(K)
    // for random-sign products. Tolerance 1.0 is loose but catches order-of-
    // magnitude divergence; tightening would require exact-bit identical
    // accumulation order, which we don't have between the GPU dot and CPU.
    assert!(
        max_abs < 1.0,
        "max_abs {max_abs:.3e} exceeds tolerance 1.0 — kernel diverges at scale"
    );
}
