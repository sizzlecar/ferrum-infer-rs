//! Debug: test gemm_v2 with actual Qwen3-0.6B dimensions
#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::metal::MetalBackend;
use ferrum_kernels::backend::Backend;

fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn test_gemm_size(name: &str, m: usize, n: usize, k: usize) {
    let a = pseudo_random(42, m * k);
    let b = pseudo_random(43, n * k);

    // CPU reference
    let mut cpu_out = CpuBackend::alloc(m * n);
    CpuBackend::gemm(&mut (), &a, &b, &mut cpu_out, m, n, k);

    // Metal
    let a_m = MetalBackend::from_slice(&a);
    let b_m = MetalBackend::from_slice(&b);
    let mut metal_out = MetalBackend::alloc(m * n);
    let mut ctx = MetalBackend::new_context();
    MetalBackend::gemm(&mut ctx, &a_m, &b_m, &mut metal_out, m, n, k);
    MetalBackend::sync(&mut ctx);

    let mv = MetalBackend::to_vec(&metal_out, m * n);
    let max_diff = cpu_out
        .iter()
        .zip(&mv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_max = cpu_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel = if cpu_max > 0.0 {
        max_diff / cpu_max
    } else {
        max_diff
    };

    eprintln!(
        "{name}: {m}x{n}x{k}  max_diff={max_diff:.2e}  rel={rel:.2e}  cpu_max={cpu_max:.2e}  {}",
        if rel < 1e-3 { "✅" } else { "❌" }
    );
    assert!(rel < 1e-3, "{name} GEMM mismatch: rel={rel}");
}

#[test]
fn test_qwen3_gemm_sizes() {
    test_gemm_size("QKV", 1, 4096, 1024);
    test_gemm_size("O-proj", 1, 1024, 2048);
    test_gemm_size("gate_up", 1, 6144, 1024);
    test_gemm_size("down", 1, 1024, 3072);
    test_gemm_size("lm_head", 1, 151936, 1024);
}
