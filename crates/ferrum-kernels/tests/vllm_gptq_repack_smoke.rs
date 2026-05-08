//! Smoke test: vLLM gptq_marlin_repack runs without error and produces
//! output of expected shape. Doesn't check correctness yet — that needs
//! a forward GEMM call which arrives in the next stage.

#![cfg(all(feature = "cuda", feature = "vllm-marlin"))]

use cudarc::driver::{CudaContext, CudaSlice};
use ferrum_kernels::vllm_marlin::vllm_gptq_marlin_repack;

#[test]
#[ignore]
fn vllm_gptq_marlin_repack_runs() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Same shape as Qwen3-MoE per-expert gate_up.
    const K: i32 = 2048;
    const N: i32 = 1536;

    // Input: [K/8, N] i32. Random GPTQ on-disk packed.
    let qw_host: Vec<i32> = (0..(K as usize / 8) * N as usize)
        .map(|i| (i as i32).wrapping_mul(0x9E3779B1))
        .collect();
    let qw_in: CudaSlice<i32> = stream.clone_htod(&qw_host).unwrap();

    // Output: same total size.
    let mut qw_out: CudaSlice<i32> = stream.alloc_zeros::<i32>(qw_host.len()).unwrap();

    vllm_gptq_marlin_repack(&stream, &qw_in, &mut qw_out, K, N).expect("repack");
    stream.synchronize().unwrap();

    let qw_out_host: Vec<i32> = stream.memcpy_dtov(&qw_out).unwrap();
    let nonzero = qw_out_host.iter().filter(|&&x| x != 0).count();
    eprintln!(
        "repack done: {} elements out, {} non-zero ({:.1}%)",
        qw_out_host.len(),
        nonzero,
        nonzero as f32 / qw_out_host.len() as f32 * 100.0
    );
    assert!(
        nonzero > qw_out_host.len() / 4,
        "repack output suspicious: only {nonzero} non-zero out of {}",
        qw_out_host.len()
    );
}
