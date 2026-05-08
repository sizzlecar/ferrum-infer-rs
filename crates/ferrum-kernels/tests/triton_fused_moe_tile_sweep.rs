//! Tile-sweep bench: load PTX from disk by tile name, run at Qwen3
//! shape, report µs/iter. Compare across tile combos to find the
//! winner vs Marlin's 183 µs/layer baseline.
//!
//! Usage (one tile per invocation):
//!   FERRUM_FUSED_MOE_TILE_BENCH=16x128x64 \
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!     --test triton_fused_moe_tile_sweep -- --ignored --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use std::time::Instant;

const NUM_EXPERTS: usize = 100;
const K: usize = 2048;
const N: usize = 1536;
const G: usize = 128;
const TOP_K: i32 = 8;
const SIZE_M: usize = 32;
const N_ITERS: usize = 50;

fn parse_meta_field<T: std::str::FromStr>(meta: &str, key: &str) -> Option<T> {
    let needle = format!("\"{key}\":");
    let pos = meta.find(&needle)? + needle.len();
    let rest = &meta[pos..];
    let end = rest.find(|c: char| c == ',' || c == '}')?;
    rest[..end].trim().parse::<T>().ok()
}

fn make_random_gptq(k: usize, n: usize, seed: u64) -> (Vec<i32>, Vec<half::f16>, Vec<i32>) {
    use half::f16;
    let mut s = seed;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let mut qw = vec![0i32; (k / 8) * n];
    for v in qw.iter_mut() {
        *v = rng() as i32;
    }
    let scales: Vec<f16> = (0..(k / G) * n)
        .map(|_| f16::from_f32(0.01 + ((rng() & 0xFF) as f32) * 0.0001))
        .collect();
    let mut qz = vec![0i32; (k / G) * (n / 8)];
    for v in qz.iter_mut() {
        *v = rng() as i32;
    }
    (qw, scales, qz)
}

#[test]
#[ignore]
fn triton_fused_moe_tile_sweep() {
    use half::f16;

    let tile = std::env::var("FERRUM_FUSED_MOE_TILE_BENCH")
        .unwrap_or_else(|_| "16x64x32".to_string());
    let parts: Vec<&str> = tile.split('x').collect();
    assert_eq!(parts.len(), 3, "tile format: BMxBNxBK");
    let bm: i32 = parts[0].parse().unwrap();
    let bn: i32 = parts[1].parse().unwrap();
    let bk: i32 = parts[2].parse().unwrap();
    eprintln!("── tile {tile}: BM={bm} BN={bn} BK={bk} ──");

    // Load PTX + meta from disk.
    let ptx_path = format!(
        "{}/triton_ptx/fused_moe_w4a16_f16_{}.ptx",
        env!("CARGO_MANIFEST_DIR"),
        tile.replace('x', "_")
    );
    let meta_path = ptx_path.replace(".ptx", ".json");
    let ptx_text = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("missing PTX {ptx_path}: {e}"));
    let meta_text = std::fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("missing meta {meta_path}: {e}"));
    let num_warps: u32 = parse_meta_field(&meta_text, "num_warps").unwrap_or(4);
    let shared_mem: u32 = parse_meta_field(&meta_text, "shared_mem").unwrap_or(0);
    eprintln!("loaded PTX ({} bytes) num_warps={num_warps} shared_mem={shared_mem}", ptx_text.len());

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Synthetic stacked weights.
    let mut qw_flat: Vec<i32> = Vec::with_capacity(NUM_EXPERTS * (K / 8) * N);
    let mut sc_flat: Vec<f16> = Vec::with_capacity(NUM_EXPERTS * (K / G) * N);
    let mut qz_flat: Vec<i32> = Vec::with_capacity(NUM_EXPERTS * (K / G) * (N / 8));
    for e in 0..NUM_EXPERTS {
        let (qw, sc, qz) = make_random_gptq(K, N, 0xCAFE0000 + e as u64);
        qw_flat.extend_from_slice(&qw);
        sc_flat.extend_from_slice(&sc);
        qz_flat.extend_from_slice(&qz);
    }
    let qw_dev: CudaSlice<i32> = stream.clone_htod(&qw_flat).unwrap();
    let sc_dev: CudaSlice<f16> = stream.clone_htod(&sc_flat).unwrap();
    let qz_dev: CudaSlice<i32> = stream.clone_htod(&qz_flat).unwrap();

    // Routing structures.
    let total_pairs = SIZE_M * (TOP_K as usize);
    let mb = bm as usize;
    let mut sorted_token_ids: Vec<i32> = Vec::new();
    let mut expert_ids: Vec<i32> = Vec::new();
    let mut idx = 0i32;
    for e in 0..NUM_EXPERTS {
        let m_e = if e < (total_pairs % NUM_EXPERTS) {
            (total_pairs / NUM_EXPERTS) + 1
        } else {
            total_pairs / NUM_EXPERTS
        };
        for i in 0..m_e {
            sorted_token_ids.push(idx + i as i32);
        }
        idx += m_e as i32;
        let pad = (mb - (m_e % mb)) % mb;
        for _ in 0..pad {
            sorted_token_ids.push(total_pairs as i32);
        }
        let blocks_for_e = ((m_e + mb - 1) / mb).max(1);
        for _ in 0..blocks_for_e {
            expert_ids.push(e as i32);
        }
    }
    let num_padded = sorted_token_ids.len() as i32;
    let st_dev: CudaSlice<i32> = stream.clone_htod(&sorted_token_ids).unwrap();
    let eid_dev: CudaSlice<i32> = stream.clone_htod(&expert_ids).unwrap();

    let a: Vec<f16> = (0..SIZE_M * K)
        .map(|i| f16::from_f32(((i as f32) * 0.0017).sin()))
        .collect();
    let a_dev: CudaSlice<f16> = stream.clone_htod(&a).unwrap();
    let mut c_dev: CudaSlice<f16> = stream
        .alloc_zeros::<f16>(SIZE_M * (TOP_K as usize) * N)
        .unwrap();

    let func = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_text))
        .unwrap()
        .load_function("fused_moe_w4a16_typed")
        .unwrap();

    // Pre-compute strides + per-expert offsets.
    let qw_per_expert = ((K / 8) * N) as i32;
    let groups = K as i32 / G as i32;
    let s_per_expert = (groups as i64 * N as i64) as i32;
    let qz_per_expert = (groups * N as i32 / 8) as i32;
    let stride_am = K as i32;
    let stride_qwk = N as i32;
    let stride_sk = N as i32;
    let stride_qzk = N as i32 / 8;
    let num_valid = SIZE_M as i32;
    let n_size = N as i32;
    let k_size = K as i32;
    let gs = G as i32;

    let blocks_m = ((num_padded + bm - 1) / bm) as u32;
    let blocks_n = ((n_size + bn - 1) / bn) as u32;

    let mut launch = || {
        let scratch_g: CudaSlice<u8> = stream.alloc_zeros::<u8>(1).unwrap();
        let scratch_p: CudaSlice<u8> = stream.alloc_zeros::<u8>(1).unwrap();
        let mut b = stream.launch_builder(&func);
        let inp = a_dev.slice(..);
        let qw = qw_dev.slice(..);
        let sc = sc_dev.slice(..);
        let qz = qz_dev.slice(..);
        let st = st_dev.slice(..);
        let eid = eid_dev.slice(..);
        b.arg(&inp);
        b.arg(&qw);
        b.arg(&sc);
        b.arg(&qz);
        b.arg(&st);
        b.arg(&eid);
        b.arg(&mut c_dev);
        b.arg(&num_valid);
        b.arg(&n_size);
        b.arg(&k_size);
        b.arg(&gs);
        b.arg(&qw_per_expert);
        b.arg(&s_per_expert);
        b.arg(&qz_per_expert);
        b.arg(&stride_am);
        b.arg(&1i32); // stride_ak
        b.arg(&stride_qwk);
        b.arg(&1i32); // stride_qwn
        b.arg(&stride_sk);
        b.arg(&1i32); // stride_sn
        b.arg(&stride_qzk);
        b.arg(&1i32); // stride_qzn
        b.arg(&n_size); // stride_cm
        b.arg(&1i32); // stride_cn
        b.arg(&scratch_g);
        b.arg(&scratch_p);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (blocks_m, blocks_n, 1),
                block_dim: (num_warps * 32, 1, 1),
                shared_mem_bytes: shared_mem,
            })
        }
        .expect("launch");
    };

    for _ in 0..3 {
        launch();
    }
    stream.synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..N_ITERS {
        launch();
    }
    stream.synchronize().unwrap();
    let elapsed = t0.elapsed();
    let per_iter = elapsed.as_micros() as f64 / N_ITERS as f64;
    eprintln!(
        "── RESULT tile={tile}: {per_iter:.1} µs/iter ({} iters in {:.1} ms)",
        N_ITERS,
        elapsed.as_secs_f64() * 1000.0
    );
}
