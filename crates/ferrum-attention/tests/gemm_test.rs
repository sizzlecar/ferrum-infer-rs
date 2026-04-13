//! Test Metal GEMM v2 kernel against Accelerate cblas_sgemm reference.

#[cfg(target_os = "macos")]
mod gemm_tests {
    use ferrum_attention::metal::pipelines::MetalPipelines;
    use metal::Device;

    extern "C" {
        fn cblas_sgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    fn ref_gemm_at_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        c
    }

    fn run_metal_gemm(
        pipes: &MetalPipelines,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let a_buf = pipes.buffer_from_data(a);
        let b_buf = pipes.buffer_from_data(b);
        let c_buf = pipes.buffer_empty(m * n);

        let cmd = pipes.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        #[repr(C)]
        struct P {
            m: i32,
            n: i32,
            k: i32,
        }
        let params = P {
            m: m as i32,
            n: n as i32,
            k: k as i32,
        };
        let params_buf = pipes.device.new_buffer_with_data(
            &params as *const _ as *const std::ffi::c_void,
            12,
            metal::MTLResourceOptions::StorageModeShared,
        );

        enc.set_compute_pipeline_state(pipes.pipeline("gemm_f32_v2"));
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        enc.set_threadgroup_memory_length(0, 12288);

        let grid = metal::MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
        let tg = metal::MTLSize::new(128, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
        MetalPipelines::read_buffer(&c_buf, m * n)
    }

    fn random_data(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(seed).wrapping_add(12345) % 10000) as f32;
                x / 10000.0 - 0.5
            })
            .collect()
    }

    #[test]
    fn test_gemm_v2_aligned() {
        let device = Device::system_default().expect("no Metal");
        let pipes = MetalPipelines::new(&device);

        // 64x32 aligned (exact tile)
        let m = 64;
        let n = 32;
        let k = 128;
        let a = random_data(m * k, 7);
        let b = random_data(n * k, 13);

        let ref_c = ref_gemm_at_bt(&a, &b, m, n, k);
        let metal_c = run_metal_gemm(&pipes, &a, &b, m, n, k);

        let max_diff = ref_c
            .iter()
            .zip(metal_c.iter())
            .map(|(r, m)| (r - m).abs())
            .fold(0.0f32, f32::max);
        eprintln!("aligned {}x{}x{}: max_diff = {:.2e}", m, n, k, max_diff);
        assert!(max_diff < 1e-4, "max_diff {} too large", max_diff);
    }

    #[test]
    fn test_gemm_v2_model_sizes() {
        let device = Device::system_default().expect("no Metal");
        let pipes = MetalPipelines::new(&device);

        // Real model GEMM sizes: Q/K/V projection, O proj, MLP
        let cases = [
            (73, 2048, 1024, "q_proj prefill"), // Q proj: [73, 1024] @ [2048, 1024]^T
            (73, 1024, 1024, "k_proj prefill"), // K proj
            (73, 1024, 2048, "o_proj prefill"), // O proj: [73, 2048] @ [1024, 2048]^T
            (73, 3072, 1024, "gate_proj prefill"), // MLP gate
            (73, 1024, 3072, "down_proj prefill"), // MLP down
            (1, 2048, 1024, "q_proj decode"),   // Single token decode
            (1, 3072, 1024, "gate_proj decode"),
        ];

        for (m, n, k, name) in cases {
            let a = random_data(m * k, 7);
            let b = random_data(n * k, 13);

            let ref_c = ref_gemm_at_bt(&a, &b, m, n, k);
            let metal_c = run_metal_gemm(&pipes, &a, &b, m, n, k);

            let max_diff = ref_c
                .iter()
                .zip(metal_c.iter())
                .map(|(r, m)| (r - m).abs())
                .fold(0.0f32, f32::max);
            eprintln!("{}: {}x{}x{} max_diff = {:.2e}", name, m, n, k, max_diff);
            assert!(max_diff < 1e-3, "{}: max_diff {} too large", name, max_diff);
        }
    }

    #[test]
    fn test_gemm_v2_perf() {
        let device = Device::system_default().expect("no Metal");
        let pipes = MetalPipelines::new(&device);

        let m = 73;
        let n = 2048;
        let k = 1024;
        let a = random_data(m * k, 7);
        let b = random_data(n * k, 13);

        // Warmup
        let _ = run_metal_gemm(&pipes, &a, &b, m, n, k);

        let start = std::time::Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let _ = run_metal_gemm(&pipes, &a, &b, m, n, k);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_secs_f64() / iters as f64 * 1000.0;
        eprintln!(
            "GEMM {}x{}x{}: {:.3}ms/iter ({} iters in {:.1}ms)",
            m,
            n,
            k,
            per_iter,
            iters,
            elapsed.as_millis()
        );
    }
}
