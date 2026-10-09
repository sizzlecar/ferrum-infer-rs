use super::*;
pub(super) struct Case {
    pub native: PreparedQ6F32Linear,
    pub p: Plan,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub xs: usize,
    pub ys: usize,
    pub x: Vec<f32>,
    pub w: Vec<u8>,
    pub input: Buffer,
    pub strict_input: Buffer,
    pub weights: Buffer,
    pub padded: Buffer,
    pub packed: Buffer,
    pub raw: Buffer,
    pub fixup: Buffer,
    pub output: Buffer,
    pub strict_output: Buffer,
    pub row_flags: Buffer,
    pub weight_flag: Buffer,
}
impl Case {
    pub fn new(s: &Arc<CudaStream>, m: usize, k: usize, n: usize, strided: bool) -> Self {
        let native = plan(s.context(), m as u32, k as u32, n as u32);
        let p = *native.geometry();
        let xs = k + if strided { 3 } else { 0 };
        let ys = n + 5;
        let mut x = vec![123.0; m * xs];
        for r in 0..m {
            for i in 0..k {
                // Include representable F32 values that an F16 roundtrip changes.
                x[r * xs + i] = (((i * 17 + r * 3) % 127) as f32 - 63.0) / 8192.0 + 1.0 / 1048576.0;
            }
        }
        let templates = oracle_blocks(GgufBlockFormat::Q6K);
        let mut w = vec![0; n * (k / 256) * 210];
        for (i, b) in w.chunks_exact_mut(210).enumerate() {
            b.copy_from_slice(&templates[(i % 2) * 210..(i % 2 + 1) * 210]);
            let d = match i % 7 {
                0 => 0,
                1 => 1,
                _ => f16::from_f32(if i % 2 == 0 { 0.003 } else { -0.004 }).to_bits(),
            };
            b[208..].copy_from_slice(&d.to_le_bytes());
        }
        let mut c = Self {
            native,
            p,
            m,
            k,
            n,
            xs,
            ys,
            x,
            w,
            input: Buffer::new(s, m * xs * 4),
            strict_input: Buffer::new(s, m * k * 4),
            weights: Buffer::new(s, p.weight_bytes as usize),
            padded: Buffer::new(s, p.converted_bytes as usize),
            packed: Buffer::new(s, p.packed_bytes as usize),
            raw: Buffer::new(s, p.output_bytes as usize),
            fixup: Buffer::new(s, p.fixup_bytes as usize),
            output: Buffer::new(s, m * ys * 4),
            strict_output: Buffer::new(s, m * ys * 4),
            row_flags: Buffer::new(s, m * 4),
            weight_flag: Buffer::new(s, 4),
        };
        c.upload(s);
        c.scan(s);
        s.synchronize().unwrap();
        c
    }
    pub fn upload(&mut self, s: &Arc<CudaStream>) {
        self.input.write(s, &bytes(&self.x));
        let compact: Vec<_> = self
            .x
            .chunks(self.xs)
            .flat_map(|r| r[..self.k].iter().copied())
            .collect();
        self.strict_input.write(s, &bytes(&compact));
        self.weights.write(s, &self.w);
    }
    pub fn scan(&self, s: &Arc<CudaStream>) {
        unsafe {
            self.native.check_weights(
                self.weights.span(s),
                self.weight_flag.span(s),
                s.cu_stream().cast(),
            )
        }
        .unwrap();
    }
    pub fn pack(&self, s: &Arc<CudaStream>) {
        unsafe {
            self.native.pack(
                self.input.span(s),
                self.xs as u32,
                self.padded.span(s),
                self.packed.span(s),
                self.row_flags.span(s),
                s.cu_stream().cast(),
            )
        }
        .unwrap();
    }
    pub fn dot(&self, s: &Arc<CudaStream>) {
        unsafe {
            self.native.dot(
                self.weights.span(s),
                self.packed.span(s),
                self.raw.span(s),
                self.fixup.span(s),
                s.cu_stream().cast(),
            )
        }
        .unwrap();
        self.publish(s);
    }
    pub fn publish(&self, s: &Arc<CudaStream>) {
        unsafe {
            self.native.publish(
                self.raw.span(s),
                self.output.span(s),
                self.ys as u32,
                self.row_flags.span(s),
                self.weight_flag.span(s),
                s.cu_stream().cast(),
            )
        }
        .unwrap();
    }
    pub fn inclusive(&self, s: &Arc<CudaStream>) {
        self.pack(s);
        self.dot(s);
    }
    pub fn strict(&self, s: &Arc<CudaStream>, kernels: &CudaNativeBlockKernels) {
        let x = self.strict_input.ptr(s) as u64;
        let w = self.weights.ptr(s) as u64;
        let y = self.strict_output.ptr(s) as u64;
        // Match the frozen strict selector: scalar for one physical row, T8
        // otherwise. Do not attribute a T8-at-M1 comparison to production.
        let (kernel, tile) = if self.m == 1 {
            (&kernels.linear_q6k_f32, 1)
        } else {
            (&kernels.linear_q6k_tiled_f32, 8)
        };
        let mut launch = s.launch_builder(kernel);
        launch.arg(&x).arg(&w).arg(&y);
        let args = [
            self.m as u32,
            self.k as u32,
            self.n as u32,
            self.ys as u32,
            0,
            14,
            256,
            210,
        ];
        for a in &args {
            launch.arg(a);
        }
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    (self.n as u32).div_ceil(4),
                    (self.m as u32).div_ceil(tile),
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    pub fn guards(&self, s: &Arc<CudaStream>) {
        for b in [
            &self.input,
            &self.strict_input,
            &self.weights,
            &self.padded,
            &self.packed,
            &self.raw,
            &self.fixup,
            &self.output,
            &self.strict_output,
            &self.row_flags,
            &self.weight_flag,
        ] {
            b.guards(s);
        }
        for b in [&self.output, &self.strict_output] {
            let data = b.read(s);
            for r in 0..self.m {
                assert!(data[(r * self.ys + self.n) * 4..(r + 1) * self.ys * 4]
                    .iter()
                    .all(|v| *v == 0x35));
            }
        }
    }
    pub fn oracle(&self, s: &Arc<CudaStream>, sparse: bool) -> serde_json::Value {
        let packed = self.packed.read(s);
        let out = floats(&self.output.read(s));
        let strict = floats(&self.strict_output.read(s));
        let mut rows: Vec<_> = if sparse {
            vec![0, self.m / 2, self.m - 1]
        } else {
            (0..self.m).collect()
        };
        rows.dedup();
        let cols: Vec<_> = if sparse {
            vec![0, self.n / 2, self.n - 1]
        } else {
            (0..self.n).collect()
        };
        let mut max_impl = 0f64;
        let mut max_quant = 0f64;
        let mut max_strict = 0f64;
        let mut count = 0;
        for r in rows {
            for &c in &cols {
                let start = c * (self.k / 256) * 210;
                let o = oracle::dot(
                    &self.w[start..start + self.k / 256 * 210],
                    &self.x[r * self.xs..r * self.xs + self.k],
                    &packed,
                    self.m,
                    r,
                );
                let a = f64::from(out[r * self.ys + c]);
                let b = f64::from(strict[r * self.ys + c]);
                // Conservative F32 reduction bound, scaled by expanded products;
                // no net-output-relative tolerance that masks cancellation.
                let gamma = (self.k as f64 + 64.0) * f64::from(f32::EPSILON);
                let tol = gamma * o[1] + 1e-7;
                let strict_tol = gamma * o[3] + 1e-7;
                assert!(a.is_finite() && b.is_finite());
                assert!(
                    (a - o[0]).abs() <= tol,
                    "actual-pack implementation error r={r} c={c}: {a} vs {} tolerance {tol}",
                    o[0]
                );
                assert!(
                    (b - o[2]).abs() <= strict_tol,
                    "strict decoded F64 mismatch"
                );
                assert!(
                    (a - o[2]).abs() <= o[4] + tol,
                    "activation-error decomposition"
                );
                max_impl = max_impl.max((a - o[0]).abs());
                max_quant = max_quant.max((o[0] - o[2]).abs());
                max_strict = max_strict.max((a - b).abs());
                count += 1;
            }
        }
        serde_json::json!({"checked_outputs":count,"sparse":sparse,"max_actual_pack_implementation_abs_error":max_impl,
            "max_activation_quantization_abs_error":max_quant,"max_difference_from_original_strict":max_strict,
            "strict_route":if self.m==1 {"original_strict_scalar"} else {"original_strict_t8"},
            "quality_status":"not evaluated; synthetic inputs, no model claim"})
    }
}
