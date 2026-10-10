use super::*;

pub(super) struct Case {
    pub p: Plan,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub xs: usize,
    pub ys: usize,
    pub output_offset: usize,
    pub weight_offset: usize,
    pub x: Vec<f16>,
    pub w: Vec<u8>,
    pub input: Buffer,
    pub strict_input: Buffer,
    pub weights: Buffer,
    pub converted: Buffer,
    pub packed: Buffer,
    pub raw: Buffer,
    pub fixup: Buffer,
    pub output: Buffer,
    pub strict_output: Buffer,
    pub row_flags: Buffer,
    pub weight_flag: Buffer,
}
impl Case {
    pub fn new(
        s: &Arc<CudaStream>,
        m: usize,
        k: usize,
        n: usize,
        strided: bool,
        leaf: bool,
    ) -> Self {
        let p = abi::plan(s.context(), m, k, n);
        let xs = k + if strided { 3 } else { 0 };
        let output_offset = if leaf {
            17408
        } else if strided {
            3
        } else {
            0
        };
        let ys = if leaf { 34816 } else { output_offset + n + 5 };
        assert!(output_offset + n <= ys);
        // The Q6 tile loader's existing contract requires a four-byte base;
        // activation and result subregions independently allow two bytes.
        let weight_offset = if strided { 4 } else { 0 };
        let mut x = vec![f16::from_f32(123.0); m * xs];
        for r in 0..m {
            for i in 0..k {
                x[r * xs + i] = f16::from_f32((((i * 17 + r * 7) % 127) as f32 - 63.0) / 8192.0);
            }
        }
        let templates = oracle_blocks(GgufBlockFormat::Q6K);
        let mut w = vec![0; n * (k / 256) * 210];
        for (i, block) in w.chunks_exact_mut(210).enumerate() {
            block.copy_from_slice(&templates[(i % 2) * 210..(i % 2 + 1) * 210]);
            let scale = match i % 7 {
                0 => 0,
                1 => 1,
                _ => f16::from_f32(if i % 2 == 0 { 0.003 } else { -0.004 }).to_bits(),
            };
            block[208..210].copy_from_slice(&scale.to_le_bytes());
        }
        let mut c = Self {
            p,
            m,
            k,
            n,
            xs,
            ys,
            output_offset,
            weight_offset,
            x,
            w,
            input: Buffer::new(s, m * xs * 2),
            strict_input: Buffer::new(s, m * k * 2),
            weights: Buffer::new(s, weight_offset + p.weight_bytes as usize),
            converted: Buffer::new(s, p.converted_bytes as usize),
            packed: Buffer::new(s, p.packed_bytes as usize),
            raw: Buffer::new(s, p.output_bytes as usize),
            fixup: Buffer::new(s, p.fixup_bytes as usize),
            output: Buffer::new(s, m * ys * 2),
            strict_output: Buffer::new(s, m * ys * 2),
            row_flags: Buffer::new(s, m * 4),
            weight_flag: Buffer::new(s, 4),
        };
        c.upload(s);
        c
    }
    pub fn upload(&mut self, s: &Arc<CudaStream>) {
        self.input.write(s, &half_bytes(&self.x));
        let compact: Vec<_> = self
            .x
            .chunks(self.xs)
            .flat_map(|row| row[..self.k].iter().copied())
            .collect();
        self.strict_input.write(s, &half_bytes(&compact));
        let mut weight_bytes = vec![0x35; self.weight_offset];
        weight_bytes.extend_from_slice(&self.w);
        self.weights.write(s, &weight_bytes);
    }
    pub fn weight_ptr(&self, s: &Arc<CudaStream>) -> *mut c_void {
        (self.weights.ptr(s) as usize + self.weight_offset) as *mut c_void
    }
    pub fn output_ptr(&self, s: &Arc<CudaStream>) -> *mut c_void {
        (self.output.ptr(s) as usize + self.output_offset * 2) as *mut c_void
    }
    pub fn scan(&self, s: &Arc<CudaStream>) {
        assert_eq!(
            unsafe {
                abi::ferrum_upstream_q6_f16_check_weights_v1(
                    &self.p,
                    self.weight_ptr(s),
                    self.weight_flag.ptr(s),
                    s.cu_stream().cast(),
                )
            },
            0
        );
    }
    pub fn pack(&self, s: &Arc<CudaStream>) {
        assert_eq!(
            unsafe {
                abi::ferrum_upstream_q6_f16_pack_v1(
                    &self.p,
                    self.input.ptr(s),
                    self.xs as u32,
                    self.converted.ptr(s),
                    self.packed.ptr(s),
                    self.row_flags.ptr(s),
                    s.cu_stream().cast(),
                )
            },
            0
        );
    }
    pub fn publish(&self, s: &Arc<CudaStream>) {
        assert_eq!(
            unsafe {
                abi::ferrum_upstream_q6_f16_cast_v1(
                    &self.p,
                    self.raw.ptr(s),
                    self.output_ptr(s),
                    self.ys as u32,
                    self.row_flags.ptr(s),
                    self.weight_flag.ptr(s),
                    s.cu_stream().cast(),
                )
            },
            0
        );
    }
    pub fn inclusive(&self, s: &Arc<CudaStream>) {
        self.pack(s);
        assert_eq!(
            unsafe {
                abi::ferrum_upstream_q6_f16_dot_v1(
                    &self.p,
                    self.weight_ptr(s),
                    self.packed.ptr(s),
                    self.raw.ptr(s),
                    self.fixup.ptr(s),
                    s.cu_stream().cast(),
                )
            },
            0
        );
        self.publish(s);
    }
    pub fn strict(&self, s: &Arc<CudaStream>, kernels: &CudaNativeBlockKernels) {
        let x = self.strict_input.ptr(s) as u64;
        let w = self.weight_ptr(s) as u64;
        let y = self.strict_output.ptr(s) as u64;
        let (kernel, tile) = if self.m == 1 {
            (&kernels.linear_q6k_f16, 1)
        } else {
            (&kernels.linear_q6k_tiled_f16, 8)
        };
        let mut launch = s.launch_builder(kernel);
        launch.arg(&x).arg(&w).arg(&y);
        let args = [
            self.m as u32,
            self.k as u32,
            self.n as u32,
            self.ys as u32,
            self.output_offset as u32,
            14,
            256,
            210,
        ];
        for arg in &args {
            launch.arg(arg);
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
        for buffer in [
            &self.input,
            &self.strict_input,
            &self.weights,
            &self.converted,
            &self.packed,
            &self.raw,
            &self.fixup,
            &self.output,
            &self.strict_output,
            &self.row_flags,
            &self.weight_flag,
        ] {
            buffer.guards(s);
        }
        for output in [&self.output, &self.strict_output] {
            let bytes = output.read(s);
            for row in 0..self.m {
                assert!(
                    bytes[row * self.ys * 2..(row * self.ys + self.output_offset) * 2]
                        .iter()
                        .all(|b| *b == 0x35)
                );
                assert!(bytes
                    [(row * self.ys + self.output_offset + self.n) * 2..(row + 1) * self.ys * 2]
                    .iter()
                    .all(|b| *b == 0x35));
            }
        }
        assert_eq!(
            self.input.read(s),
            half_bytes(&self.x),
            "input immutability"
        );
        let weight_bytes = self.weights.read(s);
        assert!(weight_bytes[..self.weight_offset]
            .iter()
            .all(|b| *b == 0x35));
        assert_eq!(&weight_bytes[self.weight_offset..], self.w);
    }
    /// All output cast bits are checked, including marker propagation. The
    /// F64 matrix oracle is exhaustive for small cases and sampled for timing.
    pub fn validate(&self, s: &Arc<CudaStream>, exhaustive: bool) -> serde_json::Value {
        let packed = self.packed.read(s);
        let raw = floats(&self.raw.read(s));
        let output = halves(&self.output.read(s));
        let strict = halves(&self.strict_output.read(s));
        assert!(self.row_flags.read(s).iter().all(|b| *b == 0));
        assert!(self.weight_flag.read(s).iter().all(|b| *b == 0));
        for r in 0..self.m {
            for c in 0..self.n {
                let i = r * self.ys + self.output_offset + c;
                assert!(
                    raw[r * self.n + c].is_finite()
                        && output[i].is_finite()
                        && strict[i].is_finite()
                );
                assert_eq!(
                    output[i].to_bits(),
                    f16::from_f32(raw[r * self.n + c]).to_bits(),
                    "F16 publication bits"
                );
            }
        }
        let mut rows = if exhaustive {
            (0..self.m).collect::<Vec<_>>()
        } else {
            vec![0, self.m / 2, self.m - 1]
        };
        let mut cols = if exhaustive {
            (0..self.n).collect::<Vec<_>>()
        } else {
            vec![0, self.n / 2, self.n - 1]
        };
        rows.dedup();
        cols.dedup();
        let mut count = 0;
        let mut max_impl = 0f64;
        let mut max_quant = 0f64;
        let mut max_strict = 0f64;
        for r in rows {
            let x: Vec<_> = self.x[r * self.xs..r * self.xs + self.k]
                .iter()
                .map(|x| x.to_f32())
                .collect();
            for &c in &cols {
                let start = c * (self.k / 256) * 210;
                let expected = oracle::dot(
                    &self.w[start..start + self.k / 256 * 210],
                    &x,
                    &packed,
                    self.m,
                    r,
                );
                let actual = f64::from(raw[r * self.n + c]);
                let strict = strict[r * self.ys + self.output_offset + c].to_f64();
                let gamma = (self.k as f64 + 64.0) * f64::from(f32::EPSILON);
                let implementation_bound = gamma * expected[1] + 1e-7;
                let strict_bound =
                    gamma * expected[3] + strict.abs() * 2f64.powi(-10) + 2f64.powi(-24);
                assert!((actual - expected[0]).abs() <= implementation_bound, "actual packed Q8 oracle r={r} c={c}: {actual} vs {} bound {implementation_bound}", expected[0]);
                assert!(
                    (strict - expected[2]).abs() <= strict_bound,
                    "strict F16 decoded oracle r={r} c={c}"
                );
                assert!(
                    (actual - expected[2]).abs() <= expected[4] + implementation_bound,
                    "activation error decomposition"
                );
                max_impl = max_impl.max((actual - expected[0]).abs());
                max_quant = max_quant.max((expected[0] - expected[2]).abs());
                max_strict = max_strict.max((strict - expected[2]).abs());
                count += 1;
            }
        }
        self.guards(s);
        serde_json::json!({"checked_f64_outputs":count,"exhaustive":exhaustive,"all_output_cast_bits":self.m*self.n,
            "max_actual_pack_implementation_abs_error":max_impl,"max_activation_quantization_abs_error":max_quant,
            "max_strict_f16_oracle_abs_error":max_strict,"cross_algorithm_bit_equality_required":false})
    }
}
