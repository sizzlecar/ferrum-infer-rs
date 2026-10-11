use super::*;
pub(super) struct Route {
    pub p: UpstreamLinearPlanV1,
    pub native: PreparedUpstreamLinear,
    pub converted: Buffer,
    pub packed: Buffer,
    pub raw: Buffer,
    pub fixup: Buffer,
    pub output: Buffer,
    pub rows: Buffer,
    pub weight_flag: Buffer,
}
impl Route {
    fn new(s: &Arc<CudaStream>, native: PreparedUpstreamLinear, stride: usize) -> Self {
        let p = *native.geometry();
        Self {
            converted: Buffer::new(s, p.converted_bytes as usize),
            packed: Buffer::new(s, p.packed_bytes as usize),
            raw: Buffer::new(s, p.output_bytes as usize),
            fixup: Buffer::new(s, p.fixup_bytes as usize),
            output: Buffer::new(s, p.request.rows as usize * stride * 2),
            rows: Buffer::new(s, p.request.rows as usize * 4),
            weight_flag: Buffer::new(s, 4),
            p,
            native,
        }
    }
    pub fn scan(&self, s: &Arc<CudaStream>, w: &Buffer) {
        unsafe {
            self.native
                .check_weights(w.span(s), self.weight_flag.span(s), s.cu_stream().cast())
                .unwrap();
        }
    }
    pub fn pack(&self, s: &Arc<CudaStream>, x: &Buffer, stride: u32) {
        unsafe {
            self.native
                .pack(
                    x.span(s),
                    stride,
                    self.converted.span(s),
                    self.packed.span(s),
                    self.rows.span(s),
                    s.cu_stream().cast(),
                )
                .unwrap();
        }
    }
    pub fn dot_cast(&self, s: &Arc<CudaStream>, w: &Buffer, stride: u32) {
        unsafe {
            self.native
                .dot(
                    w.span(s),
                    self.packed.span(s),
                    self.raw.span(s),
                    self.fixup.span(s),
                    s.cu_stream().cast(),
                )
                .unwrap();
            self.native
                .cast(
                    self.raw.span(s),
                    self.output.span(s),
                    stride,
                    self.rows.span(s),
                    self.weight_flag.span(s),
                    s.cu_stream().cast(),
                )
                .unwrap();
        }
    }
    pub fn enqueue(&self, s: &Arc<CudaStream>, x: &Buffer, xstride: u32, w: &Buffer, ystride: u32) {
        self.pack(s, x, xstride);
        self.dot_cast(s, w, ystride);
    }
    pub fn guards(&self, s: &Arc<CudaStream>) {
        for b in [
            &self.converted,
            &self.packed,
            &self.raw,
            &self.fixup,
            &self.output,
            &self.rows,
            &self.weight_flag,
        ] {
            b.guards(s);
        }
    }
}
pub(super) struct Case {
    pub format: GgufBlockFormat,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub xstride: usize,
    pub ystride: usize,
    pub x: Vec<f16>,
    pub w: Vec<u8>,
    pub bx: Buffer,
    pub bw: Buffer,
    pub baseline: Buffer,
    pub routes: Vec<Route>,
}
impl Case {
    pub fn new(
        s: &Arc<CudaStream>,
        f: GgufBlockFormat,
        m: usize,
        k: usize,
        n: usize,
        padded_input: bool,
        salt: usize,
        algorithms: &[u32],
    ) -> Self {
        let plans: Vec<_> = algorithms
            .iter()
            .map(|&a| plan(s, f, m as u32, k as u32, n as u32, a))
            .collect();
        Self::with_plans(s, f, m, k, n, padded_input, salt, plans)
    }
    pub(super) fn with_plans(
        s: &Arc<CudaStream>,
        f: GgufBlockFormat,
        m: usize,
        k: usize,
        n: usize,
        padded_input: bool,
        salt: usize,
        plans: Vec<PreparedUpstreamLinear>,
    ) -> Self {
        for p in &plans {
            let r = &p.geometry().request;
            assert_eq!(
                (r.format, r.rows, r.inputs, r.outputs),
                (f.ggml_type_id(), m as u32, k as u32, n as u32)
            );
        }
        let pn = plans
            .iter()
            .map(|p| p.geometry().padded_outputs as usize)
            .max()
            .unwrap();
        let source = oracle_blocks(f);
        let templates: Vec<_> = source.chunks_exact(f.block_bytes()).collect();
        let mut w = Vec::with_capacity(pn * (k / f.block_values()) * f.block_bytes());
        for col in 0..pn {
            for b in 0..k / f.block_values() {
                let mut block = templates[(col + b + salt) % templates.len()].to_vec();
                let d = if f == GgufBlockFormat::Q3K { 108 } else { 0 };
                let scale = if col >= n {
                    0.0
                } else {
                    ((col + salt) % 5 + 1) as f32 / 1024.0 * if b % 2 == 0 { 1.0 } else { -1.0 }
                };
                block[d..d + 2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
                if col >= n {
                    block.fill(0)
                }
                w.extend(block);
            }
        }
        let xstride = k + if padded_input { 3 } else { 0 };
        let ystride = n + 5;
        let mut x = vec![f16::from_f32(31.0); m * xstride];
        for row in 0..m {
            for i in 0..k {
                x[row * xstride + i] =
                    f16::from_f32((((row * 7 + i * 11 + salt) % 29) as f32 - 14.0) / 1024.0)
            }
        }
        let mut bx = Buffer::new(s, x.len() * 2);
        bx.write(s, &half_bytes(&x));
        let mut bw = Buffer::new(s, w.len());
        bw.write(s, &w);
        let baseline = Buffer::new(s, m * ystride * 2);
        let routes = plans
            .into_iter()
            .map(|p| Route::new(s, p, ystride))
            .collect();
        let case = Self {
            format: f,
            m,
            k,
            n,
            xstride,
            ystride,
            x,
            w,
            bx,
            bw,
            baseline,
            routes,
        };
        for r in &case.routes {
            r.scan(s, &case.bw);
        }
        s.synchronize().unwrap();
        case
    }
    pub fn upload(&mut self, s: &Arc<CudaStream>) {
        self.bx.write(s, &half_bytes(&self.x));
        self.bw.write(s, &self.w);
        for r in &self.routes {
            r.scan(s, &self.bw);
        }
        s.synchronize().unwrap();
    }
    pub fn reset(&mut self, s: &Arc<CudaStream>) {
        self.baseline.poison(s);
        for r in &mut self.routes {
            r.packed.poison(s);
            r.raw.poison(s);
            r.output.poison(s);
            r.rows.poison(s);
        }
        s.synchronize().unwrap();
    }
    pub fn native(&self, s: &Arc<CudaStream>, index: usize) {
        self.routes[index].enqueue(
            s,
            &self.bx,
            self.xstride as u32,
            &self.bw,
            self.ystride as u32,
        )
    }
    pub fn strict(&self, s: &Arc<CudaStream>, kernels: &CudaNativeBlockKernels) {
        assert_eq!(self.xstride, self.k);
        let x = self.bx.pointer(s) as u64;
        let w = self.bw.pointer(s) as u64;
        let y = self.baseline.pointer(s) as u64;
        let params = [
            self.m as u32,
            self.k as u32,
            self.n as u32,
            self.ystride as u32,
            0,
            self.format.ggml_type_id(),
            self.format.block_values() as u32,
            self.format.block_bytes() as u32,
        ];
        // Mirror the retained generic production dispatch: scalar for one
        // physical row, T8 for multiple rows. The old M8 control is unchanged.
        let row_tile = if self.m == 1 { 1 } else { 8 };
        let kernel = if self.m == 1 {
            &kernels.linear_f16
        } else {
            &kernels.linear_tiled_f16
        };
        let mut l = s.launch_builder(kernel);
        l.arg(&x).arg(&w).arg(&y);
        for p in &params {
            l.arg(p);
        }
        unsafe {
            l.launch(LaunchConfig {
                grid_dim: (
                    (self.n as u32).div_ceil(4),
                    (self.m as u32).div_ceil(row_tile),
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    pub fn group(&self, r: &Route, bytes: &[u8], row: usize, g: usize) -> ([i8; 32], f32) {
        let (off, scale) = if r.p.algorithm == 1 {
            let b = ((g / 4) * self.m + row) * 144;
            let d = f32::from_le_bytes(
                bytes[b + (g % 4) * 4..b + (g % 4) * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            (b + 16 + (g % 4) * 32, d)
        } else {
            let b = (row * (r.p.padded_inputs as usize / 32) + g) * 36;
            (
                b + 4,
                f16::from_bits(u16::from_le_bytes(bytes[b..b + 2].try_into().unwrap())).to_f32(),
            )
        };
        (std::array::from_fn(|i| bytes[off + i] as i8), scale)
    }
    pub fn validate(&self, s: &Arc<CudaStream>, route: Option<usize>) -> Vec<u16> {
        s.synchronize().unwrap();
        let (out, raw, pack, flags) = if let Some(i) = route {
            let r = &self.routes[i];
            r.guards(s);
            (
                r.output.read(s),
                Some(float_read(&r.raw.read(s))),
                Some(r.packed.read(s)),
                Some(r.rows.read(s)),
            )
        } else {
            (self.baseline.read(s), None, None, None)
        };
        self.bx.guards(s);
        self.bw.guards(s);
        self.baseline.guards(s);
        assert_eq!(self.bx.read(s), half_bytes(&self.x));
        assert_eq!(self.bw.read(s), self.w);
        let y = half_read(&out);
        let leaf_bad = self.w.chunks_exact(self.format.block_bytes()).any(|b| {
            let d = if self.format == GgufBlockFormat::Q3K {
                108
            } else {
                0
            };
            !f16::from_bits(u16::from_le_bytes([b[d], b[d + 1]])).is_finite()
        });
        if let Some(i) = route {
            let r = &self.routes[i];
            let flag = u32::from_le_bytes(r.weight_flag.read(s).try_into().unwrap());
            assert_eq!(flag != 0, leaf_bad);
            if r.p.algorithm == 1 {
                let end = self.m * r.p.padded_inputs as usize * 9 / 8;
                assert!(pack.as_ref().unwrap()[end..].iter().all(|v| *v == 0));
            }
        }
        let mut max_quant_error = 0.0_f64;
        let mut checked = 0;
        // Generated finite fixtures with |x| <= 1 and their small retained
        // scales cannot overflow F16. Unexpected nonfinite raw output must not
        // pass merely because the final marker cast faithfully poisoned it.
        let finite_fixture = !leaf_bad
            && (0..self.m).all(|row| {
                self.x[row * self.xstride..row * self.xstride + self.k]
                    .iter()
                    .all(|x| x.is_finite() && x.to_f32().abs() <= 1.0)
            });
        for row in 0..self.m {
            let row_bad = self.x[row * self.xstride..row * self.xstride + self.k]
                .iter()
                .any(|x| !x.is_finite());
            if let Some(f) = &flags {
                assert_eq!(
                    u32::from_le_bytes(f[row * 4..row * 4 + 4].try_into().unwrap()) != 0,
                    row_bad
                );
            }
            if let Some(i) = route {
                let r = &self.routes[i];
                let qbytes = pack.as_ref().unwrap();
                for g in 0..r.p.padded_inputs as usize / 32 {
                    let (q, d) = self.group(r, qbytes, row, g);
                    let zero = g * 32 >= self.k
                        || self.x[row * self.xstride + g * 32..row * self.xstride + g * 32 + 32]
                            .iter()
                            .all(|v| v.to_f32() == 0.0);
                    if zero {
                        assert!(q.iter().all(|v| *v == 0));
                        assert_eq!(d.to_bits(), 0)
                    } else if self.x[row * self.xstride + g * 32..row * self.xstride + g * 32 + 32]
                        .iter()
                        .any(|v| !v.is_finite())
                    {
                        assert!(q.iter().all(|v| *v == 0));
                        assert_eq!(d.to_bits(), 0x7fc00000, "canonical packed poison scale")
                    } else {
                        assert!(d.is_finite());
                    }
                }
            }
            for col in 0..self.ystride {
                let bits = y[row * self.ystride + col];
                if col >= self.n {
                    assert_eq!(bits, 0x3535);
                    continue;
                }
                if let Some(v) = &raw {
                    let f = v[row * self.n + col];
                    let h = f16::from_f32(f);
                    if finite_fixture {
                        assert!(
                            f.is_finite() && h.is_finite(),
                            "qualified finite fixture was poisoned"
                        );
                    }
                    assert_eq!(
                        bits,
                        if leaf_bad || row_bad || !f.is_finite() || !h.is_finite() {
                            0x7e00
                        } else {
                            h.to_bits()
                        }
                    );
                }
                if row_bad || leaf_bad {
                    if route.is_some() {
                        assert_eq!(bits, 0x7e00)
                    }
                    continue;
                }
                if raw
                    .as_ref()
                    .is_some_and(|v| !f16::from_f32(v[row * self.n + col]).is_finite())
                {
                    assert_eq!(bits, 0x7e00);
                    continue;
                }
                assert!(f16::from_bits(bits).is_finite());
                if self.n > 49
                    && !([0, self.m / 2, self.m - 1].contains(&row)
                        && [0, self.n / 2, self.n - 1].contains(&col))
                {
                    continue;
                }
                let mut target = 0.0;
                let mut magnitude = 0.0;
                let mut original = 0.0;
                let mut original_abs = 0.0;
                for g in 0..self.k / 32 {
                    let block = (col * (self.k / self.format.block_values())
                        + g / (self.format.block_values() / 32))
                        * self.format.block_bytes();
                    let wb = &self.w[block..block + self.format.block_bytes()];
                    let group = g % (self.format.block_values() / 32);
                    let decoded = oracle::decode_block(self.format, wb);
                    for j in 0..32 {
                        let term = decoded[group * 32 + j]
                            * self.x[row * self.xstride + g * 32 + j].to_f32() as f64;
                        original += term;
                        original_abs += term.abs();
                    }
                    if let Some(i) = route {
                        let (q, d) = self.group(&self.routes[i], pack.as_ref().unwrap(), row, g);
                        let (v, a) = oracle::dot_group(self.format, wb, group, &q, d);
                        target += v;
                        magnitude += a;
                    }
                }
                if route.is_none() {
                    target = original;
                    magnitude = original_abs;
                } else {
                    max_quant_error = max_quant_error.max((target - original).abs());
                }
                let epsilon = f32::EPSILON as f64;
                let count = (2 * self.k + 64) as f64;
                let gamma = count * epsilon / (1.0 - count * epsilon);
                let bound = gamma * magnitude + target.abs() / 2048.0 + 2.0_f64.powi(-24);
                let actual = f16::from_bits(bits).to_f32() as f64;
                assert!(
                    (actual - target).abs() <= bound,
                    "{:?} M{} K{} N{} route{route:?} [{row},{col}] {actual} target{target} bound{bound}",
                    self.format,
                    self.m,
                    self.k,
                    self.n
                );
                checked += 1;
            }
        }
        println!(
            "{}",
            serde_json::json!({"kind":"extra_correctness","format":format!("{:?}",self.format),"m":self.m,"k":self.k,"n":self.n,"algorithm":route.map(|i|self.routes[i].p.algorithm),"checked_outputs":checked,"oracle":"actual_pack_f64_separate_from_quantization_error","max_observed_quantization_error_at_checked_outputs":max_quant_error})
        );
        y
    }
}
