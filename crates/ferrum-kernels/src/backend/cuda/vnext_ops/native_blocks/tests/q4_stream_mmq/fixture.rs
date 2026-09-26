use super::*;
pub(super) const PAD: usize = 8;
pub(super) const SENTINEL: f32 = -12344.0;
const TEMPLATES: usize = 16;

pub(super) fn templates(format: GgufBlockFormat, k: usize, salt: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(k > 0 && k % 256 == 0);
    let bytes = format.block_bytes();
    let mut raw = vec![0_u8; TEMPLATES * (k / 256) * bytes];
    for (i, b) in raw.chunks_exact_mut(bytes).enumerate() {
        for (j, x) in b.iter_mut().enumerate() {
            *x = (i * 31 + j * 43 + salt * 67 + j / 3 * 11) as u8;
        }
        match format {
            GgufBlockFormat::Q4K => {
                b[..2].copy_from_slice(
                    &f16::from_f32((1 + (i + salt) % 3) as f32 / 4096.0)
                        .to_bits()
                        .to_le_bytes(),
                );
                b[2..4].copy_from_slice(
                    &f16::from_f32((1 + (i + salt) % 3) as f32 / 1024.0)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
            GgufBlockFormat::Q6K => {
                for (j, x) in b[192..208].iter_mut().enumerate() {
                    *x = (((i + j + salt) % 33) as i8 - 16) as u8;
                }
                b[208..210].copy_from_slice(
                    &f16::from_f32((1 + (i + salt) % 3) as f32 / 8192.0)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
            _ => unreachable!(),
        }
    }
    let mut decoded = vec![0.0; TEMPLATES * k];
    format.decode(&raw, &mut decoded).unwrap();
    (raw, decoded)
}
pub(super) fn inputs(rows: usize, k: usize, g: usize) -> Vec<f16> {
    let mut x = vec![f16::from_f32(SENTINEL); PAD + rows * k + PAD];
    for r in 0..rows {
        for i in 0..k {
            x[PAD + r * k + i] =
                f16::from_f32((((i * 13 + r * 19 + g * 29) % 127) as f32 - 63.25) / 2048.0);
        }
    }
    x
}
pub(super) struct Matrix {
    pub raw: Vec<u8>,
    pub base: usize,
    pub gpu: CudaSlice<u8>,
    pub k: usize,
    pub n: usize,
    pub format: GgufBlockFormat,
    values: Vec<f32>,
}
impl Matrix {
    pub fn new(
        s: &Arc<CudaStream>,
        format: GgufBlockFormat,
        k: usize,
        n: usize,
        salt: usize,
    ) -> Self {
        Self::new_with_base(s, format, k, n, salt, 32)
    }
    pub fn new_with_base(
        s: &Arc<CudaStream>,
        format: GgufBlockFormat,
        k: usize,
        n: usize,
        salt: usize,
        base: usize,
    ) -> Self {
        assert!(base >= 4);
        let (raw, values) = templates(format, k, salt);
        let rb = k / 256 * format.block_bytes();
        let mut padded = vec![0xcc; base];
        for col in 0..n {
            let i = col % TEMPLATES * rb;
            padded.extend_from_slice(&raw[i..i + rb]);
        }
        padded.extend([0xcc; 7]);
        let gpu = s.clone_htod(&padded).unwrap();
        Self {
            raw: padded,
            base,
            gpu,
            k,
            n,
            format,
            values,
        }
    }
    pub fn coefficient(&self, col: usize, k: usize) -> f32 {
        self.values[(col % TEMPLATES) * self.k + k]
    }
    pub fn check_immutable(&self, s: &Arc<CudaStream>) {
        assert_eq!(s.clone_dtoh(&self.gpu).unwrap(), self.raw);
    }
}
pub(super) fn check_projection(
    x: &[f16],
    y: &[f16],
    rows: usize,
    w: &Matrix,
    stride: usize,
    offset: usize,
    q8_policy: bool,
) {
    let mut refs = Vec::with_capacity(rows * TEMPLATES);
    let packed = crate::gguf_blocks::q4k_q8_reference::pack_rows(x, rows, w.k);
    for r in 0..rows {
        for col in 0..TEMPLATES {
            let (mut dot, mut abs) = (0.0f64, 0.0f64);
            if q8_policy {
                for bi in 0..w.k / 256 {
                    let block =
                        decode_q4(&w.raw[w.base + (col % w.n) * (w.k / 256) * 144 + bi * 144..]);
                    for g in 0..8 {
                        let pg = r * (w.k / 32) + bi * 8 + g;
                        let d = f64::from(packed.scales[pg]);
                        let a = f64::from(block.d.to_f32() * f32::from(block.scales[g]));
                        let b = f64::from(block.dmin.to_f32() * f32::from(block.minima[g]));
                        let (mut qdot, mut qsum) = (0i32, 0i32);
                        for lane in 0..32 {
                            let qx = i32::from(packed.quants[pg * 32 + lane]);
                            let qw = i32::from(block.quants[g * 32 + lane]);
                            qdot += qx * qw;
                            qsum += qx;
                            abs +=
                                (d * a * f64::from(qx * qw)).abs() + (d * b * f64::from(qx)).abs();
                        }
                        dot += d * (a * f64::from(qdot) - b * f64::from(qsum));
                    }
                }
            } else {
                for i in 0..w.k {
                    let term = x[r * w.k + i].to_f64() * f64::from(w.coefficient(col, i));
                    dot += term;
                    abs += term.abs();
                }
            }
            // Expanded pre-cancellation terms bound each rounded product/subtract,
            // the sequential group sum and ordered Stream-K partial sum. Every
            // partial contains >=1 K256 block, so groups+blocks+8 bounds the depth.
            let n = if q8_policy {
                w.k / 32 + w.k / 256 + 8
            } else {
                w.k
            };
            let ne = n as f64 * f64::from(f32::EPSILON);
            assert!(ne < 1.0);
            let bound = ne / (1.0 - ne) * abs + dot.abs() / 1024.0 + 1e-6;
            refs.push((dot, bound));
        }
    }
    let mut max_abs_error = 0.0f64;
    let mut max_bound_ratio = 0.0f64;
    let mut worst = (0usize, 0usize);
    for r in 0..rows {
        for col in 0..w.n {
            let observed = y[r * stride + offset + col].to_f64();
            let (dot, bound) = refs[r * TEMPLATES + col % TEMPLATES];
            let error = (observed - dot).abs();
            if error > max_abs_error {
                max_abs_error = error;
                worst = (r, col);
            }
            max_bound_ratio = max_bound_ratio.max(error / bound);
            assert!(observed.is_finite()&&(observed-dot).abs()<=bound,"projection r={r} c={col} k={} q8={q8_policy} observed={observed} expected={dot} bound={bound}",w.k);
        }
    }
    println!(
        "{}",
        serde_json::json!({"event":"projection_oracle", "q8_policy":q8_policy, "rows":rows, "inputs":w.k, "outputs":w.n, "offset":offset, "format":format!("{:?}",w.format), "max_abs_error":max_abs_error, "max_error_over_original_bound":max_bound_ratio, "worst_coordinate":worst})
    );
}

pub(super) fn decode_q4(bytes: &[u8]) -> crate::gguf_blocks::q4k_q8_reference::Q4Block {
    let s = &bytes[4..16];
    crate::gguf_blocks::q4k_q8_reference::Q4Block {
        d: f16::from_le_bytes(bytes[..2].try_into().unwrap()),
        dmin: f16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        scales: std::array::from_fn(|g| {
            if g < 4 {
                s[g] & 63
            } else {
                (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4)
            }
        }),
        minima: std::array::from_fn(|g| {
            if g < 4 {
                s[g + 4] & 63
            } else {
                (s[g + 4] >> 4) | ((s[g] >> 6) << 4)
            }
        }),
        quants: std::array::from_fn(|i| {
            (bytes[16 + (i / 64) * 32 + i % 32] >> (4 * ((i % 64) / 32))) & 15
        }),
    }
}
pub(super) struct Case {
    pub rows: usize,
    pub hidden: usize,
    pub intermediate: usize,
    input: Vec<f16>,
    dx: CudaSlice<f16>,
    gu: CudaSlice<f16>,
    act: CudaSlice<f16>,
    out: CudaSlice<f16>,
    partial: CudaSlice<f32>,
    product_workspace: CudaSlice<u8>,
    product_layout: super::super::super::stream_mmq::Workspace,
    packed: CudaSlice<u32>,
    deltas: CudaSlice<f32>,
    sums: CudaSlice<i32>,
    pub(super) gate: Matrix,
    pub(super) up: Matrix,
    pub(super) down: Matrix,
}
impl Case {
    pub fn new(
        s: &Arc<CudaStream>,
        candidate: &Candidate,
        rows: usize,
        h: usize,
        i: usize,
        down: GgufBlockFormat,
        salt: usize,
    ) -> Self {
        candidate.report_shape(rows, h, i);
        let input = inputs(rows, h, 0);
        let product_layout = candidate.product.workspace(h as u32, i as u32).unwrap();
        Self {
            product_workspace: s
                .clone_htod(&vec![0xccu8; 512 + product_layout.total_bytes as usize])
                .unwrap(),
            product_layout,
            rows,
            hidden: h,
            intermediate: i,
            dx: s.clone_htod(&input).unwrap(),
            input,
            gu: s
                .clone_htod(&vec![f16::from_f32(SENTINEL); PAD + rows * i * 2 + PAD])
                .unwrap(),
            act: s
                .clone_htod(&vec![f16::from_f32(SENTINEL); PAD + rows * i + PAD])
                .unwrap(),
            out: s
                .clone_htod(&vec![f16::from_f32(SENTINEL); PAD + rows * h + PAD])
                .unwrap(),
            partial: s
                .clone_htod(&vec![
                    SENTINEL;
                    PAD + scratch_len(rows, h, i, candidate.cta_budget)
                        + PAD
                ])
                .unwrap(),
            packed: s
                .clone_htod(&vec![0xdeadbeefu32; PAD + rows * h / 4 + PAD])
                .unwrap(),
            deltas: s
                .clone_htod(&vec![SENTINEL; PAD + rows * h / 32 + PAD])
                .unwrap(),
            sums: s
                .clone_htod(&vec![i32::MIN; PAD + rows * h / 32 + PAD])
                .unwrap(),
            gate: Matrix::new(s, GgufBlockFormat::Q4K, h, i, salt),
            up: Matrix::new(s, GgufBlockFormat::Q4K, h, i, salt + 1),
            down: Matrix::new(s, down, i, h, salt + 2),
        }
    }
    pub fn update(&mut self, s: &Arc<CudaStream>, g: usize) {
        s.synchronize().unwrap();
        self.input = inputs(self.rows, self.hidden, g);
        s.memcpy_htod(&self.input, &mut self.dx).unwrap();
        s.synchronize().unwrap();
    }
    pub fn reset(&mut self, s: &Arc<CudaStream>) {
        s.memcpy_htod(
            &vec![0xccu8; self.product_workspace.len()],
            &mut self.product_workspace,
        )
        .unwrap();
        s.memcpy_htod(&vec![SENTINEL; self.partial.len()], &mut self.partial)
            .unwrap();
        s.memcpy_htod(&vec![0xdeadbeefu32; self.packed.len()], &mut self.packed)
            .unwrap();
        s.memcpy_htod(&vec![SENTINEL; self.deltas.len()], &mut self.deltas)
            .unwrap();
        s.memcpy_htod(&vec![i32::MIN; self.sums.len()], &mut self.sums)
            .unwrap();
        for b in [&mut self.gu, &mut self.act, &mut self.out] {
            s.memcpy_htod(&vec![f16::from_f32(SENTINEL); b.len()], b)
                .unwrap();
        }
        s.synchronize().unwrap();
    }
    pub fn capture(
        &mut self,
        s: &Arc<CudaStream>,
        k: &CudaNativeBlockKernels,
        c: &Candidate,
        candidate: bool,
    ) -> CudaGraph {
        let xv = self.dx.slice(PAD..self.dx.len() - PAD);
        let gv = self.gate.gpu.slice(self.gate.base..self.gate.raw.len() - 7);
        let uv = self.up.gpu.slice(self.up.base..self.up.raw.len() - 7);
        let dv = self.down.gpu.slice(self.down.base..self.down.raw.len() - 7);
        let mut gu = self.gu.slice_mut(PAD..self.gu.len() - PAD);
        let mut av = self.act.slice_mut(PAD..self.act.len() - PAD);
        let mut ov = self.out.slice_mut(PAD..self.out.len() - PAD);
        let mut pv = self.partial.slice_mut(PAD..self.partial.len() - PAD);
        let mut qv = self.packed.slice_mut(PAD..self.packed.len() - PAD);
        let mut ddv = self.deltas.slice_mut(PAD..self.deltas.len() - PAD);
        let mut ssv = self.sums.slice_mut(PAD..self.sums.len() - PAD);
        let mut pw = self
            .product_workspace
            .slice_mut(256..self.product_workspace.len() - 256);
        let (wp, wpg) = pw.device_ptr_mut(s);
        let (q, qg) = qv.device_ptr_mut(s);
        let (dd, ddg) = ddv.device_ptr_mut(s);
        let (ss, ssg) = ssv.device_ptr_mut(s);
        let (p, pg) = pv.device_ptr_mut(s);
        let (x, xg) = xv.device_ptr(s);
        let (g, gg) = gv.device_ptr(s);
        let (u, ug) = uv.device_ptr(s);
        let (d, dg) = dv.device_ptr(s);
        let (y, yg) = gu.device_ptr_mut(s);
        let (a, ag) = av.device_ptr_mut(s);
        let (o, og) = ov.device_ptr_mut(s);
        s.synchronize().unwrap();
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        if candidate && self.rows == 8 {
            let parts: Vec<_> = [0, self.intermediate]
                .into_iter()
                .map(|offset| weights::MatrixPart {
                    component_id: WeightId::new(format!("fixture.stream.{offset}")).unwrap(),
                    format: weights::MatrixFormat::Block(GgufBlockFormat::Q4K),
                    rows: self.intermediate as u32,
                    columns: self.hidden as u32,
                    output_offset: offset as u32,
                    transform: None,
                    signs_region: None,
                })
                .collect();
            c.product
                .launch_gate_up(
                    s,
                    &parts,
                    &[g, u],
                    x,
                    y,
                    self.rows as u32,
                    self.hidden as u32,
                    self.intermediate as u32,
                    wp,
                )
                .unwrap();
        } else {
            for (w, offset) in [(g, 0), (u, self.intermediate)] {
                strict(
                    s,
                    k,
                    x,
                    w,
                    y,
                    self.rows,
                    self.hidden,
                    self.intermediate,
                    self.intermediate * 2,
                    offset,
                    GgufBlockFormat::Q4K,
                );
            }
        }
        let inter = self.intermediate as i32;
        let total = (self.rows * self.intermediate) as i32;
        unsafe {
            s.launch_builder(&c.silu)
                .arg(&y)
                .arg(&a)
                .arg(&inter)
                .arg(&total)
                .launch(LaunchConfig::for_num_elems(total as u32))
        }
        .unwrap();
        strict(
            s,
            k,
            a,
            d,
            o,
            self.rows,
            self.intermediate,
            self.hidden,
            self.hidden,
            0,
            self.down.format,
        );
        let graph = s
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .unwrap()
            .unwrap();
        drop((xg, gg, ug, dg, yg, ag, og, pg, qg, ddg, ssg, wpg));
        s.synchronize().unwrap();
        graph
    }
    pub fn validate(&self, s: &Arc<CudaStream>, candidate: bool, weights: bool) -> Vec<u16> {
        s.synchronize().unwrap();
        let gu = s.clone_dtoh(&self.gu).unwrap();
        let act = s.clone_dtoh(&self.act).unwrap();
        let out = s.clone_dtoh(&self.out).unwrap();
        let workspace = s.clone_dtoh(&self.product_workspace).unwrap();
        assert!(workspace[..256]
            .iter()
            .chain(&workspace[workspace.len() - 256..])
            .all(|v| *v == 0xcc));
        let partial = s.clone_dtoh(&self.partial).unwrap();
        assert!(
            partial[..PAD]
                .iter()
                .chain(&partial[partial.len() - PAD..])
                .all(|v| *v == SENTINEL),
            "partial guards"
        );
        // These former independent buffers are no longer passed to the product wrapper.
        assert!(s
            .clone_dtoh(&self.packed)
            .unwrap()
            .iter()
            .all(|x| *x == 0xdeadbeef));
        assert!(s
            .clone_dtoh(&self.deltas)
            .unwrap()
            .iter()
            .all(|x| *x == SENTINEL));
        assert!(s
            .clone_dtoh(&self.sums)
            .unwrap()
            .iter()
            .all(|x| *x == i32::MIN));
        assert!(partial.iter().all(|x| *x == SENTINEL));
        if candidate && self.rows == 8 {
            let raw = &workspace[256..workspace.len() - 256];
            let q_end = self.product_layout.words_bytes as usize;
            let d_end = q_end + self.product_layout.scales_bytes as usize;
            let sum_end = self.product_layout.partial_offset as usize;
            assert_eq!(sum_end, d_end + self.product_layout.scales_bytes as usize);
            let q: Vec<_> = raw[..q_end]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let d: Vec<_> = raw[q_end..d_end]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let sums: Vec<_> = raw[d_end..sum_end]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            super::qualification::check_pack_values(
                &self.input[PAD..self.input.len() - PAD],
                self.rows,
                self.hidden,
                &q,
                &d,
                &sums,
            );
            assert!(
                raw[sum_end..]
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                "product partial finite"
            );
        } else {
            assert!(
                workspace.iter().all(|v| *v == 0xcc),
                "strict fallback must not write MMQ workspace"
            );
        }
        let mut stats = Vec::new();
        for (name, v, width) in [
            ("gate_up", &gu, self.intermediate * 2),
            ("activation", &act, self.intermediate),
            ("down", &out, self.hidden),
        ] {
            assert!(
                v[..PAD]
                    .iter()
                    .chain(&v[v.len() - PAD..])
                    .all(|x| x.to_f32() == SENTINEL),
                "{name} guards"
            );
            let data = &v[PAD..v.len() - PAD];
            assert!(data.iter().all(|x| x.is_finite()), "{name} finite");
            for r in 0..self.rows {
                assert!(
                    data[r * width..(r + 1) * width]
                        .iter()
                        .any(|x| x.to_f32() != 0.0),
                    "{name} nonzero row {r}"
                );
            }
            stats.push((
                name,
                data.iter().map(|x| x.to_f32().abs()).fold(0.0, f32::max),
            ));
        }
        for (name, offset, w) in [("gate", 0, &self.gate), ("up", self.intermediate, &self.up)] {
            for r in 0..self.rows {
                assert!(
                    gu[PAD + r * self.intermediate * 2 + offset
                        ..PAD + r * self.intermediate * 2 + offset + self.intermediate]
                        .iter()
                        .any(|x| x.to_f32() != 0.0),
                    "{name} nonzero row"
                );
            }
            check_projection(
                &self.input[PAD..self.input.len() - PAD],
                &gu[PAD..gu.len() - PAD],
                self.rows,
                w,
                self.intermediate * 2,
                offset,
                candidate && self.rows == 8,
            );
        }
        for r in 0..self.rows {
            for i in 0..self.intermediate {
                let g = gu[PAD + r * self.intermediate * 2 + i].to_f32();
                let u = gu[PAD + r * self.intermediate * 2 + self.intermediate + i].to_f32();
                let reference = g / (1.0 + (-g).exp()) * u;
                let y = act[PAD + r * self.intermediate + i].to_f32();
                assert!(
                    (y - reference).abs() <= reference.abs() * 0.002 + 1e-6,
                    "SiLU actual-input oracle"
                );
            }
        }
        check_projection(
            &act[PAD..act.len() - PAD],
            &out[PAD..out.len() - PAD],
            self.rows,
            &self.down,
            self.hidden,
            0,
            false,
        );
        assert_eq!(s.clone_dtoh(&self.dx).unwrap(), self.input);
        if weights {
            for w in [&self.gate, &self.up, &self.down] {
                w.check_immutable(s);
            }
        }
        println!(
            "{}",
            serde_json::json!({"event":"stage_validation","candidate":candidate,"rows":self.rows,"hidden":self.hidden,"intermediate":self.intermediate,"maxima":stats,"oracle":"Independent F64 F32-scale Q8 group policy for gate/up, strict F64 encoded coefficient dot for down, actual GPU stage inputs","model_quality_passed":null})
        );
        gu.iter()
            .chain(&act)
            .chain(&out)
            .map(|x| x.to_bits())
            .collect()
    }
}
