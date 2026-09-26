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

/// Independent scalar implementation. Residual remains F32, never rounded to F16.
pub(super) fn pack_oracle(
    x: &[f16],
    rows: usize,
    k: usize,
    terms: usize,
) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    assert_eq!(x.len(), rows * k);
    let groups = rows * k / 32;
    let mut q = vec![0i8; terms * groups * 32];
    let mut d = vec![0f32; terms * groups];
    let mut sums = vec![0i32; terms * groups];
    for group in 0..groups {
        let mut residual: [f32; 32] = std::array::from_fn(|i| x[group * 32 + i].to_f32());
        for term in 0..terms {
            let invalid = residual.iter().any(|v| !v.is_finite());
            let mx = residual
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .map(f32::abs)
                .fold(0.0, f32::max);
            let delta = if invalid {
                f32::NAN
            } else if mx == 0.0 {
                0.0
            } else {
                mx / 127.0
            };
            let idx = term * groups + group;
            d[idx] = delta;
            for i in 0..32 {
                let qi = if invalid || mx == 0.0 {
                    0
                } else {
                    (residual[i] / delta).round().clamp(-127.0, 127.0) as i8
                };
                q[idx * 32 + i] = qi;
                sums[idx] += i32::from(qi);
                residual[i] -= delta * f32::from(qi);
            }
        }
        if x[group * 32..group * 32 + 32].iter().all(|v| v.is_finite()) {
            let mx = x[group * 32..group * 32 + 32]
                .iter()
                .map(|v| v.to_f32().abs())
                .fold(0.0, f32::max);
            // A second level reduces the absolute quantization bound by 254.
            let bound = mx / 254.0_f32.powi(terms as i32) + mx * f32::EPSILON * 4.0;
            assert!(residual
                .iter()
                .all(|v| v.abs() <= bound + f32::MIN_POSITIVE));
        }
    }
    (q, d, sums)
}
pub(super) fn check_pack(
    s: &Arc<CudaStream>,
    x: &[f16],
    rows: usize,
    k: usize,
    terms: usize,
    q: &CudaSlice<u32>,
    d: &CudaSlice<f32>,
    sum: &CudaSlice<i32>,
) {
    let qq = s.clone_dtoh(q).unwrap();
    let dd = s.clone_dtoh(d).unwrap();
    let ss = s.clone_dtoh(sum).unwrap();
    assert!(qq[..PAD]
        .iter()
        .chain(&qq[qq.len() - PAD..])
        .all(|v| *v == 0xdeadbeef));
    assert!(dd[..PAD]
        .iter()
        .chain(&dd[dd.len() - PAD..])
        .all(|v| *v == SENTINEL));
    assert!(ss[..PAD]
        .iter()
        .chain(&ss[ss.len() - PAD..])
        .all(|v| *v == i32::MIN));
    let (eq, ed, es) = pack_oracle(x, rows, k, terms);
    let groups = rows * k / 32;
    for term in 0..terms {
        for row in 0..rows {
            for group in 0..k / 32 {
                let i = term * groups + row * (k / 32) + group;
                let pg = term * groups + ((group / 8) * rows + row) * 8 + group % 8;
                if ed[i].is_nan() {
                    assert!(dd[PAD + pg].is_nan());
                } else {
                    assert_eq!(
                        dd[PAD + pg].to_bits(),
                        ed[i].to_bits(),
                        "scale term={term} row={row} group={group}"
                    );
                }
                assert_eq!(ss[PAD + pg], es[i]);
                for word in 0..8 {
                    let b = std::array::from_fn(|j| eq[i * 32 + word * 4 + j] as u8);
                    assert_eq!(qq[PAD + pg * 8 + word], u32::from_le_bytes(b));
                }
            }
        }
    }
}
pub(super) fn check_projection(
    x: &[f16],
    y: &[f16],
    rows: usize,
    w: &Matrix,
    stride: usize,
    offset: usize,
    terms: usize,
) {
    let (q, d, _) = pack_oracle(x, rows, w.k, terms.max(1));
    let groups = rows * w.k / 32;
    for r in 0..rows {
        for col in 0..w.n.min(TEMPLATES) {
            let (mut dot, mut abs) = (0f64, 0f64);
            for i in 0..w.k {
                let xv = if terms == 0 {
                    f64::from(x[r * w.k + i].to_f32())
                } else {
                    (0..terms)
                        .map(|t| {
                            let group = t * groups + r * (w.k / 32) + i / 32;
                            f64::from(d[group]) * f64::from(q[group * 32 + i % 32])
                        })
                        .sum()
                };
                let v = xv * f64::from(w.coefficient(col, i));
                dot += v;
                abs += v.abs();
            }
            let ne = (w.k * terms.max(1) + 16) as f64 * f64::from(f32::EPSILON);
            let bound = ne / (1.0 - ne) * abs + dot.abs() / 1024.0 + 1e-6;
            for c in (col..w.n).step_by(TEMPLATES) {
                let observed = y[r * stride + offset + c].to_f64();
                assert!(observed.is_finite()&&(observed-dot).abs()<=bound,"terms={terms} row={r} col={c} K={} observed={observed} expected={dot} bound={bound}",w.k);
            }
        }
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
    packed: CudaSlice<u32>,
    deltas: CudaSlice<f32>,
    sums: CudaSlice<i32>,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}
impl Case {
    pub fn new(
        s: &Arc<CudaStream>,
        c: &Candidate,
        rows: usize,
        h: usize,
        i: usize,
        down: GgufBlockFormat,
        salt: usize,
    ) -> Self {
        let input = inputs(rows, h, 0);
        let max_k = h.max(i);
        let partial_len =
            scratch_len(rows, h, i, c.cta_budget).max(scratch_len(rows, i, h, c.cta_budget));
        Self {
            rows,
            hidden: h,
            intermediate: i,
            dx: s.clone_htod(&input).unwrap(),
            input,
            gu: s
                .clone_htod(&vec![f16::from_f32(SENTINEL); 2 * PAD + rows * i * 2])
                .unwrap(),
            act: s
                .clone_htod(&vec![f16::from_f32(SENTINEL); 2 * PAD + rows * i])
                .unwrap(),
            out: s
                .clone_htod(&vec![f16::from_f32(SENTINEL); 2 * PAD + rows * h])
                .unwrap(),
            partial: s
                .clone_htod(&vec![SENTINEL; 2 * PAD + partial_len])
                .unwrap(),
            packed: s
                .clone_htod(&vec![0xdeadbeef; 2 * PAD + 2 * rows * max_k / 4])
                .unwrap(),
            deltas: s
                .clone_htod(&vec![SENTINEL; 2 * PAD + 2 * rows * max_k / 32])
                .unwrap(),
            sums: s
                .clone_htod(&vec![i32::MIN; 2 * PAD + 2 * rows * max_k / 32])
                .unwrap(),
            gate: Matrix::new(s, GgufBlockFormat::Q4K, h, i, salt),
            up: Matrix::new(s, GgufBlockFormat::Q4K, h, i, salt + 1),
            down: Matrix::new(s, down, i, h, salt + 2),
        }
    }
    pub fn update(&mut self, s: &Arc<CudaStream>, generation: usize) {
        s.synchronize().unwrap();
        self.input = inputs(self.rows, self.hidden, generation);
        s.memcpy_htod(&self.input, &mut self.dx).unwrap();
        s.synchronize().unwrap();
    }
    pub fn reset(&mut self, s: &Arc<CudaStream>) {
        s.memcpy_htod(&vec![SENTINEL; self.partial.len()], &mut self.partial)
            .unwrap();
        s.memcpy_htod(&vec![0xdeadbeef; self.packed.len()], &mut self.packed)
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
        mmq: bool,
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
        let (x, xg) = xv.device_ptr(s);
        let (g, gg) = gv.device_ptr(s);
        let (u, ug) = uv.device_ptr(s);
        let (d, dg) = dv.device_ptr(s);
        let (y, yg) = gu.device_ptr_mut(s);
        let (a, ag) = av.device_ptr_mut(s);
        let (o, og) = ov.device_ptr_mut(s);
        let (p, pg) = pv.device_ptr_mut(s);
        let (q, qg) = qv.device_ptr_mut(s);
        let (dd, ddg) = ddv.device_ptr_mut(s);
        let (ss, ssg) = ssv.device_ptr_mut(s);
        s.synchronize().unwrap();
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        if mmq {
            c.pack(s, x, q, dd, ss, self.rows, self.hidden);
        }
        for (w, offset) in [(g, 0), (u, self.intermediate)] {
            if mmq {
                c.project(
                    s,
                    q,
                    dd,
                    ss,
                    w,
                    y,
                    p,
                    self.rows,
                    self.hidden,
                    self.intermediate,
                    self.intermediate * 2,
                    offset,
                );
            } else {
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
                .launch(LaunchConfig {
                    grid_dim: ((total as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
        if mmq && self.down.format == GgufBlockFormat::Q4K {
            c.pack(s, a, q, dd, ss, self.rows, self.intermediate);
            c.project(
                s,
                q,
                dd,
                ss,
                d,
                o,
                p,
                self.rows,
                self.intermediate,
                self.hidden,
                self.hidden,
                0,
            );
        } else {
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
        }
        let graph = s
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .unwrap()
            .unwrap();
        drop((xg, gg, ug, dg, yg, ag, og, pg, qg, ddg, ssg));
        s.synchronize().unwrap();
        graph
    }
    pub fn validate(&self, s: &Arc<CudaStream>, terms: usize, weights: bool) -> Vec<u16> {
        s.synchronize().unwrap();
        let gu = s.clone_dtoh(&self.gu).unwrap();
        let act = s.clone_dtoh(&self.act).unwrap();
        let out = s.clone_dtoh(&self.out).unwrap();
        for (name, v, width) in [
            ("gate_up", &gu, self.intermediate * 2),
            ("silu", &act, self.intermediate),
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
            assert!(data.iter().all(|x| x.is_finite()));
            for r in 0..self.rows {
                assert!(
                    data[r * width..(r + 1) * width]
                        .iter()
                        .any(|x| x.to_f32() != 0.0),
                    "{name} nonzero"
                );
            }
        }
        let part = s.clone_dtoh(&self.partial).unwrap();
        assert!(part[..PAD]
            .iter()
            .chain(&part[part.len() - PAD..])
            .all(|x| *x == SENTINEL));
        if terms > 0 {
            let (px, pk) = if self.down.format == GgufBlockFormat::Q4K {
                (&act[PAD..act.len() - PAD], self.intermediate)
            } else {
                (&self.input[PAD..self.input.len() - PAD], self.hidden)
            };
            check_pack(
                s,
                px,
                self.rows,
                pk,
                terms,
                &self.packed,
                &self.deltas,
                &self.sums,
            );
        }
        for (offset, w) in [(0, &self.gate), (self.intermediate, &self.up)] {
            check_projection(
                &self.input[PAD..self.input.len() - PAD],
                &gu[PAD..gu.len() - PAD],
                self.rows,
                w,
                self.intermediate * 2,
                offset,
                terms,
            );
        }
        for r in 0..self.rows {
            for i in 0..self.intermediate {
                let g = gu[PAD + r * self.intermediate * 2 + i].to_f32();
                let u = gu[PAD + r * self.intermediate * 2 + self.intermediate + i].to_f32();
                let reference = g / (1.0 + (-g).exp()) * u;
                let y = act[PAD + r * self.intermediate + i].to_f32();
                assert!((y - reference).abs() <= reference.abs() * 0.002 + 1e-6);
            }
        }
        check_projection(
            &act[PAD..act.len() - PAD],
            &out[PAD..out.len() - PAD],
            self.rows,
            &self.down,
            self.hidden,
            0,
            if self.down.format == GgufBlockFormat::Q4K {
                terms
            } else {
                0
            },
        );
        assert_eq!(s.clone_dtoh(&self.dx).unwrap(), self.input);
        if weights {
            for w in [&self.gate, &self.up, &self.down] {
                w.check_immutable(s);
            }
        }
        gu.iter()
            .chain(&act)
            .chain(&out)
            .map(|x| x.to_bits())
            .collect()
    }
}
